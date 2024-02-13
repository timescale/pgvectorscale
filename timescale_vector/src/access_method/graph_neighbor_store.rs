use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

use crate::util::{IndexPointer, ItemPointer};

use super::stats::{StatsDistanceComparison, StatsNodeModify, StatsNodeRead};

use super::meta_page::MetaPage;
use super::neighbor_with_distance::*;
use super::storage::Storage;

/// A builderGraph is a graph that keep the neighbors in-memory in the neighbor_map below
/// The idea is that during the index build, you don't want to update the actual Postgres
/// pages every time you change the neighbors. Instead you change the neighbors in memory
/// until the build is done. Afterwards, calling the `write` method, will write out all
/// the neighbors to the right pages.
pub struct BuilderNeighborCache {
    //maps node's pointer to the representation on disk
    neighbor_map: HashMap<ItemPointer, Vec<NeighborWithDistance>>,
    rc: RefCell<ReferenceCounter>,
}

struct ReferenceCounter {
    reference_count: HashMap<ItemPointer, usize>,
}

impl ReferenceCounter {
    pub fn inc_ref_count(&mut self, ip: ItemPointer) {
        let old = self.reference_count.get_mut(&ip);
        match old {
            Some(c) => {
                *c += 1;
            }
            None => {
                self.reference_count.insert(ip, 1);
            }
        }
    }

    pub fn dec_ref_count(&mut self, ip: ItemPointer) {
        let old = self.reference_count.get_mut(&ip);
        match old {
            Some(c) => {
                *c -= 1;
                if *c < 1 {
                    pgrx::warning!("Created orphaned neighbor {:?}", ip);
                }
            }
            None => {
                panic!("Decrementing ref count of non-existing neighbor {:?}", ip);
            }
        }
    }

    pub fn check_ref_count(&self, ip: ItemPointer) -> usize {
        let old = self.reference_count.get(&ip);
        match old {
            Some(c) => {
                return *c;
            }
            None => 0,
        }
    }
    pub fn adjust_ref(&mut self, old: &Vec<NeighborWithDistance>, new: &Vec<NeighborWithDistance>) {
        let old_set = old.iter().collect::<HashSet<_>>();
        let new_set = new.iter().collect::<HashSet<_>>();
        for &n in old_set.difference(&new_set) {
            self.dec_ref_count(n.get_index_pointer_to_neighbor());
        }
        for n in new_set.difference(&old_set) {
            self.inc_ref_count(n.get_index_pointer_to_neighbor());
        }
    }
}

impl BuilderNeighborCache {
    pub fn new() -> Self {
        Self {
            neighbor_map: HashMap::with_capacity(200),
            rc: RefCell::new(ReferenceCounter {
                reference_count: HashMap::with_capacity(200),
            }),
        }
    }
    pub fn iter(&self) -> impl Iterator<Item = (&ItemPointer, &Vec<NeighborWithDistance>)> {
        self.neighbor_map.iter()
    }

    pub fn check_ref_count(&self, ip: ItemPointer) -> usize {
        self.rc.borrow().check_ref_count(ip)
    }

    pub fn adjust_ref(&self, old: &Vec<NeighborWithDistance>, new: &Vec<NeighborWithDistance>) {
        self.rc.borrow_mut().adjust_ref(old, new);
    }

    pub fn get_neighbors(&self, neighbors_of: ItemPointer) -> Vec<IndexPointer> {
        let neighbors = self.neighbor_map.get(&neighbors_of);
        match neighbors {
            Some(n) => n
                .iter()
                .map(|n| n.get_index_pointer_to_neighbor())
                .collect(),
            None => vec![],
        }
    }

    pub fn get_neighbors_with_full_vector_distances(
        &self,
        neighbors_of: ItemPointer,
        result: &mut Vec<NeighborWithDistance>,
    ) {
        let neighbors = self.neighbor_map.get(&neighbors_of);
        match neighbors {
            Some(n) => {
                for nwd in n {
                    result.push(nwd.clone());
                }
            }
            None => (),
        }
    }

    pub fn set_neighbors(
        &mut self,
        neighbors_of: ItemPointer,
        new_neighbors: Vec<NeighborWithDistance>,
    ) {
        let old_neighbors = self
            .neighbor_map
            .insert(neighbors_of, new_neighbors.clone());
        if old_neighbors.is_some() {
            self.rc
                .borrow_mut()
                .adjust_ref(&old_neighbors.unwrap(), &new_neighbors);
        } else {
            for n in new_neighbors {
                self.rc
                    .borrow_mut()
                    .inc_ref_count(n.get_index_pointer_to_neighbor())
            }
        }
    }

    pub fn max_neighbors(&self, meta_page: &MetaPage) -> usize {
        meta_page.get_max_neighbors_during_build()
    }
}

pub enum GraphNeighborStore {
    Builder(BuilderNeighborCache),
    Disk,
}

impl GraphNeighborStore {
    pub fn get_neighbors_with_full_vector_distances<
        S: Storage,
        T: StatsNodeRead + StatsDistanceComparison,
    >(
        &self,
        neighbors_of: ItemPointer,
        storage: &S,
        result: &mut Vec<NeighborWithDistance>,
        stats: &mut T,
    ) {
        match self {
            GraphNeighborStore::Builder(b) => {
                b.get_neighbors_with_full_vector_distances(neighbors_of, result)
            }
            GraphNeighborStore::Disk => storage.get_neighbors_with_full_vector_distances_from_disk(
                neighbors_of,
                result,
                stats,
            ),
        };
    }

    pub fn set_neighbors<S: Storage, T: StatsNodeModify + StatsNodeRead>(
        &mut self,
        storage: &S,
        meta_page: &MetaPage,
        neighbors_of: ItemPointer,
        new_neighbors: Vec<NeighborWithDistance>,
        stats: &mut T,
    ) {
        match self {
            GraphNeighborStore::Builder(b) => b.set_neighbors(neighbors_of, new_neighbors),
            GraphNeighborStore::Disk => storage.set_neighbors_on_disk(
                meta_page,
                neighbors_of,
                new_neighbors.as_slice(),
                stats,
            ),
        }
    }

    pub fn max_neighbors(&self, meta_page: &MetaPage) -> usize {
        match self {
            GraphNeighborStore::Builder(b) => b.max_neighbors(meta_page),
            GraphNeighborStore::Disk => meta_page.get_num_neighbors() as _,
        }
    }
}
