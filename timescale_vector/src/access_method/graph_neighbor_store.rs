use std::collections::HashMap;
use std::time::Instant;

use pgrx::*;

use crate::util::{IndexPointer, ItemPointer};

use super::stats::{PruneNeighborStats, StatsDistanceComparison, StatsNodeModify, StatsNodeRead};

use super::meta_page::MetaPage;
use super::model::*;
use super::storage::Storage;

/// A builderGraph is a graph that keep the neighbors in-memory in the neighbor_map below
/// The idea is that during the index build, you don't want to update the actual Postgres
/// pages every time you change the neighbors. Instead you change the neighbors in memory
/// until the build is done. Afterwards, calling the `write` method, will write out all
/// the neighbors to the right pages.
pub struct BuilderNeighborCache {
    //maps node's pointer to the representation on disk
    neighbor_map: HashMap<ItemPointer, Vec<NeighborWithDistance>>,
}

impl BuilderNeighborCache {
    pub fn new() -> Self {
        Self {
            neighbor_map: HashMap::with_capacity(200),
        }
    }
    pub fn iter(&self) -> impl Iterator<Item = (&ItemPointer, &Vec<NeighborWithDistance>)> {
        self.neighbor_map.iter()
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

    pub fn get_neighbors_with_full_vector_distances<S: Storage>(
        &self,
        _index: &PgRelation,
        neighbors_of: ItemPointer,
        _storage: &S,
        result: &mut Vec<NeighborWithDistance>,
    ) -> bool {
        let neighbors = self.neighbor_map.get(&neighbors_of);
        match neighbors {
            Some(n) => {
                for nwd in n {
                    result.push(nwd.clone());
                }
                true
            }
            None => false,
        }
    }

    pub fn set_neighbors(
        &mut self,
        neighbors_of: ItemPointer,
        new_neighbors: Vec<NeighborWithDistance>,
    ) {
        self.neighbor_map.insert(neighbors_of, new_neighbors);
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
        index: &PgRelation,
        neighbors_of: ItemPointer,
        storage: &S,
        result: &mut Vec<NeighborWithDistance>,
        stats: &mut T,
    ) -> bool {
        match self {
            GraphNeighborStore::Builder(b) => {
                b.get_neighbors_with_full_vector_distances(index, neighbors_of, storage, result)
            }
            GraphNeighborStore::Disk => unsafe {
                storage.get_neighbors_with_full_vector_distances_from_disk(
                    index,
                    neighbors_of,
                    result,
                    stats,
                )
            },
        }
    }

    pub fn set_neighbors<S: Storage, T: StatsNodeModify + StatsNodeRead>(
        &mut self,
        storage: &S,
        index: &PgRelation,
        meta_page: &MetaPage,
        neighbors_of: ItemPointer,
        new_neighbors: Vec<NeighborWithDistance>,
        stats: &mut T,
    ) {
        match self {
            GraphNeighborStore::Builder(b) => b.set_neighbors(neighbors_of, new_neighbors),
            GraphNeighborStore::Disk => storage.set_neighbors_on_disk(
                index,
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
