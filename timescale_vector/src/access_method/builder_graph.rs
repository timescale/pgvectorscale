use std::collections::hash_map::IterMut;
use std::collections::HashMap;
use std::time::Instant;

use pgrx::*;

use crate::util::{IndexPointer, ItemPointer};

use super::graph::PruneNeighborStats;

use super::meta_page::MetaPage;
use super::model::*;
use super::storage::StorageTrait;

/// A builderGraph is a graph that keep the neighbors in-memory in the neighbor_map below
/// The idea is that during the index build, you don't want to update the actual Postgres
/// pages every time you change the neighbors. Instead you change the neighbors in memory
/// until the build is done. Afterwards, calling the `write` method, will write out all
/// the neighbors to the right pages.
pub struct BuilderGraph {
    //maps node's pointer to the representation on disk
    neighbor_map: HashMap<ItemPointer, Vec<NeighborWithDistance>>,
}

impl BuilderGraph {
    pub fn new() -> Self {
        Self {
            neighbor_map: HashMap::with_capacity(200),
        }
    }
    pub fn iter(&self) -> impl Iterator<Item = (&ItemPointer, &Vec<NeighborWithDistance>)> {
        self.neighbor_map.iter()
    }

    pub fn iter_mut(&mut self) -> IterMut<ItemPointer, Vec<NeighborWithDistance>> {
        self.neighbor_map.iter_mut()
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

    pub fn get_neighbors_with_full_vector_distances<S: StorageTrait>(
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

    pub fn is_empty(&self) -> bool {
        self.neighbor_map.len() == 0
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

pub struct WriteStats {
    pub started: Instant,
    pub num_nodes: usize,
    pub prune_stats: PruneNeighborStats,
    pub num_neighbors: usize,
}

impl WriteStats {
    pub fn new() -> Self {
        Self {
            started: Instant::now(),
            num_nodes: 0,
            prune_stats: PruneNeighborStats::new(),
            num_neighbors: 0,
        }
    }
}
