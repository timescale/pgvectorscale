use std::collections::HashMap;
use std::time::Instant;

use pgrx::*;

use crate::util::{IndexPointer, ItemPointer};

use super::graph::Graph;

use super::model::*;
use super::storage::Storage;

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

    pub unsafe fn write(&self, index: &PgRelation, storage: &Storage, graph: &Graph) -> WriteStats {
        let mut stats = WriteStats::new();

        //TODO: OPT: do this in order of item pointers
        for (index_pointer, neighbors) in &self.neighbor_map {
            stats.num_nodes += 1;
            let prune_neighbors;
            let neighbors = if neighbors.len() > graph.get_meta_page().get_num_neighbors() as _ {
                stats.num_prunes += 1;
                stats.num_neighbors_before_prune += neighbors.len();
                (prune_neighbors, _) =
                    graph.prune_neighbors(index, *index_pointer, vec![], storage);
                stats.num_neighbors_after_prune += prune_neighbors.len();
                &prune_neighbors
            } else {
                neighbors
            };
            stats.num_neighbors += neighbors.len();

            match storage {
                Storage::None => {
                    error!("Quantizer::None not implemented")
                    /* need to update the neighbors */
                }
                Storage::PQ(_pq) => {
                    error!("Quantizer::None not implemented");
                    //pq.update_node_after_traing(index, &meta, *index_pointer, neighbors);
                }
                Storage::BQ(bq) => {
                    //TODO: OPT: this may not be needed
                    bq.update_node_after_traing(
                        index,
                        graph.get_meta_page(),
                        *index_pointer,
                        neighbors,
                    );
                }
            };
        }
        stats
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

    pub fn get_neighbors_with_distances(
        &self,
        _index: &PgRelation,
        neighbors_of: ItemPointer,
        _storage: &Storage,
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
        _index: &PgRelation,
        neighbors_of: ItemPointer,
        new_neighbors: Vec<NeighborWithDistance>,
    ) {
        self.neighbor_map.insert(neighbors_of, new_neighbors);
    }
}

pub struct WriteStats {
    pub started: Instant,
    pub num_nodes: usize,
    pub num_prunes: usize,
    pub num_neighbors_before_prune: usize,
    pub num_neighbors_after_prune: usize,
    pub num_neighbors: usize,
}

impl WriteStats {
    pub fn new() -> Self {
        Self {
            started: Instant::now(),
            num_nodes: 0,
            num_prunes: 0,
            num_neighbors_before_prune: 0,
            num_neighbors_after_prune: 0,
            num_neighbors: 0,
        }
    }
}
