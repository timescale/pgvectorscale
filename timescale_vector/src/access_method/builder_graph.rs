use std::collections::HashMap;
use std::time::Instant;

use ndarray::Array1;
use pgrx::*;
use reductive::pq::{Pq, QuantizeVector};

use crate::util::{IndexPointer, ItemPointer};

use super::graph::{Graph, VectorProvider};
use super::meta_page::MetaPage;
use super::starting_ids::StartingIds;
use super::{model::*, starting_ids};

/// A builderGraph is a graph that keep the neighbors in-memory in the neighbor_map below
/// The idea is that during the index build, you don't want to update the actual Postgres
/// pages every time you change the neighbors. Instead you change the neighbors in memory
/// until the build is done. Afterwards, calling the `write` method, will write out all
/// the neighbors to the right pages.
pub struct BuilderGraph<'a> {
    //maps node's pointer to the representation on disk
    neighbor_map: HashMap<ItemPointer, Vec<NeighborWithDistance>>,
    meta_page: MetaPage,
    starting_ids: StartingIds,
    vector_provider: VectorProvider<'a>,
}

impl<'a> BuilderGraph<'a> {
    pub fn new(meta_page: MetaPage, vp: VectorProvider<'a>, starting_ids: StartingIds) -> Self {
        Self {
            neighbor_map: HashMap::with_capacity(200),
            meta_page,
            starting_ids: starting_ids,
            vector_provider: vp,
        }
    }

    unsafe fn get_pq_vector(
        &self,
        index: &PgRelation,
        index_pointer: ItemPointer,
        pq: &Pq<f32>,
    ) -> Vec<u8> {
        let vp = self.get_vector_provider();
        let copy = vp.get_full_vector_copy_from_heap(index, index_pointer);
        let og_vec = Array1::from(copy);
        pq.quantize_vector(og_vec).to_vec()
    }

    pub unsafe fn write(&self, index: &PgRelation, pq: &Option<Pq<f32>>) -> WriteStats {
        let mut stats = WriteStats::new();
        self.starting_ids.write(index);

        //TODO: OPT: do this in order of item pointers
        for (index_pointer, neighbors) in &self.neighbor_map {
            stats.num_nodes += 1;
            let prune_neighbors;
            let neighbors = if neighbors.len() > self.meta_page.get_num_neighbors() as _ {
                stats.num_prunes += 1;
                stats.num_neighbors_before_prune += neighbors.len();
                (prune_neighbors, _) = self.prune_neighbors(index, *index_pointer, vec![]);
                stats.num_neighbors_after_prune += prune_neighbors.len();
                &prune_neighbors
            } else {
                neighbors
            };
            stats.num_neighbors += neighbors.len();

            let pqv = match pq {
                Some(pq) => Some(self.get_pq_vector(index, *index_pointer, pq)),
                None => None,
            };
            Node::update_neighbors_and_pq(
                index,
                *index_pointer,
                neighbors,
                self.get_meta_page(index),
                pqv,
            );
        }
        stats
    }
}

impl<'a> Graph for BuilderGraph<'a> {
    fn read<'b>(&self, index: &'b PgRelation, index_pointer: ItemPointer) -> ReadableNode<'b> {
        unsafe { Node::read(index, index_pointer) }
    }

    fn get_starting_ids(&mut self, query: &[f32]) -> Vec<ItemPointer> {
        self.starting_ids.get_starting_ids(query)
    }

    fn get_or_init_starting_ids(
        &mut self,
        _index: &PgRelation,
        item_pointer: ItemPointer,
        query: &[f32],
    ) -> Vec<ItemPointer> {
        self.starting_ids
            .get_or_init_starting_ids(item_pointer, query)
    }

    fn get_neighbors(
        &self,
        _index: &PgRelation,
        neighbors_of: ItemPointer,
        result: &mut Vec<IndexPointer>,
    ) -> bool {
        let neighbors = self.neighbor_map.get(&neighbors_of);
        match neighbors {
            Some(n) => {
                for nwd in n {
                    result.push(nwd.get_index_pointer_to_neighbor());
                }
                true
            }
            None => false,
        }
    }

    fn get_neighbors_with_distances(
        &self,
        _index: &PgRelation,
        neighbors_of: ItemPointer,
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

    fn is_empty(&self) -> bool {
        self.neighbor_map.len() == 0
    }

    fn get_vector_provider(&self) -> VectorProvider {
        return self.vector_provider.clone();
    }

    fn get_meta_page(&self, _index: &PgRelation) -> &MetaPage {
        &self.meta_page
    }

    fn set_neighbors(
        &mut self,
        index: &PgRelation,
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
