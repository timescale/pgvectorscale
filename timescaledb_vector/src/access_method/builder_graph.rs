use std::collections::HashMap;
use std::time::Instant;

use ndarray::Array1;
use pgrx::*;
use reductive::pq::{Pq, QuantizeVector};

use crate::util::ItemPointer;

use super::graph::Graph;
use super::meta_page::MetaPage;
use super::model::*;

/// A builderGraph is a graph that keep the neighbors in-memory in the neighbor_map below
/// The idea is that during the index build, you don't want to update the actual Postgres
/// pages every time you change the neighbors. Instead you change the neighbors in memory
/// until the build is done. Afterwards, calling the `write` method, will write out all
/// the neighbors to the right pages.
pub struct BuilderGraph {
    //maps node's pointer to the representation on disk
    neighbor_map: HashMap<ItemPointer, Vec<NeighborWithDistance>>,
    meta_page: MetaPage,
}

impl BuilderGraph {
    pub fn new(meta_page: MetaPage) -> Self {
        Self {
            neighbor_map: HashMap::with_capacity(200),
            meta_page,
        }
    }

    unsafe fn get_pq_vector(
        index: &PgRelation,
        index_pointer: ItemPointer,
        pq: &Pq<f32>,
    ) -> Vec<u8> {
        let node = Node::read(index, index_pointer);
        // todo: deal with clone
        let copy: Vec<f32> = node.get_archived_node().vector.iter().map(|f| *f).collect();
        let og_vec = Array1::from(copy);
        pq.quantize_vector(og_vec).to_vec()
    }

    pub unsafe fn write(&self, index: &PgRelation, pq: &Option<Pq<f32>>) -> WriteStats {
        let mut stats = WriteStats::new();

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
                Some(pq) => Some(BuilderGraph::get_pq_vector(index, *index_pointer, pq)),
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

impl Graph for BuilderGraph {
    fn read<'a>(&self, index: &'a PgRelation, index_pointer: ItemPointer) -> ReadableNode<'a> {
        unsafe { Node::read(index, index_pointer) }
    }

    fn get_init_ids(&mut self) -> Option<Vec<ItemPointer>> {
        //returns a vector for generality
        self.meta_page.get_init_ids()
    }

    fn get_neighbors(
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

    fn get_meta_page(&self, _index: &PgRelation) -> &MetaPage {
        &self.meta_page
    }

    fn set_neighbors(
        &mut self,
        index: &PgRelation,
        neighbors_of: ItemPointer,
        new_neighbors: Vec<NeighborWithDistance>,
    ) {
        if self.meta_page.get_init_ids().is_none() {
            //TODO probably better set off of centeroids
            MetaPage::update_init_ids(index, vec![neighbors_of]);
            self.meta_page = MetaPage::read(index);
        }
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
