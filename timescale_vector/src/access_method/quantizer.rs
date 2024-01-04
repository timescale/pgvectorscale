use pgrx::PgRelation;
use rand::seq::index;

use crate::util::{page, tape::Tape, HeapPointer, IndexPointer, ItemPointer};

use super::{
    bq::BqQuantizer,
    disk_index_graph::DiskIndexGraph,
    graph::{
        self, FullVectorDistanceState, Graph, GraphNeighborStore, GreedySearchStats,
        ListSearchNeighbor, ListSearchResult, LsrPrivateData, SearchDistanceMeasure,
    },
    meta_page::MetaPage,
    model::{NeighborWithDistance, Node},
    pq::PqQuantizer,
};

/*pub trait Quantizer {
    fn initialize_node(&self, node: &mut Node, meta_page: &MetaPage);
    fn start_training(&mut self, meta_page: &super::meta_page::MetaPage);
    fn add_sample(&mut self, sample: Vec<f32>);
    fn finish_training(&mut self);
}*/

pub enum Quantizer<'a> {
    BQ(BqQuantizer<'a>),
    PQ(PqQuantizer),
    None,
}

impl<'a> Quantizer<'a> {
    pub fn is_some(&self) -> bool {
        match self {
            Quantizer::None => false,
            _ => true,
        }
    }

    pub fn page_type(&self) -> page::PageType {
        match self {
            Quantizer::None => page::PageType::Node,
            Quantizer::PQ(_) => page::PageType::Node,
            Quantizer::BQ(_) => page::PageType::BqNode,
        }
    }

    pub fn create_node(
        &self,
        index_relation: &PgRelation,
        vector: &[f32],
        heap_pointer: HeapPointer,
        meta_page: &MetaPage,
        tape: &mut Tape,
    ) -> ItemPointer {
        match self {
            Quantizer::None | Quantizer::PQ(_) => {
                let node = Node::new(vector.to_vec(), heap_pointer, &meta_page, &self);

                let index_pointer: IndexPointer = node.write(tape);
                index_pointer
            }
            Quantizer::BQ(bq) => {
                bq.create_node(index_relation, vector, heap_pointer, meta_page, tape)
            }
        }
    }

    pub unsafe fn get_full_vector_distance_state<'i>(
        &self,
        index: &'i PgRelation,
        index_pointer: IndexPointer,
    ) -> FullVectorDistanceState<'i> {
        match self {
            Quantizer::None => pgrx::error!("not implemented"),
            Quantizer::PQ(pq) => pgrx::error!("not implemented"),
            Quantizer::BQ(bq) => bq.get_full_vector_distance_state(index, index_pointer),
        }
    }

    pub unsafe fn get_distance_pair_for_full_vectors_from_state(
        &self,
        state: &FullVectorDistanceState,
        index: &PgRelation,
        index_pointer: IndexPointer,
    ) -> f32 {
        match self {
            Quantizer::None => pgrx::error!("not implemented"),
            Quantizer::PQ(pq) => pgrx::error!("not implemented"),
            Quantizer::BQ(bq) => {
                bq.get_distance_pair_for_full_vectors_from_state(state, index, index_pointer)
            }
        }
    }

    pub fn get_search_distance_measure(
        &self,
        query: &[f32],
        distance_fn: fn(&[f32], &[f32]) -> f32,
        calc_distance_with_quantizer: bool,
    ) -> SearchDistanceMeasure {
        match self {
            Quantizer::None => pgrx::error!("not implemented"),
            Quantizer::PQ(pq) => pgrx::error!("not implemented"),
            Quantizer::BQ(bq) => {
                bq.get_search_distance_measure(query, distance_fn, calc_distance_with_quantizer)
            }
        }
    }

    pub fn get_neighbors_with_distances(
        &self,
        index: &PgRelation,
        neighbors_of: ItemPointer,
        result: &mut Vec<NeighborWithDistance>,
    ) -> bool {
        match self {
            Quantizer::None => pgrx::error!("not implemented"),
            Quantizer::PQ(pq) => pgrx::error!("not implemented"),
            Quantizer::BQ(bq) => bq.get_neighbors_with_distances(index, neighbors_of, result),
        }
    }

    pub fn visit_lsn(
        &self,
        index: &PgRelation,
        lsr: &mut ListSearchResult,
        lsn_idx: usize,
        query: &[f32],
        gns: &GraphNeighborStore,
    ) {
        match self {
            Quantizer::None => pgrx::error!("not implemented"),
            Quantizer::PQ(pq) => pgrx::error!("not implemented"),
            Quantizer::BQ(bq) => bq.visit_lsn(index, lsr, lsn_idx, query, gns),
        }
    }

    pub fn get_lsn(
        &self,
        lsr: &mut ListSearchResult,
        index: &PgRelation,
        index_pointer: ItemPointer,
        query: &[f32],
    ) -> ListSearchNeighbor {
        match self {
            Quantizer::None => pgrx::error!("not implemented"),
            Quantizer::PQ(pq) => pgrx::error!("not implemented"),
            Quantizer::BQ(bq) => bq.get_lsn(lsr, index, index_pointer, query),
        }
    }

    pub fn return_lsn(
        &self,
        index: &PgRelation,
        lsr: &mut ListSearchResult,
        idx: usize,
    ) -> (HeapPointer, IndexPointer) {
        match self {
            Quantizer::None => pgrx::error!("not implemented"),
            Quantizer::PQ(pq) => pgrx::error!("not implemented"),
            Quantizer::BQ(bq) => bq.return_lsn(index, lsr, idx),
        }
    }
}
