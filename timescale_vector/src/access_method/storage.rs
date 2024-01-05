use pgrx::PgRelation;

use crate::util::{page, tape::Tape, HeapPointer, IndexPointer, ItemPointer};

use super::{
    bq::BqStorage,
    graph::{
        FullVectorDistanceState, GraphNeighborStore, ListSearchNeighbor, ListSearchResult,
        SearchDistanceMeasure,
    },
    meta_page::MetaPage,
    model::{NeighborWithDistance, Node},
    pq::PqQuantizer,
};

pub trait StorageTrait {
    unsafe fn get_full_vector_distance_state<'i>(
        &self,
        index: &'i PgRelation,
        index_pointer: IndexPointer,
    ) -> FullVectorDistanceState<'i>;

    unsafe fn get_distance_pair_for_full_vectors_from_state(
        &self,
        state: &FullVectorDistanceState,
        index: &PgRelation,
        index_pointer: IndexPointer,
    ) -> f32;

    fn get_search_distance_measure(
        &self,
        query: &[f32],
        distance_fn: fn(&[f32], &[f32]) -> f32,
        calc_distance_with_quantizer: bool,
    ) -> SearchDistanceMeasure;

    fn get_neighbors_with_distances(
        &self,
        index: &PgRelation,
        neighbors_of: ItemPointer,
        result: &mut Vec<NeighborWithDistance>,
    ) -> bool;

    fn visit_lsn(
        &self,
        index: &PgRelation,
        lsr: &mut ListSearchResult,
        lsn_idx: usize,
        query: &[f32],
        gns: &GraphNeighborStore,
    );

    fn get_lsn(
        &self,
        lsr: &mut ListSearchResult,
        index: &PgRelation,
        index_pointer: ItemPointer,
        query: &[f32],
    ) -> ListSearchNeighbor;

    fn return_lsn(
        &self,
        index: &PgRelation,
        lsr: &mut ListSearchResult,
        idx: usize,
    ) -> (HeapPointer, IndexPointer);

    fn set_neighbors_on_disk(
        &self,
        index: &PgRelation,
        meta: &MetaPage,
        index_pointer: IndexPointer,
        neighbors: &[NeighborWithDistance],
    );
}

pub enum Storage<'a> {
    BQ(BqStorage<'a>),
    PQ(PqQuantizer),
    None,
}

impl<'a> Storage<'a> {
    pub fn is_some(&self) -> bool {
        match self {
            Storage::None => false,
            _ => true,
        }
    }

    pub fn page_type(&self) -> page::PageType {
        match self {
            Storage::None => page::PageType::Node,
            Storage::PQ(_) => page::PageType::Node,
            Storage::BQ(_) => page::PageType::BqNode,
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
            Storage::None | Storage::PQ(_) => {
                let node = Node::new(vector.to_vec(), heap_pointer, &meta_page, &self);

                let index_pointer: IndexPointer = node.write(tape);
                index_pointer
            }
            Storage::BQ(bq) => {
                bq.create_node(index_relation, vector, heap_pointer, meta_page, tape)
            }
        }
    }
}

impl<'a> StorageTrait for Storage<'a> {
    unsafe fn get_full_vector_distance_state<'i>(
        &self,
        index: &'i PgRelation,
        index_pointer: IndexPointer,
    ) -> FullVectorDistanceState<'i> {
        match self {
            Storage::None => pgrx::error!("not implemented"),
            Storage::PQ(_pq) => pgrx::error!("not implemented"),
            Storage::BQ(bq) => bq.get_full_vector_distance_state(index, index_pointer),
        }
    }

    unsafe fn get_distance_pair_for_full_vectors_from_state(
        &self,
        state: &FullVectorDistanceState,
        index: &PgRelation,
        index_pointer: IndexPointer,
    ) -> f32 {
        match self {
            Storage::None => pgrx::error!("not implemented"),
            Storage::PQ(_pq) => pgrx::error!("not implemented"),
            Storage::BQ(bq) => {
                bq.get_distance_pair_for_full_vectors_from_state(state, index, index_pointer)
            }
        }
    }

    fn get_search_distance_measure(
        &self,
        query: &[f32],
        distance_fn: fn(&[f32], &[f32]) -> f32,
        calc_distance_with_quantizer: bool,
    ) -> SearchDistanceMeasure {
        match self {
            Storage::None => pgrx::error!("not implemented"),
            Storage::PQ(_pq) => pgrx::error!("not implemented"),
            Storage::BQ(bq) => {
                bq.get_search_distance_measure(query, distance_fn, calc_distance_with_quantizer)
            }
        }
    }

    fn get_neighbors_with_distances(
        &self,
        index: &PgRelation,
        neighbors_of: ItemPointer,
        result: &mut Vec<NeighborWithDistance>,
    ) -> bool {
        match self {
            Storage::None => pgrx::error!("not implemented"),
            Storage::PQ(_pq) => pgrx::error!("not implemented"),
            Storage::BQ(bq) => bq.get_neighbors_with_distances(index, neighbors_of, result),
        }
    }

    fn visit_lsn(
        &self,
        index: &PgRelation,
        lsr: &mut ListSearchResult,
        lsn_idx: usize,
        query: &[f32],
        gns: &GraphNeighborStore,
    ) {
        match self {
            Storage::None => pgrx::error!("not implemented"),
            Storage::PQ(_pq) => pgrx::error!("not implemented"),
            Storage::BQ(bq) => bq.visit_lsn(index, lsr, lsn_idx, query, gns),
        }
    }

    fn get_lsn(
        &self,
        lsr: &mut ListSearchResult,
        index: &PgRelation,
        index_pointer: ItemPointer,
        query: &[f32],
    ) -> ListSearchNeighbor {
        match self {
            Storage::None => pgrx::error!("not implemented"),
            Storage::PQ(_pq) => pgrx::error!("not implemented"),
            Storage::BQ(bq) => bq.get_lsn(lsr, index, index_pointer, query),
        }
    }

    fn return_lsn(
        &self,
        index: &PgRelation,
        lsr: &mut ListSearchResult,
        idx: usize,
    ) -> (HeapPointer, IndexPointer) {
        match self {
            Storage::None => pgrx::error!("not implemented"),
            Storage::PQ(_pq) => pgrx::error!("not implemented"),
            Storage::BQ(bq) => bq.return_lsn(index, lsr, idx),
        }
    }

    fn set_neighbors_on_disk(
        &self,
        index: &PgRelation,
        meta: &MetaPage,
        index_pointer: IndexPointer,
        neighbors: &[NeighborWithDistance],
    ) {
        match self {
            Storage::None => pgrx::error!("not implemented"),
            Storage::PQ(_pq) => pgrx::error!("not implemented"),
            Storage::BQ(bq) => bq.set_neighbors_on_disk(index, meta, index_pointer, neighbors),
        }
    }
}
