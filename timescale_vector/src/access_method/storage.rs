use pgrx::PgRelation;

use crate::util::{
    page::{self, PageType},
    tape::Tape,
    HeapPointer, IndexPointer, ItemPointer,
};

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
    fn page_type(&self) -> PageType;

    fn create_node(
        &self,
        _index_relation: &PgRelation,
        full_vector: &[f32],
        heap_pointer: HeapPointer,
        meta_page: &MetaPage,
        tape: &mut Tape,
    ) -> ItemPointer;

    fn add_sample(&mut self, sample: &[f32]);

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
