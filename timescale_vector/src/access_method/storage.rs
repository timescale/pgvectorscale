use std::pin::Pin;

use pgrx::PgRelation;

use crate::util::{page::PageType, tape::Tape, HeapPointer, IndexPointer, ItemPointer};

use super::{
    graph::{Graph, ListSearchNeighbor, ListSearchResult},
    graph_neighbor_store::GraphNeighborStore,
    meta_page::MetaPage,
    model::NeighborWithDistance,
    pg_vector::PgVector,
    stats::{StatsDistanceComparison, StatsNodeModify, StatsNodeRead, WriteStats},
};

pub trait NodeDistanceMeasure {
    unsafe fn get_distance<S: StatsNodeRead + StatsDistanceComparison>(
        &self,
        index: &PgRelation,
        index_pointer: IndexPointer,
        stats: &mut S,
    ) -> f32;
}

pub trait ArchivedData {
    fn with_data(data: &mut [u8]) -> Pin<&mut Self>;
    fn is_deleted(&self) -> bool;
    fn delete(self: Pin<&mut Self>);
    fn get_heap_item_pointer(&self) -> HeapPointer;
    fn get_index_pointer_to_neighbors(&self) -> Vec<ItemPointer>;
}

pub trait Storage {
    type QueryDistanceMeasure;
    type ArchivedType: ArchivedData;
    type NodeDistanceMeasure<'a>: NodeDistanceMeasure
    where
        Self: 'a;

    fn page_type(&self) -> PageType;

    fn create_node(
        &self,
        _index_relation: &PgRelation,
        full_vector: &[f32],
        heap_pointer: HeapPointer,
        meta_page: &MetaPage,
        tape: &mut Tape,
    ) -> ItemPointer;

    fn start_training(&mut self, meta_page: &super::meta_page::MetaPage);
    fn add_sample(&mut self, sample: &[f32]);
    fn finish_training(&mut self, index: &PgRelation, graph: &Graph, stats: &mut WriteStats);

    unsafe fn get_full_vector_distance_state<'a, S: StatsNodeRead>(
        &'a self,
        index: &PgRelation,
        index_pointer: IndexPointer,
        stats: &mut S,
    ) -> Self::NodeDistanceMeasure<'a>;

    fn get_search_distance_measure(
        &self,
        query: PgVector,
        calc_distance_with_quantizer: bool,
    ) -> Self::QueryDistanceMeasure;

    fn visit_lsn(
        &self,
        index: &PgRelation,
        lsr: &mut ListSearchResult<Self>,
        lsn_idx: usize,
        gns: &GraphNeighborStore,
    ) where
        Self: Sized;

    fn create_lsn_for_init_id(
        &self,
        lsr: &mut ListSearchResult<Self>,
        index: &PgRelation,
        index_pointer: ItemPointer,
    ) -> ListSearchNeighbor
    where
        Self: Sized;

    fn return_lsn(
        &self,
        index: &PgRelation,
        lsr: &mut ListSearchResult<Self>,
        idx: usize,
    ) -> (HeapPointer, IndexPointer)
    where
        Self: Sized;

    fn get_neighbors_with_full_vector_distances_from_disk<
        S: StatsNodeRead + StatsDistanceComparison,
    >(
        &self,
        index: &PgRelation,
        neighbors_of: ItemPointer,
        result: &mut Vec<NeighborWithDistance>,
        stats: &mut S,
    ) -> bool;

    fn set_neighbors_on_disk<S: StatsNodeModify + StatsNodeRead>(
        &self,
        index: &PgRelation,
        meta: &MetaPage,
        index_pointer: IndexPointer,
        neighbors: &[NeighborWithDistance],
        stats: &mut S,
    );
}

pub enum StorageType {
    BQ,
    PQ,
    None,
}
