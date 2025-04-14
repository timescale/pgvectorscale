use std::pin::Pin;

use pgrx::{pg_sys, PgBox};

use crate::util::{page::PageType, tape::Tape, HeapPointer, IndexPointer, ItemPointer};

use super::{
    distance::DistanceFn,
    graph::{ListSearchNeighbor, ListSearchResult},
    graph_neighbor_store::GraphNeighborStore,
    labels::{LabelSet, LabeledVector},
    meta_page::MetaPage,
    neighbor_with_distance::NeighborWithDistance,
    stats::{
        GreedySearchStats, StatsDistanceComparison, StatsHeapNodeRead, StatsNodeModify,
        StatsNodeRead, StatsNodeWrite, WriteStats,
    },
};

/// NodeDistanceMeasure keeps the state to make distance comparisons between two nodes.
pub trait NodeDistanceMeasure {
    unsafe fn get_distance<S: StatsNodeRead + StatsDistanceComparison>(
        &self,
        index_pointer: IndexPointer,
        stats: &mut S,
    ) -> f32;
}
pub trait ArchivedData {
    fn is_deleted(&self) -> bool;
    fn get_heap_item_pointer(&self) -> HeapPointer;
    fn get_index_pointer_to_neighbors(&self) -> Vec<ItemPointer>;
}

pub trait NodeVacuum: ArchivedData {
    fn with_data(data: &mut [u8]) -> Pin<&mut Self>;
    fn delete(self: Pin<&mut Self>);
}

pub trait Storage {
    /// A QueryDistanceMeasure keeps the state to make distance comparison between a query given at initialization and a node.
    type QueryDistanceMeasure;
    /// A NodeDistanceMeasure keeps the state to make distance comparison between a node given at initialization and another node.
    type NodeDistanceMeasure<'a>: NodeDistanceMeasure
    where
        Self: 'a;
    type ArchivedType<'b>: ArchivedData
    where
        Self: 'b;
    type LSNPrivateData;

    fn page_type() -> PageType;

    fn create_node<S: StatsNodeWrite>(
        &self,
        full_vector: &[f32],
        labels: Option<LabelSet>,
        heap_pointer: HeapPointer,
        meta_page: &MetaPage,
        tape: &mut Tape,
        stats: &mut S,
    ) -> ItemPointer;

    fn start_training(&mut self, meta_page: &MetaPage);
    fn add_sample(&mut self, sample: &[f32]);
    fn finish_training(&mut self, meta_page: &mut MetaPage, stats: &mut WriteStats);

    fn finalize_node_at_end_of_build<S: StatsNodeRead + StatsNodeModify>(
        &mut self,
        meta: &MetaPage,
        index_pointer: IndexPointer,
        neighbors: &[NeighborWithDistance],
        stats: &mut S,
    );

    unsafe fn get_node_distance_measure<'a, S: StatsNodeRead>(
        &'a self,
        index_pointer: IndexPointer,
        stats: &mut S,
    ) -> Self::NodeDistanceMeasure<'a>;

    fn get_query_distance_measure(&self, query: LabeledVector) -> Self::QueryDistanceMeasure;

    fn get_full_distance_for_resort<S: StatsHeapNodeRead + StatsDistanceComparison>(
        &self,
        scan: &PgBox<pg_sys::IndexScanDescData>,
        query: &Self::QueryDistanceMeasure,
        index_pointer: IndexPointer,
        heap_pointer: HeapPointer,
        meta_page: &MetaPage,
        stats: &mut S,
    ) -> Option<f32>;

    fn visit_lsn(
        &self,
        lsr: &mut ListSearchResult<Self::QueryDistanceMeasure, Self::LSNPrivateData>,
        lsn_idx: usize,
        gns: &GraphNeighborStore,
        no_filter: bool,
    ) where
        Self: Sized;

    /// Create a ListSearchNeighbor for the start node of the search.  If start node
    /// already processed (e.g. because multiple labels use it), return None.
    fn create_lsn_for_start_node(
        &self,
        lsr: &mut ListSearchResult<Self::QueryDistanceMeasure, Self::LSNPrivateData>,
        index_pointer: ItemPointer,
        gns: &GraphNeighborStore,
    ) -> Option<ListSearchNeighbor<Self::LSNPrivateData>>
    where
        Self: Sized;

    fn return_lsn(
        &self,
        lsn: &ListSearchNeighbor<Self::LSNPrivateData>,
        stats: &mut GreedySearchStats,
    ) -> HeapPointer
    where
        Self: Sized;

    fn get_neighbors_with_distances_from_disk<S: StatsNodeRead + StatsDistanceComparison>(
        &self,
        neighbors_of: ItemPointer,
        result: &mut Vec<NeighborWithDistance>,
        stats: &mut S,
    );

    fn set_neighbors_on_disk<S: StatsNodeModify + StatsNodeRead>(
        &self,
        meta: &MetaPage,
        index_pointer: IndexPointer,
        neighbors: &[NeighborWithDistance],
        stats: &mut S,
    );

    fn get_distance_function(&self) -> DistanceFn;

    fn get_labels<S: StatsNodeRead>(
        &self,
        index_pointer: IndexPointer,
        stats: &mut S,
    ) -> Option<LabelSet>;
}

#[derive(PartialEq, Debug)]
pub enum StorageType {
    Plain = 0,
    // R.I.P. SbqSpeedup = 1,
    SbqCompression = 2,
}

pub const DEFAULT_STORAGE_TYPE_STR: &str = "memory_optimized";

impl StorageType {
    pub fn from_u8(value: u8) -> Self {
        match value {
            0 => StorageType::Plain,
            2 => StorageType::SbqCompression,
            _ => panic!("Invalid storage type"),
        }
    }

    pub fn from_str(value: &str) -> Self {
        match value.to_lowercase().as_str() {
            "plain" => StorageType::Plain,
            "bq_compression" | "memory_optimized" => StorageType::SbqCompression,
            _ => panic!("Invalid storage type. Must be either 'plain' or 'memory_optimized'"),
        }
    }
}
