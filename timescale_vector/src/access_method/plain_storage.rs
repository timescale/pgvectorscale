use super::{
    distance::distance_cosine as default_distance,
    graph::{ListSearchNeighbor, ListSearchResult},
    graph_neighbor_store::GraphNeighborStore,
    pg_vector::PgVector,
    plain_node::{ArchivedNode, Node, ReadableNode},
    stats::{
        GreedySearchStats, StatsDistanceComparison, StatsNodeModify, StatsNodeRead, StatsNodeWrite,
        WriteStats,
    },
    storage::{ArchivedData, NodeFullDistanceMeasure, Storage, StorageFullDistanceFromHeap},
    storage_common::{calculate_full_distance, HeapFullDistanceMeasure},
};
use std::{collections::HashMap, iter::once, pin::Pin};

use pgrx::{
    info,
    pg_sys::{AttrNumber, InvalidBlockNumber, InvalidOffsetNumber},
    PgRelation,
};
use rkyv::{vec::ArchivedVec, Archive, Archived, Deserialize, Serialize};

use crate::util::{
    page::PageType, table_slot::TableSlot, tape::Tape, ArchivedItemPointer, HeapPointer,
    IndexPointer, ItemPointer, ReadableBuffer,
};

use super::{meta_page::MetaPage, model::NeighborWithDistance};
use crate::util::WritableBuffer;

pub struct PlainStorage<'a> {
    pub index: &'a PgRelation,
    pub distance_fn: fn(&[f32], &[f32]) -> f32,
}

impl<'a> PlainStorage<'a> {
    pub fn new_for_build(index: &'a PgRelation) -> PlainStorage<'a> {
        Self {
            index: index,
            distance_fn: default_distance,
        }
    }

    pub fn load_for_insert(index_relation: &'a PgRelation) -> PlainStorage<'a> {
        Self {
            index: index_relation,
            distance_fn: default_distance,
        }
    }

    pub fn load_for_search(index_relation: &'a PgRelation) -> PlainStorage<'a> {
        Self {
            index: index_relation,
            distance_fn: default_distance,
        }
    }
}

pub enum PlainDistanceMeasure {
    Full(PgVector),
}

impl PlainDistanceMeasure {
    pub fn calculate_distance<S: StatsDistanceComparison>(
        distance_fn: fn(&[f32], &[f32]) -> f32,
        query: &[f32],
        vector: &[f32],
        stats: &mut S,
    ) -> f32 {
        assert!(vector.len() > 0);
        assert!(vector.len() == query.len());
        stats.record_full_distance_comparison();
        (distance_fn)(query, vector)
    }
}

/* This is only applicable to plain, so keep here not in storage_common */
pub struct IndexFullDistanceMeasure<'a> {
    readable_node: ReadableNode<'a>,
    storage: &'a PlainStorage<'a>,
}

impl<'a> IndexFullDistanceMeasure<'a> {
    pub unsafe fn new<T: StatsNodeRead>(
        storage: &'a PlainStorage<'a>,
        index_pointer: IndexPointer,
        stats: &mut T,
    ) -> Self {
        let rn = unsafe { Node::read(storage.index, index_pointer, stats) };
        Self {
            readable_node: rn,
            storage: storage,
        }
    }
}

impl<'a> NodeFullDistanceMeasure for IndexFullDistanceMeasure<'a> {
    unsafe fn get_distance<T: StatsNodeRead + StatsDistanceComparison>(
        &self,
        index_pointer: IndexPointer,
        stats: &mut T,
    ) -> f32 {
        let rn1 = Node::read(self.storage.index, index_pointer, stats);
        let rn2 = &self.readable_node;
        let node1 = rn1.get_archived_node();
        let node2 = rn2.get_archived_node();
        assert!(node1.vector.len() > 0);
        assert!(node1.vector.len() == node2.vector.len());
        let vec1 = node1.vector.as_slice();
        let vec2 = node2.vector.as_slice();
        (self.storage.get_distance_function())(vec1, vec2)
    }
}

impl<'a> Storage for PlainStorage<'a> {
    type QueryDistanceMeasure = PlainDistanceMeasure;
    type NodeFullDistanceMeasure<'b> = IndexFullDistanceMeasure<'b> where Self: 'b;
    type ArchivedType = ArchivedNode;

    fn page_type(&self) -> PageType {
        PageType::BqNode
    }

    fn create_node<S: StatsNodeWrite>(
        &self,
        full_vector: &[f32],
        heap_pointer: HeapPointer,
        meta_page: &MetaPage,
        tape: &mut Tape,
        stats: &mut S,
    ) -> ItemPointer {
        //OPT: avoid the clone?
        let node = Node::new_for_full_vector(full_vector.to_vec(), heap_pointer, meta_page);
        let index_pointer: IndexPointer = node.write(tape, stats);
        index_pointer
    }

    fn start_training(&mut self, meta_page: &super::meta_page::MetaPage) {}

    fn add_sample(&mut self, sample: &[f32]) {}

    fn finish_training(&mut self, stats: &mut WriteStats) {}

    fn finalize_node_at_end_of_build<S: StatsNodeRead + StatsNodeModify>(
        &mut self,
        meta: &MetaPage,
        index_pointer: IndexPointer,
        neighbors: &Vec<NeighborWithDistance>,
        stats: &mut S,
    ) {
        let node = unsafe { Node::modify(self.index, index_pointer, stats) };
        let mut archived = node.get_archived_node();
        archived.as_mut().set_neighbors(neighbors, &meta);
        node.commit();
    }

    unsafe fn get_full_vector_distance_state<'b, S: StatsNodeRead>(
        &'b self,
        index_pointer: IndexPointer,
        stats: &mut S,
    ) -> Self::NodeFullDistanceMeasure<'b> {
        IndexFullDistanceMeasure::new(self, index_pointer, stats)
    }

    fn get_search_distance_measure(
        &self,
        query: PgVector,
        calc_distance_with_quantizer: bool,
    ) -> PlainDistanceMeasure {
        return PlainDistanceMeasure::Full(query);
    }

    //TODO: this is a copy of the same function in BqStorage. Refactor.
    fn get_neighbors_with_full_vector_distances_from_disk<
        S: StatsNodeRead + StatsDistanceComparison,
    >(
        &self,
        neighbors_of: ItemPointer,
        result: &mut Vec<NeighborWithDistance>,
        stats: &mut S,
    ) -> bool {
        let rn = unsafe { Node::read(self.index, neighbors_of, stats) };
        let dist_state = unsafe { self.get_full_vector_distance_state(neighbors_of, stats) };
        rn.get_archived_node().apply_to_neighbors(|n| {
            let n = n.deserialize_item_pointer();
            let dist = unsafe { dist_state.get_distance(n, stats) };
            result.push(NeighborWithDistance::new(n, dist))
        });
        true
    }

    /* get_lsn and visit_lsn are different because the distance
    comparisons for BQ get the vector from different places */
    fn create_lsn_for_init_id(
        &self,
        lsr: &mut ListSearchResult<Self::QueryDistanceMeasure>,
        index_pointer: ItemPointer,
    ) -> ListSearchNeighbor {
        let rn = unsafe { Node::read(self.index, index_pointer, &mut lsr.stats) };
        let node = rn.get_archived_node();

        let distance = match lsr.sdm.as_ref().unwrap() {
            PlainDistanceMeasure::Full(query) => PlainDistanceMeasure::calculate_distance(
                self.distance_fn,
                query.to_slice(),
                node.vector.as_slice(),
                &mut lsr.stats,
            ),
        };

        if !lsr.prepare_insert(index_pointer) {
            panic!("should not have had an init id already inserted");
        }
        let lsn =
            ListSearchNeighbor::new(index_pointer, distance, super::graph::LsrPrivateData::None);

        lsn
    }

    fn visit_lsn(
        &self,
        lsr: &mut ListSearchResult<Self::QueryDistanceMeasure>,
        lsn_idx: usize,
        gns: &GraphNeighborStore,
    ) {
        //FIXME: shouldn't need to read the node again, should be in the private data
        let lsn_index_pointer = lsr.get_lsn_by_idx(lsn_idx).index_pointer;
        //Opt shouldn't need to read the node in the builder graph case.
        let rn_visiting = unsafe { Node::read(self.index, lsn_index_pointer, &mut lsr.stats) };
        let node_visiting = rn_visiting.get_archived_node();

        let neighbors = match gns {
            GraphNeighborStore::Disk => node_visiting.get_index_pointer_to_neighbors(),
            GraphNeighborStore::Builder(b) => b.get_neighbors(lsn_index_pointer),
        };

        for (i, &neighbor_index_pointer) in neighbors.iter().enumerate() {
            if !lsr.prepare_insert(neighbor_index_pointer) {
                continue;
            }

            let rn_neighbor =
                unsafe { Node::read(self.index, neighbor_index_pointer, &mut lsr.stats) };

            let distance = match lsr.sdm.as_ref().unwrap() {
                PlainDistanceMeasure::Full(query) => {
                    let rn_neighbor =
                        unsafe { Node::read(self.index, neighbor_index_pointer, &mut lsr.stats) };
                    let node_neighbor = rn_neighbor.get_archived_node();
                    PlainDistanceMeasure::calculate_distance(
                        self.distance_fn,
                        query.to_slice(),
                        node_neighbor.vector.as_slice(),
                        &mut lsr.stats,
                    )
                }
            };
            let lsn = ListSearchNeighbor::new(
                neighbor_index_pointer,
                distance,
                super::graph::LsrPrivateData::None,
            );

            lsr.insert_neighbor(lsn);
        }
    }

    fn return_lsn(&self, lsn: &ListSearchNeighbor, stats: &mut GreedySearchStats) -> HeapPointer {
        //FIXME: shouldn't need to read the node again, should be in the private data
        let lsn_index_pointer = lsn.index_pointer;
        let rn = unsafe { Node::read(self.index, lsn_index_pointer, stats) };
        let node = rn.get_archived_node();
        let heap_pointer = node.heap_item_pointer.deserialize_item_pointer();
        heap_pointer
    }

    fn set_neighbors_on_disk<S: StatsNodeModify + StatsNodeRead>(
        &self,
        meta: &MetaPage,
        index_pointer: IndexPointer,
        neighbors: &[NeighborWithDistance],
        stats: &mut S,
    ) {
        let node = unsafe { Node::modify(self.index, index_pointer, stats) };
        let mut archived = node.get_archived_node();
        archived.as_mut().set_neighbors(neighbors, &meta);
        node.commit();
    }

    fn get_distance_function(&self) -> fn(&[f32], &[f32]) -> f32 {
        self.distance_fn
    }
}

#[cfg(any(test, feature = "pg_test"))]
#[pgrx::pg_schema]
mod tests {

    use pgrx::*;

    #[pg_test]
    unsafe fn test_plain_storage_index_creation() -> spi::Result<()> {
        crate::access_method::build::tests::test_index_creation_and_accuracy_scaffold(
            "num_neighbors=38",
        )?;
        Ok(())
    }

    #[pg_test]
    unsafe fn test_plain_storage_index_creation_few_neighbors() -> spi::Result<()> {
        //a test with few neighbors tests the case that nodes share a page, which has caused deadlocks in the past.
        crate::access_method::build::tests::test_index_creation_and_accuracy_scaffold(
            "num_neighbors=10",
        )?;
        Ok(())
    }

    #[test]
    fn test_plain_storage_delete_vacuum_plain() {
        crate::access_method::vacuum::tests::test_delete_vacuum_plain_scaffold(
            "num_neighbors = 10",
        );
    }

    #[test]
    fn test_plain_storage_delete_vacuum_full() {
        crate::access_method::vacuum::tests::test_delete_vacuum_full_scaffold("num_neighbors = 38");
    }

    #[pg_test]
    unsafe fn test_plain_storage_empty_table_insert() -> spi::Result<()> {
        crate::access_method::build::tests::test_empty_table_insert_scaffold("num_neighbors=38")
    }

    #[pg_test]
    unsafe fn test_plain_storage_insert_empty_insert() -> spi::Result<()> {
        crate::access_method::build::tests::test_insert_empty_insert_scaffold("num_neighbors=38")
    }
}
