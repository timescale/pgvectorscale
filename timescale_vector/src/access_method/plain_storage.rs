use super::{
    graph::{ListSearchNeighbor, ListSearchResult},
    graph_neighbor_store::GraphNeighborStore,
    pg_vector::PgVector,
    plain_node::{ArchivedNode, Node, ReadableNode},
    stats::{
        GreedySearchStats, StatsDistanceComparison, StatsHeapNodeRead, StatsNodeModify,
        StatsNodeRead, StatsNodeWrite, WriteStats,
    },
    storage::{ArchivedData, NodeDistanceMeasure, Storage},
    storage_common::get_attribute_number_from_index,
};

use pgrx::PgRelation;

use crate::util::{
    page::PageType, table_slot::TableSlot, tape::Tape, HeapPointer, IndexPointer, ItemPointer,
};

use super::{meta_page::MetaPage, neighbor_with_distance::NeighborWithDistance};

pub struct PlainStorage<'a> {
    pub index: &'a PgRelation,
    pub distance_fn: fn(&[f32], &[f32]) -> f32,
    heap_rel: &'a PgRelation,
    heap_attr: pgrx::pg_sys::AttrNumber,
}

impl<'a> PlainStorage<'a> {
    pub fn new_for_build(
        index: &'a PgRelation,
        heap_rel: &'a PgRelation,
        distance_fn: fn(&[f32], &[f32]) -> f32,
    ) -> PlainStorage<'a> {
        Self {
            index: index,
            distance_fn: distance_fn,
            heap_rel: heap_rel,
            heap_attr: get_attribute_number_from_index(index),
        }
    }

    pub fn load_for_insert(
        index_relation: &'a PgRelation,
        heap_rel: &'a PgRelation,
        distance_fn: fn(&[f32], &[f32]) -> f32,
    ) -> PlainStorage<'a> {
        Self {
            index: index_relation,
            distance_fn: distance_fn,
            heap_rel: heap_rel,
            heap_attr: get_attribute_number_from_index(&index_relation),
        }
    }

    pub fn load_for_search(
        index_relation: &'a PgRelation,
        heap_rel: &'a PgRelation,
        distance_fn: fn(&[f32], &[f32]) -> f32,
    ) -> PlainStorage<'a> {
        Self {
            index: index_relation,
            distance_fn: distance_fn,
            heap_rel: heap_rel,
            heap_attr: get_attribute_number_from_index(&index_relation),
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
    pub unsafe fn with_index_pointer<T: StatsNodeRead>(
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

    pub unsafe fn with_readable_node(
        storage: &'a PlainStorage<'a>,
        readable_node: ReadableNode<'a>,
    ) -> Self {
        Self {
            readable_node: readable_node,
            storage: storage,
        }
    }
}

impl<'a> NodeDistanceMeasure for IndexFullDistanceMeasure<'a> {
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

//todo move to storage_common
pub struct PlainStorageLsnPrivateData {
    pub heap_pointer: HeapPointer,
    pub neighbors: Vec<ItemPointer>,
}

impl PlainStorageLsnPrivateData {
    pub fn new(
        index_pointer_to_node: IndexPointer,
        node: &ArchivedNode,
        gns: &GraphNeighborStore,
    ) -> Self {
        let heap_pointer = node.heap_item_pointer.deserialize_item_pointer();
        let neighbors = match gns {
            GraphNeighborStore::Disk => node.get_index_pointer_to_neighbors(),
            GraphNeighborStore::Builder(b) => b.get_neighbors(index_pointer_to_node),
        };
        Self {
            heap_pointer: heap_pointer,
            neighbors: neighbors,
        }
    }
}

impl<'a> Storage for PlainStorage<'a> {
    type QueryDistanceMeasure = PlainDistanceMeasure;
    type NodeDistanceMeasure<'b> = IndexFullDistanceMeasure<'b> where Self: 'b;
    type ArchivedType = ArchivedNode;
    type LSNPrivateData = PlainStorageLsnPrivateData;

    fn page_type() -> PageType {
        PageType::Node
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

    fn start_training(&mut self, _meta_page: &super::meta_page::MetaPage) {}

    fn add_sample(&mut self, _sample: &[f32]) {}

    fn finish_training(&mut self, _stats: &mut WriteStats) {}

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

    unsafe fn get_node_distance_measure<'b, S: StatsNodeRead>(
        &'b self,
        index_pointer: IndexPointer,
        stats: &mut S,
    ) -> Self::NodeDistanceMeasure<'b> {
        IndexFullDistanceMeasure::with_index_pointer(self, index_pointer, stats)
    }

    fn get_query_distance_measure(&self, query: PgVector) -> PlainDistanceMeasure {
        return PlainDistanceMeasure::Full(query);
    }
    fn get_full_distance_for_resort<S: StatsHeapNodeRead + StatsDistanceComparison>(
        &self,
        qdm: &Self::QueryDistanceMeasure,
        _index_pointer: IndexPointer,
        heap_pointer: HeapPointer,
        meta_page: &MetaPage,
        stats: &mut S,
    ) -> f32 {
        /* Plain storage only needs to resort when the index is using less dimensions than the underlying data. */
        assert!(meta_page.get_num_dimensions() > meta_page.get_num_dimensions_to_index());

        let slot = unsafe { TableSlot::new(self.heap_rel, heap_pointer, stats) };
        match qdm {
            PlainDistanceMeasure::Full(query) => {
                let datum = unsafe { slot.get_attribute(self.heap_attr).unwrap() };
                let vec = unsafe { PgVector::from_datum(datum, meta_page, false, true) };
                self.get_distance_function()(vec.to_full_slice(), query.to_full_slice())
            }
        }
    }
    fn get_neighbors_with_distances_from_disk<S: StatsNodeRead + StatsDistanceComparison>(
        &self,
        neighbors_of: ItemPointer,
        result: &mut Vec<NeighborWithDistance>,
        stats: &mut S,
    ) {
        let rn = unsafe { Node::read(self.index, neighbors_of, stats) };
        //get neighbors copy before givining ownership of rn to the distance state
        let neighbors: Vec<_> = rn.get_archived_node().iter_neighbors().collect();
        let dist_state = unsafe { IndexFullDistanceMeasure::with_readable_node(self, rn) };
        for n in neighbors {
            let dist = unsafe { dist_state.get_distance(n, stats) };
            result.push(NeighborWithDistance::new(n, dist))
        }
    }

    /* get_lsn and visit_lsn are different because the distance
    comparisons for SBQ get the vector from different places */
    fn create_lsn_for_init_id(
        &self,
        lsr: &mut ListSearchResult<Self::QueryDistanceMeasure, Self::LSNPrivateData>,
        index_pointer: ItemPointer,
        gns: &GraphNeighborStore,
    ) -> ListSearchNeighbor<Self::LSNPrivateData> {
        if !lsr.prepare_insert(index_pointer) {
            panic!("should not have had an init id already inserted");
        }

        let rn = unsafe { Node::read(self.index, index_pointer, &mut lsr.stats) };
        let node = rn.get_archived_node();

        let distance = match lsr.sdm.as_ref().unwrap() {
            PlainDistanceMeasure::Full(query) => PlainDistanceMeasure::calculate_distance(
                self.distance_fn,
                query.to_index_slice(),
                node.vector.as_slice(),
                &mut lsr.stats,
            ),
        };

        ListSearchNeighbor::new(
            index_pointer,
            distance,
            PlainStorageLsnPrivateData::new(index_pointer, node, gns),
        )
    }

    fn visit_lsn(
        &self,
        lsr: &mut ListSearchResult<Self::QueryDistanceMeasure, Self::LSNPrivateData>,
        lsn_idx: usize,
        gns: &GraphNeighborStore,
    ) {
        let lsn = lsr.get_lsn_by_idx(lsn_idx);
        //clone needed so we don't continue to borrow lsr
        let neighbors = lsn.get_private_data().neighbors.clone();

        for &neighbor_index_pointer in neighbors.iter() {
            if !lsr.prepare_insert(neighbor_index_pointer) {
                continue;
            }

            let rn_neighbor =
                unsafe { Node::read(self.index, neighbor_index_pointer, &mut lsr.stats) };
            let node_neighbor = rn_neighbor.get_archived_node();

            let distance = match lsr.sdm.as_ref().unwrap() {
                PlainDistanceMeasure::Full(query) => PlainDistanceMeasure::calculate_distance(
                    self.distance_fn,
                    query.to_index_slice(),
                    node_neighbor.vector.as_slice(),
                    &mut lsr.stats,
                ),
            };
            let lsn = ListSearchNeighbor::new(
                neighbor_index_pointer,
                distance,
                PlainStorageLsnPrivateData::new(neighbor_index_pointer, node_neighbor, gns),
            );

            lsr.insert_neighbor(lsn);
        }
    }

    fn return_lsn(
        &self,
        lsn: &ListSearchNeighbor<Self::LSNPrivateData>,
        _stats: &mut GreedySearchStats,
    ) -> HeapPointer {
        lsn.get_private_data().heap_pointer
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
    unsafe fn test_plain_storage_index_creation_many_neighbors() -> spi::Result<()> {
        crate::access_method::build::tests::test_index_creation_and_accuracy_scaffold(
            "num_neighbors=38, storage_layout = plain",
        )?;
        Ok(())
    }

    #[pg_test]
    unsafe fn test_plain_storage_index_creation_few_neighbors() -> spi::Result<()> {
        //a test with few neighbors tests the case that nodes share a page, which has caused deadlocks in the past.
        crate::access_method::build::tests::test_index_creation_and_accuracy_scaffold(
            "num_neighbors=10, storage_layout = plain",
        )?;
        Ok(())
    }

    #[test]
    fn test_plain_storage_delete_vacuum_plain() {
        crate::access_method::vacuum::tests::test_delete_vacuum_plain_scaffold(
            "num_neighbors = 38, storage_layout = plain",
        );
    }

    #[test]
    fn test_plain_storage_delete_vacuum_full() {
        crate::access_method::vacuum::tests::test_delete_vacuum_full_scaffold(
            "num_neighbors = 38, storage_layout = plain",
        );
    }

    #[pg_test]
    unsafe fn test_plain_storage_empty_table_insert() -> spi::Result<()> {
        crate::access_method::build::tests::test_empty_table_insert_scaffold(
            "num_neighbors=38, storage_layout = plain",
        )
    }

    #[pg_test]
    unsafe fn test_plain_storage_insert_empty_insert() -> spi::Result<()> {
        crate::access_method::build::tests::test_insert_empty_insert_scaffold(
            "num_neighbors=38, storage_layout = plain",
        )
    }

    #[pg_test]
    unsafe fn test_plain_storage_num_dimensions() -> spi::Result<()> {
        crate::access_method::build::tests::test_index_creation_and_accuracy_scaffold(
            "num_neighbors=38, storage_layout = plain, num_dimensions=768",
        )?;
        Ok(())
    }

    #[pg_test]
    unsafe fn test_plain_storage_index_updates() -> spi::Result<()> {
        crate::access_method::build::tests::test_index_updates(
            "storage_layout = plain, num_neighbors=30",
            50,
        )?;
        Ok(())
    }
}
