use std::pin::Pin;

use pgrx::pg_sys::{InvalidBlockNumber, InvalidOffsetNumber};
use pgrx::*;
use pgvectorscale_derive::{Readable, Writeable};
use rkyv::vec::ArchivedVec;
use rkyv::{Archive, Deserialize, Serialize};

use super::labels::{label_vec_to_set, Label, LabelSet, INVALID_LABEL, MAX_LABELS_PER_NODE};
use super::neighbor_with_distance::NeighborWithDistance;
use super::storage::ArchivedData;
use crate::util::{ArchivedItemPointer, HeapPointer, ItemPointer, ReadableBuffer, WritableBuffer};

use super::meta_page::MetaPage;

#[derive(Archive, Deserialize, Serialize, Readable, Writeable)]
#[archive(check_bytes)]
pub struct Node {
    pub vector: Vec<f32>,
    pub pq_vector: Vec<u8>,
    neighbor_index_pointers: Vec<ItemPointer>,
    pub heap_item_pointer: HeapPointer,
    labels: LabelSet,
    neighbor_labels: Vec<LabelSet>,
}

impl Node {
    fn new_internal(
        vector: Vec<f32>,
        labels: Option<&[Label]>,
        pq_vector: Vec<u8>,
        heap_item_pointer: ItemPointer,
        meta_page: &MetaPage,
    ) -> Self {
        let num_neighbors = meta_page.get_num_neighbors();
        Self {
            vector,
            // always use vectors of num_clusters on length because we never want the serialized size of a Node to change
            pq_vector,
            // always use vectors of num_neighbors on length because we never want the serialized size of a Node to change
            neighbor_index_pointers: (0..num_neighbors)
                .map(|_| ItemPointer::new(InvalidBlockNumber, InvalidOffsetNumber))
                .collect(),
            heap_item_pointer,
            labels: label_vec_to_set(labels),
            neighbor_labels: (0..num_neighbors)
                .map(|_| [INVALID_LABEL; MAX_LABELS_PER_NODE])
                .collect(),
        }
    }

    pub fn new_for_full_vector(
        vector: Vec<f32>,
        labels: Option<&[Label]>,
        heap_item_pointer: ItemPointer,
        meta_page: &MetaPage,
    ) -> Self {
        let pq_vector = Vec::with_capacity(0);
        Self::new_internal(vector, labels, pq_vector, heap_item_pointer, meta_page)
    }
}

/// contains helpers for mutate-in-place. See struct_mutable_refs in test_alloc.rs in rkyv
impl ArchivedNode {
    pub fn is_deleted(&self) -> bool {
        self.heap_item_pointer.offset == InvalidOffsetNumber
    }

    pub fn delete(self: Pin<&mut Self>) {
        //TODO: actually optimize the deletes by removing index tuples. For now just mark it.
        let mut heap_pointer = unsafe { self.map_unchecked_mut(|s| &mut s.heap_item_pointer) };
        heap_pointer.offset = InvalidOffsetNumber;
        heap_pointer.block_number = InvalidBlockNumber;
    }

    pub fn neighbor_index_pointer(
        self: Pin<&mut Self>,
    ) -> Pin<&mut ArchivedVec<ArchivedItemPointer>> {
        unsafe { self.map_unchecked_mut(|s| &mut s.neighbor_index_pointers) }
    }

    pub fn neighbor_labels(self: Pin<&mut Self>) -> Pin<&mut ArchivedVec<LabelSet>> {
        unsafe { self.map_unchecked_mut(|s| &mut s.neighbor_labels) }
    }

    pub fn num_neighbors(&self) -> usize {
        self.neighbor_index_pointers
            .iter()
            .position(|f| f.block_number == InvalidBlockNumber)
            .unwrap_or(self.neighbor_index_pointers.len())
    }

    pub fn iter_neighbors(&self) -> impl Iterator<Item = ItemPointer> + '_ {
        self.neighbor_index_pointers
            .iter()
            .take(self.num_neighbors())
            .map(|ip| ip.deserialize_item_pointer())
    }

    pub fn iter_neighbor_labels(&self) -> impl Iterator<Item = &LabelSet> + '_ {
        self.neighbor_labels.iter().take(self.num_neighbors())
    }

    pub fn set_neighbors(
        mut self: Pin<&mut Self>,
        neighbors: &[NeighborWithDistance],
        meta_page: &MetaPage,
    ) {
        for (i, new_neighbor) in neighbors.iter().enumerate() {
            let mut a_index_pointer = self.as_mut().neighbor_index_pointer().index_pin(i);
            // TODO hate that we have to set each field like this
            a_index_pointer.block_number =
                new_neighbor.get_index_pointer_to_neighbor().block_number;
            a_index_pointer.offset = new_neighbor.get_index_pointer_to_neighbor().offset;

            let mut a_labels = self.as_mut().neighbor_labels().index_pin(i);
            a_labels.copy_from_slice(new_neighbor.get_labels());
        }
        // Set the marker that the list ended
        if neighbors.len() < meta_page.get_num_neighbors() as _ {
            let mut past_last_index_pointers = self
                .as_mut()
                .neighbor_index_pointer()
                .index_pin(neighbors.len());
            past_last_index_pointers.block_number = InvalidBlockNumber;
            past_last_index_pointers.offset = InvalidOffsetNumber;

            let mut past_last_labels = self.neighbor_labels().index_pin(neighbors.len());
            past_last_labels.copy_from_slice(&[INVALID_LABEL; MAX_LABELS_PER_NODE]);
        }
    }

    pub fn get_labels(&self) -> &LabelSet {
        &self.labels
    }
}

impl ArchivedData for ArchivedNode {
    fn with_data(data: &mut [u8]) -> Pin<&mut ArchivedNode> {
        ArchivedNode::with_data(data)
    }

    fn get_index_pointer_to_neighbors(&self) -> Vec<ItemPointer> {
        self.iter_neighbors().collect()
    }

    fn is_deleted(&self) -> bool {
        self.heap_item_pointer.offset == InvalidOffsetNumber
    }

    fn delete(self: Pin<&mut Self>) {
        //TODO: actually optimize the deletes by removing index tuples. For now just mark it.
        let mut heap_pointer = unsafe { self.map_unchecked_mut(|s| &mut s.heap_item_pointer) };
        heap_pointer.offset = InvalidOffsetNumber;
        heap_pointer.block_number = InvalidBlockNumber;
    }

    fn get_heap_item_pointer(&self) -> HeapPointer {
        self.heap_item_pointer.deserialize_item_pointer()
    }
}
