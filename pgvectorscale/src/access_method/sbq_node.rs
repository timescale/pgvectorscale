use crate::access_method::node::{ReadableNode, WriteableNode};
use crate::access_method::PgRelation;
use crate::util::tape::Tape;
use crate::util::{ArchivedItemPointer, HeapPointer, ItemPointer, ReadableBuffer, WritableBuffer};
use pgrx::pg_sys::{InvalidBlockNumber, InvalidOffsetNumber};
use pgvectorscale_derive::{Readable, Writeable};
use rkyv::{vec::ArchivedVec, Archive, Deserialize, Serialize};
use std::fmt::Debug;
use std::pin::Pin;

use super::labels::{ArchivedLabelSet, LabelSet};
use super::stats::{StatsNodeModify, StatsNodeRead, StatsNodeWrite};
use super::storage::NodeVacuum;
use super::{
    meta_page::MetaPage, neighbor_with_distance::NeighborWithDistance, sbq::SbqVectorElement,
    storage::ArchivedData,
};

/// A node in the SBQ index
pub enum SbqNode {
    Classic(ClassicSbqNode),
    Labeled(LabeledSbqNode),
}

#[derive(Archive, Deserialize, Serialize, Readable, Writeable)]
#[archive(check_bytes)]
pub struct ClassicSbqNode {
    pub heap_item_pointer: HeapPointer,
    pub bq_vector: Vec<u64>, // Don't use SbqVectorElement because we don't want to change the size in on-disk format by accident
    neighbor_index_pointers: Vec<ItemPointer>,
    _neighbor_vectors: Vec<Vec<u64>>, // No longer used, but kept for backwards compatibility
}

#[derive(Archive, Deserialize, Serialize, Readable, Writeable)]
#[archive(check_bytes)]
pub struct LabeledSbqNode {
    heap_item_pointer: HeapPointer,
    bq_vector: Vec<u64>,
    neighbor_index_pointers: Vec<ItemPointer>,
    labels: LabelSet,
}

impl SbqNode {
    pub fn with_meta(
        heap_pointer: HeapPointer,
        meta_page: &MetaPage,
        bq_vector: &[SbqVectorElement],
        labels: Option<LabelSet>,
    ) -> Self {
        Self::new(
            heap_pointer,
            meta_page.get_num_neighbors() as usize,
            meta_page.has_labels(),
            bq_vector,
            labels,
        )
    }

    fn new(
        heap_pointer: HeapPointer,
        num_neighbors: usize,
        has_labels: bool,
        bq_vector: &[SbqVectorElement],
        labels: Option<LabelSet>,
    ) -> Self {
        // always use vectors of num_neighbors in length because we never want the serialized size of a Node to change
        let neighbor_index_pointers: Vec<_> = (0..num_neighbors)
            .map(|_| ItemPointer::new(InvalidBlockNumber, InvalidOffsetNumber))
            .collect();

        if has_labels {
            SbqNode::Labeled(LabeledSbqNode {
                heap_item_pointer: heap_pointer,
                bq_vector: bq_vector.to_vec(),
                neighbor_index_pointers,
                labels: labels.unwrap_or_default(),
            })
        } else {
            SbqNode::Classic(ClassicSbqNode {
                heap_item_pointer: heap_pointer,
                bq_vector: bq_vector.to_vec(),
                neighbor_index_pointers,
                _neighbor_vectors: vec![],
            })
        }
    }

    pub unsafe fn read<'a, S: StatsNodeRead>(
        index: &'a PgRelation,
        index_pointer: ItemPointer,
        has_labels: bool,
        stats: &mut S,
    ) -> ReadableSbqNode<'a> {
        if has_labels {
            ReadableSbqNode::Labeled(LabeledSbqNode::read(index, index_pointer, stats))
        } else {
            ReadableSbqNode::Classic(ClassicSbqNode::read(index, index_pointer, stats))
        }
    }

    pub unsafe fn modify<'a, S: StatsNodeModify>(
        index: &'a PgRelation,
        index_pointer: ItemPointer,
        has_labels: bool,
        stats: &mut S,
    ) -> WritableSbqNode<'a> {
        if has_labels {
            WritableSbqNode::Labeled(LabeledSbqNode::modify(index, index_pointer, stats))
        } else {
            WritableSbqNode::Classic(ClassicSbqNode::modify(index, index_pointer, stats))
        }
    }

    pub fn write<S: StatsNodeWrite>(&self, tape: &mut Tape, stats: &mut S) -> ItemPointer {
        match self {
            SbqNode::Classic(node) => node.write(tape, stats),
            SbqNode::Labeled(node) => node.write(tape, stats),
        }
    }
}

impl NodeVacuum for ArchivedClassicSbqNode {
    fn with_data(data: &mut [u8]) -> Pin<&mut Self> {
        ArchivedClassicSbqNode::with_data(data)
    }

    fn delete(self: Pin<&mut Self>) {
        //TODO: actually optimize the deletes by removing index tuples. For now just mark it.
        let mut heap_pointer = unsafe { self.map_unchecked_mut(|s| &mut s.heap_item_pointer) };
        heap_pointer.offset = InvalidOffsetNumber;
        heap_pointer.block_number = InvalidBlockNumber;
    }
}

impl NodeVacuum for ArchivedLabeledSbqNode {
    fn with_data(data: &mut [u8]) -> Pin<&mut Self> {
        ArchivedLabeledSbqNode::with_data(data)
    }

    fn delete(self: Pin<&mut Self>) {
        //TODO: actually optimize the deletes by removing index tuples. For now just mark it.
        let mut heap_pointer = unsafe { self.map_unchecked_mut(|s| &mut s.heap_item_pointer) };
        heap_pointer.offset = InvalidOffsetNumber;
        heap_pointer.block_number = InvalidBlockNumber;
    }
}

impl ArchivedData for ArchivedClassicSbqNode {
    fn is_deleted(&self) -> bool {
        self.heap_item_pointer.offset == InvalidOffsetNumber
    }

    fn get_heap_item_pointer(&self) -> HeapPointer {
        self.heap_item_pointer.deserialize_item_pointer()
    }

    fn get_index_pointer_to_neighbors(&self) -> Vec<ItemPointer> {
        self.neighbor_index_pointers
            .iter()
            .map(|p| p.deserialize_item_pointer())
            .collect()
    }
}

impl ArchivedData for ArchivedLabeledSbqNode {
    fn is_deleted(&self) -> bool {
        self.heap_item_pointer.offset == InvalidOffsetNumber
    }

    fn get_heap_item_pointer(&self) -> HeapPointer {
        self.heap_item_pointer.deserialize_item_pointer()
    }

    fn get_index_pointer_to_neighbors(&self) -> Vec<ItemPointer> {
        self.neighbor_index_pointers
            .iter()
            .map(|p| p.deserialize_item_pointer())
            .collect()
    }
}

pub enum ReadableSbqNode<'a> {
    Classic(ReadableClassicSbqNode<'a>),
    Labeled(ReadableLabeledSbqNode<'a>),
}

impl<'a> ReadableSbqNode<'a> {
    pub fn get_archived_node(&'a self) -> ArchivedSbqNode<'a> {
        match self {
            ReadableSbqNode::Classic(node) => ArchivedSbqNode::Classic(node.get_archived_node()),
            ReadableSbqNode::Labeled(node) => ArchivedSbqNode::Labeled(node.get_archived_node()),
        }
    }
}

pub enum WritableSbqNode<'a> {
    Classic(WritableClassicSbqNode<'a>),
    Labeled(WritableLabeledSbqNode<'a>),
}

impl WritableSbqNode<'_> {
    pub fn get_archived_node(&mut self) -> ArchivedMutSbqNode<'_> {
        match self {
            WritableSbqNode::Classic(node) => ArchivedMutSbqNode::Classic(node.get_archived_node()),
            WritableSbqNode::Labeled(node) => ArchivedMutSbqNode::Labeled(node.get_archived_node()),
        }
    }

    pub fn commit(self) {
        match self {
            WritableSbqNode::Classic(node) => node.commit(),
            WritableSbqNode::Labeled(node) => node.commit(),
        }
    }
}

pub enum ArchivedMutSbqNode<'a> {
    Classic(Pin<&'a mut ArchivedClassicSbqNode>),
    Labeled(Pin<&'a mut ArchivedLabeledSbqNode>),
}

pub enum ArchivedSbqNode<'a> {
    Classic(&'a ArchivedClassicSbqNode),
    Labeled(&'a ArchivedLabeledSbqNode),
}

impl ArchivedData for ArchivedSbqNode<'_> {
    fn is_deleted(&self) -> bool {
        match self {
            ArchivedSbqNode::Classic(node) => node.is_deleted(),
            ArchivedSbqNode::Labeled(node) => node.is_deleted(),
        }
    }

    fn get_heap_item_pointer(&self) -> HeapPointer {
        match self {
            ArchivedSbqNode::Classic(node) => node.get_heap_item_pointer(),
            ArchivedSbqNode::Labeled(node) => node.get_heap_item_pointer(),
        }
    }

    fn get_index_pointer_to_neighbors(&self) -> Vec<ItemPointer> {
        match self {
            ArchivedSbqNode::Classic(node) => node.get_index_pointer_to_neighbors(),
            ArchivedSbqNode::Labeled(node) => node.get_index_pointer_to_neighbors(),
        }
    }
}

impl Debug for ArchivedSbqNode<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArchivedSbqNode::Classic(node) => node.fmt(f),
            ArchivedSbqNode::Labeled(node) => node.fmt(f),
        }
    }
}

impl Debug for ArchivedClassicSbqNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArchivedClassicSbqNode")
            .field(
                "heap_item_pointer.block_number",
                &self.heap_item_pointer.block_number,
            )
            .field("heap_item_pointer.offset", &self.heap_item_pointer.offset)
            .field("bq_vector", &self.bq_vector)
            .field(
                "neighbor_index_pointers.len()",
                &self.neighbor_index_pointers.len(),
            )
            .finish()
    }
}

impl Debug for ArchivedLabeledSbqNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArchivedLabeledSbqNode")
            .field(
                "heap_item_pointer.block_number",
                &self.heap_item_pointer.block_number,
            )
            .field("heap_item_pointer.offset", &self.heap_item_pointer.offset)
            .field("bq_vector", &self.bq_vector)
            .field(
                "neighbor_index_pointers.len()",
                &self.neighbor_index_pointers.len(),
            )
            .field("labels", &self.labels)
            .finish()
    }
}

impl ArchivedSbqNode<'_> {
    pub fn num_neighbors(&self) -> usize {
        match self {
            ArchivedSbqNode::Classic(node) => node
                .neighbor_index_pointers
                .iter()
                .position(|f| f.block_number == InvalidBlockNumber)
                .unwrap_or(node.neighbor_index_pointers.len()),
            ArchivedSbqNode::Labeled(node) => node
                .neighbor_index_pointers
                .iter()
                .position(|f| f.block_number == InvalidBlockNumber)
                .unwrap_or(node.neighbor_index_pointers.len()),
        }
    }

    pub fn iter_neighbors(&self) -> impl Iterator<Item = ItemPointer> + '_ {
        let neighbor_index_pointers = match self {
            ArchivedSbqNode::Classic(node) => &node.neighbor_index_pointers,
            ArchivedSbqNode::Labeled(node) => &node.neighbor_index_pointers,
        };
        neighbor_index_pointers
            .iter()
            .take(self.num_neighbors())
            .map(|ip| ip.deserialize_item_pointer())
    }

    pub fn get_index_pointer_to_neighbors(&self) -> Vec<ItemPointer> {
        self.iter_neighbors().collect()
    }

    pub fn get_bq_vector(&self) -> &[SbqVectorElement] {
        match self {
            ArchivedSbqNode::Classic(node) => &node.bq_vector,
            ArchivedSbqNode::Labeled(node) => &node.bq_vector,
        }
    }

    pub fn get_heap_item_pointer(&self) -> HeapPointer {
        match self {
            ArchivedSbqNode::Classic(node) => node.heap_item_pointer.deserialize_item_pointer(),
            ArchivedSbqNode::Labeled(node) => node.heap_item_pointer.deserialize_item_pointer(),
        }
    }

    pub fn get_labels(&self) -> Option<&ArchivedLabelSet> {
        match self {
            ArchivedSbqNode::Classic(_) => None,
            ArchivedSbqNode::Labeled(node) => Some(&node.labels),
        }
    }
}

impl<'a> ArchivedMutSbqNode<'a> {
    fn neighbor_index_pointer(&'a mut self) -> Pin<&'a mut ArchivedVec<ArchivedItemPointer>> {
        match self {
            ArchivedMutSbqNode::Classic(node) => unsafe {
                node.as_mut()
                    .map_unchecked_mut(|s| &mut s.neighbor_index_pointers)
            },
            ArchivedMutSbqNode::Labeled(node) => unsafe {
                node.as_mut()
                    .map_unchecked_mut(|s| &mut s.neighbor_index_pointers)
            },
        }
    }

    pub fn set_neighbors(&'a mut self, neighbors: &[NeighborWithDistance], meta_page: &MetaPage) {
        let mut neighbor_index_pointer = self.neighbor_index_pointer();
        for (i, new_neighbor) in neighbors.iter().enumerate() {
            let mut a_index_pointer = neighbor_index_pointer.as_mut().index_pin(i);
            let ip = new_neighbor.get_index_pointer_to_neighbor();
            //TODO hate that we have to set each field like this
            a_index_pointer.block_number = ip.block_number;
            a_index_pointer.offset = ip.offset;
        }
        //set the marker that the list ended
        if neighbors.len() < meta_page.get_num_neighbors() as _ {
            let mut past_last_index_pointers = neighbor_index_pointer.index_pin(neighbors.len());
            past_last_index_pointers.block_number = InvalidBlockNumber;
            past_last_index_pointers.offset = InvalidOffsetNumber;
        }
    }

    pub fn num_neighbors(&self) -> usize {
        match self {
            ArchivedMutSbqNode::Classic(node) => node
                .neighbor_index_pointers
                .iter()
                .position(|f| f.block_number == InvalidBlockNumber)
                .unwrap_or(node.neighbor_index_pointers.len()),
            ArchivedMutSbqNode::Labeled(node) => node
                .neighbor_index_pointers
                .iter()
                .position(|f| f.block_number == InvalidBlockNumber)
                .unwrap_or(node.neighbor_index_pointers.len()),
        }
    }

    pub fn iter_neighbors(&self) -> impl Iterator<Item = ItemPointer> + '_ {
        let neighbor_index_pointers = match self {
            ArchivedMutSbqNode::Classic(node) => &node.neighbor_index_pointers,
            ArchivedMutSbqNode::Labeled(node) => &node.neighbor_index_pointers,
        };

        neighbor_index_pointers
            .iter()
            .take(self.num_neighbors())
            .map(|ip| ip.deserialize_item_pointer())
    }
}

impl ArchivedData for ArchivedMutSbqNode<'_> {
    fn get_index_pointer_to_neighbors(&self) -> Vec<ItemPointer> {
        self.iter_neighbors().collect()
    }

    fn is_deleted(&self) -> bool {
        self.get_heap_item_pointer().offset == InvalidOffsetNumber
    }

    fn get_heap_item_pointer(&self) -> HeapPointer {
        let hip = match self {
            ArchivedMutSbqNode::Classic(node) => &node.heap_item_pointer,
            ArchivedMutSbqNode::Labeled(node) => &node.heap_item_pointer,
        };
        hip.deserialize_item_pointer()
    }
}
