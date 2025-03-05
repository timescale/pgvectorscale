use crate::access_method::node::{ReadableNode, WriteableNode};
use crate::access_method::PgRelation;
use crate::util::tape::Tape;
use crate::util::{ArchivedItemPointer, HeapPointer, ItemPointer, ReadableBuffer, WritableBuffer};
use pgrx::pg_sys::{InvalidBlockNumber, InvalidOffsetNumber};
use pgvectorscale_derive::{Readable, Writeable};
use rkyv::{vec::ArchivedVec, Archive, Deserialize, Serialize};
use std::pin::Pin;

use super::stats::{StatsNodeModify, StatsNodeRead, StatsNodeWrite};
use super::storage::NodeVacuum;
use super::{
    meta_page::MetaPage, neighbor_with_distance::NeighborWithDistance, sbq::SbqVectorElement,
    storage::ArchivedData,
};

/// A node in the SBQ index.  Currently just classic nodes, soon to feature labeled ones.
pub enum SbqNode {
    Classic(ClassicSbqNode),
}

#[derive(Archive, Deserialize, Serialize, Readable, Writeable)]
#[archive(check_bytes)]
pub struct ClassicSbqNode {
    pub heap_item_pointer: HeapPointer,
    pub bq_vector: Vec<u64>, // Don't use SbqVectorElement because we don't want to change the size in on-disk format by accident
    neighbor_index_pointers: Vec<ItemPointer>,
    _neighbor_vectors: Vec<Vec<u64>>, // No longer used, but kept for backwards compatibility
}

impl SbqNode {
    pub fn with_meta(
        heap_pointer: HeapPointer,
        meta_page: &MetaPage,
        bq_vector: &[SbqVectorElement],
    ) -> Self {
        Self::new(
            heap_pointer,
            meta_page.get_num_neighbors() as usize,
            meta_page.get_num_dimensions_to_index() as usize,
            bq_vector,
        )
    }

    fn new(
        heap_pointer: HeapPointer,
        num_neighbors: usize,
        _num_dimensions: usize,
        bq_vector: &[SbqVectorElement],
    ) -> Self {
        // always use vectors of num_neighbors in length because we never want the serialized size of a Node to change
        let neighbor_index_pointers: Vec<_> = (0..num_neighbors)
            .map(|_| ItemPointer::new(InvalidBlockNumber, InvalidOffsetNumber))
            .collect();

        SbqNode::Classic(ClassicSbqNode {
            heap_item_pointer: heap_pointer,
            bq_vector: bq_vector.to_vec(),
            neighbor_index_pointers,
            _neighbor_vectors: vec![],
        })
    }

    pub unsafe fn read<'a, S: StatsNodeRead>(
        index: &'a PgRelation,
        index_pointer: ItemPointer,
        stats: &mut S,
    ) -> ReadableSbqNode<'a> {
        ReadableSbqNode::Classic(ClassicSbqNode::read(index, index_pointer, stats))
    }

    pub unsafe fn modify<'a, S: StatsNodeModify>(
        index: &'a PgRelation,
        index_pointer: ItemPointer,
        stats: &mut S,
    ) -> WritableSbqNode<'a> {
        WritableSbqNode::Classic(ClassicSbqNode::modify(index, index_pointer, stats))
    }

    pub fn write<S: StatsNodeWrite>(&self, tape: &mut Tape, stats: &mut S) -> ItemPointer {
        match self {
            SbqNode::Classic(node) => node.write(tape, stats),
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

pub enum ReadableSbqNode<'a> {
    Classic(ReadableClassicSbqNode<'a>),
}

impl<'a> ReadableSbqNode<'a> {
    pub fn get_archived_node(&self) -> ArchivedSbqNode<'a> {
        match self {
            ReadableSbqNode::Classic(node) => ArchivedSbqNode::Classic(node.get_archived_node()),
        }
    }
}

pub enum WritableSbqNode<'a> {
    Classic(WritableClassicSbqNode<'a>),
}

impl<'a> WritableSbqNode<'a> {
    pub fn get_archived_node(&mut self) -> ArchivedMutSbqNode<'a> {
        match self {
            WritableSbqNode::Classic(node) => ArchivedMutSbqNode::Classic(node.get_archived_node()),
        }
    }

    pub fn commit(self) {
        match self {
            WritableSbqNode::Classic(node) => node.commit(),
        }
    }
}

pub enum ArchivedMutSbqNode<'a> {
    Classic(Pin<&'a mut ArchivedClassicSbqNode>),
}

pub enum ArchivedSbqNode<'a> {
    Classic(&'a ArchivedClassicSbqNode),
}

impl ArchivedSbqNode<'_> {
    pub fn num_neighbors(&self) -> usize {
        match self {
            ArchivedSbqNode::Classic(node) => node
                .neighbor_index_pointers
                .iter()
                .position(|f| f.block_number == InvalidBlockNumber)
                .unwrap_or(node.neighbor_index_pointers.len()),
        }
    }

    pub fn iter_neighbors(&self) -> impl Iterator<Item = ItemPointer> + '_ {
        match self {
            ArchivedSbqNode::Classic(node) => node
                .neighbor_index_pointers
                .iter()
                .take(self.num_neighbors())
                .map(|ip| ip.deserialize_item_pointer()),
        }
    }

    pub fn get_index_pointer_to_neighbors(&self) -> Vec<ItemPointer> {
        self.iter_neighbors().collect()
    }

    pub fn get_bq_vector(&self) -> &[u64] {
        match self {
            ArchivedSbqNode::Classic(node) => &node.bq_vector,
        }
    }

    pub fn get_heap_item_pointer(&self) -> HeapPointer {
        match self {
            ArchivedSbqNode::Classic(node) => node.heap_item_pointer.deserialize_item_pointer(),
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
        }
    }

    pub fn iter_neighbors(&self) -> impl Iterator<Item = ItemPointer> + '_ {
        match self {
            ArchivedMutSbqNode::Classic(node) => node
                .neighbor_index_pointers
                .iter()
                .take(self.num_neighbors())
                .map(|ip| ip.deserialize_item_pointer()),
        }
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
        };
        hip.deserialize_item_pointer()
    }
}
