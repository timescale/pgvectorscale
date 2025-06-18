use crate::access_method::node::{ReadableNode, WriteableNode};
use crate::access_method::PgRelation;
use crate::util::page::PageType;
use crate::util::tape::Tape;
use crate::util::{HeapPointer, ItemPointer, ReadableBuffer, WritableBuffer};
use pgrx::pg_sys::{InvalidBlockNumber, InvalidOffsetNumber};
use pgvectorscale_derive::{Readable, Writeable};
use rkyv::{Archive, Deserialize, Serialize};
use std::fmt::Debug;
use std::pin::Pin;

use crate::access_method::{
    graph::neighbor_with_distance::NeighborWithDistance,
    labels::{ArchivedLabelSet, LabelSet},
    meta_page::MetaPage,
    stats::{StatsNodeModify, StatsNodeRead, StatsNodeWrite},
    storage::{ArchivedData, NodeVacuum},
};

use super::SbqDiskVectorElement;

/// A node in the SBQ index
pub enum SbqDiskNode {
    Classic(ClassicSbqDiskNode),
    Labeled(LabeledSbqDiskNode),
}

#[derive(Archive, Deserialize, Serialize, Readable, Writeable)]
#[archive(check_bytes)]
pub struct ClassicSbqDiskNode {
    pub heap_item_pointer: HeapPointer,
    pub bq_vector: Vec<u64>, // Don't use SbqDiskVectorElement because we don't want to change the size in on-disk format by accident
    neighbor_node_pointer: Option<ItemPointer>,
}

#[derive(Archive, Deserialize, Serialize, Readable, Writeable)]
#[archive(check_bytes)]
pub struct ClassicSbqDiskNeighborNode {
    neighbor_index_pointers: Vec<ItemPointer>,
}

impl ClassicSbqDiskNeighborNode {
    pub fn new(num_neighbors: usize) -> Self {
        let neighbor_index_pointers: Vec<_> = (0..num_neighbors)
            .map(|_| ItemPointer::new(InvalidBlockNumber, InvalidOffsetNumber))
            .collect();

        ClassicSbqDiskNeighborNode {
            neighbor_index_pointers,
        }
    }
}

#[derive(Archive, Deserialize, Serialize, Readable, Writeable)]
#[archive(check_bytes)]
pub struct LabeledSbqDiskNode {
    heap_item_pointer: HeapPointer,
    bq_vector: Vec<u64>,
    neighbor_index_pointers: Vec<ItemPointer>,
    labels: LabelSet,
}

impl SbqDiskNode {
    pub fn with_meta(
        heap_pointer: HeapPointer,
        meta_page: &MetaPage,
        bq_vector: &[SbqDiskVectorElement],
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

    pub fn write_with_meta<S: StatsNodeWrite>(
        &mut self,
        tape: &mut Tape,
        meta_page: &MetaPage,
        stats: &mut S,
    ) -> ItemPointer {
        match self {
            SbqDiskNode::Classic(node) => {
                // For Classic nodes, create the neighbor node first
                let neighbor_node =
                    ClassicSbqDiskNeighborNode::new(meta_page.get_num_neighbors() as usize);

                // Write the neighbor node using resume to try to reuse existing pages
                let mut neighbor_tape =
                    unsafe { Tape::resume(tape.get_relation(), PageType::SbqDiskNeighborNode) };
                let neighbor_pointer = neighbor_node.write(&mut neighbor_tape, stats);

                node.neighbor_node_pointer = Some(neighbor_pointer);

                assert!(node.bq_vector.len() > 0);

                // Write the main node using the WriteableNode trait
                WriteableNode::write(node, tape, stats)
            }
            SbqDiskNode::Labeled(node) => node.write(tape, stats),
        }
    }

    fn new(
        heap_pointer: HeapPointer,
        num_neighbors: usize,
        has_labels: bool,
        bq_vector: &[SbqDiskVectorElement],
        labels: Option<LabelSet>,
    ) -> Self {
        if has_labels {
            // always use vectors of num_neighbors in length because we never want the serialized size of a Node to change
            let neighbor_index_pointers: Vec<_> = (0..num_neighbors)
                .map(|_| ItemPointer::new(InvalidBlockNumber, InvalidOffsetNumber))
                .collect();

            SbqDiskNode::Labeled(LabeledSbqDiskNode {
                heap_item_pointer: heap_pointer,
                bq_vector: bq_vector.to_vec(),
                neighbor_index_pointers,
                labels: labels.unwrap_or_default(),
            })
        } else {
            SbqDiskNode::Classic(ClassicSbqDiskNode {
                heap_item_pointer: heap_pointer,
                bq_vector: bq_vector.to_vec(),
                neighbor_node_pointer: None,
            })
        }
    }

    pub unsafe fn read<'a, S: StatsNodeRead>(
        index: &'a PgRelation,
        index_pointer: ItemPointer,
        has_labels: bool,
        stats: &mut S,
    ) -> ReadableSbqDiskNode<'a> {
        if has_labels {
            ReadableSbqDiskNode::Labeled(LabeledSbqDiskNode::read(index, index_pointer, stats))
        } else {
            ReadableSbqDiskNode::Classic(ClassicSbqDiskNode::read(index, index_pointer, stats))
        }
    }

    pub unsafe fn modify<'a, S: StatsNodeModify>(
        index: &'a PgRelation,
        index_pointer: ItemPointer,
        has_labels: bool,
        stats: &mut S,
    ) -> WritableSbqDiskNode<'a> {
        if has_labels {
            WritableSbqDiskNode::Labeled(LabeledSbqDiskNode::modify(index, index_pointer, stats))
        } else {
            WritableSbqDiskNode::Classic(ClassicSbqDiskNode::modify(index, index_pointer, stats))
        }
    }
}

impl NodeVacuum for ArchivedClassicSbqDiskNode {
    fn with_data(data: &mut [u8]) -> Pin<&mut Self> {
        ArchivedClassicSbqDiskNode::with_data(data)
    }

    fn delete(self: Pin<&mut Self>) {
        //TODO: actually optimize the deletes by removing index tuples. For now just mark it.
        let mut heap_pointer = unsafe { self.map_unchecked_mut(|s| &mut s.heap_item_pointer) };
        heap_pointer.offset = InvalidOffsetNumber;
        heap_pointer.block_number = InvalidBlockNumber;
    }
}

impl NodeVacuum for ArchivedLabeledSbqDiskNode {
    fn with_data(data: &mut [u8]) -> Pin<&mut Self> {
        ArchivedLabeledSbqDiskNode::with_data(data)
    }

    fn delete(self: Pin<&mut Self>) {
        //TODO: actually optimize the deletes by removing index tuples. For now just mark it.
        let mut heap_pointer = unsafe { self.map_unchecked_mut(|s| &mut s.heap_item_pointer) };
        heap_pointer.offset = InvalidOffsetNumber;
        heap_pointer.block_number = InvalidBlockNumber;
    }
}

impl ArchivedData for ArchivedClassicSbqDiskNode {
    fn is_deleted(&self) -> bool {
        self.heap_item_pointer.offset == InvalidOffsetNumber
    }

    fn get_heap_item_pointer(&self) -> HeapPointer {
        self.heap_item_pointer.deserialize_item_pointer()
    }

    fn get_index_pointer_to_neighbors<S: StatsNodeRead>(
        &self,
        index: &PgRelation,
        stats: &mut S,
    ) -> Vec<ItemPointer> {
        if let Some(neighbor_pointer) = self.neighbor_node_pointer.as_ref() {
            let neighbor_pointer_ip = neighbor_pointer.deserialize_item_pointer();
            let neighbor_node =
                unsafe { ClassicSbqDiskNeighborNode::read(index, neighbor_pointer_ip, stats) };
            let archived_neighbor = neighbor_node.get_archived_node();
            let num_neighbors = archived_neighbor
                .neighbor_index_pointers
                .iter()
                .position(|f| f.block_number == InvalidBlockNumber)
                .unwrap_or(archived_neighbor.neighbor_index_pointers.len());
            archived_neighbor
                .neighbor_index_pointers
                .iter()
                .take(num_neighbors)
                .map(|ip| ip.deserialize_item_pointer())
                .collect()
        } else {
            vec![]
        }
    }
}

impl ArchivedClassicSbqDiskNeighborNode {
    pub fn get_index_pointer_to_neighbors(&self) -> Vec<ItemPointer> {
        let num_neighbors = self
            .neighbor_index_pointers
            .iter()
            .position(|f| f.block_number == InvalidBlockNumber)
            .unwrap_or(self.neighbor_index_pointers.len());
        self.neighbor_index_pointers
            .iter()
            .take(num_neighbors)
            .map(|ip| ip.deserialize_item_pointer())
            .collect()
    }
}

impl ArchivedData for ArchivedLabeledSbqDiskNode {
    fn is_deleted(&self) -> bool {
        self.heap_item_pointer.offset == InvalidOffsetNumber
    }

    fn get_heap_item_pointer(&self) -> HeapPointer {
        self.heap_item_pointer.deserialize_item_pointer()
    }

    fn get_index_pointer_to_neighbors<S: StatsNodeRead>(
        &self,
        _index: &PgRelation,
        _stats: &mut S,
    ) -> Vec<ItemPointer> {
        let num_neighbors = self
            .neighbor_index_pointers
            .iter()
            .position(|f| f.block_number == InvalidBlockNumber)
            .unwrap_or(self.neighbor_index_pointers.len());
        self.neighbor_index_pointers
            .iter()
            .take(num_neighbors)
            .map(|p| p.deserialize_item_pointer())
            .collect()
    }
}

pub enum ReadableSbqDiskNode<'a> {
    Classic(ReadableClassicSbqDiskNode<'a>),
    Labeled(ReadableLabeledSbqDiskNode<'a>),
}

impl<'a> ReadableSbqDiskNode<'a> {
    pub fn get_archived_node(&'a self) -> ArchivedSbqDiskNode<'a> {
        match self {
            ReadableSbqDiskNode::Classic(node) => {
                ArchivedSbqDiskNode::Classic(node.get_archived_node())
            }
            ReadableSbqDiskNode::Labeled(node) => {
                ArchivedSbqDiskNode::Labeled(node.get_archived_node())
            }
        }
    }
}

pub enum WritableSbqDiskNode<'a> {
    Classic(WritableClassicSbqDiskNode<'a>),
    Labeled(WritableLabeledSbqDiskNode<'a>),
}

impl WritableSbqDiskNode<'_> {
    pub fn get_archived_node(&mut self) -> ArchivedMutSbqDiskNode<'_> {
        match self {
            WritableSbqDiskNode::Classic(node) => {
                ArchivedMutSbqDiskNode::Classic(node.get_archived_node())
            }
            WritableSbqDiskNode::Labeled(node) => {
                ArchivedMutSbqDiskNode::Labeled(node.get_archived_node())
            }
        }
    }

    pub fn commit(self) {
        match self {
            WritableSbqDiskNode::Classic(node) => node.commit(),
            WritableSbqDiskNode::Labeled(node) => node.commit(),
        }
    }
}

pub enum ArchivedMutSbqDiskNode<'a> {
    Classic(Pin<&'a mut ArchivedClassicSbqDiskNode>),
    Labeled(Pin<&'a mut ArchivedLabeledSbqDiskNode>),
}

pub enum ArchivedSbqDiskNode<'a> {
    Classic(&'a ArchivedClassicSbqDiskNode),
    Labeled(&'a ArchivedLabeledSbqDiskNode),
}

impl ArchivedData for ArchivedSbqDiskNode<'_> {
    fn is_deleted(&self) -> bool {
        match self {
            ArchivedSbqDiskNode::Classic(node) => node.is_deleted(),
            ArchivedSbqDiskNode::Labeled(node) => node.is_deleted(),
        }
    }

    fn get_heap_item_pointer(&self) -> HeapPointer {
        match self {
            ArchivedSbqDiskNode::Classic(node) => node.get_heap_item_pointer(),
            ArchivedSbqDiskNode::Labeled(node) => node.get_heap_item_pointer(),
        }
    }

    fn get_index_pointer_to_neighbors<S: StatsNodeRead>(
        &self,
        index: &PgRelation,
        stats: &mut S,
    ) -> Vec<ItemPointer> {
        match self {
            ArchivedSbqDiskNode::Classic(node) => node.get_index_pointer_to_neighbors(index, stats),
            ArchivedSbqDiskNode::Labeled(node) => node.get_index_pointer_to_neighbors(index, stats),
        }
    }
}

impl Debug for ArchivedSbqDiskNode<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArchivedSbqDiskNode::Classic(node) => node.fmt(f),
            ArchivedSbqDiskNode::Labeled(node) => node.fmt(f),
        }
    }
}

impl ArchivedSbqDiskNode<'_> {
    pub fn num_neighbors(&self) -> usize {
        match self {
            ArchivedSbqDiskNode::Classic(_node) => {
                panic!("num_neighbors called on ClassicSbqDiskNode - use neighbor node instead")
            }
            ArchivedSbqDiskNode::Labeled(node) => node
                .neighbor_index_pointers
                .iter()
                .position(|f| f.block_number == InvalidBlockNumber)
                .unwrap_or(node.neighbor_index_pointers.len()),
        }
    }

    pub fn iter_neighbors<'a, S: StatsNodeRead>(
        &'a self,
        index: &'a PgRelation,
        stats: &'a mut S,
    ) -> Box<dyn Iterator<Item = ItemPointer> + 'a> {
        match self {
            ArchivedSbqDiskNode::Classic(node) => {
                if let Some(neighbor_pointer) = node.neighbor_node_pointer.as_ref() {
                    let neighbor_pointer_ip = neighbor_pointer.deserialize_item_pointer();
                    let neighbor_node = unsafe {
                        ClassicSbqDiskNeighborNode::read(index, neighbor_pointer_ip, stats)
                    };
                    let archived_neighbor = neighbor_node.get_archived_node();
                    let num_neighbors = archived_neighbor
                        .neighbor_index_pointers
                        .iter()
                        .position(|f| f.block_number == InvalidBlockNumber)
                        .unwrap_or(archived_neighbor.neighbor_index_pointers.len());
                    Box::new(
                        archived_neighbor
                            .neighbor_index_pointers
                            .iter()
                            .take(num_neighbors)
                            .map(|ip| ip.deserialize_item_pointer()),
                    )
                } else {
                    Box::new(std::iter::empty())
                }
            }
            ArchivedSbqDiskNode::Labeled(node) => {
                let num_neighbors = node
                    .neighbor_index_pointers
                    .iter()
                    .position(|f| f.block_number == InvalidBlockNumber)
                    .unwrap_or(node.neighbor_index_pointers.len());
                Box::new(
                    node.neighbor_index_pointers
                        .iter()
                        .take(num_neighbors)
                        .map(|ip| ip.deserialize_item_pointer()),
                )
            }
        }
    }

    pub fn get_index_pointer_to_neighbors<S: StatsNodeRead>(
        &self,
        index: &PgRelation,
        stats: &mut S,
    ) -> Vec<ItemPointer> {
        self.iter_neighbors(index, stats).collect()
    }

    pub fn get_bq_vector(&self) -> &[SbqDiskVectorElement] {
        match self {
            ArchivedSbqDiskNode::Classic(node) => &node.bq_vector,
            ArchivedSbqDiskNode::Labeled(node) => &node.bq_vector,
        }
    }

    pub fn get_heap_item_pointer(&self) -> HeapPointer {
        match self {
            ArchivedSbqDiskNode::Classic(node) => node.heap_item_pointer.deserialize_item_pointer(),
            ArchivedSbqDiskNode::Labeled(node) => node.heap_item_pointer.deserialize_item_pointer(),
        }
    }

    pub fn get_labels(&self) -> Option<&ArchivedLabelSet> {
        match self {
            ArchivedSbqDiskNode::Classic(_) => None,
            ArchivedSbqDiskNode::Labeled(node) => Some(&node.labels),
        }
    }
}

impl ArchivedData for ArchivedMutSbqDiskNode<'_> {
    fn get_index_pointer_to_neighbors<S: StatsNodeRead>(
        &self,
        index: &PgRelation,
        stats: &mut S,
    ) -> Vec<ItemPointer> {
        self.iter_neighbors(index, stats).collect()
    }

    fn is_deleted(&self) -> bool {
        self.get_heap_item_pointer().offset == InvalidOffsetNumber
    }

    fn get_heap_item_pointer(&self) -> HeapPointer {
        let hip = match self {
            ArchivedMutSbqDiskNode::Classic(node) => &node.heap_item_pointer,
            ArchivedMutSbqDiskNode::Labeled(node) => &node.heap_item_pointer,
        };
        hip.deserialize_item_pointer()
    }
}

impl Debug for ArchivedClassicSbqDiskNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArchivedClassicSbqDiskNode")
            .field(
                "heap_item_pointer.block_number",
                &self.heap_item_pointer.block_number,
            )
            .field("heap_item_pointer.offset", &self.heap_item_pointer.offset)
            .field("bq_vector", &self.bq_vector)
            .field("has_neighbor_node", &self.neighbor_node_pointer.is_some())
            .finish()
    }
}

impl Debug for ArchivedLabeledSbqDiskNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArchivedLabeledSbqDiskNode")
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

impl<'a> ArchivedMutSbqDiskNode<'a> {
    pub fn set_neighbors<S: StatsNodeModify>(
        &'a mut self,
        neighbors: &[NeighborWithDistance],
        num_neighbors: u32,
        index: &PgRelation,
        stats: &mut S,
    ) {
        match self {
            ArchivedMutSbqDiskNode::Classic(node) => {
                let neighbor_pointer = node.neighbor_node_pointer.as_ref()
                    .expect("Classic node should have neighbor node pointer - neighbor node should be created during node creation");
                let neighbor_pointer_ip = neighbor_pointer.deserialize_item_pointer();

                let mut neighbor_node = unsafe {
                    ClassicSbqDiskNeighborNode::modify(index, neighbor_pointer_ip, stats)
                };
                let mut archived_neighbor = neighbor_node.get_archived_node();
                let mut neighbor_index_pointer = unsafe {
                    archived_neighbor
                        .as_mut()
                        .map_unchecked_mut(|s| &mut s.neighbor_index_pointers)
                };
                for (i, new_neighbor) in neighbors.iter().enumerate() {
                    let mut a_index_pointer = neighbor_index_pointer.as_mut().index_pin(i);
                    let ip = new_neighbor.get_index_pointer_to_neighbor();
                    //TODO hate that we have to set each field like this
                    a_index_pointer.block_number = ip.block_number;
                    a_index_pointer.offset = ip.offset;
                }
                //set the marker that the list ended
                if neighbors.len() < num_neighbors as _ {
                    let mut past_last_index_pointers =
                        neighbor_index_pointer.index_pin(neighbors.len());
                    past_last_index_pointers.block_number = InvalidBlockNumber;
                    past_last_index_pointers.offset = InvalidOffsetNumber;
                }
                neighbor_node.commit();
            }
            ArchivedMutSbqDiskNode::Labeled(node) => {
                let mut neighbor_index_pointer = unsafe {
                    node.as_mut()
                        .map_unchecked_mut(|s| &mut s.neighbor_index_pointers)
                };
                for (i, new_neighbor) in neighbors.iter().enumerate() {
                    let mut a_index_pointer = neighbor_index_pointer.as_mut().index_pin(i);
                    let ip = new_neighbor.get_index_pointer_to_neighbor();
                    //TODO hate that we have to set each field like this
                    a_index_pointer.block_number = ip.block_number;
                    a_index_pointer.offset = ip.offset;
                }
                //set the marker that the list ended
                if neighbors.len() < num_neighbors as _ {
                    let mut past_last_index_pointers =
                        neighbor_index_pointer.index_pin(neighbors.len());
                    past_last_index_pointers.block_number = InvalidBlockNumber;
                    past_last_index_pointers.offset = InvalidOffsetNumber;
                }
            }
        }
    }

    pub fn num_neighbors<S: StatsNodeRead>(&self, index: &PgRelation, stats: &mut S) -> usize {
        match self {
            ArchivedMutSbqDiskNode::Classic(node) => {
                if let Some(neighbor_pointer) = node.neighbor_node_pointer.as_ref() {
                    let neighbor_pointer_ip = neighbor_pointer.deserialize_item_pointer();
                    let neighbor_node = unsafe {
                        ClassicSbqDiskNeighborNode::read(index, neighbor_pointer_ip, stats)
                    };
                    let archived_neighbor = neighbor_node.get_archived_node();
                    archived_neighbor
                        .neighbor_index_pointers
                        .iter()
                        .position(|f| f.block_number == InvalidBlockNumber)
                        .unwrap_or(archived_neighbor.neighbor_index_pointers.len())
                } else {
                    0
                }
            }
            ArchivedMutSbqDiskNode::Labeled(node) => node
                .neighbor_index_pointers
                .iter()
                .position(|f| f.block_number == InvalidBlockNumber)
                .unwrap_or(node.neighbor_index_pointers.len()),
        }
    }

    pub fn iter_neighbors<'b, S: StatsNodeRead>(
        &'b self,
        index: &'b PgRelation,
        stats: &'b mut S,
    ) -> Box<dyn Iterator<Item = ItemPointer> + 'b> {
        match self {
            ArchivedMutSbqDiskNode::Classic(node) => {
                if let Some(neighbor_pointer) = node.neighbor_node_pointer.as_ref() {
                    let neighbor_pointer_ip = neighbor_pointer.deserialize_item_pointer();
                    let neighbor_node = unsafe {
                        ClassicSbqDiskNeighborNode::read(index, neighbor_pointer_ip, stats)
                    };
                    let archived_neighbor = neighbor_node.get_archived_node();
                    let num_neighbors = archived_neighbor
                        .neighbor_index_pointers
                        .iter()
                        .position(|f| f.block_number == InvalidBlockNumber)
                        .unwrap_or(archived_neighbor.neighbor_index_pointers.len());
                    Box::new(
                        archived_neighbor
                            .neighbor_index_pointers
                            .iter()
                            .take(num_neighbors)
                            .map(|ip| ip.deserialize_item_pointer()),
                    )
                } else {
                    Box::new(std::iter::empty())
                }
            }
            ArchivedMutSbqDiskNode::Labeled(node) => {
                let num_neighbors = node
                    .neighbor_index_pointers
                    .iter()
                    .position(|f| f.block_number == InvalidBlockNumber)
                    .unwrap_or(node.neighbor_index_pointers.len());
                Box::new(
                    node.neighbor_index_pointers
                        .iter()
                        .take(num_neighbors)
                        .map(|ip| ip.deserialize_item_pointer()),
                )
            }
        }
    }
}
