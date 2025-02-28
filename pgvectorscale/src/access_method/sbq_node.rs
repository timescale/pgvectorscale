use crate::access_method::node::{ReadableNode, WriteableNode};
use crate::access_method::PgRelation;
use crate::util::{ArchivedItemPointer, HeapPointer, ItemPointer, ReadableBuffer, WritableBuffer};
use pgrx::pg_sys::{InvalidBlockNumber, InvalidOffsetNumber, BLCKSZ};
use pgvectorscale_derive::{Readable, Writeable};
use rkyv::{vec::ArchivedVec, Archive, Deserialize, Serialize};
use std::pin::Pin;

use super::labels::{ArchivedLabelSet, LabelSet};
use super::stats::{StatsNodeModify, StatsNodeRead};
use super::{
    meta_page::MetaPage,
    neighbor_with_distance::NeighborWithDistance,
    sbq::{SbqQuantizer, SbqVectorElement},
    storage::ArchivedData,
};

pub trait SbqNodeBase {
    type Node;

    fn with_meta(
        heap_pointer: HeapPointer,
        meta_page: &MetaPage,
        bq_vector: &[SbqVectorElement],
        labels: Option<LabelSet>,
    ) -> Self::Node;

    fn new(
        heap_pointer: HeapPointer,
        num_neighbors: usize,
        bq_vector: &[SbqVectorElement],
        labels: Option<LabelSet>,
    ) -> Self::Node;

    fn test_size(num_neighbors: usize, num_dimensions: usize, num_bits_per_dimension: u8) -> usize;
}

#[derive(Archive, Deserialize, Serialize, Readable, Writeable)]
#[archive(check_bytes)]
pub struct PlainSbqNode {
    heap_item_pointer: HeapPointer,
    bq_vector: Vec<u64>, // Don't use SbqVectorElement because we don't want to change the size in on-disk format by accident
    neighbor_index_pointers: Vec<ItemPointer>,
    _neighbor_vectors: Vec<Vec<u64>>, // No longer used, but kept for backwards compatibility
}

#[derive(Archive, Deserialize, Serialize, Readable, Writeable)]
#[archive(check_bytes)]
pub struct LabeledSbqNode {
    heap_item_pointer: HeapPointer,
    bq_vector: Vec<u64>, // Don't use SbqVectorElement because we don't want to change the size in on-disk format by accident
    neighbor_index_pointers: Vec<ItemPointer>,
    labels: Option<LabelSet>,
}

pub enum SbqNode {
    Plain(PlainSbqNode),
    Labeled(LabeledSbqNode),
}

impl From<PlainSbqNode> for SbqNode {
    fn from(n: PlainSbqNode) -> Self {
        SbqNode::Plain(n)
    }
}

impl From<LabeledSbqNode> for SbqNode {
    fn from(n: LabeledSbqNode) -> Self {
        SbqNode::Labeled(n)
    }
}

impl SbqNode {
    pub unsafe fn read<'a, S: StatsNodeRead>(
        index: &'a PgRelation,
        ip: ItemPointer,
        has_labels: bool,
        stats: &mut S,
    ) -> ReadableSbqNode<'a> {
        match has_labels {
            true => ReadableSbqNode::Labeled(LabeledSbqNode::read(index, ip, stats)),
            false => ReadableSbqNode::Plain(PlainSbqNode::read(index, ip, stats)),
        }
    }

    pub fn get_heap_item_pointer(&self) -> HeapPointer {
        match self {
            SbqNode::Plain(n) => n.heap_item_pointer,
            SbqNode::Labeled(n) => n.heap_item_pointer,
        }
    }

    pub fn get_bq_vector(&self) -> &[u64] {
        match self {
            SbqNode::Plain(n) => &n.bq_vector,
            SbqNode::Labeled(n) => &n.bq_vector,
        }
    }

    pub fn get_neighbor_index_pointers(&self) -> &[ItemPointer] {
        match self {
            SbqNode::Plain(n) => &n.neighbor_index_pointers,
            SbqNode::Labeled(n) => &n.neighbor_index_pointers,
        }
    }

    pub fn get_neighbor_index_pointers_mut(&mut self) -> &mut [ItemPointer] {
        match self {
            SbqNode::Plain(n) => &mut n.neighbor_index_pointers,
            SbqNode::Labeled(n) => &mut n.neighbor_index_pointers,
        }
    }

    pub fn get_labels(&self) -> Option<&LabelSet> {
        match self {
            SbqNode::Labeled(n) => n.labels.as_ref(),
            _ => None,
        }
    }

    fn get_default_num_neighbors<T: SbqNodeBase>(
        num_dimensions: usize,
        num_bits_per_dimension: u8,
    ) -> usize {
        //how many neighbors can fit on one page? That's what we choose.

        //we first overapproximate the number of neighbors and then double check by actually calculating the size of the SbqNode.

        //blocksize - 100 bytes for the padding/header/etc.
        let page_size = BLCKSZ as usize - 50;
        //one quantized_vector takes this many bytes
        let vec_size =
            SbqQuantizer::quantized_size_bytes(num_dimensions, num_bits_per_dimension) + 1;
        //start from the page size then subtract the heap_item_pointer and bq_vector elements of SbqNode.
        let starting = BLCKSZ as usize - std::mem::size_of::<HeapPointer>() - vec_size;
        //one neigbors contribution to neighbor_index_pointers + neighbor_vectors in SbqNode.
        let one_neighbor = vec_size + std::mem::size_of::<ItemPointer>();

        let mut num_neighbors_overapproximate: usize = starting / one_neighbor;
        while num_neighbors_overapproximate > 0 {
            let serialized_size = T::test_size(
                num_neighbors_overapproximate,
                num_dimensions,
                num_bits_per_dimension,
            );
            if serialized_size <= page_size {
                return num_neighbors_overapproximate;
            }
            num_neighbors_overapproximate -= 1;
        }
        pgrx::error!(
            "Could not find a valid number of neighbors for the default value. Please specify one."
        );
    }

    pub unsafe fn modify<'a, 'b, S: StatsNodeModify>(
        index: &'a PgRelation,
        index_pointer: ItemPointer,
        has_labels: bool,
        stats: &'b mut S,
    ) -> WriteableSbqNode<'a> {
        match has_labels {
            true => WriteableSbqNode::Labeled(LabeledSbqNode::modify(index, index_pointer, stats)),
            false => WriteableSbqNode::Plain(PlainSbqNode::modify(index, index_pointer, stats)),
        }
    }
}

pub enum ArchivedSbqNode<'a> {
    Plain(Pin<&'a mut ArchivedPlainSbqNode>),
    Labeled(Pin<&'a mut ArchivedLabeledSbqNode>),
}

enum ReadableSbqNode<'a> {
    Plain(ReadablePlainSbqNode<'a>),
    Labeled(ReadableLabeledSbqNode<'a>),
}

impl ReadableSbqNode<'_> {
    pub fn get_archived_node(&self) -> ArchivedSbqNode {
        match self {
            ReadableSbqNode::Plain(node) => ArchivedSbqNode::Plain(node.get_archived_node()),
            ReadableSbqNode::Labeled(node) => ArchivedSbqNode::Labeled(node.get_archived_node()),
        }
    }
}

enum WriteableSbqNode<'a> {
    Plain(WritablePlainSbqNode<'a>),
    Labeled(WritableLabeledSbqNode<'a>),
}

impl WriteableSbqNode<'_> {
    pub fn get_archived_node(&mut self) -> ArchivedSbqNode<'_> {
        match self {
            WriteableSbqNode::Plain(node) => ArchivedSbqNode::Plain(node.get_archived_node()),
            WriteableSbqNode::Labeled(node) => ArchivedSbqNode::Labeled(node.get_archived_node()),
        }
    }

    pub fn commit(self) {
        match self {
            WriteableSbqNode::Plain(node) => node.commit(),
            WriteableSbqNode::Labeled(node) => node.commit(),
        }
    }
}

impl SbqNodeBase for LabeledSbqNode {
    type Node = LabeledSbqNode;

    fn with_meta(
        heap_pointer: HeapPointer,
        meta_page: &MetaPage,
        bq_vector: &[SbqVectorElement],
        labels: Option<LabelSet>,
    ) -> Self::Node {
        Self::new(
            heap_pointer,
            meta_page.get_num_neighbors() as usize,
            bq_vector,
            labels,
        )
    }

    fn new(
        heap_pointer: HeapPointer,
        num_neighbors: usize,
        bq_vector: &[SbqVectorElement],
        labels: Option<LabelSet>,
    ) -> Self {
        // always use vectors of num_neighbors in length because we never want the serialized size of a Node to change
        let neighbor_index_pointers: Vec<_> = (0..num_neighbors)
            .map(|_| ItemPointer::new(InvalidBlockNumber, InvalidOffsetNumber))
            .collect();

        Self {
            heap_item_pointer: heap_pointer,
            bq_vector: bq_vector.to_vec(),
            neighbor_index_pointers,
            labels,
        }
    }

    fn test_size(num_neighbors: usize, num_dimensions: usize, num_bits_per_dimension: u8) -> usize {
        let v: Vec<SbqVectorElement> =
            vec![0; SbqQuantizer::quantized_size_internal(num_dimensions, num_bits_per_dimension)];
        let hp = HeapPointer::new(InvalidBlockNumber, InvalidOffsetNumber);
        let n = Self::new(hp, num_neighbors, &v, None);
        n.serialize_to_vec().len()
    }
}

impl SbqNodeBase for PlainSbqNode {
    type Node = PlainSbqNode;

    fn with_meta(
        heap_pointer: HeapPointer,
        meta_page: &MetaPage,
        bq_vector: &[SbqVectorElement],
        _labels: Option<LabelSet>,
    ) -> Self::Node {
        assert!(_labels.is_none());
        Self::new(
            heap_pointer,
            meta_page.get_num_neighbors() as usize,
            bq_vector,
            None,
        )
    }

    fn new(
        heap_pointer: HeapPointer,
        num_neighbors: usize,
        bq_vector: &[SbqVectorElement],
        _labels: Option<LabelSet>,
    ) -> Self {
        // always use vectors of num_neighbors in length because we never want the serialized size of a Node to change
        let neighbor_index_pointers: Vec<_> = (0..num_neighbors)
            .map(|_| ItemPointer::new(InvalidBlockNumber, InvalidOffsetNumber))
            .collect();

        Self {
            heap_item_pointer: heap_pointer,
            bq_vector: bq_vector.to_vec(),
            neighbor_index_pointers,
            _neighbor_vectors: vec![],
        }
    }

    fn test_size(num_neighbors: usize, num_dimensions: usize, num_bits_per_dimension: u8) -> usize {
        let v: Vec<SbqVectorElement> =
            vec![0; SbqQuantizer::quantized_size_internal(num_dimensions, num_bits_per_dimension)];
        let hp = HeapPointer::new(InvalidBlockNumber, InvalidOffsetNumber);
        let n = Self::new(hp, num_neighbors, &v, None);
        n.serialize_to_vec().len()
    }
}

impl<'a> ArchivedSbqNode<'a> {
    fn neighbor_index_pointer(&'a mut self) -> Pin<&'a mut ArchivedVec<ArchivedItemPointer>> {
        match self {
            ArchivedSbqNode::Plain(n) => unsafe {
                n.as_mut()
                    .map_unchecked_mut(|n| &mut n.neighbor_index_pointers)
            },
            ArchivedSbqNode::Labeled(n) => unsafe {
                n.as_mut()
                    .map_unchecked_mut(|n| &mut n.neighbor_index_pointers)
            },
        }
    }

    fn set_neighbors(&'a mut self, neighbors: &[NeighborWithDistance], meta_page: &MetaPage) {
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
}

impl ArchivedSbqNode<'_> {
    pub fn num_neighbors(&self) -> usize {
        let neighbor_index_pointers = match self {
            ArchivedSbqNode::Plain(n) => &n.neighbor_index_pointers,
            ArchivedSbqNode::Labeled(n) => &n.neighbor_index_pointers,
        };
        neighbor_index_pointers
            .iter()
            .position(|f| f.block_number == InvalidBlockNumber)
            .unwrap_or(neighbor_index_pointers.len())
    }

    pub fn iter_neighbors(&self) -> impl Iterator<Item = ItemPointer> + '_ {
        let neighbor_index_pointers = match self {
            ArchivedSbqNode::Plain(n) => &n.neighbor_index_pointers,
            ArchivedSbqNode::Labeled(n) => &n.neighbor_index_pointers,
        };
        neighbor_index_pointers
            .iter()
            .take(self.num_neighbors())
            .map(|ip| ip.deserialize_item_pointer())
    }

    pub fn get_labels(&self) -> Option<&ArchivedLabelSet> {
        match self {
            ArchivedSbqNode::Plain(_) => None,
            ArchivedSbqNode::Labeled(n) => n.labels.as_ref(),
        }
    }
}

impl<'a> ArchivedData for ArchivedSbqNode<'a> {
    type MutableSelf = &'a mut Self;
    fn get_index_pointer_to_neighbors(&self) -> Vec<ItemPointer> {
        self.iter_neighbors().collect()
    }

    fn is_deleted(&self) -> bool {
        self.get_heap_item_pointer().offset == InvalidOffsetNumber
    }

    fn delete(self: MutableSelf) {
        //TODO: actually optimize the deletes by removing index tuples. For now just mark it.
        // let mut heap_pointer = unsafe { self.map_unchecked_mut(|s| &mut s.heap_item_pointer) };
        let mut heap_pointer = unsafe {
            self.map_unchecked_mut(|s| match s {
                ArchivedSbqNode::Plain(n) => &mut n.heap_item_pointer,
                ArchivedSbqNode::Labeled(n) => &mut n.heap_item_pointer,
            })
        };
        heap_pointer.offset = InvalidOffsetNumber;
        heap_pointer.block_number = InvalidBlockNumber;
    }

    fn get_heap_item_pointer(&self) -> HeapPointer {
        let hip = match self {
            ArchivedSbqNode::Plain(n) => &n.heap_item_pointer,
            ArchivedSbqNode::Labeled(n) => &n.heap_item_pointer,
        };
        hip.deserialize_item_pointer()
    }
}
