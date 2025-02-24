use crate::access_method::PgRelation;
use crate::util::{ArchivedItemPointer, HeapPointer, ItemPointer, ReadableBuffer, WritableBuffer};
use pgrx::pg_sys::{InvalidBlockNumber, InvalidOffsetNumber, BLCKSZ};
use pgvectorscale_derive::{Readable, Writeable};
use rkyv::{vec::ArchivedVec, Archive, Deserialize, Serialize};
use std::pin::Pin;

use super::{
    meta_page::MetaPage,
    neighbor_with_distance::NeighborWithDistance,
    sbq::{SbqQuantizer, SbqVectorElement},
    storage::ArchivedData,
};

#[derive(Archive, Deserialize, Serialize, Readable, Writeable)]
#[archive(check_bytes)]
pub struct SbqNode {
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
        let n = Self::new(hp, num_neighbors, num_dimensions, &v);
        n.serialize_to_vec().len()
    }

    pub fn get_default_num_neighbors(num_dimensions: usize, num_bits_per_dimension: u8) -> usize {
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
            let serialized_size = SbqNode::test_size(
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
}

impl ArchivedSbqNode {
    fn neighbor_index_pointer(self: Pin<&mut Self>) -> Pin<&mut ArchivedVec<ArchivedItemPointer>> {
        unsafe { self.map_unchecked_mut(|s| &mut s.neighbor_index_pointers) }
    }

    pub fn set_neighbors(
        mut self: Pin<&mut Self>,
        neighbors: &[NeighborWithDistance],
        meta_page: &MetaPage,
    ) {
        for (i, new_neighbor) in neighbors.iter().enumerate() {
            let mut a_index_pointer = self.as_mut().neighbor_index_pointer().index_pin(i);
            let ip = new_neighbor.get_index_pointer_to_neighbor();
            //TODO hate that we have to set each field like this
            a_index_pointer.block_number = ip.block_number;
            a_index_pointer.offset = ip.offset;
        }
        //set the marker that the list ended
        if neighbors.len() < meta_page.get_num_neighbors() as _ {
            let mut past_last_index_pointers =
                self.neighbor_index_pointer().index_pin(neighbors.len());
            past_last_index_pointers.block_number = InvalidBlockNumber;
            past_last_index_pointers.offset = InvalidOffsetNumber;
        }
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
}

impl ArchivedData for ArchivedSbqNode {
    fn with_data(data: &mut [u8]) -> Pin<&mut ArchivedSbqNode> {
        ArchivedSbqNode::with_data(data)
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
