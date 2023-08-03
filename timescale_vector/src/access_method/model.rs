use std::cmp::Ordering;
use std::mem::size_of;
use std::pin::Pin;

use ndarray::Array3;
use pgrx::pg_sys::{InvalidBlockNumber, InvalidOffsetNumber, BLCKSZ};
use pgrx::*;
use reductive::pq::Pq;
use rkyv::vec::ArchivedVec;
use rkyv::{Archive, Archived, Deserialize, Serialize};

use crate::util::page::PageType;
use crate::util::tape::Tape;
use crate::util::{
    ArchivedItemPointer, HeapPointer, IndexPointer, ItemPointer, ReadableBuffer, WritableBuffer,
};

use super::meta_page::MetaPage;

//Ported from pg_vector code
#[repr(C)]
#[derive(Debug)]
pub struct PgVector {
    vl_len_: i32, /* varlena header (do not touch directly!) */
    pub dim: i16, /* number of dimensions */
    unused: i16,
    pub x: pg_sys::__IncompleteArrayField<std::os::raw::c_float>,
}

impl PgVector {
    pub unsafe fn from_pg_parts(
        datum_parts: *mut pg_sys::Datum,
        isnull_parts: *mut bool,
        index: usize,
    ) -> Option<*mut PgVector> {
        let isnulls = std::slice::from_raw_parts(isnull_parts, index + 1);
        if isnulls[index] {
            return None;
        }
        let datums = std::slice::from_raw_parts(datum_parts, index + 1);
        Some(Self::from_datum(datums[index]))
    }

    pub unsafe fn from_datum(datum: pg_sys::Datum) -> *mut PgVector {
        let detoasted = pg_sys::pg_detoast_datum(datum.cast_mut_ptr());
        let casted = detoasted.cast::<PgVector>();
        casted
    }

    pub fn to_slice(&self) -> &[f32] {
        let dim = (*self).dim;
        unsafe { (*self).x.as_slice(dim as _) }
        // unsafe { std::slice::from_raw_parts((*self).x, (*self).dim as _) }
    }
}

#[derive(Archive, Deserialize, Serialize)]
#[archive(check_bytes)]
pub struct Node {
    pub vector: Vec<f32>,
    pub pq_vector: Vec<u8>,
    neighbor_index_pointers: Vec<ItemPointer>,
    neighbor_distances: Vec<Distance>,
    pub heap_item_pointer: HeapPointer,
}

//ReadableNode ties an archive node to it's underlying buffer
pub struct ReadableNode<'a> {
    _rb: ReadableBuffer<'a>,
}

impl<'a> ReadableNode<'a> {
    pub fn get_archived_node(&self) -> &ArchivedNode {
        // checking the code here is expensive during build, so skip it.
        // TODO: should we check the data during queries?
        //rkyv::check_archived_root::<Node>(self._rb.get_data_slice()).unwrap()
        unsafe { rkyv::archived_root::<Node>(self._rb.get_data_slice()) }
    }
}

//WritableNode ties an archive node to it's underlying buffer that can be modified
pub struct WritableNode<'a> {
    wb: WritableBuffer<'a>,
}

impl<'a> WritableNode<'a> {
    pub fn get_archived_node(&self) -> Pin<&mut ArchivedNode> {
        ArchivedNode::with_data(self.wb.get_data_slice())
    }

    pub fn commit(self) {
        self.wb.commit()
    }
}

impl Node {
    pub fn new(vector: Vec<f32>, heap_item_pointer: ItemPointer, meta_page: &MetaPage) -> Self {
        let num_neighbors = meta_page.get_num_neighbors();
        let (vector, pq_vector) = if meta_page.get_use_pq() {
            let pq_vec_len = meta_page.get_pq_vector_length();
            (
                Vec::with_capacity(0),
                (0..pq_vec_len).map(|_| 0u8).collect(),
            )
        } else {
            (vector, Vec::with_capacity(0))
        };
        Self {
            vector,
            // always use vectors of num_clusters on length because we never want the serialized size of a Node to change
            pq_vector,
            // always use vectors of num_neighbors on length because we never want the serialized size of a Node to change
            neighbor_index_pointers: (0..num_neighbors).map(|_| ItemPointer::new(0, 0)).collect(),
            neighbor_distances: (0..num_neighbors).map(|_| Distance::NAN).collect(),
            heap_item_pointer,
        }
    }

    pub unsafe fn read<'a>(index: &'a PgRelation, index_pointer: ItemPointer) -> ReadableNode<'a> {
        let rb = index_pointer.read_bytes(index);
        ReadableNode { _rb: rb }
    }

    pub unsafe fn modify(index: &PgRelation, index_pointer: ItemPointer) -> WritableNode {
        let wb = index_pointer.modify_bytes(index);
        WritableNode { wb: wb }
    }

    pub unsafe fn update_neighbors_and_pq(
        index: &PgRelation,
        index_pointer: ItemPointer,
        neighbors: &Vec<NeighborWithDistance>,
        meta_page: &MetaPage,
        vector: Option<Vec<u8>>,
    ) {
        let node = Node::modify(index, index_pointer);
        let mut archived = node.get_archived_node();
        for (i, new_neighbor) in neighbors.iter().enumerate() {
            //TODO: why do we need to recreate the archive?
            let mut a_index_pointer = archived.as_mut().neighbor_index_pointer().index_pin(i);
            //TODO hate that we have to set each field like this
            a_index_pointer.block_number =
                new_neighbor.get_index_pointer_to_neighbor().block_number;
            a_index_pointer.offset = new_neighbor.get_index_pointer_to_neighbor().offset;

            let mut a_distance = archived.as_mut().neighbor_distances().index_pin(i);
            *a_distance = new_neighbor.get_distance() as Distance;
        }
        //set the marker that the list ended
        if neighbors.len() < meta_page.get_num_neighbors() as _ {
            //TODO: why do we need to recreate the archive?
            let archived = node.get_archived_node();
            let mut past_last_distance = archived.neighbor_distances().index_pin(neighbors.len());
            *past_last_distance = Distance::NAN;
        }

        match vector {
            Some(v) => {
                assert!(v.len() == archived.pq_vector.len());
                for i in 0..=v.len() - 1 {
                    let mut pgv = archived.as_mut().pq_vectors().index_pin(i);
                    *pgv = v[i];
                }
            }
            None => {}
        }

        node.commit()
    }
    pub fn write(&self, tape: &mut Tape) -> ItemPointer {
        let bytes = rkyv::to_bytes::<_, 256>(self).unwrap();
        unsafe { tape.write(&bytes) }
    }
}

/// contains helpers for mutate-in-place. See struct_mutable_refs in test_alloc.rs in rkyv
impl ArchivedNode {
    pub fn with_data(data: &mut [u8]) -> Pin<&mut ArchivedNode> {
        let pinned_bytes = Pin::new(data);
        unsafe { rkyv::archived_root_mut::<Node>(pinned_bytes) }
    }

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

    pub fn neighbor_distances(self: Pin<&mut Self>) -> Pin<&mut ArchivedVec<Distance>> {
        unsafe { self.map_unchecked_mut(|s| &mut s.neighbor_distances) }
    }

    pub fn pq_vectors(self: Pin<&mut Self>) -> Pin<&mut Archived<Vec<u8>>> {
        unsafe { self.map_unchecked_mut(|s| &mut s.pq_vector) }
    }

    pub fn num_neighbors(&self) -> usize {
        self.neighbor_distances
            .iter()
            .position(|&f| f.is_nan())
            .unwrap_or(self.neighbor_distances.len())
    }

    pub fn apply_to_neighbors<F>(&self, mut f: F)
    where
        F: FnMut(Distance, &ArchivedItemPointer),
    {
        for i in 0..self.num_neighbors() {
            let dist = self.neighbor_distances[i];
            let neighbor = &self.neighbor_index_pointers[i];
            f(dist, neighbor);
        }
    }
}

//TODO is this right?
pub type Distance = f32;
#[derive(Clone)]
pub struct NeighborWithDistance {
    index_pointer: IndexPointer,
    distance: Distance,
}

impl NeighborWithDistance {
    pub fn new(neighbor_index_pointer: ItemPointer, distance: Distance) -> Self {
        Self {
            index_pointer: neighbor_index_pointer,
            distance,
        }
    }

    pub fn get_index_pointer_to_neighbor(&self) -> ItemPointer {
        return self.index_pointer;
    }
    pub fn get_distance(&self) -> Distance {
        return self.distance;
    }
}

impl PartialOrd for NeighborWithDistance {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}

impl Ord for NeighborWithDistance {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance.total_cmp(&other.distance)
    }
}

impl PartialEq for NeighborWithDistance {
    fn eq(&self, other: &Self) -> bool {
        self.index_pointer == other.index_pointer
    }
}

//promise that PartialEq is reflexive
impl Eq for NeighborWithDistance {}

impl std::hash::Hash for NeighborWithDistance {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.index_pointer.hash(state);
    }
}

#[derive(Archive, Deserialize, Serialize)]
#[archive(check_bytes)]
#[repr(C)]
pub struct PqQuantizerDef {
    dim_0: usize,
    dim_1: usize,
    dim_2: usize,
    vec_len: usize,
    next_vector_pointer: ItemPointer,
}

impl PqQuantizerDef {
    pub fn new(dim_0: usize, dim_1: usize, dim_2: usize, vec_len: usize) -> PqQuantizerDef {
        {
            Self {
                dim_0,
                dim_1,
                dim_2,
                vec_len,
                next_vector_pointer: ItemPointer {
                    block_number: 0,
                    offset: 0,
                },
            }
        }
    }

    pub unsafe fn write(&self, tape: &mut Tape) -> ItemPointer {
        let bytes = rkyv::to_bytes::<_, 256>(self).unwrap();
        tape.write(&bytes)
    }
    pub unsafe fn read<'a>(
        index: &'a PgRelation,
        index_pointer: &ItemPointer,
    ) -> ReadablePqQuantizerDef<'a> {
        let rb = index_pointer.read_bytes(index);
        ReadablePqQuantizerDef { _rb: rb }
    }
}

pub struct ReadablePqQuantizerDef<'a> {
    _rb: ReadableBuffer<'a>,
}

impl<'a> ReadablePqQuantizerDef<'a> {
    pub fn get_archived_node(&self) -> &ArchivedPqQuantizerDef {
        // checking the code here is expensive during build, so skip it.
        // TODO: should we check the data during queries?
        //rkyv::check_archived_root::<Node>(self._rb.get_data_slice()).unwrap()
        unsafe { rkyv::archived_root::<PqQuantizerDef>(self._rb.get_data_slice()) }
    }
}

#[derive(Archive, Deserialize, Serialize)]
#[archive(check_bytes)]
#[repr(C)]
pub struct PqQuantizerVector {
    vec: Vec<f32>,
    next_vector_pointer: ItemPointer,
}

impl PqQuantizerVector {
    pub unsafe fn write(&self, tape: &mut Tape) -> ItemPointer {
        let bytes = rkyv::to_bytes::<_, 8192>(self).unwrap();
        tape.write(&bytes)
    }
    pub unsafe fn read<'a>(
        index: &'a PgRelation,
        index_pointer: &ItemPointer,
    ) -> ReadablePqVectorNode<'a> {
        let rb = index_pointer.read_bytes(index);
        ReadablePqVectorNode { _rb: rb }
    }
}

//ReadablePqNode ties an archive node to it's underlying buffer
pub struct ReadablePqVectorNode<'a> {
    _rb: ReadableBuffer<'a>,
}

impl<'a> ReadablePqVectorNode<'a> {
    pub fn get_archived_node(&self) -> &ArchivedPqQuantizerVector {
        // checking the code here is expensive during build, so skip it.
        // TODO: should we check the data during queries?
        //rkyv::check_archived_root::<Node>(self._rb.get_data_slice()).unwrap()
        unsafe { rkyv::archived_root::<PqQuantizerVector>(self._rb.get_data_slice()) }
    }
}

pub unsafe fn read_pq(index: &PgRelation, index_pointer: &IndexPointer) -> Pq<f32> {
    let rpq = PqQuantizerDef::read(index, &index_pointer);
    let rpn = rpq.get_archived_node();
    let mut result: Vec<f32> = Vec::new();
    let mut next = rpn.next_vector_pointer.deserialize_item_pointer();
    loop {
        if next.offset == 0 && next.block_number == 0 {
            break;
        }
        let qvn = PqQuantizerVector::read(index, &next);
        let vn = qvn.get_archived_node();
        let vs = vn.vec.as_slice();
        result.extend_from_slice(vs);
        next = vn.next_vector_pointer.deserialize_item_pointer();
    }
    let sq = Array3::from_shape_vec(
        (rpn.dim_0 as usize, rpn.dim_1 as usize, rpn.dim_2 as usize),
        result,
    )
    .unwrap();
    Pq::new(None, sq)
}

pub unsafe fn write_pq(pq: Pq<f32>, index: &PgRelation) -> ItemPointer {
    let vec = pq.subquantizers().to_slice_memory_order().unwrap().to_vec();
    let shape = pq.subquantizers().dim();
    let mut pq_node = PqQuantizerDef::new(shape.0, shape.1, shape.2, vec.len());

    let mut pqt = Tape::new(index, PageType::PqQuantizerDef);

    // write out the large vector bits.
    // we write "from the back"
    let mut prev: IndexPointer = ItemPointer {
        block_number: 0,
        offset: 0,
    };
    let mut prev_vec = vec;

    // get numbers that can fit in a page by subtracting the item pointer.
    let block_fit = (BLCKSZ as usize / size_of::<f32>()) - size_of::<ItemPointer>() - 64;
    let mut tape = Tape::new(index, PageType::PqQuantizerVector);
    loop {
        let l = prev_vec.len();
        if l == 0 {
            pq_node.next_vector_pointer = prev;
            return pq_node.write(&mut pqt);
        }
        let lv = prev_vec;
        let ni = if l > block_fit { l - block_fit } else { 0 };
        let (b, a) = lv.split_at(ni);

        let pqv_node = PqQuantizerVector {
            vec: a.to_vec(),
            next_vector_pointer: prev,
        };
        let index_pointer: IndexPointer = pqv_node.write(&mut tape);
        prev = index_pointer;
        prev_vec = b.clone().to_vec();
    }
}
