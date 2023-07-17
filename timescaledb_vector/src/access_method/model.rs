use std::cmp::Ordering;

use std::pin::Pin;

use pgrx::*;
use rkyv::vec::ArchivedVec;
use rkyv::{Archive, Deserialize, Serialize};

use crate::util::tape::Tape;
use crate::util::{
    ArchivedItemPointer, HeapPointer, IndexPointer, ItemPointer, ReadableBuffer, WritableBuffer,
};

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
    neighbor_index_pointers: Vec<ItemPointer>,
    neighbor_distances: Vec<Distance>,
    pub heap_item_pointer: HeapPointer,
    deleted: bool,
}

//ReadableNode ties an archive node to it's underlying buffer
pub struct ReadableNode {
    _rb: ReadableBuffer,
}

impl ReadableNode {
    pub fn get_archived_node(&self) -> &ArchivedNode {
        rkyv::check_archived_root::<Node>(self._rb.get_data_slice()).unwrap()
    }
}

//WritableNode ties an archive node to it's underlying buffer that can be modified
pub struct WritableNode {
    wb: WritableBuffer,
}

impl WritableNode {
    pub fn get_archived_node(&self) -> Pin<&mut ArchivedNode> {
        let pinned_bytes = Pin::new(self.wb.get_data_slice());
        unsafe { rkyv::archived_root_mut::<Node>(pinned_bytes) }
    }

    pub fn commit(self) {
        self.wb.commit()
    }
}

impl Node {
    pub fn new(
        vector: &[f32],
        heap_item_pointer: ItemPointer,
        meta_page: &super::build::TsvMetaPage,
    ) -> Self {
        let num_neighbors = meta_page.get_num_neighbors();
        Self {
            vector: vector.to_vec(),
            //always use vectors of num_neighbors on length because we never want the serialized size of a Node to change
            neighbor_index_pointers: (0..num_neighbors).map(|_| ItemPointer::new(0, 0)).collect(),
            neighbor_distances: (0..num_neighbors).map(|_| Distance::NAN).collect(),
            heap_item_pointer: heap_item_pointer,
            deleted: false,
        }
    }

    pub unsafe fn read(index: &PgRelation, index_pointer: &ItemPointer) -> ReadableNode {
        let rb = index_pointer.read_bytes((*index).as_ptr());
        ReadableNode { _rb: rb }
    }

    pub unsafe fn modify(index: &PgRelation, index_pointer: &ItemPointer) -> WritableNode {
        let wb = index_pointer.modify_bytes((*index).as_ptr());
        WritableNode { wb: wb }
    }

    pub unsafe fn write(&self, tape: &mut Tape) -> ItemPointer {
        let bytes = rkyv::to_bytes::<_, 256>(self).unwrap();
        tape.write(&bytes)
    }
}

/// contains helpers for mutate-in-place. See struct_mutable_refs in test_alloc.rs in rkyv
impl ArchivedNode {
    pub fn neighbor_index_pointer(
        self: Pin<&mut Self>,
    ) -> Pin<&mut ArchivedVec<ArchivedItemPointer>> {
        unsafe { self.map_unchecked_mut(|s| &mut s.neighbor_index_pointers) }
    }

    pub fn neighbor_distances(self: Pin<&mut Self>) -> Pin<&mut ArchivedVec<Distance>> {
        unsafe { self.map_unchecked_mut(|s| &mut s.neighbor_distances) }
    }

    pub fn num_neighbors(&self) -> usize {
        self.neighbor_distances
            .iter()
            .position(|&f| f.is_nan())
            .unwrap_or(self.neighbor_distances.len())
    }

    pub fn apply_to_neightbors<F>(&self, mut f: F)
    where
        F: FnMut(Distance, &ArchivedItemPointer),
    {
        for i in 0..self.num_neighbors() {
            let dist = self.neighbor_distances[i];
            let neigbor = &self.neighbor_index_pointers[i];
            f(dist, neigbor);
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
            distance: distance,
        }
    }

    pub fn get_index_pointer_to_neigbor(&self) -> ItemPointer {
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
