use std::borrow::BorrowMut;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::io::Read;
use std::ops::IndexMut;
use std::pin::{pin, Pin};

use pgrx::pg_sys::Item;
use pgrx::*;
use rkyv::vec::ArchivedVec;
use rkyv::{Archive, Deserialize, Serialize};

use crate::util::tape::Tape;
use crate::util::{ArchivedItemPointer, HeapPointer, ItemPointer, ReadableBuffer, WritableBuffer};

use super::build::TsvMetaPage;
use super::graph::Graph;

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
pub struct ReadableNode<'a> {
    _rb: ReadableBuffer<'a>,
    pub node: &'a ArchivedNode,
}

//WritableNode ties an archive node to it's underlying buffer that can be modified
pub struct WritableNode {
    wb: WritableBuffer,
}

impl WritableNode {
    fn get_archived_node(&self) -> Pin<&mut ArchivedNode> {
        let pinned_bytes = Pin::new(self.wb.get_data_slice());
        unsafe { rkyv::archived_root_mut::<Node>(pinned_bytes) }
    }

    fn commit(self) {
        self.wb.commit()
    }
}

impl Node {
    pub fn new(
        vector: &[f32],
        heap_item_pointer: ItemPointer,
        meta_page: &super::build::TsvMetaPage,
    ) -> Self {
        let num_neighbors = meta_page.num_neighbors;
        Self {
            vector: vector.to_vec(),
            //always use vectors of num_neighbors on length because we never want the serialized size of a Node to change
            neighbor_index_pointers: (0..num_neighbors).map(|_| ItemPointer::new(0, 0)).collect(),
            neighbor_distances: (0..num_neighbors).map(|_| Distance::NAN).collect(),
            heap_item_pointer: heap_item_pointer,
            deleted: false,
        }
    }

    fn num_neighbors(&self) -> usize {
        if let Some(index) = self
            .neighbor_distances
            .iter()
            .position(|x| *x == Distance::NAN)
        {
            index + 1
        } else {
            self.neighbor_distances.len()
        }
    }

    pub unsafe fn read<'a>(index: &PgRelation, index_pointer: &ItemPointer) -> ReadableNode<'a> {
        let rb = index_pointer.read_bytes((*index).as_ptr());
        let archived = rkyv::check_archived_root::<Node>(rb.data).unwrap();
        ReadableNode {
            _rb: rb,
            node: archived,
        }
    }

    pub unsafe fn modify(index: &PgRelation, index_pointer: &ItemPointer) -> WritableNode {
        let wb = index_pointer.modify_bytes((*index).as_ptr());
        WritableNode { wb: wb }
    }

    /*     pub unsafe fn read_with_buffer<'a>(
        index: PgRelation,
        rb: &'a ReadableBuffer,
    ) -> &'a ArchivedNode {
        let archived = rkyv::check_archived_root::<Node>(rb.data).unwrap();
        archived
    }*/

    pub unsafe fn write(&self, tape: &mut Tape) -> ItemPointer {
        let bytes = rkyv::to_bytes::<_, 256>(self).unwrap();
        tape.write(&bytes)
    }
}

/// contains helpers for mutate-in-place. See struct_mutable_refs in test_alloc.rs in rkyv
impl ArchivedNode {
    fn neighbor_index_pointer(self: Pin<&mut Self>) -> Pin<&mut ArchivedVec<ArchivedItemPointer>> {
        unsafe { self.map_unchecked_mut(|s| &mut s.neighbor_index_pointers) }
    }

    fn neighbor_distances(self: Pin<&mut Self>) -> Pin<&mut ArchivedVec<Distance>> {
        unsafe { self.map_unchecked_mut(|s| &mut s.neighbor_distances) }
    }

    pub fn apply_to_neightbors<F>(&self, mut f: F)
    where
        F: FnMut(Distance, &ArchivedItemPointer),
    {
        let mut terminate = false;
        for (dist, n) in self
            .neighbor_distances
            .iter()
            .zip(self.neighbor_index_pointers.iter())
            .filter(|(&dist, n)| {
                //stop at FIRST nan
                if dist.is_nan() {
                    terminate = true
                }
                !terminate
            })
        {
            f(*dist, n);
        }
    }
}

//TODO is this right?
pub type Distance = f32;
#[derive(Clone)]
pub struct NeighborWithDistance {
    id: ItemPointer,
    distance: Distance,
}

impl NeighborWithDistance {
    pub fn new(neighbor_index_pointer: ItemPointer, distance: Distance) -> Self {
        Self {
            id: neighbor_index_pointer,
            distance: distance,
        }
    }

    pub fn get_index_pointer_to_neigbor(&self) -> ItemPointer {
        return self.id;
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
        self.id == other.id
    }
}

//promise that PartialEq is reflexive
impl Eq for NeighborWithDistance {}

impl std::hash::Hash for NeighborWithDistance {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

pub struct NodeBuilder<'a> {
    index: &'a PgRelation,
    //maps node's pointer to the representation on disk
    map: std::collections::HashMap<ItemPointer, Vec<NeighborWithDistance>>,
    first: Option<ItemPointer>,
    meta_page: super::build::TsvMetaPage,
}

impl<'a> NodeBuilder<'a> {
    pub fn new(index: &'a PgRelation, meta_page: super::build::TsvMetaPage) -> Self {
        Self {
            index: index,
            map: HashMap::with_capacity(200),
            first: None,
            meta_page: meta_page,
        }
    }

    pub fn insert(&mut self, index_pointer: &ItemPointer, vec: &[f32]) {
        if self.map.len() == 0 {
            self.map.insert(
                index_pointer.clone(),
                Vec::<NeighborWithDistance>::with_capacity(self.meta_page.num_neighbors as _),
            );
            return;
        }

        //TODO: make configurable?
        let search_list_size = 100;
        let (l, v) = self.greedy_search(vec, 1, search_list_size);
        let neighbor_list =
            self.prune_neighbors(index_pointer.clone(), v.unwrap().into_iter().collect());

        //set forward pointers
        self.map
            .insert(index_pointer.clone(), neighbor_list.clone());

        //update back pointers
        for neighbor in neighbor_list {
            self.update_back_pointer(&neighbor.id, index_pointer, neighbor.distance)
        }
    }

    fn update_back_pointer(&mut self, from: &ItemPointer, to: &ItemPointer, distance: f32) {
        let current_links = self.map.get_mut(&from).unwrap();
        if current_links.len() + 1 < self.meta_page.num_neighbors as _ {
            current_links.push(NeighborWithDistance {
                id: to.clone(),
                distance: distance,
            })
        } else {
            let new_list = self.prune_neighbors(
                from.clone(),
                vec![NeighborWithDistance {
                    id: to.clone(),
                    distance: distance,
                }],
            );
            self.map.insert(from.clone(), new_list);
        }
    }

    pub unsafe fn write(&self) {
        //TODO: OPT: do this in order of item pointers
        for (index_pointer, neighbors) in &self.map {
            let mut node = Node::modify(self.index, index_pointer);
            let mut archived = node.get_archived_node();
            for (i, new_neighbor) in neighbors.iter().enumerate() {
                //TODO: why do we need to recreate the archive?
                let mut a_index_pointer = archived.as_mut().neighbor_index_pointer().index_pin(i);
                //TODO hate that we have to set each field like this
                a_index_pointer.block_number = new_neighbor.id.block_number;
                a_index_pointer.offset = new_neighbor.id.offset;

                let mut a_distance = archived.as_mut().neighbor_distances().index_pin(i);
                *a_distance = new_neighbor.distance as Distance;
            }
            //set the marker that the list ended
            if neighbors.len() < self.meta_page.num_neighbors as _ {
                //TODO: why do we need to recreate the archive?
                let archived = node.get_archived_node();
                let mut past_last_distance =
                    archived.neighbor_distances().index_pin(neighbors.len());
                *past_last_distance = Distance::NAN;
            }
            node.commit()
        }
    }
}

impl<'a> Graph for NodeBuilder<'a> {
    fn get_init_ids(&mut self) -> Option<Vec<ItemPointer>> {
        //TODO make this based on centroid. For now, just first node.
        //returns a vector for generality
        match &self.first {
            Some(item) => Some(vec![item.clone()]),
            None => match self.map.keys().next() {
                Some(item) => {
                    self.first = Some(item.clone());
                    Some(vec![item.clone()])
                }
                None => None,
            },
        }
    }

    fn read<'b, 'd>(&'b self, index_pointer: ItemPointer) -> ReadableNode<'d> {
        unsafe { Node::read(self.index, &index_pointer) }
    }

    fn get_neighbors(&self, neighbors_of: ItemPointer) -> Option<Vec<NeighborWithDistance>> {
        let neighbors = self.map.get(&neighbors_of);
        match neighbors {
            Some(n) => Some(n.iter().map(|v| v.clone()).collect()),
            None => None,
        }
    }

    fn get_meta_page(&self) -> &TsvMetaPage {
        &self.meta_page
    }
}

/// Debugging methods
pub fn print_graph_from_disk(index: &PgRelation, init_id: ItemPointer) {
    let mut map = HashMap::<ItemPointer, Vec<f32>>::new();
    let mut sb = String::new();
    unsafe {
        print_graph_from_disk_visitor(&index, init_id, &mut map, &mut sb, 0);
    }
    panic!("{}", sb.as_str())
}

unsafe fn print_graph_from_disk_visitor(
    index: &PgRelation,
    index_pointer: ItemPointer,
    map: &mut HashMap<ItemPointer, Vec<f32>>,
    sb: &mut String,
    level: usize,
) {
    let node = Node::read(&index, &index_pointer);
    let v = node.node.vector.as_slice();
    let copy: Vec<f32> = v.iter().map(|f| *f).collect();
    let name = format!("node {:?}", &copy);

    map.insert(index_pointer, copy);

    node.node.apply_to_neightbors(|dist, neighbor_pointer| {
        let p = neighbor_pointer.deserialize_item_pointer();
        if !map.contains_key(&p) {
            print_graph_from_disk_visitor(index, p, map, sb, level + 1);
        }
    });
    sb.push_str(&name);
    sb.push_str("\n");
    node.node.apply_to_neightbors(|dist, neighbor_pointer| {
        let ip: ItemPointer = (neighbor_pointer)
            .deserialize(&mut rkyv::Infallible)
            .unwrap();
        let neighbor = map.get(&ip).unwrap();
        sb.push_str(&format!("->{:?} dist({})\n", neighbor, dist))
    });
    sb.push_str("\n")
}
