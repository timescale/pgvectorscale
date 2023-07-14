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
use crate::util::{ArchivedItemPointer, ItemPointer, ReadableBuffer, WritableBuffer};

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
    vector: Vec<f32>,
    neighbor_index_pointers: Vec<ItemPointer>,
    neighbor_distances: Vec<f64>, //TODO distance is f64 right?
    heap_item_pointer: ItemPointer,
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
            neighbor_distances: (0..num_neighbors).map(|_| f64::NAN).collect(),
            heap_item_pointer: heap_item_pointer,
            deleted: false,
        }
    }

    fn num_neighbors(&self) -> usize {
        if let Some(index) = self.neighbor_distances.iter().position(|x| *x == f64::NAN) {
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

    fn neighbor_distances(self: Pin<&mut Self>) -> Pin<&mut ArchivedVec<f64>> {
        unsafe { self.map_unchecked_mut(|s| &mut s.neighbor_distances) }
    }

    fn apply_to_neightbors<F>(&self, mut f: F)
    where
        F: FnMut(f64, &ArchivedItemPointer),
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

struct Neighbor {
    id: ItemPointer,
    distance: f32,
    visited: bool,
}

impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}

impl PartialEq for Neighbor {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Neighbor {
    pub fn new(index_pointer: ItemPointer, distance: f32) -> Self {
        Self {
            id: index_pointer,
            distance: distance,
            visited: false,
        }
    }
}

//TODO: use slow L2 for now. Make pluggable and simd
fn distance(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());

    let norm: f32 = a
        .iter()
        .zip(b.iter())
        .map(|t| (*t.0 as f32 - *t.1 as f32) * (*t.0 as f32 - *t.1 as f32))
        .sum();
    assert!(norm >= 0.);
    norm.sqrt()
}

struct ListSearchResult {
    search_list_size: usize,
    best_candidate: Vec<Neighbor>, //keep sorted by distanced
    inserted: HashSet<ItemPointer>,
}

impl ListSearchResult {
    fn new(
        search_list_size: usize,
        builder: &NodeBuilder,
        init_ids: Vec<ItemPointer>,
        query: &[f32],
    ) -> Self {
        let mut res = Self {
            search_list_size: search_list_size,
            best_candidate: Vec::with_capacity(search_list_size),
            inserted: HashSet::with_capacity(search_list_size),
        };
        for index_pointer in init_ids {
            res.insert(builder, &index_pointer, query);
        }
        res
    }

    fn insert(&mut self, builder: &NodeBuilder, index_pointer: &ItemPointer, query: &[f32]) {
        //no point reprocessing a point. Distance calcs are expensive.
        if !self.inserted.insert(index_pointer.clone()) {
            return;
        }

        let node = builder.read(index_pointer);
        let vec = node.node.vector.as_slice();
        let distance = distance(vec, query);

        let neighbor = Neighbor::new(index_pointer.clone(), distance);
        self._insert_neighbor(neighbor);
    }

    /// Internal function
    fn _insert_neighbor(&mut self, n: Neighbor) {
        if self.best_candidate.len() >= self.search_list_size {
            let last = self.best_candidate.last().unwrap();
            if n >= *last {
                //n is too far in the list to be the best candidate.
                return;
            }
            self.best_candidate.pop();
        }

        //insert while preserving sort order.
        let idx = self.best_candidate.partition_point(|x| *x < n);
        self.best_candidate.insert(idx, n)
    }

    fn visit_closest(&mut self) -> Option<(ItemPointer, f32)> {
        //OPT: should we optimize this not to do a linear search each time?
        let neighbor = self.best_candidate.iter_mut().find(|n| !n.visited);
        match neighbor {
            Some(n) => {
                (*n).visited = true;
                Some((n.id.clone(), n.distance))
            }
            None => None,
        }
    }

    fn get_closets_nth(&self, index: usize) -> Option<ItemPointer> {
        let neighbor = self.best_candidate.get(index);
        match neighbor {
            Some(n) => Some(n.id.clone()),
            None => None,
        }
    }
}

#[derive(Clone)]
struct NodeBuilderNeighbor {
    id: ItemPointer,
    distance: f32,
}

impl PartialOrd for NodeBuilderNeighbor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}

impl Ord for NodeBuilderNeighbor {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance.total_cmp(&other.distance)
    }
}

impl PartialEq for NodeBuilderNeighbor {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

//promise that PartialEq is reflexive
impl Eq for NodeBuilderNeighbor {}

impl std::hash::Hash for NodeBuilderNeighbor {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

pub struct NodeBuilder<'a> {
    index: &'a PgRelation,
    //maps node's pointer to the representation on disk
    map: std::collections::HashMap<ItemPointer, Vec<NodeBuilderNeighbor>>,
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
                Vec::<NodeBuilderNeighbor>::with_capacity(self.meta_page.num_neighbors as _),
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
            current_links.push(NodeBuilderNeighbor {
                id: to.clone(),
                distance: distance,
            })
        } else {
            let new_list = self.prune_neighbors(
                from.clone(),
                vec![NodeBuilderNeighbor {
                    id: to.clone(),
                    distance: distance,
                }],
            );
            self.map.insert(from.clone(), new_list);
        }
    }

    /// Prune neigbors by prefering neighbors closer to the point in question
    /// than to other neighbors of the point.
    ///
    /// TODO: this is the ann-disk implementation. There may be better implementations
    /// if we save the factors or the distances and add incrementally. Not sure.
    fn prune_neighbors(
        &mut self,
        index_pointer: ItemPointer,
        new_neigbors: Vec<NodeBuilderNeighbor>,
    ) -> Vec<NodeBuilderNeighbor> {
        //TODO make configurable?
        let max_alpha = 1.2;
        //get a unique candidate pool
        let mut candidates = match self.map.get(&index_pointer) {
            Some(v) => v.clone(),
            None => vec![], //new point has no entry in the map yet
        };
        let mut hash: HashSet<ItemPointer> = candidates.iter().map(|c| c.id.clone()).collect();
        for n in new_neigbors {
            if hash.insert(n.id.clone()) {
                candidates.push(n);
            }
        }
        //remove myself
        if !hash.insert(index_pointer.clone()) {
            //prevent self-loops
            let index = candidates
                .iter()
                .position(|x| x.id == index_pointer)
                .unwrap();
            candidates.remove(index);
        }
        //TODO remove deleted nodes

        //TODO diskann has something called max_occlusion_size/max_candidate_size(default:750). Do we need to implement?

        //sort by distance
        candidates.sort();
        let mut results =
            Vec::<NodeBuilderNeighbor>::with_capacity(self.meta_page.num_neighbors as _);

        let mut max_factors: Vec<f32> = vec![0.0; candidates.len()];

        let mut alpha = 1.0;
        //first we add nodes that "pass" a small alpha. Then, if there
        //is still room we loop again with a larger alpha.
        while alpha <= max_alpha && results.len() < self.meta_page.num_neighbors as _ {
            for (i, neighbor) in candidates.iter().enumerate() {
                if results.len() >= self.meta_page.num_neighbors as _ {
                    return results;
                }
                if max_factors[i] > alpha {
                    continue;
                }

                //don't consider again
                max_factors[i] = f32::MAX;
                results.push(neighbor.clone());

                //we've now added this to the results so it's going to be a neighbor
                //rename for clarity.
                let existing_neighbor = neighbor;

                //TODO make lazy
                let existing_neighbor_node = self.read(&existing_neighbor.id);
                let existing_neighbor_vec = existing_neighbor_node.node.vector.as_slice();

                //go thru the other candidates (tail of the list)
                for (j, candidate_neighbor) in candidates.iter().enumerate().skip(i + 1) {
                    //has it been completely excluded?
                    if max_factors[j] > max_alpha {
                        continue;
                    }

                    let candidate_node = self.read(&candidate_neighbor.id);
                    let candidate_vec = candidate_node.node.vector.as_slice();
                    let distance_between_candidate_and_existing_neighbor =
                        distance(existing_neighbor_vec, candidate_vec);
                    let distance_between_candidate_and_point = candidate_neighbor.distance;
                    //factor is high if the candidate is closer to an existing neighbor than the point it's being considered for
                    let factor = if distance_between_candidate_and_existing_neighbor == 0.0 {
                        f32::MAX //avoid division by 0
                    } else {
                        distance_between_candidate_and_point
                            / distance_between_candidate_and_existing_neighbor
                    };
                    max_factors[j] = max_factors[j].max(factor)
                }
            }
            alpha = alpha * 1.2
        }
        results
    }

    pub fn get_init_ids(&mut self) -> Option<Vec<ItemPointer>> {
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

    fn read<'b, 'c, 'd>(&'b self, index_pointer: &'c ItemPointer) -> ReadableNode<'d> {
        unsafe { Node::read(self.index, index_pointer) }
    }

    fn greedy_search(
        &mut self,
        query: &[f32],
        k: usize,
        search_list_size: usize,
    ) -> (Vec<ItemPointer>, Option<HashSet<NodeBuilderNeighbor>>) {
        assert!(k <= search_list_size);

        let init_ids = self.get_init_ids();
        if let None = init_ids {
            //no nodes in the graph
            return (vec![], None);
        }
        let mut l = ListSearchResult::new(search_list_size, self, init_ids.unwrap(), query);

        //OPT: Only build v when needed.
        let mut v: HashSet<_> = HashSet::<NodeBuilderNeighbor>::with_capacity(search_list_size);
        while let Some((index_pointer, distance)) = l.visit_closest() {
            let neighbors = self.map.get(&index_pointer);
            if let None = neighbors {
                panic!("Nodes in the list search results that aren't in the builder");
            }

            for neighbor_index_pointer in neighbors.unwrap() {
                l.insert(self, &neighbor_index_pointer.id, query)
            }
            v.insert(NodeBuilderNeighbor {
                id: index_pointer,
                distance: distance,
            });
        }

        let mut k_closets = Vec::<ItemPointer>::with_capacity(k);
        for i in (0..k) {
            let item = l.get_closets_nth(i);
            match item {
                Some(pointer) => k_closets.push(pointer),
                None => break,
            }
        }
        (k_closets, Some(v))
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
                *a_distance = new_neighbor.distance as f64;
            }
            //set the marker that the list ended
            if neighbors.len() < self.meta_page.num_neighbors as _ {
                //TODO: why do we need to recreate the archive?
                let archived = node.get_archived_node();
                let mut past_last_distance =
                    archived.neighbor_distances().index_pin(neighbors.len());
                *past_last_distance = f64::NAN;
            }
            node.commit()
        }
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
        let p = neighbor_pointer.deserialize(&mut rkyv::Infallible).unwrap();
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
