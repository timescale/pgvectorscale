use std::{cmp::Ordering, collections::HashSet};

use pgrx::pg_sys::{Datum, TupleTableSlot};
use pgrx::{pg_sys, PgBox, PgRelation};

use crate::access_method::model::Node;
use crate::util::ports::slot_getattr;
use crate::util::{HeapPointer, IndexPointer, ItemPointer};

use super::model::{ArchivedNode, PgVector};
use super::quantizer::Quantizer;
use super::{
    meta_page::MetaPage,
    model::{NeighborWithDistance, ReadableNode},
};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn distance(a: &[f32], b: &[f32]) -> f32 {
    super::distance_x86::distance_opt_runtime_select(a, b)
}

//TODO: use slow L2 for now. Make pluggable and simd
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
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

struct TableSlot {
    slot: PgBox<TupleTableSlot>,
}

impl TableSlot {
    unsafe fn new(relation: &PgRelation) -> Self {
        let slot = PgBox::from_pg(pg_sys::table_slot_create(
            relation.as_ptr(),
            std::ptr::null_mut(),
        ));
        Self { slot }
    }

    unsafe fn get_attribute(&self, attribute_number: pg_sys::AttrNumber) -> Option<Datum> {
        slot_getattr(&self.slot, attribute_number)
    }
}

impl Drop for TableSlot {
    fn drop(&mut self) {
        unsafe { pg_sys::ExecDropSingleTupleTableSlot(self.slot.as_ptr()) };
    }
}

#[derive(Clone)]
pub struct VectorProvider<'a> {
    quantizer: &'a Quantizer,
    calc_distance_with_quantizer: bool,
    heap_rel: Option<&'a PgRelation>,
    heap_attr_number: Option<pg_sys::AttrNumber>,
    distance_fn: fn(&[f32], &[f32]) -> f32,
}

impl<'a> VectorProvider<'a> {
    pub fn new(
        heap_rel: Option<&'a PgRelation>,
        heap_attr_number: Option<pg_sys::AttrNumber>,
        quantizer: &'a Quantizer,
        calc_distance_with_quantizer: bool,
    ) -> Self {
        Self {
            quantizer,
            calc_distance_with_quantizer,
            heap_rel,
            heap_attr_number,
            distance_fn: distance,
        }
    }

    pub unsafe fn get_full_vector_copy_from_heap_pointer(
        &self,
        heap_pointer: ItemPointer,
    ) -> Vec<f32> {
        let slot = TableSlot::new(self.heap_rel.unwrap());
        self.init_slot(&slot, heap_pointer);
        let slice = self.get_slice(&slot);
        slice.to_vec()
    }

    unsafe fn init_slot(&self, slot: &TableSlot, heap_pointer: HeapPointer) {
        let table_am = self.heap_rel.unwrap().rd_tableam;
        let fetch_row_version = (*table_am).tuple_fetch_row_version.unwrap();
        let mut ctid: pg_sys::ItemPointerData = pg_sys::ItemPointerData {
            ..Default::default()
        };
        heap_pointer.to_item_pointer_data(&mut ctid);
        fetch_row_version(
            self.heap_rel.unwrap().as_ptr(),
            &mut ctid,
            &mut pg_sys::SnapshotAnyData,
            slot.slot.as_ptr(),
        );
    }

    unsafe fn get_slice<'s>(&self, slot: &'s TableSlot) -> &'s [f32] {
        let vector =
            PgVector::from_datum(slot.get_attribute(self.heap_attr_number.unwrap()).unwrap());

        //note pgvector slice is only valid as long as the slot is valid that's why the lifetime is tied to it.
        (*vector).to_slice()
    }

    fn get_heap_pointer(&self, index: &PgRelation, index_pointer: IndexPointer) -> HeapPointer {
        let rn = unsafe { Node::read(index, index_pointer) };
        let node = rn.get_archived_node();
        let heap_pointer = node.heap_item_pointer.deserialize_item_pointer();
        heap_pointer
    }

    fn get_distance_measure(&self, query: &[f32]) -> DistanceMeasure {
        return DistanceMeasure::new(self.quantizer, query, self.calc_distance_with_quantizer);
    }

    unsafe fn get_distance(
        &self,
        node: &ArchivedNode,
        query: &[f32],
        dm: &DistanceMeasure,
        stats: &mut GreedySearchStats,
    ) -> f32 {
        if self.calc_distance_with_quantizer {
            assert!(node.pq_vector.len() > 0);
            let vec = node.pq_vector.as_slice();
            let distance = dm.get_quantized_distance(vec);
            stats.pq_distance_comparisons += 1;
            stats.distance_comparisons += 1;
            return distance;
        }

        //now we know we're doing a distance calc on the full-sized vector
        if self.quantizer.is_some() {
            //have to get it from the heap
            let heap_pointer = node.heap_item_pointer.deserialize_item_pointer();
            let slot = TableSlot::new(self.heap_rel.unwrap());
            self.init_slot(&slot, heap_pointer);
            let slice = self.get_slice(&slot);
            let distance = dm.get_full_vector_distance(slice, query);
            stats.distance_comparisons += 1;
            return distance;
        } else {
            //have to get it from the index
            assert!(node.vector.len() > 0);
            let vec = node.vector.as_slice();
            let distance = dm.get_full_vector_distance(vec, query);
            stats.distance_comparisons += 1;
            return distance;
        }
    }

    pub unsafe fn get_full_vector_distance_state<'i>(
        &self,
        index: &'i PgRelation,
        index_pointer: IndexPointer,
    ) -> FullVectorDistanceState<'i> {
        if self.quantizer.is_some() {
            let heap_pointer = self.get_heap_pointer(index, index_pointer);
            let slot = TableSlot::new(self.heap_rel.unwrap());
            self.init_slot(&slot, heap_pointer);
            FullVectorDistanceState {
                table_slot: Some(slot),
                readable_node: None,
            }
        } else {
            let rn = Node::read(index, index_pointer);
            FullVectorDistanceState {
                table_slot: None,
                readable_node: Some(rn),
            }
        }
    }

    pub unsafe fn get_distance_pair_for_full_vectors_from_state(
        &self,
        state: &FullVectorDistanceState,
        index: &PgRelation,
        index_pointer: IndexPointer,
    ) -> f32 {
        if self.quantizer.is_some() {
            let heap_pointer = self.get_heap_pointer(index, index_pointer);
            let slot = TableSlot::new(self.heap_rel.unwrap());
            self.init_slot(&slot, heap_pointer);
            let slice1 = self.get_slice(&slot);
            let slice2 = self.get_slice(state.table_slot.as_ref().unwrap());
            (self.distance_fn)(slice1, slice2)
        } else {
            let rn1 = Node::read(index, index_pointer);
            let rn2 = state.readable_node.as_ref().unwrap();
            let node1 = rn1.get_archived_node();
            let node2 = rn2.get_archived_node();
            assert!(node1.vector.len() > 0);
            assert!(node1.vector.len() == node2.vector.len());
            let vec1 = node1.vector.as_slice();
            let vec2 = node2.vector.as_slice();
            (self.distance_fn)(vec1, vec2)
        }
    }
}

pub struct FullVectorDistanceState<'a> {
    table_slot: Option<TableSlot>,
    readable_node: Option<ReadableNode<'a>>,
}

pub struct DistanceMeasure {
    pq_distance_table: Option<super::pq::PqDistanceTable>,
}

impl DistanceMeasure {
    pub fn new(quantizer: &Quantizer, query: &[f32], calc_distance_with_quantizer: bool) -> Self {
        if !calc_distance_with_quantizer {
            return Self {
                pq_distance_table: None,
            };
        }
        match quantizer {
            Quantizer::None => Self {
                pq_distance_table: None,
            },
            Quantizer::PQ(pq) => {
                let dc = pq.get_distance_table(query, distance);
                Self {
                    pq_distance_table: Some(dc),
                }
            }
        }
    }

    fn get_quantized_distance(&self, vec: &[u8]) -> f32 {
        let dc = self.pq_distance_table.as_ref().unwrap();
        let distance = dc.distance(vec);
        distance
    }

    fn get_full_vector_distance(&self, vec: &[f32], query: &[f32]) -> f32 {
        assert!(self.pq_distance_table.is_none());
        distance(vec, query)
    }
}

struct ListSearchNeighbor {
    index_pointer: IndexPointer,
    heap_pointer: HeapPointer,
    neighbor_index_pointers: Vec<IndexPointer>,
    distance: f32,
    visited: bool,
}

impl PartialOrd for ListSearchNeighbor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}

impl PartialEq for ListSearchNeighbor {
    fn eq(&self, other: &Self) -> bool {
        self.index_pointer == other.index_pointer
    }
}

impl ListSearchNeighbor {
    pub fn new(
        index_pointer: IndexPointer,
        heap_pointer: HeapPointer,
        distance: f32,
        neighbor_index_pointers: Vec<IndexPointer>,
    ) -> Self {
        Self {
            index_pointer,
            heap_pointer,
            neighbor_index_pointers,
            distance,
            visited: false,
        }
    }
}

pub struct ListSearchResult {
    best_candidate: Vec<ListSearchNeighbor>, //keep sorted by distanced
    inserted: HashSet<ItemPointer>,
    max_history_size: Option<usize>,
    dm: DistanceMeasure,
    pub stats: GreedySearchStats,
}

impl ListSearchResult {
    fn empty() -> Self {
        Self {
            best_candidate: vec![],
            inserted: HashSet::new(),
            max_history_size: None,
            dm: DistanceMeasure {
                pq_distance_table: None,
            },
            stats: GreedySearchStats::new(),
        }
    }

    fn new<G>(
        index: &PgRelation,
        max_history_size: Option<usize>,
        graph: &G,
        init_ids: Vec<ItemPointer>,
        query: &[f32],
        dm: DistanceMeasure,
        search_list_size: usize,
        meta_page: &MetaPage,
    ) -> Self
    where
        G: Graph + ?Sized,
    {
        let neigbors = meta_page.get_num_neighbors() as usize;
        let mut res = Self {
            best_candidate: Vec::with_capacity(search_list_size * neigbors),
            inserted: HashSet::with_capacity(search_list_size * neigbors),
            max_history_size,
            stats: GreedySearchStats::new(),
            dm: dm,
        };
        res.stats.calls += 1;
        for index_pointer in init_ids {
            res.insert(index, graph, index_pointer, query);
        }
        res
    }

    fn insert<G>(
        &mut self,
        index: &PgRelation,
        graph: &G,
        index_pointer: ItemPointer,
        query: &[f32],
    ) where
        G: Graph + ?Sized,
    {
        //no point reprocessing a point. Distance calcs are expensive.
        if !self.inserted.insert(index_pointer) {
            return;
        }

        let rn = unsafe { Node::read(index, index_pointer) };
        self.stats.node_reads += 1;
        let node = rn.get_archived_node();

        let vp = graph.get_vector_provider();
        let distance = unsafe { vp.get_distance(node, query, &self.dm, &mut self.stats) };

        let neighbors = graph.get_neighbors(node, index_pointer);
        let lsn = ListSearchNeighbor::new(
            index_pointer,
            node.heap_item_pointer.deserialize_item_pointer(),
            distance,
            neighbors,
        );
        self._insert_neighbor(lsn);
    }

    /// Internal function
    fn _insert_neighbor(&mut self, n: ListSearchNeighbor) {
        if let Some(max_size) = self.max_history_size {
            if self.best_candidate.len() >= max_size {
                let last = self.best_candidate.last().unwrap();
                if n >= *last {
                    //n is too far in the list to be the best candidate.
                    return;
                }
                self.best_candidate.pop();
            }
        }
        //insert while preserving sort order.
        let idx = self.best_candidate.partition_point(|x| *x < n);
        self.best_candidate.insert(idx, n)
    }

    fn visit_closest(&mut self, pos_limit: usize) -> Option<&ListSearchNeighbor> {
        //OPT: should we optimize this not to do a linear search each time?
        let neighbor_position = self.best_candidate.iter().position(|n| !n.visited);
        match neighbor_position {
            Some(pos) => {
                if pos > pos_limit {
                    return None;
                }
                let n = &mut self.best_candidate[pos];
                n.visited = true;
                Some(n)
            }
            None => None,
        }
    }

    //removes and returns the first element. Given that the element remains in self.inserted, that means the element will never again be insereted
    //into the best_candidate list, so it will never again be returned.
    pub fn consume(&mut self) -> Option<(HeapPointer, IndexPointer)> {
        if self.best_candidate.is_empty() {
            return None;
        }
        let f = self.best_candidate.remove(0);
        return Some((f.heap_pointer, f.index_pointer));
    }
}

pub trait Graph {
    fn read<'a>(&self, index: &'a PgRelation, index_pointer: ItemPointer) -> ReadableNode<'a>;
    fn get_init_ids(&self) -> Option<Vec<ItemPointer>>;
    fn get_neighbors(&self, node: &ArchivedNode, neighbors_of: ItemPointer) -> Vec<IndexPointer>;
    fn get_neighbors_with_distances(
        &self,
        index: &PgRelation,
        neighbors_of: ItemPointer,
        result: &mut Vec<NeighborWithDistance>,
    ) -> bool;

    fn is_empty(&self) -> bool;

    fn get_vector_provider(&self) -> VectorProvider;

    fn set_neighbors(
        &mut self,
        index: &PgRelation,
        neighbors_of: ItemPointer,
        new_neighbors: Vec<NeighborWithDistance>,
    );

    fn get_meta_page(&self, index: &PgRelation) -> &MetaPage;

    /// greedy search looks for the closest neighbors to a query vector
    /// You may think that this needs the "K" parameter but it does not,
    /// instead it uses a search_list_size parameter (>K).
    ///
    /// The basic logic is you do a greedy search until you've evaluated the
    /// neighbors of the `search_list_size` closest nodes.
    ///
    /// To get the K closest neighbors, you then get the first K items in the ListSearchResult
    /// return items.
    ///
    /// Note this is the one-shot implementation that keeps only the closest `search_list_size` results in
    /// the returned ListSearchResult elements. It shouldn't be used with self.greedy_search_iterate
    fn greedy_search(
        &self,
        index: &PgRelation,
        query: &[f32],
        meta_page: &MetaPage,
    ) -> (ListSearchResult, HashSet<NeighborWithDistance>)
    where
        Self: Graph,
    {
        let init_ids = self.get_init_ids();
        if let None = init_ids {
            //no nodes in the graph
            return (ListSearchResult::empty(), HashSet::with_capacity(0));
        }
        let dm = self.get_vector_provider().get_distance_measure(query);
        let search_list_size = meta_page.get_search_list_size_for_build() as usize;

        let mut l = ListSearchResult::new(
            index,
            Some(search_list_size),
            self,
            init_ids.unwrap(),
            query,
            dm,
            search_list_size,
            meta_page,
        );
        let mut visited_nodes = HashSet::with_capacity(search_list_size);
        self.greedy_search_iterate(
            &mut l,
            index,
            query,
            search_list_size,
            Some(&mut visited_nodes),
        );
        return (l, visited_nodes);
    }

    /// Returns a ListSearchResult initialized for streaming. The output should be used with greedy_search_iterate to obtain
    /// the next elements.
    fn greedy_search_streaming_init(
        &self,
        index: &PgRelation,
        query: &[f32],
        search_list_size: usize,
        meta_page: &MetaPage,
    ) -> ListSearchResult {
        let init_ids = self.get_init_ids();
        if let None = init_ids {
            //no nodes in the graph
            return ListSearchResult::empty();
        }
        let dm = self.get_vector_provider().get_distance_measure(query);

        ListSearchResult::new(
            index,
            None,
            self,
            init_ids.unwrap(),
            query,
            dm,
            search_list_size,
            meta_page,
        )
    }

    /// Advance the state of the lsr until the closest `visit_n_closest` elements have been visited.
    fn greedy_search_iterate(
        &self,
        lsr: &mut ListSearchResult,
        index: &PgRelation,
        query: &[f32],
        visit_n_closest: usize,
        mut visited_nodes: Option<&mut HashSet<NeighborWithDistance>>,
    ) where
        Self: Graph,
    {
        //OPT: Only build v when needed.
        let mut neighbors =
            Vec::<IndexPointer>::with_capacity(self.get_meta_page(index).get_num_neighbors() as _);
        while let Some(list_search_entry) = lsr.visit_closest(visit_n_closest) {
            neighbors.clear();
            match visited_nodes {
                None => {}
                Some(ref mut visited_nodes) => {
                    visited_nodes.insert(NeighborWithDistance::new(
                        list_search_entry.index_pointer,
                        list_search_entry.distance,
                    ));
                }
            }
            neighbors.extend_from_slice(list_search_entry.neighbor_index_pointers.as_slice());
            for neighbor_index_pointer in neighbors.iter() {
                lsr.insert(index, self, *neighbor_index_pointer, query)
            }
        }
    }

    /// Prune neigbors by prefering neighbors closer to the point in question
    /// than to other neighbors of the point.
    ///
    /// TODO: this is the ann-disk implementation. There may be better implementations
    /// if we save the factors or the distances and add incrementally. Not sure.
    fn prune_neighbors(
        &self,
        index: &PgRelation,
        index_pointer: ItemPointer,
        new_neigbors: Vec<NeighborWithDistance>,
    ) -> (Vec<NeighborWithDistance>, PruneNeighborStats) {
        let mut stats = PruneNeighborStats::new();
        stats.calls += 1;
        //TODO make configurable?
        let max_alpha = self.get_meta_page(index).get_max_alpha();
        //get a unique candidate pool
        let mut candidates = Vec::<NeighborWithDistance>::with_capacity(
            (self.get_meta_page(index).get_num_neighbors() as usize) + new_neigbors.len(),
        );
        self.get_neighbors_with_distances(index, index_pointer, &mut candidates);

        let mut hash: HashSet<ItemPointer> = candidates
            .iter()
            .map(|c| c.get_index_pointer_to_neighbor())
            .collect();
        for n in new_neigbors {
            if hash.insert(n.get_index_pointer_to_neighbor()) {
                candidates.push(n);
            }
        }
        //remove myself
        if !hash.insert(index_pointer) {
            //prevent self-loops
            let index = candidates
                .iter()
                .position(|x| x.get_index_pointer_to_neighbor() == index_pointer)
                .unwrap();
            candidates.remove(index);
        }
        //TODO remove deleted nodes

        //TODO diskann has something called max_occlusion_size/max_candidate_size(default:750). Do we need to implement?

        //sort by distance
        candidates.sort();
        let mut results = Vec::<NeighborWithDistance>::with_capacity(
            self.get_meta_page(index).get_max_neighbors_during_build(),
        );

        let mut max_factors: Vec<f64> = vec![0.0; candidates.len()];

        let mut alpha = 1.0;
        //first we add nodes that "pass" a small alpha. Then, if there
        //is still room we loop again with a larger alpha.
        while alpha <= max_alpha
            && results.len() < self.get_meta_page(index).get_num_neighbors() as _
        {
            for (i, neighbor) in candidates.iter().enumerate() {
                if results.len() >= self.get_meta_page(index).get_num_neighbors() as _ {
                    return (results, stats);
                }
                if max_factors[i] > alpha {
                    continue;
                }

                //don't consider again
                max_factors[i] = f64::MAX;
                results.push(neighbor.clone());

                //we've now added this to the results so it's going to be a neighbor
                //rename for clarity.
                let existing_neighbor = neighbor;

                let vp = self.get_vector_provider();
                let dist_state = unsafe {
                    vp.get_full_vector_distance_state(
                        index,
                        existing_neighbor.get_index_pointer_to_neighbor(),
                    )
                };

                //go thru the other candidates (tail of the list)
                for (j, candidate_neighbor) in candidates.iter().enumerate().skip(i + 1) {
                    //has it been completely excluded?
                    if max_factors[j] > max_alpha {
                        continue;
                    }

                    //todo handle the non-pq case
                    let distance_between_candidate_and_existing_neighbor = unsafe {
                        vp.get_distance_pair_for_full_vectors_from_state(
                            &dist_state,
                            index,
                            candidate_neighbor.get_index_pointer_to_neighbor(),
                        )
                    };
                    stats.node_reads += 2;
                    stats.distance_comparisons += 1;
                    let distance_between_candidate_and_point = candidate_neighbor.get_distance();
                    //factor is high if the candidate is closer to an existing neighbor than the point it's being considered for
                    let factor = if distance_between_candidate_and_existing_neighbor == 0.0 {
                        f64::MAX //avoid division by 0
                    } else {
                        distance_between_candidate_and_point as f64
                            / distance_between_candidate_and_existing_neighbor as f64
                    };
                    max_factors[j] = max_factors[j].max(factor)
                }
            }
            alpha = alpha * 1.2
        }
        (results, stats)
    }

    fn insert(
        &mut self,
        index: &PgRelation,
        index_pointer: IndexPointer,
        vec: &[f32],
    ) -> InsertStats {
        let mut prune_neighbor_stats: PruneNeighborStats = PruneNeighborStats::new();
        let mut greedy_search_stats = GreedySearchStats::new();
        let meta_page = self.get_meta_page(index);
        if self.is_empty() {
            self.set_neighbors(
                index,
                index_pointer,
                Vec::<NeighborWithDistance>::with_capacity(
                    meta_page.get_max_neighbors_during_build() as _,
                ),
            );
            return InsertStats {
                prune_neighbor_stats: prune_neighbor_stats,
                greedy_search_stats: greedy_search_stats,
            };
        }

        //TODO: make configurable?
        let (l, v) = self.greedy_search(index, vec, meta_page);
        greedy_search_stats.combine(l.stats);
        let (neighbor_list, forward_stats) =
            self.prune_neighbors(index, index_pointer, v.into_iter().collect());
        prune_neighbor_stats.combine(forward_stats);

        //set forward pointers
        self.set_neighbors(index, index_pointer, neighbor_list.clone());

        //update back pointers
        let mut cnt = 0;
        for neighbor in neighbor_list {
            let (needed_prune, backpointer_stats) = self.update_back_pointer(
                index,
                neighbor.get_index_pointer_to_neighbor(),
                index_pointer,
                neighbor.get_distance(),
            );
            if needed_prune {
                cnt = cnt + 1;
            }
            prune_neighbor_stats.combine(backpointer_stats);
        }
        //info!("pruned {} neighbors", cnt);
        return InsertStats {
            prune_neighbor_stats,
            greedy_search_stats,
        };
    }

    fn update_back_pointer(
        &mut self,
        index: &PgRelation,
        from: IndexPointer,
        to: IndexPointer,
        distance: f32,
    ) -> (bool, PruneNeighborStats) {
        let mut current_links = Vec::<NeighborWithDistance>::new();
        self.get_neighbors_with_distances(index, from, &mut current_links);

        if current_links.len() < current_links.capacity() as _ {
            current_links.push(NeighborWithDistance::new(to, distance));
            self.set_neighbors(index, from, current_links);
            (false, PruneNeighborStats::new())
        } else {
            //info!("sizes {} {} {}", current_links.len() + 1, current_links.capacity(), self.meta_page.get_max_neighbors_during_build());
            //Note prune_neighbors will reduce to current_links.len() to num_neighbors while capacity is num_neighbors * 1.3
            //thus we are avoiding prunning every time
            let (new_list, stats) =
                self.prune_neighbors(index, from, vec![NeighborWithDistance::new(to, distance)]);
            self.set_neighbors(index, from, new_list);
            (true, stats)
        }
    }
}

#[derive(Debug)]
pub struct PruneNeighborStats {
    pub calls: usize,
    pub distance_comparisons: usize,
    pub node_reads: usize,
}

impl PruneNeighborStats {
    pub fn new() -> Self {
        PruneNeighborStats {
            calls: 0,
            distance_comparisons: 0,
            node_reads: 0,
        }
    }

    pub fn combine(&mut self, other: Self) {
        self.calls += other.calls;
        self.distance_comparisons += other.distance_comparisons;
        self.node_reads += other.node_reads;
    }
}

#[derive(Debug)]
pub struct GreedySearchStats {
    pub calls: usize,
    pub distance_comparisons: usize,
    pub node_reads: usize,
    pub pq_distance_comparisons: usize,
}

impl GreedySearchStats {
    pub fn new() -> Self {
        GreedySearchStats {
            calls: 0,
            distance_comparisons: 0,
            node_reads: 0,
            pq_distance_comparisons: 0,
        }
    }

    pub fn combine(&mut self, other: Self) {
        self.calls += other.calls;
        self.distance_comparisons += other.distance_comparisons;
        self.node_reads += other.node_reads;
        self.pq_distance_comparisons += other.pq_distance_comparisons;
    }
}

#[derive(Debug)]
pub struct InsertStats {
    pub prune_neighbor_stats: PruneNeighborStats,
    pub greedy_search_stats: GreedySearchStats,
}

impl InsertStats {
    pub fn new() -> Self {
        return InsertStats {
            prune_neighbor_stats: PruneNeighborStats::new(),
            greedy_search_stats: GreedySearchStats::new(),
        };
    }

    pub fn combine(&mut self, other: InsertStats) {
        self.prune_neighbor_stats
            .combine(other.prune_neighbor_stats);
        self.greedy_search_stats.combine(other.greedy_search_stats);
    }
}
