use std::{cmp::Ordering, collections::HashSet};

use pgrx::pg_sys::{Datum, TupleTableSlot};
use pgrx::{pg_sys, PgBox, PgRelation};

use crate::util::ports::slot_getattr;
use crate::util::{HeapPointer, IndexPointer, ItemPointer};

use super::builder_graph::BuilderGraph;
use super::disk_index_graph::DiskIndexGraph;
use super::distance::distance_cosine as default_distance;
use super::model::PgVector;
use super::quantizer::Quantizer;
use super::{
    meta_page::MetaPage,
    model::{NeighborWithDistance, ReadableNode},
};

pub struct TableSlot {
    slot: PgBox<TupleTableSlot>,
    attribute_number: pg_sys::AttrNumber,
}

impl TableSlot {
    pub unsafe fn new(
        heap_rel: &PgRelation,
        heap_pointer: HeapPointer,
        attribute_number: pg_sys::AttrNumber,
    ) -> Self {
        let slot = PgBox::from_pg(pg_sys::table_slot_create(
            heap_rel.as_ptr(),
            std::ptr::null_mut(),
        ));

        let table_am = heap_rel.rd_tableam;
        let fetch_row_version = (*table_am).tuple_fetch_row_version.unwrap();
        let mut ctid: pg_sys::ItemPointerData = pg_sys::ItemPointerData {
            ..Default::default()
        };
        heap_pointer.to_item_pointer_data(&mut ctid);
        fetch_row_version(
            heap_rel.as_ptr(),
            &mut ctid,
            &mut pg_sys::SnapshotAnyData,
            slot.as_ptr(),
        );

        Self {
            slot,
            attribute_number,
        }
    }

    unsafe fn get_attribute(&self, attribute_number: pg_sys::AttrNumber) -> Option<Datum> {
        slot_getattr(&self.slot, attribute_number)
    }

    pub unsafe fn get_slice(&self) -> &[f32] {
        let vector = PgVector::from_datum(self.get_attribute(self.attribute_number).unwrap());

        //note pgvector slice is only valid as long as the slot is valid that's why the lifetime is tied to it.
        (*vector).to_slice()
    }
}

impl Drop for TableSlot {
    fn drop(&mut self) {
        unsafe { pg_sys::ExecDropSingleTupleTableSlot(self.slot.as_ptr()) };
    }
}

/*

#[derive(Clone)]
pub struct VectorProvider<'a> {
    //  quantizer: &'a Quantizer,
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
            //quantizer,
            calc_distance_with_quantizer,
            heap_rel,
            heap_attr_number,
            distance_fn: default_distance,
        }
    }

    pub unsafe fn get_full_vector_copy_from_heap_pointer(
        &self,
        heap_pointer: ItemPointer,
    ) -> Vec<f32> {
        let slot = TableSlot::new(
            self.heap_rel.unwrap(),
            heap_pointer,
            self.heap_attr_number.unwrap(),
        );
        let slice = self.get_slice(&slot);
        slice.to_vec()
    }

    unsafe fn get_slice<'s>(&self, slot: &'s TableSlot) -> &'s [f32] {
        let vector =
            PgVector::from_datum(slot.get_attribute(self.heap_attr_number.unwrap()).unwrap());

        //note pgvector slice is only valid as long as the slot is valid that's why the lifetime is tied to it.
        (*vector).to_slice()
    }

    pub fn get_heap_pointer(&self, index: &PgRelation, index_pointer: IndexPointer) -> HeapPointer {
        let rn = unsafe { Node::read(index, index_pointer) };
        let node = rn.get_archived_node();
        let heap_pointer = node.heap_item_pointer.deserialize_item_pointer();
        heap_pointer
    }

    fn get_distance_measure(&self, query: &[f32]) -> DistanceMeasure {
        return DistanceMeasure::new(self.quantizer, query, self.calc_distance_with_quantizer);
    }

    pub unsafe fn get_distance(
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
            let slot = TableSlot::new(
                self.heap_rel.unwrap(),
                heap_pointer,
                self.heap_attr_number.unwrap(),
            );
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

    pub unsafe fn get_neighbors_with_distances<'i>(
        &self,
        index: &'i PgRelation,
        index_pointer: IndexPointer,
    ) -> FullVectorDistanceState<'i> {
        if self.quantizer.is_some() {
            let heap_pointer = self.get_heap_pointer(index, index_pointer);
            let slot = TableSlot::new(
                self.heap_rel.unwrap(),
                heap_pointer,
                self.heap_attr_number.unwrap(),
            );
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
            let slot = TableSlot::new(
                self.heap_rel.unwrap(),
                heap_pointer,
                self.heap_attr_number.unwrap(),
            );
            (self.distance_fn)(
                slot.get_slice(),
                state.table_slot.as_ref().unwrap().get_slice(),
            )
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
*/

pub struct FullVectorDistanceState<'a> {
    table_slot: Option<TableSlot>,
    readable_node: Option<ReadableNode<'a>>,
}

impl FullVectorDistanceState<'_> {
    pub fn with_table_slot(slot: TableSlot) -> Self {
        Self {
            table_slot: Some(slot),
            readable_node: None,
        }
    }

    /*pub fn with_readable_node(rn: ReadableNode) -> Self {
        Self {
            table_slot: None,
            readable_node: Some(rn),
        }
    }*/

    pub fn get_table_slot(&self) -> &TableSlot {
        self.table_slot.as_ref().unwrap()
    }
}

pub enum SearchDistanceMeasure {
    Invalid,
    Full(fn(&[f32], &[f32]) -> f32),
    Pq(super::pq::PqDistanceTable),
    Bq(super::bq::BqDistanceTable),
}
/*
pub struct DistanceMeasure {
    pq_distance_table: Option<super::pq::PqDistanceTable>,
    bq_distance_table: Option<super::bq::BqDistanceTable>,
}

impl DistanceMeasure {
    pub fn new(quantizer: &Quantizer, query: &[f32], calc_distance_with_quantizer: bool) -> Self {
        if !calc_distance_with_quantizer {
            return Self {
                pq_distance_table: None,
                bq_distance_table: None,
            };
        }
        match quantizer {
            Quantizer::None => Self {
                pq_distance_table: None,
                bq_distance_table: None,
            },
            Quantizer::PQ(pq) => {
                let dc = pq.get_distance_table(query, default_distance);
                Self {
                    pq_distance_table: Some(dc),
                    bq_distance_table: None,
                }
            }
            Quantizer::BQ(bq) => {
                let dc = bq.get_distance_table(query, default_distance);
                Self {
                    pq_distance_table: None,
                    bq_distance_table: Some(dc),
                }
            }
        }
    }

    pub fn get_quantized_distance(&self, vec: &[u8]) -> f32 {
        if self.pq_distance_table.is_some() {
            let dc = self.pq_distance_table.as_ref().unwrap();
            let distance = dc.distance(vec);
            distance
        } else {
            let dc = self.bq_distance_table.as_ref().unwrap();
            let distance = dc.distance(vec);
            distance
        }
    }

    pub fn get_full_vector_distance(&self, vec: &[f32], query: &[f32]) -> f32 {
        assert!(self.pq_distance_table.is_none());
        default_distance(vec, query)
    }
}
*/

pub enum LsrPrivateData {
    None,
    /* neighbors, heap_pointer */
    Node(Vec<IndexPointer>, HeapPointer),
}

pub struct ListSearchNeighbor {
    pub index_pointer: IndexPointer,
    distance: f32,
    visited: bool,
    private_data: LsrPrivateData,
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
    pub fn new(index_pointer: IndexPointer, distance: f32, private_data: LsrPrivateData) -> Self {
        Self {
            index_pointer,
            private_data,
            distance,
            visited: false,
        }
    }
}

pub struct ListSearchResult {
    pub candidate_storage: Vec<ListSearchNeighbor>, //plain storage
    best_candidate: Vec<usize>,                     //pos in candidate storage, sorted by distance
    inserted: HashSet<ItemPointer>,
    max_history_size: Option<usize>,
    pub sdm: SearchDistanceMeasure,
    pub stats: GreedySearchStats,
}

impl ListSearchResult {
    fn empty() -> Self {
        Self {
            candidate_storage: vec![],
            best_candidate: vec![],
            inserted: HashSet::new(),
            max_history_size: None,
            sdm: SearchDistanceMeasure::Invalid,
            stats: GreedySearchStats::new(),
        }
    }

    fn new(
        index: &PgRelation,
        max_history_size: Option<usize>,
        _graph: &Graph,
        init_ids: Vec<ItemPointer>,
        query: &[f32],
        sdm: SearchDistanceMeasure,
        search_list_size: usize,
        meta_page: &MetaPage,
        quantizer: &Quantizer,
    ) -> Self {
        let neigbors = meta_page.get_num_neighbors() as usize;
        let mut res = Self {
            candidate_storage: Vec::with_capacity(search_list_size * neigbors),
            best_candidate: Vec::with_capacity(search_list_size * neigbors),
            inserted: HashSet::with_capacity(search_list_size * neigbors),
            max_history_size,
            stats: GreedySearchStats::new(),
            sdm: sdm,
        };
        res.stats.calls += 1;
        for index_pointer in init_ids {
            let lsn = quantizer.get_lsn(&mut res, index, index_pointer, query);
            res.insert_neighbor(lsn);
        }
        res
    }

    pub fn prepare_insert(&mut self, ip: ItemPointer) -> bool {
        return self.inserted.insert(ip);
    }

    /// Internal function
    pub fn insert_neighbor(&mut self, n: ListSearchNeighbor) {
        if let Some(max_size) = self.max_history_size {
            if self.best_candidate.len() >= max_size {
                let last = self.best_candidate.last().unwrap();
                if n >= self.candidate_storage[*last] {
                    //n is too far in the list to be the best candidate.
                    return;
                }
                self.best_candidate.pop();
            }
        }
        //insert while preserving sort order.
        let idx = self
            .best_candidate
            .partition_point(|x| self.candidate_storage[*x] < n);
        self.candidate_storage.push(n);
        let pos = self.candidate_storage.len() - 1;
        self.best_candidate.insert(idx, pos)
    }

    fn visit_closest(&mut self, pos_limit: usize) -> Option<usize> {
        //OPT: should we optimize this not to do a linear search each time?
        let neighbor_position = self
            .best_candidate
            .iter()
            .position(|n| !self.candidate_storage[*n].visited);
        match neighbor_position {
            Some(pos) => {
                if pos > pos_limit {
                    return None;
                }
                let n = &mut self.candidate_storage[self.best_candidate[pos]];
                n.visited = true;
                Some(self.best_candidate[pos])
            }
            None => None,
        }
    }

    //removes and returns the first element. Given that the element remains in self.inserted, that means the element will never again be insereted
    //into the best_candidate list, so it will never again be returned.
    pub fn consume(
        &mut self,
        index: &PgRelation,
        quantizer: &Quantizer,
    ) -> Option<(HeapPointer, IndexPointer)> {
        if self.best_candidate.is_empty() {
            return None;
        }
        let idx = self.best_candidate.remove(0);
        //let f = &self.candidate_storage[self.best_candidate.remove(0)];
        let res = quantizer.return_lsn(index, self, idx);
        return Some(res);
    }
}

pub trait NodeNeighbor {
    fn get_index_pointer_to_neighbors(&self) -> Vec<IndexPointer>;
}

pub enum GraphNeighborStore {
    Builder(BuilderGraph),
    Disk(DiskIndexGraph),
}

impl GraphNeighborStore {
    pub fn get_neighbors_with_distances(
        &self,
        index: &PgRelation,
        neighbors_of: ItemPointer,
        quantizer: &Quantizer,
        result: &mut Vec<NeighborWithDistance>,
    ) -> bool {
        match self {
            GraphNeighborStore::Builder(b) => {
                b.get_neighbors_with_distances(index, neighbors_of, quantizer, result)
            }
            GraphNeighborStore::Disk(d) => {
                d.get_neighbors_with_distances(index, neighbors_of, quantizer, result)
            }
        }
    }

    pub fn set_neighbors(
        &mut self,
        index: &PgRelation,
        neighbors_of: ItemPointer,
        new_neighbors: Vec<NeighborWithDistance>,
    ) {
        match self {
            GraphNeighborStore::Builder(b) => b.set_neighbors(index, neighbors_of, new_neighbors),
            GraphNeighborStore::Disk(d) => d.set_neighbors(index, neighbors_of, new_neighbors),
        }
    }
}

pub struct Graph<'a> {
    neighbor_store: GraphNeighborStore,
    meta_page: &'a mut MetaPage,
}

impl<'a> Graph<'a> {
    pub fn new(neighbor_store: GraphNeighborStore, meta_page: &'a mut MetaPage) -> Self {
        Self {
            neighbor_store,
            meta_page,
        }
    }

    pub fn get_neighbor_store(&self) -> &GraphNeighborStore {
        &self.neighbor_store
    }

    fn get_init_ids(&self) -> Option<Vec<ItemPointer>> {
        self.meta_page.get_init_ids()
    }

    /*     fn get_neighbors<N: NodeNeighbor>(
        &self,
        node: &N,
        neighbors_of: ItemPointer,
    ) -> Vec<IndexPointer>;
    fn get_neighbors_with_distances(
        &self,
        index: &PgRelation,
        neighbors_of: ItemPointer,
        quantizer: &Quantizer,
        result: &mut Vec<NeighborWithDistance>,
    ) -> bool;
    */

    fn is_empty(&self) -> bool {
        match &self.neighbor_store {
            GraphNeighborStore::Builder(b) => b.is_empty(),
            GraphNeighborStore::Disk(d) => d.is_empty(&self.meta_page),
        }
    }

    fn set_neighbors(
        &mut self,
        index: &PgRelation,
        neighbors_of: ItemPointer,
        new_neighbors: Vec<NeighborWithDistance>,
    ) {
        //todo find a better place for this?
        if self.meta_page.get_init_ids().is_none() {
            //TODO probably better set off of centeroids
            MetaPage::update_init_ids(index, vec![neighbors_of]);
            *self.meta_page = MetaPage::read(index);
        }
        self.neighbor_store
            .set_neighbors(index, neighbors_of, new_neighbors);
    }

    pub fn get_meta_page(&self) -> &MetaPage {
        &self.meta_page
    }

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
    fn greedy_search_for_build(
        &self,
        index: &PgRelation,
        query: &[f32],
        meta_page: &MetaPage,
        quantizer: &Quantizer,
    ) -> (ListSearchResult, HashSet<NeighborWithDistance>) {
        let init_ids = self.get_init_ids();
        if let None = init_ids {
            //no nodes in the graph
            return (ListSearchResult::empty(), HashSet::with_capacity(0));
        }
        let dm = quantizer.get_search_distance_measure(query, default_distance, false);
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
            quantizer,
        );
        let mut visited_nodes = HashSet::with_capacity(search_list_size);
        self.greedy_search_iterate(
            &mut l,
            index,
            query,
            search_list_size,
            Some(&mut visited_nodes),
            quantizer,
        );
        return (l, visited_nodes);
    }

    /// Returns a ListSearchResult initialized for streaming. The output should be used with greedy_search_iterate to obtain
    /// the next elements.
    pub fn greedy_search_streaming_init(
        &self,
        index: &PgRelation,
        query: &[f32],
        search_list_size: usize,
        quantizer: &Quantizer,
    ) -> ListSearchResult {
        let init_ids = self.get_init_ids();
        if let None = init_ids {
            //no nodes in the graph
            return ListSearchResult::empty();
        }
        let dm = quantizer.get_search_distance_measure(query, default_distance, true);

        ListSearchResult::new(
            index,
            None,
            self,
            init_ids.unwrap(),
            query,
            dm,
            search_list_size,
            &self.meta_page,
            quantizer,
        )
    }

    /// Advance the state of the lsr until the closest `visit_n_closest` elements have been visited.
    pub fn greedy_search_iterate(
        &self,
        lsr: &mut ListSearchResult,
        index: &PgRelation,
        query: &[f32],
        visit_n_closest: usize,
        mut visited_nodes: Option<&mut HashSet<NeighborWithDistance>>,
        quantizer: &Quantizer,
    ) {
        while let Some(list_search_entry_idx) = lsr.visit_closest(visit_n_closest) {
            match visited_nodes {
                None => {}
                Some(ref mut visited_nodes) => {
                    let list_search_entry = &lsr.candidate_storage[list_search_entry_idx];
                    visited_nodes.insert(NeighborWithDistance::new(
                        list_search_entry.index_pointer,
                        list_search_entry.distance,
                    ));
                }
            }
            quantizer.visit_lsn(
                index,
                lsr,
                list_search_entry_idx,
                query,
                &self.neighbor_store,
            );
        }
    }

    /// Prune neigbors by prefering neighbors closer to the point in question
    /// than to other neighbors of the point.
    ///
    /// TODO: this is the ann-disk implementation. There may be better implementations
    /// if we save the factors or the distances and add incrementally. Not sure.
    pub fn prune_neighbors(
        &self,
        index: &PgRelation,
        index_pointer: ItemPointer,
        new_neigbors: Vec<NeighborWithDistance>,
        quantizer: &Quantizer,
    ) -> (Vec<NeighborWithDistance>, PruneNeighborStats) {
        let mut stats = PruneNeighborStats::new();
        stats.calls += 1;
        //TODO make configurable?
        let max_alpha = self.get_meta_page().get_max_alpha();
        //get a unique candidate pool
        let mut candidates = Vec::<NeighborWithDistance>::with_capacity(
            (self.get_meta_page().get_num_neighbors() as usize) + new_neigbors.len(),
        );
        self.neighbor_store.get_neighbors_with_distances(
            index,
            index_pointer,
            quantizer,
            &mut candidates,
        );

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
            self.get_meta_page().get_max_neighbors_during_build(),
        );

        let mut max_factors: Vec<f64> = vec![0.0; candidates.len()];

        let mut alpha = 1.0;
        //first we add nodes that "pass" a small alpha. Then, if there
        //is still room we loop again with a larger alpha.
        while alpha <= max_alpha && results.len() < self.get_meta_page().get_num_neighbors() as _ {
            for (i, neighbor) in candidates.iter().enumerate() {
                if results.len() >= self.get_meta_page().get_num_neighbors() as _ {
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

                let dist_state = unsafe {
                    quantizer.get_full_vector_distance_state(
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
                    let mut distance_between_candidate_and_existing_neighbor = unsafe {
                        quantizer.get_distance_pair_for_full_vectors_from_state(
                            &dist_state,
                            index,
                            candidate_neighbor.get_index_pointer_to_neighbor(),
                        )
                    };
                    stats.node_reads += 2;
                    stats.distance_comparisons += 1;
                    let mut distance_between_candidate_and_point =
                        candidate_neighbor.get_distance();

                    //We need both values to be positive.
                    //Otherwise, the case where distance_between_candidate_and_point > 0 and distance_between_candidate_and_existing_neighbor < 0 is totally wrong.
                    //If we implement inner product distance we'll have to figure something else out.
                    if distance_between_candidate_and_point < 0.0
                        && distance_between_candidate_and_point >= 0.0 - f32::EPSILON
                    {
                        distance_between_candidate_and_point = 0.0;
                    }

                    if distance_between_candidate_and_existing_neighbor < 0.0
                        && distance_between_candidate_and_existing_neighbor >= 0.0 - f32::EPSILON
                    {
                        distance_between_candidate_and_existing_neighbor = 0.0;
                    }

                    debug_assert!(distance_between_candidate_and_point >= 0.0);
                    debug_assert!(distance_between_candidate_and_existing_neighbor >= 0.0);

                    //factor is high if the candidate is closer to an existing neighbor than the point it's being considered for
                    let factor =
                        if distance_between_candidate_and_existing_neighbor < 0.0 + f32::EPSILON {
                            if distance_between_candidate_and_point < 0.0 + f32::EPSILON {
                                1.0
                            } else {
                                f64::MAX
                            }
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

    pub fn insert(
        &mut self,
        index: &PgRelation,
        index_pointer: IndexPointer,
        vec: &[f32],
        quantizer: &Quantizer,
    ) -> InsertStats {
        let mut prune_neighbor_stats: PruneNeighborStats = PruneNeighborStats::new();
        let mut greedy_search_stats = GreedySearchStats::new();
        let meta_page = self.get_meta_page();

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
        let (l, v) = self.greedy_search_for_build(index, vec, meta_page, quantizer);
        greedy_search_stats.combine(l.stats);
        let (neighbor_list, forward_stats) =
            self.prune_neighbors(index, index_pointer, v.into_iter().collect(), quantizer);
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
                quantizer,
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
        quantizer: &Quantizer,
    ) -> (bool, PruneNeighborStats) {
        let mut current_links = Vec::<NeighborWithDistance>::new();
        self.neighbor_store.get_neighbors_with_distances(
            index,
            from,
            quantizer,
            &mut current_links,
        );

        if current_links.len() < current_links.capacity() as _ {
            current_links.push(NeighborWithDistance::new(to, distance));
            self.set_neighbors(index, from, current_links);
            (false, PruneNeighborStats::new())
        } else {
            //info!("sizes {} {} {}", current_links.len() + 1, current_links.capacity(), self.meta_page.get_max_neighbors_during_build());
            //Note prune_neighbors will reduce to current_links.len() to num_neighbors while capacity is num_neighbors * 1.3
            //thus we are avoiding prunning every time
            let (new_list, stats) = self.prune_neighbors(
                index,
                from,
                vec![NeighborWithDistance::new(to, distance)],
                quantizer,
            );
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
