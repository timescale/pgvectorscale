use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::{cmp::Ordering, collections::HashSet};

use pgrx::PgRelation;

use crate::access_method::storage::NodeDistanceMeasure;

use crate::util::{HeapPointer, IndexPointer, ItemPointer};

use super::graph_neighbor_store::GraphNeighborStore;

use super::neighbor_with_distance::{Distance, DistanceWithTieBreak};
use super::pg_vector::PgVector;
use super::stats::{GreedySearchStats, InsertStats, PruneNeighborStats, StatsNodeVisit};
use super::storage::Storage;
use super::{meta_page::MetaPage, neighbor_with_distance::NeighborWithDistance};

pub struct ListSearchNeighbor<PD> {
    pub index_pointer: IndexPointer,
    distance_with_tie_break: DistanceWithTieBreak,
    private_data: PD,
}

impl<PD> PartialOrd for ListSearchNeighbor<PD> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<PD> PartialEq for ListSearchNeighbor<PD> {
    fn eq(&self, other: &Self) -> bool {
        self.index_pointer == other.index_pointer
    }
}

impl<PD> Eq for ListSearchNeighbor<PD> {}

impl<PD> Ord for ListSearchNeighbor<PD> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance_with_tie_break
            .cmp(&other.distance_with_tie_break)
    }
}

impl<PD> ListSearchNeighbor<PD> {
    pub fn new(
        index_pointer: IndexPointer,
        distance_with_tie_break: DistanceWithTieBreak,
        private_data: PD,
    ) -> Self {
        Self {
            index_pointer,
            private_data,
            distance_with_tie_break,
        }
    }

    pub fn get_private_data(&self) -> &PD {
        &self.private_data
    }
}

pub struct ListSearchResult<QDM, PD> {
    candidates: BinaryHeap<Reverse<ListSearchNeighbor<PD>>>,
    visited: Vec<ListSearchNeighbor<PD>>,
    inserted: HashSet<ItemPointer>,
    pub sdm: Option<QDM>,
    tie_break_item_pointer: Option<ItemPointer>, /* This records the item pointer of the query. It's used for tie-breaking when the distance = 0 */
    pub stats: GreedySearchStats,
}

impl<QDM, PD> ListSearchResult<QDM, PD> {
    fn empty() -> Self {
        Self {
            candidates: BinaryHeap::new(),
            visited: vec![],
            inserted: HashSet::new(),
            sdm: None,
            tie_break_item_pointer: None,
            stats: GreedySearchStats::new(),
        }
    }

    fn new<S: Storage<QueryDistanceMeasure = QDM, LSNPrivateData = PD>>(
        init_ids: Vec<ItemPointer>,
        sdm: S::QueryDistanceMeasure,
        tie_break_item_pointer: Option<ItemPointer>,
        search_list_size: usize,
        meta_page: &MetaPage,
        gns: &GraphNeighborStore,
        storage: &S,
    ) -> Self {
        let neigbors = meta_page.get_num_neighbors() as usize;
        let mut res = Self {
            tie_break_item_pointer,
            candidates: BinaryHeap::with_capacity(search_list_size * neigbors),
            visited: Vec::with_capacity(search_list_size * 2),
            //candidate_storage: Vec::with_capacity(search_list_size * neigbors),
            //best_candidate: Vec::with_capacity(search_list_size * neigbors),
            inserted: HashSet::with_capacity(search_list_size * neigbors),
            stats: GreedySearchStats::new(),
            sdm: Some(sdm),
        };
        res.stats.record_call();
        for index_pointer in init_ids {
            let lsn = storage.create_lsn_for_init_id(&mut res, index_pointer, gns);
            res.insert_neighbor(lsn);
        }
        res
    }

    pub fn prepare_insert(&mut self, ip: ItemPointer) -> bool {
        self.inserted.insert(ip)
    }

    pub fn create_distance_with_tie_break(
        &self,
        d: Distance,
        ip: ItemPointer,
    ) -> DistanceWithTieBreak {
        match self.tie_break_item_pointer {
            None => DistanceWithTieBreak::with_query(d, ip),
            Some(tie_break_item_pointer) => {
                DistanceWithTieBreak::new(d, ip, tie_break_item_pointer)
            }
        }
    }

    /// To be called by the Storage Providers only
    pub fn insert_neighbor(&mut self, n: ListSearchNeighbor<PD>) {
        self.stats.record_candidate();
        self.candidates.push(Reverse(n));
    }

    pub fn get_lsn_by_idx(&self, idx: usize) -> &ListSearchNeighbor<PD> {
        &self.visited[idx]
    }

    fn visit_closest(&mut self, pos_limit: usize) -> Option<usize> {
        if self.candidates.is_empty() {
            return None;
        }

        if self.visited.len() > pos_limit {
            let node_at_pos = &self.visited[pos_limit - 1];
            let head = self.candidates.peek().unwrap();
            if head.0 >= *node_at_pos {
                return None;
            }
        }

        let head = self.candidates.pop().unwrap();
        let idx = self.visited.partition_point(|x| *x < head.0);
        self.visited.insert(idx, head.0);
        Some(idx)
    }

    //removes and returns the first element. Given that the element remains in self.inserted, that means the element will never again be insereted
    //into the best_candidate list, so it will never again be returned.
    pub fn consume<S: Storage<QueryDistanceMeasure = QDM, LSNPrivateData = PD>>(
        &mut self,
        storage: &S,
    ) -> Option<(HeapPointer, IndexPointer)> {
        if self.visited.is_empty() {
            return None;
        }
        let lsn = self.visited.remove(0);
        let heap_pointer = storage.return_lsn(&lsn, &mut self.stats);
        Some((heap_pointer, lsn.index_pointer))
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

    fn add_neighbors<S: Storage>(
        &mut self,
        storage: &S,
        neighbors_of: ItemPointer,
        additional_neighbors: Vec<NeighborWithDistance>,
        stats: &mut PruneNeighborStats,
    ) -> (bool, Vec<NeighborWithDistance>) {
        let mut candidates = Vec::<NeighborWithDistance>::with_capacity(
            self.neighbor_store.max_neighbors(self.get_meta_page()) + additional_neighbors.len(),
        );
        self.neighbor_store
            .get_neighbors_with_full_vector_distances(
                neighbors_of,
                storage,
                &mut candidates,
                stats,
            );

        let mut hash: HashSet<ItemPointer> = candidates
            .iter()
            .map(|c| c.get_index_pointer_to_neighbor())
            .collect();
        for n in additional_neighbors {
            if hash.insert(n.get_index_pointer_to_neighbor()) {
                candidates.push(n);
            }
        }
        //remove myself
        if !hash.insert(neighbors_of) {
            //prevent self-loops
            let index = candidates
                .iter()
                .position(|x| x.get_index_pointer_to_neighbor() == neighbors_of)
                .unwrap();
            candidates.remove(index);
        }

        let (pruned, new_neighbors) =
            if candidates.len() > self.neighbor_store.max_neighbors(self.get_meta_page()) {
                let new_list = self.prune_neighbors(neighbors_of, candidates, storage, stats);
                (true, new_list)
            } else {
                (false, candidates)
            };

        //OPT: remove clone
        self.neighbor_store.set_neighbors(
            storage,
            self.meta_page,
            neighbors_of,
            new_neighbors.clone(),
            stats,
        );
        (pruned, new_neighbors)
    }

    pub fn get_meta_page(&self) -> &MetaPage {
        self.meta_page
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
    #[allow(clippy::mutable_key_type)]
    fn greedy_search_for_build<S: Storage>(
        &self,
        index_pointer: IndexPointer,
        query: PgVector,
        meta_page: &MetaPage,
        storage: &S,
        stats: &mut GreedySearchStats,
    ) -> HashSet<NeighborWithDistance> {
        let init_ids = self.get_init_ids();
        if init_ids.is_none() {
            //no nodes in the graph
            return HashSet::with_capacity(0);
        }
        let dm = storage.get_query_distance_measure(query);
        let search_list_size = meta_page.get_search_list_size_for_build() as usize;

        let mut l = ListSearchResult::new(
            init_ids.unwrap(),
            dm,
            Some(index_pointer),
            search_list_size,
            meta_page,
            self.get_neighbor_store(),
            storage,
        );
        let mut visited_nodes = HashSet::with_capacity(search_list_size);
        self.greedy_search_iterate(&mut l, search_list_size, Some(&mut visited_nodes), storage);
        stats.combine(&l.stats);
        visited_nodes
    }

    /// Returns a ListSearchResult initialized for streaming. The output should be used with greedy_search_iterate to obtain
    /// the next elements.
    pub fn greedy_search_streaming_init<S: Storage>(
        &self,
        query: PgVector,
        search_list_size: usize,
        storage: &S,
    ) -> ListSearchResult<S::QueryDistanceMeasure, S::LSNPrivateData> {
        let init_ids = self.get_init_ids();
        if init_ids.is_none() {
            //no nodes in the graph
            return ListSearchResult::empty();
        }
        let dm = storage.get_query_distance_measure(query);

        ListSearchResult::new(
            init_ids.unwrap(),
            dm,
            None,
            search_list_size,
            self.meta_page,
            self.get_neighbor_store(),
            storage,
        )
    }

    /// Advance the state of the lsr until the closest `visit_n_closest` elements have been visited.
    pub fn greedy_search_iterate<S: Storage>(
        &self,
        lsr: &mut ListSearchResult<S::QueryDistanceMeasure, S::LSNPrivateData>,
        visit_n_closest: usize,
        mut visited_nodes: Option<&mut HashSet<NeighborWithDistance>>,
        storage: &S,
    ) {
        while let Some(list_search_entry_idx) = lsr.visit_closest(visit_n_closest) {
            match visited_nodes {
                None => {}
                Some(ref mut visited_nodes) => {
                    let list_search_entry = &lsr.visited[list_search_entry_idx];
                    visited_nodes.insert(NeighborWithDistance::new(
                        list_search_entry.index_pointer,
                        list_search_entry.distance_with_tie_break.clone(),
                    ));
                }
            }
            lsr.stats.record_visit();
            storage.visit_lsn(lsr, list_search_entry_idx, &self.neighbor_store);
        }
    }

    /// Prune neigbors by prefering neighbors closer to the point in question
    /// than to other neighbors of the point.
    ///
    /// TODO: this is the ann-disk implementation. There may be better implementations
    /// if we save the factors or the distances and add incrementally. Not sure.
    pub fn prune_neighbors<S: Storage>(
        &self,
        _neighbors_of: ItemPointer,
        mut candidates: Vec<NeighborWithDistance>,
        storage: &S,
        stats: &mut PruneNeighborStats,
    ) -> Vec<NeighborWithDistance> {
        stats.calls += 1;
        //TODO make configurable?
        let max_alpha = self.get_meta_page().get_max_alpha();

        stats.num_neighbors_before_prune += candidates.len();
        //TODO remove deleted nodes

        //TODO diskann has something called max_occlusion_size/max_candidate_size(default:750). Do we need to implement?

        //sort by distance
        candidates.sort();
        let mut results = Vec::<NeighborWithDistance>::with_capacity(
            self.get_meta_page().get_num_neighbors() as _,
        );

        let mut max_factors: Vec<f64> = vec![0.0; candidates.len()];

        let mut alpha = 1.0;
        //first we add nodes that "pass" a small alpha. Then, if there
        //is still room we loop again with a larger alpha.
        while alpha <= max_alpha && results.len() < self.get_meta_page().get_num_neighbors() as _ {
            for (i, neighbor) in candidates.iter().enumerate() {
                if results.len() >= self.get_meta_page().get_num_neighbors() as _ {
                    return results;
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
                    storage.get_node_distance_measure(
                        existing_neighbor.get_index_pointer_to_neighbor(),
                        stats,
                    )
                };

                //go thru the other candidates (tail of the list)
                for (j, candidate_neighbor) in candidates.iter().enumerate().skip(i + 1) {
                    //has it been completely excluded?
                    if max_factors[j] > max_alpha {
                        continue;
                    }

                    let raw_distance_between_candidate_and_existing_neighbor = unsafe {
                        dist_state
                            .get_distance(candidate_neighbor.get_index_pointer_to_neighbor(), stats)
                    };

                    let distance_between_candidate_and_existing_neighbor =
                        DistanceWithTieBreak::new(
                            raw_distance_between_candidate_and_existing_neighbor,
                            candidate_neighbor.get_index_pointer_to_neighbor(),
                            existing_neighbor.get_index_pointer_to_neighbor(),
                        );

                    let distance_between_candidate_and_point =
                        candidate_neighbor.get_distance_with_tie_break();

                    //factor is high if the candidate is closer to an existing neighbor than the point it's being considered for
                    let factor = distance_between_candidate_and_point
                        .get_factor(&distance_between_candidate_and_existing_neighbor);

                    max_factors[j] = max_factors[j].max(factor)
                }
            }
            alpha *= 1.2
        }
        stats.num_neighbors_after_prune += results.len();
        results
    }

    pub fn insert<S: Storage>(
        &mut self,
        index: &PgRelation,
        index_pointer: IndexPointer,
        vec: PgVector,
        storage: &S,
        stats: &mut InsertStats,
    ) {
        if self.meta_page.get_init_ids().is_none() {
            //TODO probably better set off of centeroids
            MetaPage::update_init_ids(index, vec![index_pointer], stats);
            *self.meta_page = MetaPage::fetch(index);

            self.neighbor_store.set_neighbors(
                storage,
                self.meta_page,
                index_pointer,
                Vec::<NeighborWithDistance>::with_capacity(
                    self.neighbor_store.max_neighbors(self.meta_page) as _,
                ),
                stats,
            );
        }

        let meta_page = self.get_meta_page();

        //TODO: make configurable?
        #[allow(clippy::mutable_key_type)]
        let v = self.greedy_search_for_build(
            index_pointer,
            vec,
            meta_page,
            storage,
            &mut stats.greedy_search_stats,
        );

        let (_, neighbor_list) = self.add_neighbors(
            storage,
            index_pointer,
            v.into_iter().collect(),
            &mut stats.prune_neighbor_stats,
        );

        //update back pointers
        let mut cnt_contains = 0;
        let neighbor_list_len = neighbor_list.len();
        for neighbor in neighbor_list {
            let neighbor_contains_new_point = self.update_back_pointer(
                neighbor.get_index_pointer_to_neighbor(),
                index_pointer,
                neighbor.get_distance_with_tie_break(),
                storage,
                &mut stats.prune_neighbor_stats,
            );
            if neighbor_contains_new_point {
                cnt_contains += 1;
            }
        }
        if neighbor_list_len > 0 && cnt_contains == 0 {
            // in tests this should be a hard error
            debug_assert!(
                false,
                "Inserted {:?} but it became an orphan",
                index_pointer
            );
            // in production this is a warning
            pgrx::warning!("Inserted {:?} but it became an orphan", index_pointer);
        }
    }

    fn update_back_pointer<S: Storage>(
        &mut self,
        from: IndexPointer,
        to: IndexPointer,
        distance_with_tie_break: &DistanceWithTieBreak,
        storage: &S,
        prune_stats: &mut PruneNeighborStats,
    ) -> bool {
        let new = vec![NeighborWithDistance::new(
            to,
            distance_with_tie_break.clone(),
        )];
        let (_pruned, n) = self.add_neighbors(storage, from, new.clone(), prune_stats);
        n.contains(&new[0])
    }
}
