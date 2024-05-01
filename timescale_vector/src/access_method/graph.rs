use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::{cmp::Ordering, collections::HashSet};

use pgrx::PgRelation;

use crate::access_method::storage::NodeDistanceMeasure;

use crate::util::{HeapPointer, IndexPointer, ItemPointer};

use super::graph_neighbor_store::GraphNeighborStore;

use super::pg_vector::PgVector;
use super::stats::{GreedySearchStats, InsertStats, PruneNeighborStats, StatsNodeVisit};
use super::storage::Storage;
use super::{meta_page::MetaPage, neighbor_with_distance::NeighborWithDistance};

pub struct ListSearchNeighbor<PD> {
    pub index_pointer: IndexPointer,
    distance: f32,
    private_data: PD,
}

impl<PD> PartialOrd for ListSearchNeighbor<PD> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.distance.partial_cmp(&other.distance)
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
        self.distance.partial_cmp(&other.distance).unwrap()
    }
}

impl<PD> ListSearchNeighbor<PD> {
    pub fn new(index_pointer: IndexPointer, distance: f32, private_data: PD) -> Self {
        assert!(!distance.is_nan());
        debug_assert!(distance >= 0.0);
        Self {
            index_pointer,
            private_data,
            distance,
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
    pub stats: GreedySearchStats,
}

impl<QDM, PD> ListSearchResult<QDM, PD> {
    fn empty() -> Self {
        Self {
            candidates: BinaryHeap::new(),
            visited: vec![],
            inserted: HashSet::new(),
            sdm: None,
            stats: GreedySearchStats::new(),
        }
    }

    fn new<S: Storage<QueryDistanceMeasure = QDM, LSNPrivateData = PD>>(
        init_ids: Vec<ItemPointer>,
        sdm: S::QueryDistanceMeasure,
        search_list_size: usize,
        meta_page: &MetaPage,
        gns: &GraphNeighborStore,
        storage: &S,
    ) -> Self {
        let neigbors = meta_page.get_num_neighbors() as usize;
        let mut res = Self {
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
        return self.inserted.insert(ip);
    }

    /// Internal function
    pub fn insert_neighbor(&mut self, n: ListSearchNeighbor<PD>) {
        self.stats.record_candidate();
        self.candidates.push(Reverse(n));
    }

    pub fn get_lsn_by_idx(&self, idx: usize) -> &ListSearchNeighbor<PD> {
        &self.visited[idx]
    }

    fn visit_closest(&mut self, pos_limit: usize) -> Option<usize> {
        if self.candidates.len() == 0 {
            return None;
        }

        if self.visited.len() > pos_limit {
            let node_at_pos = &self.visited[pos_limit - 1];
            let head = self.candidates.peek().unwrap();
            if head.0.distance >= node_at_pos.distance {
                return None;
            }
        }

        let head = self.candidates.pop().unwrap();
        let idx = self
            .visited
            .partition_point(|x| x.distance < head.0.distance);
        self.visited.insert(idx, head.0);
        Some(idx)
    }

    //removes and returns the first element. Given that the element remains in self.inserted, that means the element will never again be insereted
    //into the best_candidate list, so it will never again be returned.
    pub fn consume<S: Storage<QueryDistanceMeasure = QDM, LSNPrivateData = PD>>(
        &mut self,
        storage: &S,
    ) -> Option<(HeapPointer, IndexPointer)> {
        if self.visited.len() == 0 {
            return None;
        }
        let lsn = self.visited.remove(0);
        let heap_pointer = storage.return_lsn(&lsn, &mut self.stats);
        return Some((heap_pointer, lsn.index_pointer));
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
            (self.neighbor_store.max_neighbors(self.get_meta_page()) as usize)
                + additional_neighbors.len(),
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
                let new_list = self.prune_neighbors(candidates, storage, stats);
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
    fn greedy_search_for_build<S: Storage>(
        &self,
        query: PgVector,
        meta_page: &MetaPage,
        storage: &S,
        stats: &mut GreedySearchStats,
    ) -> HashSet<NeighborWithDistance> {
        let init_ids = self.get_init_ids();
        if let None = init_ids {
            //no nodes in the graph
            return HashSet::with_capacity(0);
        }
        let dm = storage.get_query_distance_measure(query);
        let search_list_size = meta_page.get_search_list_size_for_build() as usize;

        let mut l = ListSearchResult::new(
            init_ids.unwrap(),
            dm,
            search_list_size,
            meta_page,
            self.get_neighbor_store(),
            storage,
        );
        let mut visited_nodes = HashSet::with_capacity(search_list_size);
        self.greedy_search_iterate(&mut l, search_list_size, Some(&mut visited_nodes), storage);
        stats.combine(&l.stats);
        return visited_nodes;
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
        if let None = init_ids {
            //no nodes in the graph
            return ListSearchResult::empty();
        }
        let dm = storage.get_query_distance_measure(query);

        ListSearchResult::new(
            init_ids.unwrap(),
            dm,
            search_list_size,
            &self.meta_page,
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
                        list_search_entry.distance,
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
        let dimension_epsilon = self.get_meta_page().get_num_dimensions() as f32 * f32::EPSILON;
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

                    let mut distance_between_candidate_and_existing_neighbor = unsafe {
                        dist_state
                            .get_distance(candidate_neighbor.get_index_pointer_to_neighbor(), stats)
                    };
                    let mut distance_between_candidate_and_point =
                        candidate_neighbor.get_distance();

                    //We need both values to be positive.
                    //Otherwise, the case where distance_between_candidate_and_point > 0 and distance_between_candidate_and_existing_neighbor < 0 is totally wrong.
                    //If we implement inner product distance we'll have to figure something else out.
                    if distance_between_candidate_and_point < 0.0
                        && distance_between_candidate_and_point >= 0.0 - dimension_epsilon
                    {
                        distance_between_candidate_and_point = 0.0;
                    }

                    if distance_between_candidate_and_existing_neighbor < 0.0
                        && distance_between_candidate_and_existing_neighbor
                            >= 0.0 - dimension_epsilon
                    {
                        distance_between_candidate_and_existing_neighbor = 0.0;
                    }

                    debug_assert!(
                        distance_between_candidate_and_point >= 0.0,
                        "distance_between_candidate_and_point is negative: {}, {}",
                        distance_between_candidate_and_point,
                        f32::EPSILON
                    );
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
        let v =
            self.greedy_search_for_build(vec, meta_page, storage, &mut stats.greedy_search_stats);

        let (_, neighbor_list) = self.add_neighbors(
            storage,
            index_pointer,
            v.into_iter().collect(),
            &mut stats.prune_neighbor_stats,
        );

        //update back pointers
        let mut cnt = 0;
        for neighbor in neighbor_list {
            let needed_prune = self.update_back_pointer(
                neighbor.get_index_pointer_to_neighbor(),
                index_pointer,
                neighbor.get_distance(),
                storage,
                &mut stats.prune_neighbor_stats,
            );
            if needed_prune {
                cnt = cnt + 1;
            }
        }
    }

    fn update_back_pointer<S: Storage>(
        &mut self,
        from: IndexPointer,
        to: IndexPointer,
        distance: f32,
        storage: &S,
        prune_stats: &mut PruneNeighborStats,
    ) -> bool {
        let new = vec![NeighborWithDistance::new(to, distance)];
        let (pruned, _) = self.add_neighbors(storage, from, new, prune_stats);
        pruned
    }
}
