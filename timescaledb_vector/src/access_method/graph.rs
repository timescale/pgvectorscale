use std::{cmp::Ordering, collections::HashSet};

use pgrx::PgRelation;

use crate::util::{HeapPointer, IndexPointer, ItemPointer};

use super::{
    build::TsvMetaPage,
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

struct ListSearchNeighbor {
    index_pointer: IndexPointer,
    heap_pointer: HeapPointer,
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
    pub fn new(index_pointer: IndexPointer, heap_pointer: HeapPointer, distance: f32) -> Self {
        Self {
            index_pointer,
            heap_pointer,
            distance: distance,
            visited: false,
        }
    }
}

pub struct ListSearchResult {
    search_list_size: usize,
    best_candidate: Vec<ListSearchNeighbor>, //keep sorted by distanced
    inserted: HashSet<ItemPointer>,
    pub stats: GreedySearchStats,
}

impl ListSearchResult {
    fn empty(search_list_size: usize) -> Self {
        Self {
            search_list_size: search_list_size,
            best_candidate: vec![],
            inserted: HashSet::new(),
            stats: GreedySearchStats::new(),
        }
    }

    fn new<G>(
        index: &PgRelation,
        search_list_size: usize,
        graph: &G,
        init_ids: Vec<ItemPointer>,
        query: &[f32],
    ) -> Self
    where
        G: Graph + ?Sized,
    {
        let mut res = Self {
            search_list_size: search_list_size,
            best_candidate: Vec::with_capacity(search_list_size),
            inserted: HashSet::with_capacity(search_list_size),
            stats: GreedySearchStats::new(),
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

        let data_node = graph.read(index, index_pointer);
        self.stats.node_reads += 1;
        let node = data_node.get_archived_node();
        let vec = node.vector.as_slice();
        let distance = distance(vec, query);
        self.stats.distance_comparisons += 1;

        let neighbor = ListSearchNeighbor::new(
            index_pointer,
            node.heap_item_pointer.deserialize_item_pointer(),
            distance,
        );
        self._insert_neighbor(neighbor);
    }

    /// Internal function
    fn _insert_neighbor(&mut self, n: ListSearchNeighbor) {
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
                Some((n.index_pointer, n.distance))
            }
            None => None,
        }
    }

    pub fn get_closets_index_pointer(&self, index: usize) -> Option<ItemPointer> {
        self.best_candidate.get(index).map(|n| n.index_pointer)
    }

    fn get_k_index_pointers(&self, k: usize) -> Vec<ItemPointer> {
        let mut k_closets = Vec::<ItemPointer>::with_capacity(k);
        for i in 0..k {
            let item = self.get_closets_index_pointer(i);
            match item {
                Some(pointer) => k_closets.push(pointer),
                None => break,
            }
        }
        k_closets
    }

    pub fn get_closest_heap_pointer(&self, i: usize) -> Option<HeapPointer> {
        self.best_candidate.get(i).map(|f| f.heap_pointer)
    }
}

pub trait Graph {
    fn read(&self, index: &PgRelation, index_pointer: ItemPointer) -> ReadableNode;
    fn get_init_ids(&mut self) -> Option<Vec<ItemPointer>>;
    fn get_neighbors(
        &self,
        index: &PgRelation,
        neighbors_of: ItemPointer,
        result: &mut Vec<NeighborWithDistance>,
    ) -> bool;
    fn get_meta_page(&self, index: &PgRelation) -> &TsvMetaPage;

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
    /// Note: there may be a streaming implementation of this possible if we don't limit the size
    /// of the ListSearchResult, and use  ListSearchResult as the state. Then you don't output item
    /// i until you've visited all item up until i+window (e.g. i+10).
    fn greedy_search(
        &mut self,
        index: &PgRelation,
        query: &[f32],
        search_list_size: usize,
    ) -> (ListSearchResult, Option<HashSet<NeighborWithDistance>>)
    where
        Self: Graph,
    {
        let init_ids = self.get_init_ids();
        if let None = init_ids {
            //no nodes in the graph
            return (ListSearchResult::empty(search_list_size), None);
        }
        let mut l = ListSearchResult::new(index, search_list_size, self, init_ids.unwrap(), query);

        //OPT: Only build v when needed.
        let mut v: HashSet<_> = HashSet::<NeighborWithDistance>::with_capacity(search_list_size);
        let mut neighbors = Vec::<NeighborWithDistance>::with_capacity(
            self.get_meta_page(index).get_num_neighbors() as _,
        );
        while let Some((index_pointer, distance)) = l.visit_closest() {
            neighbors.clear();
            let neighbors_existed = self.get_neighbors(index, index_pointer, &mut neighbors);
            if !neighbors_existed {
                panic!("Nodes in the list search results that aren't in the builder");
            }

            for neighbor_index_pointer in &neighbors {
                l.insert(
                    index,
                    self,
                    neighbor_index_pointer.get_index_pointer_to_neigbor(),
                    query,
                )
            }
            v.insert(NeighborWithDistance::new(index_pointer, distance));
        }

        (l, Some(v))
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
        self.get_neighbors(index, index_pointer, &mut candidates);

        let mut hash: HashSet<ItemPointer> = candidates
            .iter()
            .map(|c| c.get_index_pointer_to_neigbor())
            .collect();
        for n in new_neigbors {
            if hash.insert(n.get_index_pointer_to_neigbor()) {
                candidates.push(n);
            }
        }
        //remove myself
        if !hash.insert(index_pointer) {
            //prevent self-loops
            let index = candidates
                .iter()
                .position(|x| x.get_index_pointer_to_neigbor() == index_pointer)
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

                //TODO make lazy
                let data_node = self.read(index, existing_neighbor.get_index_pointer_to_neigbor());
                stats.node_reads += 1;
                let existing_neighbor_node = data_node.get_archived_node();
                let existing_neighbor_vec = existing_neighbor_node.vector.as_slice();

                //go thru the other candidates (tail of the list)
                for (j, candidate_neighbor) in candidates.iter().enumerate().skip(i + 1) {
                    //has it been completely excluded?
                    if max_factors[j] > max_alpha {
                        continue;
                    }

                    let data_node =
                        self.read(index, candidate_neighbor.get_index_pointer_to_neigbor());
                    stats.node_reads += 1;
                    let candidate_node = data_node.get_archived_node();
                    let candidate_vec = candidate_node.vector.as_slice();
                    let distance_between_candidate_and_existing_neighbor =
                        distance(existing_neighbor_vec, candidate_vec);
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
}

impl GreedySearchStats {
    pub fn new() -> Self {
        GreedySearchStats {
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
