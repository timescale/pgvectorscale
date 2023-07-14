use std::{cmp::Ordering, collections::HashSet};

use crate::util::{
    ArchivedItemPointer, HeapPointer, IndexPointer, ItemPointer, ReadableBuffer, WritableBuffer,
};

use super::{
    build::TsvMetaPage,
    model::{NeighborWithDistance, ReadableNode},
};

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
}

impl ListSearchResult {
    fn empty(search_list_size: usize) -> Self {
        Self {
            search_list_size: search_list_size,
            best_candidate: vec![],
            inserted: HashSet::new(),
        }
    }

    fn new<G>(search_list_size: usize, graph: &G, init_ids: Vec<ItemPointer>, query: &[f32]) -> Self
    where
        G: Graph + ?Sized,
    {
        let mut res = Self {
            search_list_size: search_list_size,
            best_candidate: Vec::with_capacity(search_list_size),
            inserted: HashSet::with_capacity(search_list_size),
        };
        for index_pointer in init_ids {
            res.insert(graph, index_pointer, query);
        }
        res
    }

    fn insert<G>(&mut self, graph: &G, index_pointer: ItemPointer, query: &[f32])
    where
        G: Graph + ?Sized,
    {
        //no point reprocessing a point. Distance calcs are expensive.
        if !self.inserted.insert(index_pointer.clone()) {
            return;
        }

        let node = graph.read(index_pointer);
        let vec = node.node.vector.as_slice();
        let distance = distance(vec, query);

        let neighbor = ListSearchNeighbor::new(
            index_pointer,
            node.node.heap_item_pointer.deserialize_item_pointer(),
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
                Some((n.index_pointer.clone(), n.distance))
            }
            None => None,
        }
    }

    pub fn get_closets_index_pointer(&self, index: usize) -> Option<ItemPointer> {
        self.best_candidate.get(index).map(|n| n.index_pointer)
    }

    fn get_k_index_pointers(&self, k: usize) -> Vec<ItemPointer> {
        let mut k_closets = Vec::<ItemPointer>::with_capacity(k);
        for i in (0..k) {
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
    fn read<'b, 'd>(&'b self, index_pointer: ItemPointer) -> ReadableNode<'d>;
    fn get_init_ids(&mut self) -> Option<Vec<ItemPointer>>;
    fn get_neighbors(&self, neighbors_of: ItemPointer) -> Option<Vec<NeighborWithDistance>>;
    fn get_meta_page(&self) -> &TsvMetaPage;

    fn greedy_search(
        &mut self,
        query: &[f32],
        k: usize,
        search_list_size: usize,
    ) -> (ListSearchResult, Option<HashSet<NeighborWithDistance>>)
    where
        Self: Graph,
    {
        assert!(k <= search_list_size);

        let init_ids = self.get_init_ids();
        if let None = init_ids {
            //no nodes in the graph
            return (ListSearchResult::empty(search_list_size), None);
        }
        let mut l = ListSearchResult::new(search_list_size, self, init_ids.unwrap(), query);

        //OPT: Only build v when needed.
        let mut v: HashSet<_> = HashSet::<NeighborWithDistance>::with_capacity(search_list_size);
        while let Some((index_pointer, distance)) = l.visit_closest() {
            let neighbors = self.get_neighbors(index_pointer);
            if let None = neighbors {
                panic!("Nodes in the list search results that aren't in the builder");
            }

            for neighbor_index_pointer in neighbors.unwrap() {
                l.insert(
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
        &mut self,
        index_pointer: ItemPointer,
        new_neigbors: Vec<NeighborWithDistance>,
    ) -> Vec<NeighborWithDistance> {
        //TODO make configurable?
        let max_alpha = 1.2;
        //get a unique candidate pool
        let mut candidates = match self.get_neighbors(index_pointer) {
            Some(v) => v.clone(),
            None => vec![], //new point has no entry in the map yet
        };
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
        if !hash.insert(index_pointer.clone()) {
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
        let mut results =
            Vec::<NeighborWithDistance>::with_capacity(self.get_meta_page().num_neighbors as _);

        let mut max_factors: Vec<f32> = vec![0.0; candidates.len()];

        let mut alpha = 1.0;
        //first we add nodes that "pass" a small alpha. Then, if there
        //is still room we loop again with a larger alpha.
        while alpha <= max_alpha && results.len() < self.get_meta_page().num_neighbors as _ {
            for (i, neighbor) in candidates.iter().enumerate() {
                if results.len() >= self.get_meta_page().num_neighbors as _ {
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
                let existing_neighbor_node =
                    self.read(existing_neighbor.get_index_pointer_to_neigbor());
                let existing_neighbor_vec = existing_neighbor_node.node.vector.as_slice();

                //go thru the other candidates (tail of the list)
                for (j, candidate_neighbor) in candidates.iter().enumerate().skip(i + 1) {
                    //has it been completely excluded?
                    if max_factors[j] > max_alpha {
                        continue;
                    }

                    let candidate_node =
                        self.read(candidate_neighbor.get_index_pointer_to_neigbor());
                    let candidate_vec = candidate_node.node.vector.as_slice();
                    let distance_between_candidate_and_existing_neighbor =
                        distance(existing_neighbor_vec, candidate_vec);
                    let distance_between_candidate_and_point = candidate_neighbor.get_distance();
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
}
