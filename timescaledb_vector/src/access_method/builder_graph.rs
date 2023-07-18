use std::collections::HashMap;
use std::time::Instant;

use pgrx::*;

use crate::util::{IndexPointer, ItemPointer};

use super::build::TsvMetaPage;
use super::graph::{Graph, GreedySearchStats, PruneNeighborStats};
use super::model::{Distance, NeighborWithDistance, Node, ReadableNode};

/// A builderGraph is a graph that keep the neighbors in-memory in the neighbor_map below
/// The idea is that during the index build, you don't want to update the actual Postgres
/// pages every time you change the neighbors. Instead you change the neighbors in memory
/// until the build is done. Afterwards, calling the `write` method, will write out all
/// the neighbors to the right pages.
pub struct BuilderGraph {
    //maps node's pointer to the representation on disk
    neighbor_map: std::collections::HashMap<ItemPointer, Vec<NeighborWithDistance>>,
    first: Option<ItemPointer>,
    meta_page: super::build::TsvMetaPage,
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

impl BuilderGraph {
    pub fn new(meta_page: super::build::TsvMetaPage) -> Self {
        Self {
            neighbor_map: HashMap::with_capacity(200),
            first: None,
            meta_page: meta_page,
        }
    }

    pub fn insert(
        &mut self,
        index: &PgRelation,
        index_pointer: IndexPointer,
        vec: &[f32],
    ) -> InsertStats {
        let mut prune_neighbor_stats: PruneNeighborStats = PruneNeighborStats::new();
        let mut greedy_search_stats = GreedySearchStats::new();

        if self.neighbor_map.len() == 0 {
            self.neighbor_map.insert(
                index_pointer,
                Vec::<NeighborWithDistance>::with_capacity(
                    self.meta_page.get_max_neighbors_during_build() as _,
                ),
            );
            return InsertStats {
                prune_neighbor_stats: prune_neighbor_stats,
                greedy_search_stats: greedy_search_stats,
            };
        }

        //TODO: make configurable?
        let search_list_size = 100;
        let (l, v) = self.greedy_search(index, vec, search_list_size);
        greedy_search_stats.combine(l.stats);
        let (neighbor_list, forward_stats) =
            self.prune_neighbors(index, index_pointer, v.unwrap().into_iter().collect());
        prune_neighbor_stats.combine(forward_stats);

        //set forward pointers
        self.neighbor_map
            .insert(index_pointer, neighbor_list.clone());

        //update back pointers
        let mut cnt = 0;
        for neighbor in neighbor_list {
            let (needed_prune, backpointer_stats) = self.update_back_pointer(
                index,
                neighbor.get_index_pointer_to_neigbor(),
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
            prune_neighbor_stats: prune_neighbor_stats,
            greedy_search_stats: greedy_search_stats,
        };
    }

    fn update_back_pointer(
        &mut self,
        index: &PgRelation,
        from: IndexPointer,
        to: IndexPointer,
        distance: f32,
    ) -> (bool, PruneNeighborStats) {
        let current_links = self.neighbor_map.get_mut(&from).unwrap();
        if current_links.len() < current_links.capacity() as _ {
            current_links.push(NeighborWithDistance::new(to, distance));
            (false, PruneNeighborStats::new())
        } else {
            //info!("sizes {} {} {}", current_links.len() + 1, current_links.capacity(), self.meta_page.get_max_neighbors_during_build());
            //Note prune_neighbors will reduce to current_links.len() to num_neighbors while capacity is num_neighbors * 1.3
            //thus we are avoiding prunning every time
            let (new_list, stats) =
                self.prune_neighbors(index, from, vec![NeighborWithDistance::new(to, distance)]);
            self.neighbor_map.insert(from, new_list);
            (true, stats)
        }
    }

    pub unsafe fn write(&self, index: &PgRelation) -> WriteStats {
        let mut stats = WriteStats::new();
        //TODO: OPT: do this in order of item pointers
        for (index_pointer, neighbors) in &self.neighbor_map {
            stats.num_nodes += 1;
            let prune_neighbors;
            let neighbors = if neighbors.len() > self.meta_page.get_num_neighbors() as _ {
                stats.num_prunes += 1;
                stats.num_neighbors_before_prune += neighbors.len();
                (prune_neighbors, _) = self.prune_neighbors(index, *index_pointer, vec![]);
                stats.num_neighbors_after_prune += prune_neighbors.len();
                &prune_neighbors
            } else {
                neighbors
            };
            stats.num_neighbors += neighbors.len();

            let node = Node::modify(index, index_pointer);
            let mut archived = node.get_archived_node();
            for (i, new_neighbor) in neighbors.iter().enumerate() {
                //TODO: why do we need to recreate the archive?
                let mut a_index_pointer = archived.as_mut().neighbor_index_pointer().index_pin(i);
                //TODO hate that we have to set each field like this
                a_index_pointer.block_number =
                    new_neighbor.get_index_pointer_to_neigbor().block_number;
                a_index_pointer.offset = new_neighbor.get_index_pointer_to_neigbor().offset;

                let mut a_distance = archived.as_mut().neighbor_distances().index_pin(i);
                *a_distance = new_neighbor.get_distance() as Distance;
            }
            //set the marker that the list ended
            if neighbors.len() < self.meta_page.get_num_neighbors() as _ {
                //TODO: why do we need to recreate the archive?
                let archived = node.get_archived_node();
                let mut past_last_distance =
                    archived.neighbor_distances().index_pin(neighbors.len());
                *past_last_distance = Distance::NAN;
            }
            node.commit()
        }
        stats
    }
}

impl Graph for BuilderGraph {
    fn get_init_ids(&mut self) -> Option<Vec<ItemPointer>> {
        //TODO make this based on centroid. For now, just first node.
        //returns a vector for generality
        match &self.first {
            Some(item) => Some(vec![*item]),
            None => match self.neighbor_map.keys().next() {
                Some(item) => {
                    self.first = Some(*item);
                    Some(vec![*item])
                }
                None => None,
            },
        }
    }

    fn read(&self, index: &PgRelation, index_pointer: ItemPointer) -> ReadableNode {
        unsafe { Node::read(index, &index_pointer) }
    }

    fn get_neighbors(
        &self,
        _index: &PgRelation,
        neighbors_of: ItemPointer,
        result: &mut Vec<NeighborWithDistance>,
    ) -> bool {
        let neighbors = self.neighbor_map.get(&neighbors_of);
        match neighbors {
            Some(n) => {
                for nwd in n {
                    result.push(nwd.clone());
                }
                true
            }
            None => false,
        }
    }

    fn get_meta_page(&self, _index: &PgRelation) -> &TsvMetaPage {
        &self.meta_page
    }
}

pub struct WriteStats {
    pub started: Instant,
    pub num_nodes: usize,
    pub num_prunes: usize,
    pub num_neighbors_before_prune: usize,
    pub num_neighbors_after_prune: usize,
    pub num_neighbors: usize,
}

impl WriteStats {
    pub fn new() -> Self {
        Self {
            started: Instant::now(),
            num_nodes: 0,
            num_prunes: 0,
            num_neighbors_before_prune: 0,
            num_neighbors_after_prune: 0,
            num_neighbors: 0,
        }
    }
}
