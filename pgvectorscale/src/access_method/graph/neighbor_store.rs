use std::cell::RefCell;
use std::collections::HashSet;
use std::num::NonZero;

use pgrx::debug1;

use crate::access_method::build::maintenance_work_mem_bytes;
use crate::util::lru::LruCacheWithStats;
use crate::util::{IndexPointer, ItemPointer};

use crate::access_method::graph::neighbor_with_distance::*;
use crate::access_method::labels::LabelSet;
use crate::access_method::meta_page::MetaPage;
use crate::access_method::stats::PruneNeighborStats;
use crate::access_method::storage::Storage;

use super::Graph;

/// A builderGraph is a graph that keep the neighbors in-memory in the neighbor_map below
/// The idea is that during the index build, you don't want to update the actual Postgres
/// pages every time you change the neighbors. Instead you change the neighbors in memory
/// until the build is done. Afterwards, calling the `write` method, will write out all
/// the neighbors to the right pages.
///
pub struct NeighborCacheEntry {
    pub labels: Option<LabelSet>,
    pub neighbors: Vec<NeighborWithDistance>,
}

impl NeighborCacheEntry {
    pub fn new(labels: Option<LabelSet>, neighbors: Vec<NeighborWithDistance>) -> Self {
        Self { labels, neighbors }
    }

    /// Estimate of the size of an entry in the cache in bytes.
    pub fn size(num_neighbors: usize, has_labels: bool) -> usize {
        std::mem::size_of::<Self>()
            + num_neighbors * std::mem::size_of::<NeighborWithDistance>()
            + if has_labels {
                // Heuristic: assume around 4 labels per vector, or 8 bytes of payload
                8
            } else {
                0
            }
    }
}

pub struct BuilderNeighborCache {
    /// Map of node pointer to neighbor cache entry
    neighbor_map: RefCell<LruCacheWithStats<ItemPointer, NeighborCacheEntry>>,
    num_neighbors: usize,
    max_alpha: f64,
}

impl BuilderNeighborCache {
    fn reconcile_with_disk_neighbors<S: Storage>(
        &self,
        neighbors_of: ItemPointer,
        cached_neighbors: Vec<NeighborWithDistance>,
        storage: &S,
        stats: &mut PruneNeighborStats,
    ) -> Vec<NeighborWithDistance> {
        let disk_neighbors = storage.get_neighbors_with_distances_from_disk(neighbors_of, stats);
        let mut all_neighbors = Vec::with_capacity(cached_neighbors.len() + disk_neighbors.len());

        let cached_pointers: HashSet<_> = cached_neighbors
            .iter()
            .map(|n| n.get_index_pointer_to_neighbor())
            .collect();
        all_neighbors.extend(cached_neighbors);
        all_neighbors.extend(disk_neighbors.into_iter().filter(|disk_neighbor| {
            !cached_pointers.contains(&disk_neighbor.get_index_pointer_to_neighbor())
        }));

        all_neighbors
    }

    pub fn new(memory_budget: f64, meta_page: &MetaPage, worker_count: usize) -> Self {
        let total_memory = maintenance_work_mem_bytes() as f64;
        let memory_budget = (total_memory * memory_budget).ceil() as usize;
        let capacity = memory_budget
            / NeighborCacheEntry::size(meta_page.get_num_neighbors() as _, meta_page.has_labels());
        let capacity = if worker_count > 0 {
            capacity / worker_count
        } else {
            capacity
        };

        Self {
            neighbor_map: RefCell::new(LruCacheWithStats::new(
                NonZero::new(capacity).unwrap(),
                "Builder neighbor",
            )),
            num_neighbors: meta_page.get_num_neighbors() as _,
            max_alpha: meta_page.get_max_alpha(),
        }
    }

    /// Convert cache to a sorted vector of neighbors
    fn into_sorted(self) -> Vec<(ItemPointer, NeighborCacheEntry)> {
        let (neighbor_map, stats) = self.neighbor_map.into_inner().into_parts();
        debug1!(
            "Builder neighbor cache teardown: capacity {}, stats: {:?}",
            neighbor_map.cap(),
            stats
        );
        let mut vec = neighbor_map.into_iter().collect::<Vec<_>>();
        vec.sort_by_key(|(key, _)| *key);
        vec
    }

    pub fn get_neighbors<S: Storage>(
        &self,
        neighbors_of: ItemPointer,
        storage: &S,
        stats: &mut PruneNeighborStats,
    ) -> Vec<IndexPointer> {
        let neighbors = self.get_neighbors_with_full_vector_distances(neighbors_of, storage, stats);
        neighbors
            .iter()
            .map(|n| n.get_index_pointer_to_neighbor())
            .collect()
    }

    pub fn get_neighbors_with_full_vector_distances<S: Storage>(
        &self,
        neighbors_of: ItemPointer,
        storage: &S,
        stats: &mut PruneNeighborStats,
    ) -> Vec<NeighborWithDistance> {
        let mut neighbor_map = self.neighbor_map.borrow_mut();
        let neighbors = neighbor_map.get(&neighbors_of);
        if let Some(entry) = neighbors {
            return entry.neighbors.clone();
        }
        drop(neighbor_map);
        let neighbors = storage.get_neighbors_with_distances_from_disk(neighbors_of, stats);

        self.set_neighbors(neighbors_of, None, neighbors.clone(), storage, stats);
        neighbors
    }

    pub fn set_neighbors<S: Storage>(
        &self,
        neighbors_of: ItemPointer,
        labels: Option<LabelSet>,
        mut new_neighbors: Vec<NeighborWithDistance>,
        storage: &S,
        stats: &mut PruneNeighborStats,
    ) {
        let mut neighbor_map = self.neighbor_map.borrow_mut();
        new_neighbors.shrink_to_fit();
        let evictee =
            neighbor_map.push(neighbors_of, NeighborCacheEntry::new(labels, new_neighbors));
        if let Some((key, value)) = evictee {
            let all_neighbors =
                self.reconcile_with_disk_neighbors(key, value.neighbors, storage, stats);
            let new_neighbors = Graph::prune_neighbors(
                self.max_alpha,
                self.num_neighbors,
                value.labels.as_ref(),
                all_neighbors,
                storage,
                stats,
            );
            storage.set_neighbors_on_disk(key, new_neighbors.as_slice(), stats);
        }
    }

    /// Flush cached entries to disk if cache usage is above the given threshold.
    /// This helps prevent memory buildup during parallel builds.
    pub fn flush_if_above_threshold<S: Storage>(
        &self,
        storage: &S,
        stats: &mut PruneNeighborStats,
        _threshold: f64,
    ) {
        let mut cache = self.neighbor_map.borrow_mut();
        while cache.len() > 0 {
            let (neighbors_of, entry) = cache.pop_lru().unwrap();
            drop(cache);
            let all_neighbors =
                self.reconcile_with_disk_neighbors(neighbors_of, entry.neighbors, storage, stats);

            let pruned_neighbors = if all_neighbors.len() > self.num_neighbors {
                Graph::prune_neighbors(
                    self.max_alpha,
                    self.num_neighbors,
                    entry.labels.as_ref(),
                    all_neighbors,
                    storage,
                    stats,
                )
            } else {
                all_neighbors
            };

            storage.set_neighbors_on_disk(neighbors_of, &pruned_neighbors, stats);
            cache = self.neighbor_map.borrow_mut();
        }
    }
}

pub enum GraphNeighborStore {
    Builder(BuilderNeighborCache),
    Disk,
}

impl GraphNeighborStore {
    pub fn get_neighbors_with_full_vector_distances<S: Storage>(
        &self,
        neighbors_of: ItemPointer,
        storage: &S,
        stats: &mut PruneNeighborStats,
    ) -> Vec<NeighborWithDistance> {
        match self {
            GraphNeighborStore::Builder(b) => {
                b.get_neighbors_with_full_vector_distances(neighbors_of, storage, stats)
            }
            GraphNeighborStore::Disk => {
                storage.get_neighbors_with_distances_from_disk(neighbors_of, stats)
            }
        }
    }

    pub fn set_neighbors<S: Storage>(
        &self,
        storage: &S,
        neighbors_of: ItemPointer,
        labels: Option<LabelSet>,
        new_neighbors: Vec<NeighborWithDistance>,
        stats: &mut PruneNeighborStats,
    ) {
        match self {
            GraphNeighborStore::Builder(b) => {
                b.set_neighbors(neighbors_of, labels, new_neighbors, storage, stats)
            }
            GraphNeighborStore::Disk => {
                storage.set_neighbors_on_disk(neighbors_of, new_neighbors.as_slice(), stats)
            }
        }
    }

    pub fn max_neighbors(&self, meta_page: &MetaPage) -> usize {
        match self {
            GraphNeighborStore::Builder(_) => meta_page.get_max_neighbors_during_build(),
            GraphNeighborStore::Disk => meta_page.get_num_neighbors() as _,
        }
    }

    pub fn into_sorted(self) -> Vec<(ItemPointer, NeighborCacheEntry)> {
        match self {
            GraphNeighborStore::Builder(b) => b.into_sorted(),
            GraphNeighborStore::Disk => {
                panic!("Should not be using the disk neighbor store during build")
            }
        }
    }
}
