use std::cell::RefCell;
use std::num::NonZero;

use lru::LruCache;
use pgrx::warning;

use crate::access_method::build::maintenance_work_mem_bytes;
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
                std::mem::size_of::<LabelSet>() + 4
            } else {
                0
            }
    }
}

pub struct BuilderNeighborCache {
    /// Map of node pointer to neighbor cache entry
    neighbor_map: RefCell<LruCache<ItemPointer, NeighborCacheEntry>>,
    num_neighbors: usize,
    max_alpha: f64,
    warned_full: RefCell<bool>,
}

impl BuilderNeighborCache {
    pub fn new(memory_budget: f64, meta_page: &MetaPage) -> Self {
        let total_memory = maintenance_work_mem_bytes() as f64;
        let memory_budget = (total_memory * memory_budget).ceil() as usize;
        let capacity = memory_budget
            / NeighborCacheEntry::size(meta_page.get_num_neighbors() as _, meta_page.has_labels());

        pgrx::debug1!(
            "BuilderNeighborCache::new capacity: {} memory_budget: {} total_memory: {}",
            capacity,
            memory_budget,
            total_memory
        );

        Self {
            neighbor_map: RefCell::new(LruCache::new(NonZero::new(capacity).unwrap())),
            num_neighbors: meta_page.get_num_neighbors() as _,
            max_alpha: meta_page.get_max_alpha(),
            warned_full: RefCell::new(false),
        }
    }

    /// Convert cache to a sorted vector of neighbors
    fn into_sorted(self) -> Vec<(ItemPointer, NeighborCacheEntry)> {
        let neighbor_map = self.neighbor_map.into_inner();
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
        new_neighbors: Vec<NeighborWithDistance>,
        storage: &S,
        stats: &mut PruneNeighborStats,
    ) {
        let mut neighbor_map = self.neighbor_map.borrow_mut();
        let evictee =
            neighbor_map.push(neighbors_of, NeighborCacheEntry::new(labels, new_neighbors));
        if let Some((key, value)) = evictee {
            if !*self.warned_full.borrow() {
                warning!(
                    "Vector neighbor cache is full after processing {} vectors, consider increasing maintenance_work_mem",
                    neighbor_map.len()
                );
                *self.warned_full.borrow_mut() = true;
            }
            let new_neighbors = Graph::prune_neighbors(
                self.max_alpha,
                self.num_neighbors,
                value.labels.as_ref(),
                value.neighbors,
                storage,
                stats,
            );
            storage.set_neighbors_on_disk(key, new_neighbors.as_slice(), stats);
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
