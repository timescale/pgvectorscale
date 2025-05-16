use std::num::NonZero;

use lru::LruCache;

use crate::access_method::meta_page::MetaPage;
use crate::util::{IndexPointer, ItemPointer};

use crate::access_method::graph::neighbor_with_distance::*;
use crate::access_method::labels::LabelSet;
use crate::access_method::stats::{
    StatsDistanceComparison, StatsNodeModify, StatsNodeRead, StatsNodeWrite,
};
use crate::access_method::storage::Storage;

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
    neighbor_map: LruCache<ItemPointer, NeighborCacheEntry>,
}

impl BuilderNeighborCache {
    pub fn new(memory_budget: f64, num_neighbors: usize, has_labels: bool) -> Self {
        let total_memory = unsafe { pgrx::pg_sys::maintenance_work_mem as f64 };
        let memory_budget = (total_memory * memory_budget).ceil() as usize;
        let capacity = memory_budget / NeighborCacheEntry::size(num_neighbors, has_labels);
        Self {
            neighbor_map: LruCache::new(NonZero::new(capacity).unwrap()),
        }
    }

    /// Convert cache to a sorted vector of neighbors
    pub fn drain_sorted(&mut self) -> Vec<(ItemPointer, NeighborCacheEntry)> {
        let neighbor_map = std::mem::replace(
            &mut self.neighbor_map,
            LruCache::new(NonZero::new(1).unwrap()),
        );
        let mut vec = neighbor_map.into_iter().collect::<Vec<_>>();
        vec.sort_by_key(|(key, _)| *key);
        vec
    }

    pub fn get_neighbors<
        S: Storage,
        T: StatsNodeRead + StatsDistanceComparison + StatsNodeWrite + StatsNodeModify,
    >(
        &mut self,
        neighbors_of: ItemPointer,
        storage: &S,
        stats: &mut T,
    ) -> Vec<IndexPointer> {
        let neighbors = self.get_neighbors_with_full_vector_distances(neighbors_of, storage, stats);
        neighbors
            .iter()
            .map(|n| n.get_index_pointer_to_neighbor())
            .collect()
    }

    pub fn get_neighbors_with_full_vector_distances<
        S: Storage,
        T: StatsNodeRead + StatsDistanceComparison + StatsNodeWrite + StatsNodeModify,
    >(
        &mut self,
        neighbors_of: ItemPointer,
        storage: &S,
        stats: &mut T,
    ) -> Vec<NeighborWithDistance> {
        let neighbors = self.neighbor_map.get(&neighbors_of);
        if let Some(entry) = neighbors {
            return entry.neighbors.clone();
        }
        let neighbors = storage.get_neighbors_with_distances_from_disk(neighbors_of, stats);

        self.set_neighbors(neighbors_of, None, neighbors.clone(), storage, stats);
        neighbors
    }

    pub fn set_neighbors<S: Storage, T: StatsNodeModify + StatsNodeRead>(
        &mut self,
        neighbors_of: ItemPointer,
        labels: Option<LabelSet>,
        new_neighbors: Vec<NeighborWithDistance>,
        storage: &S,
        stats: &mut T,
    ) {
        let evictee = self
            .neighbor_map
            .push(neighbors_of, NeighborCacheEntry::new(labels, new_neighbors));
        if let Some((key, value)) = evictee {
            // write to disk
            storage.set_neighbors_on_disk(key, value.neighbors.as_slice(), stats);
        }
    }
}

pub enum GraphNeighborStore {
    Builder(BuilderNeighborCache),
    Disk,
}

impl GraphNeighborStore {
    pub fn get_neighbors_with_full_vector_distances<
        S: Storage,
        T: StatsNodeRead + StatsDistanceComparison + StatsNodeWrite + StatsNodeModify,
    >(
        &mut self,
        neighbors_of: ItemPointer,
        storage: &S,
        stats: &mut T,
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

    pub fn set_neighbors<S: Storage, T: StatsNodeModify + StatsNodeRead>(
        &mut self,
        storage: &S,
        neighbors_of: ItemPointer,
        labels: Option<LabelSet>,
        new_neighbors: Vec<NeighborWithDistance>,
        stats: &mut T,
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
        meta_page.get_max_neighbors_during_build()
    }
}
