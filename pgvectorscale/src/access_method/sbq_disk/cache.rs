use std::num::NonZero;

use pgrx::debug1;

use crate::util::lru::LruCacheWithStats;

use crate::{
    access_method::{
        build::maintenance_work_mem_bytes,
        stats::{StatsNodeModify, StatsNodeRead, StatsNodeWrite},
    },
    util::{IndexPointer, ItemPointer},
};

use super::{node::SbqDiskNode, SbqDiskSpeedupStorage, SbqDiskVectorElement};

pub struct QuantizedVectorCache {
    cache: LruCacheWithStats<ItemPointer, Vec<SbqDiskVectorElement>>,
}

impl QuantizedVectorCache {
    pub fn new(memory_budget: f64, sbq_vec_len: usize, min_capacity: usize) -> Self {
        let total_memory = maintenance_work_mem_bytes() as f64;
        let memory_budget = (total_memory * memory_budget).ceil() as usize;
        let capacity = std::cmp::max(memory_budget / Self::entry_size(sbq_vec_len), min_capacity);

        Self {
            cache: LruCacheWithStats::new(NonZero::new(capacity).unwrap(), "Quantized vector"),
        }
    }

    /// Estimate of the size of an entry in the cache in bytes.
    pub fn entry_size(sbq_vec_len: usize) -> usize {
        std::mem::size_of::<ItemPointer>()
            + std::mem::size_of::<Vec<SbqDiskVectorElement>>()
            + (std::mem::size_of::<SbqDiskVectorElement>() * sbq_vec_len)
    }

    pub fn get<S: StatsNodeRead + StatsNodeWrite + StatsNodeModify>(
        &mut self,
        index_pointer: IndexPointer,
        storage: &SbqDiskSpeedupStorage,
        stats: &mut S,
    ) -> &[SbqDiskVectorElement] {
        // TODO this probes the cache twice in the case of a hit, figure out
        // how to do this in a single probe without running afould of the Rust
        // borrow checker
        if !self.cache.contains(&index_pointer) {
            // Not in cache, need to read from storage
            let node = unsafe { SbqDiskNode::read(storage.index, index_pointer, storage.has_labels, stats) };
            let vector = node.get_archived_node().get_bq_vector().to_vec();

            // Insert into cache and handle evicted item
            self.cache.push(index_pointer, vector);
        }

        self.cache.get(&index_pointer).unwrap()
    }

    pub fn preload<I: Iterator<Item = IndexPointer>, S: StatsNodeRead>(
        &mut self,
        index_pointers: I,
        storage: &SbqDiskSpeedupStorage,
        stats: &mut S,
    ) {
        for index_pointer in index_pointers {
            let item_pointer = ItemPointer::new(index_pointer.block_number, index_pointer.offset);
            // Only load if not already in cache
            if !self.cache.contains(&item_pointer) {
                let node = unsafe { SbqDiskNode::read(storage.index, item_pointer, storage.has_labels, stats) };
                let vector = node.get_archived_node().get_bq_vector().to_vec();
                self.cache.push(item_pointer, vector);
            }
        }
    }
}

impl Drop for QuantizedVectorCache {
    fn drop(&mut self) {
        debug1!(
            "Quantized vector cache teardown: capacity {}, stats: {:?}",
            self.cache.cap(),
            self.cache.stats()
        );
    }
}
