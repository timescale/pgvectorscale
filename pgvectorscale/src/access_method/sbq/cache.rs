use std::num::NonZero;

use lru::LruCache;
use pgrx::warning;

use crate::{
    access_method::stats::{StatsNodeModify, StatsNodeRead, StatsNodeWrite},
    util::{IndexPointer, ItemPointer},
};

use super::{node::SbqNode, SbqSpeedupStorage, SbqVectorElement};

pub struct QuantizedVectorCache {
    cache: LruCache<ItemPointer, Vec<SbqVectorElement>>,
    warned_full: bool,
}

impl QuantizedVectorCache {
    pub fn new(memory_budget: f64, sbq_vec_len: usize) -> Self {
        let total_memory = unsafe { pgrx::pg_sys::maintenance_work_mem as f64 };
        let memory_budget = (total_memory * memory_budget).ceil() as usize;
        let capacity = memory_budget / Self::entry_size(sbq_vec_len);
        Self {
            cache: LruCache::new(NonZero::new(capacity).unwrap()),
            warned_full: false,
        }
    }

    /// Estimate of the size of an entry in the cache in bytes.
    pub fn entry_size(sbq_vec_len: usize) -> usize {
        std::mem::size_of::<ItemPointer>()
            + std::mem::size_of::<Vec<SbqVectorElement>>()
            + (std::mem::size_of::<SbqVectorElement>() * sbq_vec_len)
    }

    pub fn get<S: StatsNodeRead + StatsNodeWrite + StatsNodeModify>(
        &mut self,
        index_pointer: IndexPointer,
        storage: &SbqSpeedupStorage,
        stats: &mut S,
    ) -> &[SbqVectorElement] {
        // TODO this probes the cache twice in the case of a hit, figure out
        // how to do this in a single probe without running afould of the Rust
        // borrow checker
        if !self.cache.contains(&index_pointer) {
            // Not in cache, need to read from storage
            let node = unsafe { SbqNode::read(storage.index, index_pointer, true, stats) };
            let vector = node.get_archived_node().get_bq_vector().to_vec();

            // Insert into cache and handle evicted item
            let evicted = self.cache.push(index_pointer, vector);
            if let Some((evicted_pointer, evicted_vector)) = evicted {
                if !self.warned_full {
                    warning!(
                        "Quantized vector cache is full after processing {} vectors, consider increasing maintenance_work_mem",
                        self.cache.len()
                    );
                    self.warned_full = true;
                }
                storage.write_quantized_vector(evicted_pointer, evicted_vector.as_slice(), stats);
            }
        }

        self.cache.get(&index_pointer).unwrap()
    }

    pub fn preload<I: Iterator<Item = IndexPointer>, S: StatsNodeRead>(
        &mut self,
        index_pointers: I,
        storage: &SbqSpeedupStorage,
        stats: &mut S,
    ) {
        for index_pointer in index_pointers {
            let item_pointer = ItemPointer::new(index_pointer.block_number, index_pointer.offset);
            // Only load if not already in cache
            if !self.cache.contains(&item_pointer) {
                let node = unsafe { SbqNode::read(storage.index, item_pointer, true, stats) };
                let vector = node.get_archived_node().get_bq_vector().to_vec();
                self.cache.push(item_pointer, vector);
            }
        }
    }
}
