use std::num::NonZero;

use pgrx::debug1;

use crate::access_method::storage::Storage;
// Use the PostgreSQL-native version
use crate::lru::pg_lru_cache_with_stats::PgLruCacheWithStats;

use crate::{
    access_method::{
        build::maintenance_work_mem_bytes,
        stats::{StatsNodeModify, StatsNodeRead, StatsNodeWrite},
    },
    util::{IndexPointer, ItemPointer},
};

use super::{node::SbqNode, SbqSpeedupStorage, SbqVectorElement};

pub struct QuantizedVectorCache {
    cache: PgLruCacheWithStats<ItemPointer, Vec<SbqVectorElement>>,
    // Cache the last accessed value to return references
    last_value: Option<Vec<SbqVectorElement>>,
}

impl QuantizedVectorCache {
    pub fn new(memory_budget: f64, sbq_vec_len: usize, min_capacity: usize) -> Self {
        let total_memory = maintenance_work_mem_bytes() as f64;
        let memory_budget = (total_memory * memory_budget).ceil() as usize;
        let capacity = std::cmp::max(memory_budget / Self::entry_size(sbq_vec_len), min_capacity);

        Self {
            cache: PgLruCacheWithStats::new(NonZero::new(capacity).unwrap(), "Quantized vector"),
            last_value: None,
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
        // Try to get from cache first
        #[cfg(not(test))]
        {
            if let Some(value) = self.cache.get(&index_pointer) {
                self.last_value = Some(value);
            } else {
                // Not in cache, need to read from storage
                let node = unsafe {
                    SbqNode::read(
                        storage.index,
                        index_pointer,
                        storage.get_has_labels(),
                        stats,
                    )
                };
                let vector = node.get_archived_node().get_bq_vector().to_vec();

                // Insert into cache
                self.cache.push(index_pointer, vector.clone());
                self.last_value = Some(vector);
            }
        }
        #[cfg(test)]
        {
            if let Some(value) = self.cache.get(&index_pointer) {
                self.last_value = Some(value.clone());
            } else {
                // Not in cache, need to read from storage
                let node = unsafe {
                    SbqNode::read(
                        storage.index,
                        index_pointer,
                        storage.get_has_labels(),
                        stats,
                    )
                };
                let vector = node.get_archived_node().get_bq_vector().to_vec();

                // Insert into cache
                self.cache.push(index_pointer, vector.clone());
                self.last_value = Some(vector);
            }
        }

        self.last_value.as_ref().unwrap()
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
                let node = unsafe {
                    SbqNode::read(storage.index, item_pointer, storage.get_has_labels(), stats)
                };
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
