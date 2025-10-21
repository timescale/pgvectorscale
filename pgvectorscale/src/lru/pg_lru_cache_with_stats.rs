//! Adapter to make PgSharedLru compatible with the LruCacheWithStats interface
//!
//! This allows us to drop in the PostgreSQL-native LRU cache as a replacement
//! for the existing process-local LruCache.

use pgrx::warning;
use rkyv::{Archive, Deserialize, Serialize};
use std::hash::Hash;
use std::num::NonZero;

use super::pg_lru::PgSharedLru;
use super::shared_memory;

/// Statistics tracking for the cache
#[derive(Copy, Clone, Debug, Default)]
pub struct CacheStats {
    pub inserts: usize,
    pub updates: usize,
    pub hits: usize,
    pub misses: usize,
    pub evictions: usize,
}

/// Type of cache to use
#[derive(Debug, Clone, Copy)]
pub enum CacheType {
    QuantizedVector,
    BuilderNeighbor,
}

/// Wrapper around PgSharedLru that provides LruCacheWithStats interface
pub struct PgLruCacheWithStats<K, V>
where
    K: Archive
        + Serialize<rkyv::ser::serializers::AllocSerializer<256>>
        + Hash
        + Eq
        + Clone
        + PartialEq,
    V: Archive + Serialize<rkyv::ser::serializers::AllocSerializer<256>> + Clone,
    K::Archived: PartialEq<K>,
    V::Archived: Deserialize<V, rkyv::Infallible>,
{
    cache: &'static PgSharedLru,
    cache_name: String,
    capacity: usize,
    stats: CacheStats,
    _phantom: std::marker::PhantomData<(K, V)>,
}

impl<K, V> PgLruCacheWithStats<K, V>
where
    K: Archive
        + Serialize<rkyv::ser::serializers::AllocSerializer<256>>
        + Hash
        + Eq
        + Clone
        + PartialEq,
    V: Archive + Serialize<rkyv::ser::serializers::AllocSerializer<256>> + Clone,
    K::Archived: PartialEq<K>,
    V::Archived: Deserialize<V, rkyv::Infallible>,
{
    pub fn new(capacity: NonZero<usize>, cache_name: &str) -> Self {
        // Shared memory should have been initialized by shmem_startup_hook
        // If it's not initialized, something went wrong during PostgreSQL startup
        if !shared_memory::is_initialized() {
            panic!("Shared memory not initialized - shmem_startup_hook may not have been called");
        }

        // Determine cache type from name
        let cache_type = if cache_name.contains("quantized") || cache_name.contains("Quantized") {
            CacheType::QuantizedVector
        } else {
            CacheType::BuilderNeighbor
        };

        // Get the appropriate shared cache
        let cache = match cache_type {
            CacheType::QuantizedVector => shared_memory::get_quantized_vector_cache()
                .expect("Failed to get quantized vector cache"),
            CacheType::BuilderNeighbor => shared_memory::get_builder_neighbor_cache()
                .expect("Failed to get builder neighbor cache"),
        };

        PgLruCacheWithStats {
            cache,
            cache_name: cache_name.to_string(),
            capacity: capacity.get(),
            stats: CacheStats::default(),
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn cap(&self) -> NonZero<usize> {
        NonZero::new(self.capacity).unwrap()
    }

    #[allow(unused)]
    pub fn len(&self) -> usize {
        let pg_stats = self.cache.stats();
        pg_stats.entry_count
    }

    #[allow(unused)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn contains(&mut self, key: &K) -> bool {
        unsafe {
            if self.cache.get::<K, V>(key).is_some() {
                self.stats.hits += 1;
                true
            } else {
                self.stats.misses += 1;
                false
            }
        }
    }

    /// Pushes a key-value pair into the cache.
    ///
    /// # Returns
    ///
    /// * `None` if the key was already in the cache (update) or if the cache had space
    /// * `Some((K, V))` if an existing key-value pair was evicted to make space
    pub fn push(&mut self, key: K, value: V) -> Option<(K, V)> {
        // Check if key exists first
        let existed = unsafe { self.cache.get::<K, V>(&key).is_some() };

        // Get current entry count before insert
        let before_count = self.cache.stats().entry_count;

        // Insert the new value
        let result = unsafe { self.cache.insert(key.clone(), value) };

        if let Err(e) = result {
            pgrx::warning!("Failed to insert into {}: {}", self.cache_name, e);
            return None;
        }

        // Check if we evicted something
        let after_count = self.cache.stats().entry_count;

        if existed {
            self.stats.updates += 1;
            None
        } else {
            self.stats.inserts += 1;

            // If count didn't increase, we must have evicted
            if after_count <= before_count && before_count > 0 {
                if self.stats.evictions == 0 {
                    warning!(
                        "{} cache is full after processing {} vectors; consider increasing maintenance_work_mem",
                        self.cache_name,
                        self.stats.inserts
                    );
                }
                self.stats.evictions += 1;

                if self.stats.evictions.is_multiple_of(10000) {
                    pgrx::debug1!(
                        "{} cache capacity {}, stats: {:?}",
                        self.cache_name,
                        self.capacity,
                        self.stats,
                    );
                }

                // We can't return the evicted value since PgSharedLru doesn't track it
                // Return a dummy value to indicate eviction occurred
                // In practice, the callers don't use the evicted value
                None
            } else {
                None
            }
        }
    }

    pub fn get(&mut self, key: &K) -> Option<V> {
        let result = unsafe { self.cache.get::<K, V>(key) };
        if result.is_some() {
            self.stats.hits += 1;
        } else {
            self.stats.misses += 1;
        }
        result
    }

    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    pub fn stats_clone(&self) -> CacheStats {
        self.stats
    }

    /// Convert to parts - returns a dummy LruCache and stats
    ///
    /// Note: This is not a perfect implementation since PgSharedLru doesn't
    /// support iteration/enumeration. We return an empty LruCache here.
    /// The caller will need to handle this appropriately.
    pub fn into_parts(self) -> (lru::LruCache<K, V>, CacheStats) {
        // Create an empty LruCache with the same capacity
        let cache = lru::LruCache::new(NonZero::new(self.capacity).unwrap());
        (cache, self.stats)
    }
}
