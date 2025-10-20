use std::hash::Hash;
use std::num::NonZero;

/// Wrapper around LruCache that tracks statistics about the cache
/// and warns on first eviction using a parameterized message.
use lru::LruCache;
use pgrx::warning;

#[derive(Copy, Clone, Debug, Default)]
pub struct CacheStats {
    pub inserts: usize,
    pub updates: usize,
    pub hits: usize,
    pub misses: usize,
    pub evictions: usize,
}

pub struct LruCacheWithStats<K: Hash + Eq + Clone, V> {
    cache: LruCache<K, V>,
    cache_name: String,
    stats: CacheStats,
}

impl<K: Hash + Eq + Clone, V> LruCacheWithStats<K, V> {
    pub fn new(capacity: NonZero<usize>, cache_name: &str) -> Self {
        LruCacheWithStats {
            cache: LruCache::new(capacity),
            cache_name: cache_name.to_string(),
            stats: CacheStats::default(),
        }
    }

    pub fn cap(&self) -> NonZero<usize> {
        self.cache.cap()
    }

    #[allow(unused)]
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    pub fn contains(&mut self, key: &K) -> bool {
        if self.cache.contains(key) {
            self.stats.hits += 1;
            true
        } else {
            self.stats.misses += 1;
            false
        }
    }

    /// Pushes a key-value pair into the cache.
    ///
    /// # Returns
    ///
    /// * `None` if the key was already in the cache (update) or if the cache had space
    /// * `Some((K, V))` if an existing key-value pair was evicted to make space
    ///
    /// # Note
    ///
    /// This differs from the underlying `LruCache::push` method, which returns:
    /// * The old value when updating an existing key
    /// * The evicted key-value pair when inserting a new key
    pub fn push(&mut self, key: K, value: V) -> Option<(K, V)> {
        let result = self.cache.push(key.clone(), value);
        if let Some((old_key, _)) = &result {
            if old_key == &key {
                // The key was already in the cache, so we didn't evict anything
                self.stats.updates += 1;
                return None;
            }
            self.stats.inserts += 1;
            // The key was evicted, so we return the old key-value pair
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
                    self.cache.cap(),
                    self.stats,
                );
            }
        } else {
            self.stats.inserts += 1;
        }
        result
    }

    pub fn get(&mut self, key: &K) -> Option<&V> {
        let result = self.cache.get(key);
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

    pub fn into_parts(self) -> (LruCache<K, V>, CacheStats) {
        (self.cache, self.stats)
    }
}
