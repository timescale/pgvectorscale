use std::hash::Hash;
use std::num::NonZero;

/// Wrapper around LruCache that tracks statistics about the cache
/// and warns on first eviction using a parameterized message.
use lru::LruCache;
use pgrx::warning;

#[derive(Copy, Clone, Debug, Default)]
pub struct CacheStats {
    pub inserts: usize,
    pub hits: usize,
    pub misses: usize,
    pub evictions: usize,
}

pub struct LruCacheWithStats<K: Hash + Eq, V> {
    cache: LruCache<K, V>,
    cache_name: String,
    stats: CacheStats,
}

impl<K: Hash + Eq, V> LruCacheWithStats<K, V> {
    pub fn new(capacity: NonZero<usize>, cache_name: &str) -> Self {
        LruCacheWithStats {
            cache: LruCache::new(capacity),
            cache_name: cache_name.to_string(),
            stats: CacheStats::default(),
        }
    }

    pub fn contains(&self, key: &K) -> bool {
        self.cache.contains(key)
    }

    pub fn push(&mut self, key: K, value: V) -> Option<(K, V)> {
        self.stats.inserts += 1;
        let result = self.cache.push(key, value);
        if result.is_some() {
            if self.stats.evictions == 0 {
                warning!("{} is full after processing {} vectors, consider increasing maintenance_work_mem", self.cache_name, self.stats.inserts);
            }
            self.stats.evictions += 1;
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

    #[allow(dead_code)]
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    pub fn into_parts(self) -> (LruCache<K, V>, CacheStats) {
        (self.cache, self.stats)
    }
}
