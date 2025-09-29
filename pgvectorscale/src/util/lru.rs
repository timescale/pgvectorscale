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
    use_mru: bool,
}

impl<K: Hash + Eq + Clone, V> LruCacheWithStats<K, V> {
    #[allow(dead_code)]
    pub fn new(capacity: NonZero<usize>, cache_name: &str) -> Self {
        LruCacheWithStats {
            cache: LruCache::new(capacity),
            cache_name: cache_name.to_string(),
            stats: CacheStats::default(),
            use_mru: false,
        }
    }

    pub fn new_mru(capacity: NonZero<usize>, cache_name: &str) -> Self {
        LruCacheWithStats {
            cache: LruCache::new(capacity),
            cache_name: cache_name.to_string(),
            stats: CacheStats::default(),
            use_mru: true,
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
    ///
    /// When use_mru is true, implements MRU (Most Recently Used) eviction by
    /// manually evicting the most recently used item before insertion when the cache is full.
    pub fn push(&mut self, key: K, value: V) -> Option<(K, V)> {
        if self.use_mru && self.cache.len() == self.cache.cap().get() && !self.cache.contains(&key)
        {
            // For MRU behavior: we need to evict the most recently used item
            // The lru crate doesn't have a direct way to get MRU, so we'll simulate it by
            // first getting the MRU key (which is the last accessed), then removing it
            let mut mru_key_to_evict = None;

            // Find the MRU key by iterating and taking the last one (most recent)
            for (k, _) in self.cache.iter() {
                mru_key_to_evict = Some(k.clone());
                break; // The first item in iter() is actually the MRU in lru crate
            }

            if let Some(mru_key) = mru_key_to_evict {
                if let Some(mru_value) = self.cache.pop(&mru_key) {
                    // Now insert the new item
                    self.cache.push(key, value);
                    self.stats.inserts += 1;
                    if self.stats.evictions == 0 {
                        warning!(
                            "{} cache is full after processing {} vectors; consider increasing maintenance_work_mem",
                            self.cache_name,
                            self.stats.inserts
                        );
                    }
                    self.stats.evictions += 1;
                    if self.stats.evictions % 10000 == 0 {
                        pgrx::debug1!(
                            "{} cache capacity {}, stats: {:?}",
                            self.cache_name,
                            self.cache.cap(),
                            self.stats,
                        );
                    }
                    return Some((mru_key, mru_value));
                }
            }
        }

        // Standard LRU behavior or MRU when cache not full
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
            if self.stats.evictions % 10000 == 0 {
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

    /// Remove and return the least recently used key-value pair (LRU mode)
    /// or the most recently used key-value pair (MRU mode)
    pub fn pop_lru(&mut self) -> Option<(K, V)> {
        if self.use_mru {
            // For MRU: evict the most recently used item
            let mru_key_to_evict = self.cache.iter().next().map(|(k, _)| k.clone());
            if let Some(mru_key) = mru_key_to_evict {
                if let Some(mru_value) = self.cache.pop(&mru_key) {
                    self.stats.evictions += 1;
                    return Some((mru_key, mru_value));
                }
            }
            None
        } else {
            // Standard LRU behavior
            if let Some((key, value)) = self.cache.pop_lru() {
                self.stats.evictions += 1;
                Some((key, value))
            } else {
                None
            }
        }
    }
}
