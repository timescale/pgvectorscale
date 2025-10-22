use std::sync::atomic::{AtomicUsize, Ordering};

/// Statistics for cache operations
#[derive(Debug, Default)]
pub struct CacheStats {
    pub inserts: AtomicUsize,
    pub updates: AtomicUsize,
    pub hits: AtomicUsize,
    pub misses: AtomicUsize,
    pub evictions: AtomicUsize,
    pub pin_count: AtomicUsize,
    pub unpin_count: AtomicUsize,
}

impl CacheStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_insert(&self) {
        self.inserts.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_update(&self) {
        self.updates.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_hit(&self) {
        self.hits.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_miss(&self) {
        self.misses.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_eviction(&self) {
        self.evictions.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_pin(&self) {
        self.pin_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_unpin(&self) {
        self.unpin_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Get a snapshot of current statistics
    pub fn snapshot(&self) -> CacheStatsSnapshot {
        CacheStatsSnapshot {
            inserts: self.inserts.load(Ordering::Relaxed),
            updates: self.updates.load(Ordering::Relaxed),
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            pin_count: self.pin_count.load(Ordering::Relaxed),
            unpin_count: self.unpin_count.load(Ordering::Relaxed),
        }
    }

    /// Calculate hit rate
    pub fn hit_rate(&self) -> f64 {
        let hits = self.hits.load(Ordering::Relaxed) as f64;
        let misses = self.misses.load(Ordering::Relaxed) as f64;
        let total = hits + misses;
        if total > 0.0 {
            hits / total
        } else {
            0.0
        }
    }
}

/// Non-atomic snapshot of cache statistics
#[derive(Debug, Clone, Copy)]
pub struct CacheStatsSnapshot {
    pub inserts: usize,
    pub updates: usize,
    pub hits: usize,
    pub misses: usize,
    pub evictions: usize,
    pub pin_count: usize,
    pub unpin_count: usize,
}

impl CacheStatsSnapshot {
    pub fn hit_rate(&self) -> f64 {
        let total = (self.hits + self.misses) as f64;
        if total > 0.0 {
            self.hits as f64 / total
        } else {
            0.0
        }
    }
}
