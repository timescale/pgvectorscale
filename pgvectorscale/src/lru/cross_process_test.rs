//! Cross-process testing for shared memory LRU cache
//!
//! This module provides tests to verify that the LRU cache works across
//! PostgreSQL backend processes.

use super::shared_memory;
use pgrx::prelude::*;

/// SQL function to insert into shared cache
#[pg_extern]
fn pgvs_shared_lru_insert(key: i32, value: String) -> bool {
    unsafe {
        if let Some(cache) = shared_memory::get_quantized_vector_cache() {
            match cache.insert(key, value) {
                Ok(_) => {
                    info!("Inserted key {} into shared LRU", key);
                    true
                }
                Err(e) => {
                    warning!("Failed to insert: {}", e);
                    false
                }
            }
        } else {
            warning!("Shared cache not available");
            false
        }
    }
}

/// SQL function to get from shared cache
#[pg_extern]
fn pgvs_shared_lru_get(key: i32) -> Option<String> {
    unsafe {
        if let Some(cache) = shared_memory::get_quantized_vector_cache() {
            let result = cache.get::<i32, String>(&key);
            if result.is_some() {
                info!("Found key {} in shared LRU", key);
            } else {
                info!("Key {} not found in shared LRU", key);
            }
            result
        } else {
            warning!("Shared cache not available");
            None
        }
    }
}

/// SQL function to get cache statistics
#[pg_extern]
fn pgvs_shared_lru_stats() -> String {
    if let Some(cache) = shared_memory::get_quantized_vector_cache() {
        let stats = cache.stats();
        format!(
            "entries: {}, memory_used: {} bytes, memory_total: {} bytes",
            stats.entry_count, stats.memory_used, stats.memory_total
        )
    } else {
        "Shared cache not available".to_string()
    }
}

/// SQL function to test cross-process visibility
///
/// To test this:
/// 1. Open two PostgreSQL sessions
/// 2. In session 1: SELECT pgvs_shared_lru_insert(1, 'test_value');
/// 3. In session 2: SELECT pgvs_shared_lru_get(1); -- Should return 'test_value'
#[pg_extern]
fn pgvs_test_cross_process() -> String {
    unsafe {
        if !shared_memory::is_initialized() {
            return "Shared memory not initialized".to_string();
        }

        let mut results = Vec::new();

        // Insert a test value
        if let Some(cache) = shared_memory::get_quantized_vector_cache() {
            let test_key = 999;
            let test_value = format!("Process PID: {}", std::process::id());

            match cache.insert(test_key, test_value.clone()) {
                Ok(_) => results.push(format!("Inserted: key={}, value={}", test_key, test_value)),
                Err(e) => results.push(format!("Insert failed: {}", e)),
            }

            // Try to read it back
            if let Some(retrieved) = cache.get::<i32, String>(&test_key) {
                results.push(format!("Retrieved: {}", retrieved));
            } else {
                results.push("Failed to retrieve value".to_string());
            }

            // Get cache stats
            let stats = cache.stats();
            results.push(format!(
                "Cache stats: {} entries, {}/{} bytes used",
                stats.entry_count, stats.memory_used, stats.memory_total
            ));
        } else {
            results.push("Cache not available".to_string());
        }

        results.join("\n")
    }
}

/// Integration test using pg_test
#[cfg(any(test, feature = "pg_test"))]
mod tests {
    use super::*;

    #[pg_test]
    fn test_shared_lru_basic() {
        unsafe {
            // Initialize shared memory (normally done during startup)
            shared_memory::init_shared_memory();

            // Get the cache
            let cache = shared_memory::get_quantized_vector_cache()
                .expect("Shared cache should be available");

            // Insert some data
            cache
                .insert(1i32, "value1".to_string())
                .expect("Insert should succeed");
            cache
                .insert(2i32, "value2".to_string())
                .expect("Insert should succeed");

            // Retrieve data
            assert_eq!(cache.get::<i32, String>(&1), Some("value1".to_string()));
            assert_eq!(cache.get::<i32, String>(&2), Some("value2".to_string()));

            // Check stats
            let stats = cache.stats();
            assert_eq!(stats.entry_count, 2);
            assert!(stats.memory_used > 0);
        }
    }

    #[pg_test]
    fn test_shared_lru_eviction() {
        unsafe {
            shared_memory::init_shared_memory();

            let cache = shared_memory::get_quantized_vector_cache()
                .expect("Shared cache should be available");

            // Fill cache to trigger eviction
            for i in 0..1000 {
                let key = i;
                let value = format!("value_{}", i);
                let _ = cache.insert(key, value);
            }

            // Check that we have entries (exact count depends on memory limit)
            let stats = cache.stats();
            assert!(stats.entry_count > 0);
            assert!(stats.memory_used > 0);
            assert!(stats.memory_used <= stats.memory_total);

            info!(
                "Cache has {} entries using {} bytes",
                stats.entry_count, stats.memory_used
            );
        }
    }
}
