//! Tests for PostgreSQL-native LRU cache
//!
//! These tests require a PostgreSQL environment to run properly
//! as they use pg_sys functions like palloc and LWLocks.

#[cfg(test)]
mod tests {
    use super::super::pg_lru::*;
    use rkyv::{Archive, Deserialize, Serialize};

    #[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Hash)]
    #[archive(compare(PartialEq))]
    #[archive_attr(derive(Debug))]
    struct TestKey {
        id: u64,
    }

    #[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq)]
    #[archive_attr(derive(Debug))]
    struct TestValue {
        data: Vec<u8>,
    }

    // Note: These tests would need to be run in a PostgreSQL extension context
    // with proper memory contexts set up. For now, they're disabled.

    #[test]
    #[ignore] // Requires PostgreSQL environment
    fn test_basic_insert_and_get() {
        unsafe {
            // Create cache with 1MB
            let cache = PgSharedLru::new(1024);

            let key = TestKey { id: 1 };
            let value = TestValue {
                data: vec![1, 2, 3, 4, 5],
            };

            // Insert
            cache
                .insert(key.clone(), value.clone())
                .expect("Failed to insert");

            // Get
            let retrieved = cache.get::<TestKey, TestValue>(&key);
            assert!(retrieved.is_some());
            assert_eq!(retrieved.unwrap(), value);
        }
    }

    #[test]
    #[ignore] // Requires PostgreSQL environment
    fn test_lru_ordering() {
        unsafe {
            // Small cache to force evictions
            let cache = PgSharedLru::new(10); // 10KB

            // Insert multiple items
            for i in 0..5 {
                let key = TestKey { id: i };
                let value = TestValue {
                    data: vec![i as u8; 100],
                };
                cache.insert(key, value).expect("Failed to insert");
            }

            // Access first item to make it most recently used
            let key0 = TestKey { id: 0 };
            assert!(cache.get::<TestKey, TestValue>(&key0).is_some());

            // Insert more items to trigger eviction
            for i in 5..10 {
                let key = TestKey { id: i };
                let value = TestValue {
                    data: vec![i as u8; 100],
                };
                cache.insert(key, value).expect("Failed to insert");
            }

            // Key 0 should still be present (was accessed recently)
            assert!(cache.get::<TestKey, TestValue>(&key0).is_some());

            // Key 1 should have been evicted (least recently used)
            let key1 = TestKey { id: 1 };
            assert!(cache.get::<TestKey, TestValue>(&key1).is_none());
        }
    }

    #[test]
    #[ignore] // Requires PostgreSQL environment
    fn test_eviction_when_full() {
        unsafe {
            // Very small cache
            let cache = PgSharedLru::new(1); // 1KB

            // Fill cache
            let key1 = TestKey { id: 1 };
            let value1 = TestValue {
                data: vec![1; 500], // 500 bytes
            };
            cache
                .insert(key1.clone(), value1.clone())
                .expect("Failed to insert");

            // Insert another item, should evict first
            let key2 = TestKey { id: 2 };
            let value2 = TestValue {
                data: vec![2; 500], // 500 bytes
            };
            cache
                .insert(key2.clone(), value2.clone())
                .expect("Failed to insert");

            // First item should be evicted
            assert!(cache.get::<TestKey, TestValue>(&key1).is_none());

            // Second item should be present
            assert!(cache.get::<TestKey, TestValue>(&key2).is_some());
        }
    }

    #[test]
    #[ignore] // Requires PostgreSQL environment
    fn test_stats() {
        unsafe {
            let cache = PgSharedLru::new(1024); // 1MB

            // Initial stats
            let stats = cache.stats();
            assert_eq!(stats.entry_count, 0);

            // Insert some items
            for i in 0..10 {
                let key = TestKey { id: i };
                let value = TestValue {
                    data: vec![i as u8; 100],
                };
                cache.insert(key, value).expect("Failed to insert");
            }

            // Check updated stats
            let stats = cache.stats();
            assert_eq!(stats.entry_count, 10);
            assert!(stats.memory_used > 0);
            assert!(stats.memory_used < stats.memory_total);
        }
    }

    #[test]
    #[ignore] // Requires PostgreSQL environment
    fn test_hash_collision_handling() {
        unsafe {
            let cache = PgSharedLru::new(1024);

            // Insert multiple items that might hash to same bucket
            for i in 0..100 {
                let key = TestKey { id: i * 1024 }; // Likely to have collisions
                let value = TestValue {
                    data: vec![i as u8; 10],
                };
                cache
                    .insert(key.clone(), value.clone())
                    .expect("Failed to insert");

                // Verify we can retrieve it
                let retrieved = cache.get::<TestKey, TestValue>(&key);
                assert!(retrieved.is_some());
                assert_eq!(retrieved.unwrap(), value);
            }
        }
    }
}
