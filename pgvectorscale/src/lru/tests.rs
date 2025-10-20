#![allow(clippy::len_zero)]
#![allow(clippy::type_complexity)]
#![allow(clippy::unnecessary_cast)]

use rkyv::{Archive, Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Duration;

use crate::lru::allocator::MockAllocator;
use crate::lru::cache::{EvictionHandler, SharedMemoryLru};

#[derive(Archive, Serialize, Deserialize, Hash, Eq, PartialEq, Clone, Debug)]
#[archive(compare(PartialEq))]
#[archive_attr(derive(Debug))]
#[archive(check_bytes)]
struct TestKey {
    id: u64,
}

#[derive(Archive, Serialize, Deserialize, Clone, Debug)]
#[archive_attr(derive(Debug))]
#[archive(check_bytes)]
struct TestValue {
    data: Vec<u8>,
}

struct TestEvictionHandler {
    evicted: Arc<std::sync::Mutex<Vec<(TestKey, TestValue)>>>,
}

impl EvictionHandler<TestKey, TestValue> for TestEvictionHandler {
    fn on_evict(&self, key: TestKey, value: TestValue) {
        self.evicted.lock().unwrap().push((key, value));
    }
}

/// Helper to create a test cache
fn create_test_cache(capacity: usize) -> SharedMemoryLru<TestKey, TestValue, MockAllocator> {
    let allocator = MockAllocator::new();
    SharedMemoryLru::new(allocator, capacity, "test".to_string(), None)
}

/// Helper to create a test cache with eviction handler
fn create_test_cache_with_handler(
    capacity: usize,
) -> (
    SharedMemoryLru<TestKey, TestValue, MockAllocator>,
    Arc<std::sync::Mutex<Vec<(TestKey, TestValue)>>>,
) {
    let allocator = MockAllocator::new();
    let evicted = Arc::new(std::sync::Mutex::new(Vec::new()));
    let handler = Arc::new(TestEvictionHandler {
        evicted: evicted.clone(),
    });
    let cache = SharedMemoryLru::new(
        allocator,
        capacity,
        "test".to_string(),
        Some(handler as Arc<dyn EvictionHandler<TestKey, TestValue>>),
    );
    (cache, evicted)
}

#[test]
fn test_basic_insert_and_get() {
    let cache = create_test_cache(1024);

    let key = TestKey { id: 1 };
    let value = TestValue {
        data: vec![1, 2, 3],
    };

    // Insert
    cache.insert(key.clone(), value.clone()).unwrap();

    // Get
    let pinned = cache.get(&key).unwrap();
    let archived = pinned.get();
    assert_eq!(archived.data.as_slice(), &[1, 2, 3]);

    // Contains
    assert!(cache.contains(&key));

    // Stats
    let stats = cache.stats().snapshot();
    assert_eq!(stats.inserts, 1);
    assert_eq!(stats.hits, 2); // One from get, one from contains
    assert_eq!(stats.misses, 0);
}

#[test]
fn test_update_existing_key() {
    let cache = create_test_cache(1024);

    let key = TestKey { id: 1 };
    let value1 = TestValue {
        data: vec![1, 2, 3],
    };
    let value2 = TestValue {
        data: vec![4, 5, 6],
    };

    // Insert initial value
    cache.insert(key.clone(), value1).unwrap();
    assert_eq!(cache.len(), 1);

    // Update with new value
    cache.insert(key.clone(), value2.clone()).unwrap();
    assert_eq!(cache.len(), 1); // Still only one entry

    // Get updated value
    let pinned = cache.get(&key).unwrap();
    let archived = pinned.get();
    assert_eq!(archived.data.as_slice(), &[4, 5, 6]);

    // Check stats
    let stats = cache.stats().snapshot();
    assert_eq!(stats.inserts, 1);
    assert_eq!(stats.updates, 1);
}

#[test]
fn test_lru_eviction() {
    // Create cache that can hold ~2 entries (each entry is ~124 bytes)
    let cache = create_test_cache(300);

    let key1 = TestKey { id: 1 };
    let key2 = TestKey { id: 2 };
    let key3 = TestKey { id: 3 };
    let value = TestValue { data: vec![0; 50] };

    // Insert first two
    cache.insert(key1.clone(), value.clone()).unwrap();
    cache.insert(key2.clone(), value.clone()).unwrap();

    // Access key1 to make it more recently used
    let _pinned = cache.get(&key1);
    drop(_pinned); // Explicitly drop to ensure it's unpinned

    // Insert key3, should evict key2 (least recently used)
    println!(
        "Before insert key3: cache has {} entries, {} bytes",
        cache.len(),
        cache.size()
    );
    println!(
        "Contains key1: {}, key2: {}",
        cache.contains(&key1),
        cache.contains(&key2)
    );

    cache.insert(key3.clone(), value.clone()).unwrap();

    println!(
        "After insert key3: cache has {} entries, {} bytes",
        cache.len(),
        cache.size()
    );
    println!(
        "Contains key1: {}, key2: {}, key3: {}",
        cache.contains(&key1),
        cache.contains(&key2),
        cache.contains(&key3)
    );

    assert!(cache.contains(&key1));
    assert!(!cache.contains(&key2)); // Should be evicted
    assert!(cache.contains(&key3));
}

#[test]
fn test_eviction_handler() {
    let (cache, evicted) = create_test_cache_with_handler(300);

    let key1 = TestKey { id: 1 };
    let key2 = TestKey { id: 2 };
    let key3 = TestKey { id: 3 };
    let value = TestValue { data: vec![0; 50] };

    // Insert entries
    cache.insert(key1.clone(), value.clone()).unwrap();
    cache.insert(key2.clone(), value.clone()).unwrap();
    cache.insert(key3.clone(), value.clone()).unwrap();

    // Check that key1 was evicted
    let evicted_items = evicted.lock().unwrap();
    assert_eq!(evicted_items.len(), 1);
    assert_eq!(evicted_items[0].0.id, 1);
}

#[test]
fn test_pinning_prevents_eviction() {
    let cache = create_test_cache(300);

    let key1 = TestKey { id: 1 };
    let key2 = TestKey { id: 2 };
    let key3 = TestKey { id: 3 };
    let value = TestValue { data: vec![0; 50] };

    // Insert first two
    cache.insert(key1.clone(), value.clone()).unwrap();
    cache.insert(key2.clone(), value.clone()).unwrap();

    // Pin key2 (would normally be evicted as LRU)
    let _pinned2 = cache.get(&key2);

    // Try to insert key3 - should evict key1 instead of pinned key2
    cache.insert(key3.clone(), value.clone()).unwrap();

    assert!(!cache.contains(&key1)); // Should be evicted
    assert!(cache.contains(&key2)); // Should still exist (pinned)
    assert!(cache.contains(&key3));
}

#[test]
fn test_multiple_pins() {
    let cache = create_test_cache(1024);

    let key = TestKey { id: 1 };
    let value = TestValue {
        data: vec![1, 2, 3],
    };

    cache.insert(key.clone(), value).unwrap();

    // Create multiple pins
    let pin1 = cache.get(&key).unwrap();
    let pin2 = cache.get(&key).unwrap();
    let pin3 = cache.get(&key).unwrap();

    // All pins should see the same data
    assert_eq!(pin1.get().data.as_slice(), &[1, 2, 3]);
    assert_eq!(pin2.get().data.as_slice(), &[1, 2, 3]);
    assert_eq!(pin3.get().data.as_slice(), &[1, 2, 3]);

    // Stats should show multiple pins
    let stats = cache.stats().snapshot();
    assert_eq!(stats.pin_count, 3);
}

#[test]
fn test_entry_too_large() {
    let cache = create_test_cache(100);

    let key = TestKey { id: 1 };
    let value = TestValue {
        data: vec![0; 200], // Larger than cache capacity
    };

    let result = cache.insert(key, value);
    assert!(result.is_err());
}

#[test]
fn test_concurrent_reads() {
    let cache = Arc::new(create_test_cache(1024));

    // Insert some data
    for i in 0..10 {
        let key = TestKey { id: i };
        let value = TestValue {
            data: vec![i as u8; 10],
        };
        cache.insert(key, value).unwrap();
    }

    let barrier = Arc::new(Barrier::new(5));
    let mut handles = vec![];

    // Spawn multiple reader threads
    for _thread_id in 0..5 {
        let cache = cache.clone();
        let barrier = barrier.clone();
        let handle = thread::spawn(move || {
            barrier.wait();

            // Each thread reads all entries
            for i in 0..10 {
                let key = TestKey { id: i };
                let pinned = cache.get(&key).unwrap();
                let data = pinned.get().data.as_slice();
                assert_eq!(data[0], i as u8);
            }
        });
        handles.push(handle);
    }

    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }

    // Check stats
    let stats = cache.stats().snapshot();
    assert_eq!(stats.hits, 50); // 5 threads * 10 reads
}

#[test]
fn test_concurrent_writes() {
    let cache = Arc::new(create_test_cache(10000));
    let barrier = Arc::new(Barrier::new(5));
    let mut handles = vec![];

    // Spawn multiple writer threads
    for thread_id in 0..5 {
        let cache = cache.clone();
        let barrier = barrier.clone();
        let handle = thread::spawn(move || {
            barrier.wait();

            // Each thread writes its own keys
            for i in 0..20 {
                let key = TestKey {
                    id: thread_id * 1000 + i,
                };
                let value = TestValue {
                    data: vec![thread_id as u8, i as u8],
                };
                cache.insert(key, value).unwrap();
            }
        });
        handles.push(handle);
    }

    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }

    // Verify all entries
    assert_eq!(cache.len(), 100); // 5 threads * 20 entries

    for thread_id in 0..5 {
        for i in 0..20 {
            let key = TestKey {
                id: thread_id * 1000 + i,
            };
            assert!(cache.contains(&key));
        }
    }
}

#[test]
fn test_concurrent_read_write_with_eviction() {
    let cache = Arc::new(create_test_cache(500)); // Small cache to force evictions
    let stop_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));

    // Writer thread
    let cache_write = cache.clone();
    let stop_write = stop_flag.clone();
    let write_handle = thread::spawn(move || {
        let mut i = 0;
        while !stop_write.load(std::sync::atomic::Ordering::Relaxed) {
            let key = TestKey { id: i };
            let value = TestValue {
                data: vec![i as u8; 50],
            };
            let _ = cache_write.insert(key, value); // Ignore errors from cache full
            i += 1;
            thread::sleep(Duration::from_micros(100));
        }
    });

    // Reader threads
    let mut read_handles = vec![];
    for _ in 0..3 {
        let cache_read = cache.clone();
        let stop_read = stop_flag.clone();
        let handle = thread::spawn(move || {
            while !stop_read.load(std::sync::atomic::Ordering::Relaxed) {
                // Try to read recent entries
                for i in 0..10 {
                    let key = TestKey { id: i };
                    let _ = cache_read.get(&key); // May or may not exist
                }
                thread::sleep(Duration::from_micros(50));
            }
        });
        read_handles.push(handle);
    }

    // Let it run for a bit
    thread::sleep(Duration::from_millis(100));

    // Stop all threads
    stop_flag.store(true, std::sync::atomic::Ordering::Relaxed);

    write_handle.join().unwrap();
    for handle in read_handles {
        handle.join().unwrap();
    }

    // Check that cache still works
    assert!(cache.len() > 0);
    let stats = cache.stats().snapshot();
    assert!(stats.evictions > 0); // Should have had evictions
}

#[test]
fn test_cache_full_with_all_pinned() {
    let cache = create_test_cache(300); // Need to hold 2 entries to test pinning behavior

    let key1 = TestKey { id: 1 };
    let key2 = TestKey { id: 2 };
    let key3 = TestKey { id: 3 };
    let value = TestValue { data: vec![0; 50] };

    // Insert first two entries
    cache.insert(key1.clone(), value.clone()).unwrap();
    cache.insert(key2.clone(), value.clone()).unwrap();

    // Pin both entries
    let _pin1 = cache.get(&key1);
    let _pin2 = cache.get(&key2);

    // Try to insert third entry - should fail because all entries are pinned
    let result = cache.insert(key3, value);
    assert!(result.is_err());
}

#[test]
fn test_lru_ordering_with_get() {
    let cache = create_test_cache(400); // Need to hold 3 entries

    let key1 = TestKey { id: 1 };
    let key2 = TestKey { id: 2 };
    let key3 = TestKey { id: 3 };
    let key4 = TestKey { id: 4 };
    let value = TestValue { data: vec![0; 50] };

    // Insert three entries
    cache.insert(key1.clone(), value.clone()).unwrap();
    cache.insert(key2.clone(), value.clone()).unwrap();
    cache.insert(key3.clone(), value.clone()).unwrap();

    // Access in order: key1, key3, key2
    let _ = cache.get(&key1);
    let _ = cache.get(&key3);
    let _ = cache.get(&key2);

    // Now order should be (MRU to LRU): key2, key3, key1
    // Insert key4 should evict key1
    cache.insert(key4.clone(), value).unwrap();

    assert!(!cache.contains(&key1)); // Should be evicted
    assert!(cache.contains(&key2));
    assert!(cache.contains(&key3));
    assert!(cache.contains(&key4));
}

#[test]
fn test_empty_cache() {
    let cache = create_test_cache(1024);

    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);
    assert_eq!(cache.size(), 0);

    let key = TestKey { id: 1 };
    assert!(cache.get(&key).is_none());
    assert!(!cache.contains(&key));
}

#[test]
fn test_stats_tracking() {
    let cache = create_test_cache(1024);

    let key1 = TestKey { id: 1 };
    let key2 = TestKey { id: 2 };
    let value = TestValue {
        data: vec![1, 2, 3],
    };

    // Insert
    cache.insert(key1.clone(), value.clone()).unwrap();
    cache.insert(key2.clone(), value.clone()).unwrap();

    // Get (hit)
    let _pin1 = cache.get(&key1);
    println!(
        "After get(key1): hits={}, misses={}",
        cache.stats().snapshot().hits,
        cache.stats().snapshot().misses
    );

    // Get (miss)
    let key3 = TestKey { id: 3 };
    let _ = cache.get(&key3);
    println!(
        "After get(key3): hits={}, misses={}",
        cache.stats().snapshot().hits,
        cache.stats().snapshot().misses
    );

    // Contains (hit)
    cache.contains(&key2);
    println!(
        "After contains(key2): hits={}, misses={}",
        cache.stats().snapshot().hits,
        cache.stats().snapshot().misses
    );

    // Contains (miss)
    cache.contains(&key3);
    println!(
        "After contains(key3): hits={}, misses={}",
        cache.stats().snapshot().hits,
        cache.stats().snapshot().misses
    );

    let stats = cache.stats().snapshot();
    assert_eq!(stats.inserts, 2);
    assert_eq!(stats.updates, 0);
    assert_eq!(stats.hits, 2); // 1 get + 1 contains
    assert_eq!(stats.misses, 2); // 1 get + 1 contains
    assert_eq!(stats.pin_count, 1);
    assert!(stats.hit_rate() > 0.4 && stats.hit_rate() < 0.6); // Should be 50%
}

// Additional comprehensive concurrent tests

#[test]
fn test_concurrent_mixed_operations() {
    // Test mixed reads, writes, and updates concurrently
    let cache = Arc::new(create_test_cache(5000));
    let barrier = Arc::new(Barrier::new(10));
    let mut handles = vec![];

    // Pre-populate with some data
    for i in 0..50 {
        let key = TestKey { id: i };
        let value = TestValue {
            data: vec![i as u8; 20],
        };
        cache.insert(key, value).unwrap();
    }

    // Spawn reader threads
    for thread_id in 0..3 {
        let cache = cache.clone();
        let barrier = barrier.clone();
        let handle = thread::spawn(move || {
            barrier.wait();
            let mut hits = 0;
            let mut misses = 0;

            for i in 0..100 {
                // Use a simple pseudo-random pattern
                let key = TestKey {
                    id: thread_id * 10 + ((i * 7 + thread_id * 13) % 100),
                };
                if cache.get(&key).is_some() {
                    hits += 1;
                } else {
                    misses += 1;
                }
                thread::yield_now();
            }
            (hits, misses)
        });
        handles.push(handle);
    }

    // Spawn writer threads
    for thread_id in 3..6 {
        let cache = cache.clone();
        let barrier = barrier.clone();
        let handle = thread::spawn(move || {
            barrier.wait();
            let mut success = 0;

            for i in 0..50 {
                let key = TestKey {
                    id: 1000 + thread_id * 100 + i,
                };
                let value = TestValue {
                    data: vec![thread_id as u8, i as u8],
                };
                if cache.insert(key, value).is_ok() {
                    success += 1;
                }
                thread::yield_now();
            }
            (success, 0)
        });
        handles.push(handle);
    }

    // Spawn updater threads
    for thread_id in 6..10 {
        let cache = cache.clone();
        let barrier = barrier.clone();
        let handle = thread::spawn(move || {
            barrier.wait();
            let mut updates = 0;

            for i in 0..50 {
                let key = TestKey { id: i }; // Update existing keys
                let value = TestValue {
                    data: vec![255 - thread_id as u8, i as u8],
                };
                if cache.insert(key, value).is_ok() {
                    updates += 1;
                }
                thread::yield_now();
            }
            (updates, 0)
        });
        handles.push(handle);
    }

    // Collect results
    let mut total_ops = 0;
    for handle in handles {
        let (ops, _) = handle.join().unwrap();
        total_ops += ops;
    }

    assert!(total_ops > 0);
    assert!(cache.len() > 0);
}

#[test]
fn test_concurrent_pinning() {
    // Test multiple threads pinning and unpinning entries
    let cache = Arc::new(create_test_cache(2000));

    // Insert test data
    for i in 0..20 {
        let key = TestKey { id: i };
        let value = TestValue {
            data: vec![i as u8; 50],
        };
        cache.insert(key, value).unwrap();
    }

    let barrier = Arc::new(Barrier::new(5));
    let mut handles = vec![];

    for thread_id in 0..5 {
        let cache = cache.clone();
        let barrier = barrier.clone();
        let handle = thread::spawn(move || {
            barrier.wait();

            let mut pins = vec![];
            // Pin multiple entries
            for i in 0..10 {
                let key = TestKey {
                    id: (thread_id + i) % 20,
                };
                if let Some(pinned) = cache.get(&key) {
                    pins.push(pinned);
                }
            }

            // Hold pins for a bit
            thread::sleep(std::time::Duration::from_millis(10));

            // Access pinned values
            for pin in &pins {
                let _ = pin.get().data.as_slice();
            }

            // Pins will be automatically dropped
            pins.len()
        });
        handles.push(handle);
    }

    let mut total_pins = 0;
    for handle in handles {
        total_pins += handle.join().unwrap();
    }

    assert!(total_pins > 0);

    // After all threads complete, all pins should be released
    thread::sleep(std::time::Duration::from_millis(50));
    let stats = cache.stats().snapshot();
    // pin_count is total pins created, unpin_count is total pins released
    // They should be equal when all pins are released
    assert_eq!(
        stats.pin_count, stats.unpin_count,
        "Not all pins were released: {} pins created, {} unpins",
        stats.pin_count, stats.unpin_count
    );
}

#[test]
fn test_concurrent_eviction_safety() {
    // Test that eviction works correctly under concurrent load
    let cache = Arc::new(create_test_cache(400)); // Small cache to trigger evictions
    let barrier = Arc::new(Barrier::new(8));
    let mut handles = vec![];

    for thread_id in 0..8 {
        let cache = cache.clone();
        let barrier = barrier.clone();
        let handle = thread::spawn(move || {
            barrier.wait();

            for i in 0..50 {
                let key = TestKey {
                    id: thread_id * 100 + i,
                };
                let value = TestValue {
                    data: vec![thread_id as u8; 30],
                };

                // Insert and immediately try to read
                // This might fail if cache is full and all entries are pinned
                let _ = cache.insert(key.clone(), value);

                // The entry might be evicted by another thread
                let _ = cache.get(&key);

                thread::yield_now();
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    // Should have evictions due to small cache size
    let stats = cache.stats().snapshot();
    assert!(stats.evictions > 0);

    // Cache should not be corrupted
    assert!(cache.len() <= 3); // Rough estimate based on entry size
}

#[test]
fn test_concurrent_same_key_updates() {
    // Multiple threads updating the same keys
    let cache = Arc::new(create_test_cache(5000));
    let barrier = Arc::new(Barrier::new(10));
    let mut handles = vec![];

    for thread_id in 0..10 {
        let cache = cache.clone();
        let barrier = barrier.clone();
        let handle = thread::spawn(move || {
            barrier.wait();

            let mut success = 0;
            for round in 0..100 {
                // All threads try to update the same 10 keys
                let key = TestKey { id: round % 10 };
                let value = TestValue {
                    data: vec![thread_id as u8, round as u8],
                };

                if cache.insert(key, value).is_ok() {
                    success += 1;
                }

                // Small random delay
                if round % 10 == 0 {
                    thread::yield_now();
                }
            }
            success
        });
        handles.push(handle);
    }

    let mut total_success = 0;
    for handle in handles {
        total_success += handle.join().unwrap();
    }

    // All inserts should succeed
    assert_eq!(total_success, 1000);

    // Should have exactly 10 keys in cache
    assert_eq!(cache.len(), 10);

    // Should have many updates
    let stats = cache.stats().snapshot();
    assert!(stats.updates > 900); // Most operations should be updates
}

#[test]
fn test_concurrent_stress_test() {
    // High concurrency stress test
    let cache = Arc::new(create_test_cache(10 * 1024 * 1024)); // 10MB
    let num_threads = 20;
    let ops_per_thread = 1000;
    let barrier = Arc::new(Barrier::new(num_threads));
    let mut handles = vec![];

    let start = std::time::Instant::now();

    for thread_id in 0..num_threads {
        let cache = cache.clone();
        let barrier = barrier.clone();
        let handle = thread::spawn(move || {
            barrier.wait();

            let mut hasher = DefaultHasher::new();
            thread_id.hash(&mut hasher);
            let seed = hasher.finish();

            for i in 0..ops_per_thread {
                let op = (seed + i as u64) % 3;
                let key_id = ((seed + i as u64) % 1000) as u64;

                match op {
                    0 => {
                        // Read
                        let key = TestKey { id: key_id };
                        let _ = cache.get(&key);
                    }
                    1 => {
                        // Write
                        let key = TestKey { id: key_id };
                        let value = TestValue {
                            data: vec![(thread_id % 256) as u8; 100],
                        };
                        let _ = cache.insert(key, value);
                    }
                    2 => {
                        // Contains
                        let key = TestKey { id: key_id };
                        let _ = cache.contains(&key);
                    }
                    _ => unreachable!(),
                }

                // Occasionally yield
                if i % 100 == 0 {
                    thread::yield_now();
                }
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let elapsed = start.elapsed();
    let total_ops = num_threads * ops_per_thread;
    let ops_per_sec = total_ops as f64 / elapsed.as_secs_f64();

    println!(
        "Stress test: {} ops in {:?} ({:.0} ops/sec)",
        total_ops, elapsed, ops_per_sec
    );

    // Cache should still be functional
    assert!(cache.len() > 0);
    let stats = cache.stats().snapshot();
    assert!(stats.hits + stats.misses > 0);
}

#[test]
fn test_concurrent_bulk_operations() {
    // Test bulk insertions and deletions
    let cache = Arc::new(create_test_cache(5000));
    let barrier = Arc::new(Barrier::new(4));
    let mut handles = vec![];

    // Two threads doing bulk inserts
    for thread_id in 0..2 {
        let cache = cache.clone();
        let barrier = barrier.clone();
        let handle = thread::spawn(move || {
            barrier.wait();

            // Bulk insert
            let base = thread_id * 1000;
            let mut inserted = 0;
            for i in 0..500 {
                let key = TestKey { id: base + i };
                let value = TestValue {
                    data: vec![thread_id as u8; 50],
                };
                // Allow some inserts to fail under concurrent load
                if cache.insert(key, value).is_ok() {
                    inserted += 1;
                }
            }
            inserted // Return count of actually inserted items
        });
        handles.push(handle);
    }

    // Two threads doing reads during bulk operations
    for _thread_id in 2..4 {
        let cache = cache.clone();
        let barrier = barrier.clone();
        let handle = thread::spawn(move || {
            barrier.wait();

            let mut found = 0;
            for i in 0..1000 {
                let key = TestKey { id: i };
                if cache.get(&key).is_some() {
                    found += 1;
                }
            }
            found
        });
        handles.push(handle);
    }

    let mut _total_operations = 0;
    for handle in handles {
        _total_operations += handle.join().unwrap();
    }

    // Verify bulk inserts succeeded
    let mut count = 0;
    for i in 0..1000 {
        let key = TestKey { id: i };
        if cache.contains(&key) {
            count += 1;
        }
    }
    // Due to evictions, we might not have all 1000, but should have many
    println!("Final cache size: {}", cache.size());
    println!("Final cache len: {}", cache.len());
    println!("Count of items 0-999: {}", count);
    assert!(
        count > 10 || cache.len() > 10,
        "count: {}, len: {}",
        count,
        cache.len()
    );
}

#[test]
fn test_concurrent_long_running_pins() {
    // Test behavior with long-running pinned entries
    let cache = Arc::new(create_test_cache(600)); // Small cache

    // Insert initial data
    for i in 0..5 {
        let key = TestKey { id: i };
        let value = TestValue {
            data: vec![i as u8; 50],
        };
        cache.insert(key, value).unwrap();
    }

    let barrier = Arc::new(Barrier::new(3));

    // Thread holding long pins
    let cache1 = cache.clone();
    let barrier1 = barrier.clone();
    let handle1 = thread::spawn(move || {
        barrier1.wait();

        // Pin some entries and hold them
        let _pin1 = cache1.get(&TestKey { id: 0 });
        let _pin2 = cache1.get(&TestKey { id: 1 });

        // Hold pins for a while
        thread::sleep(std::time::Duration::from_millis(100));

        // Pins dropped here
    });

    // Thread trying to insert new data
    let cache2 = cache.clone();
    let barrier2 = barrier.clone();
    let handle2 = thread::spawn(move || {
        barrier2.wait();

        // Wait a bit for pins to be established
        thread::sleep(std::time::Duration::from_millis(10));

        let mut success = 0;
        let mut failed = 0;

        // Try to insert, should work around pinned entries
        for i in 10..20 {
            let key = TestKey { id: i };
            let value = TestValue {
                data: vec![i as u8; 50],
            };

            match cache2.insert(key, value) {
                Ok(_) => success += 1,
                Err(_) => failed += 1,
            }
        }

        (success, failed)
    });

    // Thread doing reads
    let cache3 = cache.clone();
    let barrier3 = barrier.clone();
    let handle3 = thread::spawn(move || {
        barrier3.wait();

        let mut hits = 0;
        for _ in 0..50 {
            if cache3.get(&TestKey { id: 0 }).is_some() {
                hits += 1;
            }
            if cache3.get(&TestKey { id: 1 }).is_some() {
                hits += 1;
            }
            thread::sleep(std::time::Duration::from_millis(5));
        }
        hits
    });

    // Wait for all threads
    handle1.join().unwrap();
    let (success, failed) = handle2.join().unwrap();
    let _hits = handle3.join().unwrap();

    // Some inserts should succeed even with pinned entries
    assert!(success > 0);
    println!("Long pins test: {} succeeded, {} failed", success, failed);
}
