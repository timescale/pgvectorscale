use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::sync::Arc;
use std::thread;

use rkyv::{Archive, Deserialize, Serialize};
use vectorscale::lru::{MockAllocator, SharedMemoryLru};

#[derive(Archive, Serialize, Deserialize, Hash, Eq, PartialEq, Clone, Debug)]
#[archive(compare(PartialEq))]
#[archive_attr(derive(Debug))]
struct BenchKey {
    id: u64,
}

#[derive(Archive, Serialize, Deserialize, Clone, Debug)]
#[archive_attr(derive(Debug))]
struct BenchValue {
    data: Vec<u8>,
}

fn bench_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("lru_insert");

    for size in &[10, 100, 1000, 10000] {
        group.throughput(Throughput::Bytes(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let allocator = MockAllocator::new();
            let cache = SharedMemoryLru::<BenchKey, BenchValue, MockAllocator>::new(
                allocator,
                10 * 1024 * 1024, // 10MB cache
                "bench".to_string(),
                None,
            );

            let value = BenchValue {
                data: vec![0u8; size],
            };

            let mut counter = 0u64;
            b.iter(|| {
                let key = BenchKey { id: counter };
                counter += 1;
                cache
                    .insert(black_box(key), black_box(value.clone()))
                    .unwrap();
            });
        });
    }
    group.finish();
}

fn bench_get(c: &mut Criterion) {
    let mut group = c.benchmark_group("lru_get");

    group.bench_function("hit", |b| {
        let allocator = MockAllocator::new();
        let cache = SharedMemoryLru::<BenchKey, BenchValue, MockAllocator>::new(
            allocator,
            1024 * 1024, // 1MB cache
            "bench".to_string(),
            None,
        );

        // Pre-populate cache
        for i in 0..100 {
            let key = BenchKey { id: i };
            let value = BenchValue {
                data: vec![i as u8; 100],
            };
            cache.insert(key, value).unwrap();
        }

        b.iter(|| {
            let key = BenchKey { id: 50 }; // Always hit
            let _pinned = cache.get(black_box(&key));
        });
    });

    group.bench_function("miss", |b| {
        let allocator = MockAllocator::new();
        let cache = SharedMemoryLru::<BenchKey, BenchValue, MockAllocator>::new(
            allocator,
            1024 * 1024, // 1MB cache
            "bench".to_string(),
            None,
        );

        // Pre-populate cache
        for i in 0..100 {
            let key = BenchKey { id: i };
            let value = BenchValue {
                data: vec![i as u8; 100],
            };
            cache.insert(key, value).unwrap();
        }

        b.iter(|| {
            let key = BenchKey { id: 1000 }; // Always miss
            let _pinned = cache.get(black_box(&key));
        });
    });

    group.finish();
}

fn bench_eviction(c: &mut Criterion) {
    c.bench_function("lru_eviction", |b| {
        let allocator = MockAllocator::new();
        let cache = SharedMemoryLru::<BenchKey, BenchValue, MockAllocator>::new(
            allocator,
            10 * 1024, // Small 10KB cache to trigger evictions
            "bench".to_string(),
            None,
        );

        let value = BenchValue {
            data: vec![0u8; 1000], // 1KB values
        };

        let mut counter = 0u64;
        b.iter(|| {
            let key = BenchKey { id: counter };
            counter += 1;
            cache
                .insert(black_box(key), black_box(value.clone()))
                .unwrap();
        });
    });
}

fn bench_concurrent(c: &mut Criterion) {
    let mut group = c.benchmark_group("lru_concurrent");

    for num_threads in &[2, 4, 8] {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_threads),
            num_threads,
            |b, &num_threads| {
                let allocator = MockAllocator::new();
                let cache = Arc::new(SharedMemoryLru::<BenchKey, BenchValue, MockAllocator>::new(
                    allocator,
                    10 * 1024 * 1024, // 10MB cache
                    "bench".to_string(),
                    None,
                ));

                // Pre-populate cache
                for i in 0..1000 {
                    let key = BenchKey { id: i };
                    let value = BenchValue {
                        data: vec![i as u8; 100],
                    };
                    cache.insert(key, value).unwrap();
                }

                b.iter(|| {
                    let mut handles = vec![];
                    for _ in 0..num_threads {
                        let cache = cache.clone();
                        let handle = thread::spawn(move || {
                            for i in 0..100 {
                                let key = BenchKey { id: i % 1000 };
                                let _pinned = cache.get(&key);
                            }
                        });
                        handles.push(handle);
                    }
                    for handle in handles {
                        handle.join().unwrap();
                    }
                });
            },
        );
    }

    group.finish();
}

fn bench_mixed_workload(c: &mut Criterion) {
    c.bench_function("lru_mixed_80_20", |b| {
        let allocator = MockAllocator::new();
        let cache = SharedMemoryLru::<BenchKey, BenchValue, MockAllocator>::new(
            allocator,
            1024 * 1024, // 1MB cache
            "bench".to_string(),
            None,
        );

        // Pre-populate cache
        for i in 0..100 {
            let key = BenchKey { id: i };
            let value = BenchValue {
                data: vec![i as u8; 100],
            };
            cache.insert(key, value).unwrap();
        }

        let value = BenchValue {
            data: vec![0u8; 100],
        };

        let mut counter = 100u64;
        let mut ops = 0u64;
        b.iter(|| {
            ops += 1;
            if ops.is_multiple_of(5) {
                // 20% writes
                let key = BenchKey { id: counter };
                counter += 1;
                cache
                    .insert(black_box(key), black_box(value.clone()))
                    .unwrap();
            } else {
                // 80% reads
                let key = BenchKey { id: ops % 100 };
                let _pinned = cache.get(black_box(&key));
            }
        });
    });
}

criterion_group!(
    benches,
    bench_insert,
    bench_get,
    bench_eviction,
    bench_concurrent,
    bench_mixed_workload
);
criterion_main!(benches);
