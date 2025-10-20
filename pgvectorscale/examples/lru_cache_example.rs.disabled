/// Example demonstrating the use of SharedMemoryLru cache
/// This could be adapted for caching vector embeddings, database pages, or other data
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use rkyv::{Archive, Deserialize, Serialize};
use vectorscale::lru::{MockAllocator, SharedMemoryLru};

// Example: Caching vector embeddings for similarity search
#[derive(Archive, Serialize, Deserialize, Hash, Eq, PartialEq, Clone, Debug)]
#[archive(compare(PartialEq))]
#[archive_attr(derive(Debug))]
struct VectorId {
    index_id: u32,
    vector_id: u64,
}

#[derive(Archive, Serialize, Deserialize, Clone, Debug)]
#[archive_attr(derive(Debug))]
struct VectorEmbedding {
    dimensions: u32,
    data: Vec<f32>,
}

impl VectorEmbedding {
    fn new(dimensions: u32, seed: f32) -> Self {
        // Generate a mock embedding based on seed
        let data: Vec<f32> = (0..dimensions).map(|i| seed + (i as f32 * 0.1)).collect();
        Self { dimensions, data }
    }

    fn compute_similarity(&self, other: &Self) -> f32 {
        // Simple dot product similarity
        assert_eq!(self.dimensions, other.dimensions);
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .sum()
    }
}

fn main() {
    println!("=== LRU Cache Integration Example ===\n");

    // Create a cache with 50MB capacity
    let allocator = MockAllocator::new();
    let cache = Arc::new(
        SharedMemoryLru::<VectorId, VectorEmbedding, MockAllocator>::new(
            allocator,
            50 * 1024 * 1024, // 50MB
            "vector_cache".to_string(),
            None,
        ),
    );

    println!("Created cache with 50MB capacity\n");

    // Simulate loading vectors from a database
    println!("Loading vectors into cache...");
    let start = Instant::now();

    let dimensions = 768; // Common embedding size (e.g., BERT)
    let num_vectors = 1000;

    for i in 0..num_vectors {
        let id = VectorId {
            index_id: 1,
            vector_id: i,
        };
        let embedding = VectorEmbedding::new(dimensions, i as f32 * 0.01);
        cache.insert(id, embedding).unwrap();

        if (i + 1) % 100 == 0 {
            println!("  Loaded {} vectors", i + 1);
        }
    }

    let load_time = start.elapsed();
    println!("Loaded {} vectors in {:?}\n", num_vectors, load_time);

    // Print cache statistics
    let stats = cache.stats().snapshot();
    println!("Cache Statistics:");
    println!("  Entries: {}", cache.len());
    println!("  Size: {} MB", cache.size() / (1024 * 1024));
    println!("  Inserts: {}", stats.inserts);
    println!("  Evictions: {}", stats.evictions);
    println!();

    // Simulate similarity search queries
    println!("Performing similarity searches...");
    let query_embedding = VectorEmbedding::new(dimensions, 0.5);
    let mut similarities = Vec::new();

    let start = Instant::now();
    for i in 0..100 {
        let id = VectorId {
            index_id: 1,
            vector_id: i * 10, // Sample every 10th vector
        };

        if let Some(pinned) = cache.get(&id) {
            let embedding = pinned.get();
            let similarity = query_embedding.compute_similarity(&VectorEmbedding {
                dimensions: embedding.dimensions,
                data: embedding.data.to_vec(),
            });
            similarities.push((i * 10, similarity));
        }
    }
    let search_time = start.elapsed();

    // Find top-5 most similar
    similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("Top 5 most similar vectors:");
    for (id, score) in similarities.iter().take(5) {
        println!("  Vector {}: similarity = {:.4}", id, score);
    }
    println!("Search completed in {:?}\n", search_time);

    // Simulate concurrent access
    println!("Testing concurrent access...");
    let start = Instant::now();
    let mut handles = vec![];

    for thread_id in 0..4 {
        let cache = cache.clone();
        let handle = thread::spawn(move || {
            let mut hits = 0;
            let mut misses = 0;

            for i in 0..250 {
                let vector_id = (thread_id * 250 + i) % num_vectors;
                let id = VectorId {
                    index_id: 1,
                    vector_id,
                };

                if cache.get(&id).is_some() {
                    hits += 1;
                } else {
                    misses += 1;
                }

                // Simulate some processing
                thread::sleep(Duration::from_micros(10));
            }

            (hits, misses)
        });
        handles.push(handle);
    }

    let mut total_hits = 0;
    let mut total_misses = 0;
    for (i, handle) in handles.into_iter().enumerate() {
        let (hits, misses) = handle.join().unwrap();
        println!("  Thread {}: {} hits, {} misses", i, hits, misses);
        total_hits += hits;
        total_misses += misses;
    }

    let concurrent_time = start.elapsed();
    println!("Concurrent access completed in {:?}", concurrent_time);
    println!("Total: {} hits, {} misses", total_hits, total_misses);
    println!(
        "Hit rate: {:.2}%\n",
        (total_hits as f64 / (total_hits + total_misses) as f64) * 100.0
    );

    // Final statistics
    let final_stats = cache.stats().snapshot();
    println!("Final Cache Statistics:");
    println!("  Total hits: {}", final_stats.hits);
    println!("  Total misses: {}", final_stats.misses);
    println!("  Hit rate: {:.2}%", final_stats.hit_rate() * 100.0);
    println!("  Evictions: {}", final_stats.evictions);
    println!("  Current entries: {}", cache.len());
    println!("  Current size: {} MB", cache.size() / (1024 * 1024));

    println!("\n=== Example Complete ===");
}
