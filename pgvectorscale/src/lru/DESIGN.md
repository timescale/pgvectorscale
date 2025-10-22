# Shared Memory LRU Cache Design

## Overview

A high-performance, concurrent LRU (Least Recently Used) cache designed for Postgres shared memory environments. The cache stores serialized objects using rkyv for zero-copy access and supports multiple readers/writers.

## Key Requirements

1. **Shared Memory Compatible**: Objects stored in Postgres shared memory (DSA or conventional)
2. **Zero-Copy Access**: Use rkyv serialization for efficient access without deserialization
3. **Concurrent Access**: Support multiple readers and writers
4. **Entry Safety**: Prevent entries from being freed or modified while in use
5. **Pluggable Allocators**: Support both DSA and conventional shared memory
6. **Custom Eviction**: Support custom eviction handlers for complex behaviors

## Architecture

### Core Components

#### 1. Allocator Abstraction

```rust
pub trait SharedMemoryAllocator: Send + Sync {
    type Pointer: Copy + Clone + Debug + Eq + Hash;

    fn allocate(&self, size: usize) -> Result<Self::Pointer, AllocError>;
    unsafe fn deallocate(&self, ptr: Self::Pointer);
    unsafe fn get_address(&self, ptr: Self::Pointer) -> *mut u8;
    unsafe fn get_size(&self, ptr: Self::Pointer) -> Option<usize>;
}
```

Two implementations:
- `DsaAllocator`: Uses Postgres Dynamic Shared Areas
- `ShmemAllocator`: Uses conventional shared memory with bump allocation

#### 2. Entry Structure in Shared Memory

```rust
#[repr(C)]
struct SharedLruEntry<A: SharedMemoryAllocator> {
    // Entry metadata
    key_size: u32,
    value_size: u32,
    total_size: u32,

    // LRU list pointers
    next: Option<A::Pointer>,
    prev: Option<A::Pointer>,

    // Reference counting for safe access
    pin_count: AtomicU32,

    // Serialized data (key followed by value)
    data: [u8; 0],
}
```

#### 3. Main LRU Cache

```rust
pub struct SharedMemoryLru<K, V, A> {
    allocator: A,
    index: Arc<RwLock<HashMap<K, EntryHandle<A>>>>,
    lru_head: Arc<RwLock<Option<A::Pointer>>>,
    lru_tail: Arc<RwLock<Option<A::Pointer>>>,
    current_size: Arc<AtomicUsize>,
    capacity: usize,
    eviction_handler: Option<Box<dyn EvictionHandler<K, V>>>,
    stats: Arc<CacheStats>,
}
```

#### 4. Entry Pinning with RAII

```rust
pub struct PinnedEntry<'a, V: Archive, A: SharedMemoryAllocator> {
    entry_ptr: A::Pointer,
    allocator: &'a A,
    cache: &'a SharedMemoryLru<?, V, A>,
    _phantom: PhantomData<V>,
}

impl<'a, V: Archive, A: SharedMemoryAllocator> Drop for PinnedEntry<'a, V, A> {
    fn drop(&mut self) {
        // Decrement pin count
        unsafe {
            let entry = self.allocator.get_address(self.entry_ptr) as *mut SharedLruEntry<A>;
            (*entry).pin_count.fetch_sub(1, Ordering::Release);
        }
        // Signal potential waiters
        self.cache.notify_unpin();
    }
}
```

### Concurrency Model

#### Entry Lifecycle

1. **Insertion**: Acquires write lock on index, allocates memory, adds to LRU head
2. **Access**: Acquires read lock on index, increments pin count, returns pinned reference
3. **Eviction**: Only evicts entries with pin_count == 0, waits if necessary
4. **Modification**: Not allowed while entry is pinned (enforced by returning immutable references)

#### Lock Hierarchy

To prevent deadlocks, locks are always acquired in this order:
1. Index lock (RwLock)
2. LRU head/tail locks (RwLock)
3. Entry pin count (Atomic)

#### Safety Guarantees

- **No use-after-free**: Pin count prevents deallocation while references exist
- **No concurrent modification**: Only immutable references are handed out
- **No data races**: All shared state protected by locks or atomics
- **Memory ordering**: Use Acquire/Release for pin counts to ensure visibility

### Operations

#### Get Operation
```rust
pub fn get(&self, key: &K) -> Option<PinnedEntry<'_, V, A>> {
    // 1. Read lock on index
    // 2. Find entry pointer
    // 3. Increment pin count atomically
    // 4. Update LRU position (requires write locks on list)
    // 5. Return PinnedEntry (RAII wrapper)
}
```

#### Insert Operation
```rust
pub fn insert(&self, key: K, value: V) -> Result<(), LruError> {
    // 1. Serialize key and value
    // 2. Allocate shared memory
    // 3. Write lock on index
    // 4. Evict entries if over capacity (skipping pinned ones)
    // 5. Add to index and LRU head
    // 6. Update size tracking
}
```

#### Eviction
```rust
fn evict_lru(&self, index: &mut HashMap<K, EntryHandle<A>>) -> Result<(), LruError> {
    // 1. Find tail entry
    // 2. Check if pinned (skip if pin_count > 0)
    // 3. Remove from index and LRU list
    // 4. Call eviction handler if present
    // 5. Deallocate memory
}
```

## Testing Strategy

### Unit Tests

Located in `src/lru/tests.rs`:

1. **Basic Operations**
   - Insert and retrieve values
   - Update existing keys
   - Check contains without modifying LRU order

2. **LRU Ordering**
   - Verify most recently used moves to head
   - Verify least recently used gets evicted first
   - Test with capacity of 1, 2, and N elements

3. **Eviction**
   - Test eviction when at capacity
   - Verify custom eviction handler is called
   - Test that pinned entries are not evicted

4. **Pinning**
   - Test multiple pins on same entry
   - Verify entry not evicted while pinned
   - Test pin cleanup on drop

5. **Concurrency**
   - Multiple readers accessing same entry
   - Reader holding pin while writer tries to evict
   - Concurrent inserts and reads
   - Stress test with many threads

6. **Memory Management**
   - Verify allocations are freed on eviction
   - Test memory budget enforcement
   - Check for leaks with valgrind/ASAN

7. **Serialization**
   - Test with various key/value types
   - Large values (>1MB)
   - Empty values
   - Complex nested structures

### Integration Tests

Located in `src/lru/integration_tests.rs`:

1. **With Real Postgres Allocators**
   - Test with DSA allocator
   - Test with shared memory allocator
   - Verify cross-process visibility (if applicable)

2. **Cache Replacement**
   - Replace existing `QuantizedVectorCache`
   - Replace existing `BuilderNeighborCache`
   - Verify same behavior as original

### Performance Tests

Located in `benches/lru_bench.rs`:

1. **Throughput Benchmarks**
   ```rust
   #[bench]
   fn bench_insert_1m_entries(b: &mut Bencher) {
       // Measure inserts/second
   }

   #[bench]
   fn bench_get_hit_rate_100(b: &mut Bencher) {
       // Measure gets/second with 100% hit rate
   }

   #[bench]
   fn bench_get_hit_rate_50(b: &mut Bencher) {
       // Measure gets/second with 50% hit rate
   }

   #[bench]
   fn bench_concurrent_readers(b: &mut Bencher) {
       // Measure read throughput with N readers
   }
   ```

2. **Latency Benchmarks**
   - p50, p95, p99 latencies for get/insert
   - Worst-case eviction time
   - LRU update overhead

3. **Scalability Tests**
   - Performance vs number of entries
   - Performance vs entry size
   - Performance vs number of threads

4. **Memory Overhead**
   - Measure metadata overhead per entry
   - Compare memory usage to theoretical minimum
   - Fragmentation analysis

5. **Comparison Benchmarks**
   - Compare to existing `LruCacheWithStats`
   - Compare to standard HashMap
   - Compare serialization overhead vs native structs

### Test Infrastructure

```rust
// Test helper for creating test caches
fn create_test_cache<A: SharedMemoryAllocator>(
    allocator: A,
    capacity: usize,
) -> SharedMemoryLru<String, Vec<u8>, A> {
    SharedMemoryLru::new(
        allocator,
        capacity,
        "test".to_string(),
        None,
    )
}

// Mock allocator for testing without real shared memory
struct MockAllocator {
    allocations: RefCell<HashMap<usize, Vec<u8>>>,
    next_ptr: Cell<usize>,
}

// Property-based tests using proptest
proptest! {
    #[test]
    fn prop_never_exceed_capacity(
        operations in vec(operation_strategy(), 1..1000)
    ) {
        // Verify capacity is never exceeded
    }
}
```

## Implementation Plan

### Phase 1: Core Infrastructure
1. ✅ Create directory structure and documentation
2. Implement allocator trait and mock allocator
3. Implement basic entry structure with serialization

### Phase 2: Basic LRU
1. Implement insert/get without pinning
2. Add LRU list management
3. Implement basic eviction

### Phase 3: Safety Features
1. Add pin counting
2. Implement PinnedEntry RAII wrapper
3. Add eviction skipping for pinned entries

### Phase 4: Concurrency
1. Add RwLock protection
2. Implement atomic operations
3. Add deadlock prevention

### Phase 5: Integration
1. Implement DSA and Shmem allocators
2. Create adapters for existing caches
3. Integration testing with real Postgres

### Phase 6: Optimization
1. Performance profiling
2. Optimize hot paths
3. Consider lock-free alternatives for read path

## Design Decisions

### Why RwLock instead of lock-free?
- Simpler to implement correctly
- Postgres workloads typically not extremely high concurrency
- Can optimize later if needed

### Why pin count instead of reference counting?
- Simpler semantics - just prevents eviction
- Don't need full shared ownership
- Lighter weight than Arc

### Why separate index from shared memory?
- Faster lookups (local memory access)
- Simpler key comparison (no serialization)
- Index can be rebuilt if needed

## Future Enhancements

1. **Adaptive Eviction**: Use frequency + recency (LFU-LRU hybrid)
2. **Segmented LRU**: Separate hot/cold segments
3. **Batch Operations**: Bulk insert/preload for efficiency
4. **Compression**: Optional compression for values
5. **Partitioned Cache**: Reduce lock contention with sharding
6. **Statistics**: Detailed performance metrics and histograms

## API Example

```rust
// Create cache with DSA allocator
let dsa = DsaAllocator::new(dsa_area);
let cache = SharedMemoryLru::<ItemPointer, Vec<u64>, _>::new(
    dsa,
    1024 * 1024 * 100, // 100MB
    "quantized_vectors".to_string(),
    None,
);

// Insert value
cache.insert(item_pointer, vector)?;

// Get value (returns pinned entry)
if let Some(pinned) = cache.get(&item_pointer) {
    let archived_vec: &ArchivedVec<u64> = pinned.get();
    // Use archived_vec...
} // Pin automatically released here

// Check if contains (no pin)
if cache.contains(&item_pointer) {
    // ...
}
```