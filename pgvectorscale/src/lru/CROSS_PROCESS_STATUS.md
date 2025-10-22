# Cross-Process LRU Cache Implementation Status

## Summary

Successfully upgraded the PostgreSQL-native LRU cache (`PgSharedLru`) from process-local memory to true cross-process shared memory support.

## What Was Done

### 1. Created DSM Support (`dsm_lru.rs`)
- Implemented Dynamic Shared Memory (DSM) allocation wrapper
- Follows pgvector's parallel build patterns
- Supports parallel worker processes
- Uses `shm_toc` for shared memory table of contents

### 2. Added Conventional Shared Memory Support (`shmem_lru.rs`)
- Simpler alternative to DSM
- Uses `RequestAddinShmemSpace` and `ShmemInitStruct`
- Allocates persistent shared memory at PostgreSQL startup
- More suitable for long-lived caches

### 3. Updated Existing Shared Memory Module (`shared_memory.rs`)
- Changed from `palloc` (process-local) to `ShmemInitStruct` (cross-process)
- Now uses `RequestAddinShmemSpace` to reserve shared memory
- Requests named LWLock tranches for proper synchronization
- Creates two named caches:
  - Quantized Vector Cache (10MB)
  - Builder Neighbor Cache (10MB)

### 4. Added Cross-Process Testing (`cross_process_test.rs`)
- SQL functions for testing cross-process visibility:
  - `pgvs_shared_lru_insert()` - Insert into shared cache
  - `pgvs_shared_lru_get()` - Retrieve from shared cache
  - `pgvs_shared_lru_stats()` - Get cache statistics
  - `pgvs_test_cross_process()` - Test cross-process functionality
- Integration tests using `pg_test`

## How It Works

### Initialization Flow
1. **During `_PG_init()`**:
   - `request_shared_memory()` reserves shared memory space
   - Requests named LWLock tranches for synchronization

2. **During PostgreSQL startup**:
   - `init_shared_memory()` allocates named shared segments
   - Uses `ShmemInitStruct` for cross-process accessibility
   - Initializes LWLocks for thread-safe access

3. **At runtime**:
   - All backend processes see the same cache
   - LWLocks ensure thread-safe concurrent access
   - Data persists across backend lifetimes

### Memory Architecture
```
Shared Memory Segment
├── SharedLruCaches (control structure)
│   ├── quantized_vector_cache pointer
│   └── builder_neighbor_cache pointer
├── Quantized Vector Cache Memory (10MB)
│   ├── LruSharedHeader
│   └── Entry storage area
└── Builder Neighbor Cache Memory (10MB)
    ├── LruSharedHeader
    └── Entry storage area
```

## Testing Cross-Process Functionality

### Manual Testing
```sql
-- Session 1:
SELECT pgvs_shared_lru_insert(1, 'test_value');

-- Session 2 (different backend):
SELECT pgvs_shared_lru_get(1);  -- Returns 'test_value'
SELECT pgvs_shared_lru_stats(); -- Shows cache statistics
```

### Automated Testing
```bash
# Run PostgreSQL extension tests
cargo pgrx test
```

## Key Improvements Over Previous Implementation

1. **True Cross-Process Support**:
   - Previous: Used `palloc` (process-local memory)
   - Now: Uses `ShmemInitStruct` (shared memory)

2. **Proper Synchronization**:
   - Uses PostgreSQL LWLocks embedded in structures
   - Thread-safe across multiple backends

3. **Multiple Implementation Options**:
   - DSM for parallel operations (like index builds)
   - Conventional shared memory for persistent caches
   - Flexible architecture supports both approaches

## Limitations & Future Work

1. **Memory Fragmentation**:
   - Still uses bump allocator (no individual deallocation)
   - Same limitation as pgvector's HNSW
   - Would benefit from periodic compaction

2. **Fixed Hash Table Size**:
   - Currently uses 1024 buckets
   - Could be made configurable

3. **No Iteration Support**:
   - Can't enumerate all cache entries
   - Would need additional data structure

4. **Testing**:
   - Needs more comprehensive cross-process tests
   - Performance benchmarking under concurrent load
   - Stress testing with many backends

## Next Steps

1. **Integration Testing**: Test with actual pgvectorscale workloads
2. **Performance Tuning**: Benchmark and optimize hot paths
3. **Memory Management**: Consider implementing compaction
4. **Production Hardening**: Add monitoring, metrics, and debugging tools

## Conclusion

The shared memory LRU cache now supports true cross-process operation, enabling efficient data sharing between PostgreSQL backend processes. This is essential for scenarios like:
- Parallel index builds
- Shared query result caching
- Cross-connection state sharing
- Reduced memory usage through deduplication

The implementation follows PostgreSQL best practices and integrates properly with the extension lifecycle.