// Shared memory LRU cache implementation

pub mod allocator;
pub mod cache;
pub mod entry;
pub mod stats;

// PostgreSQL native implementation
// This uses PostgreSQL's LWLocks and shared memory directly
pub mod cross_process_test;
pub mod dsm_lru;
pub mod pg_lru;
pub mod pg_lru_cache_with_stats;
mod pg_lru_test;
pub mod shared_memory;
pub mod shmem_lru;

// Test modules
mod debug_serialize;
mod debug_test;
mod simple_test;
mod tests;

pub use allocator::{AllocError, SharedMemoryAllocator};
pub use cache::SharedMemoryLru;
pub use entry::PinnedEntry;
pub use stats::CacheStats;

// Re-export for convenience
pub use allocator::{MockAllocator, ShmemAllocator};
// DsaAllocator will be exported when Postgres integration is complete

// Export PostgreSQL-native LRU
pub use pg_lru::PgSharedLru;
