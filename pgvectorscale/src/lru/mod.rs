// Shared memory LRU cache implementation

pub mod allocator;
pub mod cache;
pub mod entry;
pub mod stats;

#[cfg(test)]
mod debug_serialize;
#[cfg(test)]
mod debug_test;
#[cfg(test)]
mod simple_test;
#[cfg(test)]
mod tests;

pub use allocator::{AllocError, SharedMemoryAllocator};
pub use cache::SharedMemoryLru;
pub use entry::PinnedEntry;
pub use stats::CacheStats;

// Re-export for convenience
pub use allocator::{MockAllocator, ShmemAllocator};
// DsaAllocator will be exported when Postgres integration is complete
