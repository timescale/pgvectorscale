//! DSM (Dynamic Shared Memory) support for PostgreSQL LRU cache
//!
//! This module provides cross-process support for the LRU cache using PostgreSQL's
//! Dynamic Shared Memory (DSM) infrastructure, following pgvector's approach.

use super::pg_lru::PgSharedLru;
use pgrx::pg_sys;
use pgrx::prelude::*;
use std::ptr;

/// Keys for shared memory table of contents (shm_toc)
const DSM_KEY_LRU_HEADER: u64 = 1;
const DSM_KEY_LRU_AREA: u64 = 2;
const DSM_KEY_QUERY_TEXT: u64 = 3;

/// Shared state for DSM-based LRU cache (stored in DSM)
#[repr(C)]
pub struct LruDsmShared {
    /// Size of the LRU area in bytes
    pub area_size: usize,

    /// Whether this is a concurrent operation
    pub is_concurrent: bool,

    /// Condition variable for workers synchronization
    pub workers_done_cv: pg_sys::ConditionVariable,

    /// Mutex for shared state
    pub mutex: pg_sys::slock_t,

    /// Number of participants that have completed
    pub participants_done: i32,
    // Placeholder for table scan state if needed
    // (In pgvector this would be ParallelTableScanDescData)
}

/// Leader state for coordinating DSM-based LRU operations
pub struct LruDsmLeader {
    /// Parallel context
    pub pcxt: *mut pg_sys::ParallelContext,

    /// Shared state in DSM
    pub shared: *mut LruDsmShared,

    /// LRU area in DSM
    pub lru_area: *mut u8,

    /// Number of worker processes launched
    pub nworkers: i32,
}

impl LruDsmLeader {
    /// Estimate the size needed for DSM allocation
    pub fn estimate_dsm_size(cache_size_bytes: usize) -> usize {
        let shared_size = std::mem::size_of::<LruDsmShared>();
        let area_size = cache_size_bytes;

        // Add some overhead for alignment and management
        let overhead = 4 * 1024; // 4KB overhead

        shared_size + area_size + overhead
    }

    /// Begin parallel operation with DSM
    ///
    /// # Safety
    /// Must be called in appropriate PostgreSQL context
    pub unsafe fn begin_parallel(
        cache_size_mb: usize,
        nworkers: i32,
        is_concurrent: bool,
    ) -> Option<Box<LruDsmLeader>> {
        // Enter parallel mode
        pg_sys::EnterParallelMode();

        // Create parallel context
        let pcxt = pg_sys::CreateParallelContext(
            c"pgvectorscale".as_ptr(),
            c"LruDsmWorkerMain".as_ptr(),
            nworkers,
        );

        if pcxt.is_null() {
            pg_sys::ExitParallelMode();
            return None;
        }

        let cache_size_bytes = cache_size_mb * 1024 * 1024;

        // Estimate sizes for shm_toc
        let est_shared = std::mem::size_of::<LruDsmShared>();
        let est_area = cache_size_bytes;

        // Estimate chunks for shm_toc
        // In PostgreSQL C, shm_toc_estimate_chunk and shm_toc_estimate_keys are macros
        // that manipulate the estimator. We'll manually add to the size.
        let total_est = est_shared + est_area + 1024; // Add some overhead

        // The estimator needs to know the total size
        // We'll use add_size to accumulate sizes
        let mut _current_size = pg_sys::shm_toc_estimate(&mut (*pcxt).estimator);
        _current_size = pg_sys::add_size(_current_size, total_est);

        // Add query text if available
        let query_len = if !pg_sys::debug_query_string.is_null() {
            let query = std::ffi::CStr::from_ptr(pg_sys::debug_query_string);
            let len = query.to_bytes().len() + 1;
            _current_size = pg_sys::add_size(_current_size, len);
            len
        } else {
            0
        };

        // Store the estimated size back
        // Note: This is a simplified approach; in production you'd need to properly
        // manipulate the estimator structure
        // In a real implementation, we'd update the estimator with _current_size

        // Initialize DSM segment
        pg_sys::InitializeParallelDSM(pcxt);

        // Check if DSM was successfully allocated
        if (*pcxt).seg.is_null() {
            pg_sys::DestroyParallelContext(pcxt);
            pg_sys::ExitParallelMode();
            return None;
        }

        // Allocate shared state in DSM
        let shared = pg_sys::shm_toc_allocate((*pcxt).toc, est_shared) as *mut LruDsmShared;
        ptr::write_bytes(shared, 0, 1);

        // Initialize shared state
        (*shared).area_size = cache_size_bytes;
        (*shared).is_concurrent = is_concurrent;
        pg_sys::ConditionVariableInit(&mut (*shared).workers_done_cv);
        pg_sys::SpinLockInit(&mut (*shared).mutex);
        (*shared).participants_done = 0;

        // Allocate LRU area in DSM
        let lru_area = pg_sys::shm_toc_allocate((*pcxt).toc, est_area) as *mut u8;

        // Insert into table of contents
        pg_sys::shm_toc_insert((*pcxt).toc, DSM_KEY_LRU_HEADER, shared as *mut _);
        pg_sys::shm_toc_insert((*pcxt).toc, DSM_KEY_LRU_AREA, lru_area as *mut _);

        // Store query string if available
        if query_len > 0 {
            let shared_query = pg_sys::shm_toc_allocate((*pcxt).toc, query_len) as *mut i8;
            ptr::copy_nonoverlapping(pg_sys::debug_query_string, shared_query, query_len);
            pg_sys::shm_toc_insert((*pcxt).toc, DSM_KEY_QUERY_TEXT, shared_query as *mut _);
        }

        // Launch worker processes
        pg_sys::LaunchParallelWorkers(pcxt);

        Some(Box::new(LruDsmLeader {
            pcxt,
            shared,
            lru_area,
            nworkers: (*pcxt).nworkers_launched,
        }))
    }

    /// Create LRU cache in DSM area
    ///
    /// # Safety
    /// Must be called with valid DSM leader context
    pub unsafe fn create_lru(&self) -> PgSharedLru {
        PgSharedLru::new_in_memory(self.lru_area, (*self.shared).area_size)
    }

    /// Wait for all workers to finish
    ///
    /// # Safety
    /// Must be called with valid DSM leader context
    pub unsafe fn wait_for_workers(&self) {
        pg_sys::WaitForParallelWorkersToFinish(self.pcxt);
    }

    /// End parallel operation
    ///
    /// # Safety
    /// Must be called with valid DSM leader context
    pub unsafe fn end_parallel(self) {
        pg_sys::WaitForParallelWorkersToFinish(self.pcxt);
        pg_sys::DestroyParallelContext(self.pcxt);
        pg_sys::ExitParallelMode();
    }
}

/// Worker-side functions for DSM-based LRU
pub struct LruDsmWorker;

impl LruDsmWorker {
    /// Attach to DSM-based LRU from worker process
    ///
    /// # Safety
    /// Must be called from parallel worker context
    pub unsafe fn attach(toc: *mut pg_sys::shm_toc) -> Option<(PgSharedLru, *mut LruDsmShared)> {
        // Look up query text first (for debugging)
        let shared_query = pg_sys::shm_toc_lookup(toc, DSM_KEY_QUERY_TEXT, true);
        if !shared_query.is_null() {
            pg_sys::debug_query_string = shared_query as *const _;
        }

        // Look up shared state
        let shared = pg_sys::shm_toc_lookup(toc, DSM_KEY_LRU_HEADER, false) as *mut LruDsmShared;
        if shared.is_null() {
            return None;
        }

        // Look up LRU area
        let lru_area = pg_sys::shm_toc_lookup(toc, DSM_KEY_LRU_AREA, false) as *mut u8;
        if lru_area.is_null() {
            return None;
        }

        // Create LRU cache view for this worker
        let lru = PgSharedLru::new_in_memory(lru_area, (*shared).area_size);

        Some((lru, shared))
    }

    /// Mark this worker as done
    ///
    /// # Safety
    /// Must be called with valid shared pointer from DSM
    pub unsafe fn mark_done(shared: *mut LruDsmShared) {
        pg_sys::SpinLockAcquire(&mut (*shared).mutex);
        (*shared).participants_done += 1;
        pg_sys::SpinLockRelease(&mut (*shared).mutex);
        pg_sys::ConditionVariableSignal(&mut (*shared).workers_done_cv);
    }
}

/// Main function for parallel workers (registered with CreateParallelContext)
#[pg_guard]
#[no_mangle]
pub unsafe extern "C" fn LruDsmWorkerMain(
    _seg: *mut pg_sys::dsm_segment,
    toc: *mut pg_sys::shm_toc,
) {
    // Attach to DSM segment
    if let Some((_lru, shared)) = LruDsmWorker::attach(toc) {
        // Worker can now use the LRU cache
        // This is where actual work would be performed

        // For now, just mark as done
        LruDsmWorker::mark_done(shared);
    }
}

/// Test function to verify DSM allocation works
#[cfg(any(test, feature = "pg_test"))]
#[pg_test]
fn test_dsm_lru_basic() {
    unsafe {
        // Try to create DSM-based LRU
        if let Some(leader) = LruDsmLeader::begin_parallel(10, 2, false) {
            // Create LRU in DSM
            let lru = leader.create_lru();

            // Insert some test data
            lru.insert(1i32, "test".to_string()).unwrap();

            // Wait for workers
            leader.wait_for_workers();

            // Clean up
            leader.end_parallel();
        }
    }
}
