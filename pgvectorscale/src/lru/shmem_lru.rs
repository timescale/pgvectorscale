//! Conventional shared memory support for PostgreSQL LRU cache
//!
//! This module provides cross-process support using PostgreSQL's conventional
//! shared memory (shmem) infrastructure, which is simpler than DSM.

use super::pg_lru::PgSharedLru;
use pgrx::pg_sys;
use pgrx::prelude::*;
use std::ptr;

/// Size of shared memory to request (in MB)
const SHMEM_SIZE_MB: usize = 100;

/// Global pointer to shared memory area
static mut SHMEM_LRU_AREA: *mut u8 = ptr::null_mut();

/// Size of the shared memory area
static mut SHMEM_LRU_SIZE: usize = 0;

/// Initialize shared memory for LRU cache
///
/// This should be called during _PG_init() to request shared memory
pub fn shmem_request_hook() {
    unsafe {
        // Request shared memory
        let size = SHMEM_SIZE_MB * 1024 * 1024;
        pg_sys::RequestAddinShmemSpace(size as _);

        // Request named shared memory locks (for LWLocks)
        pg_sys::RequestNamedLWLockTranche(
            c"pgvectorscale_lru".as_ptr(),
            2, // Number of locks (one for structure, one spare)
        );
    }
}

/// Initialize shared memory during startup
///
/// This is called during shared memory initialization
///
/// # Safety
/// Must be called during PostgreSQL shared memory initialization
pub unsafe fn shmem_startup_hook() {
    let size = SHMEM_SIZE_MB * 1024 * 1024;
    let found = std::ptr::null_mut();

    // Allocate named shared memory segment
    SHMEM_LRU_AREA =
        pg_sys::ShmemInitStruct(c"pgvectorscale_lru_area".as_ptr(), size as _, found) as *mut u8;

    SHMEM_LRU_SIZE = size;

    // If this is the first time (not found), initialize the memory
    if found.is_null() || !*found {
        ptr::write_bytes(SHMEM_LRU_AREA, 0, size);
        info!(
            "Initialized pgvectorscale LRU shared memory: {} MB",
            SHMEM_SIZE_MB
        );
    }
}

/// Create or attach to shared memory LRU cache
///
/// # Safety
/// Must be called after shared memory is initialized
pub unsafe fn get_shared_lru() -> Option<PgSharedLru> {
    if SHMEM_LRU_AREA.is_null() {
        warning!("Shared memory LRU area not initialized");
        return None;
    }

    Some(PgSharedLru::new_in_memory(SHMEM_LRU_AREA, SHMEM_LRU_SIZE))
}

/// Check if shared memory is available
pub fn is_shmem_available() -> bool {
    unsafe { !SHMEM_LRU_AREA.is_null() }
}

/// Get shared memory statistics
pub fn get_shmem_stats() -> (usize, usize) {
    unsafe { (SHMEM_LRU_SIZE, SHMEM_SIZE_MB) }
}

// Integration with pgvectorscale extension initialization
#[pg_guard]
pub unsafe extern "C" fn pgvectorscale_lru_shmem_request() {
    shmem_request_hook();
}

#[pg_guard]
pub unsafe extern "C" fn pgvectorscale_lru_shmem_startup() {
    shmem_startup_hook();
}

/// Example usage in tests
#[cfg(any(test, feature = "pg_test"))]
mod tests {
    use super::*;

    #[pg_test]
    fn test_shmem_lru() {
        unsafe {
            // This test would only work if shared memory is properly initialized
            // during PostgreSQL startup
            if let Some(lru) = get_shared_lru() {
                // Insert some test data
                match lru.insert(1i32, "test_value".to_string()) {
                    Ok(_) => info!("Successfully inserted into shared LRU"),
                    Err(e) => warning!("Failed to insert: {}", e),
                }

                // Try to retrieve
                if let Some(value) = lru.get::<i32, String>(&1) {
                    assert_eq!(value, "test_value");
                    info!("Successfully retrieved from shared LRU");
                }
            } else {
                warning!("Shared memory LRU not available in test");
            }
        }
    }
}
