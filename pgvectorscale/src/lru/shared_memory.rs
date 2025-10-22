//! Shared memory management for PostgreSQL-native LRU caches
//!
//! This module handles the allocation and initialization of shared memory
//! segments for LRU caches that need to be accessible across PostgreSQL
//! backend processes.

use pgrx::pg_sys;
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};

use super::pg_lru::PgSharedLru;

/// Global flag to track if shared memory has been initialized
static SHARED_MEMORY_INITIALIZED: AtomicBool = AtomicBool::new(false);

/// Previous shmem_request_hook (PG15+)
#[cfg(any(feature = "pg15", feature = "pg16", feature = "pg17"))]
static mut PREV_SHMEM_REQUEST_HOOK: Option<unsafe extern "C" fn()> = None;

/// Previous shmem_startup_hook
static mut PREV_SHMEM_STARTUP_HOOK: Option<unsafe extern "C" fn()> = None;

/// Size of shared memory to request for each cache type
// Reduced size for test environments with limited shared memory
const QUANTIZED_VECTOR_CACHE_SHMEM_SIZE: usize = 1024 * 1024; // 1MB
const BUILDER_NEIGHBOR_CACHE_SHMEM_SIZE: usize = 1024 * 1024; // 1MB

/// Total shared memory to request
pub const TOTAL_SHMEM_SIZE: usize =
    QUANTIZED_VECTOR_CACHE_SHMEM_SIZE + BUILDER_NEIGHBOR_CACHE_SHMEM_SIZE;

/// Structure to hold pointers to our shared memory cache data
#[repr(C)]
pub struct SharedLruCacheData {
    pub quantized_vector_cache_base: *mut u8,
    pub builder_neighbor_cache_base: *mut u8,
}

/// Global pointer to shared cache data pointers (set during initialization)
static mut SHARED_CACHE_DATA: *mut SharedLruCacheData = ptr::null_mut();

/// Hook callback for shmem_request_hook (PG15+)
///
/// This is called during PostgreSQL startup to request shared memory
#[cfg(any(feature = "pg15", feature = "pg16", feature = "pg17"))]
unsafe extern "C" fn pgvectorscale_shmem_request() {
    // Call previous hook if it exists
    if let Some(prev_hook) = PREV_SHMEM_REQUEST_HOOK {
        prev_hook();
    }

    // Request shared memory for our caches
    pg_sys::RequestAddinShmemSpace(TOTAL_SHMEM_SIZE as _);

    // Also request named LWLock tranches for our caches
    pg_sys::RequestNamedLWLockTranche(
        c"pgvectorscale_qv_cache".as_ptr(),
        2, // Two locks per cache
    );
    pg_sys::RequestNamedLWLockTranche(
        c"pgvectorscale_bn_cache".as_ptr(),
        2, // Two locks per cache
    );
}

/// Hook callback for shmem_startup_hook
///
/// This is called during PostgreSQL startup to initialize shared memory
unsafe extern "C" fn pgvectorscale_shmem_startup() {
    // Call previous hook if it exists
    if let Some(prev_hook) = PREV_SHMEM_STARTUP_HOOK {
        prev_hook();
    }

    // Initialize our shared memory structures
    init_shared_memory();
}

/// Register shared memory hooks during _PG_init
///
/// For PG15+, this sets up the shmem_request_hook
/// For older versions, it calls RequestAddinShmemSpace directly
///
/// # Safety
/// Must be called during PostgreSQL extension initialization
pub unsafe fn register_hooks() {
    #[cfg(any(feature = "pg15", feature = "pg16", feature = "pg17"))]
    {
        // PG15+: Use shmem_request_hook
        PREV_SHMEM_REQUEST_HOOK = pg_sys::shmem_request_hook;
        pg_sys::shmem_request_hook = Some(pgvectorscale_shmem_request);
    }

    #[cfg(not(any(feature = "pg15", feature = "pg16", feature = "pg17")))]
    {
        // PG < 15: Call directly in _PG_init
        pg_sys::RequestAddinShmemSpace(TOTAL_SHMEM_SIZE as _);

        pg_sys::RequestNamedLWLockTranche(
            c"pgvectorscale_qv_cache".as_ptr(),
            2, // Two locks per cache
        );
        pg_sys::RequestNamedLWLockTranche(
            c"pgvectorscale_bn_cache".as_ptr(),
            2, // Two locks per cache
        );
    }

    // Register shmem_startup_hook for all versions
    PREV_SHMEM_STARTUP_HOOK = pg_sys::shmem_startup_hook;
    pg_sys::shmem_startup_hook = Some(pgvectorscale_shmem_startup);
}

/// Initialize shared memory structures
///
/// This should be called during shared memory initialization hook.
/// It uses ShmemInitStruct to allocate named shared memory segments
/// that are accessible across all backend processes.
///
/// # Safety
/// Must be called during PostgreSQL shared memory initialization phase
pub unsafe fn init_shared_memory() {
    // Only initialize once
    if SHARED_MEMORY_INITIALIZED.load(Ordering::Acquire) {
        return;
    }

    let mut found_data = false;
    let mut found_qv = false;
    let mut found_bn = false;

    // Allocate shared structure for cache data pointers
    SHARED_CACHE_DATA = pg_sys::ShmemInitStruct(
        c"pgvectorscale_lru_cache_data".as_ptr(),
        size_of::<SharedLruCacheData>() as _,
        &mut found_data as *mut bool,
    ) as *mut SharedLruCacheData;

    if !found_data {
        // First time initialization
        ptr::write_bytes(SHARED_CACHE_DATA, 0, 1);
    }

    // Allocate shared memory for quantized vector cache
    let qv_cache_mem = pg_sys::ShmemInitStruct(
        c"pgvectorscale_qv_cache_mem".as_ptr(),
        QUANTIZED_VECTOR_CACHE_SHMEM_SIZE as _,
        &mut found_qv as *mut bool,
    ) as *mut u8;

    if !found_qv {
        // First time initialization
        PgSharedLru::new_in_memory(qv_cache_mem, QUANTIZED_VECTOR_CACHE_SHMEM_SIZE);
    }

    // Always store the base pointer (same for all processes)
    (*SHARED_CACHE_DATA).quantized_vector_cache_base = qv_cache_mem;

    // Allocate shared memory for builder neighbor cache
    let bn_cache_mem = pg_sys::ShmemInitStruct(
        c"pgvectorscale_bn_cache_mem".as_ptr(),
        BUILDER_NEIGHBOR_CACHE_SHMEM_SIZE as _,
        &mut found_bn as *mut bool,
    ) as *mut u8;

    if !found_bn {
        // First time initialization
        PgSharedLru::new_in_memory(bn_cache_mem, BUILDER_NEIGHBOR_CACHE_SHMEM_SIZE);
    }

    // Always store the base pointer (same for all processes)
    (*SHARED_CACHE_DATA).builder_neighbor_cache_base = bn_cache_mem;

    // Mark as initialized
    SHARED_MEMORY_INITIALIZED.store(true, Ordering::Release);
}

/// Get the shared quantized vector cache
///
/// Returns None if shared memory has not been initialized.
/// Creates a per-process handle pointing to the shared cache data.
pub fn get_quantized_vector_cache() -> Option<PgSharedLru> {
    if !SHARED_MEMORY_INITIALIZED.load(Ordering::Acquire) {
        return None;
    }

    unsafe {
        if SHARED_CACHE_DATA.is_null() || (*SHARED_CACHE_DATA).quantized_vector_cache_base.is_null()
        {
            None
        } else {
            // Create a per-process handle pointing to shared memory
            let base = (*SHARED_CACHE_DATA).quantized_vector_cache_base;
            Some(PgSharedLru::from_existing(
                base,
                QUANTIZED_VECTOR_CACHE_SHMEM_SIZE,
            ))
        }
    }
}

/// Get the shared builder neighbor cache
///
/// Returns None if shared memory has not been initialized.
/// Creates a per-process handle pointing to the shared cache data.
pub fn get_builder_neighbor_cache() -> Option<PgSharedLru> {
    if !SHARED_MEMORY_INITIALIZED.load(Ordering::Acquire) {
        return None;
    }

    unsafe {
        if SHARED_CACHE_DATA.is_null() || (*SHARED_CACHE_DATA).builder_neighbor_cache_base.is_null()
        {
            None
        } else {
            // Create a per-process handle pointing to shared memory
            let base = (*SHARED_CACHE_DATA).builder_neighbor_cache_base;
            Some(PgSharedLru::from_existing(
                base,
                BUILDER_NEIGHBOR_CACHE_SHMEM_SIZE,
            ))
        }
    }
}

/// Check if shared memory has been initialized
pub fn is_initialized() -> bool {
    SHARED_MEMORY_INITIALIZED.load(Ordering::Acquire)
}
