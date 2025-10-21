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

/// Size of shared memory to request for each cache type
const QUANTIZED_VECTOR_CACHE_SHMEM_SIZE: usize = 10 * 1024 * 1024; // 10MB
const BUILDER_NEIGHBOR_CACHE_SHMEM_SIZE: usize = 10 * 1024 * 1024; // 10MB

/// Total shared memory to request
pub const TOTAL_SHMEM_SIZE: usize =
    QUANTIZED_VECTOR_CACHE_SHMEM_SIZE + BUILDER_NEIGHBOR_CACHE_SHMEM_SIZE;

/// Structure to hold pointers to our shared memory caches
#[repr(C)]
pub struct SharedLruCaches {
    pub quantized_vector_cache: *mut PgSharedLru,
    pub builder_neighbor_cache: *mut PgSharedLru,
}

/// Global pointer to shared caches (set during initialization)
static mut SHARED_CACHES: *mut SharedLruCaches = ptr::null_mut();

/// Request shared memory space during _PG_init
///
/// This must be called during _PG_init to reserve shared memory
/// that will be allocated during PostgreSQL startup.
///
/// # Safety
/// Must be called during PostgreSQL extension initialization
pub unsafe fn request_shared_memory() {
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

    let mut found_caches = false;
    let mut found_qv = false;
    let mut found_bn = false;

    // Allocate shared structure for cache pointers
    SHARED_CACHES = pg_sys::ShmemInitStruct(
        c"pgvectorscale_lru_caches".as_ptr(),
        size_of::<SharedLruCaches>() as _,
        &mut found_caches as *mut bool,
    ) as *mut SharedLruCaches;

    if !found_caches {
        // First time initialization
        ptr::write_bytes(SHARED_CACHES, 0, 1);
    }

    // Allocate shared memory for quantized vector cache
    let qv_cache_mem = pg_sys::ShmemInitStruct(
        c"pgvectorscale_qv_cache_mem".as_ptr(),
        QUANTIZED_VECTOR_CACHE_SHMEM_SIZE as _,
        &mut found_qv as *mut bool,
    ) as *mut u8;

    if !found_qv {
        // Initialize the cache structure
        let qv_cache = PgSharedLru::new_in_memory(qv_cache_mem, QUANTIZED_VECTOR_CACHE_SHMEM_SIZE);

        // Allocate space for the cache handle in shared memory
        (*SHARED_CACHES).quantized_vector_cache = pg_sys::ShmemInitStruct(
            c"pgvectorscale_qv_cache_handle".as_ptr(),
            size_of::<PgSharedLru>() as _,
            &mut found_qv as *mut bool,
        ) as *mut PgSharedLru;

        ptr::write((*SHARED_CACHES).quantized_vector_cache, qv_cache);
    }

    // Allocate shared memory for builder neighbor cache
    let bn_cache_mem = pg_sys::ShmemInitStruct(
        c"pgvectorscale_bn_cache_mem".as_ptr(),
        BUILDER_NEIGHBOR_CACHE_SHMEM_SIZE as _,
        &mut found_bn as *mut bool,
    ) as *mut u8;

    if !found_bn {
        // Initialize the cache structure
        let bn_cache = PgSharedLru::new_in_memory(bn_cache_mem, BUILDER_NEIGHBOR_CACHE_SHMEM_SIZE);

        // Allocate space for the cache handle in shared memory
        (*SHARED_CACHES).builder_neighbor_cache = pg_sys::ShmemInitStruct(
            c"pgvectorscale_bn_cache_handle".as_ptr(),
            size_of::<PgSharedLru>() as _,
            &mut found_bn as *mut bool,
        ) as *mut PgSharedLru;

        ptr::write((*SHARED_CACHES).builder_neighbor_cache, bn_cache);
    }

    // Mark as initialized
    SHARED_MEMORY_INITIALIZED.store(true, Ordering::Release);
}

/// Get the shared quantized vector cache
///
/// Returns None if shared memory has not been initialized.
pub fn get_quantized_vector_cache() -> Option<&'static PgSharedLru> {
    if !SHARED_MEMORY_INITIALIZED.load(Ordering::Acquire) {
        return None;
    }

    unsafe {
        if SHARED_CACHES.is_null() || (*SHARED_CACHES).quantized_vector_cache.is_null() {
            None
        } else {
            Some(&*(*SHARED_CACHES).quantized_vector_cache)
        }
    }
}

/// Get the shared builder neighbor cache
///
/// Returns None if shared memory has not been initialized.
pub fn get_builder_neighbor_cache() -> Option<&'static PgSharedLru> {
    if !SHARED_MEMORY_INITIALIZED.load(Ordering::Acquire) {
        return None;
    }

    unsafe {
        if SHARED_CACHES.is_null() || (*SHARED_CACHES).builder_neighbor_cache.is_null() {
            None
        } else {
            Some(&*(*SHARED_CACHES).builder_neighbor_cache)
        }
    }
}

/// Check if shared memory has been initialized
pub fn is_initialized() -> bool {
    SHARED_MEMORY_INITIALIZED.load(Ordering::Acquire)
}
