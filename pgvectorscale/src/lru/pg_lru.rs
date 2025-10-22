//! PostgreSQL-native shared memory LRU cache
//!
//! This implementation uses PostgreSQL's shared memory system and LWLocks
//! for proper cross-process synchronization. It's designed to work with
//! maintenance_work_mem for memory allocation.
//!
//! ## Memory Allocation Strategy
//!
//! Following pgvector's approach for HNSW index building, we use:
//! - **Single-process mode**: Uses `palloc` for memory allocation (current implementation)
//! - **Multi-process mode**: Would use DSM (Dynamic Shared Memory) with `shm_toc_allocate`
//!   similar to pgvector's parallel builds
//!
//! Within the allocated memory, we use a simple bump allocator that increments
//! `next_free_offset`, exactly like pgvector's `memoryUsed` approach. This avoids
//! the complexity of DSA (Dynamic Shared Areas) which pgvector also doesn't use.
//!
//! ## Limitations
//!
//! - **No memory reclamation**: Like pgvector, we don't free individual entries.
//!   This leads to fragmentation over time. A production implementation would need
//!   periodic compaction or a restart to reclaim memory.
//! - **Single-process only**: Current implementation uses `palloc` which is not
//!   shared across processes. For true cross-process support, would need to use
//!   DSM or conventional shared memory.
//! - **Fixed hash table size**: Uses 1024 buckets regardless of cache size.

use pgrx::pg_sys;
use pgrx::pg_sys::LWLockMode::{LW_EXCLUSIVE, LW_SHARED};
use pgrx::prelude::*;
use rkyv::{Archive, Deserialize, Serialize};
use std::mem::size_of;
use std::ptr;
use std::slice;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

/// Offset type for relative pointers in shared memory
type Offset = u32;

/// Invalid offset marker
const INVALID_OFFSET: Offset = 0;

/// Lock tranche ID for our LWLocks
static mut LRU_LOCK_TRANCHE_ID: i32 = 0;

/// Get or initialize the lock tranche ID
unsafe fn get_lock_tranche_id() -> i32 {
    if LRU_LOCK_TRANCHE_ID == 0 {
        initialize_lock_tranche();
    }
    LRU_LOCK_TRANCHE_ID
}

/// Initialize lock tranche for LRU cache locks
unsafe fn initialize_lock_tranche() {
    // Use a fixed tranche ID for our extension
    // In production, might want to register this properly with PostgreSQL
    // Following pgvector's approach which also uses a fixed ID
    LRU_LOCK_TRANCHE_ID = 1001; // Arbitrary ID for our extension
}

/// Shared memory header for the LRU cache
#[repr(C)]
struct LruSharedHeader {
    /// Lock for the entire structure
    structure_lock: pg_sys::LWLock,

    /// Head of LRU list (most recently used)
    head_offset: Offset,

    /// Tail of LRU list (least recently used)
    tail_offset: Offset,

    /// Current memory used
    memory_used: AtomicUsize,

    /// Total memory available
    memory_total: usize,

    /// Number of entries
    entry_count: AtomicUsize,

    /// Next free offset for bump allocation
    next_free_offset: AtomicUsize,

    /// Simple hash table for index (fixed size)
    /// We'll use a simple hash table with chaining
    hash_buckets: [Offset; 1024], // Fixed 1024 buckets
}

/// Entry in the shared memory LRU cache
#[repr(C)]
struct LruEntry {
    /// Per-entry lock
    lock: pg_sys::LWLock,

    /// Next entry in LRU list
    next_offset: Offset,

    /// Previous entry in LRU list
    prev_offset: Offset,

    /// Next entry in hash chain
    hash_next_offset: Offset,

    /// Hash of the key
    key_hash: u64,

    /// Size of serialized key
    key_size: u32,

    /// Size of serialized value
    value_size: u32,

    /// Pin count for safe access
    pin_count: AtomicU32,

    /// Whether entry is being deleted
    deleted: AtomicU32,

    /// Data follows (key then value)
    data: [u8; 0],
}

impl LruEntry {
    /// Get key bytes
    unsafe fn key_bytes(&self) -> &[u8] {
        slice::from_raw_parts(self.data.as_ptr(), self.key_size as usize)
    }

    /// Get value bytes
    unsafe fn value_bytes(&self) -> &[u8] {
        slice::from_raw_parts(
            self.data.as_ptr().add(self.key_size as usize),
            self.value_size as usize,
        )
    }

    /// Try to pin the entry
    fn try_pin(&self) -> bool {
        loop {
            let current = self.pin_count.load(Ordering::Acquire);
            if current == u32::MAX {
                return false; // Being deleted
            }

            match self.pin_count.compare_exchange_weak(
                current,
                current + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return true,
                Err(_) => continue,
            }
        }
    }

    /// Unpin the entry
    fn unpin(&self) {
        self.pin_count.fetch_sub(1, Ordering::Release);
    }
}

/// PostgreSQL shared memory LRU cache
pub struct PgSharedLru {
    /// Base pointer to shared memory
    base: *mut u8,

    /// Header in shared memory
    header: *mut LruSharedHeader,
}

impl PgSharedLru {
    /// Create a new shared memory LRU cache
    ///
    /// # Safety
    /// Must be called with appropriate PostgreSQL memory context
    pub unsafe fn new(maintenance_work_mem_kb: i64) -> PgSharedLru {
        let size = (maintenance_work_mem_kb * 1024) as usize;

        // Allocate memory using palloc for single-process use
        // For cross-process use, this would need DSM or conventional shared memory
        // Following pgvector's approach for serial builds
        let base = pg_sys::palloc(size as _) as *mut u8;
        if base.is_null() {
            error!("Failed to allocate {} bytes of memory", size);
        }

        Self::new_in_memory(base, size)
    }

    /// Create a handle to an existing shared memory LRU cache
    ///
    /// # Safety
    /// The memory at base must already be initialized with a valid cache
    pub unsafe fn from_existing(base: *mut u8, _size: usize) -> PgSharedLru {
        let header = base as *mut LruSharedHeader;

        pgrx::debug2!(
            "LRU from_existing: creating handle for base={:p}, header={:p}",
            base,
            header
        );
        pgrx::debug2!(
            "LRU from_existing: header state - head={}, tail={}, entry_count={}, memory_used={}",
            (*header).head_offset,
            (*header).tail_offset,
            (*header).entry_count.load(Ordering::Acquire),
            (*header).memory_used.load(Ordering::Acquire)
        );

        PgSharedLru { base, header }
    }

    /// Create a new shared memory LRU cache in pre-allocated memory
    ///
    /// # Safety
    /// The provided memory must be valid and large enough
    pub unsafe fn new_in_memory(base: *mut u8, size: usize) -> PgSharedLru {
        pgrx::debug2!(
            "LRU new_in_memory: initializing cache at base={:p}, size={}",
            base,
            size
        );

        // Initialize header
        let header = base as *mut LruSharedHeader;
        ptr::write_bytes(header, 0, 1);

        // Initialize locks
        let tranche_id = get_lock_tranche_id();
        pg_sys::LWLockInitialize(&mut (*header).structure_lock, tranche_id);

        // Initialize header fields
        (*header).head_offset = INVALID_OFFSET;
        (*header).tail_offset = INVALID_OFFSET;
        (*header).memory_used = AtomicUsize::new(size_of::<LruSharedHeader>());
        (*header).memory_total = size;
        (*header).entry_count = AtomicUsize::new(0);
        (*header).next_free_offset = AtomicUsize::new(size_of::<LruSharedHeader>());

        // Initialize hash buckets
        for bucket in &mut (*header).hash_buckets {
            *bucket = INVALID_OFFSET;
        }

        pgrx::debug2!("LRU new_in_memory: initialized cache at base={:p}, header={:p}, header_size={}, available={}",
            base, header, size_of::<LruSharedHeader>(), size - size_of::<LruSharedHeader>());

        PgSharedLru { base, header }
    }

    /// Convert offset to pointer
    unsafe fn offset_to_ptr<T>(&self, offset: Offset) -> *mut T {
        if offset == INVALID_OFFSET {
            ptr::null_mut()
        } else {
            self.base.add(offset as usize) as *mut T
        }
    }

    /// Hash a key
    fn hash_key<K: Archive + Serialize<rkyv::ser::serializers::AllocSerializer<256>>>(
        &self,
        key: &K,
    ) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let key_bytes = rkyv::to_bytes::<_, 256>(key).expect("Failed to serialize key");

        let mut hasher = DefaultHasher::new();
        key_bytes.hash(&mut hasher);
        hasher.finish()
    }

    /// Get a value from the cache
    ///
    /// # Safety
    /// Must be called with valid PostgreSQL memory context and proper initialization
    pub unsafe fn get<K, V>(&self, key: &K) -> Option<V>
    where
        K: Archive + Serialize<rkyv::ser::serializers::AllocSerializer<256>> + PartialEq,
        V: Archive,
        K::Archived: PartialEq<K>,
        V::Archived: Deserialize<V, rkyv::Infallible>,
    {
        let hash = self.hash_key(key);
        let bucket_idx = (hash % (*self.header).hash_buckets.len() as u64) as usize;

        // Serialize key for comparison
        let key_bytes = rkyv::to_bytes::<_, 256>(key).ok()?;

        pgrx::debug2!(
            "LRU get: hash={:x}, bucket={}, base={:p}, header={:p}",
            hash,
            bucket_idx,
            self.base,
            self.header
        );

        // Acquire shared lock for reading
        // We don't update LRU on reads to avoid lock contention and race conditions
        pg_sys::LWLockAcquire(
            &mut (*self.header).structure_lock as *mut _ as _,
            LW_SHARED as _,
        );

        let mut current_offset = (*self.header).hash_buckets[bucket_idx];
        pgrx::debug2!("LRU get: chain start offset={}", current_offset);

        while current_offset != INVALID_OFFSET {
            let entry = self.offset_to_ptr::<LruEntry>(current_offset);

            pgrx::debug2!(
                "LRU get: checking entry at offset={}, ptr={:p}, hash={:x}",
                current_offset,
                entry,
                (*entry).key_hash
            );

            // Check hash first
            if (*entry).key_hash == hash {
                // Check if keys match
                if (*entry).key_bytes() == key_bytes.as_ref() {
                    // Try to pin the entry to prevent eviction while we read
                    if (*entry).try_pin() {
                        pgrx::debug2!(
                            "LRU get: found and pinned entry at offset={}",
                            current_offset
                        );

                        // Copy value bytes while pinned and lock held
                        let value_bytes = (*entry).value_bytes();
                        let mut aligned_value = vec![0u8; value_bytes.len()];
                        aligned_value.copy_from_slice(value_bytes);

                        // Unpin
                        (*entry).unpin();

                        // Release lock
                        pg_sys::LWLockRelease(&mut (*self.header).structure_lock as *mut _ as _);

                        // Deserialize and return
                        let archived = rkyv::archived_root::<V>(&aligned_value);
                        return Some(archived.deserialize(&mut rkyv::Infallible).unwrap());
                    } else {
                        pgrx::debug2!(
                            "LRU get: entry at offset={} could not be pinned (being deleted)",
                            current_offset
                        );
                    }
                }
            }

            current_offset = (*entry).hash_next_offset;
        }

        pgrx::debug2!("LRU get: miss for hash={:x}", hash);
        pg_sys::LWLockRelease(&mut (*self.header).structure_lock as *mut _ as _);
        None
    }

    /// Insert a key-value pair into the cache
    ///
    /// # Safety
    /// Must be called with valid PostgreSQL memory context and proper initialization
    pub unsafe fn insert<K, V>(&self, key: K, value: V) -> Result<(), String>
    where
        K: Archive + Serialize<rkyv::ser::serializers::AllocSerializer<256>>,
        V: Archive + Serialize<rkyv::ser::serializers::AllocSerializer<256>>,
    {
        // Serialize key and value
        let key_bytes = rkyv::to_bytes::<_, 256>(&key)
            .map_err(|e| format!("Failed to serialize key: {}", e))?;
        let value_bytes = rkyv::to_bytes::<_, 256>(&value)
            .map_err(|e| format!("Failed to serialize value: {}", e))?;

        let entry_size = size_of::<LruEntry>() + key_bytes.len() + value_bytes.len();
        let hash = self.hash_key(&key);
        let bucket_idx = (hash % (*self.header).hash_buckets.len() as u64) as usize;

        pgrx::debug2!(
            "LRU insert: hash={:x}, bucket={}, entry_size={}, base={:p}",
            hash,
            bucket_idx,
            entry_size,
            self.base
        );

        // Check if we have space
        pg_sys::LWLockAcquire(
            &mut (*self.header).structure_lock as *mut _ as _,
            LW_EXCLUSIVE as _,
        );

        let before_free = (*self.header).next_free_offset.load(Ordering::Acquire);
        pgrx::debug2!(
            "LRU insert: before insert - next_free={}, total={}, need={}",
            before_free,
            (*self.header).memory_total,
            entry_size
        );

        // Check if we need to evict entries to make room
        // Use memory_used instead of next_free_offset because bump allocation
        // means next_free_offset never decreases, but eviction does decrease memory_used
        while (*self.header).memory_used.load(Ordering::Acquire) + entry_size
            > (*self.header).memory_total
        {
            pgrx::debug2!(
                "LRU insert: need to evict - free={}, total={}, need={}",
                (*self.header).next_free_offset.load(Ordering::Acquire),
                (*self.header).memory_total,
                entry_size
            );

            if !self.evict_lru() {
                pg_sys::LWLockRelease(&mut (*self.header).structure_lock as *mut _ as _);
                return Err("Cannot evict enough memory - all entries are pinned".to_string());
            }
        }

        // Simple bump allocation
        let offset = (*self.header)
            .next_free_offset
            .fetch_add(entry_size, Ordering::SeqCst);

        pgrx::debug2!(
            "LRU insert: allocated at offset={}, size={}",
            offset,
            entry_size
        );

        // Initialize entry
        let entry = self.offset_to_ptr::<LruEntry>(offset as Offset);
        ptr::write_bytes(entry, 0, 1);

        // Initialize lock
        let tranche_id = get_lock_tranche_id();
        pg_sys::LWLockInitialize(&mut (*entry).lock, tranche_id);

        // Set entry fields
        (*entry).key_hash = hash;
        (*entry).key_size = key_bytes.len() as u32;
        (*entry).value_size = value_bytes.len() as u32;
        (*entry).pin_count = AtomicU32::new(0);
        (*entry).deleted = AtomicU32::new(0);

        // Copy data
        ptr::copy_nonoverlapping(
            key_bytes.as_ptr(),
            (*entry).data.as_mut_ptr(),
            key_bytes.len(),
        );
        ptr::copy_nonoverlapping(
            value_bytes.as_ptr(),
            (*entry).data.as_mut_ptr().add(key_bytes.len()),
            value_bytes.len(),
        );

        // Add to hash chain
        (*entry).hash_next_offset = (*self.header).hash_buckets[bucket_idx];
        (*self.header).hash_buckets[bucket_idx] = offset as Offset;

        // Add to head of LRU list
        (*entry).prev_offset = INVALID_OFFSET;
        (*entry).next_offset = (*self.header).head_offset;

        if (*self.header).head_offset != INVALID_OFFSET {
            let old_head = self.offset_to_ptr::<LruEntry>((*self.header).head_offset);
            (*old_head).prev_offset = offset as Offset;
        }

        (*self.header).head_offset = offset as Offset;

        if (*self.header).tail_offset == INVALID_OFFSET {
            (*self.header).tail_offset = offset as Offset;
        }

        (*self.header).entry_count.fetch_add(1, Ordering::Release);
        (*self.header)
            .memory_used
            .fetch_add(entry_size, Ordering::Release);

        pg_sys::LWLockRelease(&mut (*self.header).structure_lock as *mut _ as _);

        Ok(())
    }

    /// Evict the least recently used entry
    /// Returns true if an entry was evicted, false if all entries are pinned
    unsafe fn evict_lru(&self) -> bool {
        // Must be called with exclusive lock held
        let tail_offset = (*self.header).tail_offset;

        pgrx::debug2!(
            "LRU evict: attempting eviction, tail_offset={}",
            tail_offset
        );

        if tail_offset == INVALID_OFFSET {
            pgrx::debug2!("LRU evict: cache is empty, cannot evict");
            return false; // Empty cache
        }

        let tail_entry = self.offset_to_ptr::<LruEntry>(tail_offset);
        let pin_count = (*tail_entry).pin_count.load(Ordering::Acquire);

        pgrx::debug2!(
            "LRU evict: tail entry at offset={}, ptr={:p}, pin_count={}, hash={:x}",
            tail_offset,
            tail_entry,
            pin_count,
            (*tail_entry).key_hash
        );

        // Check if entry is pinned
        if pin_count > 0 {
            pgrx::debug2!(
                "LRU evict: tail entry is pinned (count={}), cannot evict",
                pin_count
            );
            // Can't evict pinned entries
            // In a more sophisticated implementation, we'd walk the list
            // to find an unpinned entry
            return false;
        }

        // Remove from LRU list
        let prev_offset = (*tail_entry).prev_offset;
        pgrx::debug2!(
            "LRU evict: removing from LRU list, prev_offset={}",
            prev_offset
        );

        (*self.header).tail_offset = prev_offset;

        if prev_offset != INVALID_OFFSET {
            let prev = self.offset_to_ptr::<LruEntry>(prev_offset);
            (*prev).next_offset = INVALID_OFFSET;
        } else {
            // Was the only entry
            pgrx::debug2!("LRU evict: was the only entry, clearing head");
            (*self.header).head_offset = INVALID_OFFSET;
        }

        // Remove from hash chain
        let hash = (*tail_entry).key_hash;
        let bucket_idx = (hash % (*self.header).hash_buckets.len() as u64) as usize;
        let mut current_offset = (*self.header).hash_buckets[bucket_idx];
        let mut prev_hash_offset = INVALID_OFFSET;

        pgrx::debug2!(
            "LRU evict: removing from hash bucket {}, chain start={}",
            bucket_idx,
            current_offset
        );

        while current_offset != INVALID_OFFSET {
            if current_offset == tail_offset {
                // Found it, unlink from hash chain
                if prev_hash_offset == INVALID_OFFSET {
                    (*self.header).hash_buckets[bucket_idx] = (*tail_entry).hash_next_offset;
                } else {
                    let prev = self.offset_to_ptr::<LruEntry>(prev_hash_offset);
                    (*prev).hash_next_offset = (*tail_entry).hash_next_offset;
                }
                pgrx::debug2!("LRU evict: removed from hash chain");
                break;
            }

            let entry = self.offset_to_ptr::<LruEntry>(current_offset);
            prev_hash_offset = current_offset;
            current_offset = (*entry).hash_next_offset;
        }

        // Update counts
        (*self.header).entry_count.fetch_sub(1, Ordering::Release);
        let entry_size = size_of::<LruEntry>()
            + (*tail_entry).key_size as usize
            + (*tail_entry).value_size as usize;
        (*self.header)
            .memory_used
            .fetch_sub(entry_size, Ordering::Release);

        pgrx::debug2!(
            "LRU evict: successfully evicted entry at offset={}, freed {} bytes",
            tail_offset,
            entry_size
        );

        // Note: In a real implementation with proper memory management,
        // we'd need to handle memory reclamation here. With bump allocation,
        // we can't actually free individual entries, so memory gets fragmented.
        // A production implementation would need a more sophisticated allocator
        // or periodic compaction.

        true
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        unsafe {
            CacheStats {
                entry_count: (*self.header).entry_count.load(Ordering::Acquire),
                memory_used: (*self.header).memory_used.load(Ordering::Acquire),
                memory_total: (*self.header).memory_total,
            }
        }
    }
}

/// Cache statistics
#[derive(Debug)]
pub struct CacheStats {
    pub entry_count: usize,
    pub memory_used: usize,
    pub memory_total: usize,
}

// Mark as no_mangle so it can be called from C
#[no_mangle]
pub extern "C" fn pgvectorscale_lru_init() {
    unsafe {
        initialize_lock_tranche();
    }
}
