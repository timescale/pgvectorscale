use std::marker::PhantomData;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Condvar};

use rkyv::{Archive, Archived};

use crate::lru::allocator::SharedMemoryAllocator;
use crate::lru::stats::CacheStats;

/// Entry stored in shared memory
#[repr(C)]
pub struct SharedLruEntry<A: SharedMemoryAllocator> {
    /// Size of serialized key
    pub key_size: u32,
    /// Size of serialized value
    pub value_size: u32,
    /// Total size of this allocation
    pub total_size: u32,
    /// Padding for alignment
    _padding: u32,

    /// Next entry in LRU list
    pub next: Option<A::Pointer>,
    /// Previous entry in LRU list
    pub prev: Option<A::Pointer>,

    /// Reference count for safe access
    pub pin_count: AtomicU32,

    /// Serialized data follows (key then value)
    pub data: [u8; 0],
}

impl<A: SharedMemoryAllocator> SharedLruEntry<A> {
    /// Get pointer to key data
    ///
    /// # Safety
    /// Caller must ensure the entry is still valid and not deallocated
    pub unsafe fn key_ptr(&self) -> *const u8 {
        self.data.as_ptr()
    }

    /// Get pointer to value data
    ///
    /// # Safety
    /// Caller must ensure the entry is still valid and not deallocated
    pub unsafe fn value_ptr(&self) -> *const u8 {
        self.data.as_ptr().add(self.key_size as usize)
    }

    /// Get key bytes
    ///
    /// # Safety
    /// Caller must ensure the entry is still valid and not deallocated
    pub unsafe fn key_bytes(&self) -> &[u8] {
        std::slice::from_raw_parts(self.key_ptr(), self.key_size as usize)
    }

    /// Get value bytes
    ///
    /// # Safety
    /// Caller must ensure the entry is still valid and not deallocated
    pub unsafe fn value_bytes(&self) -> &[u8] {
        std::slice::from_raw_parts(self.value_ptr(), self.value_size as usize)
    }

    /// Check if entry is pinned
    pub fn is_pinned(&self) -> bool {
        self.pin_count.load(Ordering::Acquire) > 0
    }

    /// Try to pin the entry
    /// Returns true if successfully pinned, false if entry is being deleted
    pub fn try_pin(&self) -> bool {
        // Use a CAS loop to increment pin count
        // But fail if pin count is u32::MAX (marker for deletion)
        loop {
            let current = self.pin_count.load(Ordering::Acquire);
            if current == u32::MAX {
                return false; // Entry is being deleted
            }
            if self
                .pin_count
                .compare_exchange_weak(current, current + 1, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return true;
            }
        }
    }

    /// Unpin the entry
    pub fn unpin(&self) {
        let old = self.pin_count.fetch_sub(1, Ordering::Release);
        debug_assert!(old > 0 && old != u32::MAX);
    }

    /// Mark entry for deletion (prevents new pins)
    pub fn mark_for_deletion(&self) -> bool {
        // Set pin count to MAX to prevent new pins
        // Return true if we can delete immediately (was 0)
        self.pin_count
            .compare_exchange(0, u32::MAX, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }
}

/// RAII wrapper for pinned cache entries
pub struct PinnedEntry<'a, K, V, A>
where
    K: Archive,
    V: Archive,
    A: SharedMemoryAllocator,
{
    entry_ptr: A::Pointer,
    allocator: &'a A,
    stats: Arc<CacheStats>,
    unpin_notify: Arc<Condvar>,
    _phantom: PhantomData<(K, V)>,
}

impl<'a, K, V, A> PinnedEntry<'a, K, V, A>
where
    K: Archive,
    V: Archive,
    A: SharedMemoryAllocator,
{
    /// Create a new pinned entry
    ///
    /// # Safety
    /// The entry_ptr must point to a valid SharedLruEntry that has been successfully pinned
    pub(crate) unsafe fn new(
        entry_ptr: A::Pointer,
        allocator: &'a A,
        stats: Arc<CacheStats>,
        unpin_notify: Arc<Condvar>,
    ) -> Self {
        Self {
            entry_ptr,
            allocator,
            stats,
            unpin_notify,
            _phantom: PhantomData,
        }
    }

    /// Get the archived value without deserializing
    pub fn get(&self) -> &Archived<V> {
        unsafe {
            let entry = self.allocator.get_address(self.entry_ptr) as *const SharedLruEntry<A>;
            let value_bytes = (*entry).value_bytes();
            rkyv::archived_root::<V>(value_bytes)
        }
    }

    /// Get the archived key without deserializing
    pub fn key(&self) -> &Archived<K> {
        unsafe {
            let entry = self.allocator.get_address(self.entry_ptr) as *const SharedLruEntry<A>;
            let key_ptr = (*entry).key_ptr();
            &*(key_ptr as *const Archived<K>)
        }
    }

    /// Get raw value bytes
    pub fn value_bytes(&self) -> &[u8] {
        unsafe {
            let entry = self.allocator.get_address(self.entry_ptr) as *const SharedLruEntry<A>;
            (*entry).value_bytes()
        }
    }

    /// Get raw key bytes
    pub fn key_bytes(&self) -> &[u8] {
        unsafe {
            let entry = self.allocator.get_address(self.entry_ptr) as *const SharedLruEntry<A>;
            (*entry).key_bytes()
        }
    }
}

impl<'a, K, V, A> Drop for PinnedEntry<'a, K, V, A>
where
    K: Archive,
    V: Archive,
    A: SharedMemoryAllocator,
{
    fn drop(&mut self) {
        // Only unpin if the entry is still valid
        if self.allocator.is_valid(self.entry_ptr) {
            unsafe {
                let entry = self.allocator.get_address(self.entry_ptr) as *const SharedLruEntry<A>;
                (*entry).unpin();
            }
        }
        // Notify cache that an entry was unpinned
        self.stats.record_unpin();
        self.unpin_notify.notify_all();
    }
}

// PinnedEntry should be Send if the types are Send
unsafe impl<'a, K, V, A> Send for PinnedEntry<'a, K, V, A>
where
    K: Archive + Send,
    V: Archive + Send,
    A: SharedMemoryAllocator,
    A::Pointer: Send,
{
}

// PinnedEntry should be Sync if the types are Sync
unsafe impl<'a, K, V, A> Sync for PinnedEntry<'a, K, V, A>
where
    K: Archive + Sync,
    V: Archive + Sync,
    A: SharedMemoryAllocator,
    A::Pointer: Sync,
{
}

/// Handle to an entry in the cache index
#[derive(Clone, Debug)]
pub struct EntryHandle<A: SharedMemoryAllocator> {
    pub ptr: A::Pointer,
    pub size: usize,
}

impl<A: SharedMemoryAllocator> EntryHandle<A> {
    pub fn new(ptr: A::Pointer, size: usize) -> Self {
        Self { ptr, size }
    }
}
