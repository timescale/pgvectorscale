use std::collections::HashMap;
use std::hash::Hash;
use std::mem::size_of;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};

#[cfg(not(test))]
use pgrx::{debug1, warning};
use rkyv::ser::serializers::AllocSerializer;
use rkyv::{Archive, Deserialize, Serialize};

use crate::lru::allocator::{AllocError, SharedMemoryAllocator};
use crate::lru::entry::{EntryHandle, PinnedEntry, SharedLruEntry};
use crate::lru::stats::CacheStats;

/// Error types for LRU operations
#[derive(Debug, Clone)]
pub enum LruError {
    AllocationFailed(AllocError),
    SerializationFailed,
    DeserializationFailed,
    EntryTooLarge,
    CacheFull,
}

impl From<AllocError> for LruError {
    fn from(err: AllocError) -> Self {
        LruError::AllocationFailed(err)
    }
}

impl std::fmt::Display for LruError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LruError::AllocationFailed(e) => write!(f, "Allocation failed: {}", e),
            LruError::SerializationFailed => write!(f, "Serialization failed"),
            LruError::DeserializationFailed => write!(f, "Deserialization failed"),
            LruError::EntryTooLarge => write!(f, "Entry too large for cache"),
            LruError::CacheFull => write!(f, "Cache is full and cannot evict"),
        }
    }
}

impl std::error::Error for LruError {}

/// Trait for handling eviction events
pub trait EvictionHandler<K, V>: Send + Sync {
    fn on_evict(&self, key: K, value: V);
}

/// Main LRU cache implementation for shared memory
pub struct SharedMemoryLru<K, V, A>
where
    K: Archive + Serialize<AllocSerializer<256>> + Hash + Eq,
    V: Archive + Serialize<AllocSerializer<256>>,
    A: SharedMemoryAllocator,
{
    /// The allocator for shared memory
    allocator: Arc<A>,

    /// Index mapping keys to shared memory locations
    index: Arc<RwLock<HashMap<K, EntryHandle<A>>>>,

    /// Head of LRU list (most recently used)
    lru_head: Arc<RwLock<Option<A::Pointer>>>,

    /// Tail of LRU list (least recently used)
    lru_tail: Arc<RwLock<Option<A::Pointer>>>,

    /// Current size in bytes
    current_size: Arc<AtomicUsize>,

    /// Maximum size in bytes
    capacity: usize,

    /// Optional eviction handler
    eviction_handler: Option<Arc<dyn EvictionHandler<K, V>>>,

    /// Statistics
    stats: Arc<CacheStats>,

    /// Name for debugging
    #[allow(dead_code)]
    cache_name: String,

    /// Condition variable for unpin notifications
    unpin_notify: Arc<Condvar>,

    /// Mutex for condition variable (paired with unpin_notify)
    #[allow(dead_code)]
    unpin_mutex: Arc<Mutex<()>>,
}

// Basic methods without Deserialize requirement
impl<K, V, A> SharedMemoryLru<K, V, A>
where
    K: Archive + Serialize<AllocSerializer<256>> + Hash + Eq,
    V: Archive + Serialize<AllocSerializer<256>>,
    A: SharedMemoryAllocator,
{
    /// Get current number of entries
    pub fn len(&self) -> usize {
        self.index.read().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get current size in bytes
    pub fn size(&self) -> usize {
        self.current_size.load(Ordering::Relaxed)
    }

    /// Get capacity in bytes
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get cache statistics
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }
}

impl<K, V, A> SharedMemoryLru<K, V, A>
where
    K: Archive + Serialize<AllocSerializer<256>> + Hash + Eq + Clone,
    V: Archive + Serialize<AllocSerializer<256>>,
    K::Archived: Deserialize<K, rkyv::Infallible>,
    V::Archived: Deserialize<V, rkyv::Infallible>,
    A: SharedMemoryAllocator,
{
    /// Create a new shared memory LRU cache
    pub fn new(
        allocator: A,
        capacity: usize,
        cache_name: String,
        eviction_handler: Option<Arc<dyn EvictionHandler<K, V>>>,
    ) -> Self {
        Self {
            allocator: Arc::new(allocator),
            index: Arc::new(RwLock::new(HashMap::new())),
            lru_head: Arc::new(RwLock::new(None)),
            lru_tail: Arc::new(RwLock::new(None)),
            current_size: Arc::new(AtomicUsize::new(0)),
            capacity,
            eviction_handler,
            stats: Arc::new(CacheStats::new()),
            cache_name,
            unpin_notify: Arc::new(Condvar::new()),
            unpin_mutex: Arc::new(Mutex::new(())),
        }
    }

    /// Get a value from the cache
    pub fn get(&self, key: &K) -> Option<PinnedEntry<'_, K, V, A>> {
        // Read lock on index
        let index = self.index.read().unwrap();
        let handle = match index.get(key) {
            Some(h) => h,
            None => {
                drop(index);
                self.stats.record_miss();
                return None;
            }
        };
        let ptr = handle.ptr;
        drop(index);

        // Try to pin the entry
        unsafe {
            let entry = self.allocator.get_address(ptr) as *const SharedLruEntry<A>;
            if entry.is_null() {
                // Entry was already deallocated (race condition)
                self.stats.record_miss();
                return None;
            }
            if !(*entry).try_pin() {
                // Entry is being deleted
                self.stats.record_miss();
                return None;
            }
            self.stats.record_pin();
        }

        // Update LRU position
        self.move_to_head(ptr);

        self.stats.record_hit();

        // Return pinned entry
        Some(unsafe {
            PinnedEntry::new(
                ptr,
                &self.allocator,
                self.stats.clone(),
                self.unpin_notify.clone(),
            )
        })
    }

    /// Insert or update a value in the cache
    pub fn insert(&self, key: K, value: V) -> Result<(), LruError> {
        // Serialize key and value
        let key_bytes =
            rkyv::to_bytes::<_, 256>(&key).map_err(|_| LruError::SerializationFailed)?;
        let value_bytes =
            rkyv::to_bytes::<_, 256>(&value).map_err(|_| LruError::SerializationFailed)?;

        let key_size = key_bytes.len();
        let value_size = value_bytes.len();
        let entry_size = size_of::<SharedLruEntry<A>>() + key_size + value_size;

        // Check if entry is too large
        if entry_size > self.capacity {
            return Err(LruError::EntryTooLarge);
        }

        // Allocate shared memory
        let ptr = self.allocator.allocate(entry_size)?;

        // Initialize entry in shared memory
        unsafe {
            let entry = self.allocator.get_address(ptr) as *mut SharedLruEntry<A>;
            (*entry).key_size = key_size as u32;
            (*entry).value_size = value_size as u32;
            (*entry).total_size = entry_size as u32;
            (*entry).next = None;
            (*entry).prev = None;
            (*entry).pin_count = AtomicU32::new(0);

            // Copy serialized data
            let data_ptr = (*entry).data.as_mut_ptr();
            std::ptr::copy_nonoverlapping(key_bytes.as_ptr(), data_ptr, key_size);
            std::ptr::copy_nonoverlapping(value_bytes.as_ptr(), data_ptr.add(key_size), value_size);
        }

        // Update index and handle evictions
        let mut index = self.index.write().unwrap();

        // Evict entries if needed
        while self.current_size.load(Ordering::Relaxed) + entry_size > self.capacity {
            if !self.try_evict_lru(&mut index)? {
                // Could not evict anything (all pinned?)
                unsafe {
                    self.allocator.deallocate(ptr);
                }
                return Err(LruError::CacheFull);
            }
        }

        // Check if key already exists
        if let Some(old_handle) = index.insert(key.clone(), EntryHandle::new(ptr, entry_size)) {
            // Try to mark old entry for deletion
            unsafe {
                let old_entry =
                    self.allocator.get_address(old_handle.ptr) as *const SharedLruEntry<A>;
                if self.allocator.is_valid(old_handle.ptr) && (*old_entry).mark_for_deletion() {
                    // Entry was not pinned, safe to remove and deallocate
                    self.remove_from_list(old_handle.ptr);
                    self.current_size
                        .fetch_sub(old_handle.size, Ordering::Relaxed);
                    self.allocator.deallocate(old_handle.ptr);
                } else {
                    // Entry is pinned, just remove from list but don't deallocate
                    // The pinned entry will be deallocated when unpinned
                    self.remove_from_list(old_handle.ptr);
                    self.current_size
                        .fetch_sub(old_handle.size, Ordering::Relaxed);
                }
            }
            self.stats.record_update();
        } else {
            self.stats.record_insert();
        }

        // Add to head of LRU list
        self.add_to_head(ptr);
        self.current_size.fetch_add(entry_size, Ordering::Relaxed);

        // Warn on first eviction
        #[cfg(not(test))]
        {
            let evictions = self.stats.evictions.load(Ordering::Relaxed);
            if evictions == 1 {
                warning!(
                    "{} cache is full after processing {} items; consider increasing memory",
                    self.cache_name,
                    self.stats.inserts.load(Ordering::Relaxed)
                );
            }
        }

        Ok(())
    }

    /// Check if cache contains a key (doesn't update LRU)
    pub fn contains(&self, key: &K) -> bool {
        let index = self.index.read().unwrap();
        let result = index.contains_key(key);
        if result {
            self.stats.record_hit();
        } else {
            self.stats.record_miss();
        }
        result
    }

    /// Try to evict the LRU entry
    /// Returns true if an entry was evicted, false if nothing could be evicted
    fn try_evict_lru(&self, index: &mut HashMap<K, EntryHandle<A>>) -> Result<bool, LruError> {
        // Hold write locks during eviction to prevent concurrent modifications
        let mut head = self.lru_head.write().unwrap();
        let mut tail = self.lru_tail.write().unwrap();

        let tail_ptr = match *tail {
            Some(ptr) => ptr,
            None => return Ok(false), // Empty cache
        };

        // Check if entry is pinned
        unsafe {
            let entry = self.allocator.get_address(tail_ptr) as *const SharedLruEntry<A>;
            if (*entry).is_pinned() {
                // Try to find an unpinned entry to evict
                drop(head);
                drop(tail);
                return self.find_and_evict_unpinned(index);
            }

            // Mark for deletion to prevent new pins
            if !(*entry).mark_for_deletion() {
                // Entry got pinned while we were checking, try another
                drop(head);
                drop(tail);
                return self.find_and_evict_unpinned(index);
            }

            // Safe to evict this entry
            self.evict_entry_locked(tail_ptr, index, &mut *head, &mut *tail)?;
        }

        Ok(true)
    }

    /// Find an unpinned entry and evict it
    fn find_and_evict_unpinned(
        &self,
        index: &mut HashMap<K, EntryHandle<A>>,
    ) -> Result<bool, LruError> {
        // Hold write locks during traversal to prevent concurrent list modifications
        let mut _head = self.lru_head.write().unwrap();
        let mut _tail = self.lru_tail.write().unwrap();

        // Traverse from tail to head looking for unpinned entry
        let mut current = *_tail;

        while let Some(ptr) = current {
            unsafe {
                let entry = self.allocator.get_address(ptr) as *const SharedLruEntry<A>;

                // Get next before potentially evicting this entry
                let next = (*entry).next;

                if !(*entry).is_pinned() && (*entry).mark_for_deletion() {
                    // Found unpinned entry, evict it
                    self.evict_entry_locked(ptr, index, &mut *_head, &mut *_tail)?;
                    return Ok(true);
                }

                // Move to next (more recently used) entry
                current = next;
            }
        }

        // All entries are pinned
        Ok(false)
    }

    /// Evict a specific entry (must already be marked for deletion)
    /// Assumes LRU locks are already held by caller
    unsafe fn evict_entry_locked(
        &self,
        ptr: A::Pointer,
        index: &mut HashMap<K, EntryHandle<A>>,
        head: &mut Option<A::Pointer>,
        tail: &mut Option<A::Pointer>,
    ) -> Result<(), LruError> {
        let entry = self.allocator.get_address(ptr) as *mut SharedLruEntry<A>;

        // CRITICAL: Remove from LRU list FIRST to prevent other threads
        // from finding this entry while we're evicting it
        self.remove_from_list_locked(ptr, head, tail);

        // Find the key in the index by searching for the matching pointer
        let key_to_remove = index
            .iter()
            .find(|(_, handle)| handle.ptr == ptr)
            .map(|(k, _)| k.clone());

        if let Some(key) = key_to_remove {
            // Remove from index AFTER removing from LRU list
            index.remove(&key);

            // Call eviction handler if present
            if let Some(ref handler) = self.eviction_handler {
                // Get archived key and value
                let key_bytes = (*entry).key_bytes();
                let value_bytes = (*entry).value_bytes();

                // Copy to aligned buffers before deserializing
                // This is necessary because our data might not be aligned in shared memory
                let mut aligned_key = vec![0u8; key_bytes.len()];
                let mut aligned_value = vec![0u8; value_bytes.len()];
                aligned_key.copy_from_slice(key_bytes);
                aligned_value.copy_from_slice(value_bytes);

                // Now we can safely deserialize from aligned buffers
                let archived_key = rkyv::archived_root::<K>(&aligned_key);
                let archived_value = rkyv::archived_root::<V>(&aligned_value);

                // Deserialize to get the actual types
                let deserialized_key: K = archived_key
                    .deserialize(&mut rkyv::Infallible)
                    .expect("Infallible deserializer should not fail");
                let deserialized_value: V = archived_value
                    .deserialize(&mut rkyv::Infallible)
                    .expect("Infallible deserializer should not fail");

                handler.on_evict(deserialized_key, deserialized_value);
            }
        }

        // Update size
        let size = (*entry).total_size as usize;
        self.current_size.fetch_sub(size, Ordering::Relaxed);

        // Deallocate memory LAST, after all references are gone
        self.allocator.deallocate(ptr);

        self.stats.record_eviction();

        Ok(())
    }

    /// Move an entry to the head of the LRU list
    fn move_to_head(&self, ptr: A::Pointer) {
        // Hold locks for the entire operation to make it atomic
        let mut head = self.lru_head.write().unwrap();
        let mut tail = self.lru_tail.write().unwrap();

        unsafe {
            // Remove from current position
            self.remove_from_list_locked(ptr, &mut *head, &mut *tail);

            // Check if entry is still valid after removal
            if !self.allocator.is_valid(ptr) {
                // Entry was deallocated, nothing more to do
                return;
            }

            // Add to head
            let entry = self.allocator.get_address(ptr) as *mut SharedLruEntry<A>;

            (*entry).prev = None;
            (*entry).next = *head;

            if let Some(old_head) = *head {
                let old_head_entry = self.allocator.get_address(old_head) as *mut SharedLruEntry<A>;
                (*old_head_entry).prev = Some(ptr);
            }

            *head = Some(ptr);

            if tail.is_none() {
                *tail = Some(ptr);
            }
        }
    }

    /// Add an entry to the head of the LRU list
    fn add_to_head(&self, ptr: A::Pointer) {
        let mut head = self.lru_head.write().unwrap();
        let mut tail = self.lru_tail.write().unwrap();

        unsafe {
            let entry = self.allocator.get_address(ptr) as *mut SharedLruEntry<A>;

            (*entry).prev = None;
            (*entry).next = *head;

            if let Some(old_head) = *head {
                let old_head_entry = self.allocator.get_address(old_head) as *mut SharedLruEntry<A>;
                (*old_head_entry).prev = Some(ptr);
            }

            *head = Some(ptr);

            if tail.is_none() {
                *tail = Some(ptr);
            }
        }
    }

    /// Remove an entry from the LRU list (assumes locks are held)
    unsafe fn remove_from_list_locked(
        &self,
        ptr: A::Pointer,
        head: &mut Option<A::Pointer>,
        tail: &mut Option<A::Pointer>,
    ) {
        // Check if the entry is still valid (not deallocated)
        if !self.allocator.is_valid(ptr) {
            // Entry was already deallocated by another thread, nothing to do
            return;
        }

        let entry = self.allocator.get_address(ptr) as *mut SharedLruEntry<A>;
        let prev = (*entry).prev;
        let next = (*entry).next;

        // Update previous entry
        if let Some(prev_ptr) = prev {
            // Only update if the previous entry is still valid
            if self.allocator.is_valid(prev_ptr) {
                let prev_entry = self.allocator.get_address(prev_ptr) as *mut SharedLruEntry<A>;
                (*prev_entry).next = next;
            }
        } else {
            // This was the head
            *head = next;
        }

        // Update next entry
        if let Some(next_ptr) = next {
            // Only update if the next entry is still valid
            if self.allocator.is_valid(next_ptr) {
                let next_entry = self.allocator.get_address(next_ptr) as *mut SharedLruEntry<A>;
                (*next_entry).prev = prev;
            }
        } else {
            // This was the tail
            *tail = prev;
        }
    }

    /// Remove an entry from the LRU list
    fn remove_from_list(&self, ptr: A::Pointer) {
        let mut head = self.lru_head.write().unwrap();
        let mut tail = self.lru_tail.write().unwrap();
        unsafe {
            self.remove_from_list_locked(ptr, &mut *head, &mut *tail);
        }
    }
}

impl<K, V, A> Drop for SharedMemoryLru<K, V, A>
where
    K: Archive + Serialize<AllocSerializer<256>> + Hash + Eq,
    V: Archive + Serialize<AllocSerializer<256>>,
    A: SharedMemoryAllocator,
{
    fn drop(&mut self) {
        #[cfg(not(test))]
        {
            let stats = self.stats.snapshot();
            debug1!(
                "{} cache teardown: capacity {}, entries {}, size {}, stats: {:?}",
                self.cache_name,
                self.capacity,
                self.len(),
                self.size(),
                stats
            );
        }
    }
}

// SharedMemoryLru is Send if all components are Send
unsafe impl<K, V, A> Send for SharedMemoryLru<K, V, A>
where
    K: Archive + Serialize<AllocSerializer<256>> + Hash + Eq + Send,
    V: Archive + Serialize<AllocSerializer<256>> + Send,
    A: SharedMemoryAllocator,
    A::Pointer: Send,
{
}

// SharedMemoryLru is Sync if all components are Sync
unsafe impl<K, V, A> Sync for SharedMemoryLru<K, V, A>
where
    K: Archive + Serialize<AllocSerializer<256>> + Hash + Eq + Sync,
    V: Archive + Serialize<AllocSerializer<256>> + Sync,
    A: SharedMemoryAllocator,
    A::Pointer: Sync,
{
}
