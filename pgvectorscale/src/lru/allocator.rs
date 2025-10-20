use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};

#[derive(Debug, Clone)]
pub enum AllocError {
    OutOfMemory,
    InvalidPointer,
    AllocationTooLarge,
}

impl std::fmt::Display for AllocError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AllocError::OutOfMemory => write!(f, "Out of memory"),
            AllocError::InvalidPointer => write!(f, "Invalid pointer"),
            AllocError::AllocationTooLarge => write!(f, "Allocation too large"),
        }
    }
}

impl std::error::Error for AllocError {}

/// Trait for different shared memory allocation strategies
pub trait SharedMemoryAllocator: Send + Sync {
    /// Pointer type used by this allocator
    type Pointer: Copy + Clone + Debug + Eq + Hash + Send + Sync;

    /// Allocate memory of given size
    fn allocate(&self, size: usize) -> Result<Self::Pointer, AllocError>;

    /// Free previously allocated memory
    ///
    /// # Safety
    /// Caller must ensure the pointer was allocated by this allocator
    /// and is not being used elsewhere
    unsafe fn deallocate(&self, ptr: Self::Pointer);

    /// Get raw pointer to allocated memory
    ///
    /// # Safety
    /// Caller must ensure the pointer is valid and allocated by this allocator
    unsafe fn get_address(&self, ptr: Self::Pointer) -> *mut u8;

    /// Get size of allocation (if tracked by allocator)
    ///
    /// # Safety
    /// Caller must ensure the pointer is valid and allocated by this allocator
    unsafe fn get_size(&self, ptr: Self::Pointer) -> Option<usize>;

    /// Check if a pointer is still valid (not deallocated)
    fn is_valid(&self, ptr: Self::Pointer) -> bool;
}

// === DSA (Dynamic Shared Area) Implementation ===
// TODO: Re-enable when integrating with actual Postgres
/*
/// Type alias for DSA pointers
pub type DsaPointer = pg_sys::dsa_pointer;

/// Invalid DSA pointer value
pub const INVALID_DSA_POINTER: DsaPointer = 0;

/// Allocator using Postgres Dynamic Shared Areas
pub struct DsaAllocator {
    dsa: *mut pg_sys::dsa_area,
}

unsafe impl Send for DsaAllocator {}
unsafe impl Sync for DsaAllocator {}

impl DsaAllocator {
    /// Create a new DSA allocator
    ///
    /// # Safety
    /// The dsa_area pointer must be valid for the lifetime of this allocator
    pub unsafe fn new(dsa: *mut pg_sys::dsa_area) -> Self {
        Self { dsa }
    }
}

impl SharedMemoryAllocator for DsaAllocator {
    type Pointer = DsaPointer;

    fn allocate(&self, size: usize) -> Result<DsaPointer, AllocError> {
        unsafe {
            let ptr = pg_sys::dsa_allocate(self.dsa, size);
            if ptr == INVALID_DSA_POINTER {
                Err(AllocError::OutOfMemory)
            } else {
                Ok(ptr)
            }
        }
    }

    unsafe fn deallocate(&self, ptr: DsaPointer) {
        pg_sys::dsa_free(self.dsa, ptr);
    }

    unsafe fn get_address(&self, ptr: DsaPointer) -> *mut u8 {
        pg_sys::dsa_get_address(self.dsa, ptr) as *mut u8
    }

    unsafe fn get_size(&self, _ptr: DsaPointer) -> Option<usize> {
        None // DSA doesn't track sizes
    }
}
*/

// === Conventional Shared Memory Implementation ===

/// Allocator using conventional shared memory with bump allocation
pub struct ShmemAllocator {
    base: *mut u8,
    size: usize,
    offset: AtomicUsize,
}

unsafe impl Send for ShmemAllocator {}
unsafe impl Sync for ShmemAllocator {}

impl ShmemAllocator {
    /// Create a new shared memory allocator
    ///
    /// # Safety
    /// The base pointer must point to valid shared memory of at least `size` bytes
    pub unsafe fn new(base: *mut u8, size: usize) -> Self {
        Self {
            base,
            size,
            offset: AtomicUsize::new(0),
        }
    }

    /// Get current offset (for debugging/monitoring)
    pub fn current_offset(&self) -> usize {
        self.offset.load(Ordering::Relaxed)
    }

    /// Get remaining space
    pub fn remaining(&self) -> usize {
        let offset = self.offset.load(Ordering::Relaxed);
        self.size.saturating_sub(offset)
    }
}

impl SharedMemoryAllocator for ShmemAllocator {
    type Pointer = usize; // Offset from base

    fn allocate(&self, size: usize) -> Result<usize, AllocError> {
        // Align to 8 bytes
        let aligned_size = (size + 7) & !7;

        // Try to allocate using atomic fetch_add
        let offset = self.offset.fetch_add(aligned_size, Ordering::SeqCst);

        // Check if allocation fits
        if offset + aligned_size > self.size {
            // Roll back the allocation
            self.offset.fetch_sub(aligned_size, Ordering::SeqCst);
            Err(AllocError::OutOfMemory)
        } else {
            Ok(offset)
        }
    }

    unsafe fn deallocate(&self, _ptr: usize) {
        // No-op for bump allocator
        // In a real implementation, we might maintain a free list
    }

    unsafe fn get_address(&self, ptr: usize) -> *mut u8 {
        self.base.add(ptr)
    }

    unsafe fn get_size(&self, _ptr: usize) -> Option<usize> {
        None // Bump allocator doesn't track individual allocation sizes
    }

    fn is_valid(&self, ptr: usize) -> bool {
        // For bump allocator, any pointer less than current offset is potentially valid
        // In practice, we can't track deallocations, so we assume all allocated pointers remain valid
        ptr < self.offset.load(Ordering::Relaxed) && ptr > 0
    }
}

// === Mock Allocator for Testing ===

/// Mock allocator for testing without real shared memory
/// Uses raw pointers to ensure memory stability
pub struct MockAllocator {
    allocations: Mutex<HashMap<usize, (*mut u8, usize)>>, // (pointer, size)
    next_ptr: Mutex<usize>,
    fail_after: Mutex<Option<usize>>,
}

impl MockAllocator {
    pub fn new() -> Self {
        Self {
            allocations: Mutex::new(HashMap::new()),
            next_ptr: Mutex::new(1), // Start at 1 to avoid 0 (null)
            fail_after: Mutex::new(None),
        }
    }

    /// Make allocations fail after n successful allocations (for testing)
    pub fn fail_after(&self, n: usize) {
        *self.fail_after.lock().unwrap() = Some(n);
    }

    /// Get number of active allocations
    pub fn allocation_count(&self) -> usize {
        self.allocations.lock().unwrap().len()
    }

    /// Get total allocated bytes
    pub fn allocated_bytes(&self) -> usize {
        self.allocations
            .lock()
            .unwrap()
            .values()
            .map(|(_, size)| *size)
            .sum()
    }
}

impl Default for MockAllocator {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for MockAllocator {
    fn drop(&mut self) {
        // Clean up any remaining allocations
        // Take ownership to avoid borrowing issues during cleanup
        let allocations = self.allocations.get_mut().unwrap();
        for (_, (raw_ptr, size)) in allocations.drain() {
            unsafe {
                let layout = std::alloc::Layout::from_size_align_unchecked(size, 8);
                std::alloc::dealloc(raw_ptr, layout);
            }
        }
    }
}

unsafe impl Send for MockAllocator {}
unsafe impl Sync for MockAllocator {}

impl SharedMemoryAllocator for MockAllocator {
    type Pointer = usize;

    fn allocate(&self, size: usize) -> Result<usize, AllocError> {
        // Hold both locks to make the operation atomic
        let mut allocations = self.allocations.lock().unwrap();
        let mut next_ptr = self.next_ptr.lock().unwrap();

        // Check if we should fail
        if let Some(fail_after) = *self.fail_after.lock().unwrap() {
            if allocations.len() >= fail_after {
                return Err(AllocError::OutOfMemory);
            }
        }

        // Allocate raw memory first
        let layout = std::alloc::Layout::from_size_align(size, 8)
            .map_err(|_| AllocError::AllocationTooLarge)?;
        let raw_ptr = unsafe { std::alloc::alloc_zeroed(layout) };

        if raw_ptr.is_null() {
            return Err(AllocError::OutOfMemory);
        }

        // Get the pointer value and increment atomically
        let ptr = *next_ptr;
        *next_ptr += 1;

        // Insert into map while still holding the lock
        allocations.insert(ptr, (raw_ptr, size));

        Ok(ptr)
    }

    unsafe fn deallocate(&self, ptr: usize) {
        if let Some((raw_ptr, size)) = self.allocations.lock().unwrap().remove(&ptr) {
            let layout = std::alloc::Layout::from_size_align_unchecked(size, 8);
            std::alloc::dealloc(raw_ptr, layout);
        }
    }

    unsafe fn get_address(&self, ptr: usize) -> *mut u8 {
        self.allocations
            .lock()
            .unwrap()
            .get(&ptr)
            .map(|(raw_ptr, _)| *raw_ptr)
            .unwrap_or(std::ptr::null_mut())
    }

    unsafe fn get_size(&self, ptr: usize) -> Option<usize> {
        self.allocations
            .lock()
            .unwrap()
            .get(&ptr)
            .map(|(_, size)| *size)
    }

    fn is_valid(&self, ptr: usize) -> bool {
        self.allocations.lock().unwrap().contains_key(&ptr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_allocator() {
        let alloc = MockAllocator::new();

        // Test allocation
        let ptr1 = alloc.allocate(100).unwrap();
        assert_eq!(alloc.allocation_count(), 1);
        assert_eq!(alloc.allocated_bytes(), 100);

        let ptr2 = alloc.allocate(200).unwrap();
        assert_eq!(alloc.allocation_count(), 2);
        assert_eq!(alloc.allocated_bytes(), 300);

        // Test that we can write to and read from the allocated memory
        unsafe {
            let addr1 = alloc.get_address(ptr1);
            assert!(!addr1.is_null());
            // Write some data
            addr1.write(42);
            // Read it back
            assert_eq!(addr1.read(), 42);
        }

        // Test deallocation
        unsafe {
            alloc.deallocate(ptr1);
        }
        assert_eq!(alloc.allocation_count(), 1);
        assert_eq!(alloc.allocated_bytes(), 200);

        // Test fail_after
        alloc.fail_after(1); // Already have 1 allocation
        assert!(alloc.allocate(50).is_err());

        // Clean up
        unsafe {
            alloc.deallocate(ptr2);
        }
        assert_eq!(alloc.allocation_count(), 0);
    }

    #[test]
    fn test_shmem_allocator() {
        let mut buffer = vec![0u8; 1024];
        let alloc = unsafe { ShmemAllocator::new(buffer.as_mut_ptr(), buffer.len()) };

        // Test allocation
        let ptr1 = alloc.allocate(100).unwrap();
        assert_eq!(ptr1, 0);
        assert_eq!(alloc.current_offset(), 104); // Aligned to 8 bytes

        let ptr2 = alloc.allocate(200).unwrap();
        assert_eq!(ptr2, 104);
        assert_eq!(alloc.current_offset(), 304);

        // Test remaining space
        assert_eq!(alloc.remaining(), 1024 - 304);

        // Test out of memory
        assert!(alloc.allocate(800).is_err());
        assert_eq!(alloc.current_offset(), 304); // Should roll back
    }
}
