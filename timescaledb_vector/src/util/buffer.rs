//! A buffer is a Postgres abstraction to identify a slot in memory, almost always shared memory.
//! Under the hood, it is just an index into the shared memory array.
//! To use the slot, certain pins and locks need to be taken
//! See src/backend/storage/buffer/README in the Postgres src.

use std::ops::Deref;

use pgrx::*;

use pgrx::pg_sys::{
    Buffer, ForkNumber, ForkNumber_MAIN_FORKNUM, InvalidBlockNumber, ReadBufferMode_RBM_NORMAL,
    Relation,
};

/// LockedBufferExclusive is an RAII-guarded buffer that
/// has been locked for exclusive access.
///
/// It is probably not a good idea to hold on to this too long.
pub struct LockedBufferExclusive {
    buffer: Buffer,
}

impl LockedBufferExclusive {
    /// new return an allocated buffer for a new block in a relation.
    /// The block is obtained by extending the relation.
    ///
    /// The returned block will be pinned and locked in exclusive mode
    pub unsafe fn new(index: Relation) -> Self {
        let fork_number = ForkNumber_MAIN_FORKNUM;

        //should really be using  ExtendBufferedRel but it's not in pgrx
        let buf = pg_sys::ReadBufferExtended(
            index,
            fork_number,
            InvalidBlockNumber,
            ReadBufferMode_RBM_NORMAL,
            std::ptr::null_mut(),
        );

        pg_sys::LockBuffer(buf, pg_sys::BUFFER_LOCK_EXCLUSIVE as i32);
        LockedBufferExclusive { buffer: buf }
    }
}

impl Drop for LockedBufferExclusive {
    /// drop both unlock and unpins the buffer.
    fn drop(&mut self) {
        unsafe { pg_sys::UnlockReleaseBuffer(self.buffer) };
    }
}

impl Deref for LockedBufferExclusive {
    type Target = Buffer;
    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}
