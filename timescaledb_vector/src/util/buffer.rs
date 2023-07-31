//! A buffer is a Postgres abstraction to identify a slot in memory, almost always shared memory.
//! Under the hood, it is just an index into the shared memory array.
//! To use the slot, certain pins and locks need to be taken
//! See src/backend/storage/buffer/README in the Postgres src.

use std::ops::Deref;

use pgrx::*;

use pgrx::pg_sys::{
    BlockNumber, Buffer, BufferGetBlockNumber, ForkNumber_MAIN_FORKNUM, InvalidBlockNumber,
    ReadBufferMode_RBM_NORMAL, Relation,
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
        //should really be using  ExtendBufferedRel but it's not in pgrx so go thru the read path with InvalidBlockNumber
        Self::read(index, InvalidBlockNumber)
    }

    pub unsafe fn read(index: Relation, block: BlockNumber) -> Self {
        let fork_number = ForkNumber_MAIN_FORKNUM;

        let buf = pg_sys::ReadBufferExtended(
            index,
            fork_number,
            block,
            ReadBufferMode_RBM_NORMAL,
            std::ptr::null_mut(),
        );

        pg_sys::LockBuffer(buf, pg_sys::BUFFER_LOCK_EXCLUSIVE as i32);
        LockedBufferExclusive { buffer: buf }
    }

    /// Get an exclusive lock for cleanup (vacuum) operations.
    /// Obtaining this lock is more restrictive. It will only be obtained once the pin
    /// count is 1. Refer to the PG code for `LockBufferForCleanup` for more info
    pub unsafe fn read_for_cleanup(index: Relation, block: BlockNumber) -> Self {
        let fork_number = ForkNumber_MAIN_FORKNUM;

        let buf = pg_sys::ReadBufferExtended(
            index,
            fork_number,
            block,
            ReadBufferMode_RBM_NORMAL,
            std::ptr::null_mut(),
        );

        pg_sys::LockBufferForCleanup(buf);
        LockedBufferExclusive { buffer: buf }
    }

    pub fn get_block_number(&self) -> BlockNumber {
        unsafe { BufferGetBlockNumber(self.buffer) }
    }
}

impl Drop for LockedBufferExclusive {
    /// drop both unlock and unpins the buffer.
    fn drop(&mut self) {
        unsafe {
            // Only unlock while in a transaction state. Should not be unlocking during abort or commit.
            // During abort, the system will unlock stuff itself. During commit, the release should have already happened.
            if pgrx::pg_sys::IsTransactionState() {
                pg_sys::UnlockReleaseBuffer(self.buffer);
            }
        }
    }
}

impl Deref for LockedBufferExclusive {
    type Target = Buffer;
    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

/// LockedBufferShare is an RAII-guarded buffer that
/// has been locked for share access.
///
/// It is probably not a good idea to hold on to this too long.
pub struct LockedBufferShare {
    buffer: Buffer,
}

impl LockedBufferShare {
    /// read return buffer for the given blockNumber in a relation.
    ///
    /// The returned block will be pinned and locked in share mode
    pub unsafe fn read(index: Relation, block: BlockNumber) -> Self {
        let fork_number = ForkNumber_MAIN_FORKNUM;

        let buf = pg_sys::ReadBufferExtended(
            index,
            fork_number,
            block,
            ReadBufferMode_RBM_NORMAL,
            std::ptr::null_mut(),
        );

        pg_sys::LockBuffer(buf, pg_sys::BUFFER_LOCK_SHARE as i32);
        LockedBufferShare { buffer: buf }
    }
}

impl Drop for LockedBufferShare {
    /// drop both unlock and unpins the buffer.
    fn drop(&mut self) {
        unsafe {
            // Only unlock while in a transaction state. Should not be unlocking during abort or commit.
            // During abort, the system will unlock stuff itself. During commit, the release should have already happened.
            if pgrx::pg_sys::IsTransactionState() {
                pg_sys::UnlockReleaseBuffer(self.buffer);
            }
        }
    }
}

impl Deref for LockedBufferShare {
    type Target = Buffer;
    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}
