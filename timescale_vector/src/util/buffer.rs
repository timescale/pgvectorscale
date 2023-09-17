//! A buffer is a Postgres abstraction to identify a slot in memory, almost always shared memory.
//! Under the hood, it is just an index into the shared memory array.
//! To use the slot, certain pins and locks need to be taken
//! See src/backend/storage/buffer/README in the Postgres src.

use std::ops::Deref;

use pgrx::*;

use pgrx::pg_sys::{
    BlockNumber, Buffer, BufferGetBlockNumber, ForkNumber_MAIN_FORKNUM, InvalidBlockNumber,
    ReadBufferMode_RBM_NORMAL,
};

pub struct LockRelationForExtension<'a> {
    relation: &'a PgRelation,
}

impl<'a> LockRelationForExtension<'a> {
    pub fn new(index: &'a PgRelation) -> Self {
        unsafe {
            pg_sys::LockRelationForExtension(
                index.as_ptr(),
                pg_sys::ExclusiveLock as pg_sys::LOCKMODE,
            )
        }
        Self { relation: index }
    }
}

impl<'a> Drop for LockRelationForExtension<'a> {
    /// drop both unlock and unpins the buffer.
    fn drop(&mut self) {
        unsafe {
            // Only unlock while in a transaction state. Should not be unlocking during abort or commit.
            // During abort, the system will unlock stuff itself. During commit, the release should have already happened.
            if pgrx::pg_sys::IsTransactionState() {
                pg_sys::UnlockRelationForExtension(
                    self.relation.as_ptr(),
                    pg_sys::ExclusiveLock as pg_sys::LOCKMODE,
                );
            }
        }
    }
}

/// LockedBufferExclusive is an RAII-guarded buffer that
/// has been locked for exclusive access.
///
/// It is probably not a good idea to hold on to this too long.
pub struct LockedBufferExclusive<'a> {
    _relation: &'a PgRelation,
    buffer: Buffer,
}

impl<'a> LockedBufferExclusive<'a> {
    /// new return an allocated buffer for a new block in a relation.
    /// The block is obtained by extending the relation.
    ///
    /// The returned block will be pinned and locked in exclusive mode
    ///
    /// Safety: safe because it locks the relation for extension.
    pub fn new(index: &'a PgRelation) -> Self {
        //ReadBufferExtended requires the caller to ensure that only one backend extends the relation at one time.
        let _lock = LockRelationForExtension::new(index);
        //should really be using  ExtendBufferedRel but it's not in pgrx so go thru the read path with InvalidBlockNumber
        unsafe { Self::read_unchecked(index, InvalidBlockNumber) }
    }

    /// Safety: Safe because it checks the block number doesn't overflow. ReadBufferExtended will throw an error if the block number is out of range for the relation
    pub fn read(index: &'a PgRelation, block: BlockNumber) -> Self {
        unsafe { Self::read_unchecked(index, block) }
    }

    /// Safety: unsafe because tje block number is not verifiwed
    unsafe fn read_unchecked(index: &'a PgRelation, block: BlockNumber) -> Self {
        let fork_number = ForkNumber_MAIN_FORKNUM;

        let buf = pg_sys::ReadBufferExtended(
            index.as_ptr(),
            fork_number,
            block,
            ReadBufferMode_RBM_NORMAL,
            std::ptr::null_mut(),
        );

        pg_sys::LockBuffer(buf, pg_sys::BUFFER_LOCK_EXCLUSIVE as i32);
        LockedBufferExclusive {
            _relation: index,
            buffer: buf,
        }
    }

    /// Get an exclusive lock for cleanup (vacuum) operations.
    /// Obtaining this lock is more restrictive. It will only be obtained once the pin
    /// count is 1. Refer to the PG code for `LockBufferForCleanup` for more info
    pub unsafe fn read_for_cleanup(index: &'a PgRelation, block: BlockNumber) -> Self {
        let fork_number = ForkNumber_MAIN_FORKNUM;

        let buf = pg_sys::ReadBufferExtended(
            index.as_ptr(),
            fork_number,
            block,
            ReadBufferMode_RBM_NORMAL,
            std::ptr::null_mut(),
        );

        pg_sys::LockBufferForCleanup(buf);
        LockedBufferExclusive {
            _relation: index,
            buffer: buf,
        }
    }

    pub fn get_block_number(&self) -> BlockNumber {
        unsafe { BufferGetBlockNumber(self.buffer) }
    }
}

impl<'a> Drop for LockedBufferExclusive<'a> {
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

impl<'a> Deref for LockedBufferExclusive<'a> {
    type Target = Buffer;
    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

/// LockedBufferShare is an RAII-guarded buffer that
/// has been locked for share access.
///
/// It is probably not a good idea to hold on to this too long.
pub struct LockedBufferShare<'a> {
    _relation: &'a PgRelation,
    buffer: Buffer,
}

impl<'a> LockedBufferShare<'a> {
    /// read return buffer for the given blockNumber in a relation.
    ///
    /// The returned block will be pinned and locked in share mode
    ///
    /// Safety: Safe because it checks the block number doesn't overflow. ReadBufferExtended will throw an error if the block number is out of range for the relation
    pub fn read(index: &'a PgRelation, block: BlockNumber) -> Self {
        let fork_number = ForkNumber_MAIN_FORKNUM;

        unsafe {
            let buf = pg_sys::ReadBufferExtended(
                index.as_ptr(),
                fork_number,
                block,
                ReadBufferMode_RBM_NORMAL,
                std::ptr::null_mut(),
            );

            pg_sys::LockBuffer(buf, pg_sys::BUFFER_LOCK_SHARE as i32);
            LockedBufferShare {
                _relation: index,
                buffer: buf,
            }
        }
    }
}

impl<'a> Drop for LockedBufferShare<'a> {
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

impl<'a> Deref for LockedBufferShare<'a> {
    type Target = Buffer;
    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}
