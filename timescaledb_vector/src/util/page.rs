//! A Page is a Postgres abstraction for a slice of memory you can write to
//! It is usually 8kb and has a special layout. See https://www.postgresql.org/docs/current/storage-page-layout.html

use pg_sys::{Buffer, Page, Relation};
use pgrx::{
    pg_sys::{BlockNumber, BufferGetPage},
    *,
};
use std::ops::Deref;

use super::buffer::{LockedBufferExclusive, LockedBufferShare};
pub struct WritablePage {
    buffer: LockedBufferExclusive,
    page: Page,
    state: *mut pg_sys::GenericXLogState,
    committed: bool,
}

pub const TSV_PAGE_ID: u16 = 0xAE24; /* magic number, generated randomly */

/// This is the Tsv-specific data that goes on every "tsv-owned" page
/// It is placed at the end of a page in the "special" area
struct TsvPageOpaqueData {
    page_id: u16, //  A magic ID for debuging to identify the page as a "tsv-owned"
}

#[inline(always)]
unsafe fn get_tsv_opaque_data(page: Page) -> *mut TsvPageOpaqueData {
    let sp = super::ports::PageGetSpecialPointer(page);
    sp.cast::<TsvPageOpaqueData>()
}

/// WritablePage implements and RAII-guarded Page that you can write to.
/// All writes will be WAL-logged.
///
/// It is probably not a good idea to hold on to a WritablePage for a long time.
impl WritablePage {
    /// new creates a totally new page on a relation by extending the relation
    pub unsafe fn new(index: Relation) -> Self {
        let buffer = LockedBufferExclusive::new(index);
        let state = pg_sys::GenericXLogStart(index);
        //TODO do we need a GENERIC_XLOG_FULL_IMAGE option?
        let page = pg_sys::GenericXLogRegisterBuffer(state, *buffer, 0);
        pg_sys::PageInit(
            page,
            pg_sys::BLCKSZ as usize,
            std::mem::size_of::<TsvPageOpaqueData>(),
        );
        (*get_tsv_opaque_data(page)).page_id = TSV_PAGE_ID;
        Self {
            buffer: buffer,
            page: page,
            state: state,
            committed: false,
        }
    }

    pub unsafe fn modify(index: Relation, block: BlockNumber) -> Self {
        let buffer = LockedBufferExclusive::read(index, block);
        let state = pg_sys::GenericXLogStart(index);
        let page = pg_sys::GenericXLogRegisterBuffer(state, *buffer, 0);
        Self {
            buffer: buffer,
            page: page,
            state: state,
            committed: false,
        }
    }

    pub fn get_buffer(&self) -> &LockedBufferExclusive {
        &self.buffer
    }

    pub fn get_block_number(&self) -> BlockNumber {
        self.buffer.get_block_number()
    }

    pub fn get_free_space(&self) -> usize {
        unsafe { pg_sys::PageGetFreeSpace(self.page) }
    }

    /// commit saves all the changes to the page.
    /// Note that this will consume the page and make it unusable after the call.
    pub fn commit(mut self) {
        unsafe {
            pg_sys::MarkBufferDirty(*self.buffer);
            pg_sys::GenericXLogFinish(self.state);
        }
        self.committed = true;
    }
}

impl Drop for WritablePage {
    // drop aborts the xlog if it has not been committed.
    fn drop(&mut self) {
        if !self.committed {
            unsafe {
                pg_sys::GenericXLogAbort(self.state);
            };
        }
    }
}

impl Deref for WritablePage {
    type Target = Page;
    fn deref(&self) -> &Self::Target {
        &self.page
    }
}

pub struct ReadablePage {
    buffer: LockedBufferShare,
    page: Page,
}

impl ReadablePage {
    /// new creates a totally new page on a relation by extending the relation
    pub unsafe fn read(index: Relation, block: BlockNumber) -> Self {
        let buffer = LockedBufferShare::read(index, block);
        let page = BufferGetPage(*buffer);
        Self {
            buffer: buffer,
            page: page,
        }
    }

    pub fn get_buffer(&self) -> &LockedBufferShare {
        &self.buffer
    }

    pub fn get_free_space(&self) -> usize {
        unsafe { pg_sys::PageGetFreeSpace(self.page) }
    }
}

impl Deref for ReadablePage {
    type Target = Page;
    fn deref(&self) -> &Self::Target {
        &self.page
    }
}
