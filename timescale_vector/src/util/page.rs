//! A Page is a Postgres abstraction for a slice of memory you can write to
//! It is usually 8kb and has a special layout. See https://www.postgresql.org/docs/current/storage-page-layout.html

use pg_sys::Page;
use pgrx::{
    pg_sys::{BlockNumber, BufferGetPage},
    *,
};
use std::ops::Deref;

use super::buffer::{LockedBufferExclusive, LockedBufferShare};
pub struct WritablePage<'a> {
    buffer: LockedBufferExclusive<'a>,
    page: Page,
    state: *mut pg_sys::GenericXLogState,
    committed: bool,
}

pub const TSV_PAGE_ID: u16 = 0xAE24; /* magic number, generated randomly */

/// PageType identifies different types of pages in our index.
/// The layout of any one type should be consistent
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PageType {
    Meta = 0,
    Node = 1,
    PqQuantizerDef = 2,
    PqQuantizerVector = 3,
    StartingIds = 4,
}

impl PageType {
    fn from_u8(value: u8) -> Self {
        match value {
            0 => PageType::Meta,
            1 => PageType::Node,
            2 => PageType::PqQuantizerDef,
            3 => PageType::PqQuantizerVector,
            4 => PageType::StartingIds,
            _ => panic!("Unknown PageType number {}", value),
        }
    }
}
/// This is the Tsv-specific data that goes on every "tsv-owned" page
/// It is placed at the end of a page in the "special" area

#[repr(C)]
struct TsvPageOpaqueData {
    page_type: u8, // stores the PageType enum as an integer (u8 because we doubt we'll have more than 256 types).
    _reserved: u8, // don't waste bytes, may be able to reuse later. For now: 0
    page_id: u16, //  A magic ID for debuging to identify the page as a "tsv-owned". Should be last.
}

impl TsvPageOpaqueData {
    fn new(page_type: PageType) -> Self {
        Self {
            page_type: page_type as u8,
            _reserved: 0,
            page_id: TSV_PAGE_ID,
        }
    }

    /// Safety: unsafe because no verification done. Blind cast.
    #[inline(always)]
    unsafe fn with_page(page: Page) -> *mut TsvPageOpaqueData {
        let sp = super::ports::PageGetSpecialPointer(page);
        sp.cast::<TsvPageOpaqueData>()
    }

    /// Safety: Safe because of the verify call that checks a magic number
    fn read_from_page(page: &Page) -> &TsvPageOpaqueData {
        unsafe {
            let ptr = Self::with_page(*page);
            (*ptr).verify();
            ptr.as_ref().unwrap()
        }
    }

    fn verify(&self) {
        assert_eq!(self.page_id, TSV_PAGE_ID);
        PageType::from_u8(self.page_type);
    }
}

/// WritablePage implements and RAII-guarded Page that you can write to.
/// All writes will be WAL-logged.
///
/// It is probably not a good idea to hold on to a WritablePage for a long time.
impl<'a> WritablePage<'a> {
    /// new creates a totally new page on a relation by extending the relation
    pub fn new(index: &'a PgRelation, page_type: PageType) -> Self {
        let buffer = LockedBufferExclusive::new(index);
        unsafe {
            let state = pg_sys::GenericXLogStart(index.as_ptr());
            //TODO do we need a GENERIC_XLOG_FULL_IMAGE option?
            let page = pg_sys::GenericXLogRegisterBuffer(state, *buffer, 0);
            pg_sys::PageInit(
                page,
                pg_sys::BLCKSZ as usize,
                std::mem::size_of::<TsvPageOpaqueData>(),
            );
            *TsvPageOpaqueData::with_page(page) = TsvPageOpaqueData::new(page_type);
            Self {
                buffer: buffer,
                page: page,
                state: state,
                committed: false,
            }
        }
    }

    pub fn modify(index: &'a PgRelation, block: BlockNumber) -> Self {
        let buffer = LockedBufferExclusive::read(index, block);
        Self::modify_with_buffer(index, buffer)
    }

    /// get a writable page for cleanup(vacuum) operations.
    pub unsafe fn cleanup(index: &'a PgRelation, block: BlockNumber) -> Self {
        let buffer = LockedBufferExclusive::read_for_cleanup(index, block);
        Self::modify_with_buffer(index, buffer)
    }

    // Safety: Safe because it verifies the page
    fn modify_with_buffer(index: &'a PgRelation, buffer: LockedBufferExclusive<'a>) -> Self {
        unsafe {
            let state = pg_sys::GenericXLogStart(index.as_ptr());
            let page = pg_sys::GenericXLogRegisterBuffer(state, *buffer, 0);
            //this check the page
            _ = TsvPageOpaqueData::read_from_page(&page);
            Self {
                buffer: buffer,
                page: page,
                state: state,
                committed: false,
            }
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

    pub fn get_type(&self) -> PageType {
        unsafe {
            let opaque_data =
            //safe to do because self.page was already verified during construction
            TsvPageOpaqueData::with_page(self.page);

            PageType::from_u8((*opaque_data).page_type)
        }
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

impl<'a> Drop for WritablePage<'a> {
    // drop aborts the xlog if it has not been committed.
    fn drop(&mut self) {
        if !self.committed {
            unsafe {
                pg_sys::GenericXLogAbort(self.state);
            };
        }
    }
}

impl<'a> Deref for WritablePage<'a> {
    type Target = Page;
    fn deref(&self) -> &Self::Target {
        &self.page
    }
}

pub struct ReadablePage<'a> {
    buffer: LockedBufferShare<'a>,
    page: Page,
}

impl<'a> ReadablePage<'a> {
    /// new creates a totally new page on a relation by extending the relation
    pub unsafe fn read(index: &'a PgRelation, block: BlockNumber) -> Self {
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
}

impl<'a> Deref for ReadablePage<'a> {
    type Target = Page;
    fn deref(&self) -> &Self::Target {
        &self.page
    }
}
