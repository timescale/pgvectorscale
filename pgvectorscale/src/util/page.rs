//! A Page is a Postgres abstraction for a slice of memory you can write to
//! It is usually 8kb and has a special layout. See https://www.postgresql.org/docs/current/storage-page-layout.html

use pg_sys::Page;
use pgrx::{
    pg_sys::{BlockNumber, BufferGetPage, OffsetNumber, BLCKSZ},
    *,
};
use std::ops::Deref;

use super::{
    buffer::{LockedBufferExclusive, LockedBufferShare},
    ports::{PageGetItem, PageGetItemId},
    ReadableBuffer,
};
pub struct WritablePage<'a> {
    buffer: LockedBufferExclusive<'a>,
    page: Page,
    state: *mut pg_sys::GenericXLogState,
    committed: bool,
}

pub const TSV_PAGE_ID: u16 = 0xAE24; /* magic number, generated randomly */

/// PageType identifies different types of pages in our index.
/// The layout of any one type should be consistent
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PageType {
    MetaV1 = 0,
    Node = 1,
    PqQuantizerDef = 2,
    PqQuantizerVector = 3,
    SbqMeansV1 = 4,
    SbqNode = 5,
    MetaV2 = 6,
    SbqMeans = 7,
    Meta = 8,
}

impl PageType {
    pub fn from_u8(value: u8) -> Self {
        match value {
            0 => PageType::MetaV1,
            1 => PageType::Node,
            2 => PageType::PqQuantizerDef,
            3 => PageType::PqQuantizerVector,
            4 => PageType::SbqMeansV1,
            5 => PageType::SbqNode,
            6 => PageType::MetaV2,
            7 => PageType::SbqMeans,
            8 => PageType::Meta,
            _ => panic!("Unknown PageType number {}", value),
        }
    }

    /// `ChainTape` supports chaining of pages that might contain large data.
    /// This is not supported for all page types.  Note that `Tape` requires
    /// that the page type not be chained.
    pub fn is_chained(self) -> bool {
        matches!(self, PageType::SbqMeans) || matches!(self, PageType::Meta)
    }
}

/// This is the Tsv-specific data that goes on every "diskann-owned" page
/// It is placed at the end of a page in the "special" area
#[repr(C)]
struct TsvPageOpaqueData {
    page_type: u8, // stores the PageType enum as an integer (u8 because we doubt we'll have more than 256 types).
    _reserved: u8, // don't waste bytes, may be able to reuse later. For now: 0
    page_id: u16, //  A magic ID for debuging to identify the page as a "diskann-owned". Should be last.
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
            let mut new = Self {
                buffer,
                page,
                state,
                committed: false,
            };
            new.reinit(page_type);
            new
        }
    }

    pub fn reinit(&mut self, page_type: PageType) {
        unsafe {
            pg_sys::PageInit(
                self.page,
                pg_sys::BLCKSZ as usize,
                std::mem::size_of::<TsvPageOpaqueData>(),
            );
            *TsvPageOpaqueData::with_page(self.page) = TsvPageOpaqueData::new(page_type);
        }
    }

    pub fn modify(index: &'a PgRelation, block: BlockNumber) -> Self {
        let buffer = LockedBufferExclusive::read(index, block);
        Self::modify_with_buffer(index, buffer)
    }

    pub fn add_item(&mut self, data: &[u8]) -> OffsetNumber {
        let size = data.len();
        assert!(self.get_free_space() >= size);
        unsafe { self.add_item_unchecked(data) }
    }

    pub unsafe fn add_item_unchecked(&mut self, data: &[u8]) -> OffsetNumber {
        let size = data.len();
        assert!(size < BLCKSZ as usize);

        let offset_number = pg_sys::PageAddItemExtended(
            self.page,
            data.as_ptr() as _,
            size,
            pg_sys::InvalidOffsetNumber,
            0,
        );

        assert!(offset_number != pg_sys::InvalidOffsetNumber);
        offset_number
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
                buffer,
                page,
                state,
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

    fn get_free_space(&self) -> usize {
        unsafe { pg_sys::PageGetFreeSpace(self.page) }
    }

    /// The actual free space that can be used to store data.
    /// See https://github.com/postgres/postgres/blob/0164a0f9ee12e0eff9e4c661358a272ecd65c2d4/src/backend/storage/page/bufpage.c#L304
    pub fn get_aligned_free_space(&self) -> usize {
        let free_space = self.get_free_space();
        free_space - (free_space % 8)
    }

    pub fn get_type(&self) -> PageType {
        unsafe {
            let opaque_data =
            //safe to do because self.page was already verified during construction
            TsvPageOpaqueData::with_page(self.page);

            PageType::from_u8((*opaque_data).page_type)
        }
    }

    pub fn set_types(&self, new: PageType) {
        unsafe {
            let opaque_data =
            //safe to do because self.page was already verified during construction
            TsvPageOpaqueData::with_page(self.page);

            (*opaque_data).page_type = new as u8;
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

impl Drop for WritablePage<'_> {
    // drop aborts the xlog if it has not been committed.
    fn drop(&mut self) {
        if !self.committed {
            unsafe {
                pg_sys::GenericXLogAbort(self.state);
            };
        }
    }
}

impl Deref for WritablePage<'_> {
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
        Self { buffer, page }
    }

    pub fn get_type(&self) -> PageType {
        let opaque_data = TsvPageOpaqueData::read_from_page(&self.page);
        PageType::from_u8(opaque_data.page_type)
    }

    pub fn get_buffer(&self) -> &LockedBufferShare {
        &self.buffer
    }

    // Safety: unsafe because no verification of the offset is done.
    pub unsafe fn get_item_unchecked(
        self,
        offset: pgrx::pg_sys::OffsetNumber,
    ) -> ReadableBuffer<'a> {
        let item_id = PageGetItemId(self.page, offset);
        let item = PageGetItem(self.page, item_id) as *mut u8;
        let len = (*item_id).lp_len();
        ReadableBuffer {
            _page: self,
            ptr: item,
            len: len as _,
        }
    }
}

impl Deref for ReadablePage<'_> {
    type Target = Page;
    fn deref(&self) -> &Self::Target {
        &self.page
    }
}
