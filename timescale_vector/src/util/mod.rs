pub mod buffer;
pub mod page;
pub mod ports;
pub mod tape;

use pgrx::{
    error,
    pg_sys::{BufferGetPage, ForkNumber_MAIN_FORKNUM, BLCKSZ},
    PgRelation,
};
use rkyv::{Archive, Deserialize, Serialize};

use std::{io::Error, os::raw::c_void};

use self::{
    page::{ReadablePage, WritablePage},
    ports::{PageGetItem, PageGetItemId},
};

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Eq, Hash, Clone, Copy)]
#[archive(check_bytes)]
#[repr(C)] // Added this so we can compute size via sizeof
pub struct ItemPointer {
    pub block_number: pgrx::pg_sys::BlockNumber,
    pub offset: pgrx::pg_sys::OffsetNumber,
}

impl ArchivedItemPointer {
    pub fn deserialize_item_pointer(&self) -> ItemPointer {
        self.deserialize(&mut rkyv::Infallible).unwrap()
    }
}

pub struct ReadableBuffer<'a> {
    _page: ReadablePage<'a>,
    len: usize,
    ptr: *const u8,
}

impl<'a> ReadableBuffer<'a> {
    pub fn get_data_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

pub struct WritableBuffer<'a> {
    _page: WritablePage<'a>,
    len: usize,
    ptr: *mut u8,
}

impl<'a> WritableBuffer<'a> {
    pub fn get_data_slice(&self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    pub fn commit(self) {
        self._page.commit();
    }
}

impl ItemPointer {
    pub fn new(
        block_number: pgrx::pg_sys::BlockNumber,
        offset: pgrx::pg_sys::OffsetNumber,
    ) -> Self {
        Self {
            block_number: block_number,
            offset: offset,
        }
    }

    pub unsafe fn with_page(page: &page::WritablePage, offset: pgrx::pg_sys::OffsetNumber) -> Self {
        Self {
            block_number: pgrx::pg_sys::BufferGetBlockNumber(**(page.get_buffer())),
            offset: offset,
        }
    }

    pub unsafe fn with_item_pointer_data(ctid: pgrx::pg_sys::ItemPointerData) -> Self {
        let ip = pgrx::item_pointer_get_block_number(&ctid);
        let off = pgrx::item_pointer_get_offset_number(&ctid);
        Self::new(ip, off)
    }

    pub fn to_item_pointer_data(&self, ctid: &mut pgrx::pg_sys::ItemPointerData) {
        pgrx::item_pointer_set_all(ctid, self.block_number, self.offset)
    }

    pub unsafe fn prefetch(self, index: &PgRelation) {
        let res = pgrx::pg_sys::PrefetchBuffer(
            index.as_ptr(),
            ForkNumber_MAIN_FORKNUM,
            self.block_number,
        );
        if res.recent_buffer > 0 {
            let page_size = 4096;
            let ptr = BufferGetPage(res.recent_buffer) as *mut c_void;
            let off = ptr.align_offset(page_size);
            let (ptr, sz) = if off > 0 {
                (
                    ptr.offset((off as isize) - (page_size as isize)),
                    page_size * 3,
                )
            } else {
                (ptr, page_size * 2)
            };
            let mres = libc::madvise(ptr, sz, libc::MADV_WILLNEED);
            if mres != 0 {
                let err = Error::last_os_error();
                error!("Error in madvise: {}", err);
            }
        }
    }

    pub unsafe fn read_bytes(self, index: &PgRelation) -> ReadableBuffer {
        let page = ReadablePage::read(index, self.block_number);
        let item_id = PageGetItemId(*page, self.offset);
        let item = PageGetItem(*page, item_id) as *mut u8;
        let len = (*item_id).lp_len();
        ReadableBuffer {
            _page: page,
            ptr: item,
            len: len as _,
        }
    }
    pub unsafe fn modify_bytes(self, index: &PgRelation) -> WritableBuffer {
        let page = WritablePage::modify(index, self.block_number);
        let item_id = PageGetItemId(*page, self.offset);
        let item = PageGetItem(*page, item_id) as *mut u8;
        let len = (*item_id).lp_len();
        WritableBuffer {
            _page: page,
            ptr: item,
            len: len as _,
        }
    }
}

pub type IndexPointer = ItemPointer;
pub type HeapPointer = ItemPointer;
