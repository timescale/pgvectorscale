pub mod buffer;
pub mod page;
pub mod ports;
pub mod tape;

use pgrx::pg_sys::{Page, Relation};
use rkyv::{Archive, Deserialize, Serialize};

use self::{
    page::{ReadablePage, WritablePage},
    ports::{PageGetItem, PageGetItemId},
};

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Eq, Hash, Clone)]
#[archive(check_bytes)]
pub struct ItemPointer {
    pub block_number: pgrx::pg_sys::BlockNumber,
    pub offset: pgrx::pg_sys::OffsetNumber,
}

/*

TODO change not to have a liftime here.
pub struct ReadableBuffer {
    _page: ReadablePage,
    len: usize,
    ptr: *const u8,
}
impl ReadableBuffer {
    fn get_data_slice(&self) -> &[u8] {
        std::slice::from_raw_parts(self.ptr, self.len)
    )
}
*/

pub struct ReadableBuffer<'a> {
    _page: ReadablePage,
    pub data: &'a [u8],
}

pub struct WritableBuffer {
    _page: WritablePage,
    len: usize,
    ptr: *mut u8,
}

impl WritableBuffer {
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

    pub unsafe fn read_bytes<'b, 'a>(&'b self, index: Relation) -> ReadableBuffer<'a> {
        let page = ReadablePage::read(index, self.block_number);
        let item_id = PageGetItemId(*page, self.offset);
        let item = PageGetItem(*page, item_id) as *mut u8;
        let len = (*item_id).lp_len();
        let data = std::slice::from_raw_parts(item, len as _);
        ReadableBuffer {
            _page: page,
            data: data,
        }
    }
    pub unsafe fn modify_bytes(&self, index: Relation) -> WritableBuffer {
        let page = WritablePage::modify(index, self.block_number);
        let item_id = PageGetItemId(*page, self.offset);
        let mut item = PageGetItem(*page, item_id) as *mut u8;
        let len = (*item_id).lp_len();
        WritableBuffer {
            _page: page,
            ptr: item,
            len: len as _,
        }
    }
}
