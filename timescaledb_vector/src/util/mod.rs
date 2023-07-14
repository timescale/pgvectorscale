pub mod buffer;
pub mod page;
pub mod ports;
pub mod tape;

use pgrx::pg_sys::Relation;
use rkyv::{Archive, Deserialize, Serialize};

use self::{
    page::{ReadablePage, WritablePage},
    ports::{PageGetItem, PageGetItemId},
};

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Eq, Hash, Clone, Copy)]
#[archive(check_bytes)]
pub struct ItemPointer {
    pub block_number: pgrx::pg_sys::BlockNumber,
    pub offset: pgrx::pg_sys::OffsetNumber,
}

impl ArchivedItemPointer {
    pub fn deserialize_item_pointer(&self) -> ItemPointer {
        self.deserialize(&mut rkyv::Infallible).unwrap()
    }
}

pub struct ReadableBuffer {
    _page: ReadablePage,
    len: usize,
    ptr: *const u8,
}

impl ReadableBuffer {
    pub fn get_data_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
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

    pub fn to_item_pointer_data(&self, ctid: &mut pgrx::pg_sys::ItemPointerData) {
        pgrx::item_pointer_set_all(ctid, self.block_number, self.offset)
    }

    pub unsafe fn read_bytes(&self, index: Relation) -> ReadableBuffer {
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
    pub unsafe fn modify_bytes(&self, index: Relation) -> WritableBuffer {
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
