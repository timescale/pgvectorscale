pub mod buffer;
pub mod chain;
pub mod page;
pub mod ports;
pub mod table_slot;
pub mod tape;

use pgrx::PgRelation;
use rkyv::{Archive, Deserialize, Serialize};

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

impl PartialOrd for ItemPointer {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ItemPointer {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.block_number
            .cmp(&other.block_number)
            .then_with(|| self.offset.cmp(&other.offset))
    }
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

    pub fn get_owned_page(self) -> ReadablePage<'a> {
        self._page
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn advance(&mut self, len: usize) {
        assert!(self.len >= len);
        self.ptr = unsafe { self.ptr.add(len) };
        self.len -= len;
    }
}

pub struct WritableBuffer<'a> {
    _page: WritablePage<'a>,
    len: usize,
    ptr: *mut u8,
}

impl<'a> WritableBuffer<'a> {
    pub fn get_data_slice(&mut self) -> &mut [u8] {
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
            block_number,
            offset,
        }
    }

    pub fn new_invalid() -> Self {
        Self {
            block_number: pgrx::pg_sys::InvalidBlockNumber,
            offset: pgrx::pg_sys::InvalidOffsetNumber,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.block_number != pgrx::pg_sys::InvalidBlockNumber
            && self.offset != pgrx::pg_sys::InvalidOffsetNumber
    }

    pub unsafe fn with_page(page: &page::WritablePage, offset: pgrx::pg_sys::OffsetNumber) -> Self {
        Self {
            block_number: pgrx::pg_sys::BufferGetBlockNumber(**(page.get_buffer())),
            offset,
        }
    }

    pub unsafe fn with_item_pointer_data(ctid: pgrx::pg_sys::ItemPointerData) -> Self {
        let ip = pgrx::itemptr::item_pointer_get_block_number(&ctid);
        let off = pgrx::itemptr::item_pointer_get_offset_number(&ctid);
        Self::new(ip, off)
    }

    pub fn to_item_pointer_data(self, ctid: &mut pgrx::pg_sys::ItemPointerData) {
        pgrx::itemptr::item_pointer_set_all(ctid, self.block_number, self.offset)
    }

    pub unsafe fn read_bytes(self, index: &PgRelation) -> ReadableBuffer {
        let page = ReadablePage::read(index, self.block_number);
        page.get_item_unchecked(self.offset)
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

    pub fn ip_distance(self, other: Self) -> usize {
        let block_diff = self.block_number as isize - other.block_number as isize;
        let offset_diff = self.offset as isize - other.offset as isize;
        debug_assert!(offset_diff < pgrx::pg_sys::MaxOffsetNumber as _);
        (block_diff * (pgrx::pg_sys::MaxOffsetNumber as isize) + offset_diff).unsigned_abs()
    }
}

pub type IndexPointer = ItemPointer;
pub type HeapPointer = ItemPointer;
