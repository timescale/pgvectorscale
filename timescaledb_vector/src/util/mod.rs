pub mod buffer;
pub mod page;
pub mod ports;
pub mod tape;

use pgrx::pg_sys::Page;
use rkyv::{Archive, Deserialize, Serialize};

#[derive(Archive, Deserialize, Serialize, Debug)]
pub struct ItemPointer {
    block_number: pgrx::pg_sys::BlockNumber,
    offset: pgrx::pg_sys::OffsetNumber,
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
}
