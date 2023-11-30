//! Tape provides a simple infinite-tape-writing abstraction over postgres pages.

use std::io::Write;

use crate::util::ItemPointer;

use super::page::{PageType, WritablePage};
use pgrx::{
    pg_sys::{BlockNumber, InvalidBlockNumber, BLCKSZ},
    *,
};

pub struct MultiblockTape<'a> {
    page_type: PageType,
    index: &'a PgRelation,
}

impl<'a> MultiblockTape<'a> {
    pub unsafe fn new(index: &'a PgRelation, page_type: PageType) -> Self {
        Self {
            page_type,
            index: index,
        }
    }

    fn get_free_space(&self, page: &WritablePage) -> usize {
        let free_size = page.get_free_space();
        //account for alignment
        free_size.saturating_sub(free_size % 8).saturating_sub(100)
    }

    pub unsafe fn add(&mut self, data: &[u8]) -> super::ItemPointer {
        let mut size = data.len();
        let mut ip: Option<ItemPointer> = None;
        let mut data = data;

        //do not spread items across pages. rethink later
        let page = WritablePage::new(self.index, self.page_type);
        let mut current_block_number = page.get_block_number();
        page.commit();

        while size > 0 {
            let current_page = WritablePage::modify(self.index, current_block_number);
            assert!(self.get_free_space(&current_page) > 0);
            let this_page_size = size.min(self.get_free_space(&current_page));

            let offset_number = pg_sys::PageAddItemExtended(
                *current_page,
                data.as_ptr() as _,
                this_page_size,
                pg_sys::InvalidOffsetNumber,
                0,
            );
            assert!(offset_number != pg_sys::InvalidOffsetNumber);
            //only set first time
            if ip.is_none() {
                ip = Some(super::ItemPointer::with_page(&current_page, offset_number));
            }
            size = size.saturating_sub(this_page_size);
            if size > 0 {
                data = &data[this_page_size..];
                let new_page = WritablePage::new(self.index, self.page_type);
                current_block_number = new_page.get_block_number();
                current_page.set_next_block_number(current_block_number);
                new_page.commit();
            }
            current_page.commit();
        }
        ip.unwrap()
    }

    pub unsafe fn overwrite(&mut self, ip: ItemPointer, data: &[u8]) {
        let mut current_ip = ip;
        let mut data = data;

        let mut size = data.len();
        while size > 0 {
            let buf = current_ip.modify_bytes(self.index);
            let mut buf_slice = buf.get_data_slice();
            let this_page_size = buf_slice.len();
            assert!(size > this_page_size || size == this_page_size);
            buf_slice.write_all(&data[..this_page_size]).unwrap();
            size = size.saturating_sub(this_page_size);
            if size > 0 {
                data = &data[this_page_size..];
                let blk = buf._page.get_next_block_number();
                assert!(blk != InvalidBlockNumber);
                current_ip = ItemPointer::new(blk, 1);
            }
            buf.commit();
        }
    }

    pub unsafe fn read(&mut self, ip: ItemPointer) -> Vec<u8> {
        let mut current_ip = Some(ip);
        let mut res = Vec::with_capacity(8094);

        while current_ip.is_some() {
            let buf = current_ip.unwrap().read_bytes(self.index);
            res.extend_from_slice(buf.get_data_slice());
            let blk = buf._page.get_next_block_number();
            if blk == InvalidBlockNumber {
                current_ip = None
            } else {
                current_ip = Some(ItemPointer::new(blk, 1));
            }
        }
        res
    }

    pub fn close(self) {
        std::mem::drop(self)
    }
}
