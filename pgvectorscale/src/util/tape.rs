//! Tape provides a simple infinite-tape-writing abstraction over postgres pages.

use super::page::{PageType, WritablePage};
use pgrx::{
    pg_sys::{BlockNumber, BLCKSZ},
    *,
};

pub struct Tape<'a> {
    page_type: PageType,
    index: &'a PgRelation,
    current: BlockNumber,
}

impl<'a> Tape<'a> {
    pub unsafe fn new(index: &'a PgRelation, page_type: PageType) -> Self {
        let page = WritablePage::new(index, page_type);
        let block_number = page.get_block_number();
        page.commit();
        Self {
            page_type,
            index: index,
            current: block_number,
        }
    }

    pub unsafe fn write(&mut self, data: &[u8]) -> super::ItemPointer {
        let size = data.len();
        assert!(size < BLCKSZ as usize);

        let mut current_page = WritablePage::modify(self.index, self.current);

        //don't split data over pages. Depending on packing,
        //we may have to implement that in the future.
        if current_page.get_free_space() < size {
            //TODO update forward pointer;

            current_page = WritablePage::new(self.index, self.page_type);
            self.current = current_page.get_block_number();
            if current_page.get_free_space() < size {
                panic!("Not enough free space on new page");
            }
        }
        let offset_number = current_page.add_item_unchecked(data);

        let item_pointer = super::ItemPointer::with_page(&current_page, offset_number);
        current_page.commit();
        item_pointer
    }

    pub fn close(self) {
        std::mem::drop(self)
    }
}
