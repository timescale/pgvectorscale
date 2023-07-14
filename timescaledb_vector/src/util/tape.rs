//! Tape provides a simple infinite-tape-writing abstraction over postgres pages.

use super::page::WritablePage;
use pg_sys::Relation;
use pgrx::{
    pg_sys::{BlockNumber, BLCKSZ},
    *,
};

pub struct Tape {
    index: Relation,
    current: BlockNumber,
}

impl Tape {
    pub unsafe fn new(index: Relation) -> Self {
        let page = WritablePage::new(index);
        let block_number = page.get_block_number();
        page.commit();
        Self {
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

            current_page = WritablePage::new(self.index);
            self.current = current_page.get_block_number();
            if current_page.get_free_space() < size {
                panic!("Not enough free space on new page");
            }
        }
        let offset_number = pg_sys::PageAddItemExtended(
            *current_page,
            data.as_ptr() as _,
            size,
            pg_sys::InvalidOffsetNumber,
            0,
        );

        assert!(offset_number != pg_sys::InvalidOffsetNumber);
        let index_pointer = super::ItemPointer::with_page(&current_page, offset_number);
        current_page.commit();
        index_pointer
    }

    pub fn close(self) {
        std::mem::drop(self)
    }
}
