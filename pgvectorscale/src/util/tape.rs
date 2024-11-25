//! Tape provides a simple infinite-tape-writing abstraction over postgres pages.

use super::page::{PageType, ReadablePage, WritablePage};
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
    /// Create a Tape that starts writing on a new page.
    pub unsafe fn new(index: &'a PgRelation, page_type: PageType) -> Self {
        let page = WritablePage::new(index, page_type);
        let block_number = page.get_block_number();
        page.commit();
        Self {
            page_type,
            index,
            current: block_number,
        }
    }

    /// Create a Tape that resumes writing on the newest page of the given type, if possible.
    pub unsafe fn resume(index: &'a PgRelation, page_type: PageType) -> Self {
        let nblocks = pg_sys::RelationGetNumberOfBlocksInFork(
            index.as_ptr(),
            pg_sys::ForkNumber::MAIN_FORKNUM,
        );
        let mut current_block = None;
        for block in (0..nblocks).rev() {
            if ReadablePage::read(index, block).get_type() == page_type {
                current_block = Some(block);
                break;
            }
        }
        match current_block {
            Some(current) => Tape {
                index,
                page_type,
                current,
            },
            None => Tape::new(index, page_type),
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

    pub fn close(self) {}
}

#[cfg(any(test, feature = "pg_test"))]
#[pgrx::pg_schema]
mod tests {
    use super::*;

    fn make_test_relation() -> PgRelation {
        Spi::run(
            "CREATE TABLE test(encoding vector(3));
        CREATE INDEX idxtest
                  ON test
               USING diskann(encoding)
                WITH (num_neighbors=30);",
        )
        .unwrap();

        let index_oid = Spi::get_one::<pg_sys::Oid>("SELECT 'idxtest'::regclass::oid")
            .unwrap()
            .expect("oid was null");
        unsafe { PgRelation::from_pg(pg_sys::RelationIdGetRelation(index_oid)) }
    }

    #[pg_test]
    fn tape_resume() {
        let indexrel = make_test_relation();
        unsafe {
            let node_page = {
                let mut tape = Tape::new(&indexrel, PageType::Node);
                let node_page = tape.current;
                let ip = tape.write(&[1, 2, 3]);
                assert_eq!(
                    ip.block_number, node_page,
                    "Tape block number should match IP"
                );
                assert_eq!(ip.offset, 1, "IP offset should be correct");
                let ip = tape.write(&[4, 5, 6]);
                assert_eq!(
                    ip.block_number, node_page,
                    "Tape block number should match IP"
                );
                assert_eq!(
                    tape.current, node_page,
                    "Data should be written to page with enough room"
                );
                node_page
            };

            {
                let mut tape = Tape::resume(&indexrel, PageType::SbqMeans);
                let ip = tape.write(&[99]);
                assert_eq!(
                    ip.block_number, tape.current,
                    "Tape block number should match IP"
                );
                assert_ne!(
                    tape.current, node_page,
                    "Data can only be written to page of its type"
                );
            }

            {
                let mut tape = Tape::resume(&indexrel, PageType::PqQuantizerVector);
                let ip = tape.write(&[99]);
                assert_eq!(
                    ip.block_number, tape.current,
                    "Tape block number should match IP"
                );
                assert_eq!(
                    tape.current,
                    node_page + 1,
                    "An unseen page type must create a new page"
                );
            }

            {
                let mut tape = Tape::resume(&indexrel, PageType::Node);
                let ip = tape.write(&[7, 8, 9]);
                assert_eq!(
                    ip.block_number, tape.current,
                    "Tape block number should match IP"
                );
                tape.write(&[10, 11, 12]);
                assert_eq!(
                    ip.block_number, tape.current,
                    "Tape block number should match IP"
                );
                assert_eq!(
                    tape.current, node_page,
                    "Data should be written to existing page when there is room"
                );

                let page = WritablePage::modify(tape.index, tape.current);
                assert_eq!(page.get_free_space(), 8108);
            }

            {
                let mut tape = Tape::resume(&indexrel, PageType::Node);
                let ip = tape.write(&[42; 8109]);
                assert_eq!(
                    ip.block_number, tape.current,
                    "Tape block number should match IP"
                );
                assert_ne!(
                    tape.current, node_page,
                    "Writing more than available forces a new page"
                );
            }
        }
    }
}
