//! `Tape`` provides a simple infinite-tape read/write abstraction over postgres pages.
//! Page types fall into two categories: chained and non-chained. Chained pages can
//! contain more data than a single page, and are used for large data payloads.
//! Non-chained pages are used for small data payloads that fit within a single page.

use anyhow::Result;

use super::{
    page::{PageType, ReadablePage, WritablePage},
    ItemPointer, ReadableBuffer,
};
use pgrx::{
    pg_sys::{
        BlockNumber, ForkNumber, InvalidBlockNumber, RelationGetNumberOfBlocksInFork, BLCKSZ,
    },
    PgRelation,
};
use rkyv::{AlignedVec, Archive, Deserialize, Serialize};

pub struct Tape<'a> {
    page_type: PageType,
    index: &'a PgRelation,
    current: BlockNumber,
}

impl<'a> Tape<'a> {
    /// Create a `Tape` that starts writing on a new page.
    pub fn new(index: &'a PgRelation, page_type: PageType) -> Self {
        let page = WritablePage::new(index, page_type);
        let block_number = page.get_block_number();
        page.commit();
        Self {
            page_type,
            index,
            current: block_number,
        }
    }

    /// Create a `Tape` that resumes writing on the newest page of the given type, if possible.
    pub unsafe fn resume(index: &'a PgRelation, page_type: PageType) -> Self {
        let nblocks = RelationGetNumberOfBlocksInFork(index.as_ptr(), ForkNumber::MAIN_FORKNUM);
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

    /// Replace the `Tape` that begins at the given block_number with a new, empty one.
    pub fn replace(
        index: &'a PgRelation,
        page_type: PageType,
        block_number: BlockNumber,
    ) -> Result<Self> {
        let mut page = WritablePage::modify(index, block_number);
        if page.get_type() != page_type {
            return Err(anyhow::anyhow!(
                "Page type mismatch: expected {:?}, got {:?}",
                page_type,
                page.get_type()
            ));
        }

        // TODO: free the rest of the pages in the chain

        page.reinit(page_type);

        Ok(Tape {
            page_type,
            index,
            current: block_number,
        })
    }

    /// Allocate a new page
    unsafe fn alloc_page(&mut self) -> WritablePage<'a> {
        let mut page = WritablePage::new(self.index, self.page_type);
        if self.page_type.is_chained() {
            let header = ChainedPageHeader::new(InvalidBlockNumber);
            let bytes = header.to_bytes();
            let off = page.add_item(&bytes);
            assert_eq!(
                off, CHAINED_PAGE_HEADER_POS,
                "Chained page header should be at offset {CHAINED_PAGE_HEADER_POS}"
            );
        }
        page
    }

    /// Read data from the tape at the given `ItemPointer`.  Should only be used for non-chained pages.
    pub unsafe fn read(&self, page_type: PageType, ip: ItemPointer) -> Result<ReadableBuffer<'a>> {
        assert!(!page_type.is_chained());
        let page = ReadablePage::read(self.index, ip.block_number);
        if page_type != page.get_type() {
            return Err(anyhow::anyhow!(
                "Page type mismatch: expected {:?}, got {:?}",
                page_type,
                page.get_type()
            ));
        }
        Ok(page.get_item_unchecked(ip.offset))
    }

    /// Read chained data from the tape at the given `ItemPointer`.
    pub fn read_chained(
        &self,
        page_type: PageType,
        ip: ItemPointer,
        len: usize,
    ) -> SegmentIterator<'a> {
        assert!(page_type.is_chained());
        assert!(ip.offset > CHAINED_PAGE_FIRST_ITEM_POS);
        SegmentIterator {
            index: self.index,
            ip,
            len,
        }
    }

    /// Write data to the tape, returning an `ItemPointer` to the start of the data.
    pub unsafe fn write(&mut self, data: &[u8]) -> super::ItemPointer {
        let size = data.len();
        assert!(self.page_type.is_chained() || size < BLCKSZ as usize);
        let mut current_page = WritablePage::modify(self.index, self.current);

        if current_page.get_free_space() >= size {
            let offset_number = current_page.add_item_unchecked(data);
            let result = ItemPointer::with_page(&current_page, offset_number);
            current_page.commit();
            return result;
        }

        if self.page_type.is_chained() {
            if current_page.get_free_space() == 0 {
                current_page = self.alloc_page();
                self.current = current_page.get_block_number();
            }

            // Write as much data as will fit on the current page.
            let offset_number = current_page.add_item(&data[..current_page.get_free_space()]);
            let start_pos = ItemPointer::with_page(&current_page, offset_number);

            // Write the rest of the data on new pages.
            let mut data = &data[current_page.get_free_space()..];
            while data.len() > 0 {
                let next_page = self.alloc_page();
                let header = ChainedPageHeader::new(next_page.get_block_number());
                let bytes = header.to_bytes();
                current_page.replace_item(CHAINED_PAGE_HEADER_POS, &bytes);
                current_page.commit();

                current_page = next_page;
                self.current = current_page.get_block_number();

                current_page.add_item(&data[..current_page.get_free_space()]);
                data = &data[current_page.get_free_space()..];
            }
            current_page.commit();
            start_pos
        } else {
            // We do not split data across pages for non-chained pages
            current_page = WritablePage::new(self.index, self.page_type);
            self.current = current_page.get_block_number();
            if current_page.get_free_space() < size {
                panic!("Not enough free space on new page");
            }
            let offset_number = current_page.add_item_unchecked(data);
            let start_pos = ItemPointer::with_page(&current_page, offset_number);
            current_page.commit();
            start_pos
        }
    }

    pub fn close(self) {}
}

#[derive(Clone, PartialEq, Archive, Deserialize, Serialize)]
#[archive(check_bytes)]
struct ChainedPageHeader {
    next_page: pgrx::pg_sys::BlockNumber,
}

const CHAINED_PAGE_HEADER_POS: u16 = 1;
const CHAINED_PAGE_FIRST_ITEM_POS: u16 = 2;

impl ChainedPageHeader {
    fn new(next_page: BlockNumber) -> Self {
        Self { next_page }
    }

    fn to_bytes(self) -> AlignedVec {
        rkyv::to_bytes::<_, 256>(&self).unwrap()
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        rkyv::from_bytes::<Self>(bytes).unwrap()
    }
}

pub struct TapeIterator<'a> {
    index: &'a PgRelation,
    ip: ItemPointer,
    page_type: PageType,
}

impl<'a> Iterator for TapeIterator<'a> {
    type Item = ReadableBuffer<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.page_type.is_chained() {
            unsafe {
                let page = ReadablePage::read(self.index, ip.block_number);
                return Some(page.get_item_unchecked(ip.offset));
            }
        }

        if self.len == 0 {
            return None;
        }

        unsafe {
            let mut page = ReadablePage::read(self.index, self.ip.block_number);
            let header_buf = page.get_item_unchecked(CHAINED_PAGE_HEADER_POS);
            let header = ChainedPageHeader::from_bytes(header_buf.get_data_slice());
            page = header_buf.get_owned_page();

            let item = page.get_item_unchecked(self.ip.offset);
            assert!(item.len() <= self.len);
            self.len -= item.len();
            if self.len > 0 {
                assert!(header.next_page != InvalidBlockNumber);
                self.ip = ItemPointer {
                    block_number: header.next_page,
                    offset: CHAINED_PAGE_FIRST_ITEM_POS,
                };
            }

            Some(item)
        }
    }
}

#[cfg(any(test, feature = "pg_test"))]
#[pgrx::pg_schema]
mod tests {
    use pgrx::{pg_sys, pg_test, Spi};

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
    fn test_multipage_payload() {
        let indexrel = make_test_relation();
        unsafe {
            const PAYLOAD_SIZE: usize = (BLCKSZ + BLCKSZ / 2) as usize;
            let mut tape = Tape::new(&indexrel, PageType::Node);
            let ip = tape.write(&[1; PAYLOAD_SIZE]);
            assert_eq!(
                ip.block_number, tape.current,
                "Tape block number should match IP"
            );
            assert_eq!(ip.offset, 1, "IP offset should be correct");
            let ip = tape.write(&[2; PAYLOAD_SIZE]);
            assert_eq!(
                ip.block_number, tape.current,
                "Tape block number should match IP"
            );
            assert_eq!(ip.offset, 2, "IP offset should be correct");
            let ip = tape.write(&[3; PAYLOAD_SIZE]);
            assert_eq!(
                ip.block_number, tape.current,
                "Tape block number should match IP"
            );
            assert_eq!(ip.offset, 3, "IP offset should be correct");
        }
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
