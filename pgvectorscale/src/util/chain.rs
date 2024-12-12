//! This module defines the `ChainTape` data structure, which is used to store large data items that
//! are too big to fit in a single page.  See `Tape` for a simpler version that assumes each data
//! item fits in a single page.
//!
//! All page entries begin with a header that contains a block number of the next page in the chain,
//! if applicable.  If there is a next page, the entry continues (at position 1) on that page.
//!
//! The implementation supports an append-only sequence of writes via `ChainTapeWriter` and reads
//! via `ChainTapeReader`.  The writer returns an `ItemPointer` that can be used to read the data
//! back.  Reads are done via an iterator that returns `ReadableBuffer` objects for the segments
//! of the data.

use anyhow::Result;

use pgrx::{
    pg_sys::{BlockNumber, InvalidBlockNumber},
    PgRelation,
};
use rkyv::{Archive, Deserialize, Serialize};

use crate::access_method::stats::{StatsNodeRead, StatsNodeWrite};

use super::{
    page::{PageType, ReadablePage, WritablePage},
    ItemPointer, ReadableBuffer,
};

#[derive(Clone, PartialEq, Archive, Deserialize, Serialize)]
#[archive(check_bytes)]
struct ChainItemHeader {
    next: BlockNumber,
}

const CHAIN_HEADER_SIZE: usize = std::mem::size_of::<ArchivedChainItemHeader>();

// Empirically-measured slop factor for how much `pg_sys::PageGetFreeSpace` can
// overestimate the free space in a page in our usage patterns.
const PG_SLOP_SIZE: usize = 4;

pub struct ChainTapeWriter<'a, S: StatsNodeWrite> {
    page_type: PageType,
    index: &'a PgRelation,
    current: BlockNumber,
    stats: &'a mut S,
}

impl<'a, S: StatsNodeWrite> ChainTapeWriter<'a, S> {
    /// Create a ChainingTape that starts writing on a new page.
    pub fn new(index: &'a PgRelation, page_type: PageType, stats: &'a mut S) -> Self {
        assert!(page_type.is_chained());
        let page = WritablePage::new(index, page_type);
        let block_number = page.get_block_number();
        page.commit();
        Self {
            page_type,
            index,
            current: block_number,
            stats,
        }
    }

    /// Write chained data to the tape, returning an `ItemPointer` to the start of the data.
    pub fn write(&mut self, mut data: &[u8]) -> Result<super::ItemPointer> {
        let mut current_page = WritablePage::modify(self.index, self.current);

        // If there isn't enough space for the header plus some data, start a new page.
        if current_page.get_free_space() + PG_SLOP_SIZE < CHAIN_HEADER_SIZE + 1 {
            current_page = WritablePage::new(self.index, self.page_type);
            self.current = current_page.get_block_number();
        }

        // ItemPointer to the first item in the chain.
        let mut result: Option<super::ItemPointer> = None;

        // Write the data in chunks, creating new pages as needed.
        while CHAIN_HEADER_SIZE + data.len() + PG_SLOP_SIZE > current_page.get_free_space() {
            let next_page = WritablePage::new(self.index, self.page_type);
            let header = ChainItemHeader {
                next: next_page.get_block_number(),
            };
            let header_bytes = rkyv::to_bytes::<_, 256>(&header).unwrap();
            let data_size = current_page.get_free_space() - PG_SLOP_SIZE - CHAIN_HEADER_SIZE;
            let chunk = &data[..data_size];
            let combined = [header_bytes.as_slice(), chunk].concat();
            let offset_number = current_page.add_item(combined.as_ref());
            result.get_or_insert_with(|| {
                ItemPointer::new(current_page.get_block_number(), offset_number)
            });
            current_page.commit();
            self.stats.record_write();
            current_page = next_page;
            data = &data[data_size..];
        }

        // Write the last chunk of data.
        let header = ChainItemHeader {
            next: InvalidBlockNumber,
        };
        let header_bytes = rkyv::to_bytes::<_, 256>(&header).unwrap();
        let combined = [header_bytes.as_slice(), data].concat();
        let offset_number = current_page.add_item(combined.as_ref());
        let result = result
            .unwrap_or_else(|| ItemPointer::new(current_page.get_block_number(), offset_number));
        self.current = current_page.get_block_number();
        current_page.commit();
        self.stats.record_write();

        Ok(result)
    }
}

pub struct ChainTapeReader<'a, S: StatsNodeRead> {
    page_type: PageType,
    index: &'a PgRelation,
    stats: &'a mut S,
}

impl<'a, S: StatsNodeRead> ChainTapeReader<'a, S> {
    pub fn new(index: &'a PgRelation, page_type: PageType, stats: &'a mut S) -> Self {
        assert!(page_type.is_chained());
        Self {
            page_type,
            index,
            stats,
        }
    }

    pub fn read(&'a mut self, ip: ItemPointer) -> ChainedItemIterator<'a, S> {
        ChainedItemIterator {
            index: self.index,
            ip,
            page_type: self.page_type,
            stats: self.stats,
        }
    }
}

pub struct ChainedItemIterator<'a, S: StatsNodeRead> {
    index: &'a PgRelation,
    ip: ItemPointer,
    page_type: PageType,
    stats: &'a mut S,
}

impl<'a, S: StatsNodeRead> Iterator for ChainedItemIterator<'a, S> {
    type Item = ReadableBuffer<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.ip.block_number == InvalidBlockNumber {
            return None;
        }

        unsafe {
            let page = ReadablePage::read(self.index, self.ip.block_number);
            self.stats.record_read();
            assert!(page.get_type() == self.page_type);
            let mut item = page.get_item_unchecked(self.ip.offset);
            let slice = item.get_data_slice();
            let header_slice = &slice[..CHAIN_HEADER_SIZE];
            let header =
                rkyv::from_bytes::<ChainItemHeader>(header_slice).expect("failed to read header");
            self.ip = ItemPointer::new(header.next, 1);
            item.advance(CHAIN_HEADER_SIZE);

            Some(item)
        }
    }
}

#[cfg(any(test, feature = "pg_test"))]
#[pgrx::pg_schema]
mod tests {
    use pgrx::{
        pg_sys::{self, BLCKSZ},
        pg_test, Spi,
    };

    use crate::access_method::stats::InsertStats;

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
    #[allow(clippy::needless_range_loop)]
    fn test_chain_tape() {
        let mut rstats = InsertStats::default();
        let mut wstats = InsertStats::default();

        let index = make_test_relation();
        {
            // ChainTape can be used for small items too
            let mut tape = ChainTapeWriter::new(&index, PageType::SbqMeans, &mut wstats);
            for _ in 0..100 {
                let data = b"hello world";
                let ip = tape.write(data).unwrap();
                let mut reader = ChainTapeReader::new(&index, PageType::SbqMeans, &mut rstats);

                let mut iter = reader.read(ip);
                let item = iter.next().unwrap();
                assert_eq!(item.get_data_slice(), data);
                assert!(iter.next().is_none());
            }
        }

        for data_size in BLCKSZ - 100..BLCKSZ + 100 {
            // Exhaustively test around the neighborhood of a page size
            let mut bigdata = vec![0u8; data_size as usize];
            for i in 0..bigdata.len() {
                bigdata[i] = (i % 256) as u8;
            }

            let mut tape = ChainTapeWriter::new(&index, PageType::SbqMeans, &mut wstats);
            for _ in 0..10 {
                let ip = tape.write(&bigdata).unwrap();
                let mut count = 0;
                let mut reader = ChainTapeReader::new(&index, PageType::SbqMeans, &mut rstats);
                for item in reader.read(ip) {
                    assert_eq!(item.get_data_slice(), &bigdata[count..count + item.len]);
                    count += item.len;
                }
                assert_eq!(count, bigdata.len());
            }
        }

        for data_size in (2 * BLCKSZ - 100)..(2 * BLCKSZ + 100) {
            // Exhaustively test around the neighborhood of a 2-page size
            let mut bigdata = vec![0u8; data_size as usize];
            for i in 0..bigdata.len() {
                bigdata[i] = (i % 256) as u8;
            }

            let mut tape = ChainTapeWriter::new(&index, PageType::SbqMeans, &mut wstats);
            for _ in 0..10 {
                let ip = tape.write(&bigdata).unwrap();
                let mut count = 0;
                let mut reader = ChainTapeReader::new(&index, PageType::SbqMeans, &mut rstats);
                for item in reader.read(ip) {
                    assert_eq!(item.get_data_slice(), &bigdata[count..count + item.len]);
                    count += item.len;
                }
                assert_eq!(count, bigdata.len());
            }
        }

        for data_size in (3 * BLCKSZ - 100)..(3 * BLCKSZ + 100) {
            // Exhaustively test around the neighborhood of a 3-page size
            let mut bigdata = vec![0u8; data_size as usize];
            for i in 0..bigdata.len() {
                bigdata[i] = (i % 256) as u8;
            }

            let mut tape = ChainTapeWriter::new(&index, PageType::SbqMeans, &mut wstats);
            for _ in 0..10 {
                let ip = tape.write(&bigdata).unwrap();
                let mut count = 0;
                let mut reader = ChainTapeReader::new(&index, PageType::SbqMeans, &mut rstats);
                for item in reader.read(ip) {
                    assert_eq!(item.get_data_slice(), &bigdata[count..count + item.len]);
                    count += item.len;
                }
                assert_eq!(count, bigdata.len());
            }
        }
    }
}
