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

const ARCHIVED_CHAIN_HEADER_SIZE: usize = std::mem::size_of::<ArchivedChainItemHeader>();

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
        let header_size = ARCHIVED_CHAIN_HEADER_SIZE;

        if current_page.get_free_space() <= header_size {
            current_page = WritablePage::new(self.index, self.page_type);
            self.current = current_page.get_block_number();
        }

        let mut result: Option<super::ItemPointer> = None;

        while header_size + data.len() > current_page.get_free_space() {
            let next_page = WritablePage::new(self.index, self.page_type);
            let header = ChainItemHeader {
                next: next_page.get_block_number(),
            };
            let header_bytes = rkyv::to_bytes::<_, 256>(&header).unwrap();
            let data_size = current_page.get_free_space() - header_size;
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
            let header_slice = &slice[..ARCHIVED_CHAIN_HEADER_SIZE];
            let header =
                rkyv::from_bytes::<ChainItemHeader>(header_slice).expect("failed to read header");
            self.ip = ItemPointer::new(header.next, 1);
            item.advance(ARCHIVED_CHAIN_HEADER_SIZE);

            Some(item)
        }
    }
}
