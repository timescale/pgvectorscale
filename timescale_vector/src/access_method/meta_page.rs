use pgrx::pg_sys::{BufferGetBlockNumber, Pointer};
use pgrx::*;

use crate::access_method::options::TSVIndexOptions;
use crate::util::page;
use crate::util::*;

use super::storage::StorageType;

const TSV_MAGIC_NUMBER: u32 = 768756476; //Magic number, random
const TSV_VERSION: u32 = 2;
const GRAPH_SLACK_FACTOR: f64 = 1.3_f64;
/// This is old metadata version for extension versions <=0.0.2.
/// Note it is NOT repr(C)
#[derive(Clone)]
pub struct MetaPageV1 {
    /// random magic number for identifying the index
    magic_number: u32,
    /// version number for future-proofing
    version: u32,
    /// number of dimensions in the vector
    num_dimensions: u32,
    /// max number of outgoing edges a node in the graph can have (R in the papers)
    num_neighbors: u32,
    search_list_size: u32,
    max_alpha: f64,
    init_ids_block_number: pg_sys::BlockNumber,
    init_ids_offset: pg_sys::OffsetNumber,
    use_pq: bool,
    pq_vector_length: usize,
    pq_block_number: pg_sys::BlockNumber,
    pq_block_offset: pg_sys::OffsetNumber,
}

impl MetaPageV1 {
    /// Returns the MetaPage from a page.
    /// Should only be called from the very first page in a relation.
    unsafe fn page_get_meta(page: pg_sys::Page, buffer: pg_sys::Buffer) -> *mut MetaPageV1 {
        assert_eq!(BufferGetBlockNumber(buffer), 0);
        let meta_page = ports::PageGetContents(page) as *mut MetaPageV1;
        assert_eq!((*meta_page).magic_number, TSV_MAGIC_NUMBER);
        assert_eq!((*meta_page).version, 1);
        meta_page
    }

    pub fn get_new_meta(&self) -> MetaPage {
        MetaPage {
            magic_number: self.magic_number,
            version: self.version,
            num_dimensions: self.num_dimensions,
            num_neighbors: self.num_neighbors,
            search_list_size: self.search_list_size,
            max_alpha: self.max_alpha,
            init_ids_block_number: self.init_ids_block_number,
            init_ids_offset: self.init_ids_offset,
            use_pq: self.use_pq,
            pq_vector_length: self.pq_vector_length,
            pq_block_number: self.pq_block_number,
            pq_block_offset: self.pq_block_offset,
            use_bq: false,
        }
    }
}

/// This is metadata about the entire index.
/// Stored as the first page in the index relation.
#[derive(Clone, PartialEq)]
pub struct MetaPage {
    /// random magic number for identifying the index
    magic_number: u32,
    /// version number for future-proofing
    version: u32,
    /// number of dimensions in the vector
    num_dimensions: u32,
    /// max number of outgoing edges a node in the graph can have (R in the papers)
    num_neighbors: u32,
    search_list_size: u32,
    max_alpha: f64,
    init_ids_block_number: pg_sys::BlockNumber,
    init_ids_offset: pg_sys::OffsetNumber,
    use_pq: bool,
    pq_vector_length: usize,
    pq_block_number: pg_sys::BlockNumber,
    pq_block_offset: pg_sys::OffsetNumber,
    use_bq: bool,
}

impl MetaPage {
    /// Number of dimensions in the vectors being stored.
    /// Has to be the same for all vectors in the graph and cannot change.
    pub fn get_num_dimensions(&self) -> u32 {
        self.num_dimensions
    }

    /// Maximum number of neigbors per node. Given that we pre-allocate
    /// these many slots for each node, this cannot change after the graph is built.
    pub fn get_num_neighbors(&self) -> u32 {
        self.num_neighbors
    }

    pub fn get_pq_vector_length(&self) -> usize {
        self.pq_vector_length
    }

    pub fn get_search_list_size_for_build(&self) -> u32 {
        self.search_list_size
    }

    pub fn get_max_alpha(&self) -> f64 {
        self.max_alpha
    }

    fn get_use_pq(&self) -> bool {
        self.use_pq
    }

    pub fn get_storage_type(&self) -> StorageType {
        if self.get_use_pq() {
            StorageType::PqCompression
        } else if self.use_bq {
            StorageType::BqSpeedup
        } else {
            StorageType::Plain
        }
    }

    pub fn get_max_neighbors_during_build(&self) -> usize {
        return ((self.get_num_neighbors() as f64) * GRAPH_SLACK_FACTOR).ceil() as usize;
    }

    pub fn get_init_ids(&self) -> Option<Vec<IndexPointer>> {
        if self.init_ids_block_number == 0 && self.init_ids_offset == 0 {
            return None;
        }

        let ptr = HeapPointer::new(self.init_ids_block_number, self.init_ids_offset);
        Some(vec![ptr])
    }

    pub fn get_pq_pointer(&self) -> Option<IndexPointer> {
        if (!self.use_pq && !self.use_bq)
            || (self.pq_block_number == 0 && self.pq_block_offset == 0)
        {
            return None;
        }

        let ptr = IndexPointer::new(self.pq_block_number, self.pq_block_offset);
        Some(ptr)
    }

    /// Returns the MetaPage from a page.
    /// Should only be called from the very first page in a relation.
    unsafe fn page_get_meta(
        page: pg_sys::Page,
        buffer: pg_sys::Buffer,
        new: bool,
    ) -> *mut MetaPage {
        assert_eq!(BufferGetBlockNumber(buffer), 0);
        let meta_page = ports::PageGetContents(page) as *mut MetaPage;
        if !new {
            assert_eq!((*meta_page).magic_number, TSV_MAGIC_NUMBER);
        }
        meta_page
    }

    /// Write out a new meta page.
    /// Has to be done as the first write to a new relation.
    pub unsafe fn create(
        index: &PgRelation,
        num_dimensions: u32,
        opt: PgBox<TSVIndexOptions>,
    ) -> MetaPage {
        let meta = MetaPage {
            magic_number: TSV_MAGIC_NUMBER,
            version: TSV_VERSION,
            num_dimensions,
            num_neighbors: (*opt).num_neighbors,
            search_list_size: (*opt).search_list_size,
            max_alpha: (*opt).max_alpha,
            init_ids_block_number: 0,
            init_ids_offset: 0,
            use_pq: (*opt).use_pq,
            pq_vector_length: (*opt).pq_vector_length,
            pq_block_number: 0,
            pq_block_offset: 0,
            use_bq: (*opt).use_bq,
        };
        let page = page::WritablePage::new(index, crate::util::page::PageType::Meta);
        meta.write_to_page(page);
        meta
    }

    pub unsafe fn write_to_page(&self, page: page::WritablePage) {
        let meta = Self::page_get_meta(*page, *(*(page.get_buffer())), true);
        (*meta) = self.clone();
        let header = page.cast::<pgrx::pg_sys::PageHeaderData>();

        let meta_end = (meta as Pointer).add(std::mem::size_of::<MetaPage>());
        let page_start = (*page) as Pointer;
        (*header).pd_lower = meta_end.offset_from(page_start) as _;

        page.commit();
    }

    /// Read the meta page for an index
    pub fn read(index: &PgRelation) -> MetaPage {
        unsafe {
            let page = page::ReadablePage::read(index, 0);
            let page_type = page.get_type();
            if page_type == crate::util::page::PageType::MetaV1 {
                let old_meta = MetaPageV1::page_get_meta(*page, *(*(page.get_buffer())));
                let new_meta = (*old_meta).get_new_meta();

                std::mem::drop(page);
                let page = page::WritablePage::modify(index, 0);
                page.set_types(crate::util::page::PageType::Meta);
                new_meta.write_to_page(page);

                let page = page::ReadablePage::read(index, 0);
                let page_type = page.get_type();
                if page_type != crate::util::page::PageType::Meta {
                    pgrx::error!(
                        "Problem upgrading meta page: wrong page type: {:?}",
                        page_type
                    );
                }
                let meta = Self::page_get_meta(*page, *(*(page.get_buffer())), false);
                if *meta != new_meta {
                    pgrx::error!("Problem upgrading meta page: meta mismatch");
                }
                return new_meta;
            }
            let meta = Self::page_get_meta(*page, *(*(page.get_buffer())), false);
            (*meta).clone()
        }
    }

    /// Change the init ids for an index.
    pub fn update_init_ids(index: &PgRelation, init_ids: Vec<IndexPointer>) {
        assert_eq!(init_ids.len(), 1); //change this if we support multiple
        let id = init_ids[0];

        unsafe {
            let page = page::WritablePage::modify(index, 0);
            let meta = Self::page_get_meta(*page, *(*(page.get_buffer())), false);
            (*meta).init_ids_block_number = id.block_number;
            (*meta).init_ids_offset = id.offset;
            page.commit()
        }
    }

    pub fn update_pq_pointer(index: &PgRelation, pq_pointer: IndexPointer) {
        unsafe {
            let page = page::WritablePage::modify(index, 0);
            let meta = Self::page_get_meta(*page, *(*(page.get_buffer())), false);
            (*meta).pq_block_number = pq_pointer.block_number;
            (*meta).pq_block_offset = pq_pointer.offset;
            page.commit()
        }
    }
}
