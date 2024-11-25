use pgrx::pg_sys::{BufferGetBlockNumber, InvalidBlockNumber, InvalidOffsetNumber};
use pgrx::*;
use pgvectorscale_derive::{Readable, Writeable};
use rkyv::{Archive, Deserialize, Serialize};
use semver::Version;

use crate::access_method::options::TSVIndexOptions;
use crate::util::page;
use crate::util::*;

use super::distance::{DistanceFn, DistanceType};
use super::options::{
    NUM_DIMENSIONS_DEFAULT_SENTINEL, NUM_NEIGHBORS_DEFAULT_SENTINEL,
    SBQ_NUM_BITS_PER_DIMENSION_DEFAULT_SENTINEL,
};
use super::sbq::SbqNode;
use super::stats::StatsNodeModify;
use super::storage::StorageType;

const TSV_MAGIC_NUMBER: u32 = 768756476; //Magic number, random
const TSV_VERSION: u32 = 2;
const GRAPH_SLACK_FACTOR: f64 = 1.3_f64;

const META_BLOCK_NUMBER: pg_sys::BlockNumber = 0;
const META_HEADER_OFFSET: pgrx::pg_sys::OffsetNumber = 1;
const META_OFFSET: pgrx::pg_sys::OffsetNumber = 2;
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
    _pq_vector_length: usize,
    _pq_block_number: pg_sys::BlockNumber,
    _pq_block_offset: pg_sys::OffsetNumber,
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
        if self.use_pq {
            pgrx::error!("PQ is no longer supported. Please rebuild the TSV index.");
        }

        MetaPage {
            magic_number: TSV_MAGIC_NUMBER,
            version: TSV_VERSION,
            extension_version_when_built: "0.0.2".to_string(),
            distance_type: DistanceType::L2 as u16,
            num_dimensions: self.num_dimensions,
            num_dimensions_to_index: self.num_dimensions,
            bq_num_bits_per_dimension: 1,
            num_neighbors: self.num_neighbors,
            storage_type: StorageType::Plain as u8,
            search_list_size: self.search_list_size,
            max_alpha: self.max_alpha,
            init_ids: ItemPointer::new(self.init_ids_block_number, self.init_ids_offset),
            quantizer_metadata: ItemPointer::new(InvalidBlockNumber, InvalidOffsetNumber),
        }
    }
}

/// This is metadata header. It contains just the magic number and version number.
/// Stored as the first page (offset 1) in the index relation.
/// The header is separate from the actual metadata to allow for future-proofing.
/// In particular, if the metadata format changes, we can still read the header to check the version.
#[derive(Clone, PartialEq, Archive, Deserialize, Serialize, Readable, Writeable)]
#[archive(check_bytes)]
pub struct MetaPageHeader {
    /// random magic number for identifying the index
    magic_number: u32,
    /// version number for future-proofing
    version: u32,
}

/// This is metadata about the entire index.
/// Stored as the first page (offset 2) in the index relation.
#[derive(Clone, PartialEq, Archive, Deserialize, Serialize, Readable, Writeable)]
#[archive(check_bytes)]
pub struct MetaPage {
    /// repeat the magic number and version from MetaPageHeader for sanity checks
    magic_number: u32,
    version: u32,
    extension_version_when_built: String,
    /// The value of the DistanceType enum
    distance_type: u16,
    /// number of total_dimensions in the vector
    num_dimensions: u32,
    //number of dimensions in the vectors stored in the index
    num_dimensions_to_index: u32,
    bq_num_bits_per_dimension: u8,
    /// the value of the TSVStorageLayout enum
    storage_type: u8,
    /// max number of outgoing edges a node in the graph can have (R in the papers)
    num_neighbors: u32,
    search_list_size: u32,
    max_alpha: f64,
    init_ids: ItemPointer,
    quantizer_metadata: ItemPointer,
}

impl MetaPage {
    /// Number of dimensions in the vectors being stored.
    /// Has to be the same for all vectors in the graph and cannot change.
    pub fn get_num_dimensions(&self) -> u32 {
        self.num_dimensions
    }

    pub fn get_num_dimensions_to_index(&self) -> u32 {
        self.num_dimensions_to_index
    }

    pub fn get_bq_num_bits_per_dimension(&self) -> u8 {
        self.bq_num_bits_per_dimension
    }

    pub fn get_num_dimensions_for_neighbors(&self) -> u32 {
        match StorageType::from_u8(self.storage_type) {
            StorageType::Plain => {
                error!("get_num_dimensions_for_neighbors should not be called for Plain storage")
            }
            StorageType::SbqSpeedup => self.num_dimensions_to_index,
            StorageType::SbqCompression => 0,
        }
    }

    /// Maximum number of neigbors per node. Given that we pre-allocate
    /// these many slots for each node, this cannot change after the graph is built.
    pub fn get_num_neighbors(&self) -> u32 {
        self.num_neighbors
    }

    pub fn get_search_list_size_for_build(&self) -> u32 {
        self.search_list_size
    }

    pub fn get_max_alpha(&self) -> f64 {
        self.max_alpha
    }

    pub fn get_distance_function(&self) -> DistanceFn {
        DistanceType::from_u16(self.distance_type).get_distance_function()
    }

    pub fn get_distance_type(&self) -> DistanceType {
        DistanceType::from_u16(self.distance_type)
    }

    pub fn get_storage_type(&self) -> StorageType {
        StorageType::from_u8(self.storage_type)
    }

    pub fn get_max_neighbors_during_build(&self) -> usize {
        ((self.get_num_neighbors() as f64) * GRAPH_SLACK_FACTOR).ceil() as usize
    }

    pub fn get_init_ids(&self) -> Option<Vec<IndexPointer>> {
        if !self.init_ids.is_valid() {
            return None;
        }

        Some(vec![self.init_ids])
    }

    pub fn get_quantizer_metadata_pointer(&self) -> Option<IndexPointer> {
        if !self.quantizer_metadata.is_valid() {
            return None;
        }

        match self.get_storage_type() {
            StorageType::Plain => None,
            StorageType::SbqSpeedup | StorageType::SbqCompression => Some(self.quantizer_metadata),
        }
    }

    fn calculate_num_neighbors(
        num_dimensions: u32,
        num_bits_per_dimension: u8,
        opt: &PgBox<TSVIndexOptions>,
    ) -> u32 {
        let num_neighbors = (*opt).get_num_neighbors();
        if num_neighbors == NUM_NEIGHBORS_DEFAULT_SENTINEL {
            match (*opt).get_storage_type() {
                StorageType::Plain => 50,
                StorageType::SbqSpeedup => SbqNode::get_default_num_neighbors(
                    num_dimensions as usize,
                    num_dimensions as usize,
                    num_bits_per_dimension,
                ) as u32,
                StorageType::SbqCompression => 50,
            }
        } else {
            num_neighbors as u32
        }
    }

    /// Write out a new meta page.
    /// Has to be done as the first write to a new relation.
    pub unsafe fn create(
        index: &PgRelation,
        num_dimensions: u32,
        distance_type: DistanceType,
        opt: PgBox<TSVIndexOptions>,
    ) -> MetaPage {
        let version = Version::parse(env!("CARGO_PKG_VERSION")).unwrap();

        let num_dimensions_to_index = if opt.num_dimensions == NUM_DIMENSIONS_DEFAULT_SENTINEL {
            num_dimensions
        } else {
            opt.num_dimensions
        };

        let bq_num_bits_per_dimension =
            if opt.bq_num_bits_per_dimension == SBQ_NUM_BITS_PER_DIMENSION_DEFAULT_SENTINEL {
                if (*opt).get_storage_type() == StorageType::SbqCompression
                    && num_dimensions_to_index < 900
                {
                    2
                } else {
                    1
                }
            } else {
                opt.bq_num_bits_per_dimension as u8
            };

        if bq_num_bits_per_dimension > 1 && num_dimensions_to_index > 930 {
            //limited by SbqMeans fitting on a page
            pgrx::error!("SBQ with more than 1 bit per dimension is not supported for more than 900 dimensions");
        }
        if bq_num_bits_per_dimension > 1 && (*opt).get_storage_type() != StorageType::SbqCompression
        {
            pgrx::error!(
                "SBQ with more than 1 bit per dimension is only supported with the memory_optimized storage layout"
            );
        }

        let meta = MetaPage {
            magic_number: TSV_MAGIC_NUMBER,
            version: TSV_VERSION,
            extension_version_when_built: version.to_string(),
            distance_type: distance_type as u16,
            num_dimensions,
            num_dimensions_to_index,
            storage_type: (*opt).get_storage_type() as u8,
            num_neighbors: Self::calculate_num_neighbors(
                num_dimensions,
                bq_num_bits_per_dimension,
                &opt,
            ),
            bq_num_bits_per_dimension,
            search_list_size: opt.search_list_size,
            max_alpha: opt.max_alpha,
            init_ids: ItemPointer::new(InvalidBlockNumber, InvalidOffsetNumber),
            quantizer_metadata: ItemPointer::new(InvalidBlockNumber, InvalidOffsetNumber),
        };
        let page = page::WritablePage::new(index, crate::util::page::PageType::Meta);
        meta.write_to_page(page);
        meta
    }

    unsafe fn write_to_page(&self, mut page: page::WritablePage) {
        let header = MetaPageHeader {
            magic_number: self.magic_number,
            version: self.version,
        };

        assert!(header.magic_number == TSV_MAGIC_NUMBER);
        assert!(header.version == TSV_VERSION);

        //serialize the header
        let bytes = header.serialize_to_vec();
        let off = page.add_item(&bytes);
        assert!(off == META_HEADER_OFFSET);

        //serialize the meta
        let bytes = self.serialize_to_vec();
        let off = page.add_item(&bytes);
        assert!(off == META_OFFSET);

        page.commit();
    }

    unsafe fn overwrite(index: &PgRelation, new_meta: &MetaPage) {
        let mut page = page::WritablePage::modify(index, META_BLOCK_NUMBER);
        page.reinit(crate::util::page::PageType::Meta);
        new_meta.write_to_page(page);

        let page = page::ReadablePage::read(index, META_BLOCK_NUMBER);
        let page_type = page.get_type();
        if page_type != crate::util::page::PageType::Meta {
            pgrx::error!(
                "Problem upgrading meta page: wrong page type: {:?}",
                page_type
            );
        }
        let meta = Self::get_meta_from_page(page);
        if meta != *new_meta {
            pgrx::error!("Problem upgrading meta page: meta mismatch");
        }
    }

    /// Read the meta page for an index
    pub fn fetch(index: &PgRelation) -> MetaPage {
        unsafe {
            let page = page::ReadablePage::read(index, META_BLOCK_NUMBER);
            let page_type = page.get_type();
            if page_type == crate::util::page::PageType::MetaV1 {
                let old_meta = MetaPageV1::page_get_meta(*page, *(*(page.get_buffer())));
                let new_meta = (*old_meta).get_new_meta();

                //release the page
                std::mem::drop(page);

                Self::overwrite(index, &new_meta);
                return new_meta;
            }
            Self::get_meta_from_page(page)
        }
    }

    unsafe fn get_meta_from_page(page: page::ReadablePage) -> MetaPage {
        //check the header. In the future, we can use this to check the version
        let rb = page.get_item_unchecked(META_HEADER_OFFSET);
        let meta = ReadableMetaPageHeader::with_readable_buffer(rb);
        let archived = meta.get_archived_node();
        assert!(archived.magic_number == TSV_MAGIC_NUMBER);
        assert!(archived.version == TSV_VERSION);

        let page = meta.get_owned_page();

        //retrieve the MetaPage itself and deserialize it
        let rb = page.get_item_unchecked(META_OFFSET);
        let meta = ReadableMetaPage::with_readable_buffer(rb);
        let archived = meta.get_archived_node();
        assert!(archived.magic_number == TSV_MAGIC_NUMBER);
        assert!(archived.version == TSV_VERSION);

        archived.deserialize(&mut rkyv::Infallible).unwrap()
    }

    /// Change the init ids for an index.
    pub fn update_init_ids<S: StatsNodeModify>(
        index: &PgRelation,
        init_ids: Vec<IndexPointer>,
        stats: &mut S,
    ) {
        assert_eq!(init_ids.len(), 1); //change this if we support multiple
        let id = init_ids[0];

        let mut meta = Self::fetch(index);
        meta.init_ids = id;

        unsafe {
            Self::overwrite(index, &meta);
            stats.record_modify();
        };
    }

    pub fn update_quantizer_metadata_pointer<S: StatsNodeModify>(
        index: &PgRelation,
        quantizer_pointer: IndexPointer,
        stats: &mut S,
    ) {
        let mut meta = Self::fetch(index);
        meta.quantizer_metadata = quantizer_pointer;

        unsafe {
            Self::overwrite(index, &meta);
            stats.record_modify();
        };
    }
}
