use pgrx::pg_sys::{BufferGetBlockNumber, InvalidBlockNumber, InvalidOffsetNumber};
use pgrx::*;
use pgvectorscale_derive::{Readable, Writeable};
use rkyv::{Archive, Deserialize, Serialize};
use semver::Version;

use super::distance::{DistanceFn, DistanceType};
use super::options::{
    NUM_DIMENSIONS_DEFAULT_SENTINEL, NUM_NEIGHBORS_DEFAULT_SENTINEL,
    SBQ_NUM_BITS_PER_DIMENSION_DEFAULT_SENTINEL,
};
use super::start_nodes::StartNodes;
use super::storage::StorageType;
use super::storage_common::get_num_index_attributes;
use crate::access_method::node::{ReadableNode, WriteableNode};
use crate::access_method::options::TSVIndexOptions;
use crate::access_method::stats::WriteStats;
use crate::util::chain::{ChainItemReader, ChainTapeWriter};
use crate::util::page::{self, PageType};
use crate::util::*;

const TSV_MAGIC_NUMBER: u32 = 768756476; //Magic number, random
const TSV_VERSION: u32 = 3;
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
}

impl From<&MetaPageV1> for MetaPage {
    fn from(meta: &MetaPageV1) -> Self {
        if meta.use_pq {
            pgrx::error!("PQ is no longer supported. Please rebuild the TSV index.");
        }

        let start_nodes = StartNodes::new(ItemPointer::new(
            meta.init_ids_block_number,
            meta.init_ids_offset,
        ));

        MetaPage {
            magic_number: meta.magic_number,
            version: meta.version,
            extension_version_when_built: "0.0.2".to_string(),
            distance_type: DistanceType::L2 as u16,
            num_dimensions: meta.num_dimensions,
            num_dimensions_to_index: meta.num_dimensions,
            bq_num_bits_per_dimension: 1,
            storage_type: StorageType::Plain as u8,
            num_neighbors: meta.num_neighbors,
            search_list_size: meta.search_list_size,
            max_alpha: meta.max_alpha,
            start_nodes: Some(start_nodes),
            quantizer_metadata: ItemPointer::new(InvalidBlockNumber, InvalidOffsetNumber),
            has_labels: false,
        }
    }
}

/// This is metadata about the entire index.
/// Stored as the first page (offset 2) in the index relation.
#[derive(Clone, PartialEq, Archive, Deserialize, Serialize, Readable, Writeable)]
#[archive(check_bytes)]
pub struct MetaPageV2 {
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

impl MetaPageV2 {
    unsafe fn from_page(page: page::ReadablePage) -> Self {
        //check the header. In the future, we can use this to check the version
        let rb = page.get_item_unchecked(META_HEADER_OFFSET);
        let meta = ReadableMetaPageHeader::with_readable_buffer(rb);
        let archived = meta.get_archived_node();
        assert_eq!(archived.magic_number, TSV_MAGIC_NUMBER);
        assert_eq!(archived.version, 2);

        let page = meta.get_owned_page();

        //retrieve the MetaPage itself and deserialize it
        let rb = page.get_item_unchecked(META_OFFSET);
        let meta = ReadableMetaPageV2::with_readable_buffer(rb);
        let archived = meta.get_archived_node();
        assert_eq!(archived.magic_number, TSV_MAGIC_NUMBER);
        assert_eq!(archived.version, 2);

        archived.deserialize(&mut rkyv::Infallible).unwrap()
    }
}

impl From<MetaPageV2> for MetaPage {
    fn from(meta: MetaPageV2) -> Self {
        let start_nodes = StartNodes::new(meta.init_ids);

        MetaPage {
            magic_number: meta.magic_number,
            version: meta.version,
            extension_version_when_built: meta.extension_version_when_built,
            distance_type: meta.distance_type,
            num_dimensions: meta.num_dimensions,
            num_dimensions_to_index: meta.num_dimensions_to_index,
            bq_num_bits_per_dimension: meta.bq_num_bits_per_dimension,
            storage_type: meta.storage_type,
            num_neighbors: meta.num_neighbors,
            search_list_size: meta.search_list_size,
            max_alpha: meta.max_alpha,
            start_nodes: Some(start_nodes),
            quantizer_metadata: meta.quantizer_metadata,
            has_labels: false,
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
#[derive(Clone, Debug, PartialEq, Archive, Deserialize, Serialize, Readable, Writeable)]
#[archive(check_bytes)]
pub struct MetaPage {
    /// Magic number from MetaPageHeader for sanity check
    magic_number: u32,
    /// Version number from MetaPageHeader for sanity check
    version: u32,
    /// Version of the extension when the index was built
    extension_version_when_built: String,
    /// The value of the DistanceType enum
    distance_type: u16,
    /// Number of vector dimensions
    num_dimensions: u32,
    /// Number of vector dimensions that are indexed
    num_dimensions_to_index: u32,
    /// Number of bits per dimension for Sbq
    bq_num_bits_per_dimension: u8,
    /// The value of the TSVStorageLayout enum
    storage_type: u8,
    /// Max number of outgoing edges a node in the graph can have (R in the papers)
    num_neighbors: u32,
    /// Search list size (L in the papers)
    search_list_size: u32,
    /// Maximal alpha value for the index
    max_alpha: f64,
    /// Start nodes for search, one for each label.
    start_nodes: Option<StartNodes>,
    /// Sbq means metadata
    quantizer_metadata: ItemPointer,
    /// Whether the index has labels
    has_labels: bool,
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

    pub fn has_labels(&self) -> bool {
        self.has_labels
    }

    pub fn get_start_nodes(&self) -> Option<&StartNodes> {
        self.start_nodes.as_ref()
    }

    pub fn get_start_nodes_mut(&mut self) -> Option<&mut StartNodes> {
        self.start_nodes.as_mut()
    }

    pub fn set_start_nodes(&mut self, start_nodes: StartNodes) {
        self.start_nodes = Some(start_nodes);
    }

    pub fn get_quantizer_metadata_pointer(&self) -> Option<IndexPointer> {
        if !self.quantizer_metadata.is_valid() {
            return None;
        }

        match self.get_storage_type() {
            StorageType::Plain => None,
            StorageType::SbqCompression => Some(self.quantizer_metadata),
        }
    }

    fn calculate_num_neighbors(opt: &PgBox<TSVIndexOptions>) -> u32 {
        let num_neighbors = (*opt).get_num_neighbors();
        if num_neighbors == NUM_NEIGHBORS_DEFAULT_SENTINEL {
            match (*opt).get_storage_type() {
                StorageType::Plain => 50,
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

        let has_labels = get_num_index_attributes(index) == 2;

        let meta = MetaPage {
            magic_number: TSV_MAGIC_NUMBER,
            version: TSV_VERSION,
            extension_version_when_built: version.to_string(),
            distance_type: distance_type as u16,
            num_dimensions,
            num_dimensions_to_index,
            storage_type: (*opt).get_storage_type() as u8,
            num_neighbors: Self::calculate_num_neighbors(&opt),
            bq_num_bits_per_dimension,
            search_list_size: opt.search_list_size,
            max_alpha: opt.max_alpha,
            start_nodes: None,
            quantizer_metadata: ItemPointer::new(InvalidBlockNumber, InvalidOffsetNumber),
            has_labels,
        };

        meta.store(index, true);
        meta
    }

    pub unsafe fn store(&self, index: &PgRelation, first_time: bool) {
        let header = MetaPageHeader {
            magic_number: self.magic_number,
            version: self.version,
        };

        assert!(header.magic_number == TSV_MAGIC_NUMBER);
        assert!(header.version == TSV_VERSION);

        let mut stats = WriteStats::new();
        let mut tape = if first_time {
            ChainTapeWriter::new(index, PageType::Meta, &mut stats)
        } else {
            ChainTapeWriter::reinit(index, PageType::Meta, &mut stats, META_BLOCK_NUMBER)
        };

        // Serialize the header
        let bytes = header.serialize_to_vec();
        let off = tape.write(&bytes);
        assert_eq!(off, ItemPointer::new(META_BLOCK_NUMBER, META_HEADER_OFFSET));

        // Serialize the meta
        let bytes = self.serialize_to_vec();
        let off = tape.write(&bytes);
        assert_eq!(off, ItemPointer::new(META_BLOCK_NUMBER, META_OFFSET));
    }

    unsafe fn load(index: &PgRelation) -> MetaPage {
        let mut stats = WriteStats::new();
        let mut tape = ChainItemReader::new(index, PageType::Meta, &mut stats);

        let mut buf: Vec<u8> = Vec::new();
        for item in tape.read(ItemPointer::new(META_BLOCK_NUMBER, META_OFFSET)) {
            buf.extend_from_slice(item.get_data_slice());
        }
        let result = rkyv::from_bytes::<MetaPage>(&buf).unwrap();
        result
    }

    /// Read the meta page for an index
    pub fn fetch(index: &PgRelation) -> MetaPage {
        unsafe {
            let page = page::ReadablePage::read(index, META_BLOCK_NUMBER);
            let page_type = page.get_type();
            match page_type {
                PageType::MetaV1 => {
                    let old_meta = MetaPageV1::page_get_meta(*page, *(*(page.get_buffer())));
                    let new_meta: MetaPage = (&*old_meta).into();

                    //release the page
                    std::mem::drop(page);

                    new_meta.store(index, false);
                    new_meta
                }
                PageType::MetaV2 => MetaPageV2::from_page(page).into(),
                PageType::Meta => Self::load(index),
                _ => pgrx::error!("Meta page is not of type Meta"),
            }
        }
    }

    pub fn set_quantizer_metadata_pointer(&mut self, quantizer_pointer: IndexPointer) {
        self.quantizer_metadata = quantizer_pointer;
    }
}
