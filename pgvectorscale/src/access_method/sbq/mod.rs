mod cache;
pub mod node;
pub mod quantize;
pub mod storage;
mod tests;

use super::{
    distance::distance_xor_optimized,
    graph::neighbor_store::GraphNeighborStore,
    labels::LabeledVector,
    stats::{StatsDistanceComparison, StatsNodeModify, StatsNodeRead, StatsNodeWrite},
    storage::NodeDistanceMeasure,
};

use quantize::SbqQuantizer;

use pgrx::PgRelation;
use rkyv::{Archive, Deserialize, Serialize};
use storage::SbqSpeedupStorage;

use super::meta_page::MetaPage;
use crate::access_method::node::{ReadableNode, WriteableNode};
use crate::util::{
    chain::{ChainItemReader, ChainTapeWriter},
    page::{PageType, ReadablePage},
    tape::Tape,
    IndexPointer, ItemPointer, ReadableBuffer, WritableBuffer,
};
use pgvectorscale_derive::{Readable, Writeable};

pub type SbqVectorElement = u64;
const BITS_STORE_TYPE_SIZE: usize = 64;

#[derive(Archive, Deserialize, Serialize, Readable, Writeable)]
#[archive(check_bytes)]
#[repr(C)]
pub struct SbqMeansV1 {
    count: u64,
    means: Vec<f32>,
    m2: Vec<f32>,
}

impl SbqMeansV1 {
    pub unsafe fn load<S: StatsNodeRead>(
        index: &PgRelation,
        mut quantizer: SbqQuantizer,
        qip: ItemPointer,
        stats: &mut S,
    ) -> SbqQuantizer {
        assert!(quantizer.use_mean);
        let bq = SbqMeansV1::read(index, qip, stats);
        let archived = bq.get_archived_node();

        quantizer.load(
            archived.count,
            archived.means.to_vec(),
            archived.m2.to_vec(),
        );
        quantizer
    }

    pub unsafe fn store<S: StatsNodeWrite>(
        index: &PgRelation,
        quantizer: &SbqQuantizer,
        stats: &mut S,
    ) -> ItemPointer {
        let mut tape = Tape::new(index, PageType::SbqMeans);
        let node = SbqMeansV1 {
            count: quantizer.count,
            means: quantizer.mean.to_vec(),
            m2: quantizer.m2.to_vec(),
        };
        let ptr = node.write(&mut tape, stats);
        tape.close();
        ptr
    }
}

#[derive(Archive, Deserialize, Serialize)]
#[archive(check_bytes)]
#[repr(C)]
pub struct SbqMeans {
    count: u64,
    means: Vec<f32>,
    m2: Vec<f32>,
}

impl SbqMeans {
    pub unsafe fn load<S: StatsNodeRead>(
        index: &PgRelation,
        meta_page: &MetaPage,
        stats: &mut S,
    ) -> SbqQuantizer {
        let mut quantizer = SbqQuantizer::new(meta_page);
        if !quantizer.use_mean {
            return quantizer;
        }
        let qip = meta_page
            .get_quantizer_metadata_pointer()
            .unwrap_or_else(|| pgrx::error!("No SBQ pointer found in meta page"));

        let page = ReadablePage::read(index, qip.block_number);
        let page_type = page.get_type();
        match page_type {
            PageType::SbqMeansV1 => SbqMeansV1::load(index, quantizer, qip, stats),
            PageType::SbqMeans => {
                let mut tape_reader = ChainItemReader::new(index, PageType::SbqMeans, stats);
                let mut buf: Vec<u8> = Vec::new();
                for item in tape_reader.read(qip) {
                    buf.extend_from_slice(item.get_data_slice());
                }

                let means = rkyv::from_bytes::<SbqMeans>(buf.as_slice()).unwrap();
                quantizer.load(means.count, means.means, means.m2);
                quantizer
            }
            _ => {
                pgrx::error!("Invalid page type {} for SbqMeans", page_type as u8);
            }
        }
    }

    pub unsafe fn store<S: StatsNodeWrite>(
        index: &PgRelation,
        quantizer: &SbqQuantizer,
        stats: &mut S,
    ) -> ItemPointer {
        let bq = SbqMeans {
            count: quantizer.count,
            means: quantizer.mean.clone(),
            m2: quantizer.m2.clone(),
        };
        let mut tape = ChainTapeWriter::new(index, PageType::SbqMeans, stats);
        let buf = rkyv::to_bytes::<_, 1024>(&bq).unwrap();
        tape.write(&buf)
    }
}

pub struct SbqSearchDistanceMeasure {
    vec: Vec<SbqVectorElement>,
    query: LabeledVector,
}

impl SbqSearchDistanceMeasure {
    pub fn new(quantizer: &SbqQuantizer, query: LabeledVector) -> Self {
        let vec = quantizer.quantize(query.vec().to_index_slice());
        Self { vec, query }
    }

    pub fn calculate_bq_distance<S: StatsDistanceComparison>(
        &self,
        bq_vector: &[SbqVectorElement],
        _gns: &GraphNeighborStore,
        stats: &mut S,
    ) -> f32 {
        stats.record_quantized_distance_comparison();
        distance_xor_optimized(bq_vector, &self.vec) as f32
    }
}

pub struct SbqNodeDistanceMeasure<'a> {
    vec: Vec<SbqVectorElement>,
    storage: &'a SbqSpeedupStorage<'a>,
}

impl<'a> SbqNodeDistanceMeasure<'a> {
    pub unsafe fn with_index_pointer<T: StatsNodeRead + StatsNodeWrite + StatsNodeModify>(
        storage: &'a SbqSpeedupStorage<'a>,
        index_pointer: IndexPointer,
        stats: &mut T,
    ) -> Self {
        let mut cache = storage.cache().borrow_mut();
        let vec = cache.get(index_pointer, storage, stats).to_vec();
        Self { vec, storage }
    }
}

impl NodeDistanceMeasure for SbqNodeDistanceMeasure<'_> {
    unsafe fn get_distance<
        T: StatsNodeRead + StatsDistanceComparison + StatsNodeWrite + StatsNodeModify,
    >(
        &self,
        index_pointer: IndexPointer,
        stats: &mut T,
    ) -> f32 {
        let mut cache = self.storage.cache().borrow_mut();
        let vec1 = cache.get(index_pointer, self.storage, stats);
        distance_xor_optimized(vec1, self.vec.as_slice()) as f32
    }
}
