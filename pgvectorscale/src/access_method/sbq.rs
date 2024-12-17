use super::{
    distance::{distance_xor_optimized, DistanceFn},
    graph::{ListSearchNeighbor, ListSearchResult},
    graph_neighbor_store::GraphNeighborStore,
    neighbor_with_distance::DistanceWithTieBreak,
    pg_vector::PgVector,
    stats::{
        GreedySearchStats, StatsDistanceComparison, StatsHeapNodeRead, StatsNodeModify,
        StatsNodeRead, StatsNodeWrite, WriteStats,
    },
    storage::{ArchivedData, NodeDistanceMeasure, Storage},
    storage_common::get_attribute_number_from_index,
};
use std::{cell::RefCell, collections::HashMap, iter::once, marker::PhantomData, pin::Pin};

use pgrx::{
    pg_sys::{InvalidBlockNumber, InvalidOffsetNumber, BLCKSZ},
    PgBox, PgRelation,
};
use rkyv::{vec::ArchivedVec, Archive, Deserialize, Serialize};

use crate::util::{
    chain::{ChainItemReader, ChainTapeWriter},
    page::{PageType, ReadablePage},
    table_slot::TableSlot,
    tape::Tape,
    ArchivedItemPointer, HeapPointer, IndexPointer, ItemPointer, ReadableBuffer,
};

use super::{meta_page::MetaPage, neighbor_with_distance::NeighborWithDistance};
use crate::util::WritableBuffer;

type SbqVectorElement = u64;
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
        meta_page: &super::meta_page::MetaPage,
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

#[derive(Clone)]
pub struct SbqQuantizer {
    pub use_mean: bool,
    training: bool,
    pub count: u64,
    pub mean: Vec<f32>,
    pub m2: Vec<f32>,
    pub num_bits_per_dimension: u8,
}

impl SbqQuantizer {
    fn new(meta_page: &super::meta_page::MetaPage) -> SbqQuantizer {
        Self {
            use_mean: true,
            training: false,
            count: 0,
            mean: vec![],
            m2: vec![],
            num_bits_per_dimension: meta_page.get_bq_num_bits_per_dimension(),
        }
    }

    fn load(&mut self, count: u64, mean: Vec<f32>, m2: Vec<f32>) {
        self.count = count;
        self.mean = mean;
        self.m2 = m2
    }

    fn quantized_size(&self, full_vector_size: usize) -> usize {
        Self::quantized_size_internal(full_vector_size, self.num_bits_per_dimension)
    }

    fn quantized_size_internal(full_vector_size: usize, num_bits_per_dimension: u8) -> usize {
        let num_bits = full_vector_size * num_bits_per_dimension as usize;

        if num_bits % BITS_STORE_TYPE_SIZE == 0 {
            num_bits / BITS_STORE_TYPE_SIZE
        } else {
            (num_bits / BITS_STORE_TYPE_SIZE) + 1
        }
    }

    fn quantized_size_bytes(num_dimensions: usize, num_bits_per_dimension: u8) -> usize {
        Self::quantized_size_internal(num_dimensions, num_bits_per_dimension)
            * std::mem::size_of::<SbqVectorElement>()
    }

    fn quantize(&self, full_vector: &[f32]) -> Vec<SbqVectorElement> {
        assert!(!self.training);
        if self.use_mean {
            let mut res_vector = vec![0; self.quantized_size(full_vector.len())];

            if self.num_bits_per_dimension == 1 {
                for (i, &v) in full_vector.iter().enumerate() {
                    if v > self.mean[i] {
                        res_vector[i / BITS_STORE_TYPE_SIZE] |= 1 << (i % BITS_STORE_TYPE_SIZE);
                    }
                }
            } else {
                for (i, &v) in full_vector.iter().enumerate() {
                    let mean = self.mean[i];
                    let variance = self.m2[i] / self.count as f32;
                    let std_dev = variance.sqrt();
                    let ranges = self.num_bits_per_dimension + 1;

                    let v_z_score = (v - mean) / std_dev;
                    let index = (v_z_score + 2.0) / (4.0 / ranges as f32); //we consider z scores between -2 and 2 and divide them into {ranges} ranges

                    let bit_position = i * self.num_bits_per_dimension as usize;
                    if index < 1.0 {
                        //all zeros
                    } else {
                        let count_ones =
                            (index.floor() as usize).min(self.num_bits_per_dimension as usize);
                        //fill in count_ones bits from the left
                        // ex count_ones=1: 100
                        // ex count_ones=2: 110
                        // ex count_ones=3: 111
                        for j in 0..count_ones {
                            res_vector[(bit_position + j) / BITS_STORE_TYPE_SIZE] |=
                                1 << ((bit_position + j) % BITS_STORE_TYPE_SIZE);
                        }
                    }
                }
            }
            res_vector
        } else {
            let mut res_vector = vec![0; self.quantized_size(full_vector.len())];

            for (i, &v) in full_vector.iter().enumerate() {
                if v > 0.0 {
                    res_vector[i / BITS_STORE_TYPE_SIZE] |= 1 << (i % BITS_STORE_TYPE_SIZE);
                }
            }

            res_vector
        }
    }

    fn start_training(&mut self, meta_page: &super::meta_page::MetaPage) {
        self.training = true;
        if self.use_mean {
            self.count = 0;
            self.mean = vec![0.0; meta_page.get_num_dimensions_to_index() as _];
            if self.num_bits_per_dimension > 1 {
                self.m2 = vec![0.0; meta_page.get_num_dimensions_to_index() as _];
            }
        }
    }

    fn add_sample(&mut self, sample: &[f32]) {
        if self.use_mean {
            self.count += 1;
            assert!(self.mean.len() == sample.len());

            if self.num_bits_per_dimension > 1 {
                assert!(self.m2.len() == sample.len());
                let delta: Vec<_> = self
                    .mean
                    .iter()
                    .zip(sample.iter())
                    .map(|(m, s)| s - *m)
                    .collect();

                self.mean
                    .iter_mut()
                    .zip(sample.iter())
                    .for_each(|(m, s)| *m += (s - *m) / self.count as f32);

                let delta2 = self.mean.iter().zip(sample.iter()).map(|(m, s)| s - *m);

                self.m2
                    .iter_mut()
                    .zip(delta.iter())
                    .zip(delta2)
                    .for_each(|((m2, d), d2)| *m2 += d * d2);
            } else {
                self.mean
                    .iter_mut()
                    .zip(sample.iter())
                    .for_each(|(m, s)| *m += (s - *m) / self.count as f32);
            }
        }
    }

    fn finish_training(&mut self) {
        self.training = false;
    }

    fn vector_for_new_node(
        &self,
        _meta_page: &super::meta_page::MetaPage,
        full_vector: &[f32],
    ) -> Vec<SbqVectorElement> {
        self.quantize(full_vector)
    }
}

pub struct SbqSearchDistanceMeasure {
    quantized_vector: Vec<SbqVectorElement>,
    query: PgVector,
    num_dimensions_for_neighbors: usize,
    quantized_dimensions: usize,
}

impl SbqSearchDistanceMeasure {
    pub fn new(
        quantizer: &SbqQuantizer,
        query: PgVector,
        num_dimensions_for_neighbors: usize,
    ) -> SbqSearchDistanceMeasure {
        SbqSearchDistanceMeasure {
            quantized_vector: quantizer.quantize(query.to_index_slice()),
            query,
            num_dimensions_for_neighbors,
            quantized_dimensions: quantizer.quantized_size(num_dimensions_for_neighbors),
        }
    }

    pub fn calculate_bq_distance<S: StatsDistanceComparison>(
        &self,
        bq_vector: &[SbqVectorElement],
        gns: &GraphNeighborStore,
        stats: &mut S,
    ) -> f32 {
        assert!(!bq_vector.is_empty());
        stats.record_quantized_distance_comparison();
        let (a, b) = match gns {
            GraphNeighborStore::Disk => {
                if self.num_dimensions_for_neighbors > 0 {
                    debug_assert!(self.quantized_vector.len() >= self.quantized_dimensions);
                    debug_assert!(bq_vector.len() >= self.quantized_dimensions);
                    (
                        &self.quantized_vector.as_slice()[..self.quantized_dimensions],
                        &bq_vector[..self.quantized_dimensions],
                    )
                } else {
                    debug_assert!(
                        self.quantized_vector.len() == bq_vector.len(),
                        "self.quantized_vector.len()={} bq_vector.len()={}",
                        self.quantized_vector.len(),
                        bq_vector.len()
                    );
                    (self.quantized_vector.as_slice(), bq_vector)
                }
            }
            GraphNeighborStore::Builder(_b) => {
                debug_assert!(
                    self.quantized_vector.len() == bq_vector.len(),
                    "self.quantized_vector.len()={} bq_vector.len()={}",
                    self.quantized_vector.len(),
                    bq_vector.len()
                );
                (self.quantized_vector.as_slice(), bq_vector)
            }
        };

        let count_ones = distance_xor_optimized(a, b);
        //dot product is LOWER the more xors that lead to 1 becaues that means a negative times a positive = negative component
        //but the distance is 1 - dot product, so the more count_ones the higher the distance.
        // one other check for distance(a,a), xor=0, count_ones=0, distance=0
        count_ones as f32
    }
}

pub struct SbqNodeDistanceMeasure<'a> {
    vec: Vec<SbqVectorElement>,
    storage: &'a SbqSpeedupStorage<'a>,
}

impl<'a> SbqNodeDistanceMeasure<'a> {
    pub unsafe fn with_index_pointer<T: StatsNodeRead>(
        storage: &'a SbqSpeedupStorage<'a>,
        index_pointer: IndexPointer,
        stats: &mut T,
    ) -> Self {
        let cache = &mut storage.qv_cache.borrow_mut();
        Self {
            vec: cache.get(index_pointer, storage, stats).to_vec(),
            storage,
        }
    }
}

impl<'a> NodeDistanceMeasure for SbqNodeDistanceMeasure<'a> {
    unsafe fn get_distance<T: StatsNodeRead + StatsDistanceComparison>(
        &self,
        index_pointer: IndexPointer,
        stats: &mut T,
    ) -> f32 {
        let cache = &mut self.storage.qv_cache.borrow_mut();
        let vec1 = cache.get(index_pointer, self.storage, stats);
        distance_xor_optimized(vec1, self.vec.as_slice()) as f32
    }
}

struct QuantizedVectorCache {
    quantized_vector_map: HashMap<ItemPointer, Vec<SbqVectorElement>>,
}

/* should be a LRU cache for quantized vector. For now cheat and never evict
    TODO: implement LRU cache
*/
impl QuantizedVectorCache {
    fn new(capacity: usize) -> Self {
        Self {
            quantized_vector_map: HashMap::with_capacity(capacity),
        }
    }

    fn get<S: StatsNodeRead>(
        &mut self,
        index_pointer: IndexPointer,
        storage: &SbqSpeedupStorage,
        stats: &mut S,
    ) -> &[SbqVectorElement] {
        self.quantized_vector_map
            .entry(index_pointer)
            .or_insert_with(|| {
                storage.get_quantized_vector_from_index_pointer(index_pointer, stats)
            })
    }

    fn must_get(&self, index_pointer: IndexPointer) -> &[SbqVectorElement] {
        self.quantized_vector_map.get(&index_pointer).unwrap()
    }

    /* Ensure that all these elements are in the cache. If the capacity isn't big enough throw an error.
    must_get must succeed on all the elements after this call prior to another get or preload call */

    fn preload<I: Iterator<Item = IndexPointer>, S: StatsNodeRead>(
        &mut self,
        index_pointers: I,
        storage: &SbqSpeedupStorage,
        stats: &mut S,
    ) {
        for index_pointer in index_pointers {
            self.get(index_pointer, storage, stats);
        }
    }
}

pub struct SbqSpeedupStorage<'a> {
    pub index: &'a PgRelation,
    pub distance_fn: DistanceFn,
    quantizer: SbqQuantizer,
    heap_rel: &'a PgRelation,
    heap_attr: pgrx::pg_sys::AttrNumber,
    qv_cache: RefCell<QuantizedVectorCache>,
    num_dimensions_for_neighbors: usize,
}

impl<'a> SbqSpeedupStorage<'a> {
    pub fn new_for_build(
        index: &'a PgRelation,
        heap_rel: &'a PgRelation,
        meta_page: &super::meta_page::MetaPage,
    ) -> SbqSpeedupStorage<'a> {
        Self {
            index,
            distance_fn: meta_page.get_distance_function(),
            quantizer: SbqQuantizer::new(meta_page),
            heap_rel,
            heap_attr: get_attribute_number_from_index(index),
            qv_cache: RefCell::new(QuantizedVectorCache::new(1000)),
            num_dimensions_for_neighbors: meta_page.get_num_dimensions_for_neighbors() as usize,
        }
    }

    fn load_quantizer<S: StatsNodeRead>(
        index_relation: &PgRelation,
        meta_page: &super::meta_page::MetaPage,
        stats: &mut S,
    ) -> SbqQuantizer {
        unsafe { SbqMeans::load(index_relation, meta_page, stats) }
    }

    pub fn load_for_insert<S: StatsNodeRead>(
        heap_rel: &'a PgRelation,
        index_relation: &'a PgRelation,
        meta_page: &super::meta_page::MetaPage,
        stats: &mut S,
    ) -> SbqSpeedupStorage<'a> {
        Self {
            index: index_relation,
            distance_fn: meta_page.get_distance_function(),
            quantizer: Self::load_quantizer(index_relation, meta_page, stats),
            heap_rel,
            heap_attr: get_attribute_number_from_index(index_relation),
            qv_cache: RefCell::new(QuantizedVectorCache::new(1000)),
            num_dimensions_for_neighbors: meta_page.get_num_dimensions_for_neighbors() as usize,
        }
    }

    pub fn load_for_search(
        index_relation: &'a PgRelation,
        heap_relation: &'a PgRelation,
        quantizer: &SbqQuantizer,
        meta_page: &super::meta_page::MetaPage,
    ) -> SbqSpeedupStorage<'a> {
        Self {
            index: index_relation,
            distance_fn: meta_page.get_distance_function(),
            //OPT: get rid of clone
            quantizer: quantizer.clone(),
            heap_rel: heap_relation,
            heap_attr: get_attribute_number_from_index(index_relation),
            qv_cache: RefCell::new(QuantizedVectorCache::new(1000)),
            num_dimensions_for_neighbors: meta_page.get_num_dimensions_for_neighbors() as usize,
        }
    }

    fn get_quantized_vector_from_index_pointer<S: StatsNodeRead>(
        &self,
        index_pointer: IndexPointer,
        stats: &mut S,
    ) -> Vec<SbqVectorElement> {
        let rn = unsafe { SbqNode::read(self.index, index_pointer, stats) };
        let node = rn.get_archived_node();
        node.bq_vector.as_slice().to_vec()
    }

    fn write_quantizer_metadata<S: StatsNodeWrite + StatsNodeModify>(&self, stats: &mut S) {
        if self.quantizer.use_mean {
            let index_pointer = unsafe { SbqMeans::store(self.index, &self.quantizer, stats) };
            super::meta_page::MetaPage::update_quantizer_metadata_pointer(
                self.index,
                index_pointer,
                stats,
            );
        }
    }

    fn visit_lsn_internal(
        &self,
        lsr: &mut ListSearchResult<
            <SbqSpeedupStorage<'a> as Storage>::QueryDistanceMeasure,
            <SbqSpeedupStorage<'a> as Storage>::LSNPrivateData,
        >,
        lsn_index_pointer: IndexPointer,
        gns: &GraphNeighborStore,
    ) {
        match gns {
            GraphNeighborStore::Disk => {
                let rn_visiting =
                    unsafe { SbqNode::read(self.index, lsn_index_pointer, &mut lsr.stats) };
                let node_visiting = rn_visiting.get_archived_node();
                //OPT: get neighbors from private data just like plain storage in the self.num_dimensions_for_neighbors == 0 case
                let neighbors = node_visiting.get_index_pointer_to_neighbors();

                for (i, &neighbor_index_pointer) in neighbors.iter().enumerate() {
                    if !lsr.prepare_insert(neighbor_index_pointer) {
                        continue;
                    }

                    let distance = if self.num_dimensions_for_neighbors > 0 {
                        let bq_vector = node_visiting.neighbor_vectors[i].as_slice();
                        lsr.sdm.as_ref().unwrap().calculate_bq_distance(
                            bq_vector,
                            gns,
                            &mut lsr.stats,
                        )
                    } else {
                        let rn_neighbor = unsafe {
                            SbqNode::read(self.index, neighbor_index_pointer, &mut lsr.stats)
                        };
                        let node_neighbor = rn_neighbor.get_archived_node();
                        let bq_vector = node_neighbor.bq_vector.as_slice();
                        lsr.sdm.as_ref().unwrap().calculate_bq_distance(
                            bq_vector,
                            gns,
                            &mut lsr.stats,
                        )
                    };

                    let lsn = ListSearchNeighbor::new(
                        neighbor_index_pointer,
                        lsr.create_distance_with_tie_break(distance, neighbor_index_pointer),
                        PhantomData::<bool>,
                    );

                    lsr.insert_neighbor(lsn);
                }
            }
            GraphNeighborStore::Builder(b) => {
                let neighbors = b.get_neighbors(lsn_index_pointer);
                for &neighbor_index_pointer in neighbors.iter() {
                    if !lsr.prepare_insert(neighbor_index_pointer) {
                        continue;
                    }
                    let mut cache = self.qv_cache.borrow_mut();
                    let bq_vector = cache.get(neighbor_index_pointer, self, &mut lsr.stats);
                    let distance = lsr.sdm.as_ref().unwrap().calculate_bq_distance(
                        bq_vector,
                        gns,
                        &mut lsr.stats,
                    );

                    let lsn = ListSearchNeighbor::new(
                        neighbor_index_pointer,
                        lsr.create_distance_with_tie_break(distance, neighbor_index_pointer),
                        PhantomData::<bool>,
                    );

                    lsr.insert_neighbor(lsn);
                }
            }
        }
    }
}

pub type SbqSpeedupStorageLsnPrivateData = PhantomData<bool>; //no data stored

impl<'a> Storage for SbqSpeedupStorage<'a> {
    type QueryDistanceMeasure = SbqSearchDistanceMeasure;
    type NodeDistanceMeasure<'b> = SbqNodeDistanceMeasure<'b> where Self: 'b;
    type ArchivedType = ArchivedSbqNode;
    type LSNPrivateData = SbqSpeedupStorageLsnPrivateData; //no data stored

    fn page_type() -> PageType {
        PageType::SbqNode
    }

    fn create_node<S: StatsNodeWrite>(
        &self,
        full_vector: &[f32],
        heap_pointer: HeapPointer,
        meta_page: &MetaPage,
        tape: &mut Tape,
        stats: &mut S,
    ) -> ItemPointer {
        let bq_vector = self.quantizer.vector_for_new_node(meta_page, full_vector);

        let node = SbqNode::with_meta(
            &self.quantizer,
            heap_pointer,
            meta_page,
            bq_vector.as_slice(),
        );

        let index_pointer: IndexPointer = node.write(tape, stats);
        index_pointer
    }

    fn start_training(&mut self, meta_page: &super::meta_page::MetaPage) {
        self.quantizer.start_training(meta_page);
    }

    fn add_sample(&mut self, sample: &[f32]) {
        self.quantizer.add_sample(sample);
    }

    fn finish_training(&mut self, stats: &mut WriteStats) {
        self.quantizer.finish_training();
        self.write_quantizer_metadata(stats);
    }

    fn finalize_node_at_end_of_build<S: StatsNodeRead + StatsNodeModify>(
        &mut self,
        meta: &MetaPage,
        index_pointer: IndexPointer,
        neighbors: &[NeighborWithDistance],
        stats: &mut S,
    ) {
        let mut cache = self.qv_cache.borrow_mut();
        /* It's important to preload cache with all the items since you can run into deadlocks
        if you try to fetch a quantized vector while holding the SbqNode::modify lock */
        let iter = neighbors
            .iter()
            .map(|n| n.get_index_pointer_to_neighbor())
            .chain(once(index_pointer));
        cache.preload(iter, self, stats);

        let mut node = unsafe { SbqNode::modify(self.index, index_pointer, stats) };
        let mut archived = node.get_archived_node();
        archived.as_mut().set_neighbors(neighbors, meta, &cache);

        node.commit();
    }

    unsafe fn get_node_distance_measure<'b, S: StatsNodeRead>(
        &'b self,
        index_pointer: IndexPointer,
        stats: &mut S,
    ) -> SbqNodeDistanceMeasure<'b> {
        SbqNodeDistanceMeasure::with_index_pointer(self, index_pointer, stats)
    }

    fn get_query_distance_measure(&self, query: PgVector) -> SbqSearchDistanceMeasure {
        SbqSearchDistanceMeasure::new(&self.quantizer, query, self.num_dimensions_for_neighbors)
    }

    fn get_full_distance_for_resort<S: StatsHeapNodeRead + StatsDistanceComparison>(
        &self,
        scan: &PgBox<pgrx::pg_sys::IndexScanDescData>,
        qdm: &Self::QueryDistanceMeasure,
        _index_pointer: IndexPointer,
        heap_pointer: HeapPointer,
        meta_page: &MetaPage,
        stats: &mut S,
    ) -> Option<f32> {
        let slot_opt = unsafe {
            TableSlot::from_index_heap_pointer(self.heap_rel, heap_pointer, scan.xs_snapshot, stats)
        };

        let slot = slot_opt?;

        let datum = unsafe {
            slot.get_attribute(self.heap_attr)
                .expect("vector attribute should exist in the heap")
        };
        let vec = unsafe { PgVector::from_datum(datum, meta_page, false, true) };
        Some(self.get_distance_function()(
            vec.to_full_slice(),
            qdm.query.to_full_slice(),
        ))
    }

    fn get_neighbors_with_distances_from_disk<S: StatsNodeRead + StatsDistanceComparison>(
        &self,
        neighbors_of: ItemPointer,
        result: &mut Vec<NeighborWithDistance>,
        stats: &mut S,
    ) {
        let rn = unsafe { SbqNode::read(self.index, neighbors_of, stats) };
        let archived = rn.get_archived_node();
        let q = archived.bq_vector.as_slice();

        for n in rn.get_archived_node().iter_neighbors() {
            //OPT: we can optimize this if num_dimensions_for_neighbors == num_dimensions_to_index
            let rn1 = unsafe { SbqNode::read(self.index, n, stats) };
            stats.record_quantized_distance_comparison();
            let dist = distance_xor_optimized(q, rn1.get_archived_node().bq_vector.as_slice());
            result.push(NeighborWithDistance::new(
                n,
                DistanceWithTieBreak::new(dist as f32, neighbors_of, n),
            ))
        }
    }

    /* get_lsn and visit_lsn are different because the distance
    comparisons for SBQ get the vector from different places */
    fn create_lsn_for_init_id(
        &self,
        lsr: &mut ListSearchResult<Self::QueryDistanceMeasure, Self::LSNPrivateData>,
        index_pointer: ItemPointer,
        gns: &GraphNeighborStore,
    ) -> ListSearchNeighbor<Self::LSNPrivateData> {
        if !lsr.prepare_insert(index_pointer) {
            panic!("should not have had an init id already inserted");
        }

        let rn = unsafe { SbqNode::read(self.index, index_pointer, &mut lsr.stats) };
        let node = rn.get_archived_node();

        let distance = lsr.sdm.as_ref().unwrap().calculate_bq_distance(
            node.bq_vector.as_slice(),
            gns,
            &mut lsr.stats,
        );

        ListSearchNeighbor::new(
            index_pointer,
            lsr.create_distance_with_tie_break(distance, index_pointer),
            PhantomData::<bool>,
        )
    }

    fn visit_lsn(
        &self,
        lsr: &mut ListSearchResult<Self::QueryDistanceMeasure, Self::LSNPrivateData>,
        lsn_idx: usize,
        gns: &GraphNeighborStore,
    ) {
        let lsn_index_pointer = lsr.get_lsn_by_idx(lsn_idx).index_pointer;
        self.visit_lsn_internal(lsr, lsn_index_pointer, gns);
    }

    fn return_lsn(
        &self,
        lsn: &ListSearchNeighbor<Self::LSNPrivateData>,
        stats: &mut GreedySearchStats,
    ) -> HeapPointer {
        let lsn_index_pointer = lsn.index_pointer;
        let rn = unsafe { SbqNode::read(self.index, lsn_index_pointer, stats) };
        let node = rn.get_archived_node();

        node.heap_item_pointer.deserialize_item_pointer()
    }

    fn set_neighbors_on_disk<S: StatsNodeModify + StatsNodeRead>(
        &self,
        meta: &MetaPage,
        index_pointer: IndexPointer,
        neighbors: &[NeighborWithDistance],
        stats: &mut S,
    ) {
        let mut cache = QuantizedVectorCache::new(neighbors.len() + 1);

        /* It's important to preload cache with all the items since you can run into deadlocks
        if you try to fetch a quantized vector while holding the SbqNode::modify lock */
        let iter = neighbors
            .iter()
            .map(|n| n.get_index_pointer_to_neighbor())
            .chain(once(index_pointer));
        cache.preload(iter, self, stats);

        let mut node = unsafe { SbqNode::modify(self.index, index_pointer, stats) };
        let mut archived = node.get_archived_node();
        archived.as_mut().set_neighbors(neighbors, meta, &cache);
        node.commit();
    }

    fn get_distance_function(&self) -> DistanceFn {
        self.distance_fn
    }
}

use pgvectorscale_derive::{Readable, Writeable};

#[derive(Archive, Deserialize, Serialize, Readable, Writeable)]
#[archive(check_bytes)]
pub struct SbqNode {
    pub heap_item_pointer: HeapPointer,
    pub bq_vector: Vec<u64>, //don't use SbqVectorElement because we don't want to change the size in on-disk format by accident
    neighbor_index_pointers: Vec<ItemPointer>,
    neighbor_vectors: Vec<Vec<u64>>, //don't use SbqVectorElement because we don't want to change the size in on-disk format by accident
}

impl SbqNode {
    pub fn with_meta(
        quantizer: &SbqQuantizer,
        heap_pointer: HeapPointer,
        meta_page: &MetaPage,
        bq_vector: &[SbqVectorElement],
    ) -> Self {
        Self::new(
            heap_pointer,
            meta_page.get_num_neighbors() as usize,
            meta_page.get_num_dimensions_to_index() as usize,
            meta_page.get_num_dimensions_for_neighbors() as usize,
            quantizer.num_bits_per_dimension,
            bq_vector,
        )
    }

    fn new(
        heap_pointer: HeapPointer,
        num_neighbors: usize,
        _num_dimensions: usize,
        num_dimensions_for_neighbors: usize,
        num_bits_per_dimension: u8,
        bq_vector: &[SbqVectorElement],
    ) -> Self {
        // always use vectors of num_neighbors in length because we never want the serialized size of a Node to change
        let neighbor_index_pointers: Vec<_> = (0..num_neighbors)
            .map(|_| ItemPointer::new(InvalidBlockNumber, InvalidOffsetNumber))
            .collect();

        let neighbor_vectors: Vec<_> = if num_dimensions_for_neighbors > 0 {
            (0..num_neighbors)
                .map(|_| {
                    vec![
                        0;
                        SbqQuantizer::quantized_size_internal(
                            num_dimensions_for_neighbors as _,
                            num_bits_per_dimension
                        )
                    ]
                })
                .collect()
        } else {
            vec![]
        };

        Self {
            heap_item_pointer: heap_pointer,
            bq_vector: bq_vector.to_vec(),
            neighbor_index_pointers,
            neighbor_vectors,
        }
    }

    fn test_size(
        num_neighbors: usize,
        num_dimensions: usize,
        num_dimensions_for_neighbors: usize,
        num_bits_per_dimension: u8,
    ) -> usize {
        let v: Vec<SbqVectorElement> =
            vec![0; SbqQuantizer::quantized_size_internal(num_dimensions, num_bits_per_dimension)];
        let hp = HeapPointer::new(InvalidBlockNumber, InvalidOffsetNumber);
        let n = Self::new(
            hp,
            num_neighbors,
            num_dimensions,
            num_dimensions_for_neighbors,
            num_bits_per_dimension,
            &v,
        );
        n.serialize_to_vec().len()
    }

    pub fn get_default_num_neighbors(
        num_dimensions: usize,
        num_dimensions_for_neighbors: usize,
        num_bits_per_dimension: u8,
    ) -> usize {
        //how many neighbors can fit on one page? That's what we choose.

        //we first overapproximate the number of neighbors and then double check by actually calculating the size of the SbqNode.

        //blocksize - 100 bytes for the padding/header/etc.
        let page_size = BLCKSZ as usize - 50;
        //one quantized_vector takes this many bytes
        let vec_size =
            SbqQuantizer::quantized_size_bytes(num_dimensions, num_bits_per_dimension) + 1;
        //start from the page size then subtract the heap_item_pointer and bq_vector elements of SbqNode.
        let starting = BLCKSZ as usize - std::mem::size_of::<HeapPointer>() - vec_size;
        //one neigbors contribution to neighbor_index_pointers + neighbor_vectors in SbqNode.
        let one_neighbor = vec_size + std::mem::size_of::<ItemPointer>();

        let mut num_neighbors_overapproximate: usize = starting / one_neighbor;
        while num_neighbors_overapproximate > 0 {
            let serialized_size = SbqNode::test_size(
                num_neighbors_overapproximate,
                num_dimensions,
                num_dimensions_for_neighbors,
                num_bits_per_dimension,
            );
            if serialized_size <= page_size {
                return num_neighbors_overapproximate;
            }
            num_neighbors_overapproximate -= 1;
        }
        pgrx::error!(
            "Could not find a valid number of neighbors for the default value. Please specify one."
        );
    }
}

impl ArchivedSbqNode {
    fn neighbor_index_pointer(self: Pin<&mut Self>) -> Pin<&mut ArchivedVec<ArchivedItemPointer>> {
        unsafe { self.map_unchecked_mut(|s| &mut s.neighbor_index_pointers) }
    }

    fn neighbor_vector(self: Pin<&mut Self>) -> Pin<&mut ArchivedVec<ArchivedVec<u64>>> {
        unsafe { self.map_unchecked_mut(|s| &mut s.neighbor_vectors) }
    }

    fn set_neighbors(
        mut self: Pin<&mut Self>,
        neighbors: &[NeighborWithDistance],
        meta_page: &MetaPage,
        cache: &QuantizedVectorCache,
    ) {
        for (i, new_neighbor) in neighbors.iter().enumerate() {
            let mut a_index_pointer = self.as_mut().neighbor_index_pointer().index_pin(i);
            let ip = new_neighbor.get_index_pointer_to_neighbor();
            //TODO hate that we have to set each field like this
            a_index_pointer.block_number = ip.block_number;
            a_index_pointer.offset = ip.offset;

            if meta_page.get_num_dimensions_for_neighbors() > 0 {
                let quantized = &cache.must_get(ip)[..SbqQuantizer::quantized_size_internal(
                    meta_page.get_num_dimensions_for_neighbors() as _,
                    meta_page.get_bq_num_bits_per_dimension(),
                )];
                let mut neighbor_vector = self.as_mut().neighbor_vector().index_pin(i);
                for (index_in_q_vec, val) in quantized.iter().enumerate() {
                    let mut x = neighbor_vector.as_mut().index_pin(index_in_q_vec);
                    *x = *val;
                }
            }
        }
        //set the marker that the list ended
        if neighbors.len() < meta_page.get_num_neighbors() as _ {
            let mut past_last_index_pointers =
                self.neighbor_index_pointer().index_pin(neighbors.len());
            past_last_index_pointers.block_number = InvalidBlockNumber;
            past_last_index_pointers.offset = InvalidOffsetNumber;
        }
    }

    pub fn num_neighbors(&self) -> usize {
        self.neighbor_index_pointers
            .iter()
            .position(|f| f.block_number == InvalidBlockNumber)
            .unwrap_or(self.neighbor_index_pointers.len())
    }

    pub fn iter_neighbors(&self) -> impl Iterator<Item = ItemPointer> + '_ {
        self.neighbor_index_pointers
            .iter()
            .take(self.num_neighbors())
            .map(|ip| ip.deserialize_item_pointer())
    }
}

impl ArchivedData for ArchivedSbqNode {
    fn with_data(data: &mut [u8]) -> Pin<&mut ArchivedSbqNode> {
        ArchivedSbqNode::with_data(data)
    }

    fn get_index_pointer_to_neighbors(&self) -> Vec<ItemPointer> {
        self.iter_neighbors().collect()
    }

    fn is_deleted(&self) -> bool {
        self.heap_item_pointer.offset == InvalidOffsetNumber
    }

    fn delete(self: Pin<&mut Self>) {
        //TODO: actually optimize the deletes by removing index tuples. For now just mark it.
        let mut heap_pointer = unsafe { self.map_unchecked_mut(|s| &mut s.heap_item_pointer) };
        heap_pointer.offset = InvalidOffsetNumber;
        heap_pointer.block_number = InvalidBlockNumber;
    }

    fn get_heap_item_pointer(&self) -> HeapPointer {
        self.heap_item_pointer.deserialize_item_pointer()
    }
}

#[cfg(any(test, feature = "pg_test"))]
#[pgrx::pg_schema]
mod tests {
    use pgrx::*;

    use crate::access_method::distance::DistanceType;

    #[pg_test]
    unsafe fn test_bq_speedup_storage_index_creation_default_neighbors() -> spi::Result<()> {
        crate::access_method::build::tests::test_index_creation_and_accuracy_scaffold(
            DistanceType::Cosine,
            "storage_layout = io_optimized",
            "bq_speedup_default_neighbors",
            1536,
        )?;
        Ok(())
    }

    #[pg_test]
    unsafe fn test_bq_speedup_storage_index_creation_few_neighbors() -> spi::Result<()> {
        //a test with few neighbors tests the case that nodes share a page, which has caused deadlocks in the past.
        crate::access_method::build::tests::test_index_creation_and_accuracy_scaffold(
            DistanceType::Cosine,
            "num_neighbors=10, storage_layout = io_optimized",
            "bq_speedup_few_neighbors",
            1536,
        )?;
        Ok(())
    }

    #[test]
    fn test_bq_speedup_storage_delete_vacuum_plain() {
        crate::access_method::vacuum::tests::test_delete_vacuum_plain_scaffold(
            "num_neighbors = 10, storage_layout = io_optimized",
        );
    }

    #[test]
    fn test_bq_speedup_storage_delete_vacuum_full() {
        crate::access_method::vacuum::tests::test_delete_vacuum_full_scaffold(
            "num_neighbors = 38, storage_layout = io_optimized",
        );
    }

    #[pg_test]
    unsafe fn test_bq_speedup_storage_empty_table_insert() -> spi::Result<()> {
        crate::access_method::build::tests::test_empty_table_insert_scaffold(
            "num_neighbors=38, storage_layout = io_optimized",
        )
    }

    #[pg_test]
    unsafe fn test_bq_speedup_storage_insert_empty_insert() -> spi::Result<()> {
        crate::access_method::build::tests::test_insert_empty_insert_scaffold(
            "num_neighbors=38, storage_layout = io_optimized",
        )
    }

    #[pg_test]
    unsafe fn test_bq_speedup_storage_index_creation_num_dimensions_cosine() -> spi::Result<()> {
        crate::access_method::build::tests::test_index_creation_and_accuracy_scaffold(
            DistanceType::Cosine,
            "storage_layout = io_optimized, num_dimensions=768",
            "bq_speedup_num_dimensions",
            3072,
        )?;
        Ok(())
    }

    #[pg_test]
    unsafe fn test_bq_speedup_storage_index_creation_num_dimensions_l2() -> spi::Result<()> {
        crate::access_method::build::tests::test_index_creation_and_accuracy_scaffold(
            DistanceType::L2,
            "storage_layout = io_optimized, num_dimensions=768",
            "bq_speedup_num_dimensions",
            3072,
        )?;
        Ok(())
    }

    #[pg_test]
    unsafe fn test_bq_speedup_storage_index_creation_num_dimensions_ip() -> spi::Result<()> {
        crate::access_method::build::tests::test_index_creation_and_accuracy_scaffold(
            DistanceType::InnerProduct,
            "storage_layout = io_optimized, num_dimensions=768",
            "bq_speedup_num_dimensions",
            3072,
        )?;
        Ok(())
    }

    #[pg_test]
    unsafe fn test_bq_speedup_storage_index_updates_cosine() -> spi::Result<()> {
        crate::access_method::build::tests::test_index_updates(
            DistanceType::Cosine,
            "storage_layout = io_optimized, num_neighbors=10",
            300,
            "bq_speedup",
        )?;
        Ok(())
    }

    #[pg_test]
    unsafe fn test_bq_speedup_storage_index_updates_l2() -> spi::Result<()> {
        crate::access_method::build::tests::test_index_updates(
            DistanceType::L2,
            "storage_layout = io_optimized, num_neighbors=10",
            300,
            "bq_speedup",
        )?;
        Ok(())
    }

    #[pg_test]
    unsafe fn test_bq_speedup_storage_index_updates_ip() -> spi::Result<()> {
        crate::access_method::build::tests::test_index_updates(
            DistanceType::InnerProduct,
            "storage_layout = io_optimized, num_neighbors=10",
            300,
            "bq_speedup",
        )?;
        Ok(())
    }

    #[pg_test]
    unsafe fn test_bq_compressed_index_creation_default_neighbors() -> spi::Result<()> {
        crate::access_method::build::tests::test_index_creation_and_accuracy_scaffold(
            DistanceType::Cosine,
            "storage_layout = memory_optimized",
            "bq_compressed_default_neighbors",
            1536,
        )?;
        Ok(())
    }

    #[pg_test]
    unsafe fn test_bq_compressed_storage_index_creation_few_neighbors() -> spi::Result<()> {
        //a test with few neighbors tests the case that nodes share a page, which has caused deadlocks in the past.
        crate::access_method::build::tests::test_index_creation_and_accuracy_scaffold(
            DistanceType::Cosine,
            "num_neighbors=10, storage_layout = memory_optimized",
            "bq_compressed_few_neighbors",
            1536,
        )?;
        Ok(())
    }

    #[test]
    fn test_bq_compressed_storage_delete_vacuum_plain() {
        crate::access_method::vacuum::tests::test_delete_vacuum_plain_scaffold(
            "num_neighbors = 10, storage_layout = memory_optimized",
        );
    }

    #[test]
    fn test_bq_compressed_storage_delete_vacuum_full() {
        crate::access_method::vacuum::tests::test_delete_vacuum_full_scaffold(
            "num_neighbors = 38, storage_layout = memory_optimized",
        );
    }

    #[test]
    fn test_bq_compressed_storage_update_with_null() {
        crate::access_method::vacuum::tests::test_update_with_null_scaffold(
            "num_neighbors = 38, storage_layout = memory_optimized",
        );
    }
    #[pg_test]
    unsafe fn test_bq_compressed_storage_empty_table_insert() -> spi::Result<()> {
        crate::access_method::build::tests::test_empty_table_insert_scaffold(
            "num_neighbors=38, storage_layout = memory_optimized",
        )
    }

    #[pg_test]
    unsafe fn test_bq_compressed_storage_insert_empty_insert() -> spi::Result<()> {
        crate::access_method::build::tests::test_insert_empty_insert_scaffold(
            "num_neighbors=38, storage_layout = memory_optimized",
        )
    }

    #[pg_test]
    unsafe fn test_bq_compressed_storage_index_creation_num_dimensions() -> spi::Result<()> {
        crate::access_method::build::tests::test_index_creation_and_accuracy_scaffold(
            DistanceType::Cosine,
            "storage_layout = memory_optimized, num_dimensions=768",
            "bq_compressed_num_dimensions",
            3072,
        )?;
        Ok(())
    }

    #[pg_test]
    unsafe fn test_bq_compressed_storage_index_updates_cosine() -> spi::Result<()> {
        crate::access_method::build::tests::test_index_updates(
            DistanceType::Cosine,
            "storage_layout = memory_optimized, num_neighbors=10",
            300,
            "bq_compressed",
        )?;
        Ok(())
    }

    #[pg_test]
    unsafe fn test_bq_compressed_storage_index_updates_l2() -> spi::Result<()> {
        crate::access_method::build::tests::test_index_updates(
            DistanceType::L2,
            "storage_layout = memory_optimized, num_neighbors=10",
            300,
            "bq_compressed",
        )?;
        Ok(())
    }

    #[pg_test]
    unsafe fn test_bq_compressed_storage_index_updates_ip() -> spi::Result<()> {
        crate::access_method::build::tests::test_index_updates(
            DistanceType::InnerProduct,
            "storage_layout = memory_optimized, num_neighbors=10",
            300,
            "bq_compressed",
        )?;
        Ok(())
    }
}
