use super::{
    distance::{distance_xor_optimized, DistanceFn},
    graph::{ListSearchNeighbor, ListSearchResult},
    graph_neighbor_store::GraphNeighborStore,
    labels::{LabelSet, LabelSetView, LabeledVector},
    neighbor_with_distance::DistanceWithTieBreak,
    pg_vector::PgVector,
    sbq_node::{ArchivedSbqNode, SbqNode},
    stats::{
        GreedySearchStats, StatsDistanceComparison, StatsHeapNodeRead, StatsNodeModify,
        StatsNodeRead, StatsNodeWrite, WriteStats,
    },
    storage::{NodeDistanceMeasure, Storage},
    storage_common::get_index_vector_attribute,
};
use std::{cell::RefCell, collections::HashMap, iter::once, marker::PhantomData};

use pgrx::{pg_sys::AttrNumber, PgBox, PgRelation};
use rkyv::{Archive, Deserialize, Serialize};

use super::{meta_page::MetaPage, neighbor_with_distance::NeighborWithDistance};
use crate::access_method::node::{ReadableNode, WriteableNode};
use crate::util::{
    chain::{ChainItemReader, ChainTapeWriter},
    page::{PageType, ReadablePage},
    table_slot::TableSlot,
    tape::Tape,
    HeapPointer, IndexPointer, ItemPointer, ReadableBuffer, WritableBuffer,
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
    fn new(meta_page: &MetaPage) -> SbqQuantizer {
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

    pub fn quantized_size(&self, full_vector_size: usize) -> usize {
        Self::quantized_size_internal(full_vector_size, self.num_bits_per_dimension)
    }

    pub fn quantized_size_internal(full_vector_size: usize, num_bits_per_dimension: u8) -> usize {
        let num_bits = full_vector_size * num_bits_per_dimension as usize;

        if num_bits % BITS_STORE_TYPE_SIZE == 0 {
            num_bits / BITS_STORE_TYPE_SIZE
        } else {
            (num_bits / BITS_STORE_TYPE_SIZE) + 1
        }
    }

    pub fn quantized_size_bytes(num_dimensions: usize, num_bits_per_dimension: u8) -> usize {
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

    fn start_training(&mut self, meta_page: &MetaPage) {
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
        _meta_page: &MetaPage,
        full_vector: &[f32],
    ) -> Vec<SbqVectorElement> {
        self.quantize(full_vector)
    }
}

pub struct SbqSearchDistanceMeasure {
    quantized_vector: Vec<SbqVectorElement>,
    query: LabeledVector,
}

impl SbqSearchDistanceMeasure {
    pub fn new(quantizer: &SbqQuantizer, query: LabeledVector) -> SbqSearchDistanceMeasure {
        SbqSearchDistanceMeasure {
            quantized_vector: quantizer.quantize(query.vec().to_index_slice()),
            query,
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
                debug_assert!(
                    self.quantized_vector.len() == bq_vector.len(),
                    "self.quantized_vector.len()={} bq_vector.len()={}",
                    self.quantized_vector.len(),
                    bq_vector.len()
                );
                (self.quantized_vector.as_slice(), bq_vector)
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
        let vec = cache.get(index_pointer, storage, stats);
        Self {
            vec: vec.to_vec(),
            storage,
        }
    }
}

impl NodeDistanceMeasure for SbqNodeDistanceMeasure<'_> {
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
        let vec = self
            .quantized_vector_map
            .entry(index_pointer)
            .or_insert_with(|| {
                storage.get_quantized_vector_from_index_pointer(index_pointer, stats)
            });
        vec.as_slice()
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
    heap_attr: AttrNumber,
    qv_cache: RefCell<QuantizedVectorCache>,
    has_labels: bool,
}

impl<'a> SbqSpeedupStorage<'a> {
    pub fn new_for_build(
        index: &'a PgRelation,
        heap_rel: &'a PgRelation,
        meta_page: &MetaPage,
    ) -> SbqSpeedupStorage<'a> {
        Self {
            index,
            distance_fn: meta_page.get_distance_function(),
            quantizer: SbqQuantizer::new(meta_page),
            heap_rel,
            heap_attr: get_index_vector_attribute(index),
            qv_cache: RefCell::new(QuantizedVectorCache::new(1000)),
            has_labels: meta_page.has_labels(),
        }
    }

    fn load_quantizer<S: StatsNodeRead>(
        index_relation: &PgRelation,
        meta_page: &MetaPage,
        stats: &mut S,
    ) -> SbqQuantizer {
        unsafe { SbqMeans::load(index_relation, meta_page, stats) }
    }

    pub fn load_for_insert<S: StatsNodeRead>(
        heap_rel: &'a PgRelation,
        index_relation: &'a PgRelation,
        meta_page: &MetaPage,
        stats: &mut S,
    ) -> SbqSpeedupStorage<'a> {
        Self {
            index: index_relation,
            distance_fn: meta_page.get_distance_function(),
            quantizer: Self::load_quantizer(index_relation, meta_page, stats),
            heap_rel,
            heap_attr: get_index_vector_attribute(index_relation),
            qv_cache: RefCell::new(QuantizedVectorCache::new(1000)),
            has_labels: meta_page.has_labels(),
        }
    }

    pub fn load_for_search(
        index_relation: &'a PgRelation,
        heap_relation: &'a PgRelation,
        quantizer: &SbqQuantizer,
        meta_page: &MetaPage,
    ) -> SbqSpeedupStorage<'a> {
        Self {
            index: index_relation,
            distance_fn: meta_page.get_distance_function(),
            //OPT: get rid of clone
            quantizer: quantizer.clone(),
            heap_rel: heap_relation,
            heap_attr: get_index_vector_attribute(index_relation),
            qv_cache: RefCell::new(QuantizedVectorCache::new(1000)),
            has_labels: meta_page.has_labels(),
        }
    }

    fn write_quantizer_metadata<S: StatsNodeWrite + StatsNodeModify>(
        &self,
        meta_page: &mut MetaPage,
        stats: &mut S,
    ) {
        if self.quantizer.use_mean {
            let index_pointer = unsafe { SbqMeans::store(self.index, &self.quantizer, stats) };
            meta_page.set_quantizer_metadata_pointer(index_pointer);
        }
    }

    fn get_quantized_vector_from_index_pointer<S: StatsNodeRead>(
        &self,
        index_pointer: IndexPointer,
        stats: &mut S,
    ) -> Vec<SbqVectorElement> {
        let rn = unsafe { SbqNode::read(self.index, index_pointer, self.has_labels, stats) };
        let node = rn.get_archived_node();
        node.get_bq_vector().to_vec()
    }

    fn visit_lsn_internal(
        &self,
        lsr: &mut ListSearchResult<
            <SbqSpeedupStorage<'a> as Storage>::QueryDistanceMeasure,
            <SbqSpeedupStorage<'a> as Storage>::LSNPrivateData,
        >,
        lsn_index_pointer: IndexPointer,
        gns: &GraphNeighborStore,
        no_filter: bool,
    ) {
        match gns {
            GraphNeighborStore::Disk => {
                let rn_visiting = unsafe {
                    SbqNode::read(
                        self.index,
                        lsn_index_pointer,
                        self.has_labels,
                        &mut lsr.stats,
                    )
                };
                let node_visiting = rn_visiting.get_archived_node();
                //OPT: get neighbors from private data just like plain storage in the self.num_dimensions_for_neighbors == 0 case
                let neighbors = node_visiting.get_index_pointer_to_neighbors();

                for &neighbor_index_pointer in neighbors.iter() {
                    if !lsr.prepare_insert(neighbor_index_pointer) {
                        continue;
                    }

                    let rn_neighbor = unsafe {
                        SbqNode::read(
                            self.index,
                            neighbor_index_pointer,
                            self.has_labels,
                            &mut lsr.stats,
                        )
                    };

                    let node_neighbor = rn_neighbor.get_archived_node();

                    // Skip neighbors that have no matching labels with the query
                    if let Some(labels) = lsr.sdm.as_ref().expect("sdm is Some").query.labels() {
                        if !no_filter
                            && !labels
                                .overlaps(node_neighbor.get_labels().expect("Unlabeled neighbor?"))
                        {
                            continue;
                        }
                    }

                    let bq_vector = node_neighbor.get_bq_vector();
                    let distance = lsr
                        .sdm
                        .as_ref()
                        .expect("sdm is Some")
                        .calculate_bq_distance(bq_vector, gns, &mut lsr.stats);

                    let lsn = ListSearchNeighbor::new(
                        neighbor_index_pointer,
                        lsr.create_distance_with_tie_break(distance, neighbor_index_pointer),
                        PhantomData::<bool>,
                        node_neighbor.get_labels().map(Into::into),
                    );

                    lsr.insert_neighbor(lsn);
                }
            }
            GraphNeighborStore::Builder(b) => {
                let mut neighbors: Vec<NeighborWithDistance> = Vec::new();
                b.get_neighbors_with_full_vector_distances(lsn_index_pointer, &mut neighbors);
                for neighbor in neighbors.iter() {
                    let neighbor_index_pointer = neighbor.get_index_pointer_to_neighbor();
                    if !lsr.prepare_insert(neighbor_index_pointer) {
                        continue;
                    }

                    // Skip neighbors that have no matching labels with the query
                    if let Some(labels) = lsr.sdm.as_ref().expect("lsr.sdm is None").query.labels()
                    {
                        if !no_filter && !labels.overlaps(neighbor.get_labels().unwrap()) {
                            continue;
                        }
                    }

                    let mut cache = self.qv_cache.borrow_mut();
                    let bq_vector = cache.get(neighbor_index_pointer, self, &mut lsr.stats);
                    let distance = lsr
                        .sdm
                        .as_ref()
                        .expect("lsr.sdm is None")
                        .calculate_bq_distance(bq_vector, gns, &mut lsr.stats);

                    let lsn = ListSearchNeighbor::new(
                        neighbor_index_pointer,
                        lsr.create_distance_with_tie_break(distance, neighbor_index_pointer),
                        PhantomData::<bool>,
                        neighbor.get_labels().cloned(),
                    );

                    lsr.insert_neighbor(lsn);
                }
            }
        }
    }
}

impl Storage for SbqSpeedupStorage<'_> {
    type QueryDistanceMeasure = SbqSearchDistanceMeasure;
    type NodeDistanceMeasure<'a>
        = SbqNodeDistanceMeasure<'a>
    where
        Self: 'a;
    type ArchivedType<'b>
        = ArchivedSbqNode<'b>
    where
        Self: 'b;
    type LSNPrivateData = SbqSpeedupStorageLsnPrivateData; //no data stored

    fn page_type() -> PageType {
        PageType::SbqNode
    }

    fn create_node<S: StatsNodeWrite>(
        &self,
        full_vector: &[f32],
        labels: Option<LabelSet>,
        heap_pointer: HeapPointer,
        meta_page: &MetaPage,
        tape: &mut Tape,
        stats: &mut S,
    ) -> ItemPointer {
        let bq_vector = self.quantizer.vector_for_new_node(meta_page, full_vector);

        let node = SbqNode::with_meta(heap_pointer, meta_page, bq_vector.as_slice(), labels);

        let index_pointer: IndexPointer = node.write(tape, stats);
        index_pointer
    }

    fn start_training(&mut self, meta_page: &MetaPage) {
        self.quantizer.start_training(meta_page);
    }

    fn add_sample(&mut self, sample: &[f32]) {
        self.quantizer.add_sample(sample);
    }

    fn finish_training(&mut self, meta_page: &mut MetaPage, stats: &mut WriteStats) {
        self.quantizer.finish_training();
        self.write_quantizer_metadata(meta_page, stats);
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

        let mut node =
            unsafe { SbqNode::modify(self.index, index_pointer, self.has_labels, stats) };
        let mut archived = node.get_archived_node();
        archived.set_neighbors(neighbors, meta);

        node.commit();
    }

    unsafe fn get_node_distance_measure<'b, S: StatsNodeRead>(
        &'b self,
        index_pointer: IndexPointer,
        stats: &mut S,
    ) -> SbqNodeDistanceMeasure<'b> {
        SbqNodeDistanceMeasure::with_index_pointer(self, index_pointer, stats)
    }

    fn get_query_distance_measure(&self, query: LabeledVector) -> SbqSearchDistanceMeasure {
        SbqSearchDistanceMeasure::new(&self.quantizer, query)
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
            qdm.query.vec().to_full_slice(),
        ))
    }

    fn get_neighbors_with_distances_from_disk<S: StatsNodeRead + StatsDistanceComparison>(
        &self,
        neighbors_of: ItemPointer,
        result: &mut Vec<NeighborWithDistance>,
        stats: &mut S,
    ) {
        let rn = unsafe { SbqNode::read(self.index, neighbors_of, self.has_labels, stats) };
        let archived = rn.get_archived_node();
        let q = archived.get_bq_vector();

        for n in rn.get_archived_node().iter_neighbors() {
            //OPT: we can optimize this if num_dimensions_for_neighbors == num_dimensions_to_index
            let rn1 = unsafe { SbqNode::read(self.index, n, self.has_labels, stats) };
            let arch = rn1.get_archived_node();
            stats.record_quantized_distance_comparison();
            let dist = distance_xor_optimized(q, arch.get_bq_vector());
            result.push(NeighborWithDistance::new(
                n,
                DistanceWithTieBreak::new(dist as f32, neighbors_of, n),
                arch.get_labels().map(Into::into),
            ))
        }
    }

    /* get_lsn and visit_lsn are different because the distance
    comparisons for SBQ get the vector from different places */
    fn create_lsn_for_start_node(
        &self,
        lsr: &mut ListSearchResult<Self::QueryDistanceMeasure, Self::LSNPrivateData>,
        index_pointer: ItemPointer,
        gns: &GraphNeighborStore,
    ) -> Option<ListSearchNeighbor<Self::LSNPrivateData>> {
        if !lsr.prepare_insert(index_pointer) {
            // Already processed this start node
            return None;
        }

        let rn =
            unsafe { SbqNode::read(self.index, index_pointer, self.has_labels, &mut lsr.stats) };
        let node = rn.get_archived_node();
        let distance = lsr.sdm.as_ref().unwrap().calculate_bq_distance(
            node.get_bq_vector(),
            gns,
            &mut lsr.stats,
        );

        Some(ListSearchNeighbor::new(
            index_pointer,
            lsr.create_distance_with_tie_break(distance, index_pointer),
            PhantomData::<bool>,
            node.get_labels().map(Into::into),
        ))
    }

    fn visit_lsn(
        &self,
        lsr: &mut ListSearchResult<Self::QueryDistanceMeasure, Self::LSNPrivateData>,
        lsn_idx: usize,
        gns: &GraphNeighborStore,
        no_filter: bool,
    ) {
        let lsn_index_pointer = lsr.get_lsn_by_idx(lsn_idx).index_pointer;
        self.visit_lsn_internal(lsr, lsn_index_pointer, gns, no_filter);
    }

    fn return_lsn(
        &self,
        lsn: &ListSearchNeighbor<Self::LSNPrivateData>,
        stats: &mut GreedySearchStats,
    ) -> HeapPointer {
        let lsn_index_pointer = lsn.index_pointer;
        let rn = unsafe { SbqNode::read(self.index, lsn_index_pointer, self.has_labels, stats) };
        let node = rn.get_archived_node();

        node.get_heap_item_pointer()
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

        let mut node =
            unsafe { SbqNode::modify(self.index, index_pointer, self.has_labels, stats) };
        let mut archived = node.get_archived_node();
        archived.set_neighbors(neighbors, meta);
        node.commit();
    }

    fn get_distance_function(&self) -> DistanceFn {
        self.distance_fn
    }

    fn get_labels<S: StatsNodeRead>(
        &self,
        index_pointer: IndexPointer,
        stats: &mut S,
    ) -> Option<LabelSet> {
        if !self.has_labels {
            return None;
        }
        let rn = unsafe { SbqNode::read(self.index, index_pointer, true, stats) };
        let node = rn.get_archived_node();
        node.get_labels().map(Into::into)
    }
}

pub type SbqSpeedupStorageLsnPrivateData = PhantomData<bool>; //no data stored

#[cfg(any(test, feature = "pg_test"))]
#[pgrx::pg_schema]
mod tests {
    use pgrx::*;

    use crate::access_method::distance::DistanceType;

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
