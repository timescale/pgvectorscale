use super::{
    distance::distance_xor_optimized,
    graph::{ListSearchNeighbor, ListSearchResult},
    graph_neighbor_store::GraphNeighborStore,
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
    pg_sys::{FirstOffsetNumber, InvalidBlockNumber, InvalidOffsetNumber, BLCKSZ},
    PgRelation,
};
use rkyv::{vec::ArchivedVec, Archive, Deserialize, Serialize};

use crate::util::{
    page::{PageType, ReadablePage},
    ports::{PageGetItem, PageGetItemId, PageGetMaxOffsetNumber},
    table_slot::TableSlot,
    tape::Tape,
    ArchivedItemPointer, HeapPointer, IndexPointer, ItemPointer, ReadableBuffer,
};

use super::{meta_page::MetaPage, neighbor_with_distance::NeighborWithDistance};
use crate::util::WritableBuffer;

type BqVectorElement = u64;
const BITS_STORE_TYPE_SIZE: usize = 64;

#[derive(Archive, Deserialize, Serialize, Readable, Writeable)]
#[archive(check_bytes)]
#[repr(C)]
pub struct BqMeans {
    count: u64,
    means: Vec<f32>,
}

impl BqMeans {
    pub unsafe fn load<S: StatsNodeRead>(
        index: &PgRelation,
        meta_page: &super::meta_page::MetaPage,
        stats: &mut S,
    ) -> BqQuantizer {
        let mut quantizer = BqQuantizer::new();
        if quantizer.use_mean {
            if meta_page.get_quantizer_metadata_pointer().is_none() {
                pgrx::error!("No BQ pointer found in meta page");
            }
            let quantizer_item_pointer = meta_page.get_quantizer_metadata_pointer().unwrap();
            let bq = BqMeans::read(index, quantizer_item_pointer, stats);
            let archived = bq.get_archived_node();

            quantizer.load(archived.count, archived.means.to_vec());
        }
        quantizer
    }

    pub unsafe fn store<S: StatsNodeWrite>(
        index: &PgRelation,
        quantizer: &BqQuantizer,
        stats: &mut S,
    ) -> ItemPointer {
        let mut tape = Tape::new(index, PageType::BqMeans);
        let node = BqMeans {
            count: quantizer.count,
            means: quantizer.mean.to_vec(),
        };
        let ptr = node.write(&mut tape, stats);
        tape.close();
        ptr
    }
}

#[derive(Clone)]
pub struct BqQuantizer {
    pub use_mean: bool,
    training: bool,
    pub count: u64,
    pub mean: Vec<f32>,
}

impl BqQuantizer {
    fn new() -> BqQuantizer {
        Self {
            use_mean: true,
            training: false,
            count: 0,
            mean: vec![],
        }
    }

    fn load(&mut self, count: u64, mean: Vec<f32>) {
        self.count = count;
        self.mean = mean;
    }

    fn quantized_size(full_vector_size: usize) -> usize {
        if full_vector_size % BITS_STORE_TYPE_SIZE == 0 {
            full_vector_size / BITS_STORE_TYPE_SIZE
        } else {
            (full_vector_size / BITS_STORE_TYPE_SIZE) + 1
        }
    }

    fn quantized_size_bytes(num_dimensions: usize) -> usize {
        Self::quantized_size(num_dimensions) * std::mem::size_of::<BqVectorElement>()
    }

    fn quantize(&self, full_vector: &[f32]) -> Vec<BqVectorElement> {
        assert!(!self.training);
        if self.use_mean {
            let mut res_vector = vec![0; Self::quantized_size(full_vector.len())];

            for (i, &v) in full_vector.iter().enumerate() {
                if v > self.mean[i] {
                    res_vector[i / BITS_STORE_TYPE_SIZE] |= 1 << (i % BITS_STORE_TYPE_SIZE);
                }
            }

            res_vector
        } else {
            let mut res_vector = vec![0; Self::quantized_size(full_vector.len())];

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
        }
    }

    fn add_sample(&mut self, sample: &[f32]) {
        if self.use_mean {
            self.count += 1;
            assert!(self.mean.len() == sample.len());

            self.mean
                .iter_mut()
                .zip(sample.iter())
                .for_each(|(m, s)| *m += (s - *m) / self.count as f32);
        }
    }

    fn finish_training(&mut self) {
        self.training = false;
    }

    fn vector_for_new_node(
        &self,
        _meta_page: &super::meta_page::MetaPage,
        full_vector: &[f32],
    ) -> Vec<BqVectorElement> {
        self.quantize(&full_vector)
    }
}

pub struct BqSearchDistanceMeasure {
    quantized_vector: Vec<BqVectorElement>,
    query: PgVector,
    num_dimensions_for_neighbors: usize,
}

impl BqSearchDistanceMeasure {
    pub fn new(
        quantized_vector: Vec<BqVectorElement>,
        query: PgVector,
        num_dimensions_for_neighbors: usize,
    ) -> BqSearchDistanceMeasure {
        BqSearchDistanceMeasure {
            quantized_vector,
            query,
            num_dimensions_for_neighbors,
        }
    }

    pub fn calculate_bq_distance<S: StatsDistanceComparison>(
        &self,
        bq_vector: &[BqVectorElement],
        gns: &GraphNeighborStore,
        stats: &mut S,
    ) -> f32 {
        assert!(bq_vector.len() > 0);
        stats.record_quantized_distance_comparison();
        let (a, b) = match gns {
            GraphNeighborStore::Disk => {
                if self.num_dimensions_for_neighbors > 0 {
                    let quantized_dimensions =
                        BqQuantizer::quantized_size(self.num_dimensions_for_neighbors);
                    debug_assert!(self.quantized_vector.len() >= quantized_dimensions);
                    debug_assert!(bq_vector.len() >= quantized_dimensions);
                    (
                        &self.quantized_vector.as_slice()[..quantized_dimensions],
                        &bq_vector[..quantized_dimensions],
                    )
                } else {
                    debug_assert!(self.quantized_vector.len() == bq_vector.len());
                    (self.quantized_vector.as_slice(), bq_vector)
                }
            }
            GraphNeighborStore::Builder(_b) => {
                debug_assert!(self.quantized_vector.len() == bq_vector.len());
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

pub struct BqNodeDistanceMeasure<'a> {
    vec: Vec<BqVectorElement>,
    storage: &'a BqSpeedupStorage<'a>,
}

impl<'a> BqNodeDistanceMeasure<'a> {
    pub unsafe fn with_index_pointer<T: StatsNodeRead>(
        storage: &'a BqSpeedupStorage<'a>,
        index_pointer: IndexPointer,
        stats: &mut T,
    ) -> Self {
        let cache = &mut storage.qv_cache.borrow_mut();
        Self {
            vec: cache.get(index_pointer, storage, stats).to_vec(),
            storage: storage,
        }
    }
}

impl<'a> NodeDistanceMeasure for BqNodeDistanceMeasure<'a> {
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
    quantized_vector_map: HashMap<ItemPointer, Vec<BqVectorElement>>,
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
        storage: &BqSpeedupStorage,
        stats: &mut S,
    ) -> &[BqVectorElement] {
        self.quantized_vector_map
            .entry(index_pointer)
            .or_insert_with(|| {
                storage.get_quantized_vector_from_index_pointer(index_pointer, stats)
            })
    }

    fn get_and_cache_all_on_page<S: StatsNodeRead>(
        &mut self,
        index_pointer: IndexPointer,
        storage: &BqSpeedupStorage,
        stats: &mut S,
    ) -> &[BqVectorElement] {
        if self.quantized_vector_map.contains_key(&index_pointer) {
            return self.must_get(index_pointer);
        } else {
            unsafe {
                stats.record_read();
                let mut page = ReadablePage::read(storage.index, index_pointer.block_number);
                let max_offset = PageGetMaxOffsetNumber(&page);
                for offset_number in FirstOffsetNumber..(max_offset + 1) as _ {
                    let rb = page.get_item_unchecked(offset_number);
                    let node = ReadableBqNode::with_readable_buffer(rb);
                    let vec = node.get_archived_node().bq_vector.as_slice().to_vec();
                    self.quantized_vector_map.insert(
                        ItemPointer::new(index_pointer.block_number, offset_number),
                        vec,
                    );
                    page = node.get_owned_page();
                }
                return self.must_get(index_pointer);
            }
        }
    }

    fn must_get(&self, index_pointer: IndexPointer) -> &[BqVectorElement] {
        self.quantized_vector_map.get(&index_pointer).unwrap()
    }

    /* Ensure that all these elements are in the cache. If the capacity isn't big enough throw an error.
    must_get must succeed on all the elements after this call prior to another get or preload call */

    fn preload<I: Iterator<Item = IndexPointer>, S: StatsNodeRead>(
        &mut self,
        index_pointers: I,
        storage: &BqSpeedupStorage,
        stats: &mut S,
    ) {
        for index_pointer in index_pointers {
            self.get(index_pointer, storage, stats);
        }
    }
}

pub struct BqSpeedupStorage<'a> {
    pub index: &'a PgRelation,
    pub distance_fn: fn(&[f32], &[f32]) -> f32,
    quantizer: BqQuantizer,
    heap_rel: &'a PgRelation,
    heap_attr: pgrx::pg_sys::AttrNumber,
    qv_cache: RefCell<QuantizedVectorCache>,
    num_dimensions_for_neighbors: usize,
}

impl<'a> BqSpeedupStorage<'a> {
    pub fn new_for_build(
        index: &'a PgRelation,
        heap_rel: &'a PgRelation,
        meta_page: &super::meta_page::MetaPage,
    ) -> BqSpeedupStorage<'a> {
        Self {
            index: index,
            distance_fn: meta_page.get_distance_function(),
            quantizer: BqQuantizer::new(),
            heap_rel: heap_rel,
            heap_attr: get_attribute_number_from_index(index),
            qv_cache: RefCell::new(QuantizedVectorCache::new(1000)),
            num_dimensions_for_neighbors: meta_page.get_num_dimensions_for_neighbors() as usize,
        }
    }

    fn load_quantizer<S: StatsNodeRead>(
        index_relation: &PgRelation,
        meta_page: &super::meta_page::MetaPage,
        stats: &mut S,
    ) -> BqQuantizer {
        unsafe { BqMeans::load(&index_relation, meta_page, stats) }
    }

    pub fn load_for_insert<S: StatsNodeRead>(
        heap_rel: &'a PgRelation,
        index_relation: &'a PgRelation,
        meta_page: &super::meta_page::MetaPage,
        stats: &mut S,
    ) -> BqSpeedupStorage<'a> {
        Self {
            index: index_relation,
            distance_fn: meta_page.get_distance_function(),
            quantizer: Self::load_quantizer(index_relation, meta_page, stats),
            heap_rel: heap_rel,
            heap_attr: get_attribute_number_from_index(index_relation),
            qv_cache: RefCell::new(QuantizedVectorCache::new(1000)),
            num_dimensions_for_neighbors: meta_page.get_num_dimensions_for_neighbors() as usize,
        }
    }

    pub fn load_for_search(
        index_relation: &'a PgRelation,
        heap_relation: &'a PgRelation,
        quantizer: &BqQuantizer,
        meta_page: &super::meta_page::MetaPage,
    ) -> BqSpeedupStorage<'a> {
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
    ) -> Vec<BqVectorElement> {
        let rn = unsafe { BqNode::read(self.index, index_pointer, stats) };
        let node = rn.get_archived_node();
        node.bq_vector.as_slice().to_vec()
    }

    fn write_quantizer_metadata<S: StatsNodeWrite + StatsNodeModify>(&self, stats: &mut S) {
        if self.quantizer.use_mean {
            let index_pointer = unsafe { BqMeans::store(&self.index, &self.quantizer, stats) };
            super::meta_page::MetaPage::update_quantizer_metadata_pointer(
                &self.index,
                index_pointer,
                stats,
            );
        }
    }

    fn visit_lsn_internal(
        &self,
        lsr: &mut ListSearchResult<
            <BqSpeedupStorage<'a> as Storage>::QueryDistanceMeasure,
            <BqSpeedupStorage<'a> as Storage>::LSNPrivateData,
        >,
        lsn_index_pointer: IndexPointer,
        gns: &GraphNeighborStore,
    ) {
        match gns {
            GraphNeighborStore::Disk => {
                let rn_visiting =
                    unsafe { BqNode::read(self.index, lsn_index_pointer, &mut lsr.stats) };
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
                        //todo: probably better as distance cache but first see if it helps

                        let mut cachie = self.qv_cache.borrow_mut();
                        let bq_vector = cache.get_and_cache_all_on_page(
                            neighbor_index_pointer,
                            self,
                            &mut lsr.stats,
                        );

                        /*let rn_neighbor = unsafe {
                            BqNode::read(self.index, neighbor_index_pointer, &mut lsr.stats)
                        };
                        let node_neighbor = rn_neighbor.get_archived_node();
                        let bq_vector = node_neighbor.bq_vector.as_slice();*/
                        lsr.sdm.as_ref().unwrap().calculate_bq_distance(
                            bq_vector,
                            gns,
                            &mut lsr.stats,
                        )
                    };

                    let lsn = ListSearchNeighbor::new(
                        neighbor_index_pointer,
                        distance,
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
                        distance,
                        PhantomData::<bool>,
                    );

                    lsr.insert_neighbor(lsn);
                }
            }
        }
    }

    unsafe fn get_heap_table_slot_from_heap_pointer<T: StatsHeapNodeRead>(
        &self,
        heap_pointer: HeapPointer,
        stats: &mut T,
    ) -> TableSlot {
        TableSlot::new(self.heap_rel, heap_pointer, stats)
    }
}

pub type BqSpeedupStorageLsnPrivateData = PhantomData<bool>; //no data stored

impl<'a> Storage for BqSpeedupStorage<'a> {
    type QueryDistanceMeasure = BqSearchDistanceMeasure;
    type NodeDistanceMeasure<'b> = BqNodeDistanceMeasure<'b> where Self: 'b;
    type ArchivedType = ArchivedBqNode;
    type LSNPrivateData = BqSpeedupStorageLsnPrivateData; //no data stored

    fn page_type() -> PageType {
        PageType::BqNode
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

        let node = BqNode::with_meta(heap_pointer, &meta_page, bq_vector.as_slice());

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
        neighbors: &Vec<NeighborWithDistance>,
        stats: &mut S,
    ) {
        let mut cache = self.qv_cache.borrow_mut();
        /* It's important to preload cache with all the items since you can run into deadlocks
        if you try to fetch a quantized vector while holding the BqNode::modify lock */
        let iter = neighbors
            .iter()
            .map(|n| n.get_index_pointer_to_neighbor())
            .chain(once(index_pointer));
        cache.preload(iter, self, stats);

        let node = unsafe { BqNode::modify(self.index, index_pointer, stats) };
        let mut archived = node.get_archived_node();
        archived.as_mut().set_neighbors(neighbors, &meta, &cache);

        node.commit();
    }

    unsafe fn get_node_distance_measure<'b, S: StatsNodeRead>(
        &'b self,
        index_pointer: IndexPointer,
        stats: &mut S,
    ) -> BqNodeDistanceMeasure<'b> {
        BqNodeDistanceMeasure::with_index_pointer(self, index_pointer, stats)
    }

    fn get_query_distance_measure(&self, query: PgVector) -> BqSearchDistanceMeasure {
        return BqSearchDistanceMeasure::new(
            self.quantizer.quantize(query.to_index_slice()),
            query,
            self.num_dimensions_for_neighbors,
        );
    }

    fn get_full_distance_for_resort<S: StatsHeapNodeRead + StatsDistanceComparison>(
        &self,
        qdm: &Self::QueryDistanceMeasure,
        _index_pointer: IndexPointer,
        heap_pointer: HeapPointer,
        meta_page: &MetaPage,
        stats: &mut S,
    ) -> f32 {
        let slot = unsafe { self.get_heap_table_slot_from_heap_pointer(heap_pointer, stats) };

        let datum = unsafe { slot.get_attribute(self.heap_attr).unwrap() };
        let vec = unsafe { PgVector::from_datum(datum, meta_page, false, true) };
        self.get_distance_function()(vec.to_full_slice(), qdm.query.to_full_slice())
    }

    fn get_neighbors_with_distances_from_disk<S: StatsNodeRead + StatsDistanceComparison>(
        &self,
        neighbors_of: ItemPointer,
        result: &mut Vec<NeighborWithDistance>,
        stats: &mut S,
    ) {
        let rn = unsafe { BqNode::read(self.index, neighbors_of, stats) };
        let archived = rn.get_archived_node();
        let q = archived.bq_vector.as_slice();

        for n in rn.get_archived_node().iter_neighbors() {
            //OPT: we can optimize this if num_dimensions_for_neighbors == num_dimensions_to_index
            let rn1 = unsafe { BqNode::read(self.index, n, stats) };
            stats.record_quantized_distance_comparison();
            let dist = distance_xor_optimized(q, rn1.get_archived_node().bq_vector.as_slice());
            result.push(NeighborWithDistance::new(n, dist as f32))
        }
    }

    /* get_lsn and visit_lsn are different because the distance
    comparisons for BQ get the vector from different places */
    fn create_lsn_for_init_id(
        &self,
        lsr: &mut ListSearchResult<Self::QueryDistanceMeasure, Self::LSNPrivateData>,
        index_pointer: ItemPointer,
        gns: &GraphNeighborStore,
    ) -> ListSearchNeighbor<Self::LSNPrivateData> {
        if !lsr.prepare_insert(index_pointer) {
            panic!("should not have had an init id already inserted");
        }

        let rn = unsafe { BqNode::read(self.index, index_pointer, &mut lsr.stats) };
        let node = rn.get_archived_node();

        let distance = lsr.sdm.as_ref().unwrap().calculate_bq_distance(
            node.bq_vector.as_slice(),
            gns,
            &mut lsr.stats,
        );

        ListSearchNeighbor::new(index_pointer, distance, PhantomData::<bool>)
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
        let rn = unsafe { BqNode::read(self.index, lsn_index_pointer, stats) };
        let node = rn.get_archived_node();
        let heap_pointer = node.heap_item_pointer.deserialize_item_pointer();
        heap_pointer
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
        if you try to fetch a quantized vector while holding the BqNode::modify lock */
        let iter = neighbors
            .iter()
            .map(|n| n.get_index_pointer_to_neighbor())
            .chain(once(index_pointer));
        cache.preload(iter, self, stats);

        let node = unsafe { BqNode::modify(self.index, index_pointer, stats) };
        let mut archived = node.get_archived_node();
        archived.as_mut().set_neighbors(neighbors, &meta, &cache);
        node.commit();
    }

    fn get_distance_function(&self) -> fn(&[f32], &[f32]) -> f32 {
        self.distance_fn
    }
}

use timescale_vector_derive::{Readable, Writeable};

#[derive(Archive, Deserialize, Serialize, Readable, Writeable)]
#[archive(check_bytes)]
pub struct BqNode {
    pub heap_item_pointer: HeapPointer,
    pub bq_vector: Vec<u64>, //don't use BqVectorElement because we don't want to change the size in on-disk format by accident
    neighbor_index_pointers: Vec<ItemPointer>,
    neighbor_vectors: Vec<Vec<u64>>, //don't use BqVectorElement because we don't want to change the size in on-disk format by accident
}

impl BqNode {
    pub fn with_meta(
        heap_pointer: HeapPointer,
        meta_page: &MetaPage,
        bq_vector: &[BqVectorElement],
    ) -> Self {
        Self::new(
            heap_pointer,
            meta_page.get_num_neighbors() as usize,
            meta_page.get_num_dimensions_to_index() as usize,
            meta_page.get_num_dimensions_for_neighbors() as usize,
            bq_vector,
        )
    }

    fn new(
        heap_pointer: HeapPointer,
        num_neighbors: usize,
        _num_dimensions: usize,
        num_dimensions_for_neighbors: usize,
        bq_vector: &[BqVectorElement],
    ) -> Self {
        // always use vectors of num_neighbors in length because we never want the serialized size of a Node to change
        let neighbor_index_pointers: Vec<_> = (0..num_neighbors)
            .map(|_| ItemPointer::new(InvalidBlockNumber, InvalidOffsetNumber))
            .collect();

        let neighbor_vectors: Vec<_> = if num_dimensions_for_neighbors > 0 {
            (0..num_neighbors)
                .map(|_| vec![0; BqQuantizer::quantized_size(num_dimensions_for_neighbors as _)])
                .collect()
        } else {
            vec![]
        };

        Self {
            heap_item_pointer: heap_pointer,
            bq_vector: bq_vector.to_vec(),
            neighbor_index_pointers: neighbor_index_pointers,
            neighbor_vectors: neighbor_vectors,
        }
    }

    fn test_size(
        num_neighbors: usize,
        num_dimensions: usize,
        num_dimensions_for_neighbors: usize,
    ) -> usize {
        let v: Vec<BqVectorElement> = vec![0; BqQuantizer::quantized_size(num_dimensions)];
        let hp = HeapPointer::new(InvalidBlockNumber, InvalidOffsetNumber);
        let n = Self::new(
            hp,
            num_neighbors,
            num_dimensions,
            num_dimensions_for_neighbors,
            &v,
        );
        n.serialize_to_vec().len()
    }

    pub fn get_default_num_neighbors(
        num_dimensions: usize,
        num_dimensions_for_neighbors: usize,
    ) -> usize {
        //how many neighbors can fit on one page? That's what we choose.

        //we first overapproximate the number of neighbors and then double check by actually calculating the size of the BqNode.

        //blocksize - 100 bytes for the padding/header/etc.
        let page_size = BLCKSZ as usize - 50;
        //one quantized_vector takes this many bytes
        let vec_size = BqQuantizer::quantized_size_bytes(num_dimensions as usize) + 1;
        //start from the page size then subtract the heap_item_pointer and bq_vector elements of BqNode.
        let starting = BLCKSZ as usize - std::mem::size_of::<HeapPointer>() - vec_size;
        //one neigbors contribution to neighbor_index_pointers + neighbor_vectors in BqNode.
        let one_neighbor = vec_size + std::mem::size_of::<ItemPointer>();

        let mut num_neighbors_overapproximate: usize = starting / one_neighbor;
        while num_neighbors_overapproximate > 0 {
            let serialized_size = BqNode::test_size(
                num_neighbors_overapproximate as usize,
                num_dimensions as usize,
                num_dimensions_for_neighbors as usize,
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

impl ArchivedBqNode {
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
                let quantized = &cache.must_get(ip)[..BqQuantizer::quantized_size(
                    meta_page.get_num_dimensions_for_neighbors() as _,
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

impl ArchivedData for ArchivedBqNode {
    fn with_data(data: &mut [u8]) -> Pin<&mut ArchivedBqNode> {
        ArchivedBqNode::with_data(data)
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

    #[pg_test]
    unsafe fn test_bq_speedup_storage_index_creation_default_neighbors() -> spi::Result<()> {
        crate::access_method::build::tests::test_index_creation_and_accuracy_scaffold(
            "storage_layout = io_optimized",
        )?;
        Ok(())
    }

    #[pg_test]
    unsafe fn test_bq_speedup_storage_index_creation_few_neighbors() -> spi::Result<()> {
        //a test with few neighbors tests the case that nodes share a page, which has caused deadlocks in the past.
        crate::access_method::build::tests::test_index_creation_and_accuracy_scaffold(
            "num_neighbors=10, storage_layout = io_optimized",
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
    unsafe fn test_bq_speedup_storage_index_creation_num_dimensions() -> spi::Result<()> {
        crate::access_method::build::tests::test_index_creation_and_accuracy_scaffold(
            "storage_layout = io_optimized, num_dimensions=768",
        )?;
        Ok(())
    }

    #[pg_test]
    unsafe fn test_bq_speedup_storage_index_updates() -> spi::Result<()> {
        crate::access_method::build::tests::test_index_updates(
            "storage_layout = io_optimized, num_neighbors=10",
            300,
        )?;
        Ok(())
    }

    #[pg_test]
    unsafe fn test_bq_speedup_compressed_index_creation_default_neighbors() -> spi::Result<()> {
        crate::access_method::build::tests::test_index_creation_and_accuracy_scaffold(
            "storage_layout = memory_optimized",
        )?;
        Ok(())
    }

    #[pg_test]
    unsafe fn test_bq_compressed_storage_index_creation_few_neighbors() -> spi::Result<()> {
        //a test with few neighbors tests the case that nodes share a page, which has caused deadlocks in the past.
        crate::access_method::build::tests::test_index_creation_and_accuracy_scaffold(
            "num_neighbors=10, storage_layout = memory_optimized",
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
            "storage_layout = memory_optimized, num_dimensions=768",
        )?;
        Ok(())
    }

    #[pg_test]
    unsafe fn test_bq_compressed_storage_index_updates() -> spi::Result<()> {
        crate::access_method::build::tests::test_index_updates(
            "storage_layout = memory_optimized, num_neighbors=10",
            300,
        )?;
        Ok(())
    }
}
