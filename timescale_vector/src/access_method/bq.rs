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
    pg_sys::{InvalidBlockNumber, InvalidOffsetNumber},
    PgRelation,
};
use rkyv::{vec::ArchivedVec, Archive, Archived, Deserialize, Serialize};

use crate::util::{
    page::PageType, table_slot::TableSlot, tape::Tape, ArchivedItemPointer, HeapPointer,
    IndexPointer, ItemPointer, ReadableBuffer,
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
            if meta_page.get_pq_pointer().is_none() {
                pgrx::error!("No BQ pointer found in meta page");
            }
            let pq_item_pointer = meta_page.get_pq_pointer().unwrap();
            let rpq = BqMeans::read(index, pq_item_pointer, stats);
            let rpn = rpq.get_archived_node();

            quantizer.load(rpn.count, rpn.means.to_vec());
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
            self.mean = vec![0.0; meta_page.get_num_dimensions() as _];
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

    fn get_distance_table(
        &self,
        query: &[f32],
        _distance_fn: fn(&[f32], &[f32]) -> f32,
    ) -> BqDistanceTable {
        BqDistanceTable::new(self.quantize(query))
    }
}

/// DistanceCalculator encapsulates the code to generate distances between a BQ vector and a query.
pub struct BqDistanceTable {
    quantized_vector: Vec<BqVectorElement>,
}

impl BqDistanceTable {
    pub fn new(query: Vec<BqVectorElement>) -> BqDistanceTable {
        BqDistanceTable {
            quantized_vector: query,
        }
    }

    /// distance emits the sum of distances between each centroid in the quantized vector.
    pub fn distance(&self, bq_vector: &[BqVectorElement]) -> f32 {
        let count_ones = distance_xor_optimized(&self.quantized_vector, bq_vector);
        //dot product is LOWER the more xors that lead to 1 becaues that means a negative times a positive = negative component
        //but the distance is 1 - dot product, so the more count_ones the higher the distance.
        // one other check for distance(a,a), xor=0, count_ones=0, distance=0
        count_ones as f32
    }
}

//FIXME: cleanup make this into a struct
pub enum BqSearchDistanceMeasure {
    Bq(BqDistanceTable, PgVector),
}

impl BqSearchDistanceMeasure {
    pub fn calculate_bq_distance<S: StatsDistanceComparison>(
        table: &BqDistanceTable,
        bq_vector: &[BqVectorElement],
        stats: &mut S,
    ) -> f32 {
        assert!(bq_vector.len() > 0);
        let vec = bq_vector;
        stats.record_quantized_distance_comparison();
        table.distance(vec)
    }
}

pub struct BqNodeDistanceMeasure<'a> {
    readable_node: ReadableBqNode<'a>,
    storage: &'a BqSpeedupStorage<'a>,
}

impl<'a> BqNodeDistanceMeasure<'a> {
    pub unsafe fn with_index_pointer<T: StatsNodeRead>(
        storage: &'a BqSpeedupStorage<'a>,
        index_pointer: IndexPointer,
        stats: &mut T,
    ) -> Self {
        let rn = unsafe { BqNode::read(storage.index, index_pointer, stats) };
        Self {
            readable_node: rn,
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
        //OPT: should I get and memoize the vector from self.readable_node in with_index_pointer above?
        let rn1 = BqNode::read(self.storage.index, index_pointer, stats);
        let rn2 = &self.readable_node;
        let node1 = rn1.get_archived_node();
        let node2 = rn2.get_archived_node();
        assert!(node1.bq_vector.len() > 0);
        assert!(node1.bq_vector.len() == node2.bq_vector.len());
        let vec1 = node1.bq_vector.as_slice();
        let vec2 = node2.bq_vector.as_slice();
        distance_xor_optimized(vec1, vec2) as f32
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
}

impl<'a> BqSpeedupStorage<'a> {
    pub fn new_for_build(
        index: &'a PgRelation,
        heap_rel: &'a PgRelation,
        heap_attr: pgrx::pg_sys::AttrNumber,
        distance_fn: fn(&[f32], &[f32]) -> f32,
    ) -> BqSpeedupStorage<'a> {
        Self {
            index: index,
            distance_fn: distance_fn,
            quantizer: BqQuantizer::new(),
            heap_rel: heap_rel,
            heap_attr: heap_attr,
            qv_cache: RefCell::new(QuantizedVectorCache::new(1000)),
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
        heap_attr: pgrx::pg_sys::AttrNumber,
        index_relation: &'a PgRelation,
        meta_page: &super::meta_page::MetaPage,
        stats: &mut S,
    ) -> BqSpeedupStorage<'a> {
        Self {
            index: index_relation,
            distance_fn: meta_page.get_distance_function(),
            quantizer: Self::load_quantizer(index_relation, meta_page, stats),
            heap_rel: heap_rel,
            heap_attr: heap_attr,
            qv_cache: RefCell::new(QuantizedVectorCache::new(1000)),
        }
    }

    pub fn load_for_search(
        index_relation: &'a PgRelation,
        heap_relation: &'a PgRelation,
        quantizer: &BqQuantizer,
        distance_fn: fn(&[f32], &[f32]) -> f32,
    ) -> BqSpeedupStorage<'a> {
        Self {
            index: index_relation,
            distance_fn: distance_fn,
            //OPT: get rid of clone
            quantizer: quantizer.clone(),
            heap_rel: heap_relation,
            heap_attr: get_attribute_number_from_index(index_relation),
            qv_cache: RefCell::new(QuantizedVectorCache::new(1000)),
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
            super::meta_page::MetaPage::update_pq_pointer(&self.index, index_pointer, stats);
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
        //Opt shouldn't need to read the node in the builder graph case.
        let rn_visiting = unsafe { BqNode::read(self.index, lsn_index_pointer, &mut lsr.stats) };
        let node_visiting = rn_visiting.get_archived_node();

        let neighbors = match gns {
            GraphNeighborStore::Disk => node_visiting.get_index_pointer_to_neighbors(),
            GraphNeighborStore::Builder(b) => b.get_neighbors(lsn_index_pointer),
        };

        for (i, &neighbor_index_pointer) in neighbors.iter().enumerate() {
            if !lsr.prepare_insert(neighbor_index_pointer) {
                continue;
            }

            let distance = match lsr.sdm.as_ref().unwrap() {
                BqSearchDistanceMeasure::Bq(table, _) => {
                    /* Note: there is no additional node reads here. We get all of our info from node_visiting
                     * This is what gives us a speedup in BQ Speedup */
                    match gns {
                        GraphNeighborStore::Disk => {
                            let bq_vector = node_visiting.neighbor_vectors[i].as_slice();
                            BqSearchDistanceMeasure::calculate_bq_distance(
                                table,
                                bq_vector,
                                &mut lsr.stats,
                            )
                        }
                        GraphNeighborStore::Builder(_) => {
                            let mut cache = self.qv_cache.borrow_mut();
                            let bq_vector = cache.get(neighbor_index_pointer, self, &mut lsr.stats);
                            let dist = BqSearchDistanceMeasure::calculate_bq_distance(
                                table,
                                bq_vector,
                                &mut lsr.stats,
                            );
                            dist
                        }
                    }
                    //let bq_vector = node_visiting.neighbor_vectors[i].as_slice();
                    //BqSearchDistanceMeasure::calculate_bq_distance(table, bq_vector, &mut lsr.stats)
                }
            };
            let lsn =
                ListSearchNeighbor::new(neighbor_index_pointer, distance, PhantomData::<bool>);

            lsr.insert_neighbor(lsn);
        }
    }

    unsafe fn get_heap_table_slot_from_heap_pointer<T: StatsHeapNodeRead>(
        &self,
        heap_pointer: HeapPointer,
        stats: &mut T,
    ) -> TableSlot {
        TableSlot::new(self.heap_rel, heap_pointer, self.heap_attr, stats)
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

        let node = BqNode::new(heap_pointer, &meta_page, bq_vector.as_slice());

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
        return BqSearchDistanceMeasure::Bq(
            self.quantizer
                .get_distance_table(query.to_slice(), self.distance_fn),
            query,
        );
    }

    fn get_full_distance_for_resort<S: StatsHeapNodeRead + StatsDistanceComparison>(
        &self,
        qdm: &Self::QueryDistanceMeasure,
        _index_pointer: IndexPointer,
        heap_pointer: HeapPointer,
        stats: &mut S,
    ) -> f32 {
        let slot = unsafe { self.get_heap_table_slot_from_heap_pointer(heap_pointer, stats) };
        match qdm {
            BqSearchDistanceMeasure::Bq(_, query) => self.get_distance_function()(
                unsafe { slot.get_pg_vector().to_slice() },
                query.to_slice(),
            ),
        }
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

        for (i, n) in rn.get_archived_node().iter_neighbors().enumerate() {
            //let dist = unsafe { dist_state.get_distance(n, stats) };
            assert!(i < archived.neighbor_vectors.len());
            let neighbor_q = archived.neighbor_vectors[i].as_slice();
            stats.record_quantized_distance_comparison();
            let dist = distance_xor_optimized(q, neighbor_q);
            result.push(NeighborWithDistance::new(n, dist as f32))
        }
    }

    /* get_lsn and visit_lsn are different because the distance
    comparisons for BQ get the vector from different places */
    fn create_lsn_for_init_id(
        &self,
        lsr: &mut ListSearchResult<Self::QueryDistanceMeasure, Self::LSNPrivateData>,
        index_pointer: ItemPointer,
        _gns: &GraphNeighborStore,
    ) -> ListSearchNeighbor<Self::LSNPrivateData> {
        if !lsr.prepare_insert(index_pointer) {
            panic!("should not have had an init id already inserted");
        }

        let rn = unsafe { BqNode::read(self.index, index_pointer, &mut lsr.stats) };
        let node = rn.get_archived_node();

        let distance = match lsr.sdm.as_ref().unwrap() {
            BqSearchDistanceMeasure::Bq(table, _) => {
                BqSearchDistanceMeasure::calculate_bq_distance(
                    table,
                    node.bq_vector.as_slice(),
                    &mut lsr.stats,
                )
            }
        };

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
    pub fn new(
        heap_pointer: HeapPointer,
        meta_page: &MetaPage,
        bq_vector: &[BqVectorElement],
    ) -> Self {
        let num_neighbors = meta_page.get_num_neighbors();
        // always use vectors of num_neighbors in length because we never want the serialized size of a Node to change
        let neighbor_index_pointers: Vec<_> = (0..num_neighbors)
            .map(|_| ItemPointer::new(InvalidBlockNumber, InvalidOffsetNumber))
            .collect();

        let neighbor_vectors: Vec<_> = (0..num_neighbors)
            .map(|_| vec![0; BqQuantizer::quantized_size(meta_page.get_num_dimensions() as _)])
            .collect();

        Self {
            heap_item_pointer: heap_pointer,
            bq_vector: bq_vector.to_vec(),
            neighbor_index_pointers: neighbor_index_pointers,
            neighbor_vectors: neighbor_vectors,
        }
    }
}

impl ArchivedBqNode {
    pub fn neighbor_index_pointer(
        self: Pin<&mut Self>,
    ) -> Pin<&mut ArchivedVec<ArchivedItemPointer>> {
        unsafe { self.map_unchecked_mut(|s| &mut s.neighbor_index_pointers) }
    }

    pub fn neighbor_vector(self: Pin<&mut Self>) -> Pin<&mut ArchivedVec<ArchivedVec<u64>>> {
        unsafe { self.map_unchecked_mut(|s| &mut s.neighbor_vectors) }
    }

    pub fn bq_vector(self: Pin<&mut Self>) -> Pin<&mut Archived<Vec<BqVectorElement>>> {
        unsafe { self.map_unchecked_mut(|s| &mut s.bq_vector) }
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

            let quantized = cache.must_get(ip);

            let mut neighbor_vector = self.as_mut().neighbor_vector().index_pin(i);
            for (index_in_q_vec, val) in quantized.iter().enumerate() {
                let mut x = neighbor_vector.as_mut().index_pin(index_in_q_vec);
                *x = *val;
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
    unsafe fn test_bq_storage_index_creation() -> spi::Result<()> {
        crate::access_method::build::tests::test_index_creation_and_accuracy_scaffold(
            "num_neighbors=38, storage_layout = io_optimized",
        )?;
        Ok(())
    }

    #[pg_test]
    unsafe fn test_bq_storage_index_creation_few_neighbors() -> spi::Result<()> {
        //a test with few neighbors tests the case that nodes share a page, which has caused deadlocks in the past.
        crate::access_method::build::tests::test_index_creation_and_accuracy_scaffold(
            "num_neighbors=10, storage_layout = io_optimized",
        )?;
        Ok(())
    }

    #[test]
    fn test_bq_storage_delete_vacuum_plain() {
        crate::access_method::vacuum::tests::test_delete_vacuum_plain_scaffold(
            "num_neighbors = 10, storage_layout = io_optimized",
        );
    }

    #[test]
    fn test_bq_storage_delete_vacuum_full() {
        crate::access_method::vacuum::tests::test_delete_vacuum_full_scaffold(
            "num_neighbors = 38, storage_layout = io_optimized",
        );
    }

    #[pg_test]
    unsafe fn test_bq_storage_empty_table_insert() -> spi::Result<()> {
        crate::access_method::build::tests::test_empty_table_insert_scaffold(
            "num_neighbors=38, storage_layout = io_optimized",
        )
    }

    #[pg_test]
    unsafe fn test_bq_storage_insert_empty_insert() -> spi::Result<()> {
        crate::access_method::build::tests::test_insert_empty_insert_scaffold(
            "num_neighbors=38, storage_layout = io_optimized",
        )
    }
}
