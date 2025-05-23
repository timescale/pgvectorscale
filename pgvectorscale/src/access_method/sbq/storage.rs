use std::{cell::RefCell, iter::once, marker::PhantomData};

use pgrx::{pg_sys::AttrNumber, PgBox, PgRelation};

use crate::{
    access_method::{
        build::QUANTIZED_VECTOR_CACHE_SIZE,
        distance::{distance_xor_optimized, DistanceFn},
        graph::{
            neighbor_store::GraphNeighborStore,
            neighbor_with_distance::{DistanceWithTieBreak, NeighborWithDistance},
            ListSearchNeighbor, ListSearchResult,
        },
        labels::{LabelSet, LabelSetView, LabeledVector},
        meta_page::MetaPage,
        pg_vector::PgVector,
        stats::{
            GreedySearchStats, StatsDistanceComparison, StatsHeapNodeRead, StatsNodeModify,
            StatsNodeRead, StatsNodeWrite, WriteStats,
        },
        storage::Storage,
        storage_common::get_index_vector_attribute,
    },
    util::{
        page::PageType, table_slot::TableSlot, tape::Tape, HeapPointer, IndexPointer, ItemPointer,
    },
};

use super::{
    cache::QuantizedVectorCache,
    node::{ArchivedSbqNode, SbqNode},
    quantize::SbqQuantizer,
    SbqMeans, SbqNodeDistanceMeasure, SbqSearchDistanceMeasure,
};

pub struct SbqSpeedupStorage<'a> {
    pub index: &'a PgRelation,
    pub distance_fn: DistanceFn,
    quantizer: SbqQuantizer,
    heap_rel: &'a PgRelation,
    heap_attr: AttrNumber,
    qv_cache: Option<RefCell<QuantizedVectorCache>>,
    has_labels: bool,
    num_neighbors: u32,
}

impl<'a> SbqSpeedupStorage<'a> {
    fn sbq_vec_len(meta_page: &MetaPage) -> usize {
        meta_page.get_num_dimensions() as usize * meta_page.get_bq_num_bits_per_dimension() as usize
            / 64
    }

    pub fn new_for_build(
        index: &'a PgRelation,
        heap_rel: &'a PgRelation,
        meta_page: &MetaPage,
    ) -> Self {
        Self {
            index,
            distance_fn: meta_page.get_distance_function(),
            quantizer: SbqQuantizer::new(meta_page),
            heap_rel,
            heap_attr: get_index_vector_attribute(index),
            qv_cache: Some(RefCell::new(QuantizedVectorCache::new(
                QUANTIZED_VECTOR_CACHE_SIZE,
                Self::sbq_vec_len(meta_page),
                meta_page.get_num_neighbors() as usize,
            ))),
            has_labels: meta_page.has_labels(),
            num_neighbors: meta_page.get_num_neighbors(),
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
    ) -> Self {
        Self {
            index: index_relation,
            distance_fn: meta_page.get_distance_function(),
            quantizer: Self::load_quantizer(index_relation, meta_page, stats),
            heap_rel,
            heap_attr: get_index_vector_attribute(index_relation),
            qv_cache: Some(RefCell::new(QuantizedVectorCache::new(
                QUANTIZED_VECTOR_CACHE_SIZE,
                Self::sbq_vec_len(meta_page),
                meta_page.get_num_neighbors() as usize,
            ))),
            has_labels: meta_page.has_labels(),
            num_neighbors: meta_page.get_num_neighbors(),
        }
    }

    pub fn load_for_search(
        index_relation: &'a PgRelation,
        heap_relation: &'a PgRelation,
        quantizer: &SbqQuantizer,
        meta_page: &MetaPage,
    ) -> Self {
        Self {
            index: index_relation,
            distance_fn: meta_page.get_distance_function(),
            //OPT: get rid of clone
            quantizer: quantizer.clone(),
            heap_rel: heap_relation,
            heap_attr: get_index_vector_attribute(index_relation),
            qv_cache: None,
            has_labels: meta_page.has_labels(),
            num_neighbors: meta_page.get_num_neighbors(),
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

    fn visit_lsn_internal(
        &self,
        lsr: &mut ListSearchResult<
            <SbqSpeedupStorage<'a> as Storage>::QueryDistanceMeasure,
            <SbqSpeedupStorage<'a> as Storage>::LSNPrivateData,
        >,
        lsn_index_pointer: IndexPointer,
        gns: &mut GraphNeighborStore,
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
                let neighbors = b.get_neighbors_with_full_vector_distances(
                    lsn_index_pointer,
                    self,
                    &mut lsr.prune_stats,
                );
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

                    let mut cache = self.qv_cache.as_ref().unwrap().borrow_mut();
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

    pub fn cache(&self) -> Option<&RefCell<QuantizedVectorCache>> {
        self.qv_cache.as_ref()
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
        index_pointer: IndexPointer,
        neighbors: &[NeighborWithDistance],
        stats: &mut S,
    ) {
        let mut cache = self.qv_cache.as_ref().unwrap().borrow_mut();
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
        archived.set_neighbors(neighbors, self.num_neighbors);
        node.commit();
    }

    unsafe fn get_node_distance_measure<'b, S: StatsNodeRead + StatsNodeWrite + StatsNodeModify>(
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
        stats: &mut S,
    ) -> Vec<NeighborWithDistance> {
        let rn = unsafe { SbqNode::read(self.index, neighbors_of, self.has_labels, stats) };
        let archived = rn.get_archived_node();
        let q = archived.get_bq_vector();

        rn.get_archived_node()
            .iter_neighbors()
            .map(|n| {
                //OPT: we can optimize this if num_dimensions_for_neighbors == num_dimensions_to_index
                let rn1 = unsafe { SbqNode::read(self.index, n, self.has_labels, stats) };
                let arch = rn1.get_archived_node();
                stats.record_quantized_distance_comparison();
                let dist = distance_xor_optimized(q, arch.get_bq_vector());
                NeighborWithDistance::new(
                    n,
                    DistanceWithTieBreak::new(dist as f32, neighbors_of, n),
                    arch.get_labels().map(Into::into),
                )
            })
            .collect()
    }

    /// The distance comparison implementations differ between get_lsn and visit_lsn
    /// because they handle vectors from different sources:
    ///
    /// - get_lsn (create_lsn_for_start_node): Gets vectors directly from the node
    ///   being read, used for the initial search node
    ///
    /// - visit_lsn: Handles vectors differently based on storage location:
    ///   - For disk storage: Gets vectors directly from nodes
    ///   - For builder storage: Uses a cache to access quantized vectors
    fn create_lsn_for_start_node(
        &self,
        lsr: &mut ListSearchResult<Self::QueryDistanceMeasure, Self::LSNPrivateData>,
        index_pointer: ItemPointer,
        gns: &mut GraphNeighborStore,
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
        gns: &mut GraphNeighborStore,
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
        index_pointer: IndexPointer,
        neighbors: &[NeighborWithDistance],
        stats: &mut S,
    ) {
        let mut cache = self.cache().as_ref().unwrap().borrow_mut();

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
        archived.set_neighbors(neighbors, self.num_neighbors);
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
