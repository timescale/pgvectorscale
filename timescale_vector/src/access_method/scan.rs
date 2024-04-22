use std::collections::BinaryHeap;

use pgrx::{pg_sys::InvalidOffsetNumber, *};

use crate::{
    access_method::{
        bq::BqSpeedupStorage, graph_neighbor_store::GraphNeighborStore, meta_page::MetaPage,
        pg_vector::PgVector,
    },
    util::{buffer::PinnedBufferShare, HeapPointer, IndexPointer},
};

use super::{
    bq::{BqMeans, BqQuantizer, BqSearchDistanceMeasure, BqSpeedupStorageLsnPrivateData},
    graph::{Graph, ListSearchResult},
    plain_storage::{PlainDistanceMeasure, PlainStorage, PlainStorageLsnPrivateData},
    stats::QuantizerStats,
    storage::{Storage, StorageType},
};

/* Be very careful not to transfer PgRelations in the state, as they can change between calls. That means we shouldn't be
using lifetimes here. Everything should be owned */
enum StorageState {
    BqSpeedup(
        BqQuantizer,
        TSVResponseIterator<BqSearchDistanceMeasure, BqSpeedupStorageLsnPrivateData>,
    ),
    Plain(TSVResponseIterator<PlainDistanceMeasure, PlainStorageLsnPrivateData>),
}

/* no lifetime usage here. */
struct TSVScanState {
    storage: *mut StorageState,
    distance_fn: Option<fn(&[f32], &[f32]) -> f32>,
    meta_page: MetaPage,
    last_buffer: Option<PinnedBufferShare>,
}

impl TSVScanState {
    fn new(meta_page: MetaPage) -> Self {
        Self {
            storage: std::ptr::null_mut(),
            distance_fn: None,
            meta_page: meta_page,
            last_buffer: None,
        }
    }

    fn initialize(
        &mut self,
        index: &PgRelation,
        heap: &PgRelation,
        query: PgVector,
        search_list_size: usize,
    ) {
        let meta_page = MetaPage::fetch(&index);
        let storage = meta_page.get_storage_type();
        let distance = meta_page.get_distance_function();

        let store_type = match storage {
            StorageType::Plain => {
                let stats = QuantizerStats::new();
                let bq =
                    PlainStorage::load_for_search(index, heap, meta_page.get_distance_function());
                let it =
                    TSVResponseIterator::new(&bq, index, query, search_list_size, meta_page, stats);
                StorageState::Plain(it)
            }
            StorageType::BqSpeedup => {
                let mut stats = QuantizerStats::new();
                let quantizer = unsafe { BqMeans::load(index, &meta_page, &mut stats) };
                let bq = BqSpeedupStorage::load_for_search(
                    index,
                    heap,
                    &quantizer,
                    meta_page.get_distance_function(),
                );
                let it =
                    TSVResponseIterator::new(&bq, index, query, search_list_size, meta_page, stats);
                StorageState::BqSpeedup(quantizer, it)
            }
        };

        self.storage = PgMemoryContexts::CurrentMemoryContext.leak_and_drop_on_delete(store_type);
        self.distance_fn = Some(distance);
    }
}

struct ResortData {
    heap_pointer: HeapPointer,
    index_pointer: IndexPointer,
    exact_distance: f32,
    approx_distance: f32,
}

impl PartialEq for ResortData {
    fn eq(&self, other: &Self) -> bool {
        self.heap_pointer == other.heap_pointer
    }
}

impl PartialOrd for ResortData {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        //notice the reverse here. Other is the one that is being compared to self
        //this allows us to have a min heap
        other.exact_distance.partial_cmp(&self.exact_distance)
    }
}

impl Eq for ResortData {}

impl Ord for ResortData {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

struct TSVResponseIterator<QDM, PD> {
    lsr: ListSearchResult<QDM, PD>,
    search_list_size: usize,
    meta_page: MetaPage,
    quantizer_stats: QuantizerStats,
    resort_size: usize,
    resort_buffer: BinaryHeap<ResortData>,
    max_approx_distance_in_resort_buffer: f32,
}

impl<QDM, PD> TSVResponseIterator<QDM, PD> {
    fn new<S: Storage<QueryDistanceMeasure = QDM, LSNPrivateData = PD>>(
        storage: &S,
        index: &PgRelation,
        query: PgVector,
        search_list_size: usize,
        //FIXME?
        _meta_page: MetaPage,
        quantizer_stats: QuantizerStats,
    ) -> Self {
        let mut meta_page = MetaPage::fetch(&index);
        let graph = Graph::new(GraphNeighborStore::Disk, &mut meta_page);

        let lsr = graph.greedy_search_streaming_init(query, search_list_size, storage);
        let resort_size = super::guc::TSV_RESORT_SIZE.get() as usize;

        Self {
            search_list_size,
            lsr,
            meta_page,
            quantizer_stats,
            resort_size,
            resort_buffer: BinaryHeap::with_capacity(resort_size),
            max_approx_distance_in_resort_buffer: 0.0,
        }
    }
}

impl<QDM, PD> TSVResponseIterator<QDM, PD> {
    fn next<S: Storage<QueryDistanceMeasure = QDM, LSNPrivateData = PD>>(
        &mut self,
        storage: &S,
    ) -> Option<(HeapPointer, IndexPointer, f32)> {
        let graph = Graph::new(GraphNeighborStore::Disk, &mut self.meta_page);

        /* Iterate until we find a non-deleted tuple */
        loop {
            graph.greedy_search_iterate(&mut self.lsr, self.search_list_size, None, storage);

            let item = self.lsr.consume(storage);

            match item {
                Some((heap_pointer, index_pointer, approx_distance)) => {
                    if heap_pointer.offset == InvalidOffsetNumber {
                        /* deleted tuple */
                        continue;
                    }
                    return Some((heap_pointer, index_pointer, approx_distance));
                }
                None => {
                    return None;
                }
            }
        }
    }

    fn next_with_resort<S: Storage<QueryDistanceMeasure = QDM, LSNPrivateData = PD>>(
        &mut self,
        index: &PgRelation,
        storage: &S,
    ) -> Option<(HeapPointer, IndexPointer)> {
        if self.resort_buffer.capacity() == 0 {
            return self
                .next(storage)
                .map(|(heap_pointer, index_pointer, _)| (heap_pointer, index_pointer));
        }

        while self.resort_buffer.len() < 2
            || self.max_approx_distance_in_resort_buffer
                - self.resort_buffer.peek().unwrap().approx_distance
                < self.resort_size as f32
        {
            match self.next(storage) {
                Some((heap_pointer, index_pointer, approx_distance)) => {
                    let distance = storage.get_full_distance_for_resort(
                        self.lsr.sdm.as_ref().unwrap(),
                        index_pointer,
                        heap_pointer,
                        &self.meta_page,
                        &mut self.lsr.stats,
                    );

                    self.resort_buffer.push(ResortData {
                        heap_pointer,
                        index_pointer,
                        exact_distance: distance,
                        approx_distance,
                    });
                    self.max_approx_distance_in_resort_buffer = self
                        .max_approx_distance_in_resort_buffer
                        .max(approx_distance);
                }
                None => {
                    break;
                }
            }
        }

        match self.resort_buffer.pop() {
            Some(rd) => Some((rd.heap_pointer, rd.index_pointer)),
            None => None,
        }
    }
}

/*
struct TSVScanState<'a, 'b> {
    iterator: *mut TSVResponseIterator<'a, 'b>,
}
*/
#[pg_guard]
pub extern "C" fn ambeginscan(
    index_relation: pg_sys::Relation,
    nkeys: ::std::os::raw::c_int,
    norderbys: ::std::os::raw::c_int,
) -> pg_sys::IndexScanDesc {
    let mut scandesc: PgBox<pg_sys::IndexScanDescData> = unsafe {
        PgBox::from_pg(pg_sys::RelationGetIndexScan(
            index_relation,
            nkeys,
            norderbys,
        ))
    };
    let indexrel = unsafe { PgRelation::from_pg(index_relation) };
    let meta_page = MetaPage::fetch(&indexrel);

    let state: TSVScanState = TSVScanState::new(meta_page);
    scandesc.opaque =
        PgMemoryContexts::CurrentMemoryContext.leak_and_drop_on_delete(state) as void_mut_ptr;

    scandesc.into_pg()
}

#[pg_guard]
pub extern "C" fn amrescan(
    scan: pg_sys::IndexScanDesc,
    _keys: pg_sys::ScanKey,
    nkeys: ::std::os::raw::c_int,
    orderbys: pg_sys::ScanKey,
    norderbys: ::std::os::raw::c_int,
) {
    if norderbys == 0 {
        panic!("No order by keys provided");
    }
    if norderbys > 1 {
        panic!("Too many order by provided");
    }
    let mut scan: PgBox<pg_sys::IndexScanDescData> = unsafe { PgBox::from_pg(scan) };
    let indexrel = unsafe { PgRelation::from_pg(scan.indexRelation) };
    let heaprel = unsafe { PgRelation::from_pg(scan.heapRelation) };

    if nkeys > 0 {
        scan.xs_recheck = true;
    }

    let orderby_keys = unsafe {
        std::slice::from_raw_parts(orderbys as *const pg_sys::ScanKeyData, norderbys as _)
    };

    let search_list_size = super::guc::TSV_QUERY_SEARCH_LIST_SIZE.get() as usize;

    let state = unsafe { (scan.opaque as *mut TSVScanState).as_mut() }.expect("no scandesc state");

    let query = unsafe {
        PgVector::from_datum(
            orderby_keys[0].sk_argument,
            &state.meta_page,
            true, /* needed for search */
            true, /* needed for resort */
        )
    };
    state.initialize(&indexrel, &heaprel, query, search_list_size);
}

#[pg_guard]
pub extern "C" fn amgettuple(
    scan: pg_sys::IndexScanDesc,
    _direction: pg_sys::ScanDirection,
) -> bool {
    let scan: PgBox<pg_sys::IndexScanDescData> = unsafe { PgBox::from_pg(scan) };
    let state = unsafe { (scan.opaque as *mut TSVScanState).as_mut() }.expect("no scandesc state");
    //let iter = unsafe { state.iterator.as_mut() }.expect("no iterator in state");

    let indexrel = unsafe { PgRelation::from_pg(scan.indexRelation) };
    let heaprel = unsafe { PgRelation::from_pg(scan.heapRelation) };

    let mut storage = unsafe { state.storage.as_mut() }.expect("no storage in state");
    match &mut storage {
        StorageState::BqSpeedup(quantizer, iter) => {
            let bq = BqSpeedupStorage::load_for_search(
                &indexrel,
                &heaprel,
                quantizer,
                state.distance_fn.unwrap(),
            );
            let next = iter.next_with_resort(&indexrel, &bq);
            get_tuple(state, next, scan)
        }
        StorageState::Plain(iter) => {
            let storage =
                PlainStorage::load_for_search(&indexrel, &heaprel, state.distance_fn.unwrap());
            let next = if state.meta_page.get_num_dimensions()
                == state.meta_page.get_num_dimensions_to_index()
            {
                /* no need to resort */
                iter.next(&storage).map(|(hp, ip, _)| (hp, ip))
            } else {
                iter.next_with_resort(&indexrel, &storage)
            };
            get_tuple(state, next, scan)
        }
    }
}

fn get_tuple(
    state: &mut TSVScanState,
    next: Option<(HeapPointer, IndexPointer)>,
    mut scan: PgBox<pg_sys::IndexScanDescData>,
) -> bool {
    scan.xs_recheckorderby = false;
    match next {
        Some((heap_pointer, index_pointer)) => {
            let tid_to_set = &mut scan.xs_heaptid;
            heap_pointer.to_item_pointer_data(tid_to_set);

            /*
             * An index scan must maintain a pin on the index page holding the
             * item last returned by amgettuple
             *
             * https://www.postgresql.org/docs/current/index-locking.html
             */
            let indexrel = unsafe { PgRelation::from_pg(scan.indexRelation) };
            state.last_buffer = Some(PinnedBufferShare::read(
                &indexrel,
                index_pointer.block_number,
            ));
            true
        }
        None => {
            state.last_buffer = None;
            false
        }
    }
}

#[pg_guard]
pub extern "C" fn amendscan(scan: pg_sys::IndexScanDesc) {
    let min_level = unsafe {
        let l = pg_sys::log_min_messages;
        let c = pg_sys::client_min_messages;
        std::cmp::min(l, c)
    };
    if min_level <= pg_sys::DEBUG1 as _ {
        let scan: PgBox<pg_sys::IndexScanDescData> = unsafe { PgBox::from_pg(scan) };
        let state =
            unsafe { (scan.opaque as *mut TSVScanState).as_mut() }.expect("no scandesc state");

        let mut storage = unsafe { state.storage.as_mut() }.expect("no storage in state");
        match &mut storage {
            StorageState::BqSpeedup(_bq, iter) => end_scan::<BqSpeedupStorage>(iter),
            StorageState::Plain(iter) => end_scan::<PlainStorage>(iter),
        }
    }
}

fn end_scan<S: Storage>(
    iter: &mut TSVResponseIterator<S::QueryDistanceMeasure, S::LSNPrivateData>,
) {
    debug1!(
        "Query stats - node reads:{}, calls: {}, total distance comparisons: {}, quantized distance comparisons: {}, quantizer r/w: {}/{}",
        iter.lsr.stats.get_node_reads(),
        iter.lsr.stats.get_calls(),
        iter.lsr.stats.get_total_distance_comparisons(),
        iter.lsr.stats.get_quantized_distance_comparisons(),
        iter.quantizer_stats.node_reads,
        iter.quantizer_stats.node_writes,
    );
}
