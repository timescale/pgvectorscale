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
}

impl TSVScanState {
    fn new() -> Self {
        Self {
            storage: std::ptr::null_mut(),
            distance_fn: None,
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
                let bq = PlainStorage::load_for_search(index, meta_page.get_distance_function());
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
    distance: f32,
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
        other.distance.partial_cmp(&self.distance)
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
    current: usize,
    last_buffer: Option<PinnedBufferShare>,
    meta_page: MetaPage,
    quantizer_stats: QuantizerStats,
    resort_buffer: BinaryHeap<ResortData>,
}

impl<QDM, PD> TSVResponseIterator<QDM, PD> {
    fn new<S: Storage<QueryDistanceMeasure = QDM, LSNPrivateData = PD>>(
        storage: &S,
        index: &PgRelation,
        query: PgVector,
        search_list_size: usize,
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
            current: 0,
            last_buffer: None,
            meta_page,
            quantizer_stats,
            resort_buffer: BinaryHeap::with_capacity(resort_size),
        }
    }
}

impl<QDM, PD> TSVResponseIterator<QDM, PD> {
    fn next<S: Storage<QueryDistanceMeasure = QDM, LSNPrivateData = PD>>(
        &mut self,
        index: &PgRelation,
        storage: &S,
    ) -> Option<(HeapPointer, IndexPointer)> {
        let graph = Graph::new(GraphNeighborStore::Disk, &mut self.meta_page);

        /* Iterate until we find a non-deleted tuple */
        loop {
            graph.greedy_search_iterate(&mut self.lsr, self.search_list_size, None, storage);

            let item = self.lsr.consume(storage);

            match item {
                Some((heap_pointer, index_pointer)) => {
                    /*
                     * An index scan must maintain a pin on the index page holding the
                     * item last returned by amgettuple
                     *
                     * https://www.postgresql.org/docs/current/index-locking.html
                     */
                    self.last_buffer =
                        Some(PinnedBufferShare::read(index, index_pointer.block_number));

                    self.current = self.current + 1;
                    if heap_pointer.offset == InvalidOffsetNumber {
                        /* deleted tuple */
                        continue;
                    }
                    return Some((heap_pointer, index_pointer));
                }
                None => {
                    self.last_buffer = None;
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
            return self.next(index, storage);
        }

        while self.resort_buffer.len() < self.resort_buffer.capacity() {
            match self.next(index, storage) {
                Some((heap_pointer, index_pointer)) => {
                    let distance = storage.get_full_distance_for_resort(
                        self.lsr.sdm.as_ref().unwrap(),
                        index_pointer,
                        heap_pointer,
                        &mut self.lsr.stats,
                    );

                    self.resort_buffer.push(ResortData {
                        heap_pointer,
                        index_pointer,
                        distance,
                    });
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

    let state: TSVScanState = TSVScanState::new();
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
    let meta_page = MetaPage::fetch(&indexrel);
    let _storage = meta_page.get_storage_type();

    if nkeys > 0 {
        scan.xs_recheck = true;
    }

    let orderby_keys = unsafe {
        std::slice::from_raw_parts(orderbys as *const pg_sys::ScanKeyData, norderbys as _)
    };
    let query = unsafe { PgVector::from_datum(orderby_keys[0].sk_argument) };

    //TODO need to set search_list_size correctly
    //TODO right now doesn't handle more than LIMIT 100;
    let search_list_size = super::guc::TSV_QUERY_SEARCH_LIST_SIZE.get() as usize;

    let state = unsafe { (scan.opaque as *mut TSVScanState).as_mut() }.expect("no scandesc state");
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
            get_tuple(next, scan)
        }
        StorageState::Plain(iter) => {
            let storage = PlainStorage::load_for_search(&indexrel, state.distance_fn.unwrap());
            let next = iter.next(&indexrel, &storage);
            get_tuple(next, scan)
        }
    }
}

fn get_tuple(
    next: Option<(HeapPointer, IndexPointer)>,
    mut scan: PgBox<pg_sys::IndexScanDescData>,
) -> bool {
    scan.xs_recheckorderby = false;
    match next {
        Some((heap_pointer, _)) => {
            let tid_to_set = &mut scan.xs_heaptid;
            heap_pointer.to_item_pointer_data(tid_to_set);
            true
        }
        None => false,
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
