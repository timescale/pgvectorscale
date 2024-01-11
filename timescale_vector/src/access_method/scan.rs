use pgrx::{pg_sys::InvalidOffsetNumber, *};

use crate::{
    access_method::{
        bq::BqStorage, graph_neighbor_store::GraphNeighborStore, meta_page::MetaPage,
        pg_vector::PgVector,
    },
    util::{buffer::PinnedBufferShare, HeapPointer},
};

use super::{
    graph::{Graph, ListSearchResult},
    stats::QuantizerStats,
    storage::{Storage, StorageType},
};

enum StorageState<'a, 'b> {
    BQ(BqStorage<'a>, TSVResponseIterator<'b, BqStorage<'a>>),
}

struct TSVScanState<'a, 'b> {
    storage: *mut StorageState<'a, 'b>,
}

impl<'a, 'b> TSVScanState<'a, 'b> {
    fn new() -> Self {
        Self {
            storage: std::ptr::null_mut(),
        }
    }

    fn initialize(&mut self, index: &PgRelation, query: PgVector, search_list_size: usize) {
        let meta_page = MetaPage::read(&index);
        let storage = meta_page.get_storage_type();

        let store_type = match storage {
            StorageType::None => {
                pgrx::error!("not implemented");
            }
            StorageType::PQ => {
                //pq.load(index, &meta_page);
                pgrx::error!("not implemented");
            }
            StorageType::BQ => {
                let mut stats = QuantizerStats::new();
                let bq = BqStorage::load_for_search(index, &meta_page, &mut stats);
                let it =
                    TSVResponseIterator::new(&bq, index, query, search_list_size, meta_page, stats);
                StorageState::BQ(bq, it)
            }
        };

        self.storage = PgMemoryContexts::CurrentMemoryContext.leak_and_drop_on_delete(store_type);
    }
}

struct TSVResponseIterator<'a, S: Storage> {
    lsr: ListSearchResult<S>,
    search_list_size: usize,
    current: usize,
    last_buffer: Option<PinnedBufferShare<'a>>,
    meta_page: MetaPage,
}

impl<'a, S: Storage> TSVResponseIterator<'a, S> {
    fn new(
        storage: &S,
        index: &PgRelation,
        query: PgVector,
        search_list_size: usize,
        _meta_page: MetaPage,
        stats: QuantizerStats,
    ) -> Self {
        let mut meta_page = MetaPage::read(&index);
        let graph = Graph::new(GraphNeighborStore::Disk, &mut meta_page);

        let mut lsr = graph.greedy_search_streaming_init(&index, query, search_list_size, storage);
        lsr.stats.set_quantizer_stats(stats);

        Self {
            search_list_size,
            lsr,
            current: 0,
            last_buffer: None,
            meta_page,
        }
    }
}

impl<'a, S: Storage> TSVResponseIterator<'a, S> {
    fn next(&mut self, index: &'a PgRelation, storage: &S) -> Option<HeapPointer> {
        let graph = Graph::new(GraphNeighborStore::Disk, &mut self.meta_page);

        /* Iterate until we find a non-deleted tuple */
        loop {
            graph.greedy_search_iterate(&mut self.lsr, index, self.search_list_size, None, storage);

            let item = self.lsr.consume(index, storage);

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
                    return Some(heap_pointer);
                }
                None => {
                    self.last_buffer = None;
                    return None;
                }
            }
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
    let meta_page = MetaPage::read(&indexrel);
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
    state.initialize(&indexrel, query, search_list_size);
    /*match &mut storage {
        Storage::None => pgrx::error!("not implemented"),
        Storage::PQ(_pq) => pgrx::error!("not implemented"),
        Storage::BQ(_bq) => {
            let state =
                unsafe { (scan.opaque as *mut TSVScanState).as_mut() }.expect("no scandesc state");

            let res = TSVResponseIterator::new(&indexrel, query, search_list_size);
            state.iterator = PgMemoryContexts::CurrentMemoryContext.leak_and_drop_on_delete(res);
        }
    }*/
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

    let mut storage = unsafe { state.storage.as_mut() }.expect("no storage in state");
    match &mut storage {
        StorageState::BQ(bq, iter) => get_tuple(bq, &indexrel, iter, scan),
    }
}

fn get_tuple<'a, S: Storage>(
    storage: &S,
    index: &'a PgRelation,
    iter: &'a mut TSVResponseIterator<'a, S>,
    mut scan: PgBox<pg_sys::IndexScanDescData>,
) -> bool {
    scan.xs_recheckorderby = false;
    match iter.next(&index, storage) {
        Some(heap_pointer) => {
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
            StorageState::BQ(_bq, iter) => end_scan(iter),
        }
    }
}

fn end_scan<S: Storage>(iter: &mut TSVResponseIterator<S>) {
    debug1!(
        "Query stats - node reads:{}, calls: {}, total distance comparisons: {}, quantized distance comparisons: {}",
        iter.lsr.stats.get_node_reads(),
        iter.lsr.stats.get_calls(),
        iter.lsr.stats.get_total_distance_comparisons(),
        iter.lsr.stats.get_quantized_distance_comparisons(),
    );
}

#[cfg(any(test, feature = "pg_test"))]
#[pgrx::pg_schema]
mod tests {
    use pgrx::*;

    //TODO: add test where inserting and querying with vectors that are all the same.

    #[pg_test]
    unsafe fn test_index_scan() -> spi::Result<()> {
        Spi::run(&format!(
            "CREATE TABLE test(embedding vector(3));

        INSERT INTO test(embedding) VALUES ('[1,2,3]'), ('[4,5,6]'), ('[7,8,10]');

        INSERT INTO test(embedding) SELECT ('[' || g::text ||', 1.0, 1.0]')::vector FROM generate_series(0, 100) g;

        CREATE INDEX idxtest
              ON test
           USING tsv(embedding)
            WITH (num_neighbors=30);"
        ))?;

        Spi::run(&format!(
            "
        set enable_seqscan = 0;
        select * from test order by embedding <=> '[0,0,0]';
        explain analyze select * from test order by embedding <=> '[0,0,0]';
        ",
        ))?;

        Spi::run(&format!(
            "
        set enable_seqscan = 0;
        set tsv.query_search_list_size = 2;
        select * from test order by embedding <=> '[0,0,0]';
        ",
        ))?;

        let res: Option<i64> = Spi::get_one(&format!(
            "
        set enable_seqscan = 0;
        set tsv.query_search_list_size = 2;
        WITH cte as (select * from test order by embedding <=> '[0,0,0]') SELECT count(*) from cte;
        ",
        ))?;

        assert_eq!(104, res.unwrap(), "Testing query over entire table");

        Spi::run(&format!(
            "
        drop index idxtest;
        ",
        ))?;

        Ok(())
    }

    #[pg_test]
    unsafe fn test_index_scan_on_empty_table() -> spi::Result<()> {
        Spi::run(&format!(
            "CREATE TABLE test(embedding vector(3));

        CREATE INDEX idxtest
              ON test
           USING tsv(embedding)
            WITH (num_neighbors=30);"
        ))?;

        Spi::run(&format!(
            "
        set enable_seqscan = 0;
        select * from test order by embedding <=> '[0,0,0]';
        explain analyze select * from test order by embedding <=> '[0,0,0]';
        ",
        ))?;

        Spi::run(&format!(
            "
        drop index idxtest;
        ",
        ))?;

        Ok(())
    }
}
