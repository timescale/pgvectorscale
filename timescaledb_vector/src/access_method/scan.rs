use pgrx::{pg_sys::InvalidOffsetNumber, *};

use crate::{
    access_method::{disk_index_graph::DiskIndexGraph, model::PgVector},
    util::{buffer::LockedBufferShare, HeapPointer},
};

use super::graph::ListSearchResult;

struct TSVResponseIterator {
    query: Vec<f32>,
    lsr: ListSearchResult,
    search_list_size: usize,
    current: usize,
    last_buffer: Option<LockedBufferShare>,
}

impl TSVResponseIterator {
    fn new(index: &PgRelation, query: &[f32], search_list_size: usize) -> Self {
        let mut graph = DiskIndexGraph::new(&index);
        use super::graph::Graph;
        let lsr = graph.greedy_search_streaming_init(&index, query);
        Self {
            query: query.to_vec(),
            search_list_size,
            lsr,
            current: 0,
            last_buffer: None,
        }
    }
}

impl TSVResponseIterator {
    fn next(&mut self, index: &PgRelation) -> Option<HeapPointer> {
        let mut graph = DiskIndexGraph::new(&index);
        use super::graph::Graph;

        /* Iterate until we find a non-deleted tuple */
        loop {
            graph.greedy_search_iterate(&mut self.lsr, index, &self.query, self.search_list_size);

            let item = self.lsr.consume();

            match item {
                Some((heap_pointer, index_pointer)) => {
                    /*
                     * An index scan must maintain a pin on the index page holding the
                     * item last returned by amgettuple
                     *
                     * https://www.postgresql.org/docs/current/index-locking.html
                     */
                    self.last_buffer = unsafe {
                        Some(LockedBufferShare::read(
                            index.as_ptr(),
                            index_pointer.block_number,
                        ))
                    };

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

struct TSVScanState {
    iterator: *mut TSVResponseIterator,
}

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
    let state = TSVScanState {
        iterator: std::ptr::null_mut(),
    };

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
    let state = unsafe { (scan.opaque as *mut TSVScanState).as_mut() }.expect("no scandesc state");

    if nkeys > 0 {
        scan.xs_recheck = true;
    }

    let orderby_keys = unsafe {
        std::slice::from_raw_parts(orderbys as *const pg_sys::ScanKeyData, norderbys as _)
    };
    let vec = unsafe { PgVector::from_datum(orderby_keys[0].sk_argument) };
    let query = unsafe { (*vec).to_slice() };

    //TODO need to set search_list_size correctly
    //TODO right now doesn't handle more than LIMIT 100;
    let search_list_size = super::guc::TSV_QUERY_SEARCH_LIST_SIZE.get() as usize;

    let res = TSVResponseIterator::new(&indexrel, query, search_list_size);

    state.iterator = PgMemoryContexts::CurrentMemoryContext.leak_and_drop_on_delete(res);
}

#[pg_guard]
pub extern "C" fn amgettuple(
    scan: pg_sys::IndexScanDesc,
    _direction: pg_sys::ScanDirection,
) -> bool {
    let mut scan: PgBox<pg_sys::IndexScanDescData> = unsafe { PgBox::from_pg(scan) };
    let state = unsafe { (scan.opaque as *mut TSVScanState).as_mut() }.expect("no scandesc state");
    let iter = unsafe { state.iterator.as_mut() }.expect("no iterator in state");

    let indexrel = unsafe { PgRelation::from_pg(scan.indexRelation) };

    /* no need to recheck stuff for now */
    scan.xs_recheckorderby = false;
    match iter.next(&indexrel) {
        Some(heap_pointer) => {
            let tid_to_set = &mut scan.xs_heaptid;
            heap_pointer.to_item_pointer_data(tid_to_set);
            true
        }
        None => false,
    }
}

#[pg_guard]
pub extern "C" fn amendscan(_scan: pg_sys::IndexScanDesc) {
    // nothing to do here
}

#[cfg(any(test, feature = "pg_test"))]
#[pgrx::pg_schema]
mod tests {
    use pgrx::*;

    #[pg_test]
    unsafe fn test_index_scan() -> spi::Result<()> {
        Spi::run(&format!(
            "CREATE TABLE test(embedding vector(3));

        INSERT INTO test(embedding) VALUES ('[1,2,3]'), ('[4,5,6]'), ('[7,8,10]');

        INSERT INTO test(embedding) SELECT ('[' || g::text ||', 0, 0]')::vector FROM generate_series(0, 100) g;

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
