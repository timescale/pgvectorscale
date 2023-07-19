use pgrx::*;
use rkyv::ArchiveUnsized;

use crate::{
    access_method::{disk_index_graph::DiskIndexGraph, model::PgVector},
    util::{buffer::LockedBufferShare, ItemPointer},
};

use super::graph::ListSearchResult;

struct TSVResponseIterator {
    index: PgRelation,
    lsr: ListSearchResult,
    current: usize,
    last_buffer: Option<LockedBufferShare>,
}

impl TSVResponseIterator {
    fn new(index: PgRelation, lsr: ListSearchResult) -> Self {
        Self {
            index: index,
            lsr: lsr,
            current: 0,
            last_buffer: None,
        }
    }
}

impl Iterator for TSVResponseIterator {
    type Item = ItemPointer;

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.lsr.get_closest_heap_pointer(self.current);
        match item {
            Some(heap_pointer) => {
                let index_pointer = self.lsr.get_closets_index_pointer(self.current).unwrap();
                /*
                 * An index scan must maintain a pin on the index page holding the
                 * item last returned by amgettuple
                 *
                 * https://www.postgresql.org/docs/current/index-locking.html
                 */
                self.last_buffer = unsafe {
                    Some(LockedBufferShare::read(
                        self.index.as_ptr(),
                        index_pointer.block_number,
                    ))
                };

                self.current = self.current + 1;
                Some(heap_pointer)
            }
            None => None,
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
    let mut state =
        unsafe { (scan.opaque as *mut TSVScanState).as_mut() }.expect("no scandesc state");

    if nkeys > 0 {
        scan.xs_recheck = true;
    }

    let orderby_keys = unsafe {
        std::slice::from_raw_parts(orderbys as *const pg_sys::ScanKeyData, norderbys as _)
    };
    let vec = unsafe { PgVector::from_datum(orderby_keys[0].sk_argument) };
    let query = unsafe { (*vec).to_slice() };

    //TODO: use real init id
    let mut graph = DiskIndexGraph::new(&indexrel, vec![ItemPointer::new(1, 1)]);

    //TODO need to set search_list_size correctly
    //TODO right now doesn't handle more than LIMIT 100;
    let search_list_size = super::guc::TSV_QUERY_SEARCH_LIST_SIZE.get() as usize;
    use super::graph::Graph;
    let (lsr, _) = graph.greedy_search(&indexrel, query, search_list_size);
    let res = TSVResponseIterator::new(indexrel, lsr);

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

    /* no need to recheck stuff for now */
    scan.xs_recheckorderby = false;
    match iter.next() {
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
