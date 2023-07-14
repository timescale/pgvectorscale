use pgrx::pg_sys::{BufferGetBlockNumber, Pointer};
use pgrx::*;

use crate::access_method::model::PgVector;
use crate::util::page;
use crate::util::tape::Tape;
use crate::util::*;

use super::builder_graph::BuilderGraph;
use super::model::{self};

const TSV_MAGIC_NUMBER: u32 = 768756476; //Magic number, random
const TSV_VERSION: u32 = 1;

/// This is metadata about the entire index.
/// Stored as the first page in the index relation.
#[derive(Clone)]
pub struct TsvMetaPage {
    /// random magic number for identifying the index
    magic_number: u32,
    /// version number for future-proofing
    version: u32,
    /// number of dimensions in the vector
    pub num_dimensions: u32,
    /// max number of outgoing edges a node in the graph can have (R in the papers)
    pub num_neighbors: u32,
}

struct BuildState {
    memcxt: PgMemoryContexts,
    meta_page: TsvMetaPage,
    ntuples: usize,
    tape: Tape,
    node_builder: BuilderGraph,
}

impl BuildState {
    fn new(index_relation: &PgRelation, meta_page: TsvMetaPage) -> Self {
        let tape = unsafe { Tape::new((**index_relation).as_ptr()) };
        //TODO: some ways to get rid of meta_page.clone?
        BuildState {
            memcxt: PgMemoryContexts::new("tsv build context"),
            ntuples: 0,
            meta_page: meta_page.clone(),
            tape: tape,
            node_builder: BuilderGraph::new(meta_page),
        }
    }
}

/// Returns the TsvMetaPage from a page.
/// Should only be called from the very first page in a relation.
unsafe fn page_get_meta(page: pg_sys::Page, buffer: pg_sys::Buffer, new: bool) -> *mut TsvMetaPage {
    assert_eq!(BufferGetBlockNumber(buffer), 0);
    let meta_page = ports::PageGetContents(page) as *mut TsvMetaPage;
    if !new {
        assert_eq!((*meta_page).magic_number, TSV_MAGIC_NUMBER);
    }
    meta_page
}

/// Write out a new meta page.
/// Has to be done as the first write to a new relation.
unsafe fn write_meta_page(
    index: pg_sys::Relation,
    num_dimensions: u32,
    num_neighbors: u32,
) -> TsvMetaPage {
    let page = page::WritablePage::new(index);
    let meta = page_get_meta(*page, *(*(page.get_buffer())), true);
    (*meta).magic_number = TSV_MAGIC_NUMBER;
    (*meta).version = TSV_VERSION;
    (*meta).num_dimensions = num_dimensions;
    (*meta).num_neighbors = num_neighbors;
    let header = page.cast::<pgrx::pg_sys::PageHeaderData>();

    let meta_end = (meta as Pointer).add(std::mem::size_of::<TsvMetaPage>());
    let page_start = (*page) as Pointer;
    (*header).pd_lower = meta_end.offset_from(page_start) as _;

    let mp = (*meta).clone();
    page.commit();
    mp
}

pub unsafe fn read_meta_page(index: &PgRelation) -> TsvMetaPage {
    let page = page::ReadablePage::read(index.as_ptr(), 0);
    let meta = page_get_meta(*page, *(*(page.get_buffer())), false);
    (*meta).clone()
}

#[pg_guard]
pub extern "C" fn ambuild(
    heaprel: pg_sys::Relation,
    indexrel: pg_sys::Relation,
    index_info: *mut pg_sys::IndexInfo,
) -> *mut pg_sys::IndexBuildResult {
    let heap_relation = unsafe { PgRelation::from_pg(heaprel) };
    let index_relation = unsafe { PgRelation::from_pg(indexrel) };

    let dimensions = index_relation.tuple_desc().get(0).unwrap().atttypmod;
    assert!(dimensions > 0 && dimensions < 2000);
    let meta_page = unsafe { write_meta_page(indexrel, dimensions as _, 50) };

    let ntuples = do_heap_scan(index_info, &heap_relation, &index_relation, meta_page);

    let mut result = unsafe { PgBox::<pg_sys::IndexBuildResult>::alloc0() };
    result.heap_tuples = ntuples as f64;
    result.index_tuples = ntuples as f64;

    result.into_pg()
}

#[pg_guard]
pub unsafe extern "C" fn aminsert(
    _index_relation: pg_sys::Relation,
    _values: *mut pg_sys::Datum,
    _isnull: *mut bool,
    _heap_tid: pg_sys::ItemPointer,
    _heap_relation: pg_sys::Relation,
    _check_unique: pg_sys::IndexUniqueCheck,
    _index_unchanged: bool,
    _index_info: *mut pg_sys::IndexInfo,
) -> bool {
    panic!("Not yet implemented")
}

#[pg_guard]
pub extern "C" fn ambuildempty(_index_relation: pg_sys::Relation) {
    panic!("Not yet implemented")
}

fn do_heap_scan<'a>(
    index_info: *mut pg_sys::IndexInfo,
    heap_relation: &'a PgRelation,
    index_relation: &'a PgRelation,
    meta_page: TsvMetaPage,
) -> usize {
    let mut state = BuildState::new(index_relation, meta_page);

    unsafe {
        pg_sys::IndexBuildHeapScan(
            heap_relation.as_ptr(),
            index_relation.as_ptr(),
            index_info,
            Some(build_callback),
            &mut state,
        );
    }

    unsafe { state.node_builder.write(index_relation) };
    /*print_graph_from_disk(
        index_relation,
        ItemPointer {
            block_number: 1,
            offset: 1,
        },
    );*/
    let ntuples = state.ntuples;

    warning!("Indexed {} tuples", ntuples);
    ntuples
}

#[pg_guard]
unsafe extern "C" fn build_callback(
    index: pg_sys::Relation,
    ctid: pg_sys::ItemPointer,
    values: *mut pg_sys::Datum,
    _isnull: *mut bool,
    _tuple_is_alive: bool,
    state: *mut std::os::raw::c_void,
) {
    let index_relation = unsafe { PgRelation::from_pg(index) };
    build_callback_internal(index_relation, *ctid, values, state);
}

#[inline(always)]
unsafe fn build_callback_internal(
    index: PgRelation,
    ctid: pg_sys::ItemPointerData,
    values: *mut pg_sys::Datum,
    state: *mut std::os::raw::c_void,
) {
    check_for_interrupts!();

    let state = (state as *mut BuildState).as_mut().unwrap();
    let mut old_context = state.memcxt.set_as_current();

    state.ntuples = state.ntuples + 1;

    //warning!("values {:?} {:?}", values, *values);
    //let vec = PgVector::from_datum(*values);
    let values = std::slice::from_raw_parts(values, 1);
    let vec = PgVector::from_datum(values[0]);

    let vector = (*vec).to_slice();
    let heap_pointer = ItemPointer::with_item_pointer_data(ctid);

    let node = model::Node::new(vector, heap_pointer, &state.meta_page);
    let index_pointer: IndexPointer = node.write(&mut state.tape);
    state.node_builder.insert(&index, index_pointer, vector);

    old_context.set_as_current();
    state.memcxt.reset();
}

#[cfg(any(test, feature = "pg_test"))]
#[pgrx::pg_schema]
mod tests {
    use pgrx::*;

    #[pg_test]
    unsafe fn test_index_creation() -> spi::Result<()> {
        Spi::run(&format!(
            "CREATE TABLE test(embedding vector(3));

            INSERT INTO test(embedding) VALUES ('[1,2,3]'), ('[4,5,6]'), ('[7,8,10]');

            CREATE INDEX idxtest
                  ON test
               USING tsv(embedding)
                WITH (placeholder=30);

            set enable_seqscan =0;
            select * from test order by embedding <=> '[0,0,0]';
            explain analyze select * from test order by embedding <=> '[0,0,0]';
                ",
        ))?;
        Ok(())
    }
}
