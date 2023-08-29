use std::time::Instant;

use ndarray::{Array1, Array2};
use pgrx::pg_sys::{BufferGetBlockNumber, Pointer};
use pgrx::*;
use rand::Rng;
use reductive::pq::{Pq, QuantizeVector, TrainPq};

use crate::access_method::disk_index_graph::DiskIndexGraph;
use crate::access_method::graph::Graph;
use crate::access_method::graph::InsertStats;
use crate::access_method::model::{read_pq, PgVector};
use crate::access_method::options::TSVIndexOptions;
use crate::util::page;
use crate::util::tape::Tape;
use crate::util::*;

use super::builder_graph::BuilderGraph;
use super::model::{self};

const TSV_MAGIC_NUMBER: u32 = 768756476; //Magic number, random
const TSV_VERSION: u32 = 1;
const GRAPH_SLACK_FACTOR: f64 = 1.3_f64;
const PQ_TRAINING_ITERATIONS: usize = 20;
const NUM_SUBQUANTIZER_BITS: u32 = 8;

const NUM_TRAINING_ATTEMPTS: usize = 1;

const NUM_TRAINING_SET_SIZE: usize = 100000;
/// This is metadata about the entire index.
/// Stored as the first page in the index relation.
#[derive(Clone)]
pub struct TsvMetaPage {
    /// random magic number for identifying the index
    magic_number: u32,
    /// version number for future-proofing
    version: u32,
    /// number of dimensions in the vector
    num_dimensions: u32,
    /// max number of outgoing edges a node in the graph can have (R in the papers)
    num_neighbors: u32,
    search_list_size: u32,
    max_alpha: f64,
    init_ids_block_number: pg_sys::BlockNumber,
    init_ids_offset: pg_sys::OffsetNumber,
    use_pq: bool,
    num_clusters: usize,
    pq_block_number: pg_sys::BlockNumber,
    pq_block_offset: pg_sys::OffsetNumber,
}

impl TsvMetaPage {
    /// Number of dimensions in the vectors being stored.
    /// Has to be the same for all vectors in the graph and cannot change.
    pub fn get_num_dimensions(&self) -> u32 {
        self.num_dimensions
    }

    /// Maximum number of neigbors per node. Given that we pre-allocate
    /// these many slots for each node, this cannot change after the graph is built.
    pub fn get_num_neighbors(&self) -> u32 {
        self.num_neighbors
    }

    pub fn get_num_clusters(&self) -> usize {
        self.num_clusters
    }

    pub fn get_search_list_size_for_build(&self) -> u32 {
        self.search_list_size
    }

    pub fn get_max_alpha(&self) -> f64 {
        self.max_alpha
    }

    pub fn get_use_pq(&self) -> bool {
        self.use_pq
    }

    pub fn get_max_neighbors_during_build(&self) -> usize {
        return ((self.get_num_neighbors() as f64) * GRAPH_SLACK_FACTOR).ceil() as usize;
    }

    pub fn get_init_ids(&self) -> Option<Vec<IndexPointer>> {
        if self.init_ids_block_number == 0 && self.init_ids_offset == 0 {
            return None;
        }

        let ptr = HeapPointer::new(self.init_ids_block_number, self.init_ids_offset);
        Some(vec![ptr])
    }
    pub fn get_pq_pointer(&self) -> Option<IndexPointer> {
        if !self.use_pq || (self.pq_block_number == 0 && self.pq_block_offset == 0) {
            return None;
        }

        let ptr = HeapPointer::new(self.pq_block_number, self.pq_block_offset);
        Some(ptr)
    }
}

struct BuildState {
    memcxt: PgMemoryContexts,
    meta_page: TsvMetaPage,
    ntuples: usize,
    tape: Tape, //The tape is a memory abstraction over Postgres pages for writing data.
    node_builder: BuilderGraph,
    started: Instant,
    stats: InsertStats,
    vectors: Vec<Vec<f32>>,
}

impl BuildState {
    fn new(index_relation: &PgRelation, meta_page: TsvMetaPage) -> Self {
        let tape = unsafe { Tape::new((**index_relation).as_ptr(), page::PageType::Node) };
        //TODO: some ways to get rid of meta_page.clone?
        BuildState {
            memcxt: PgMemoryContexts::new("tsv build context"),
            ntuples: 0,
            meta_page: meta_page.clone(),
            tape,
            node_builder: BuilderGraph::new(meta_page),
            started: Instant::now(),
            stats: InsertStats::new(),
            vectors: Vec::new(),
        }
    }

    fn train_pq(&self) -> Pq<f32> {
        notice!("Training pq with {} vectors", self.vectors.len());
        let training_set = self.vectors.iter().map(|x| x.to_vec()).flatten().collect();
        let shape = (self.vectors.len(), self.vectors[0].len());
        let instances = Array2::<f32>::from_shape_vec(shape, training_set).unwrap();
        Pq::train_pq(
            self.meta_page.num_clusters,
            NUM_SUBQUANTIZER_BITS,
            PQ_TRAINING_ITERATIONS,
            NUM_TRAINING_ATTEMPTS,
            instances,
        )
        .unwrap()
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
    opt: PgBox<TSVIndexOptions>,
) -> TsvMetaPage {
    let page = page::WritablePage::new(index, crate::util::page::PageType::Meta);
    let meta = page_get_meta(*page, *(*(page.get_buffer())), true);
    (*meta).magic_number = TSV_MAGIC_NUMBER;
    (*meta).version = TSV_VERSION;
    (*meta).num_dimensions = num_dimensions;
    (*meta).num_neighbors = (*opt).num_neighbors;
    (*meta).search_list_size = (*opt).search_list_size;
    (*meta).max_alpha = (*opt).max_alpha;
    (*meta).use_pq = (*opt).use_pq;
    (*meta).num_clusters = (*opt).num_clusters;
    (*meta).pq_block_number = 0;
    (*meta).pq_block_offset = 0;
    (*meta).init_ids_block_number = 0;
    (*meta).init_ids_offset = 0;
    let header = page.cast::<pgrx::pg_sys::PageHeaderData>();

    let meta_end = (meta as Pointer).add(std::mem::size_of::<TsvMetaPage>());
    let page_start = (*page) as Pointer;
    (*header).pd_lower = meta_end.offset_from(page_start) as _;

    let mp = (*meta).clone();
    page.commit();
    mp
}

pub fn read_meta_page(index: &PgRelation) -> TsvMetaPage {
    unsafe {
        let page = page::ReadablePage::read(index.as_ptr(), 0);
        let meta = page_get_meta(*page, *(*(page.get_buffer())), false);
        (*meta).clone()
    }
}

pub fn update_meta_page_init_ids(index: &PgRelation, init_ids: Vec<IndexPointer>) {
    assert_eq!(init_ids.len(), 1); //change this if we support multiple
    let id = init_ids[0];

    unsafe {
        let page = page::WritablePage::modify(index.as_ptr(), 0);
        let meta = page_get_meta(*page, *(*(page.get_buffer())), false);
        (*meta).init_ids_block_number = id.block_number;
        (*meta).init_ids_offset = id.offset;
        page.commit()
    }
}

pub fn update_meta_page_pq_pointer(index: &PgRelation, pq_pointer: IndexPointer) {
    unsafe {
        let page = page::WritablePage::modify(index.as_ptr(), 0);
        let meta = page_get_meta(*page, *(*(page.get_buffer())), false);
        (*meta).pq_block_number = pq_pointer.block_number;
        (*meta).pq_block_offset = pq_pointer.offset;
        page.commit()
    }
}

#[pg_guard]
pub extern "C" fn ambuild(
    heaprel: pg_sys::Relation,
    indexrel: pg_sys::Relation,
    index_info: *mut pg_sys::IndexInfo,
) -> *mut pg_sys::IndexBuildResult {
    let heap_relation = unsafe { PgRelation::from_pg(heaprel) };
    let index_relation = unsafe { PgRelation::from_pg(indexrel) };
    let opt = TSVIndexOptions::from_relation(&index_relation);

    notice!(
        "Starting index build. num_neighbors={} search_list_size={}, max_alpha={}, use_pq={}, num_clusters={}",
        opt.num_neighbors,
        opt.search_list_size,
        opt.max_alpha,
        opt.use_pq,
        opt.num_clusters
    );

    let dimensions = index_relation.tuple_desc().get(0).unwrap().atttypmod;
    // PQ is only applicable to high dimension vectors.
    if opt.use_pq {
        if dimensions < opt.num_clusters as i32 {
            error!("use_pq can only be applied to vectors with greater than {} dimensions. {} dimensions provided", opt.num_clusters, dimensions)
        };
        if dimensions % opt.num_clusters as i32 != 0 {
            error!("use_pq can only be applied to vectors where the number of dimensions {} is divisible by the number of clusters {} ", dimensions, opt.num_clusters)
        };
    }
    assert!(dimensions > 0 && dimensions < 2000);
    let meta_page = unsafe { write_meta_page(indexrel, dimensions as _, opt.clone()) };
    let (ntuples, pq_opt) = do_heap_scan(index_info, &heap_relation, &index_relation, meta_page);

    // When using PQ, we initialize a node to store the model we use to quantize the vectors.
    unsafe {
        if opt.use_pq {
            let pq = pq_opt.unwrap();
            let index_pointer: IndexPointer = model::write_pq(pq, &index_relation);
            update_meta_page_pq_pointer(&index_relation, index_pointer)
        }
    }

    let mut result = unsafe { PgBox::<pg_sys::IndexBuildResult>::alloc0() };
    result.heap_tuples = ntuples as f64;
    result.index_tuples = ntuples as f64;

    result.into_pg()
}

#[pg_guard]
pub unsafe extern "C" fn aminsert(
    indexrel: pg_sys::Relation,
    values: *mut pg_sys::Datum,
    isnull: *mut bool,
    heap_tid: pg_sys::ItemPointer,
    _heap_relation: pg_sys::Relation,
    _check_unique: pg_sys::IndexUniqueCheck,
    _index_unchanged: bool,
    _index_info: *mut pg_sys::IndexInfo,
) -> bool {
    let index_relation = unsafe { PgRelation::from_pg(indexrel) };
    let vec = PgVector::from_pg_parts(values, isnull, 0);
    if let None = vec {
        //todo handle NULLs?
        return false;
    }
    let vec = vec.unwrap();
    let vector = (*vec).to_slice();
    let heap_pointer = ItemPointer::with_item_pointer_data(*heap_tid);
    let mut graph = DiskIndexGraph::new(&index_relation);
    let meta_page = read_meta_page(&index_relation);

    let mut node = model::Node::new(vector.to_vec(), heap_pointer, &meta_page);
    if meta_page.use_pq {
        let pq_id = meta_page.get_pq_pointer().unwrap();
        let pq = read_pq(&index_relation, &pq_id);
        let og_vec = Array1::from(vector.to_vec());
        node.pq_vector = pq.quantize_vector(og_vec).to_vec();
    }

    let mut tape = unsafe { Tape::new((*index_relation).as_ptr(), page::PageType::Node) };
    let index_pointer: IndexPointer = node.write(&mut tape);

    let _stats = graph.insert(&index_relation, index_pointer, vector);
    false
}

#[pg_guard]
pub extern "C" fn ambuildempty(_index_relation: pg_sys::Relation) {
    panic!("ambuildempty: not yet implemented")
}

fn do_heap_scan<'a>(
    index_info: *mut pg_sys::IndexInfo,
    heap_relation: &'a PgRelation,
    index_relation: &'a PgRelation,
    meta_page: TsvMetaPage,
) -> (usize, Option<Pq<f32>>) {
    let mut state = BuildState::new(index_relation, meta_page.clone());
    let mut pq: Option<Pq<f32>> = None;
    unsafe {
        pg_sys::IndexBuildHeapScan(
            heap_relation.as_ptr(),
            index_relation.as_ptr(),
            index_info,
            Some(build_callback),
            &mut state,
        );
    }

    // we train the quantizer and add prepare to write quantized values to the nodes.
    if meta_page.use_pq {
        let v = state.train_pq();
        pq = Some(v)
    }
    let write_stats = unsafe { state.node_builder.write(index_relation, pq.clone()) };
    assert_eq!(write_stats.num_nodes, state.ntuples);

    let writing_took = Instant::now()
        .duration_since(write_stats.started)
        .as_secs_f64();
    if write_stats.num_nodes > 0 {
        info!(
            "Writing took {}s or {}s/tuple.  Avg neighbors: {}",
            writing_took,
            writing_took / write_stats.num_nodes as f64,
            write_stats.num_neighbors / write_stats.num_nodes
        );
    }
    if write_stats.num_prunes > 0 {
        info!(
            "When pruned for cleanup: avg neighbors before/after {}/{} of {} prunes",
            write_stats.num_neighbors_before_prune / write_stats.num_prunes,
            write_stats.num_neighbors_after_prune / write_stats.num_prunes,
            write_stats.num_prunes
        );
    }
    let ntuples = state.ntuples;

    warning!("Indexed {} tuples", ntuples);
    (ntuples, pq)
}

#[pg_guard]
unsafe extern "C" fn build_callback(
    index: pg_sys::Relation,
    ctid: pg_sys::ItemPointer,
    values: *mut pg_sys::Datum,
    isnull: *mut bool,
    _tuple_is_alive: bool,
    state: *mut std::os::raw::c_void,
) {
    let index_relation = unsafe { PgRelation::from_pg(index) };
    let vec = PgVector::from_pg_parts(values, isnull, 0);
    if let Some(vec) = vec {
        build_callback_internal(index_relation, *ctid, (*vec).to_slice(), state);
    }
    //todo: what do we do with nulls?
}

#[inline(always)]
unsafe fn build_callback_internal(
    index: PgRelation,
    ctid: pg_sys::ItemPointerData,
    vector: &[f32],
    state: *mut std::os::raw::c_void,
) {
    check_for_interrupts!();
    let mut rng = rand::thread_rng();

    let state = (state as *mut BuildState).as_mut().unwrap();
    let mut old_context = state.memcxt.set_as_current();

    state.ntuples = state.ntuples + 1;

    if state.ntuples % 1000 == 0 {
        info!(
            "Processed {} tuples in {}s which is {}s/tuple. Dist/tuple: Prune: {} search: {}. Stats: {:?}",
            state.ntuples,
            Instant::now().duration_since(state.started).as_secs_f64(),
            (Instant::now().duration_since(state.started) / state.ntuples as u32).as_secs_f64(),
            state.stats.prune_neighbor_stats.distance_comparisons / state.ntuples,
            state.stats.greedy_search_stats.distance_comparisons / state.ntuples,
            state.stats,
        );
    }

    let heap_pointer = ItemPointer::with_item_pointer_data(ctid);

    // We collect the training data and reservoir sample into a training buffer of size
    // NUM_TRAINING_SET_SIZE.
    if state.meta_page.get_use_pq() {
        if state.vectors.len() >= NUM_TRAINING_SET_SIZE {
            let index = rng.gen_range(0..state.ntuples + 1);
            if index < NUM_TRAINING_SET_SIZE {
                state.vectors[index] = vector.to_vec();
            }
        } else {
            state.vectors.push(vector.to_vec());
        }
    }
    let node = model::Node::new(vector.to_vec(), heap_pointer, &state.meta_page);
    let index_pointer: IndexPointer = node.write(&mut state.tape);
    let new_stats = state.node_builder.insert(&index, index_pointer, vector);
    state.stats.combine(new_stats);
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
                WITH (num_neighbors=30);

            set enable_seqscan =0;
            select * from test order by embedding <=> '[0,0,0]';
            explain analyze select * from test order by embedding <=> '[0,0,0]';
            drop index idxtest;
            ",
        ))?;
        Ok(())
    }

    #[pg_test]
    unsafe fn test_insert() -> spi::Result<()> {
        Spi::run(&format!(
            "CREATE TABLE test(embedding vector(3));

            INSERT INTO test(embedding) VALUES ('[1,2,3]'), ('[4,5,6]'), ('[7,8,10]');

            CREATE INDEX idxtest
                  ON test
               USING tsv(embedding)
                WITH (num_neighbors=30);

            INSERT INTO test(embedding) VALUES ('[11,12,13]');
            ",
        ))?;

        let res: Option<i64> = Spi::get_one(&format!(
            "   set enable_seqscan = 0;
                WITH cte as (select * from test order by embedding <=> '[0,0,0]') SELECT count(*) from cte;",
        ))?;
        assert_eq!(4, res.unwrap());

        Spi::run(&format!(
            "INSERT INTO test(embedding) VALUES ('[11,12,13]'),  ('[14,15,16]');",
        ))?;
        let res: Option<i64> = Spi::get_one(&format!(
            "   set enable_seqscan = 0;
                WITH cte as (select * from test order by embedding <=> '[0,0,0]') SELECT count(*) from cte;",
        ))?;
        assert_eq!(6, res.unwrap());

        Spi::run(&format!("drop index idxtest;",))?;

        Ok(())
    }

    #[pg_test]
    unsafe fn test_empty_table_insert() -> spi::Result<()> {
        Spi::run(&format!(
            "CREATE TABLE test(embedding vector(3));

            CREATE INDEX idxtest
                  ON test
               USING tsv(embedding)
                WITH (num_neighbors=30);

            INSERT INTO test(embedding) VALUES ('[1,2,3]'), ('[4,5,6]'), ('[7,8,10]');
            ",
        ))?;

        let res: Option<i64> = Spi::get_one(&format!(
            "   set enable_seqscan = 0;
                WITH cte as (select * from test order by embedding <=> '[0,0,0]') SELECT count(*) from cte;",
        ))?;
        assert_eq!(3, res.unwrap());

        Spi::run(&format!("drop index idxtest;",))?;

        Ok(())
    }

    #[pg_test]
    unsafe fn test_insert_empty_insert() -> spi::Result<()> {
        Spi::run(&format!(
            "CREATE TABLE test(embedding vector(3));

            CREATE INDEX idxtest
                  ON test
               USING tsv(embedding)
                WITH (num_neighbors=30);

            INSERT INTO test(embedding) VALUES ('[1,2,3]'), ('[4,5,6]'), ('[7,8,10]');
            DELETE FROM test;
            INSERT INTO test(embedding) VALUES ('[1,2,3]'), ('[14,15,16]');
            ",
        ))?;

        let res: Option<i64> = Spi::get_one(&format!(
            "   set enable_seqscan = 0;
                WITH cte as (select * from test order by embedding <=> '[0,0,0]') SELECT count(*) from cte;",
        ))?;
        assert_eq!(2, res.unwrap());

        Spi::run(&format!("drop index idxtest;",))?;

        Ok(())
    }
}
