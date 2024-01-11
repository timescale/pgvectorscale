use std::time::Instant;

use pgrx::*;

use crate::access_method::graph::Graph;
use crate::access_method::graph_neighbor_store::GraphNeighborStore;
use crate::access_method::options::TSVIndexOptions;
use crate::access_method::pg_vector::PgVector;
use crate::access_method::stats::{InsertStats, WriteStats};

use crate::util::page::PageType;
use crate::util::tape::Tape;
use crate::util::*;

use super::bq::BqStorage;
use super::graph_neighbor_store::BuilderNeighborCache;

use super::meta_page::MetaPage;

use super::storage::{Storage, StorageType};

enum StorageBuildState<'a, 'b, 'c, 'd, 'e> {
    BQ(&'d mut BqStorage<'c>, &'e mut BuildState<'a, 'b>),
}

struct BuildState<'a, 'b> {
    memcxt: PgMemoryContexts,
    meta_page: MetaPage,
    ntuples: usize,
    tape: Tape<'a>, //The tape is a memory abstraction over Postgres pages for writing data.
    graph: Graph<'b>,
    started: Instant,
    stats: InsertStats,
}

impl<'a, 'b> BuildState<'a, 'b> {
    fn new(
        index_relation: &'a PgRelation,
        meta_page: MetaPage,
        graph: Graph<'b>,
        page_type: PageType,
    ) -> Self {
        let tape = unsafe { Tape::new(index_relation, page_type) };

        //TODO: some ways to get rid of meta_page.clone?
        BuildState {
            memcxt: PgMemoryContexts::new("tsv build context"),
            ntuples: 0,
            meta_page: meta_page,
            tape,
            graph: graph,
            started: Instant::now(),
            stats: InsertStats::new(),
        }
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
        "Starting index build. num_neighbors={} search_list_size={}, max_alpha={}, use_pq={}, pq_vector_length={}",
        opt.num_neighbors,
        opt.search_list_size,
        opt.max_alpha,
        opt.use_pq,
        opt.pq_vector_length
    );

    let dimensions = index_relation.tuple_desc().get(0).unwrap().atttypmod;
    // PQ is only applicable to high dimension vectors.
    if opt.use_pq {
        if dimensions < opt.pq_vector_length as i32 {
            error!("use_pq can only be applied to vectors with greater than {} dimensions. {} dimensions provided", opt.pq_vector_length, dimensions)
        };
        if dimensions % opt.pq_vector_length as i32 != 0 {
            error!("use_pq can only be applied to vectors where the number of dimensions {} is divisible by the pq_vector_length {} ", dimensions, opt.pq_vector_length)
        };
    }
    assert!(dimensions > 0 && dimensions < 2000);
    let meta_page = unsafe { MetaPage::create(&index_relation, dimensions as _, opt.clone()) };

    let ntuples = do_heap_scan(index_info, &heap_relation, &index_relation, meta_page);

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
    heaprel: pg_sys::Relation,
    _check_unique: pg_sys::IndexUniqueCheck,
    _index_unchanged: bool,
    index_info: *mut pg_sys::IndexInfo,
) -> bool {
    let index_relation = unsafe { PgRelation::from_pg(indexrel) };
    let heap_relation = unsafe { PgRelation::from_pg(heaprel) };
    let vec = PgVector::from_pg_parts(values, isnull, 0);
    if let None = vec {
        //todo handle NULLs?
        return false;
    }
    let vec = vec.unwrap();
    let heap_pointer = ItemPointer::with_item_pointer_data(*heap_tid);
    let mut meta_page = MetaPage::read(&index_relation);

    let mut storage = meta_page.get_storage_type();
    let mut stats = InsertStats::new();
    match &mut storage {
        StorageType::None => {}
        StorageType::PQ => {
            //pq.load(&index_relation, &meta_page);
            //let _stats = insert_storage(&pq, &index_relation, vector, heap_pointer, &mut meta_page);
            pgrx::error!("not implemented");
        }
        StorageType::BQ => {
            let bq = BqStorage::load_for_insert(
                &heap_relation,
                get_attribute_number(index_info),
                &index_relation,
                &meta_page,
                &mut stats.quantizer_stats,
            );
            insert_storage(
                &bq,
                &index_relation,
                vec,
                heap_pointer,
                &mut meta_page,
                &mut stats,
            );
        }
    }
    false
}

unsafe fn insert_storage<S: Storage>(
    storage: &S,
    index_relation: &PgRelation,
    vector: PgVector,
    heap_pointer: ItemPointer,
    meta_page: &mut MetaPage,
    stats: &mut InsertStats,
) {
    let mut tape = Tape::new(&index_relation, storage.page_type());
    let index_pointer = storage.create_node(
        vector.to_slice(),
        heap_pointer,
        &meta_page,
        &mut tape,
        stats,
    );

    let mut graph = Graph::new(GraphNeighborStore::Disk, meta_page);
    graph.insert(&index_relation, index_pointer, vector, storage, stats)
}

#[pg_guard]
pub extern "C" fn ambuildempty(_index_relation: pg_sys::Relation) {
    panic!("ambuildempty: not yet implemented")
}

fn get_attribute_number(index_info: *mut pg_sys::IndexInfo) -> pg_sys::AttrNumber {
    unsafe { assert!((*index_info).ii_NumIndexAttrs == 1) };
    unsafe { (*index_info).ii_IndexAttrNumbers[0] }
}

fn do_heap_scan<'a>(
    index_info: *mut pg_sys::IndexInfo,
    heap_relation: &'a PgRelation,
    index_relation: &'a PgRelation,
    meta_page: MetaPage,
) -> usize {
    let mut storage = meta_page.get_storage_type();

    let mut mp2 = meta_page.clone();
    let graph = Graph::new(
        GraphNeighborStore::Builder(BuilderNeighborCache::new()),
        &mut mp2,
    );
    match storage {
        StorageType::None => {
            pgrx::error!("not implemented");
        }
        StorageType::PQ => {
            //pq.start_training(&meta_page);
            pgrx::error!("not implemented");
        }
        StorageType::BQ => {
            let mut bq = BqStorage::new_for_build(
                index_relation,
                heap_relation,
                get_attribute_number(index_info),
            );
            bq.start_training(&meta_page);
            let page_type = bq.page_type();
            let mut bs = BuildState::new(index_relation, meta_page, graph, page_type);
            let mut state = StorageBuildState::BQ(&mut bq, &mut bs);

            unsafe {
                pg_sys::IndexBuildHeapScan(
                    heap_relation.as_ptr(),
                    index_relation.as_ptr(),
                    index_info,
                    Some(build_callback),
                    &mut state,
                );
            }

            do_heap_scan_with_state(&mut bq, &mut bs)
        }
    }
}

fn do_heap_scan_with_state<S: Storage>(storage: &mut S, state: &mut BuildState) -> usize {
    // we train the quantizer and add prepare to write quantized values to the nodes.\
    let mut write_stats = WriteStats::new();
    storage.finish_training(&mut write_stats);

    match state.graph.get_neighbor_store() {
        GraphNeighborStore::Builder(builder) => {
            for (&index_pointer, neighbors) in builder.iter() {
                write_stats.num_nodes += 1;
                let prune_neighbors;
                let neighbors =
                    if neighbors.len() > state.graph.get_meta_page().get_num_neighbors() as _ {
                        //OPT: get rid of this clone
                        prune_neighbors = state.graph.prune_neighbors(
                            neighbors.clone(),
                            storage,
                            &mut write_stats.prune_stats,
                        );
                        &prune_neighbors
                    } else {
                        neighbors
                    };
                write_stats.num_neighbors += neighbors.len();

                storage.finalize_node_at_end_of_build(
                    &state.meta_page,
                    index_pointer,
                    neighbors,
                    &mut write_stats,
                );
            }
        }
        GraphNeighborStore::Disk => {
            panic!("Should not be using the disk neighbor store during build");
        }
    }

    info!("write done");
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
    if write_stats.prune_stats.calls > 0 {
        info!(
            "When pruned for cleanup: avg neighbors before/after {}/{} of {} prunes",
            write_stats.prune_stats.num_neighbors_before_prune / write_stats.prune_stats.calls,
            write_stats.prune_stats.num_neighbors_after_prune / write_stats.prune_stats.calls,
            write_stats.prune_stats.calls
        );
    }
    let ntuples = state.ntuples;

    warning!("Indexed {} tuples", ntuples);

    ntuples
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
        let state = (state as *mut StorageBuildState).as_mut().unwrap();
        let heap_pointer = ItemPointer::with_item_pointer_data(*ctid);

        match state {
            StorageBuildState::BQ(bq, state) => {
                build_callback_memory_wrapper(index_relation, heap_pointer, vec, state, *bq);
            }
        }
    }
    //todo: what do we do with nulls?
}

#[inline(always)]
unsafe fn build_callback_memory_wrapper<S: Storage>(
    index: PgRelation,
    heap_pointer: ItemPointer,
    vector: PgVector,
    state: &mut BuildState,
    storage: &mut S,
) {
    let mut old_context = state.memcxt.set_as_current();

    build_callback_internal(index, heap_pointer, vector, state, storage);

    old_context.set_as_current();
    state.memcxt.reset();
}

#[inline(always)]
fn build_callback_internal<S: Storage>(
    index: PgRelation,
    heap_pointer: ItemPointer,
    vector: PgVector,
    state: &mut BuildState,
    storage: &mut S,
) {
    check_for_interrupts!();

    state.ntuples = state.ntuples + 1;

    if state.ntuples % 1000 == 0 {
        info!(
            "Processed {} tuples in {}s which is {}s/tuple. Dist/tuple: Prune: {} search: {}. Stats: {:?}",
            state.ntuples,
            Instant::now().duration_since(state.started).as_secs_f64(),
            (Instant::now().duration_since(state.started) / state.ntuples as u32).as_secs_f64(),
            state.stats.prune_neighbor_stats.distance_comparisons / state.ntuples,
            state.stats.greedy_search_stats.get_total_distance_comparisons() / state.ntuples,
            state.stats,
        );
    }

    storage.add_sample(vector.to_slice());

    let index_pointer = storage.create_node(
        vector.to_slice(),
        heap_pointer,
        &state.meta_page,
        &mut state.tape,
        &mut state.stats,
    );

    state
        .graph
        .insert(&index, index_pointer, vector, storage, &mut state.stats);
}

#[cfg(any(test, feature = "pg_test"))]
#[pgrx::pg_schema]
pub mod tests {
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
    unsafe fn test_pq_index_creation() -> spi::Result<()> {
        Spi::run(&format!(
            "CREATE TABLE test_pq (
                embedding vector (1536)
            );

           -- generate 300 vectors
            INSERT INTO test_pq (embedding)
            SELECT
                *
            FROM (
                SELECT
                    ('[' || array_to_string(array_agg(random()), ',', '0') || ']')::vector AS embedding
                FROM
                    generate_series(1, 1536 * 300) i
                GROUP BY
                    i % 300) g;

            CREATE INDEX idx_tsv_pq ON test_pq USING tsv (embedding) WITH (num_neighbors = 64, search_list_size = 125, max_alpha = 1.0, use_pq = TRUE, pq_vector_length = 64);

            ;

            SET enable_seqscan = 0;
            -- perform index scans on the vectors
            SELECT
                *
            FROM
                test_pq
            ORDER BY
                embedding <=> (
                    SELECT
                        ('[' || array_to_string(array_agg(random()), ',', '0') || ']')::vector AS embedding
            FROM generate_series(1, 1536));

            EXPLAIN ANALYZE
            SELECT
                *
            FROM
                test_pq
            ORDER BY
                embedding <=> (
                    SELECT
                        ('[' || array_to_string(array_agg(random()), ',', '0') || ']')::vector AS embedding
            FROM generate_series(1, 1536));

            DROP INDEX idx_tsv_pq;
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

    #[cfg(any(test, feature = "pg_test"))]
    pub unsafe fn test_empty_table_insert_scaffold(index_options: &str) -> spi::Result<()> {
        Spi::run(&format!(
            "CREATE TABLE test(embedding vector(3));

            CREATE INDEX idxtest
                  ON test
               USING tsv(embedding)
                WITH ({index_options});

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

    #[cfg(any(test, feature = "pg_test"))]
    pub unsafe fn test_insert_empty_insert_scaffold(index_options: &str) -> spi::Result<()> {
        Spi::run(&format!(
            "CREATE TABLE test(embedding vector(3));

            CREATE INDEX idxtest
                  ON test
               USING tsv(embedding)
                WITH ({index_options});

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

    #[pg_test]
    unsafe fn test_empty_table_insert() -> spi::Result<()> {
        crate::access_method::build::tests::test_empty_table_insert_scaffold("num_neighbors=30")
    }

    #[pg_test]
    unsafe fn test_insert_empty_insert() -> spi::Result<()> {
        crate::access_method::build::tests::test_insert_empty_insert_scaffold("num_neighbors=30")
    }
}
