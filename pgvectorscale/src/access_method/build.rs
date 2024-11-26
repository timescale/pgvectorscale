use std::time::Instant;

use pg_sys::{FunctionCall0Coll, InvalidOid};
use pgrx::pg_sys::{index_getprocinfo, pgstat_progress_update_param, AsPgCStr};
use pgrx::*;

use crate::access_method::distance::DistanceType;
use crate::access_method::graph::Graph;
use crate::access_method::graph_neighbor_store::GraphNeighborStore;
use crate::access_method::options::TSVIndexOptions;
use crate::access_method::pg_vector::PgVector;
use crate::access_method::stats::{InsertStats, WriteStats};

use crate::access_method::DISKANN_DISTANCE_TYPE_PROC;
use crate::util::page::PageType;
use crate::util::tape::Tape;
use crate::util::*;

use self::ports::PROGRESS_CREATE_IDX_SUBPHASE;

use super::graph_neighbor_store::BuilderNeighborCache;
use super::sbq::SbqSpeedupStorage;

use super::meta_page::MetaPage;

use super::plain_storage::PlainStorage;
use super::storage::{Storage, StorageType};

enum StorageBuildState<'a, 'b, 'c, 'd, 'e> {
    SbqSpeedup(&'a mut SbqSpeedupStorage<'b>, &'c mut BuildState<'d, 'e>),
    Plain(&'a mut PlainStorage<'b>, &'c mut BuildState<'d, 'e>),
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

        BuildState {
            memcxt: PgMemoryContexts::new("diskann build context"),
            ntuples: 0,
            meta_page,
            tape,
            graph,
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
        "Starting index build. num_neighbors={} search_list_size={}, max_alpha={}, storage_layout={:?}",
        opt.get_num_neighbors(),
        opt.search_list_size,
        opt.max_alpha,
        opt.get_storage_type(),
    );

    let dimensions = index_relation.tuple_desc().get(0).unwrap().atttypmod;

    let distance_type = unsafe {
        let fmgr_info = index_getprocinfo(indexrel, 1, DISKANN_DISTANCE_TYPE_PROC);
        if fmgr_info.is_null() {
            error!("No distance type function found for index");
        }
        let result = FunctionCall0Coll(fmgr_info, InvalidOid).value() as u16;
        DistanceType::from_u16(result)
    };

    if distance_type == DistanceType::InnerProduct && opt.get_storage_type() == StorageType::Plain {
        error!("Inner product distance type is not supported with plain storage");
    }

    let meta_page =
        unsafe { MetaPage::create(&index_relation, dimensions as _, distance_type, opt) };

    assert!(
        meta_page.get_num_dimensions_to_index() > 0
            && meta_page.get_num_dimensions_to_index() <= 2000
    );

    let ntuples = do_heap_scan(index_info, &heap_relation, &index_relation, meta_page);

    let mut result = unsafe { PgBox::<pg_sys::IndexBuildResult>::alloc0() };
    result.heap_tuples = ntuples as f64;
    result.index_tuples = ntuples as f64;

    result.into_pg()
}

#[cfg(any(feature = "pg14", feature = "pg15", feature = "pg16", feature = "pg17"))]
#[pg_guard]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn aminsert(
    indexrel: pg_sys::Relation,
    values: *mut pg_sys::Datum,
    isnull: *mut bool,
    heap_tid: pg_sys::ItemPointer,
    heaprel: pg_sys::Relation,
    _check_unique: pg_sys::IndexUniqueCheck::Type,
    _index_unchanged: bool,
    _index_info: *mut pg_sys::IndexInfo,
) -> bool {
    aminsert_internal(indexrel, values, isnull, heap_tid, heaprel)
}

#[cfg(feature = "pg13")]
#[pg_guard]
pub unsafe extern "C" fn aminsert(
    indexrel: pg_sys::Relation,
    values: *mut pg_sys::Datum,
    isnull: *mut bool,
    heap_tid: pg_sys::ItemPointer,
    heaprel: pg_sys::Relation,
    _check_unique: pg_sys::IndexUniqueCheck::Type,
    _index_info: *mut pg_sys::IndexInfo,
) -> bool {
    aminsert_internal(indexrel, values, isnull, heap_tid, heaprel)
}

unsafe fn aminsert_internal(
    indexrel: pg_sys::Relation,
    values: *mut pg_sys::Datum,
    isnull: *mut bool,
    heap_tid: pg_sys::ItemPointer,
    heaprel: pg_sys::Relation,
) -> bool {
    let index_relation = PgRelation::from_pg(indexrel);
    let heap_relation = PgRelation::from_pg(heaprel);
    let mut meta_page = MetaPage::fetch(&index_relation);
    let vec = PgVector::from_pg_parts(values, isnull, 0, &meta_page, true, false);
    if vec.is_none() {
        //todo handle NULLs?
        return false;
    }
    let vec = vec.unwrap();
    let heap_pointer = ItemPointer::with_item_pointer_data(*heap_tid);

    let mut storage = meta_page.get_storage_type();
    let mut stats = InsertStats::new();
    match &mut storage {
        StorageType::Plain => {
            let plain = PlainStorage::load_for_insert(
                &index_relation,
                &heap_relation,
                meta_page.get_distance_function(),
            );
            insert_storage(
                &plain,
                &index_relation,
                vec,
                heap_pointer,
                &mut meta_page,
                &mut stats,
            );
        }
        StorageType::SbqSpeedup | StorageType::SbqCompression => {
            let bq = SbqSpeedupStorage::load_for_insert(
                &heap_relation,
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
    let mut tape = Tape::resume(index_relation, S::page_type());
    let index_pointer = storage.create_node(
        vector.to_index_slice(),
        heap_pointer,
        meta_page,
        &mut tape,
        stats,
    );

    let mut graph = Graph::new(GraphNeighborStore::Disk, meta_page);
    graph.insert(index_relation, index_pointer, vector, storage, stats)
}

#[pg_guard]
pub extern "C" fn ambuildempty(_index_relation: pg_sys::Relation) {
    panic!("ambuildempty: not yet implemented")
}

fn do_heap_scan<'a>(
    index_info: *mut pg_sys::IndexInfo,
    heap_relation: &'a PgRelation,
    index_relation: &'a PgRelation,
    meta_page: MetaPage,
) -> usize {
    let storage = meta_page.get_storage_type();

    let mut mp2 = meta_page.clone();
    let graph = Graph::new(
        GraphNeighborStore::Builder(BuilderNeighborCache::new()),
        &mut mp2,
    );
    let mut write_stats = WriteStats::new();
    match storage {
        StorageType::Plain => {
            let mut plain = PlainStorage::new_for_build(
                index_relation,
                heap_relation,
                meta_page.get_distance_function(),
            );
            plain.start_training(&meta_page);
            let page_type = PlainStorage::page_type();
            let mut bs = BuildState::new(index_relation, meta_page, graph, page_type);
            let mut state = StorageBuildState::Plain(&mut plain, &mut bs);

            unsafe {
                pg_sys::IndexBuildHeapScan(
                    heap_relation.as_ptr(),
                    index_relation.as_ptr(),
                    index_info,
                    Some(build_callback),
                    &mut state,
                );
            }

            finalize_index_build(&mut plain, &mut bs, write_stats)
        }
        StorageType::SbqSpeedup | StorageType::SbqCompression => {
            let mut bq =
                SbqSpeedupStorage::new_for_build(index_relation, heap_relation, &meta_page);

            let page_type = SbqSpeedupStorage::page_type();

            unsafe {
                pgstat_progress_update_param(PROGRESS_CREATE_IDX_SUBPHASE, BUILD_PHASE_TRAINING);
            }

            bq.start_training(&meta_page);

            let mut bs = BuildState::new(index_relation, meta_page, graph, page_type);
            let mut state = StorageBuildState::SbqSpeedup(&mut bq, &mut bs);

            unsafe {
                pg_sys::IndexBuildHeapScan(
                    heap_relation.as_ptr(),
                    index_relation.as_ptr(),
                    index_info,
                    Some(build_callback_bq_train),
                    &mut state,
                );
            }
            bq.finish_training(&mut write_stats);

            unsafe {
                pgstat_progress_update_param(
                    PROGRESS_CREATE_IDX_SUBPHASE,
                    BUILD_PHASE_BUILDING_GRAPH,
                );
            }

            let mut state = StorageBuildState::SbqSpeedup(&mut bq, &mut bs);

            unsafe {
                pg_sys::IndexBuildHeapScan(
                    heap_relation.as_ptr(),
                    index_relation.as_ptr(),
                    index_info,
                    Some(build_callback),
                    &mut state,
                );
            }

            unsafe {
                pgstat_progress_update_param(
                    PROGRESS_CREATE_IDX_SUBPHASE,
                    BUILD_PHASE_FINALIZING_GRAPH,
                );
            }
            finalize_index_build(&mut bq, &mut bs, write_stats)
        }
    }
}

fn finalize_index_build<S: Storage>(
    storage: &mut S,
    state: &mut BuildState,
    mut write_stats: WriteStats,
) -> usize {
    match state.graph.get_neighbor_store() {
        GraphNeighborStore::Builder(builder) => {
            for (&index_pointer, neighbors) in builder.iter() {
                write_stats.num_nodes += 1;
                let prune_neighbors;
                let neighbors =
                    if neighbors.len() > state.graph.get_meta_page().get_num_neighbors() as _ {
                        //OPT: get rid of this clone
                        prune_neighbors = state.graph.prune_neighbors(
                            index_pointer,
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

    debug1!("write done");
    assert_eq!(write_stats.num_nodes, state.ntuples);

    let writing_took = Instant::now()
        .duration_since(write_stats.started)
        .as_secs_f64();
    if write_stats.num_nodes > 0 {
        debug1!(
            "Writing took {}s or {}s/tuple.  Avg neighbors: {}",
            writing_took,
            writing_took / write_stats.num_nodes as f64,
            write_stats.num_neighbors / write_stats.num_nodes
        );
    }
    if write_stats.prune_stats.calls > 0 {
        debug1!(
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
unsafe extern "C" fn build_callback_bq_train(
    _index: pg_sys::Relation,
    _ctid: pg_sys::ItemPointer,
    values: *mut pg_sys::Datum,
    isnull: *mut bool,
    _tuple_is_alive: bool,
    state: *mut std::os::raw::c_void,
) {
    let state = (state as *mut StorageBuildState).as_mut().unwrap();
    match state {
        StorageBuildState::SbqSpeedup(bq, state) => {
            let vec = PgVector::from_pg_parts(values, isnull, 0, &state.meta_page, true, false);
            if let Some(vec) = vec {
                bq.add_sample(vec.to_index_slice());
            }
        }
        StorageBuildState::Plain(_, _) => {
            panic!("Should not be training with plain storage");
        }
    }
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
    let state = (state as *mut StorageBuildState).as_mut().unwrap();
    match state {
        StorageBuildState::SbqSpeedup(bq, state) => {
            let vec = PgVector::from_pg_parts(values, isnull, 0, &state.meta_page, true, false);
            if let Some(vec) = vec {
                let heap_pointer = ItemPointer::with_item_pointer_data(*ctid);
                build_callback_memory_wrapper(index_relation, heap_pointer, vec, state, *bq);
            }
        }
        StorageBuildState::Plain(plain, state) => {
            let vec = PgVector::from_pg_parts(values, isnull, 0, &state.meta_page, true, false);
            if let Some(vec) = vec {
                let heap_pointer = ItemPointer::with_item_pointer_data(*ctid);
                build_callback_memory_wrapper(index_relation, heap_pointer, vec, state, *plain);
            }
        }
    }
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

    state.ntuples += 1;

    if state.ntuples % 1000 == 0 {
        debug1!(
            "Processed {} tuples in {}s which is {}s/tuple. Dist/tuple: Prune: {} search: {}. Stats: {:?}",
            state.ntuples,
            Instant::now().duration_since(state.started).as_secs_f64(),
            (Instant::now().duration_since(state.started) / state.ntuples as u32).as_secs_f64(),
            state.stats.prune_neighbor_stats.distance_comparisons / state.ntuples,
            state.stats.greedy_search_stats.get_total_distance_comparisons() / state.ntuples,
            state.stats,
        );
    }

    let index_pointer = storage.create_node(
        vector.to_index_slice(),
        heap_pointer,
        &state.meta_page,
        &mut state.tape,
        &mut state.stats,
    );

    state
        .graph
        .insert(&index, index_pointer, vector, storage, &mut state.stats);
}

const BUILD_PHASE_TRAINING: i64 = 0;
const BUILD_PHASE_BUILDING_GRAPH: i64 = 1;
const BUILD_PHASE_FINALIZING_GRAPH: i64 = 2;

#[pg_guard]
pub unsafe extern "C" fn ambuildphasename(phasenum: i64) -> *mut ffi::c_char {
    match phasenum {
        BUILD_PHASE_TRAINING => "training quantizer".as_pg_cstr(),
        BUILD_PHASE_BUILDING_GRAPH => "building graph".as_pg_cstr(),
        BUILD_PHASE_FINALIZING_GRAPH => "finalizing graph".as_pg_cstr(),
        _ => error!("Unknown phase number {}", phasenum),
    }
}

#[cfg(any(test, feature = "pg_test"))]
#[pgrx::pg_schema]
pub mod tests {
    use std::collections::HashSet;

    use crate::access_method::distance::DistanceType;
    use pgrx::*;

    //TODO: add test where inserting and querying with vectors that are all the same.

    #[cfg(any(test, feature = "pg_test"))]
    pub unsafe fn test_index_creation_and_accuracy_scaffold(
        distance_type: DistanceType,
        index_options: &str,
        name: &str,
        vector_dimensions: usize,
    ) -> spi::Result<()> {
        let operator = distance_type.get_operator();
        let operator_class = distance_type.get_operator_class();
        let table_name = format!("test_data_icaa_{}", name);
        Spi::run(&format!(
            "CREATE TABLE {table_name} (
                embedding vector ({vector_dimensions})
            );

            select setseed(0.5);
           -- generate 300 vectors
            INSERT INTO {table_name} (embedding)
            SELECT
                *
            FROM (
                SELECT
                    ('[' || array_to_string(array_agg(random()), ',', '0') || ']')::vector AS embedding
                FROM
                    generate_series(1, {vector_dimensions} * 300) i
                GROUP BY
                    i % 300) g;

            CREATE INDEX ON {table_name} USING diskann (embedding {operator_class}) WITH ({index_options});


            SET enable_seqscan = 0;
            -- perform index scans on the vectors
            SELECT
                *
            FROM
                {table_name}
            ORDER BY
                embedding {operator} (
                    SELECT
                        ('[' || array_to_string(array_agg(random()), ',', '0') || ']')::vector AS embedding
            FROM generate_series(1, {vector_dimensions}));"))?;

        let test_vec: Option<Vec<f32>> = Spi::get_one(
            &format!("SELECT('{{' || array_to_string(array_agg(1.0), ',', '0') || '}}')::real[] AS embedding
    FROM generate_series(1, {vector_dimensions})")
                .to_string(),
        )?;

        let cnt: Option<i64> = Spi::get_one_with_args(
                &format!(
                    "
            SET enable_seqscan = 0;
            SET enable_indexscan = 1;
            SET diskann.query_search_list_size = 2;
            WITH cte as (select * from {table_name} order by embedding {operator} $1::vector) SELECT count(*) from cte;
            ",
                ),
                vec![(
                    pgrx::PgOid::Custom(pgrx::pg_sys::FLOAT4ARRAYOID),
                    test_vec.clone().into_datum(),
                )],
            )?;

        //FIXME: should work in all cases
        if !index_options.contains("num_neighbors=10") {
            assert_eq!(cnt.unwrap(), 300, "initial count");
        }

        Spi::run(&format!("
            -- test insert 2 vectors
            INSERT INTO {table_name} (embedding)
            SELECT
                *
            FROM (
                SELECT
                    ('[' || array_to_string(array_agg(random()), ',', '0') || ']')::vector AS embedding
                FROM
                    generate_series(1, {vector_dimensions} * 2) i
                GROUP BY
                    i % 2) g;


            EXPLAIN ANALYZE
            SELECT
                *
            FROM
                {table_name}
            ORDER BY
                embedding {operator} (
                    SELECT
                        ('[' || array_to_string(array_agg(random()), ',', '0') || ']')::vector AS embedding
            FROM generate_series(1, {vector_dimensions}));

            -- test insert 10 vectors to search for that aren't random
            INSERT INTO {table_name} (embedding)
            SELECT
                *
            FROM (
                SELECT
                    ('[' || array_to_string(array_agg(1.0), ',', '0') || ']')::vector AS embedding
                FROM
                    generate_series(1, {vector_dimensions} * 10) i
                GROUP BY
                    i % 10) g;

            ",
        ))?;

        let with_index: Option<Vec<String>> = Spi::get_one_with_args(
            &format!(
                "
        SET enable_seqscan = 0;
        SET enable_indexscan = 1;
        SET diskann.query_search_list_size = 25;
        WITH cte AS (
            SELECT
                ctid::TEXT
            FROM
                {table_name}
            ORDER BY
                embedding {operator} $1::vector
            LIMIT 10
        )
        SELECT array_agg(ctid) from cte;"
            ),
            vec![(
                pgrx::PgOid::Custom(pgrx::pg_sys::FLOAT4ARRAYOID),
                test_vec.clone().into_datum(),
            )],
        )?;

        /* Test that the explain plan is generated ok */
        let explain: Option<pgrx::datum::Json> = Spi::get_one_with_args(
            &format!(
                "
        SET enable_seqscan = 0;
        SET enable_indexscan = 1;
        EXPLAIN (format json) WITH cte AS (
            SELECT
                ctid
            FROM
                {table_name}
            ORDER BY
                embedding {operator} $1::vector
            LIMIT 10
        )
        SELECT array_agg(ctid) from cte;"
            ),
            vec![(
                pgrx::PgOid::Custom(pgrx::pg_sys::FLOAT4ARRAYOID),
                test_vec.clone().into_datum(),
            )],
        )?;
        assert!(explain.is_some());
        //warning!("explain: {}", explain.unwrap().0);

        let without_index: Vec<String> = Spi::get_one_with_args(
            &format!(
                "
        SET enable_seqscan = 1;
        SET enable_indexscan = 0;
        WITH cte AS (
            SELECT
                ctid::TEXT
            FROM
                {table_name}
            ORDER BY
                embedding {operator} $1::vector
            LIMIT 10
        )
        SELECT array_agg(ctid) from cte;"
            ),
            vec![(
                pgrx::PgOid::Custom(pgrx::pg_sys::FLOAT4ARRAYOID),
                test_vec.clone().into_datum(),
            )],
        )?
        .unwrap();

        let set: HashSet<_> = without_index.iter().collect();

        let mut matches = 0;
        for ctid in with_index.unwrap() {
            if set.contains(&ctid) {
                matches += 1;
            }
        }
        assert!(matches > 9, "Low number of matches: {}", matches);

        //FIXME: should work in all cases
        if !index_options.contains("num_neighbors=10") {
            //make sure you can scan entire table with index
            let cnt: Option<i64> = Spi::get_one_with_args(
            &format!(
                "
        SET enable_seqscan = 0;
        SET enable_indexscan = 1;
        SET diskann.query_search_list_size = 2;
        WITH cte as (select * from {table_name} order by embedding {operator} $1::vector) SELECT count(*) from cte;
        ",
            ),
            vec![(
                pgrx::PgOid::Custom(pgrx::pg_sys::FLOAT4ARRAYOID),
                test_vec.into_datum(),
            )],
        )?;

            assert_eq!(cnt.unwrap(), 312);
        }

        Ok(())
    }

    #[pg_test]
    pub unsafe fn test_l2_sanity_check() -> spi::Result<()> {
        Spi::run(&format!(
            "CREATE TABLE test(embedding vector(3));

            CREATE INDEX idxtest
                  ON test
               USING diskann(embedding vector_l2_ops)
                WITH (num_neighbors=10, search_list_size=10);

            INSERT INTO test(embedding) VALUES ('[1,1,1]'), ('[2,2,2]'), ('[3,3,3]');
            ",
        ))?;

        // Query vector [1,1,1] should return [1,1,1]; [2,2,2] should return [2,2,2];
        // and [3,3,3] should return [3,3,3].  (Note that if vectors or the query vector
        // were normalized, then the results would be different.)
        let res: Option<Vec<String>> = Spi::get_one(
            "WITH cte as (select * from test order by embedding <-> '[1,1,1]' LIMIT 1)
            SELECT array_agg(embedding::text) from cte;",
        )?;
        assert_eq!(vec!["[1,1,1]"], res.unwrap());

        let res: Option<Vec<String>> = Spi::get_one(
            "WITH cte as (select * from test order by embedding <-> '[2,2,2]' LIMIT 1)
            SELECT array_agg(embedding::text) from cte;",
        )?;
        assert_eq!(vec!["[2,2,2]"], res.unwrap());

        let res: Option<Vec<String>> = Spi::get_one(
            "WITH cte as (select * from test order by embedding <-> '[3,3,3]' LIMIT 1)
            SELECT array_agg(embedding::text) from cte;",
        )?;
        assert_eq!(vec!["[3,3,3]"], res.unwrap());

        Spi::run(&"drop index idxtest;".to_string())?;

        Ok(())
    }

    #[pg_test]
    pub unsafe fn test_ip_sanity_check() -> spi::Result<()> {
        Spi::run(&format!(
            "CREATE TABLE test(embedding vector(3));

            CREATE INDEX idxtest
                  ON test
               USING diskann(embedding vector_ip_ops)
                WITH (num_neighbors=10, search_list_size=10);

            INSERT INTO test(embedding) VALUES ('[1,1,1]'), ('[2,2,2]'), ('[3,3,3]');
            ",
        ))?;

        let res: Option<Vec<String>> = Spi::get_one(
            "WITH cte as (select * from test order by embedding <#> '[1,1,1]' LIMIT 1)
            SELECT array_agg(embedding::text) from cte;",
        )?;
        assert_eq!(vec!["[3,3,3]"], res.unwrap());

        let res: Option<Vec<String>> = Spi::get_one(
            "WITH cte as (select * from test order by embedding <#> '[2,2,2]' LIMIT 1)
            SELECT array_agg(embedding::text) from cte;",
        )?;
        assert_eq!(vec!["[3,3,3]"], res.unwrap());

        let res: Option<Vec<String>> = Spi::get_one(
            "WITH cte as (select * from test order by embedding <#> '[3,3,3]' LIMIT 1)
            SELECT array_agg(embedding::text) from cte;",
        )?;
        assert_eq!(vec!["[3,3,3]"], res.unwrap());

        Spi::run(&"drop index idxtest;".to_string())?;

        Ok(())
    }

    #[cfg(any(test, feature = "pg_test"))]
    pub unsafe fn test_empty_table_insert_scaffold(index_options: &str) -> spi::Result<()> {
        Spi::run(&format!(
            "CREATE TABLE test(embedding vector(3));

            CREATE INDEX idxtest
                  ON test
               USING diskann(embedding)
                WITH ({index_options});

            INSERT INTO test(embedding) VALUES ('[1,2,3]'), ('[4,5,6]'), ('[7,8,10]');
            ",
        ))?;

        let res: Option<i64> = Spi::get_one(&"   set enable_seqscan = 0;
                WITH cte as (select * from test order by embedding <=> '[0,0,0]') SELECT count(*) from cte;".to_string())?;
        assert_eq!(3, res.unwrap());

        Spi::run(
            &"
        set enable_seqscan = 0;
        explain analyze select * from test order by embedding <=> '[0,0,0]';
        "
            .to_string(),
        )?;

        Spi::run(&"drop index idxtest;".to_string())?;

        Ok(())
    }

    #[cfg(any(test, feature = "pg_test"))]
    pub unsafe fn test_insert_empty_insert_scaffold(index_options: &str) -> spi::Result<()> {
        Spi::run(&format!(
            "CREATE TABLE test(embedding vector(3));

            CREATE INDEX idxtest
                  ON test
               USING diskann(embedding)
                WITH ({index_options});

            INSERT INTO test(embedding) VALUES ('[1,2,3]'), ('[4,5,6]'), ('[7,8,10]');
            DELETE FROM test;
            INSERT INTO test(embedding) VALUES ('[1,2,3]'), ('[14,15,16]');
            ",
        ))?;

        let res: Option<i64> = Spi::get_one(&"   set enable_seqscan = 0;
                WITH cte as (select * from test order by embedding <=> '[0,0,0]') SELECT count(*) from cte;".to_string())?;
        assert_eq!(2, res.unwrap());

        Spi::run(&"drop index idxtest;".to_string())?;

        Ok(())
    }

    #[cfg(any(test, feature = "pg_test"))]
    pub unsafe fn test_index_updates(
        distance_type: DistanceType,
        index_options: &str,
        expected_cnt: i64,
        name: &str,
    ) -> spi::Result<()> {
        let operator_class = distance_type.get_operator_class();
        let operator = distance_type.get_operator();

        let table_name = format!("test_data_index_updates_{}", name);
        Spi::run(&format!(
            "CREATE TABLE {table_name} (
                id int,
                embedding vector (1536)
            );

            select setseed(0.5);
           -- generate 300 vectors
            INSERT INTO {table_name} (id, embedding)
            SELECT
                *
            FROM (
                SELECT
                    i % {expected_cnt},
                    ('[' || array_to_string(array_agg(random()), ',', '0') || ']')::vector AS embedding
                FROM
                    generate_series(1, 1536 * {expected_cnt}) i
                GROUP BY
                    i % {expected_cnt}) g;

            CREATE INDEX ON {table_name} USING diskann (embedding {operator_class}) WITH ({index_options});


            SET enable_seqscan = 0;
            -- perform index scans on the vectors
            SELECT
                *
            FROM
                {table_name}
            ORDER BY
                embedding {operator} (
                    SELECT
                        ('[' || array_to_string(array_agg(random()), ',', '0') || ']')::vector AS embedding
            FROM generate_series(1, 1536));"))?;

        let test_vec: Option<Vec<f32>> = Spi::get_one(
            &"SELECT('{' || array_to_string(array_agg(1.0), ',', '0') || '}')::real[] AS embedding
    FROM generate_series(1, 1536)"
                .to_string(),
        )?;

        let cnt: Option<i64> = Spi::get_one_with_args(
                &format!(
                    "
            SET enable_seqscan = 0;
            SET enable_indexscan = 1;
            SET diskann.query_search_list_size = 2;
            WITH cte as (select * from {table_name} order by embedding {operator} $1::vector) SELECT count(*) from cte;
            ",
                ),
                vec![(
                    pgrx::PgOid::Custom(pgrx::pg_sys::FLOAT4ARRAYOID),
                    test_vec.clone().into_datum(),
                )],
            )?;

        assert!(cnt.unwrap() == expected_cnt, "initial count");

        Spi::run(&format!(
            "

        --CREATE INDEX idx_id ON {table_name}(id);

        WITH CTE as (
            SELECT
                i % {expected_cnt} as id,
                ('[' || array_to_string(array_agg(random()), ',', '0') || ']')::vector AS embedding
            FROM
                generate_series(1, 1536 * {expected_cnt}) i
            GROUP BY
            i % {expected_cnt}
        )
        UPDATE {table_name} SET embedding = cte.embedding
        FROM cte
        WHERE {table_name}.id = cte.id;

        --DROP INDEX idx_id;
            ",
        ))?;

        let cnt: Option<i64> = Spi::get_one_with_args(
            &format!(
                "
        SET enable_seqscan = 0;
        SET enable_indexscan = 1;
        SET diskann.query_search_list_size = 2;
        WITH cte as (select * from {table_name} order by embedding {operator} $1::vector) SELECT count(*) from cte;
        ",
            ),
            vec![(
                pgrx::PgOid::Custom(pgrx::pg_sys::FLOAT4ARRAYOID),
                test_vec.clone().into_datum(),
            )],
        )?;

        assert!(cnt.unwrap() == expected_cnt, "after update count");

        Ok(())
    }

    pub fn verify_index_accuracy(expected_cnt: i64, dimensions: usize) -> spi::Result<()> {
        let test_vec: Option<Vec<f32>> = Spi::get_one(&format!(
            "SELECT('{{' || array_to_string(array_agg(1.0), ',', '0') || '}}')::real[] AS embedding
    FROM generate_series(1, {dimensions})"
        ))?;

        let cnt: Option<i64> = Spi::get_one_with_args(
                &format!(
                    "
            SET enable_seqscan = 0;
            SET enable_indexscan = 1;
            SET diskann.query_search_list_size = 2;
            WITH cte as (select * from test_data order by embedding <=> $1::vector) SELECT count(*) from cte;
            ",
                ),
                vec![(
                    pgrx::PgOid::Custom(pgrx::pg_sys::FLOAT4ARRAYOID),
                    test_vec.clone().into_datum(),
                )],
            )?;

        if cnt.unwrap() != expected_cnt {
            // better debugging
            let id: Option<String> = Spi::get_one_with_args(
                &format!(
                    "
            SET enable_seqscan = 0;
            SET enable_indexscan = 1;
            SET diskann.query_search_list_size = 2;
            WITH cte as (select id from test_data EXCEPT (select id from test_data order by embedding <=> $1::vector)) SELECT ctid::text || ' ' || id from test_data where id in (select id from cte limit 1);
            ",
                ),
                vec![(
                    pgrx::PgOid::Custom(pgrx::pg_sys::FLOAT4ARRAYOID),
                    test_vec.clone().into_datum(),
                )],
            )?;

            assert!(
                cnt.unwrap() == expected_cnt,
                "initial count is {} id is {}",
                cnt.unwrap(),
                id.unwrap()
            );
        }

        assert!(
            cnt.unwrap() == expected_cnt,
            "initial count is {}",
            cnt.unwrap()
        );
        Ok(())
    }

    #[pg_test]
    pub unsafe fn test_index_small_accuracy() -> spi::Result<()> {
        // Test for the creation of connected graphs when the number of dimensions is small as is the
        // number of neighborss
        // small num_neighbors is especially challenging for making sure no nodes get disconnected
        let index_options = "num_neighbors=10, search_list_size=10";
        let expected_cnt = 1000;
        let dimensions = 2;

        Spi::run(&format!(
            "CREATE TABLE test_data (
                id int,
                embedding vector ({dimensions})
            );

            select setseed(0.5);
           -- generate 300 vectors
            INSERT INTO test_data (id, embedding)
            SELECT
                *
            FROM (
                SELECT
                    i % {expected_cnt},
                    ('[' || array_to_string(array_agg(random()), ',', '0') || ']')::vector AS embedding
                FROM
                    generate_series(1, {dimensions} * {expected_cnt}) i
                GROUP BY
                    i % {expected_cnt}) g;

            CREATE INDEX idx_diskann_bq ON test_data USING diskann (embedding) WITH ({index_options});


            SET enable_seqscan = 0;
            -- perform index scans on the vectors
            SELECT
                *
            FROM
                test_data
            ORDER BY
                embedding <=> (
                    SELECT
                        ('[' || array_to_string(array_agg(random()), ',', '0') || ']')::vector AS embedding
            FROM generate_series(1, {dimensions}));"))?;

        verify_index_accuracy(expected_cnt, dimensions)?;
        Ok(())
    }

    #[pg_test]
    pub unsafe fn test_index_small_accuracy_insert_after_index_created() -> spi::Result<()> {
        // Test for the creation of connected graphs when the number of dimensions is small as is the
        // number of neighborss
        // small num_neighbors is especially challenging for making sure no nodes get disconnected
        let index_options = "num_neighbors=10, search_list_size=10";
        let expected_cnt = 1000;
        let dimensions = 2;

        Spi::run(&format!(
            "CREATE TABLE test_data (
                id int,
                embedding vector ({dimensions})
            );
            
            CREATE INDEX idx_diskann_bq ON test_data USING diskann (embedding) WITH ({index_options});

            select setseed(0.5);
           -- generate 300 vectors
            INSERT INTO test_data (id, embedding)
            SELECT
                *
            FROM (
                SELECT
                    i % {expected_cnt},
                    ('[' || array_to_string(array_agg(random()), ',', '0') || ']')::vector AS embedding
                FROM
                    generate_series(1, {dimensions} * {expected_cnt}) i
                GROUP BY
                    i % {expected_cnt}) g;

            SET enable_seqscan = 0;
            -- perform index scans on the vectors
            SELECT
                *
            FROM
                test_data
            ORDER BY
                embedding <=> (
                    SELECT
                        ('[' || array_to_string(array_agg(random()), ',', '0') || ']')::vector AS embedding
            FROM generate_series(1, {dimensions}));"))?;

        verify_index_accuracy(expected_cnt, dimensions)?;
        Ok(())
    }
}
