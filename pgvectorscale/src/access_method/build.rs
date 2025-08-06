use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;

use pg_sys::{FunctionCall0Coll, InvalidOid};
use pgrx::ffi::c_char;
use pgrx::pg_sys::{index_getprocinfo, pgstat_progress_update_param, AsPgCStr, Oid};
use pgrx::*;

use crate::access_method::distance::DistanceType;
use crate::access_method::graph::neighbor_store::GraphNeighborStore;
use crate::access_method::graph::Graph;
use crate::access_method::options::TSVIndexOptions;
use crate::access_method::pg_vector::PgVector;
use crate::access_method::stats::{InsertStats, WriteStats};
use crate::util::ports::acquire_index_lock;

use crate::access_method::DISKANN_DISTANCE_TYPE_PROC;
use crate::util::page::PageType;
use crate::util::tape::Tape;
use crate::util::*;
use crate::util::ports::IndexBuildHeapScanParallel;

use self::ports::PROGRESS_CREATE_IDX_SUBPHASE;

use super::graph::neighbor_store::BuilderNeighborCache;
use super::labels::LabeledVector;
use super::sbq::quantize::SbqQuantizer;
use super::sbq::storage::SbqSpeedupStorage;

use super::meta_page::MetaPage;

use super::plain::storage::PlainStorage;
use super::sbq::SbqMeans;
use super::storage::{Storage, StorageType};

mod parallel;

struct SbqTrainState<'a, 'b> {
    quantizer: &'a mut SbqQuantizer,
    meta_page: &'b MetaPage,
}

enum StorageBuildState<'a, 'b, 'c, 'd> {
    SbqSpeedup(&'a mut SbqSpeedupStorage<'b>, &'c mut BuildState<'d>),
    Plain(&'a mut PlainStorage<'b>, &'c mut BuildState<'d>),
}

/// Storage build state for parallel builds using shared state
enum StorageBuildStateParallel<'a, 'b, 'c> {
    SbqSpeedup(
        &'a mut SbqSpeedupStorage<'b>,
        &'c mut BuildStateParallel<'b>,
    ),
    Plain(&'a mut PlainStorage<'b>, &'c mut BuildStateParallel<'b>),
}

struct BuildState<'a> {
    memcxt: PgMemoryContexts,
    ntuples: usize,
    tape: Tape<'a>, //The tape is a memory abstraction over Postgres pages for writing data.
    graph: Graph<'a>,
    stats: InsertStats,
}

/// Wrapper for BuildState that shares statistics with parallel workers
struct BuildStateParallel<'a> {
    memcxt: PgMemoryContexts,
    tape: Tape<'a>,
    graph: Graph<'a>,
    shared_state: &'a ParallelShared,
    local_stats: InsertStats,
    local_ntuples: usize,
    is_initializing_worker: bool,
}

impl<'a> BuildState<'a> {
    fn new(index_relation: &'a PgRelation, graph: Graph<'a>, page_type: PageType) -> Self {
        let tape = unsafe { Tape::new(index_relation, page_type) };

        BuildState {
            memcxt: PgMemoryContexts::new("diskann build context"),
            ntuples: 0,
            tape,
            graph,
            stats: InsertStats::default(),
        }
    }
}

impl<'a> BuildStateParallel<'a> {
    fn new(
        index_relation: &'a PgRelation,
        graph: Graph<'a>,
        page_type: PageType,
        shared_state: &'a ParallelShared,
        is_initializing_worker: bool,
    ) -> Self {
        let tape = unsafe { Tape::new(index_relation, page_type) };

        BuildStateParallel {
            memcxt: PgMemoryContexts::new("diskann build context"),
            tape,
            graph,
            shared_state,
            local_stats: InsertStats::default(),
            local_ntuples: 0,
            is_initializing_worker,
        }
    }

    fn increment_ntuples(&mut self) {
        self.local_ntuples += 1;
        // Only update shared counter for the initializing worker until threshold is reached
        if self.is_initializing_worker && self.local_ntuples <= parallel::INITIAL_START_NODES_COUNT {
            self.shared_state.build_state.ntuples.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn into_build_state(self) -> BuildState<'a> {
        // Signal that the initializing worker is done if this is the initializing worker
        if self.is_initializing_worker {
            self.shared_state.build_state.initializing_worker_done.store(true, Ordering::Relaxed);
        }
        
        self.update_shared_ntuples();

        let ntuples = self.local_ntuples;

        BuildState {
            memcxt: self.memcxt,
            ntuples,
            tape: self.tape,
            graph: self.graph,
            stats: self.local_stats,
        }
    }

    fn update_shared_ntuples(&self) {
        if self.is_initializing_worker {
            // For initializing worker, only add tuples beyond the initial threshold to avoid double counting
            let remaining = self.local_ntuples.saturating_sub(parallel::INITIAL_START_NODES_COUNT);
            if remaining > 0 {
                self.shared_state.build_state.ntuples.fetch_add(remaining, Ordering::Relaxed);
            }
        } else {
            // For non-initializing workers, add all local tuples
            self.shared_state.build_state.ntuples.fetch_add(self.local_ntuples, Ordering::Relaxed);
        }
    }
}

/// Maximum number of dimensions supported by pgvector's vector type.  Also
/// the maximum number of dimensions that can be indexed with diskann.
pub const MAX_DIMENSION: u32 = 16000;

/// Maximum number of dimensions that can be indexed with diskann without
/// using the SBQ storage type.
pub const MAX_DIMENSION_NO_SBQ: u32 = 2000;

/// Data about parallel index build that never changes.
#[derive(Debug, Copy, Clone)]
#[cfg_attr(not(feature = "build_parallel"), allow(dead_code))]
struct ParallelSharedParams {
    heaprelid: Oid,
    indexrelid: Oid,
    is_concurrent: bool,
}

/// Shared build state for parallel index builds.
#[derive(Debug)]
#[cfg_attr(not(feature = "build_parallel"), allow(dead_code))]
struct ParallelBuildState {
    ntuples: AtomicUsize,
    start_nodes_initialized: AtomicBool,
    initializing_worker_done: AtomicBool,
}

/// Status data for parallel index builds, shared among all parallel workers.
#[derive(Debug)]
#[cfg_attr(not(feature = "build_parallel"), allow(dead_code))]
struct ParallelShared {
    params: ParallelSharedParams,
    build_state: ParallelBuildState,
}

/// Information about parallel build passed to heap scan.
#[derive(Debug)]
#[cfg_attr(not(feature = "build_parallel"), allow(dead_code))]
struct ParallelBuildInfo {
    parallel_shared: *mut ParallelShared,
    is_initializing_worker: bool,
    tablescandesc: *mut pg_sys::ParallelTableScanDescData,
}

fn get_meta_page(
    indexrel: pg_sys::Relation,
    index_relation: &PgRelation,
    opt: PgBox<TSVIndexOptions>,
) -> MetaPage {
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

    if meta_page.get_num_dimensions_to_index() == 0 {
        error!("No dimensions to index");
    }

    if meta_page.get_num_dimensions_to_index() > MAX_DIMENSION {
        error!("Too many dimensions to index (max is {})", MAX_DIMENSION);
    }

    if meta_page.get_num_dimensions_to_index() > MAX_DIMENSION_NO_SBQ
        && meta_page.get_storage_type() == StorageType::Plain
    {
        error!(
            "Too many dimensions to index with plain storage (max is {}).  Use storage_layout=memory_optimized instead.",
            MAX_DIMENSION_NO_SBQ
        );
    }

    if meta_page.has_labels() && meta_page.get_storage_type() == StorageType::Plain {
        error!("Labeled filtering is not supported with plain storage");
    }

    meta_page
}

#[pg_guard]
pub extern "C-unwind" fn ambuild(
    heaprel: pg_sys::Relation,
    indexrel: pg_sys::Relation,
    index_info: *mut pg_sys::IndexInfo,
) -> *mut pg_sys::IndexBuildResult {
    let heap_relation = unsafe { PgRelation::from_pg(heaprel) };
    let index_relation = unsafe { PgRelation::from_pg(indexrel) };
    let opt = TSVIndexOptions::from_relation(&index_relation);
    let mut meta_page = get_meta_page(indexrel, &index_relation, opt);
    let opt = TSVIndexOptions::from_relation(&index_relation);

    notice!(
        "Starting index build with num_neighbors={}, search_list_size={}, max_alpha={}, storage_layout={:?}.",
        opt.get_num_neighbors(),
        opt.search_list_size,
        opt.max_alpha,
        opt.get_storage_type(),
    );

    // Train quantizer before doing anything in parallel
    let write_stats =
        maybe_train_quantizer(index_info, &heap_relation, &index_relation, &mut meta_page);
    unsafe {
        meta_page.store(&index_relation, false);
    };

    // TODO: unsafe { (*index_info).ii_ParallelWorkers };
    let workers = if cfg!(feature = "build_parallel") {
        8 // TODO: set properly
    } else {
        0
    };
    let is_concurrent = unsafe { (*index_info).ii_Concurrent };
    struct ParallelData {
        pcxt: *mut pg_sys::ParallelContext,
        snapshot: *mut pg_sys::SnapshotData,
    }
    let parallel_data = if workers > 0 {
        notice!("Parallel build with {} workers", workers);
        unsafe {
            pg_sys::EnterParallelMode();
            const EXTENSION_NAME: *const c_char = {
                static NAME: &str =
                    concat!(env!("CARGO_PKG_NAME"), "-", env!("CARGO_PKG_VERSION"), "\0");
                NAME.as_ptr() as *const c_char
            };

            let pcxt = pg_sys::CreateParallelContext(EXTENSION_NAME, PARALLEL_BUILD_MAIN, workers);
            let snapshot = if is_concurrent {
                pg_sys::RegisterSnapshot(pg_sys::GetTransactionSnapshot())
            } else {
                &raw mut pg_sys::SnapshotAnyData
            };

            // Estimate things we put in shared memory
            parallel::toc_estimate_single_chunk(pcxt, size_of::<ParallelShared>());
            let tablescandesc_size_estimate =
                pg_sys::table_parallelscan_estimate(heaprel, snapshot);
            parallel::toc_estimate_single_chunk(pcxt, tablescandesc_size_estimate);

            pg_sys::InitializeParallelDSM(pcxt);
            if (*pcxt).seg.is_null() {
                parallel::cleanup_pcxt(pcxt, snapshot);
                None
            } else {
                let parallel_shared =
                    pg_sys::shm_toc_allocate((*pcxt).toc, size_of::<ParallelShared>())
                        .cast::<ParallelShared>();
                parallel_shared.write(ParallelShared {
                    params: ParallelSharedParams {
                        heaprelid: heap_relation.rd_id,
                        indexrelid: index_relation.rd_id,
                        is_concurrent,
                    },
                    build_state: ParallelBuildState {
                        ntuples: AtomicUsize::new(0),
                        start_nodes_initialized: AtomicBool::new(false),
                        initializing_worker_done: AtomicBool::new(false),
                    },
                });
                let tablescandesc =
                    pg_sys::shm_toc_allocate((*pcxt).toc, tablescandesc_size_estimate)
                        .cast::<pg_sys::ParallelTableScanDescData>();
                pg_sys::table_parallelscan_initialize(heaprel, tablescandesc, snapshot);

                pg_sys::shm_toc_insert(
                    (*pcxt).toc,
                    parallel::SHM_TOC_SHARED_KEY,
                    parallel_shared.cast(),
                );
                pg_sys::shm_toc_insert(
                    (*pcxt).toc,
                    parallel::SHM_TOC_TABLESCANDESC_KEY,
                    tablescandesc.cast(),
                );

                pg_sys::LaunchParallelWorkers(pcxt);
                if (*pcxt).nworkers_launched == 0 {
                    warning!("No workers launched");
                    parallel::cleanup_pcxt(pcxt, snapshot);
                    None
                } else {
                    pg_sys::WaitForParallelWorkersToAttach(pcxt);
                    Some(ParallelData { pcxt, snapshot })
                }
            }
        }
    } else {
        None
    };

    let ntuples = if let Some(ParallelData { pcxt, snapshot }) = parallel_data {
        unsafe {
            pg_sys::WaitForParallelWorkersToFinish(pcxt);
            let parallel_shared: *mut ParallelShared =
                pg_sys::shm_toc_lookup((*pcxt).toc, parallel::SHM_TOC_SHARED_KEY, false)
                    .cast::<ParallelShared>();
            let ntuples = (*parallel_shared).build_state.ntuples.load(Ordering::Relaxed);
            parallel::cleanup_pcxt(pcxt, snapshot);
            ntuples
        }
    } else {
        do_heap_scan(
            index_info,
            &heap_relation,
            &index_relation,
            meta_page,
            write_stats,
            None,
        )
    };

    let mut result = unsafe { PgBox::<pg_sys::IndexBuildResult>::alloc0() };
    result.heap_tuples = ntuples as f64;
    result.index_tuples = ntuples as f64;

    result.into_pg()
}

#[cfg(any(
    feature = "pg14",
    feature = "pg15",
    feature = "pg16",
    feature = "pg17",
    feature = "pg18"
))]
#[pg_guard]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C-unwind" fn aminsert(
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
pub unsafe extern "C-unwind" fn aminsert(
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

    // Acquire advisory txn-level lock to serialize all index operations.
    // This prevents concurrent update races (and snapshot-isolation anomalies)
    // in meta_page, tape, and index pages.  TODO: allow more concurrency.
    acquire_index_lock(&index_relation);
    let mut meta_page = MetaPage::fetch(&index_relation);

    let vec = LabeledVector::from_datums(values, isnull, &meta_page);
    if vec.is_none() {
        //todo handle NULLs?
        return false;
    }
    let vec = vec.unwrap();

    let heap_pointer = ItemPointer::with_item_pointer_data(*heap_tid);
    let mut storage = meta_page.get_storage_type();
    let mut stats = InsertStats::default();

    match &mut storage {
        StorageType::Plain => {
            let plain = PlainStorage::load_for_insert(&index_relation, &heap_relation, &meta_page);
            assert!(vec.labels().is_none());
            insert_storage(
                &plain,
                &index_relation,
                vec,
                heap_pointer,
                &mut meta_page,
                &mut stats,
            );
        }
        StorageType::SbqCompression => {
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
    vector: LabeledVector,
    heap_pointer: ItemPointer,
    meta_page: &mut MetaPage,
    stats: &mut InsertStats,
) {
    let mut tape = Tape::resume(index_relation, S::page_type());

    let index_pointer = storage.create_node(
        vector.vec().to_index_slice(),
        vector.labels().cloned(),
        heap_pointer,
        meta_page,
        &mut tape,
        stats,
    );

    let mut graph = Graph::new(GraphNeighborStore::Disk, meta_page);
    graph.insert(index_relation, index_pointer, vector, storage, stats);
}

#[pg_guard]
pub extern "C-unwind" fn ambuildempty(_index_relation: pg_sys::Relation) {
    panic!("ambuildempty: not yet implemented")
}

/// The fraction of maintenance_work_mem that should be used for various large in-memory
/// data structures.
pub const BUILDER_NEIGHBOR_CACHE_SIZE: f64 = 0.8;
pub const QUANTIZED_VECTOR_CACHE_SIZE: f64 = 0.2;

pub fn maintenance_work_mem_bytes() -> usize {
    unsafe { pg_sys::maintenance_work_mem as usize * 1024 }
}

fn maybe_train_quantizer(
    index_info: *mut pg_sys::IndexInfo,
    heap_relation: &PgRelation,
    index_relation: &PgRelation,
    meta_page: &mut MetaPage,
) -> WriteStats {
    let mut write_stats = WriteStats::default();
    let storage = meta_page.get_storage_type();
    match storage {
        StorageType::Plain => {}
        StorageType::SbqCompression => {
            unsafe {
                pgstat_progress_update_param(PROGRESS_CREATE_IDX_SUBPHASE, BUILD_PHASE_TRAINING);
            }
            let mut quantizer = SbqQuantizer::new(meta_page);
            quantizer.start_training(meta_page);

            let mut state = SbqTrainState {
                quantizer: &mut quantizer,
                meta_page,
            };

            unsafe {
                pg_sys::IndexBuildHeapScan(
                    heap_relation.as_ptr(),
                    index_relation.as_ptr(),
                    index_info,
                    Some(build_callback_bq_train),
                    &mut state,
                );
            }
            quantizer.finish_training();
            if quantizer.use_mean {
                let index_pointer =
                    unsafe { SbqMeans::store(index_relation, &quantizer, &mut write_stats) };
                meta_page.set_quantizer_metadata_pointer(index_pointer);
            }
        }
    }
    write_stats
}

const PARALLEL_BUILD_MAIN: *const c_char = c"_vectorscale_build_main".as_ptr();
#[pg_guard]
#[unsafe(no_mangle)]
#[cfg(feature = "build_parallel")]
pub extern "C-unwind" fn _vectorscale_build_main(
    _seg: *mut pg_sys::dsm_segment,
    shm_toc: *mut pg_sys::shm_toc,
) {
    let status_flags = unsafe { (*pg_sys::MyProc).statusFlags };
    assert!(
        status_flags == 0 || status_flags == pg_sys::PROC_IN_SAFE_IC as u8,
        "Status flags for an index build process must be unset or PROC_IN_SAFE_IC (in a safe index creation)"
    );

    let parallel_shared: *mut ParallelShared = unsafe {
        pg_sys::shm_toc_lookup(shm_toc, parallel::SHM_TOC_SHARED_KEY, false)
            .cast::<ParallelShared>()
    };
    let tablescandesc = unsafe {
        pg_sys::shm_toc_lookup(shm_toc, parallel::SHM_TOC_TABLESCANDESC_KEY, false)
            .cast::<pg_sys::ParallelTableScanDescData>()
    };

    let params = unsafe {
        // SAFETY: these parameters never change, so no data races
        (*parallel_shared).params
    };

    // Check if this worker should handle the first 1024 nodes for start node initialization
    let should_initialize = unsafe {
        (*parallel_shared).build_state.start_nodes_initialized.compare_exchange(
            false, 
            true, 
            Ordering::SeqCst, 
            Ordering::SeqCst
        ).is_ok()
    };

    if !should_initialize {
        loop {
            let ntuples = unsafe { (*parallel_shared).build_state.ntuples.load(Ordering::Relaxed) };
            let init_done = unsafe { (*parallel_shared).build_state.initializing_worker_done.load(Ordering::Relaxed) };
            if ntuples >= parallel::INITIAL_START_NODES_COUNT || init_done {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }

    let (heap_lockmode, index_lockmode) = if params.is_concurrent {
        (
            pg_sys::ShareLock as pg_sys::LOCKMODE,
            pg_sys::AccessExclusiveLock as pg_sys::LOCKMODE,
        )
    } else {
        (
            pg_sys::ShareUpdateExclusiveLock as pg_sys::LOCKMODE,
            pg_sys::RowExclusiveLock as pg_sys::LOCKMODE,
        )
    };

    let heaprel = unsafe { pg_sys::table_open(params.heaprelid, heap_lockmode) };
    let indexrel = unsafe { pg_sys::index_open(params.indexrelid, index_lockmode) };
    let index_info = unsafe { pg_sys::BuildIndexInfo(indexrel) };
    let heap_relation = unsafe { PgRelation::from_pg(heaprel) };
    let index_relation = unsafe { PgRelation::from_pg(indexrel) };
    let meta_page = MetaPage::fetch(&index_relation);

    do_heap_scan(
        index_info,
        &heap_relation,
        &index_relation,
        meta_page,
        WriteStats::default(),
        Some(ParallelBuildInfo { 
            parallel_shared, 
            is_initializing_worker: should_initialize,
            tablescandesc,
        }),
    );
}

fn do_heap_scan(
    index_info: *mut pg_sys::IndexInfo,
    heap_relation: &PgRelation,
    index_relation: &PgRelation,
    mut meta_page: MetaPage,
    mut write_stats: WriteStats,
    parallel_build_info: Option<ParallelBuildInfo>,
) -> usize {
    unsafe {
        pgstat_progress_update_param(PROGRESS_CREATE_IDX_SUBPHASE, BUILD_PHASE_BUILDING_GRAPH);
    }

    let storage = meta_page.get_storage_type();

    if let Some(parallel_info) = parallel_build_info {
        let shared_state = unsafe { &*parallel_info.parallel_shared };

        // In parallel mode, timing is handled locally by each worker
        // No shared timing state needed across processes

        let graph = Graph::new(
            GraphNeighborStore::Builder(BuilderNeighborCache::new(
                BUILDER_NEIGHBOR_CACHE_SIZE,
                &meta_page,
            )),
            &mut meta_page,
        );

        match storage {
            StorageType::Plain => {
                let mut plain = PlainStorage::new_for_build(
                    index_relation,
                    heap_relation,
                    graph.get_meta_page(),
                );
                let page_type = PlainStorage::page_type();
                let mut bs =
                    BuildStateParallel::new(index_relation, graph, page_type, shared_state, parallel_info.is_initializing_worker);
                let mut state = StorageBuildStateParallel::Plain(&mut plain, &mut bs);

                unsafe {
                    IndexBuildHeapScanParallel(
                        heap_relation.as_ptr(),
                        index_relation.as_ptr(),
                        index_info,
                        Some(build_callback_parallel),
                        &mut state,
                        parallel_info.tablescandesc,
                    );
                }

                // In parallel mode, nodes are finalized during insertion via streaming
                // Just need to handle any remaining cached nodes and update meta page
                finalize_remaining_parallel_nodes(&mut plain, bs, index_relation, write_stats)
            }
            StorageType::SbqCompression => {
                let mut bq = unsafe {
                    SbqSpeedupStorage::new_for_build(
                        index_relation,
                        heap_relation,
                        graph.get_meta_page(),
                        &mut write_stats,
                    )
                };

                let page_type = SbqSpeedupStorage::page_type();
                let mut bs =
                    BuildStateParallel::new(index_relation, graph, page_type, shared_state, parallel_info.is_initializing_worker);
                let mut state = StorageBuildStateParallel::SbqSpeedup(&mut bq, &mut bs);

                unsafe {
                    IndexBuildHeapScanParallel(
                        heap_relation.as_ptr(),
                        index_relation.as_ptr(),
                        index_info,
                        Some(build_callback_parallel),
                        &mut state,
                        parallel_info.tablescandesc,
                    );
                }

                unsafe {
                    pgstat_progress_update_param(
                        PROGRESS_CREATE_IDX_SUBPHASE,
                        BUILD_PHASE_FINALIZING_GRAPH,
                    );
                }

                // In parallel mode, nodes are finalized during insertion via streaming
                // Just need to handle any remaining cached nodes and update meta page
                finalize_remaining_parallel_nodes(&mut bq, bs, index_relation, write_stats)
            }
        }
    } else {
        // Serial build: use local state
        let graph = Graph::new(
            GraphNeighborStore::Builder(BuilderNeighborCache::new(
                BUILDER_NEIGHBOR_CACHE_SIZE,
                &meta_page,
            )),
            &mut meta_page,
        );

        match storage {
            StorageType::Plain => {
                let mut plain = PlainStorage::new_for_build(
                    index_relation,
                    heap_relation,
                    graph.get_meta_page(),
                );
                let page_type = PlainStorage::page_type();
                let mut bs = BuildState::new(index_relation, graph, page_type);
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

                finalize_index_build(&mut plain, bs, index_relation, write_stats)
            }
            StorageType::SbqCompression => {
                let mut bq = unsafe {
                    SbqSpeedupStorage::new_for_build(
                        index_relation,
                        heap_relation,
                        graph.get_meta_page(),
                        &mut write_stats,
                    )
                };

                let page_type = SbqSpeedupStorage::page_type();
                let mut bs = BuildState::new(index_relation, graph, page_type);
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

                finalize_index_build(&mut bq, bs, index_relation, write_stats)
            }
        }
    }
}

fn finalize_remaining_parallel_nodes<S: Storage>(
    storage: &mut S,
    state: BuildStateParallel,
    index_relation: &PgRelation,
    write_stats: WriteStats,
) -> usize {
    // Convert parallel state to regular build state for final processing
    let build_state = state.into_build_state();
    finalize_index_build(storage, build_state, index_relation, write_stats)
}

fn finalize_index_build<S: Storage>(
    storage: &mut S,
    state: BuildState,
    index_relation: &PgRelation,
    mut write_stats: WriteStats,
) -> usize {
    let BuildState { graph, ntuples, .. } = state;
    let (neighbor_store, meta_page) = graph.into_parts();
    let cache_entries = neighbor_store.into_sorted();

    for (index_pointer, entry) in cache_entries {
        write_stats.num_nodes += 1;
        let prune_neighbors;
        let neighbors = if entry.neighbors.len() > meta_page.get_num_neighbors() as _ {
            prune_neighbors = Graph::prune_neighbors(
                meta_page.get_max_alpha(),
                meta_page.get_num_neighbors() as _,
                entry.labels.as_ref(),
                entry.neighbors,
                storage,
                &mut write_stats.prune_stats,
            );
            prune_neighbors
        } else {
            entry.neighbors
        };
        write_stats.num_neighbors += neighbors.len();

        storage.finalize_node_at_end_of_build(
            index_pointer,
            neighbors.as_slice(),
            &mut write_stats,
        );
    }
    unsafe {
        meta_page.store(index_relation, false);
    }

    debug1!("write done");

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

    notice!("Indexed {} tuples", ntuples);

    ntuples
}

#[pg_guard]
unsafe extern "C-unwind" fn build_callback_bq_train(
    _index: pg_sys::Relation,
    _ctid: pg_sys::ItemPointer,
    values: *mut pg_sys::Datum,
    isnull: *mut bool,
    _tuple_is_alive: bool,
    state: *mut std::os::raw::c_void,
) {
    let state = (state as *mut SbqTrainState).as_mut().unwrap();
    let vec = PgVector::from_pg_parts(values, isnull, 0, state.meta_page, true, false);
    if let Some(vec) = vec {
        state.quantizer.add_sample(vec.to_index_slice());
    }
}

#[pg_guard]
unsafe extern "C-unwind" fn build_callback(
    index: pg_sys::Relation,
    ctid: pg_sys::ItemPointer,
    values: *mut pg_sys::Datum,
    isnull: *mut bool,
    _tuple_is_alive: bool,
    state: *mut std::os::raw::c_void,
) {
    let heap_pointer = ItemPointer::with_item_pointer_data(*ctid);
    let index_relation = PgRelation::from_pg(index);
    let state = (state as *mut StorageBuildState).as_mut().unwrap();
    match state {
        StorageBuildState::SbqSpeedup(bq, state) => {
            let vec = LabeledVector::from_datums(values, isnull, state.graph.get_meta_page());
            if let Some(vec) = vec {
                build_callback_memory_wrapper(&index_relation, heap_pointer, vec, state, *bq);
            }
        }
        StorageBuildState::Plain(plain, state) => {
            let vec = LabeledVector::from_datums(values, isnull, state.graph.get_meta_page());
            if let Some(vec) = vec {
                build_callback_memory_wrapper(&index_relation, heap_pointer, vec, state, *plain);
            }
        }
    }
}

#[pg_guard]
unsafe extern "C" fn build_callback_parallel(
    index: pg_sys::Relation,
    ctid: pg_sys::ItemPointer,
    values: *mut pg_sys::Datum,
    isnull: *mut bool,
    _tuple_is_alive: bool,
    state: *mut std::os::raw::c_void,
) {
    let heap_pointer = ItemPointer::with_item_pointer_data(*ctid);
    let index_relation = PgRelation::from_pg(index);
    let state = (state as *mut StorageBuildStateParallel).as_mut().unwrap();
    match state {
        StorageBuildStateParallel::SbqSpeedup(bq, state) => {
            let vec = LabeledVector::from_datums(values, isnull, state.graph.get_meta_page());
            if let Some(vec) = vec {
                let spare_vec =
                    LabeledVector::from_datums(values, isnull, state.graph.get_meta_page())
                        .unwrap();
                build_callback_parallel_memory_wrapper(
                    &index_relation,
                    heap_pointer,
                    vec,
                    spare_vec,
                    state,
                    *bq,
                );
            }
        }
        StorageBuildStateParallel::Plain(plain, state) => {
            let vec = LabeledVector::from_datums(values, isnull, state.graph.get_meta_page());
            if let Some(vec) = vec {
                let spare_vec =
                    LabeledVector::from_datums(values, isnull, state.graph.get_meta_page())
                        .unwrap();
                build_callback_parallel_memory_wrapper(
                    &index_relation,
                    heap_pointer,
                    vec,
                    spare_vec,
                    state,
                    *plain,
                );
            }
        }
    }
}

#[inline(always)]
unsafe fn build_callback_memory_wrapper<S: Storage>(
    index: &PgRelation,
    heap_pointer: ItemPointer,
    vector: LabeledVector,
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
    index: &PgRelation,
    heap_pointer: ItemPointer,
    vector: LabeledVector,
    state: &mut BuildState,
    storage: &mut S,
) {
    check_for_interrupts!();

    state.ntuples += 1;

    /*if state.ntuples % 1000 == 0 {
        debug1!(
            "Processed {} tuples in {}s which is {}s/tuple. Dist/tuple: Prune: {} search: {}. Stats: {:?}",
            state.ntuples,
            Instant::now().duration_since(state.started).as_secs_f64(),
            (Instant::now().duration_since(state.started) / state.ntuples as u32).as_secs_f64(),
            state.stats.prune_neighbor_stats.distance_comparisons / state.ntuples,
            state.stats.greedy_search_stats.get_total_distance_comparisons() / state.ntuples,
            state.stats,
        );
    }*/

    let index_pointer = storage.create_node(
        vector.vec().to_index_slice(),
        vector.labels().cloned(),
        heap_pointer,
        state.graph.get_meta_page(),
        &mut state.tape,
        &mut state.stats,
    );

    state
        .graph
        .insert(index, index_pointer, vector, storage, &mut state.stats);
}

#[inline(always)]
unsafe fn build_callback_parallel_memory_wrapper<S: Storage>(
    index: &PgRelation,
    heap_pointer: ItemPointer,
    vector: LabeledVector,
    spare_vector: LabeledVector,
    state: &mut BuildStateParallel,
    storage: &mut S,
) {
    let mut old_context = state.memcxt.set_as_current();

    build_callback_parallel_internal(index, heap_pointer, vector, spare_vector, state, storage);

    old_context.set_as_current();
    state.memcxt.reset();
}

#[inline(always)]
fn build_callback_parallel_internal<S: Storage>(
    index: &PgRelation,
    heap_pointer: ItemPointer,
    vector: LabeledVector,
    spare_vector: LabeledVector,
    state: &mut BuildStateParallel,
    storage: &mut S,
) {
    check_for_interrupts!();

    state.increment_ntuples();

    /*if current_ntuples % 1000 == 0 {
        if let Some(started) = state.get_started() {
            debug1!(
                "Processed {} tuples in {}s which is {}s/tuple. Dist/tuple: Prune: {} search: {}. Stats: {:?}",
                current_ntuples,
                Instant::now().duration_since(started).as_secs_f64(),
                (Instant::now().duration_since(started) / current_ntuples as u32).as_secs_f64(),
                state.local_stats.prune_neighbor_stats.distance_comparisons / current_ntuples,
                state.local_stats.greedy_search_stats.get_total_distance_comparisons() / current_ntuples,
                state.local_stats,
            );
        }
    }*/

    // Create node using local tape - PostgreSQL page locking handles concurrency
    let index_pointer = storage.create_node(
        vector.vec().to_index_slice(),
        vector.labels().cloned(),
        heap_pointer,
        state.graph.get_meta_page(),
        &mut state.tape,
        &mut state.local_stats,
    );

    // Insert node into graph with parallel build mode enabled
    // PostgreSQL page locking handles concurrency when finalizing nodes
    state.graph.insert(
        index,
        index_pointer,
        vector,
        spare_vector,
        storage,
        &mut state.local_stats,
        true,
    );
}

const BUILD_PHASE_TRAINING: i64 = 0;
const BUILD_PHASE_BUILDING_GRAPH: i64 = 1;
const BUILD_PHASE_FINALIZING_GRAPH: i64 = 2;

#[pg_guard]
pub unsafe extern "C-unwind" fn ambuildphasename(phasenum: i64) -> *mut ffi::c_char {
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
        test_index_creation_and_accuracy_scaffold_bounded_memory(
            distance_type,
            index_options,
            name,
            vector_dimensions,
            None,
        )
    }

    #[cfg(any(test, feature = "pg_test"))]
    pub unsafe fn test_index_creation_and_accuracy_scaffold_bounded_memory(
        distance_type: DistanceType,
        index_options: &str,
        name: &str,
        vector_dimensions: usize,
        maintenance_work_mem_kb: Option<usize>,
    ) -> spi::Result<()> {
        let operator = distance_type.get_operator();
        let operator_class = distance_type.get_operator_class();
        let table_name = format!("test_data_icaa_{}", name);
        let maintenance_work_mem_clause =
            if let Some(maintenance_work_mem_kb) = maintenance_work_mem_kb {
                format!("SET maintenance_work_mem = {};", maintenance_work_mem_kb)
            } else {
                "".to_string()
            };

        let sql = format!(
            "CREATE TABLE {table_name} (
                embedding vector ({vector_dimensions})
            );

            {maintenance_work_mem_clause}

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
            FROM generate_series(1, {vector_dimensions}));");
        Spi::run(&sql)?;

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
                &[unsafe { pgrx::datum::DatumWithOid::new(test_vec.clone().into_datum(), pgrx::pg_sys::FLOAT4ARRAYOID) }],
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
            &[unsafe {
                pgrx::datum::DatumWithOid::new(
                    test_vec.clone().into_datum(),
                    pgrx::pg_sys::FLOAT4ARRAYOID,
                )
            }],
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
            &[unsafe {
                pgrx::datum::DatumWithOid::new(
                    test_vec.clone().into_datum(),
                    pgrx::pg_sys::FLOAT4ARRAYOID,
                )
            }],
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
            &[unsafe {
                pgrx::datum::DatumWithOid::new(
                    test_vec.clone().into_datum(),
                    pgrx::pg_sys::FLOAT4ARRAYOID,
                )
            }],
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
            &[unsafe { pgrx::datum::DatumWithOid::new(test_vec.into_datum(), pgrx::pg_sys::FLOAT4ARRAYOID) }],
        )?;

            assert_eq!(cnt.unwrap(), 312);
        }

        Ok(())
    }

    #[pg_test]
    pub unsafe fn test_no_rescore() -> spi::Result<()> {
        // Create a table with 2 vectors.  Create an index on the table before adding
        // data to ensure that SBQ uses default 0 means and therefore cannot distinguish
        // between the vectors.
        Spi::run(
            "CREATE TABLE test(embedding vector(3));

            CREATE INDEX idxtest
                  ON test
               USING diskann(embedding vector_l2_ops);

            INSERT INTO test(embedding) VALUES ('[1,1,1]'), ('[2,2,2]');
            ",
        )?;

        // Query the table with scoring disabled.  The result should be "wrong" for at
        // least one of the queries if rescoring is disabled.
        let res_a: Option<Vec<String>> = Spi::get_one(
            "set enable_seqscan = 0;
            SET diskann.query_rescore = 0;
            WITH cte as (select * from test order by embedding <-> '[1,1,1]' LIMIT 1)
            SELECT array_agg(embedding::text) from cte;",
        )?;
        let wrong_a = res_a.unwrap() != vec!["[1,1,1]"];
        let res_b: Option<Vec<String>> = Spi::get_one(
            "set enable_seqscan = 0;
            SET diskann.query_rescore = 0;
            WITH cte as (select * from test order by embedding <-> '[2,2,2]' LIMIT 1)
            SELECT array_agg(embedding::text) from cte;",
        )?;
        let wrong_b = res_b.unwrap() != vec!["[2,2,2]"];
        assert!(wrong_a || wrong_b);

        // Now query the table with scoring enabled.  The result should be correct for
        // both queries.
        let res_a: Option<Vec<String>> = Spi::get_one(
            "set enable_seqscan = 0;
            SET diskann.query_rescore = 2;
            WITH cte as (select * from test order by embedding <-> '[1,1,1]' LIMIT 1)
            SELECT array_agg(embedding::text) from cte;",
        )?;
        let res_b: Option<Vec<String>> = Spi::get_one(
            "set enable_seqscan = 0;
            SET diskann.query_rescore = 2;
            WITH cte as (select * from test order by embedding <-> '[2,2,2]' LIMIT 1)
            SELECT array_agg(embedding::text) from cte;",
        )?;
        assert_eq!(vec!["[1,1,1]"], res_a.unwrap());
        assert_eq!(vec!["[2,2,2]"], res_b.unwrap());

        Spi::run("DROP INDEX idxtest;")?;

        Ok(())
    }

    #[pg_test]
    pub unsafe fn test_l2_sanity_check() -> spi::Result<()> {
        Spi::run(
            "CREATE TABLE test(embedding vector(3));

            CREATE INDEX idxtest
                  ON test
               USING diskann(embedding vector_l2_ops)
                WITH (num_neighbors=10, search_list_size=10);

            INSERT INTO test(embedding) VALUES ('[1,1,1]'), ('[2,2,2]'), ('[3,3,3]');
            ",
        )?;

        // Query vector [1,1,1] should return [1,1,1]; [2,2,2] should return [2,2,2];
        // and [3,3,3] should return [3,3,3].  (Note that if vectors or the query vector
        // were normalized, then the results would be different.)
        let res: Option<Vec<String>> = Spi::get_one(
            "set enable_seqscan = 0;
            WITH cte as (select * from test order by embedding <-> '[1,1,1]' LIMIT 1)
            SELECT array_agg(embedding::text) from cte;",
        )?;
        assert_eq!(vec!["[1,1,1]"], res.unwrap());

        let res: Option<Vec<String>> = Spi::get_one(
            "set enable_seqscan = 0;
            WITH cte as (select * from test order by embedding <-> '[2,2,2]' LIMIT 1)
            SELECT array_agg(embedding::text) from cte;",
        )?;
        assert_eq!(vec!["[2,2,2]"], res.unwrap());

        let res: Option<Vec<String>> = Spi::get_one(
            "set enable_seqscan = 0;
            WITH cte as (select * from test order by embedding <-> '[3,3,3]' LIMIT 1)
            SELECT array_agg(embedding::text) from cte;",
        )?;
        assert_eq!(vec!["[3,3,3]"], res.unwrap());

        Spi::run("drop index idxtest;")?;

        Ok(())
    }

    #[pg_test]
    pub unsafe fn test_ip_sanity_check() -> spi::Result<()> {
        Spi::run(
            "CREATE TABLE test(embedding vector(3));

            CREATE INDEX idxtest
                  ON test
               USING diskann(embedding vector_ip_ops)
                WITH (num_neighbors=10, search_list_size=10);

            INSERT INTO test(embedding) VALUES ('[1,1,1]'), ('[2,2,2]'), ('[3,3,3]');
            ",
        )?;

        let res: Option<Vec<String>> = Spi::get_one(
            "set enable_seqscan = 0;
            WITH cte as (select * from test order by embedding <#> '[1,1,1]' LIMIT 1)
            SELECT array_agg(embedding::text) from cte;",
        )?;
        assert_eq!(vec!["[3,3,3]"], res.unwrap());

        let res: Option<Vec<String>> = Spi::get_one(
            "set enable_seqscan = 0;
            WITH cte as (select * from test order by embedding <#> '[2,2,2]' LIMIT 1)
            SELECT array_agg(embedding::text) from cte;",
        )?;
        assert_eq!(vec!["[3,3,3]"], res.unwrap());

        let res: Option<Vec<String>> = Spi::get_one(
            "set enable_seqscan = 0;
            WITH cte as (select * from test order by embedding <#> '[3,3,3]' LIMIT 1)
            SELECT array_agg(embedding::text) from cte;",
        )?;
        assert_eq!(vec!["[3,3,3]"], res.unwrap());

        Spi::run("drop index idxtest;")?;

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

        let res: Option<i64> = Spi::get_one("   set enable_seqscan = 0;
                WITH cte as (select * from test order by embedding <=> '[0,0,0]') SELECT count(*) from cte;")?;
        assert_eq!(3, res.unwrap());

        Spi::run(
            "
        set enable_seqscan = 0;
        explain analyze select * from test order by embedding <=> '[0,0,0]';
        ",
        )?;

        Spi::run("drop index idxtest;")?;

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

        let res: Option<i64> = Spi::get_one("   set enable_seqscan = 0;
                WITH cte as (select * from test order by embedding <=> '[0,0,0]') SELECT count(*) from cte;")?;
        assert_eq!(2, res.unwrap());

        Spi::run("drop index idxtest;")?;

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
           -- generate {expected_cnt} vectors
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
            "SELECT('{' || array_to_string(array_agg(1.0), ',', '0') || '}')::real[] AS embedding
    FROM generate_series(1, 1536)",
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
                &[unsafe { pgrx::datum::DatumWithOid::new(test_vec.clone().into_datum(), pgrx::pg_sys::FLOAT4ARRAYOID) }],
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
            &[unsafe { pgrx::datum::DatumWithOid::new(test_vec.clone().into_datum(), pgrx::pg_sys::FLOAT4ARRAYOID) }],
        )?;

        assert!(cnt.unwrap() == expected_cnt, "after update count");

        Ok(())
    }

    pub fn verify_index_accuracy(
        expected_cnt: i64,
        dimensions: usize,
        table_name: &str,
    ) -> spi::Result<()> {
        let test_vec: Option<Vec<f32>> = Spi::get_one(&format!(
            "SELECT('{{' || array_to_string(array_agg(1.0), ',', '0') || '}}')::real[] AS embedding
    FROM generate_series(1, {dimensions})"
        ))?;

        let cnt: Option<i64> = Spi::get_one_with_args(
                    &format!("
            SET enable_seqscan = 0;
            SET enable_indexscan = 1;
            SET diskann.query_search_list_size = 2;
            WITH cte as (select * from {table_name} order by embedding <=> $1::vector) SELECT count(*) from cte;
            "),
                &[unsafe { pgrx::datum::DatumWithOid::new(test_vec.clone().into_datum(), pgrx::pg_sys::FLOAT4ARRAYOID) }],
            )?;

        if cnt.unwrap() != expected_cnt {
            // better debugging
            let id: Option<String> = Spi::get_one_with_args(
                    &format!("
            SET enable_seqscan = 0;
            SET enable_indexscan = 1;
            SET diskann.query_search_list_size = 2;
            WITH cte as (select id from {table_name} EXCEPT (select id from {table_name} order by embedding <=> $1::vector)) SELECT ctid::text || ' ' || id from {table_name} where id in (select id from cte limit 1);
            "),
                &[unsafe { pgrx::datum::DatumWithOid::new(test_vec.clone().into_datum(), pgrx::pg_sys::FLOAT4ARRAYOID) }],
            )?;

            assert_eq!(cnt.unwrap(), expected_cnt, "id is {}", id.unwrap());
        }

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
           -- generate {expected_cnt} vectors
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

        verify_index_accuracy(expected_cnt, dimensions, "test_data")?;
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

        // Cleanup any leftover state from previous test runs
        let _ = Spi::run("DROP TABLE IF EXISTS test_data_insert_after CASCADE;");

        Spi::run(&format!(
            "CREATE TABLE test_data_insert_after (
                id int,
                embedding vector ({dimensions})
            );

            CREATE INDEX idx_diskann_insert_after ON test_data_insert_after USING diskann (embedding) WITH ({index_options});

            select setseed(0.5);
           -- generate {expected_cnt} vectors
            INSERT INTO test_data_insert_after (id, embedding)
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
                test_data_insert_after
            ORDER BY
                embedding <=> (
                    SELECT
                        ('[' || array_to_string(array_agg(random()), ',', '0') || ']')::vector AS embedding
            FROM generate_series(1, {dimensions}));"))?;

        verify_index_accuracy(expected_cnt, dimensions, "test_data_insert_after")?;
        Spi::run("DROP TABLE test_data_insert_after CASCADE;")?;
        Ok(())
    }

    #[pg_test]
    pub unsafe fn test_high_dimension_index() -> spi::Result<()> {
        for dimensions in [4000, 8000, 12000, 16000] {
            test_sized_index_scaffold(
                "num_neighbors=20, search_list_size=10",
                dimensions,
                1000,
                None,
            )?;
        }
        Ok(())
    }

    #[cfg(any(test, feature = "pg_test"))]
    pub unsafe fn test_sized_index_scaffold(
        index_options: &str,
        dimensions: usize,
        vector_count: usize,
        maintenance_work_mem_kb: Option<usize>,
    ) -> spi::Result<()> {
        let maintenance_work_mem_kb = maintenance_work_mem_kb.unwrap_or(64 * 1024);
        Spi::run(&format!(
            "
            CREATE TABLE test_data (
                id int,
                embedding vector ({dimensions})
            );

            select setseed(0.5);

            -- generate {vector_count} vectors
            INSERT INTO test_data (id, embedding)
            SELECT
                *
            FROM (
                SELECT
                    i % {vector_count},
                    ('[' || array_to_string(array_agg(random()), ',', '0') || ']')::vector AS embedding
                FROM
                    generate_series(1, {dimensions} * {vector_count}) i
                GROUP BY
                    i % {vector_count}) g;

            SET enable_seqscan = 0;

            SET maintenance_work_mem = {maintenance_work_mem_kb};
            CREATE INDEX ON test_data USING diskann (embedding) WITH ({index_options});

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

        verify_index_accuracy(vector_count as i64, dimensions, "test_data")?;

        Spi::run("DROP TABLE test_data CASCADE;")?;
        Ok(())
    }

    #[pg_test]
    pub unsafe fn test_labeled_index() -> spi::Result<()> {
        let index_options = "num_neighbors=15, search_list_size=10";
        let expected_cnt = 50;
        let dimension = 128;

        // Cleanup any leftover state from previous test runs
        let _ = Spi::run("DROP TABLE IF EXISTS test_data_labeled CASCADE;");

        let query = &format!(
            "CREATE TABLE test_data_labeled (
                id int,
                embedding vector ({dimension}),
                labels smallint[]
            );

            CREATE INDEX idx_diskann_labeled ON test_data_labeled USING diskann (embedding, labels) WITH ({index_options});

            select setseed(0.5);

            -- generate {expected_cnt} vectors along with labels
            INSERT INTO test_data_labeled (id, embedding, labels)
            SELECT
                gs AS id,
                ARRAY[
                    (random() * 2 - 1)::float8, (random() * 2 - 1)::float8, (random() * 2 - 1)::float8,
                    (random() * 2 - 1)::float8, (random() * 2 - 1)::float8, (random() * 2 - 1)::float8,
                    (random() * 2 - 1)::float8, (random() * 2 - 1)::float8, (random() * 2 - 1)::float8,
                    (random() * 2 - 1)::float8, (random() * 2 - 1)::float8, (random() * 2 - 1)::float8,
                    (random() * 2 - 1)::float8, (random() * 2 - 1)::float8, (random() * 2 - 1)::float8,
                    (random() * 2 - 1)::float8, (random() * 2 - 1)::float8, (random() * 2 - 1)::float8,
                    (random() * 2 - 1)::float8, (random() * 2 - 1)::float8, (random() * 2 - 1)::float8,
                    (random() * 2 - 1)::float8, (random() * 2 - 1)::float8, (random() * 2 - 1)::float8,
                    (random() * 2 - 1)::float8, (random() * 2 - 1)::float8, (random() * 2 - 1)::float8,
                    (random() * 2 - 1)::float8, (random() * 2 - 1)::float8, (random() * 2 - 1)::float8,
                    (random() * 2 - 1)::float8, (random() * 2 - 1)::float8, (random() * 2 - 1)::float8,
                    (random() * 2 - 1)::float8, (random() * 2 - 1)::float8, (random() * 2 - 1)::float8,
                    (random() * 2 - 1)::float8, (random() * 2 - 1)::float8, (random() * 2 - 1)::float8,
                    (random() * 2 - 1)::float8, (random() * 2 - 1)::float8, (random() * 2 - 1)::float8,
                    (random() * 2 - 1)::float8, (random() * 2 - 1)::float8, (random() * 2 - 1)::float8,
                    (random() * 2 - 1)::float8, (random() * 2 - 1)::float8, (random() * 2 - 1)::float8,
                    (random() * 2 - 1)::float8, (random() * 2 - 1)::float8, (random() * 2 - 1)::float8,
                    (random() * 2 - 1)::float8, (random() * 2 - 1)::float8, (random() * 2 - 1)::float8,
                    (random() * 2 - 1)::float8, (random() * 2 - 1)::float8, (random() * 2 - 1)::float8,
                    (random() * 2 - 1)::float8, (random() * 2 - 1)::float8, (random() * 2 - 1)::float8,
                    (random() * 2 - 1)::float8, (random() * 2 - 1)::float8, (random() * 2 - 1)::float8,
                    (random() * 2 - 1)::float8, (random() * 2 - 1)::float8, (random() * 2 - 1)::float8,
                    (random() * 2 - 1)::float8, (random() * 2 - 1)::float8, (random() * 2 - 1)::float8,
                    (random() * 2 - 1)::float8, (random() * 2 - 1)::float8, (random() * 2 - 1)::float8,
                    (random() * 2 - 1)::float8, (random() * 2 - 1)::float8, (random() * 2 - 1)::float8,
                    (random() * 2 - 1)::float8, (random() * 2 - 1)::float8, (random() * 2 - 1)::float8,
                    (random() * 2 - 1)::float8, (random() * 2 - 1)::float8, (random() * 2 - 1)::float8,
                    (random() * 2 - 1)::float8, (random() * 2 - 1)::float8, (random() * 2 - 1)::float8,
                    (random() * 2 - 1)::float8, (random() * 2 - 1)::float8, (random() * 2 - 1)::float8,
                    (random() * 2 - 1)::float8, (random() * 2 - 1)::float8, (random() * 2 - 1)::float8,
                    (random() * 2 - 1)::float8, (random() * 2 - 1)::float8, (random() * 2 - 1)::float8,
                    (random() * 2 - 1)::float8, (random() * 2 - 1)::float8, (random() * 2 - 1)::float8,
                    (random() * 2 - 1)::float8, (random() * 2 - 1)::float8, (random() * 2 - 1)::float8,
                    (random() * 2 - 1)::float8, (random() * 2 - 1)::float8, (random() * 2 - 1)::float8,
                    (random() * 2 - 1)::float8, (random() * 2 - 1)::float8, (random() * 2 - 1)::float8,
                    (random() * 2 - 1)::float8, (random() * 2 - 1)::float8, (random() * 2 - 1)::float8,
                    (random() * 2 - 1)::float8, (random() * 2 - 1)::float8, (random() * 2 - 1)::float8,
                    (random() * 2 - 1)::float8, (random() * 2 - 1)::float8, (random() * 2 - 1)::float8,
                    (random() * 2 - 1)::float8, (random() * 2 - 1)::float8, (random() * 2 - 1)::float8,
                    (random() * 2 - 1)::float8, (random() * 2 - 1)::float8, (random() * 2 - 1)::float8,
                    (random() * 2 - 1)::float8, (random() * 2 - 1)::float8, (random() * 2 - 1)::float8,
                    (random() * 2 - 1)::float8, (random() * 2 - 1)::float8, (random() * 2 - 1)::float8,
                    (random() * 2 - 1)::float8, (random() * 2 - 1)::float8
                ]::vector(128),
                ARRAY[
                    (floor(random() * 16 + 1))::smallint,
                    (floor(random() * 16 + 1))::smallint
                ]
            FROM generate_series(1, {expected_cnt}) gs;

            SET enable_seqscan = 0;
            -- perform index scans on the vectors
            SELECT
                *
            FROM
                test_data_labeled
            ORDER BY
                embedding <=> (
                    SELECT
                        ('[' || array_to_string(array_agg(random()), ',', '0') || ']')::vector AS embedding
            FROM generate_series(1, {dimension}));");

        warning!("Running query: {}", query);
        Spi::run(query)?;

        verify_index_accuracy(expected_cnt, dimension, "test_data_labeled")?;

        Spi::run("DROP TABLE test_data_labeled CASCADE;")?;
        Ok(())
    }

    #[pg_test]
    pub unsafe fn test_null_vector_scan() -> spi::Result<()> {
        // Test for issue #238 - NULL vectors should not crash index scans
        // Instead the index scan should return all vectors in some arbitrary order.

        Spi::run(
            "CREATE TABLE test(embedding vector(3));

            CREATE INDEX idxtest
                  ON test
               USING diskann(embedding vector_l2_ops)
                WITH (num_neighbors=10, search_list_size=10);

            INSERT INTO test(embedding) VALUES ('[1,1,1]'), ('[2,2,2]'), ('[3,3,3]');
            ",
        )?;

        // Scan the table with a NULL vector - this should not crash
        // The main goal is to verify NULL vector handling doesn't cause segfaults
        let count: Option<i64> = Spi::get_one(
            "set enable_seqscan = 0;
            SELECT COUNT(*) FROM (SELECT embedding FROM test ORDER BY embedding <-> NULL LIMIT 3) t;",
        )?;

        // Should return 3 rows (all vectors, since the index scan completes successfully)
        assert_eq!(count, Some(3));
        // Clean up
        Spi::run("DROP TABLE test CASCADE;")?;
        Ok(())
    }
}
