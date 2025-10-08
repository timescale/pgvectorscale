use pgrx::{pg_sys::AsPgCStr, *};

pub static TSV_QUERY_SEARCH_LIST_SIZE: GucSetting<i32> = GucSetting::<i32>::new(100);
pub static TSV_RESORT_SIZE: GucSetting<i32> = GucSetting::<i32>::new(50);
pub static TSV_PARALLEL_FLUSH_RATE: GucSetting<i32> = GucSetting::<i32>::new(4096);
pub static TSV_PARALLEL_INITIAL_START_NODES_COUNT: GucSetting<i32> = GucSetting::<i32>::new(1024);
pub static TSV_MIN_VECTORS_FOR_PARALLEL_BUILD: GucSetting<i32> = GucSetting::<i32>::new(65536);

pub fn init() {
    GucRegistry::define_int_guc(
        unsafe { std::ffi::CStr::from_ptr("diskann.query_search_list_size".as_pg_cstr()) },
        unsafe {
            std::ffi::CStr::from_ptr("The size of the search list used in queries".as_pg_cstr())
        },
        unsafe {
            std::ffi::CStr::from_ptr(
                "Higher value increases recall at the cost of speed.".as_pg_cstr(),
            )
        },
        &TSV_QUERY_SEARCH_LIST_SIZE,
        1,
        10000,
        GucContext::Userset,
        GucFlags::default(),
    );

    GucRegistry::define_int_guc(
        unsafe { std::ffi::CStr::from_ptr("diskann.query_rescore".as_pg_cstr()) },
        unsafe {
            std::ffi::CStr::from_ptr(
                "The number of elements rescored (0 to disable rescoring)".as_pg_cstr(),
            )
        },
        unsafe {
            std::ffi::CStr::from_ptr("Rescoring takes the query_rescore number of elements that have the smallest approximate distance, rescores them with the exact distance, returning the closest ones with the exact distance.".as_pg_cstr())
        },
        &TSV_RESORT_SIZE,
        0,
        1000,
        GucContext::Userset,
        GucFlags::default(),
    );

    GucRegistry::define_int_guc(
        "diskann.parallel_flush_rate",
        "The number of tuples processed before flushing neighbor cache in parallel builds",
        "Controls how often the neighbor cache is flushed during parallel index builds. Higher values use more memory but may improve performance.",
        &TSV_PARALLEL_FLUSH_RATE,
        1,
        100000,
        GucContext::Suset,
        GucFlags::default(),
    );

    GucRegistry::define_int_guc(
        "diskann.parallel_initial_start_nodes_count",
        "The number of initial start nodes to process before starting parallel workers",
        "Determines how many nodes the initializing worker processes before other workers begin. Affects parallel build coordination.",
        &TSV_PARALLEL_INITIAL_START_NODES_COUNT,
        1,
        10000,
        GucContext::Suset,
        GucFlags::default(),
    );

    GucRegistry::define_int_guc(
        "diskann.min_vectors_for_parallel_build",
        "Minimum number of vectors required to enable parallel building",
        "If the table has fewer vectors than this threshold, parallel building will be disabled and serial building will be used instead.",
        &TSV_MIN_VECTORS_FOR_PARALLEL_BUILD,
        1,
        i32::MAX,
        GucContext::Suset,
        GucFlags::default(),
    );
}
