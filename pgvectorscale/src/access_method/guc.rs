use pgrx::{pg_sys::AsPgCStr, *};

pub static TSV_QUERY_SEARCH_LIST_SIZE: GucSetting<i32> = GucSetting::<i32>::new(100);
pub static TSV_RESORT_SIZE: GucSetting<i32> = GucSetting::<i32>::new(50);

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
}
