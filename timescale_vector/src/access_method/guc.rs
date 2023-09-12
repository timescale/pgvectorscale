use pgrx::*;

pub static TSV_QUERY_SEARCH_LIST_SIZE: GucSetting<i32> = GucSetting::<i32>::new(100);

pub fn init() {
    GucRegistry::define_int_guc(
        "tsv.query_search_list_size",
        "The size of the search list used in queries",
        "Higher value increases recall at the cost of speed.",
        &TSV_QUERY_SEARCH_LIST_SIZE,
        1,
        10000,
        GucContext::Userset,
        GucFlags::default(),
    );
}
