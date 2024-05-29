use pgrx::*;

pub static TSV_QUERY_SEARCH_LIST_SIZE: GucSetting<i32> = GucSetting::<i32>::new(100);
pub static TSV_RESORT_SIZE: GucSetting<i32> = GucSetting::<i32>::new(50);

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

    GucRegistry::define_int_guc(
        "tsv.query_rescore",
        "The number of elements rescored (0 to disable rescoring)",
        "Rescoring takes the query_rescore number of elements that have the smallest approximate distance, rescores them with the exact distance, returning the closest ones with the exact distance.",
        &TSV_RESORT_SIZE,
        1,
        1000,
        GucContext::Userset,
        GucFlags::default(),
    );
}
