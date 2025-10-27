use pgrx::*;

/// cost estimate function loosely based on how ivfflat does things
#[pg_guard(immutable, parallel_safe)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C-unwind" fn amcostestimate(
    root: *mut pg_sys::PlannerInfo,
    path: *mut pg_sys::IndexPath,
    loop_count: f64,
    index_startup_cost: *mut pg_sys::Cost,
    index_total_cost: *mut pg_sys::Cost,
    index_selectivity: *mut pg_sys::Selectivity,
    index_correlation: *mut f64,
    index_pages: *mut f64,
) {
    if (*path).indexorderbys.is_null() {
        // Can't use index without order-bys
        *index_startup_cost = f64::MAX;
        *index_total_cost = f64::MAX;
        *index_selectivity = 0.;
        *index_correlation = 0.;
        *index_pages = 0.;
        #[cfg(any(feature = "pg18"))]
        {
            // Following in the footsteps of pgvector's PG18+ cost estimate change
            // https://github.com/pgvector/pgvector/commit/1291b12090bbb03bd92b92e42a1567ae5b1c96ad
            (*path).path.disabled_nodes = 2;
        }
        return;
    }
    let path_ref = path.as_ref().expect("path argument is NULL");

    let total_index_tuples = (*path_ref.indexinfo).tuples;

    let mut generic_costs = pg_sys::GenericCosts {
        numIndexTuples: total_index_tuples / 100., //TODO need better estimate
        ..Default::default()
    };

    pg_sys::genericcostestimate(root, path, loop_count, &mut generic_costs);

    //TODO probably have to adjust costs more here

    *index_startup_cost = generic_costs.indexTotalCost;
    *index_total_cost = generic_costs.indexTotalCost;
    *index_selectivity = generic_costs.indexSelectivity;
    *index_correlation = generic_costs.indexCorrelation;
    *index_pages = generic_costs.numIndexPages;
    //pg_sys::cpu_index_tuple_cost;
}
