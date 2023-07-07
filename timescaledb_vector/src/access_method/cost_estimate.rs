use pgrx::*;

//TODO

#[pg_guard(immutable, parallel_safe)]
pub unsafe extern "C" fn amcostestimate(
    _root: *mut pg_sys::PlannerInfo,
    path: *mut pg_sys::IndexPath,
    _loop_count: f64,
    index_startup_cost: *mut pg_sys::Cost,
    index_total_cost: *mut pg_sys::Cost,
    index_selectivity: *mut pg_sys::Selectivity,
    index_correlation: *mut f64,
    index_pages: *mut f64,
) {
    let path = path.as_ref().expect("path argument is NULL");
    let indexinfo = path.indexinfo.as_ref().expect("indexinfo in path is NULL");
    let index_relation = unsafe {
        PgRelation::with_lock(
            indexinfo.indexoid,
            pg_sys::AccessShareLock as pg_sys::LOCKMODE,
        )
    };
    let heap_relation = index_relation
        .heap_relation()
        .expect("failed to get heap relation for index");

    *index_correlation = 1.0;
    *index_startup_cost = 0.0;
    *index_pages = 0.0;
    *index_total_cost = 0.0;

    *index_selectivity = 1.0;

    let index_clauses = PgList::<pg_sys::IndexClause>::from_pg(path.indexclauses);

    for clause in index_clauses.iter_ptr() {
        let ri = clause
            .as_ref()
            .unwrap()
            .rinfo
            .as_ref()
            .expect("restrict info in index clause is NULL");

        if ri.norm_selec > 0f64 {
            *index_selectivity = ri.norm_selec.min(*index_selectivity);
        }
    }

    let reltuples = heap_relation.reltuples().unwrap_or(1f32) as f64;
    *index_total_cost += *index_selectivity * reltuples * pg_sys::cpu_index_tuple_cost;
}
