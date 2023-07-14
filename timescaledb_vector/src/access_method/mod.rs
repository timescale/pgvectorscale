use pgrx::*;

mod build;
mod builder_graph;
mod cost_estimate;
mod debugging;
mod disk_index_graph;
mod graph;
mod model;
pub mod options;
mod scan;
mod vacuum;

#[pg_extern(sql = "
    CREATE OR REPLACE FUNCTION tsv_amhandler(internal) RETURNS index_am_handler PARALLEL SAFE IMMUTABLE STRICT COST 0.0001 LANGUAGE c AS 'MODULE_PATHNAME', '@FUNCTION_NAME@';
    CREATE ACCESS METHOD tsv TYPE INDEX HANDLER tsv_amhandler;
")]
fn amhandler(_fcinfo: pg_sys::FunctionCallInfo) -> PgBox<pg_sys::IndexAmRoutine> {
    let mut amroutine =
        unsafe { PgBox::<pg_sys::IndexAmRoutine>::alloc_node(pg_sys::NodeTag_T_IndexAmRoutine) };

    amroutine.amstrategies = 0;
    amroutine.amsupport = 0; //TODO
    amroutine.amoptsprocnum = 0;

    amroutine.amcanorder = false;
    amroutine.amcanorderbyop = true;
    amroutine.amcanbackward = false; /* can change direction mid-scan */
    amroutine.amcanunique = false;
    amroutine.amcanmulticol = false;
    amroutine.amoptionalkey = true;
    amroutine.amsearcharray = false;
    amroutine.amsearchnulls = false;
    amroutine.amstorage = false;
    amroutine.amclusterable = false;
    amroutine.ampredlocks = false;
    amroutine.amcanparallel = false; //TODO
    amroutine.amcaninclude = false; //TODO
    amroutine.amusemaintenanceworkmem = false; /* not used during VACUUM */
    //amroutine.amparallelvacuumoptions = pg_sys  VACUUM_OPTION_PARALLEL_BULKDEL; //TODO
    amroutine.amkeytype = pg_sys::InvalidOid;

    amroutine.amvalidate = Some(amvalidate);
    amroutine.ambuild = Some(build::ambuild);
    amroutine.ambuildempty = Some(build::ambuildempty);
    amroutine.aminsert = Some(build::aminsert);
    amroutine.ambulkdelete = Some(vacuum::ambulkdelete);
    amroutine.amvacuumcleanup = Some(vacuum::amvacuumcleanup);
    amroutine.amcostestimate = Some(cost_estimate::amcostestimate);
    amroutine.amoptions = Some(options::amoptions);
    amroutine.ambeginscan = Some(scan::ambeginscan);
    amroutine.amrescan = Some(scan::amrescan);
    amroutine.amgettuple = Some(scan::amgettuple);
    amroutine.amgetbitmap = None;
    amroutine.amendscan = Some(scan::amendscan);

    amroutine.into_pg_boxed()
}

extension_sql!(
    r#"
CREATE OPERATOR CLASS vector_cosine_ops DEFAULT
FOR TYPE vector USING tsv AS
	OPERATOR 1 <=> (vector, vector) FOR ORDER BY float_ops
;
"#,
    name = "tsv_ops_operator"
);

#[pg_guard]
pub extern "C" fn amvalidate(_opclassoid: pg_sys::Oid) -> bool {
    true
}
