use pgrx::*;
mod build;
mod cost_estimate;
mod debugging;
mod graph;
mod graph_neighbor_store;
pub mod guc;
mod meta_page;
mod neighbor_with_distance;
pub mod options;
pub mod pg_vector;
mod plain_node;
mod plain_storage;
mod scan;
pub mod stats;
mod storage;
mod storage_common;
mod upgrade_test;
mod vacuum;

pub mod distance;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod distance_x86;
mod sbq;

/// Access method support function numbers
pub const DISKANN_DISTANCE_TYPE_PROC: u16 = 1;

#[pg_extern(sql = "
    CREATE OR REPLACE FUNCTION diskann_amhandler(internal) RETURNS index_am_handler PARALLEL SAFE IMMUTABLE STRICT COST 0.0001 LANGUAGE c AS '@MODULE_PATHNAME@', '@FUNCTION_NAME@';

    DO $$
    DECLARE
        c int;
    BEGIN
        SELECT count(*)
        INTO c
        FROM pg_catalog.pg_am a
        WHERE a.amname = 'diskann';

        IF c = 0 THEN
            CREATE ACCESS METHOD diskann TYPE INDEX HANDLER diskann_amhandler;
        END IF;
    END;
    $$;
")]
fn amhandler(_fcinfo: pg_sys::FunctionCallInfo) -> PgBox<pg_sys::IndexAmRoutine> {
    let mut amroutine =
        unsafe { PgBox::<pg_sys::IndexAmRoutine>::alloc_node(pg_sys::NodeTag::T_IndexAmRoutine) };

    amroutine.amstrategies = 0;
    amroutine.amsupport = 1;

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
    amroutine.amoptsprocnum = 0;
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

    amroutine.ambuildphasename = Some(build::ambuildphasename);

    amroutine.into_pg_boxed()
}

// This SQL is made idempotent so that we can use the same script for the installation and the upgrade.
extension_sql!(
    r#"
    DROP OPERATOR CLASS IF EXISTS vector_cosine_ops USING diskann;

    CREATE OPERATOR CLASS vector_cosine_ops
        DEFAULT FOR TYPE vector USING diskann AS
        OPERATOR 1 <=> (vector, vector) FOR ORDER BY float_ops,
        FUNCTION 1 distance_type_cosine();


    DROP OPERATOR CLASS IF EXISTS vector_l2_ops USING diskann;

    CREATE OPERATOR CLASS vector_l2_ops
    FOR TYPE vector USING diskann AS
        OPERATOR 1 <-> (vector, vector) FOR ORDER BY float_ops,
        FUNCTION 1 distance_type_l2();
"#,
    name = "diskann_ops_operator",
    requires = [amhandler, distance_type_cosine, distance_type_l2]
);

#[pg_guard]
pub extern "C" fn amvalidate(_opclassoid: pg_sys::Oid) -> bool {
    true
}
