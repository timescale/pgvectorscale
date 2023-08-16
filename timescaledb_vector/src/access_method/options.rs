use memoffset::*;
use pgrx::pg_sys::AsPgCStr;
use pgrx::prelude::*;
use pgrx::*;
use std::fmt::Debug;

#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct TSVIndexOptions {
    /* varlena header (do not touch directly!) */
    #[allow(dead_code)]
    vl_len_: i32,

    pub num_neighbors: u32,
    pub search_list_size: u32,
    pub max_alpha: f64,
    pub use_pq: bool,
}

impl TSVIndexOptions {
    pub fn from_relation(relation: &PgRelation) -> PgBox<TSVIndexOptions> {
        if relation.rd_index.is_null() {
            panic!("'{}' is not a TSV index", relation.name())
        } else if relation.rd_options.is_null() {
            // use defaults
            let mut ops = unsafe { PgBox::<TSVIndexOptions>::alloc0() };
            ops.num_neighbors = 50;
            ops.search_list_size = 65;
            ops.max_alpha = 1.0;
            ops.use_pq = false;
            unsafe {
                set_varsize(
                    ops.as_ptr().cast(),
                    std::mem::size_of::<TSVIndexOptions>() as i32,
                );
            }
            ops.into_pg_boxed()
        } else {
            unsafe { PgBox::from_pg(relation.rd_options as *mut TSVIndexOptions) }
        }
    }
}

const NUM_REL_OPTS: usize = 4;
static mut RELOPT_KIND_TSV: pg_sys::relopt_kind = 0;

#[allow(clippy::unneeded_field_pattern)] // b/c of offset_of!()
#[pg_guard]
pub unsafe extern "C" fn amoptions(
    reloptions: pg_sys::Datum,
    validate: bool,
) -> *mut pg_sys::bytea {
    // TODO:  how to make this const?  we can't use offset_of!() macro in const definitions, apparently
    let tab: [pg_sys::relopt_parse_elt; NUM_REL_OPTS] = [
        pg_sys::relopt_parse_elt {
            optname: "num_neighbors".as_pg_cstr(),
            opttype: pg_sys::relopt_type_RELOPT_TYPE_INT,
            offset: offset_of!(TSVIndexOptions, num_neighbors) as i32,
        },
        pg_sys::relopt_parse_elt {
            optname: "search_list_size".as_pg_cstr(),
            opttype: pg_sys::relopt_type_RELOPT_TYPE_INT,
            offset: offset_of!(TSVIndexOptions, search_list_size) as i32,
        },
        pg_sys::relopt_parse_elt {
            optname: "max_alpha".as_pg_cstr(),
            opttype: pg_sys::relopt_type_RELOPT_TYPE_REAL,
            offset: offset_of!(TSVIndexOptions, max_alpha) as i32,
        },
        pg_sys::relopt_parse_elt {
            optname: "use_pq".as_pg_cstr(),
            opttype: pg_sys::relopt_type_RELOPT_TYPE_BOOL,
            offset: offset_of!(TSVIndexOptions, use_pq) as i32,
        },
    ];

    build_relopts(reloptions, validate, tab)
}

#[cfg(any(feature = "pg13", feature = "pg14", feature = "pg15"))]
unsafe fn build_relopts(
    reloptions: pg_sys::Datum,
    validate: bool,
    tab: [pg_sys::relopt_parse_elt; NUM_REL_OPTS],
) -> *mut pg_sys::bytea {
    let rdopts;

    /* Parse the user-given reloptions */
    rdopts = pg_sys::build_reloptions(
        reloptions,
        validate,
        RELOPT_KIND_TSV,
        std::mem::size_of::<TSVIndexOptions>(),
        tab.as_ptr(),
        NUM_REL_OPTS as i32,
    );

    rdopts as *mut pg_sys::bytea
}

pub unsafe fn init() {
    RELOPT_KIND_TSV = pg_sys::add_reloption_kind();

    pg_sys::add_int_reloption(
        RELOPT_KIND_TSV,
        "num_neighbors".as_pg_cstr(),
        "Maximum number of neighbors in the graph".as_pg_cstr(),
        50,
        10,
        1000,
        pg_sys::AccessExclusiveLock as pg_sys::LOCKMODE,
    );

    pg_sys::add_int_reloption(
        RELOPT_KIND_TSV,
        "search_list_size".as_pg_cstr(),
        "The search list size to use during a build".as_pg_cstr(),
        65,
        10,
        1000,
        pg_sys::AccessExclusiveLock as pg_sys::LOCKMODE,
    );

    pg_sys::add_real_reloption(
        RELOPT_KIND_TSV,
        "max_alpha".as_pg_cstr(),
        "The maximum alpha used in pruning".as_pg_cstr(),
        1.0,
        1.0,
        5.0,
        pg_sys::AccessExclusiveLock as pg_sys::LOCKMODE,
    );
    pg_sys::add_bool_reloption(
        RELOPT_KIND_TSV,
        "use_pq".as_pg_cstr(),
        "Enable product quantization".as_pg_cstr(),
        false,
        pg_sys::AccessExclusiveLock as pg_sys::LOCKMODE,
    );
}

#[cfg(any(test, feature = "pg_test"))]
#[pgrx::pg_schema]
mod tests {
    use crate::access_method::options::TSVIndexOptions;
    use pgrx::*;

    #[pg_test]
    unsafe fn test_index_options() -> spi::Result<()> {
        Spi::run(&format!(
            "CREATE TABLE test(encoding vector(3));
        CREATE INDEX idxtest
                  ON test
               USING tsv(encoding)
                WITH (num_neighbors=30);",
        ))?;

        let index_oid =
            Spi::get_one::<pg_sys::Oid>("SELECT 'idxtest'::regclass::oid")?.expect("oid was null");
        let indexrel = PgRelation::from_pg(pg_sys::RelationIdGetRelation(index_oid));
        let options = TSVIndexOptions::from_relation(&indexrel);
        assert_eq!(options.num_neighbors, 30);
        Ok(())
    }

    #[pg_test]
    unsafe fn test_index_options_defaults() -> spi::Result<()> {
        Spi::run(&format!(
            "CREATE TABLE test(encoding vector(3));
        CREATE INDEX idxtest
                  ON test
               USING tsv(encoding);",
        ))?;

        let index_oid =
            Spi::get_one::<pg_sys::Oid>("SELECT 'idxtest'::regclass::oid")?.expect("oid was null");
        let indexrel = PgRelation::from_pg(pg_sys::RelationIdGetRelation(index_oid));
        let options = TSVIndexOptions::from_relation(&indexrel);
        assert_eq!(options.num_neighbors, 50);
        assert_eq!(options.search_list_size, 65);
        assert_eq!(options.max_alpha, 1.0);
        assert_eq!(options.use_pq, false);
        Ok(())
    }
}
