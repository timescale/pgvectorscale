use memoffset::*;
use pgrx::pg_sys::AsPgCStr;
use pgrx::prelude::*;
use pgrx::*;
use std::fmt::Debug;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct TSVIndexOptions {
    /* varlena header (do not touch directly!) */
    #[allow(dead_code)]
    vl_len_: i32,

    placeholder: i32,
}

impl TSVIndexOptions {
    pub fn from_relation(relation: &PgRelation) -> PgBox<TSVIndexOptions> {
        if relation.rd_index.is_null() {
            panic!("'{}' is not a TSV index", relation.name())
        } else if relation.rd_options.is_null() {
            // use defaults
            let mut ops = unsafe { PgBox::<TSVIndexOptions>::alloc0() };
            ops.placeholder = 47;
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

const NUM_REL_OPTS: usize = 1;
static mut RELOPT_KIND_TSV: pg_sys::relopt_kind = 0;

#[allow(clippy::unneeded_field_pattern)] // b/c of offset_of!()
#[pg_guard]
pub unsafe extern "C" fn amoptions(
    reloptions: pg_sys::Datum,
    validate: bool,
) -> *mut pg_sys::bytea {
    // TODO:  how to make this const?  we can't use offset_of!() macro in const definitions, apparently
    let tab: [pg_sys::relopt_parse_elt; NUM_REL_OPTS] = [pg_sys::relopt_parse_elt {
        optname: "placeholder".as_pg_cstr(),
        opttype: pg_sys::relopt_type_RELOPT_TYPE_INT,
        offset: offset_of!(TSVIndexOptions, placeholder) as i32,
    }];

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
        "placeholder".as_pg_cstr(),
        "Placeholder option".as_pg_cstr(),
        47,
        0,
        50,
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
            "CREATE TABLE test(encoding vector);
        CREATE INDEX idxtest
                  ON test
               USING tsv(encoding)
                WITH (placeholder=30);",
        ))?;

        let index_oid =
            Spi::get_one::<pg_sys::Oid>("SELECT 'idxtest'::regclass::oid")?.expect("oid was null");
        let indexrel = PgRelation::from_pg(pg_sys::RelationIdGetRelation(index_oid));
        let options = TSVIndexOptions::from_relation(&indexrel);
        assert_eq!(options.placeholder, 30);
        Ok(())
    }

    #[pg_test]
    unsafe fn test_index_options_defaults() -> spi::Result<()> {
        Spi::run(&format!(
            "CREATE TABLE test(encoding vector);
        CREATE INDEX idxtest
                  ON test
               USING tsv(encoding);",
        ))?;

        let index_oid =
            Spi::get_one::<pg_sys::Oid>("SELECT 'idxtest'::regclass::oid")?.expect("oid was null");
        let indexrel = PgRelation::from_pg(pg_sys::RelationIdGetRelation(index_oid));
        let options = TSVIndexOptions::from_relation(&indexrel);
        assert_eq!(options.placeholder, 47);
        Ok(())
    }
}
