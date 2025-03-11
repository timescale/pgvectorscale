use memoffset::*;
use pgrx::{pg_sys::AsPgCStr, prelude::*, set_varsize_4b, void_ptr, PgRelation};
use std::{ffi::CStr, fmt::Debug};

use super::storage::StorageType;

//DO NOT derive Clone for this struct. The storage layout string comes at the end and wouldn't be copied properly.
#[derive(Debug, PartialEq)]
#[repr(C)]
pub struct TSVIndexOptions {
    /* varlena header (do not touch directly!) */
    #[allow(dead_code)]
    vl_len_: i32,

    pub storage_layout_offset: i32,
    num_neighbors: i32,
    pub search_list_size: u32,
    pub num_dimensions: u32,
    pub max_alpha: f64,
    pub bq_num_bits_per_dimension: u32,
}

pub const NUM_NEIGHBORS_DEFAULT_SENTINEL: i32 = -1;
pub const NUM_DIMENSIONS_DEFAULT_SENTINEL: u32 = 0;
pub const SBQ_NUM_BITS_PER_DIMENSION_DEFAULT_SENTINEL: u32 = 0;
const DEFAULT_MAX_ALPHA: f64 = 1.2;

impl TSVIndexOptions {
    //note: this should only be used when building a new index. The options aren't really versioned.
    //therefore, we should move all the options to the meta page when building the index (meta pages are properly versioned).
    pub fn from_relation(relation: &PgRelation) -> PgBox<TSVIndexOptions> {
        if relation.rd_index.is_null() {
            panic!("'{}' is not a TSV index", relation.name())
        } else if relation.rd_options.is_null() {
            // use defaults
            let mut ops = unsafe { PgBox::<TSVIndexOptions>::alloc0() };
            ops.storage_layout_offset = 0;
            ops.num_neighbors = NUM_NEIGHBORS_DEFAULT_SENTINEL;
            ops.search_list_size = 100;
            ops.max_alpha = DEFAULT_MAX_ALPHA;
            ops.num_dimensions = NUM_DIMENSIONS_DEFAULT_SENTINEL;
            ops.bq_num_bits_per_dimension = SBQ_NUM_BITS_PER_DIMENSION_DEFAULT_SENTINEL;
            unsafe {
                set_varsize_4b(
                    ops.as_ptr().cast(),
                    std::mem::size_of::<TSVIndexOptions>() as i32,
                );
            }
            ops.into_pg_boxed()
        } else {
            unsafe { PgBox::from_pg(relation.rd_options as *mut TSVIndexOptions) }
        }
    }

    pub fn get_num_neighbors(&self) -> i32 {
        if self.num_neighbors == NUM_NEIGHBORS_DEFAULT_SENTINEL {
            //specify to use the default value here
            //we can't derive the default at this point in the code because the default is based on the number of dimensions in the vector in the io_optimized case.
            NUM_NEIGHBORS_DEFAULT_SENTINEL
        } else {
            if self.num_neighbors < 10 {
                panic!("num_neighbors must be greater than 10, or -1 for default")
            }
            self.num_neighbors
        }
    }

    pub fn get_storage_type(&self) -> StorageType {
        let s = self.get_str(self.storage_layout_offset, || {
            super::storage::DEFAULT_STORAGE_TYPE_STR.to_owned()
        });

        StorageType::from_str(s.as_str())
    }

    fn get_str<F: FnOnce() -> String>(&self, offset: i32, default: F) -> String {
        if offset == 0 {
            default()
        } else {
            let opts = self as *const _ as void_ptr as usize;
            let value =
                unsafe { CStr::from_ptr((opts + offset as usize) as *const std::os::raw::c_char) };

            value.to_str().unwrap().to_owned()
        }
    }
}

static mut RELOPT_KIND_TSV: pg_sys::relopt_kind::Type = 0;

// amoptions is a function that gets a datum of text[] data from pg_class.reloptions (which contains text in the format "key=value") and returns a bytea for the struct for the parsed options.
// this is used to fill the rd_options field in the index relation.
// except for during build the validate parameter should be false.
// any option that is no longer recognized that exists in the reloptions will simply be ignored when validate is false.
// therefore, it is safe to change the options struct and add/remove new options without breaking existing indexes.
// but note that the standard parsing way has no ability to put "migration" logic in here. So all new options will have to have defaults value when reading old indexes.
// we could do additional logic to fix this here, but instead we just move the option values to the meta page when building the index, and do versioning there.
// side note: this logic is not used in \d+ and similar psql commands to get description info. Those commands use the text array in pg_class.reloptions directly.
// so when displaying the info, they'll show the old options and their values as set when the index was created.
#[allow(clippy::unneeded_field_pattern)] // b/c of offset_of!()
#[pg_guard]
pub unsafe extern "C" fn amoptions(
    reloptions: pg_sys::Datum,
    validate: bool,
) -> *mut pg_sys::bytea {
    // TODO:  how to make this const?  we can't use offset_of!() macro in const definitions, apparently
    let tab: [pg_sys::relopt_parse_elt; 6] = [
        pg_sys::relopt_parse_elt {
            optname: "storage_layout".as_pg_cstr(),
            opttype: pg_sys::relopt_type::RELOPT_TYPE_STRING,
            offset: offset_of!(TSVIndexOptions, storage_layout_offset) as i32,
        },
        pg_sys::relopt_parse_elt {
            optname: "num_neighbors".as_pg_cstr(),
            opttype: pg_sys::relopt_type::RELOPT_TYPE_INT,
            offset: offset_of!(TSVIndexOptions, num_neighbors) as i32,
        },
        pg_sys::relopt_parse_elt {
            optname: "search_list_size".as_pg_cstr(),
            opttype: pg_sys::relopt_type::RELOPT_TYPE_INT,
            offset: offset_of!(TSVIndexOptions, search_list_size) as i32,
        },
        pg_sys::relopt_parse_elt {
            optname: "num_dimensions".as_pg_cstr(),
            opttype: pg_sys::relopt_type::RELOPT_TYPE_INT,
            offset: offset_of!(TSVIndexOptions, num_dimensions) as i32,
        },
        pg_sys::relopt_parse_elt {
            optname: "num_bits_per_dimension".as_pg_cstr(),
            opttype: pg_sys::relopt_type::RELOPT_TYPE_INT,
            offset: offset_of!(TSVIndexOptions, bq_num_bits_per_dimension) as i32,
        },
        pg_sys::relopt_parse_elt {
            optname: "max_alpha".as_pg_cstr(),
            opttype: pg_sys::relopt_type::RELOPT_TYPE_REAL,
            offset: offset_of!(TSVIndexOptions, max_alpha) as i32,
        },
    ];

    build_relopts(reloptions, validate, &tab)
}

unsafe fn build_relopts(
    reloptions: pg_sys::Datum,
    validate: bool,
    tab: &[pg_sys::relopt_parse_elt],
) -> *mut pg_sys::bytea {
    /* Parse the user-given reloptions */
    let rdopts = pg_sys::build_reloptions(
        reloptions,
        validate,
        RELOPT_KIND_TSV,
        std::mem::size_of::<TSVIndexOptions>(),
        tab.as_ptr(),
        tab.len() as i32,
    );

    rdopts as *mut pg_sys::bytea
}

#[pg_guard]
extern "C" fn validate_storage_layout(value: *const std::os::raw::c_char) {
    if value.is_null() {
        // use a default value
        return;
    }

    let value = unsafe { CStr::from_ptr(value) }
        .to_str()
        .expect("failed to parse storage_layout value");
    _ = StorageType::from_str(value);
}

/// # Safety
///
/// TODO
pub unsafe fn init() {
    RELOPT_KIND_TSV = pg_sys::add_reloption_kind();

    pg_sys::add_string_reloption(
        RELOPT_KIND_TSV,
        "storage_layout".as_pg_cstr(),
        "Storage layout: either memory_optimized or plain".as_pg_cstr(),
        super::storage::DEFAULT_STORAGE_TYPE_STR.as_pg_cstr(),
        Some(validate_storage_layout),
        pg_sys::AccessExclusiveLock as pg_sys::LOCKMODE,
    );

    pg_sys::add_int_reloption(
        RELOPT_KIND_TSV,
        "num_neighbors".as_pg_cstr(),
        "Maximum number of neighbors in the graph".as_pg_cstr(),
        NUM_NEIGHBORS_DEFAULT_SENTINEL,
        -1,
        1000,
        pg_sys::AccessExclusiveLock as pg_sys::LOCKMODE,
    );

    pg_sys::add_int_reloption(
        RELOPT_KIND_TSV,
        "search_list_size".as_pg_cstr(),
        "The search list size to use during a build".as_pg_cstr(),
        100,
        10,
        1000,
        pg_sys::AccessExclusiveLock as pg_sys::LOCKMODE,
    );

    pg_sys::add_real_reloption(
        RELOPT_KIND_TSV,
        "max_alpha".as_pg_cstr(),
        "The maximum alpha used in pruning".as_pg_cstr(),
        DEFAULT_MAX_ALPHA,
        1.0,
        5.0,
        pg_sys::AccessExclusiveLock as pg_sys::LOCKMODE,
    );

    pg_sys::add_int_reloption(
        RELOPT_KIND_TSV,
        "num_dimensions".as_pg_cstr(),
        "The number of dimensions to index (0 to index all dimensions)".as_pg_cstr(),
        0,
        0,
        5000,
        pg_sys::AccessExclusiveLock as pg_sys::LOCKMODE,
    );

    pg_sys::add_int_reloption(
        RELOPT_KIND_TSV,
        "num_bits_per_dimension".as_pg_cstr(),
        "The number of bits to use per dimension for compressed storage".as_pg_cstr(),
        SBQ_NUM_BITS_PER_DIMENSION_DEFAULT_SENTINEL as _,
        0,
        32,
        pg_sys::AccessExclusiveLock as pg_sys::LOCKMODE,
    );
}

#[cfg(any(test, feature = "pg_test"))]
#[pgrx::pg_schema]
mod tests {
    use crate::access_method::{
        options::{
            TSVIndexOptions, DEFAULT_MAX_ALPHA, NUM_DIMENSIONS_DEFAULT_SENTINEL,
            NUM_NEIGHBORS_DEFAULT_SENTINEL, SBQ_NUM_BITS_PER_DIMENSION_DEFAULT_SENTINEL,
        },
        storage::StorageType,
    };
    use pgrx::*;

    #[pg_test]
    unsafe fn test_index_options() -> spi::Result<()> {
        Spi::run(
            "CREATE TABLE test(encoding vector(3));
        CREATE INDEX idxtest
                  ON test
               USING diskann(encoding)
                WITH (num_neighbors=30);",
        )?;

        let index_oid =
            Spi::get_one::<pg_sys::Oid>("SELECT 'idxtest'::regclass::oid")?.expect("oid was null");
        let indexrel = PgRelation::from_pg(pg_sys::RelationIdGetRelation(index_oid));
        let options = TSVIndexOptions::from_relation(&indexrel);
        assert_eq!(options.num_neighbors, 30);
        assert_eq!(options.num_dimensions, NUM_DIMENSIONS_DEFAULT_SENTINEL);
        assert_eq!(
            options.bq_num_bits_per_dimension,
            SBQ_NUM_BITS_PER_DIMENSION_DEFAULT_SENTINEL,
        );
        Ok(())
    }

    #[pg_test]
    unsafe fn test_index_options_defaults() -> spi::Result<()> {
        Spi::run(
            "CREATE TABLE test(encoding vector(3));
        CREATE INDEX idxtest
                  ON test
               USING diskann(encoding);",
        )?;

        let index_oid =
            Spi::get_one::<pg_sys::Oid>("SELECT 'idxtest'::regclass::oid")?.expect("oid was null");
        let indexrel = PgRelation::from_pg(pg_sys::RelationIdGetRelation(index_oid));
        let options = TSVIndexOptions::from_relation(&indexrel);
        assert_eq!(options.get_num_neighbors(), NUM_NEIGHBORS_DEFAULT_SENTINEL);
        assert_eq!(options.search_list_size, 100);
        assert_eq!(options.max_alpha, DEFAULT_MAX_ALPHA);
        assert_eq!(options.num_dimensions, NUM_DIMENSIONS_DEFAULT_SENTINEL);
        assert_eq!(options.get_storage_type(), StorageType::SbqCompression);
        assert_eq!(
            options.bq_num_bits_per_dimension,
            SBQ_NUM_BITS_PER_DIMENSION_DEFAULT_SENTINEL,
        );
        Ok(())
    }

    #[pg_test]
    unsafe fn test_index_options_plain() -> spi::Result<()> {
        Spi::run(
            "CREATE TABLE test(encoding vector(3));
        CREATE INDEX idxtest
                  ON test
               USING diskann(encoding)
               WITH (storage_layout = plain);",
        )?;

        let index_oid =
            Spi::get_one::<pg_sys::Oid>("SELECT 'idxtest'::regclass::oid")?.expect("oid was null");
        let indexrel = PgRelation::from_pg(pg_sys::RelationIdGetRelation(index_oid));
        let options = TSVIndexOptions::from_relation(&indexrel);
        assert_eq!(options.get_num_neighbors(), NUM_NEIGHBORS_DEFAULT_SENTINEL);
        assert_eq!(options.search_list_size, 100);
        assert_eq!(options.max_alpha, DEFAULT_MAX_ALPHA);
        assert_eq!(options.get_storage_type(), StorageType::Plain);
        Ok(())
    }

    #[pg_test]
    unsafe fn test_index_options_custom() -> spi::Result<()> {
        Spi::run("CREATE TABLE test(encoding vector(3));
        CREATE INDEX idxtest
                  ON test
               USING diskann(encoding)
               WITH (storage_layout = plain, num_neighbors=40, search_list_size=18, num_dimensions=20, max_alpha=1.4);")?;

        let index_oid =
            Spi::get_one::<pg_sys::Oid>("SELECT 'idxtest'::regclass::oid")?.expect("oid was null");
        let indexrel = PgRelation::from_pg(pg_sys::RelationIdGetRelation(index_oid));
        let options = TSVIndexOptions::from_relation(&indexrel);
        assert_eq!(options.get_num_neighbors(), 40);
        assert_eq!(options.search_list_size, 18);
        assert_eq!(options.max_alpha, 1.4);
        assert_eq!(options.get_storage_type(), StorageType::Plain);
        assert_eq!(options.num_dimensions, 20);
        assert_eq!(
            options.bq_num_bits_per_dimension,
            SBQ_NUM_BITS_PER_DIMENSION_DEFAULT_SENTINEL
        );
        Ok(())
    }

    #[pg_test]
    unsafe fn test_index_options_custom_mem_optimized() -> spi::Result<()> {
        Spi::run("CREATE TABLE test(encoding vector(3));
        CREATE INDEX idxtest
                  ON test
               USING diskann(encoding)
               WITH (storage_layout = memory_optimized, num_neighbors=40, search_list_size=18, num_dimensions=20, max_alpha=1.4, num_bits_per_dimension=5);")?;

        let index_oid =
            Spi::get_one::<pg_sys::Oid>("SELECT 'idxtest'::regclass::oid")?.expect("oid was null");
        let indexrel = PgRelation::from_pg(pg_sys::RelationIdGetRelation(index_oid));
        let options = TSVIndexOptions::from_relation(&indexrel);
        assert_eq!(options.get_num_neighbors(), 40);
        assert_eq!(options.search_list_size, 18);
        assert_eq!(options.max_alpha, 1.4);
        assert_eq!(options.get_storage_type(), StorageType::SbqCompression);
        assert_eq!(options.num_dimensions, 20);
        assert_eq!(options.bq_num_bits_per_dimension, 5);
        Ok(())
    }
}
