use once_cell::sync::OnceCell;
use pgrx::pg_sys::{self, GETSTRUCT};
use pgrx::varlena::varsize_any;
use pgrx::*;

use super::build::MAX_DIMENSION;

static PGVECTOR_VECTOR_OID: OnceCell<pg_sys::Oid> = OnceCell::new();

/// OID of the `vector` type owned by the installed `vector` (pgvector) extension.
///
/// Resolved via `pg_extension` / `pg_type`, never through `search_path`.
pub unsafe fn pgvector_vector_oid() -> pg_sys::Oid {
    *PGVECTOR_VECTOR_OID.get_or_init(|| lookup_pgvector_vector_oid())
}

unsafe fn lookup_pgvector_vector_oid() -> pg_sys::Oid {
    let extname = c"vector";
    let ext_oid = pg_sys::get_extension_oid(extname.as_ptr(), false);
    if ext_oid == pg_sys::InvalidOid {
        error!("diskann: required extension \"vector\" is not installed");
    }

    let ext_tup =
        pg_sys::SearchSysCache1(pg_sys::SysCacheIdentifier::EXTENSIONOID as i32, ext_oid.into());
    if ext_tup.is_null() {
        error!("diskann: could not look up pg_extension entry for \"vector\"");
    }
    let ext_form = GETSTRUCT(ext_tup) as pg_sys::Form_pg_extension;
    let nsp_oid = (*ext_form).extnamespace;
    pg_sys::ReleaseSysCache(ext_tup);

    let mut typname = pg_sys::NameData::default();
    pg_sys::namestrcpy(
        &mut typname as *mut pg_sys::NameData,
        c"vector".as_ptr(),
    );

    let typ_tup = pg_sys::SearchSysCache2(
        pg_sys::SysCacheIdentifier::TYPENAMENSP as i32,
        pg_sys::Datum::from(std::ptr::addr_of!(typname) as usize),
        nsp_oid.into(),
    );
    if typ_tup.is_null() {
        error!("diskann: could not find type \"vector\" in the vector extension schema");
    }
    let typ_form = GETSTRUCT(typ_tup) as pg_sys::Form_pg_type;
    let oid = (*typ_form).oid;
    pg_sys::ReleaseSysCache(typ_tup);
    oid
}

/// Reject index attributes whose base type is not pgvector's `vector`.
///
/// Must run before any detoast of index/query datums.
pub unsafe fn ensure_index_vector_type(index: &PgRelation) {
    let att = index
        .tuple_desc()
        .get(0)
        .unwrap_or_else(|| error!("diskann: index is missing its vector attribute"));
    let base = pg_sys::getBaseType(att.atttypid);
    let expected = pgvector_vector_oid();
    if base != expected {
        error!(
            "diskann: index column type is not pgvector's vector type (found oid {}, expected {})",
            att.atttypid, expected
        );
    }
}

/// Validate a signed typmod as a vector dimension and return it as u32.
pub fn dimension_from_typmod(atttypmod: i32) -> u32 {
    if atttypmod < 1 || atttypmod as u32 > MAX_DIMENSION {
        error!(
            "diskann: indexed column has no valid vector dimension (atttypmod = {}); the column must be pgvector's vector(N)",
            atttypmod
        );
    }
    atttypmod as u32
}

/// Validate persisted full and indexed dimensions.
pub fn ensure_valid_dimensions(num_dimensions: u32, num_dimensions_to_index: u32) {
    if num_dimensions < 1 || num_dimensions > MAX_DIMENSION {
        error!(
            "diskann: invalid full dimension {} (max is {})",
            num_dimensions, MAX_DIMENSION
        );
    }
    if num_dimensions_to_index < 1 || num_dimensions_to_index > num_dimensions {
        error!(
            "diskann: num_dimensions={} must be between 1 and {}",
            num_dimensions_to_index, num_dimensions
        );
    }
}

/// After detoast, require a well-formed pgvector layout before building any slice.
///
/// Returns the signed dimension as usize.
pub unsafe fn checked_vector_dim(ptr: *const super::pg_vector::PgVectorInternal) -> usize {
    let header_bytes = std::mem::size_of::<i32>() + 2 * std::mem::size_of::<i16>();
    let total = varsize_any(ptr.cast());
    if total < header_bytes {
        error!("diskann: vector datum shorter than its header");
    }

    let dim = (*ptr).dim as i32;
    if dim < 1 || dim as u32 > MAX_DIMENSION {
        error!("diskann: vector dimension {dim} out of range");
    }

    let unused = (*ptr).unused.assume_init();
    if unused != 0 {
        error!("diskann: vector datum has non-zero reserved field");
    }

    let expect = header_bytes
        .checked_add(
            (dim as usize)
                .checked_mul(std::mem::size_of::<f32>())
                .unwrap(),
        )
        .unwrap();
    if total != expect {
        error!("diskann: vector datum is {total} bytes, dimension {dim} needs {expect}");
    }

    dim as usize
}
