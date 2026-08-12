use pgrx::pg_sys::{self, GETSTRUCT};
use pgrx::varlena::varsize_any;
use pgrx::*;

use super::build::MAX_DIMENSION;

/// Return the pgvector-owned `vector` base type for `type_oid`.
///
/// This verifies extension ownership and the type name without relying on
/// `search_path` or a backend-lifetime cache.
pub unsafe fn pgvector_vector_base_oid(type_oid: pg_sys::Oid) -> Option<pg_sys::Oid> {
    let base_oid = pg_sys::getBaseType(type_oid);
    let vector_extension_oid = pg_sys::get_extension_oid(c"vector".as_ptr(), true);
    if vector_extension_oid == pg_sys::InvalidOid
        || pg_sys::getExtensionOfObject(pg_sys::TypeRelationId, base_oid) != vector_extension_oid
    {
        return None;
    }

    let type_tuple =
        pg_sys::SearchSysCache1(pg_sys::SysCacheIdentifier::TYPEOID as i32, base_oid.into());
    if type_tuple.is_null() {
        return None;
    }

    let type_form = GETSTRUCT(type_tuple) as pg_sys::Form_pg_type;
    let is_vector =
        core::ffi::CStr::from_ptr((*type_form).typname.data.as_ptr()).to_bytes() == b"vector";
    pg_sys::ReleaseSysCache(type_tuple);

    is_vector.then_some(base_oid)
}

/// Reject index attributes whose base type is not pgvector's `vector`.
///
/// Must run before any detoast of index/query datums.
pub unsafe fn ensure_index_vector_type(index: &PgRelation) {
    let tuple_desc = index.tuple_desc();
    let attribute = tuple_desc
        .get(0)
        .unwrap_or_else(|| error!("diskann: index is missing its vector attribute"));
    let attribute_type_oid = attribute.atttypid;
    if pgvector_vector_base_oid(attribute_type_oid).is_none() {
        error!(
            "diskann: index column type is not pgvector's vector type (found oid {})",
            attribute_type_oid
        );
    }
}

/// Validate a signed typmod as a vector dimension and return it as u32.
pub fn dimension_from_typmod(atttypmod: i32) -> Result<u32, String> {
    if atttypmod < 1 || atttypmod as u32 > MAX_DIMENSION {
        return Err(format!(
            "diskann: indexed column has no valid vector dimension (atttypmod = {}); the column must be pgvector's vector(N)",
            atttypmod
        ));
    }
    Ok(atttypmod as u32)
}

/// Validate persisted full and indexed dimensions.
pub fn ensure_valid_dimensions(
    num_dimensions: u32,
    num_dimensions_to_index: u32,
) -> Result<(), String> {
    if num_dimensions < 1 || num_dimensions > MAX_DIMENSION {
        return Err(format!(
            "diskann: invalid full dimension {} (max is {})",
            num_dimensions, MAX_DIMENSION
        ));
    }
    if num_dimensions_to_index < 1 || num_dimensions_to_index > num_dimensions {
        return Err(format!(
            "diskann: num_dimensions={} must be between 1 and {}",
            num_dimensions_to_index, num_dimensions
        ));
    }
    Ok(())
}

pub fn ensure_datum_dimension(datum_dimension: usize, index_dimension: u32) -> Result<(), String> {
    if datum_dimension != index_dimension as usize {
        return Err(format!(
            "diskann: vector datum dimension {} does not match index dimension {}",
            datum_dimension, index_dimension
        ));
    }
    Ok(())
}

fn validate_vector_layout(total: usize, dim: i32, reserved: i16) -> Result<usize, String> {
    let header_bytes = std::mem::size_of::<i32>() + 2 * std::mem::size_of::<i16>();
    if total < header_bytes {
        return Err("diskann: vector datum shorter than its header".to_string());
    }
    if dim < 1 || dim as u32 > MAX_DIMENSION {
        return Err(format!("diskann: vector dimension {dim} out of range"));
    }
    if reserved != 0 {
        return Err("diskann: vector datum has non-zero reserved field".to_string());
    }

    let expected = header_bytes
        .checked_add(
            (dim as usize)
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| "diskann: vector datum size overflow".to_string())?,
        )
        .ok_or_else(|| "diskann: vector datum size overflow".to_string())?;
    if total != expected {
        return Err(format!(
            "diskann: vector datum is {total} bytes, dimension {dim} needs {expected}"
        ));
    }

    Ok(dim as usize)
}

/// After detoast, require a well-formed pgvector layout before building any slice.
///
/// Returns the signed dimension as usize.
pub unsafe fn checked_vector_dim(
    ptr: *const super::pg_vector::PgVectorInternal,
) -> Result<usize, String> {
    let total = varsize_any(ptr.cast());
    let dim = (*ptr).dim as i32;
    validate_vector_layout(total, dim, (*ptr).reserved())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_typmods() {
        assert!(dimension_from_typmod(-1).is_err());
        assert!(dimension_from_typmod(0).is_err());
        assert!(dimension_from_typmod((MAX_DIMENSION + 1) as i32).is_err());
        assert_eq!(dimension_from_typmod(3).unwrap(), 3);
    }

    #[test]
    fn rejects_invalid_metapage_dimensions() {
        assert!(ensure_valid_dimensions(0, 0).is_err());
        assert!(ensure_valid_dimensions(MAX_DIMENSION + 1, 1).is_err());
        assert!(ensure_valid_dimensions(3, 0).is_err());
        assert!(ensure_valid_dimensions(3, 4).is_err());
        assert!(ensure_valid_dimensions(3, 3).is_ok());
    }

    #[test]
    fn rejects_malformed_vector_layouts() {
        assert!(validate_vector_layout(4, 1, 0).is_err());
        assert!(validate_vector_layout(16, 10, 0).is_err());
        assert!(validate_vector_layout(16, 2, 1).is_err());
        assert!(validate_vector_layout(16, 2, 0).is_ok());
    }

    #[test]
    fn rejects_datum_dimension_mismatch() {
        assert!(ensure_datum_dimension(2, 3).is_err());
        assert!(ensure_datum_dimension(3, 3).is_ok());
    }
}
