use pgrx::*;

use crate::access_method::distance::DistanceType;

use super::{distance::preprocess_cosine, meta_page};

//Ported from pg_vector code
#[repr(C)]
#[derive(Debug)]
pub struct PgVectorInternal {
    vl_len_: i32, /* varlena header (do not touch directly!) */
    pub dim: i16, /* number of dimensions */
    unused: i16,
    pub x: pg_sys::__IncompleteArrayField<std::os::raw::c_float>,
}

impl PgVectorInternal {
    pub fn to_slice(&self) -> &[f32] {
        let dim = self.dim;
        let raw_slice = unsafe { self.x.as_slice(dim as _) };
        raw_slice
    }
}

pub struct PgVector {
    index_distance: Option<*mut PgVectorInternal>,
    index_distance_needs_pfree: bool,
    full_distance: Option<*mut PgVectorInternal>,
    full_distance_needs_pfree: bool,
}

impl Drop for PgVector {
    fn drop(&mut self) {
        if self.index_distance_needs_pfree {
            unsafe {
                if self.index_distance.is_some() {
                    pg_sys::pfree(self.index_distance.unwrap().cast());
                }
            }
        }
        if self.full_distance_needs_pfree {
            unsafe {
                if self.full_distance.is_some() {
                    pg_sys::pfree(self.full_distance.unwrap().cast());
                }
            }
        }
    }
}

impl PgVector {
    /// # Safety
    ///
    /// TODO
    pub unsafe fn from_pg_parts(
        datum_parts: *mut pg_sys::Datum,
        isnull_parts: *mut bool,
        index: usize,
        meta_page: &meta_page::MetaPage,
        index_distance: bool,
        full_distance: bool,
    ) -> Option<PgVector> {
        let isnulls = std::slice::from_raw_parts(isnull_parts, index + 1);
        if isnulls[index] {
            return None;
        }
        let datums = std::slice::from_raw_parts(datum_parts, index + 1);
        Some(Self::from_datum(
            datums[index],
            meta_page,
            index_distance,
            full_distance,
        ))
    }

    unsafe fn create_inner(
        datum: pg_sys::Datum,
        meta_page: &meta_page::MetaPage,
        is_index_distance: bool,
    ) -> *mut PgVectorInternal {
        //TODO: we are using a copy here to avoid lifetime issues and because in some cases we have to
        //modify the datum in preprocess_cosine. We should find a way to avoid the copy if the vector is
        //normalized and preprocess_cosine is a noop;
        let detoasted = pg_sys::pg_detoast_datum_copy(datum.cast_mut_ptr());
        let is_copy = !std::ptr::eq(
            detoasted.cast::<PgVectorInternal>(),
            datum.cast_mut_ptr::<PgVectorInternal>(),
        );

        /* if is_copy every changes, need to change needs_pfree */
        assert!(is_copy, "Datum should be a copy");
        let casted = detoasted.cast::<PgVectorInternal>();

        if is_index_distance
            && meta_page.get_num_dimensions() != meta_page.get_num_dimensions_to_index()
        {
            assert!((*casted).dim > meta_page.get_num_dimensions_to_index() as _);
            (*casted).dim = meta_page.get_num_dimensions_to_index() as _;
        }

        let dim = (*casted).dim;
        let raw_slice = unsafe { (*casted).x.as_mut_slice(dim as _) };

        if meta_page.get_distance_type() == DistanceType::Cosine {
            preprocess_cosine(raw_slice);
        }
        casted
    }

    /// # Safety
    ///
    /// TODO
    pub unsafe fn from_datum(
        datum: pg_sys::Datum,
        meta_page: &meta_page::MetaPage,
        index_distance: bool,
        full_distance: bool,
    ) -> PgVector {
        if meta_page.get_num_dimensions() == meta_page.get_num_dimensions_to_index() {
            /* optimization if the num dimensions are the same */
            let inner = Self::create_inner(datum, meta_page, true);
            return PgVector {
                index_distance: Some(inner),
                index_distance_needs_pfree: true,
                full_distance: Some(inner),
                full_distance_needs_pfree: false,
            };
        }

        let idx = if index_distance {
            Some(Self::create_inner(datum, meta_page, true))
        } else {
            None
        };

        let full = if full_distance {
            Some(Self::create_inner(datum, meta_page, false))
        } else {
            None
        };

        PgVector {
            index_distance: idx,
            index_distance_needs_pfree: true,
            full_distance: full,
            full_distance_needs_pfree: true,
        }
    }

    pub fn to_index_slice(&self) -> &[f32] {
        unsafe { (*self.index_distance.unwrap()).to_slice() }
    }

    pub fn to_full_slice(&self) -> &[f32] {
        unsafe { (*self.full_distance.unwrap()).to_slice() }
    }
}
