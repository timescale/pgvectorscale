use pgrx::*;

use super::distance::preprocess_cosine;

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
        let dim = (*self).dim;
        let raw_slice = unsafe { (*self).x.as_slice(dim as _) };
        raw_slice
    }
}

pub struct PgVector {
    inner: *mut PgVectorInternal,
    need_pfree: bool,
}

impl Drop for PgVector {
    fn drop(&mut self) {
        if self.need_pfree {
            unsafe {
                pg_sys::pfree(self.inner.cast());
            }
        }
    }
}

impl PgVector {
    pub unsafe fn from_pg_parts(
        datum_parts: *mut pg_sys::Datum,
        isnull_parts: *mut bool,
        index: usize,
    ) -> Option<PgVector> {
        let isnulls = std::slice::from_raw_parts(isnull_parts, index + 1);
        if isnulls[index] {
            return None;
        }
        let datums = std::slice::from_raw_parts(datum_parts, index + 1);
        Some(Self::from_datum(datums[index]))
    }

    pub unsafe fn from_datum(datum: pg_sys::Datum) -> PgVector {
        //FIXME: we are using a copy here to avoid lifetime issues and because in some cases we have to
        //modify the datum in preprocess_cosine. We should find a way to avoid the copy if the vector is
        //normalized and preprocess_cosine is a noop;
        let detoasted = pg_sys::pg_detoast_datum_copy(datum.cast_mut_ptr());
        let is_copy = !std::ptr::eq(
            detoasted.cast::<PgVectorInternal>(),
            datum.cast_mut_ptr::<PgVectorInternal>(),
        );
        let casted = detoasted.cast::<PgVectorInternal>();

        let dim = (*casted).dim;
        let raw_slice = unsafe { (*casted).x.as_mut_slice(dim as _) };
        preprocess_cosine(raw_slice);

        PgVector {
            inner: casted,
            need_pfree: is_copy,
        }
    }

    pub fn to_slice(&self) -> &[f32] {
        unsafe { (*self.inner).to_slice() }
    }
}
