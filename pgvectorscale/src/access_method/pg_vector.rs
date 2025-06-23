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

#[derive(Debug)]
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
    /// Creates a zero-filled PgVector with the specified dimensions
    pub fn zeros(meta_page: &meta_page::MetaPage) -> Self {
        let num_dimensions = meta_page.get_num_dimensions();
        let num_dimensions_to_index = meta_page.get_num_dimensions_to_index();

        unsafe {
            if num_dimensions == num_dimensions_to_index {
                // Optimization: same pointer for both index and full distance
                let inner = Self::create_zeros_inner(num_dimensions as i16);
                PgVector {
                    index_distance: Some(inner),
                    index_distance_needs_pfree: true,
                    full_distance: Some(inner),
                    full_distance_needs_pfree: false,
                }
            } else {
                // Different dimensions for index vs full
                let index_inner = Self::create_zeros_inner(num_dimensions_to_index as i16);
                let full_inner = Self::create_zeros_inner(num_dimensions as i16);
                PgVector {
                    index_distance: Some(index_inner),
                    index_distance_needs_pfree: true,
                    full_distance: Some(full_inner),
                    full_distance_needs_pfree: true,
                }
            }
        }
    }

    unsafe fn create_zeros_inner(dimensions: i16) -> *mut PgVectorInternal {
        // Calculate total size needed: header + array of f32s
        let header_size = std::mem::size_of::<PgVectorInternal>();
        let array_size = dimensions as usize * std::mem::size_of::<f32>();
        let total_size = header_size + array_size;

        // Allocate PostgreSQL memory
        let ptr = pg_sys::palloc0(total_size) as *mut PgVectorInternal;

        // Initialize the header
        (*ptr).vl_len_ = total_size as i32;
        (*ptr).dim = dimensions;
        (*ptr).unused = 0;

        // The array is already zero-filled due to palloc0
        ptr
    }

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
        assert!(!datum.is_null(), "Datum should not be NULL");

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

impl Clone for PgVector {
    fn clone(&self) -> Self {
        unsafe {
            let index_distance = self
                .index_distance
                .map(|original| Self::clone_inner(original));

            let full_distance = if let Some(original) = self.full_distance {
                // Check if full_distance points to the same memory as index_distance
                if self.index_distance.is_some()
                    && std::ptr::eq(original, self.index_distance.unwrap())
                {
                    // Reuse the same cloned pointer
                    index_distance
                } else {
                    // Clone separately
                    Some(Self::clone_inner(original))
                }
            } else {
                None
            };

            PgVector {
                index_distance,
                index_distance_needs_pfree: index_distance.is_some(),
                full_distance,
                full_distance_needs_pfree: full_distance.is_some()
                    && !std::ptr::eq(
                        full_distance.unwrap_or(std::ptr::null_mut()),
                        index_distance.unwrap_or(std::ptr::null_mut()),
                    ),
            }
        }
    }
}

impl PgVector {
    unsafe fn clone_inner(original: *mut PgVectorInternal) -> *mut PgVectorInternal {
        let dim = (*original).dim;
        let slice = (*original).to_slice();

        // Calculate total size needed: header + array of f32s
        let header_size = std::mem::size_of::<PgVectorInternal>();
        let array_size = dim as usize * std::mem::size_of::<f32>();
        let total_size = header_size + array_size;

        // Allocate new PostgreSQL memory
        let new_ptr = pg_sys::palloc(total_size) as *mut PgVectorInternal;

        // Copy the header
        (*new_ptr).vl_len_ = (*original).vl_len_;
        (*new_ptr).dim = dim;
        (*new_ptr).unused = (*original).unused;

        // Copy the vector data
        let new_slice = (*new_ptr).x.as_mut_slice(dim as _);
        new_slice.copy_from_slice(slice);

        new_ptr
    }
}

#[cfg(any(test, feature = "pg_test"))]
#[pgrx::pg_schema]
mod tests {
    use super::*;
    use crate::access_method::meta_page::MetaPage;
    use pgrx::{pg_sys, spi::Result as spi_Result, PgRelation, Spi};

    #[pg_test]
    unsafe fn test_pgvector_clone() -> spi_Result<()> {
        // Create a test table to get a valid MetaPage context
        Spi::run(
            "CREATE TABLE test_clone(embedding vector(3));
            CREATE INDEX test_clone_idx ON test_clone USING diskann(embedding vector_l2_ops)
                WITH (num_neighbors=10, search_list_size=10);",
        )?;

        let index_oid = Spi::get_one::<pg_sys::Oid>("SELECT 'test_clone_idx'::regclass::oid")?
            .expect("oid was null");
        let indexrel = PgRelation::from_pg(pg_sys::RelationIdGetRelation(index_oid));
        let meta_page = MetaPage::fetch(&indexrel);

        // Create an original PgVector with test data
        let original = PgVector::zeros(&meta_page);

        // Modify the original vector to have specific test values
        {
            let slice = original.to_index_slice();
            let slice_mut = std::slice::from_raw_parts_mut(slice.as_ptr() as *mut f32, slice.len());
            slice_mut[0] = 1.0;
            slice_mut[1] = 2.0;
            slice_mut[2] = 3.0;
        }

        // Clone the vector
        let cloned = original.clone();

        // Verify the data was copied correctly
        let original_slice = original.to_index_slice();
        let cloned_slice = cloned.to_index_slice();

        assert_eq!(original_slice.len(), cloned_slice.len());
        assert_eq!(original_slice[0], cloned_slice[0]);
        assert_eq!(original_slice[1], cloned_slice[1]);
        assert_eq!(original_slice[2], cloned_slice[2]);

        // Verify they point to different memory locations
        assert_ne!(
            original_slice.as_ptr(),
            cloned_slice.as_ptr(),
            "Clone should create new memory allocation"
        );

        // Modify the original to ensure independence
        {
            let slice = original.to_index_slice();
            let slice_mut = std::slice::from_raw_parts_mut(slice.as_ptr() as *mut f32, slice.len());
            slice_mut[0] = 99.0; // Change original
        }

        // Verify clone is unaffected
        let cloned_slice_after = cloned.to_index_slice();
        assert_eq!(
            cloned_slice_after[0], 1.0,
            "Clone should be independent of original"
        );

        // Clean up
        Spi::run("DROP TABLE test_clone CASCADE;")?;
        Ok(())
    }

    #[pg_test]
    unsafe fn test_pgvector_clone_shared_pointers() -> spi_Result<()> {
        // Test case where index_distance and full_distance point to same memory
        Spi::run(
            "CREATE TABLE test_shared(embedding vector(3));
            CREATE INDEX test_shared_idx ON test_shared USING diskann(embedding vector_l2_ops)
                WITH (num_neighbors=10, search_list_size=10);",
        )?;

        let index_oid = Spi::get_one::<pg_sys::Oid>("SELECT 'test_shared_idx'::regclass::oid")?
            .expect("oid was null");
        let indexrel = PgRelation::from_pg(pg_sys::RelationIdGetRelation(index_oid));
        let meta_page = MetaPage::fetch(&indexrel);

        // Create a vector where both pointers are the same (optimization case)
        let original = PgVector::zeros(&meta_page);

        // Verify the optimization case is active
        if std::ptr::eq(
            original.index_distance.unwrap(),
            original.full_distance.unwrap(),
        ) {
            let cloned = original.clone();

            // In the clone, they should also point to the same memory
            assert!(
                std::ptr::eq(
                    cloned.index_distance.unwrap(),
                    cloned.full_distance.unwrap()
                ),
                "Cloned vector should preserve pointer optimization"
            );

            // But clone should have different memory than original
            assert_ne!(
                original.index_distance.unwrap(),
                cloned.index_distance.unwrap(),
                "Clone should have different memory than original"
            );
        }

        // Clean up
        Spi::run("DROP TABLE test_shared CASCADE;")?;
        Ok(())
    }
}
