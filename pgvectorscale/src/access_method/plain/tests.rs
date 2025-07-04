#[cfg(any(test, feature = "pg_test"))]
#[pgrx::pg_schema]
mod tests {

    use pgrx::*;

    use crate::access_method::distance::DistanceType;

    #[pg_test]
    unsafe fn test_plain_storage_index_creation_many_neighbors() -> spi::Result<()> {
        crate::access_method::build::tests::test_index_creation_and_accuracy_scaffold(
            DistanceType::Cosine,
            "num_neighbors=38, storage_layout = plain",
            "plain_many_neighbors",
            1536,
        )?;
        Ok(())
    }

    #[pg_test]
    unsafe fn test_plain_storage_index_creation_low_memory() -> spi::Result<()> {
        crate::access_method::build::tests::test_index_creation_and_accuracy_scaffold_bounded_memory(
            DistanceType::Cosine,
            "num_neighbors=10, storage_layout = plain",
            "plain_many_neighbors",
            1536,
            Some(1024),
        )?;
        Ok(())
    }

    #[pg_test]
    unsafe fn test_plain_storage_index_creation_few_neighbors() -> spi::Result<()> {
        //a test with few neighbors tests the case that nodes share a page, which has caused deadlocks in the past.
        crate::access_method::build::tests::test_index_creation_and_accuracy_scaffold(
            DistanceType::Cosine,
            "num_neighbors=20, storage_layout = plain",
            "plain_few_neighbors",
            1536,
        )?;
        Ok(())
    }

    #[test]
    fn test_plain_storage_delete_vacuum_plain() {
        crate::access_method::vacuum::tests::test_delete_vacuum_plain_scaffold(
            "num_neighbors = 38, storage_layout = plain",
        );
    }

    #[test]
    fn test_plain_storage_delete_vacuum_full() {
        crate::access_method::vacuum::tests::test_delete_vacuum_full_scaffold(
            "num_neighbors = 38, storage_layout = plain",
        );
    }

    #[test]
    fn test_plain_storage_update_with_null() {
        crate::access_method::vacuum::tests::test_update_with_null_scaffold(
            "num_neighbors = 38, storage_layout = plain",
        );
    }

    #[pg_test]
    unsafe fn test_plain_storage_empty_table_insert() -> spi::Result<()> {
        crate::access_method::build::tests::test_empty_table_insert_scaffold(
            "num_neighbors=38, storage_layout = plain",
        )
    }

    #[pg_test]
    unsafe fn test_plain_storage_insert_empty_insert() -> spi::Result<()> {
        crate::access_method::build::tests::test_insert_empty_insert_scaffold(
            "num_neighbors=38, storage_layout = plain",
        )
    }

    #[pg_test]
    unsafe fn test_plain_storage_num_dimensions_cosine() -> spi::Result<()> {
        crate::access_method::build::tests::test_index_creation_and_accuracy_scaffold(
            DistanceType::Cosine,
            "num_neighbors=38, storage_layout = plain, num_dimensions=768",
            "plain_num_dimensions",
            3072,
        )?;
        Ok(())
    }

    #[pg_test]
    unsafe fn test_plain_storage_num_dimensions_l2() -> spi::Result<()> {
        crate::access_method::build::tests::test_index_creation_and_accuracy_scaffold(
            DistanceType::L2,
            "num_neighbors=38, storage_layout = plain, num_dimensions=768",
            "plain_num_dimensions",
            3072,
        )?;
        Ok(())
    }

    #[pg_test]
    #[should_panic]
    unsafe fn test_plain_storage_num_dimensions_ip() -> spi::Result<()> {
        // Should panic because combination of inner product and plain storage
        // is not supported.
        crate::access_method::build::tests::test_index_creation_and_accuracy_scaffold(
            DistanceType::InnerProduct,
            "num_neighbors=38, storage_layout = plain, num_dimensions=768",
            "plain_num_dimensions",
            3072,
        )?;
        Ok(())
    }

    #[pg_test]
    unsafe fn test_plain_storage_index_updates_cosine() -> spi::Result<()> {
        crate::access_method::build::tests::test_index_updates(
            DistanceType::Cosine,
            "storage_layout = plain, num_neighbors=30",
            50,
            "plain",
        )?;
        Ok(())
    }

    #[pg_test]
    unsafe fn test_plain_storage_index_updates_l2() -> spi::Result<()> {
        crate::access_method::build::tests::test_index_updates(
            DistanceType::L2,
            "storage_layout = plain, num_neighbors=30",
            50,
            "plain",
        )?;
        Ok(())
    }

    #[pg_test]
    #[should_panic]
    unsafe fn test_plain_storage_index_updates_ip() -> spi::Result<()> {
        // Should panic because combination of inner product and plain storage
        // is not supported.
        crate::access_method::build::tests::test_index_updates(
            DistanceType::InnerProduct,
            "storage_layout = plain, num_neighbors=30",
            50,
            "plain",
        )?;
        Ok(())
    }
}
