#[cfg(any(test, feature = "pg_test"))]
#[pgrx::pg_schema]
mod tests {
    use pgrx::*;

    use crate::access_method::distance::DistanceType;

    #[pg_test]
    unsafe fn test_bq_compressed_index_creation_default_neighbors() -> spi::Result<()> {
        crate::access_method::build::tests::test_index_creation_and_accuracy_scaffold(
            DistanceType::Cosine,
            "storage_layout = memory_optimized",
            "bq_compressed_default_neighbors",
            1536,
        )?;
        Ok(())
    }

    #[pg_test]
    unsafe fn test_bq_compressed_storage_index_creation_few_neighbors() -> spi::Result<()> {
        //a test with few neighbors tests the case that nodes share a page, which has caused deadlocks in the past.
        crate::access_method::build::tests::test_index_creation_and_accuracy_scaffold(
            DistanceType::Cosine,
            "num_neighbors=10, storage_layout = memory_optimized",
            "bq_compressed_few_neighbors",
            1536,
        )?;
        Ok(())
    }

    #[test]
    fn test_bq_compressed_storage_delete_vacuum_plain() {
        crate::access_method::vacuum::tests::test_delete_vacuum_plain_scaffold(
            "num_neighbors = 10, storage_layout = memory_optimized",
        );
    }

    #[test]
    fn test_bq_compressed_storage_delete_vacuum_full() {
        crate::access_method::vacuum::tests::test_delete_vacuum_full_scaffold(
            "num_neighbors = 38, storage_layout = memory_optimized",
        );
    }

    #[test]
    fn test_bq_compressed_storage_update_with_null() {
        crate::access_method::vacuum::tests::test_update_with_null_scaffold(
            "num_neighbors = 38, storage_layout = memory_optimized",
        );
    }
    #[pg_test]
    unsafe fn test_bq_compressed_storage_empty_table_insert() -> spi::Result<()> {
        crate::access_method::build::tests::test_empty_table_insert_scaffold(
            "num_neighbors=38, storage_layout = memory_optimized",
        )
    }

    #[pg_test]
    unsafe fn test_bq_compressed_storage_insert_empty_insert() -> spi::Result<()> {
        crate::access_method::build::tests::test_insert_empty_insert_scaffold(
            "num_neighbors=38, storage_layout = memory_optimized",
        )
    }

    #[pg_test]
    unsafe fn test_bq_compressed_storage_index_creation_num_dimensions() -> spi::Result<()> {
        crate::access_method::build::tests::test_index_creation_and_accuracy_scaffold(
            DistanceType::Cosine,
            "storage_layout = memory_optimized, num_dimensions=768",
            "bq_compressed_num_dimensions",
            3072,
        )?;
        Ok(())
    }

    #[pg_test]
    unsafe fn test_bq_compressed_storage_index_updates_cosine() -> spi::Result<()> {
        crate::access_method::build::tests::test_index_updates(
            DistanceType::Cosine,
            "storage_layout = memory_optimized, num_neighbors=10",
            300,
            "bq_compressed",
        )?;
        Ok(())
    }

    #[pg_test]
    unsafe fn test_bq_compressed_storage_index_updates_l2() -> spi::Result<()> {
        crate::access_method::build::tests::test_index_updates(
            DistanceType::L2,
            "storage_layout = memory_optimized, num_neighbors=10",
            300,
            "bq_compressed",
        )?;
        Ok(())
    }

    #[pg_test]
    unsafe fn test_bq_compressed_storage_index_updates_ip() -> spi::Result<()> {
        crate::access_method::build::tests::test_index_updates(
            DistanceType::InnerProduct,
            "storage_layout = memory_optimized, num_neighbors=10",
            300,
            "bq_compressed",
        )?;
        Ok(())
    }
}
