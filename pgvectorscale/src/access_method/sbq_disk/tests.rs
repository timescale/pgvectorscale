#[cfg(any(test, feature = "pg_test"))]
#[pgrx::pg_schema]
mod tests {
    use pgrx::*;

    use crate::access_method::distance::DistanceType;

    #[pg_test]
    unsafe fn test_bq_compressed_disk_index_creation_default_neighbors() -> spi::Result<()> {
        crate::access_method::build::tests::test_index_creation_and_accuracy_scaffold(
            DistanceType::Cosine,
            "storage_layout = disk_optimized",
            "bq_compressed_default_neighbors",
            1536,
        )?;
        Ok(())
    }

    #[pg_test]
    unsafe fn test_bq_compressed_storage_index_creation_few_neighbors_disk() -> spi::Result<()> {
        //a test with few neighbors tests the case that nodes share a page, which has caused deadlocks in the past.
        crate::access_method::build::tests::test_index_creation_and_accuracy_scaffold(
            DistanceType::Cosine,
            "num_neighbors=10, storage_layout = disk_optimized",
            "bq_compressed_few_neighbors",
            1536,
        )?;
        Ok(())
    }

    #[pg_test]
    unsafe fn test_bq_compressed_storage_index_creation_low_memory_disk() -> spi::Result<()> {
        crate::access_method::build::tests::test_sized_index_scaffold(
            "num_neighbors=40, storage_layout = disk_optimized",
            1536,
            2000,
            Some(1024),
        )?;
        Ok(())
    }

    #[test]
    fn test_bq_compressed_storage_delete_vacuum_plain_disk() {
        crate::access_method::vacuum::tests::test_delete_vacuum_plain_scaffold(
            "num_neighbors = 10, storage_layout = disk_optimized",
        );
    }

    #[test]
    fn test_bq_compressed_storage_delete_vacuum_full_disk() {
        crate::access_method::vacuum::tests::test_delete_vacuum_full_scaffold(
            "num_neighbors = 38, storage_layout = disk_optimized",
        );
    }

    #[test]
    fn test_bq_compressed_storage_update_with_null_disk() {
        crate::access_method::vacuum::tests::test_update_with_null_scaffold(
            "num_neighbors = 38, storage_layout = disk_optimized",
        );
    }
    #[pg_test]
    unsafe fn test_bq_compressed_storage_empty_table_insert_disk() -> spi::Result<()> {
        crate::access_method::build::tests::test_empty_table_insert_scaffold(
            "num_neighbors=38, storage_layout = disk_optimized",
        )
    }

    #[pg_test]
    unsafe fn test_bq_compressed_storage_insert_empty_insert_disk() -> spi::Result<()> {
        crate::access_method::build::tests::test_insert_empty_insert_scaffold(
            "num_neighbors=38, storage_layout = disk_optimized",
        )
    }

    #[pg_test]
    unsafe fn test_bq_compressed_storage_index_creation_num_dimensions_disk() -> spi::Result<()> {
        crate::access_method::build::tests::test_index_creation_and_accuracy_scaffold(
            DistanceType::Cosine,
            "storage_layout = disk_optimized, num_dimensions=768",
            "bq_compressed_num_dimensions",
            3072,
        )?;
        Ok(())
    }

    #[pg_test]
    unsafe fn test_bq_compressed_storage_index_updates_cosine_disk() -> spi::Result<()> {
        crate::access_method::build::tests::test_index_updates(
            DistanceType::Cosine,
            "storage_layout = disk_optimized, num_neighbors=10",
            300,
            "bq_compressed",
        )?;
        Ok(())
    }

    #[pg_test]
    unsafe fn test_bq_compressed_storage_index_updates_l2_disk() -> spi::Result<()> {
        crate::access_method::build::tests::test_index_updates(
            DistanceType::L2,
            "storage_layout = disk_optimized, num_neighbors=10",
            300,
            "bq_compressed",
        )?;
        Ok(())
    }

    #[pg_test]
    unsafe fn test_bq_compressed_storage_index_updates_ip_disk() -> spi::Result<()> {
        crate::access_method::build::tests::test_index_updates(
            DistanceType::InnerProduct,
            "storage_layout = disk_optimized, num_neighbors=10",
            300,
            "bq_compressed",
        )?;
        Ok(())
    }
}
