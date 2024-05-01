use pgrx::{
    pg_sys::{FirstOffsetNumber, IndexBulkDeleteResult},
    *,
};

use crate::{
    access_method::{bq::BqSpeedupStorage, meta_page::MetaPage, plain_storage::PlainStorage},
    util::{
        page::WritablePage,
        ports::{PageGetItem, PageGetItemId, PageGetMaxOffsetNumber},
        ItemPointer,
    },
};

use crate::access_method::storage::ArchivedData;

use super::storage::{Storage, StorageType};

#[pg_guard]
pub extern "C" fn ambulkdelete(
    info: *mut pg_sys::IndexVacuumInfo,
    stats: *mut pg_sys::IndexBulkDeleteResult,
    callback: pg_sys::IndexBulkDeleteCallback,
    callback_state: *mut ::std::os::raw::c_void,
) -> *mut pg_sys::IndexBulkDeleteResult {
    let results = if stats.is_null() {
        unsafe { PgBox::<pg_sys::IndexBulkDeleteResult>::alloc0().into_pg() }
    } else {
        stats
    };

    let index_relation = unsafe { PgRelation::from_pg((*info).index) };
    let nblocks = unsafe {
        pg_sys::RelationGetNumberOfBlocksInFork(
            index_relation.as_ptr(),
            pg_sys::ForkNumber_MAIN_FORKNUM,
        )
    };

    let meta_page = MetaPage::fetch(&index_relation);
    let storage = meta_page.get_storage_type();
    match storage {
        StorageType::BqSpeedup | StorageType::BqCompression => {
            bulk_delete_for_storage::<BqSpeedupStorage>(
                &index_relation,
                nblocks,
                results,
                callback,
                callback_state,
            );
        }
        StorageType::Plain => {
            bulk_delete_for_storage::<PlainStorage>(
                &index_relation,
                nblocks,
                results,
                callback,
                callback_state,
            );
        }
    }
    results
}

fn bulk_delete_for_storage<S: Storage>(
    index: &PgRelation,
    nblocks: u32,
    results: *mut IndexBulkDeleteResult,
    callback: pg_sys::IndexBulkDeleteCallback,
    callback_state: *mut ::std::os::raw::c_void,
) {
    for block_number in 0..nblocks {
        let page = unsafe { WritablePage::cleanup(&index, block_number) };
        if page.get_type() != S::page_type() {
            continue;
        }
        let mut modified = false;

        unsafe { pg_sys::vacuum_delay_point() };

        let max_offset = unsafe { PageGetMaxOffsetNumber(*page) };
        for offset_number in FirstOffsetNumber..(max_offset + 1) as _ {
            unsafe {
                let item_id = PageGetItemId(*page, offset_number);
                let item = PageGetItem(*page, item_id) as *mut u8;
                let len = (*item_id).lp_len();
                let data = std::slice::from_raw_parts_mut(item, len as _);
                let node = S::ArchivedType::with_data(data);

                if node.is_deleted() {
                    continue;
                }

                let heap_pointer: ItemPointer = node.get_heap_item_pointer();
                let mut ctid: pg_sys::ItemPointerData = pg_sys::ItemPointerData {
                    ..Default::default()
                };
                heap_pointer.to_item_pointer_data(&mut ctid);

                let deleted = callback.unwrap()(&mut ctid, callback_state);
                if deleted {
                    node.delete();
                    modified = true;
                    (*results).tuples_removed += 1.0;
                } else {
                    (*results).num_index_tuples += 1.0;
                }
            }
        }
        if modified {
            page.commit();
        }
    }
}

#[pg_guard]
pub extern "C" fn amvacuumcleanup(
    vinfo: *mut pg_sys::IndexVacuumInfo,
    stats: *mut pg_sys::IndexBulkDeleteResult,
) -> *mut pg_sys::IndexBulkDeleteResult {
    unsafe {
        if stats.is_null() || (*vinfo).analyze_only {
            return stats;
        }

        let index_relation = PgRelation::from_pg((*vinfo).index);

        (*stats).num_pages = pg_sys::RelationGetNumberOfBlocksInFork(
            index_relation.as_ptr(),
            pg_sys::ForkNumber_MAIN_FORKNUM,
        );

        stats
    }
}

#[cfg(any(test, feature = "pg_test"))]
#[pgrx::pg_schema]
pub mod tests {
    use once_cell::sync::Lazy;
    use pgrx::*;
    use std::sync::Mutex;

    static VAC_PLAIN_MUTEX: Lazy<Mutex<()>> = Lazy::new(Mutex::default);

    #[cfg(test)]
    pub fn test_delete_vacuum_plain_scaffold(index_options: &str) {
        //do not run this test in parallel. (pgrx tests run in a txn rolled back after each test, but we do not have that luxury here).
        let _lock = VAC_PLAIN_MUTEX.lock().unwrap();

        //we need to run vacuum in this test which cannot be run from SPI.
        //so we cannot use the pg_test framework here. Thus we do a bit of
        //hackery to bring up the test db and then use a client to run queries against it.

        //bring up the test db by running a fake test on a fake fn
        pgrx_tests::run_test(
            "test_delete_mock_fn",
            None,
            crate::pg_test::postgresql_conf_options(),
        )
        .unwrap();

        let suffix = (1..=253)
            .map(|i| format!("{}", i))
            .collect::<Vec<String>>()
            .join(", ");

        let (mut client, _) = pgrx_tests::client().unwrap();

        client
            .batch_execute(&format!(
                "CREATE TABLE test_vac(embedding vector(256));

        select setseed(0.5);
        -- generate 300 vectors
        INSERT INTO test_vac (embedding)
        SELECT
         *
        FROM (
            SELECT
        ('[ 0 , ' || array_to_string(array_agg(random()), ',', '0') || ']')::vector AS embedding
        FROM
         generate_series(1, 255 * 300) i
        GROUP BY
        i % 300) g;


        INSERT INTO test_vac(embedding) VALUES ('[1,2,3,{suffix}]'), ('[4,5,6,{suffix}]'), ('[7,8,10,{suffix}]');

        CREATE INDEX idxtest_vac
              ON test_vac
           USING tsv(embedding)
            WITH ({index_options});
            "
            ))
            .unwrap();

        client.execute("set enable_seqscan = 0;", &[]).unwrap();
        let cnt: i64 = client.query_one(&format!("WITH cte as (select * from test_vac order by embedding <=> '[1,1,1,{suffix}]') SELECT count(*) from cte;"), &[]).unwrap().get(0);

        assert_eq!(cnt, 303, "initial count");

        client
            .execute(
                &format!("DELETE FROM test_vac WHERE embedding = '[1,2,3,{suffix}]';"),
                &[],
            )
            .unwrap();

        client.close().unwrap();

        let (mut client, _) = pgrx_tests::client().unwrap();

        client.execute("VACUUM test_vac", &[]).unwrap();

        //inserts into the previous 1,2,3 spot that was deleted
        client
            .execute(
                &format!("INSERT INTO test_vac(embedding) VALUES ('[10,12,13,{suffix}]');"),
                &[],
            )
            .unwrap();

        client.execute("set enable_seqscan = 0;", &[]).unwrap();
        let cnt: i64 = client.query_one(&format!("WITH cte as (select * from test_vac order by embedding <=> '[1,1,1,{suffix}]') SELECT count(*) from cte;"), &[]).unwrap().get(0);
        //if the old index is still used the count is 304
        assert_eq!(cnt, 303, "count after vacuum");

        //do another delete for same items (noop)
        client
            .execute(
                &format!("DELETE FROM test_vac WHERE embedding = '[1,2,3,{suffix}]';"),
                &[],
            )
            .unwrap();

        client.execute("set enable_seqscan = 0;", &[]).unwrap();
        let cnt: i64 = client.query_one(&format!("WITH cte as (select * from test_vac order by embedding <=> '[1,1,1,{suffix}]') SELECT count(*) from cte;"), &[]).unwrap().get(0);
        //if the old index is still used the count is 304
        assert_eq!(cnt, 303, "count after delete");

        client.execute("DROP INDEX idxtest_vac", &[]).unwrap();
        client.execute("DROP TABLE test_vac", &[]).unwrap();
    }

    static VAC_FULL_MUTEX: Lazy<Mutex<()>> = Lazy::new(Mutex::default);

    #[cfg(test)]
    pub fn test_delete_vacuum_full_scaffold(index_options: &str) {
        //do not run this test in parallel
        let _lock = VAC_FULL_MUTEX.lock().unwrap();

        //we need to run vacuum in this test which cannot be run from SPI.
        //so we cannot use the pg_test framework here. Thus we do a bit of
        //hackery to bring up the test db and then use a client to run queries against it.

        //bring up the test db by running a fake test on a fake fn
        pgrx_tests::run_test(
            "test_delete_mock_fn",
            None,
            crate::pg_test::postgresql_conf_options(),
        )
        .unwrap();

        let (mut client, _) = pgrx_tests::client().unwrap();

        let suffix = (1..=253)
            .map(|i| format!("{}", i))
            .collect::<Vec<String>>()
            .join(", ");

        client
            .batch_execute(&format!(
                "CREATE TABLE test_vac_full(embedding vector(256));

        select setseed(0.5);
        -- generate 300 vectors
        INSERT INTO test_vac_full (embedding)
        SELECT
         *
        FROM (
            SELECT
        ('[ 0 , ' || array_to_string(array_agg(random()), ',', '0') || ']')::vector AS embedding
        FROM
         generate_series(1, 255 * 300) i
        GROUP BY
        i % 300) g;

        INSERT INTO test_vac_full(embedding) VALUES ('[1,2,3,{suffix}]'), ('[4,5,6,{suffix}]'), ('[7,8,10,{suffix}]');

        CREATE INDEX idxtest_vac_full
              ON test_vac_full
           USING tsv(embedding)
            WITH ({index_options});
            "
            ))
            .unwrap();

        client.execute("set enable_seqscan = 0;", &[]).unwrap();
        let cnt: i64 = client.query_one(&format!("WITH cte as (select * from test_vac_full order by embedding <=> '[1,1,1,{suffix}]') SELECT count(*) from cte;"), &[]).unwrap().get(0);
        std::thread::sleep(std::time::Duration::from_millis(10000));
        assert_eq!(cnt, 303, "initial count");

        client.execute("DELETE FROM test_vac_full", &[]).unwrap();

        client
            .execute(
                &format!(
                    "
                    INSERT INTO test_vac_full (embedding)
                    SELECT
                     *
                    FROM (
                        SELECT
                    ('[ 0 , ' || array_to_string(array_agg(random()), ',', '0') || ']')::vector AS embedding
                    FROM
                     generate_series(1, 255 * 300) i
                    GROUP BY
                    i % 300) g;
                    "
                ),
                &[],
            )
            .unwrap();

        client.close().unwrap();

        let (mut client, _) = pgrx_tests::client().unwrap();
        client.execute("VACUUM FULL test_vac_full", &[]).unwrap();

        client
            .execute(
                &format!("INSERT INTO test_vac_full(embedding) VALUES ('[1,2,3,{suffix}]');"),
                &[],
            )
            .unwrap();

        client.execute("set enable_seqscan = 0;", &[]).unwrap();
        let cnt: i64 = client.query_one(&format!("WITH cte as (select * from test_vac_full order by embedding <=> '[1,1,1,{suffix}]') SELECT count(*) from cte;"), &[]).unwrap().get(0);
        assert_eq!(cnt, 301, "count after full vacuum");

        client.execute("DROP INDEX idxtest_vac_full", &[]).unwrap();
        client.execute("DROP TABLE test_vac_full", &[]).unwrap();
    }

    #[pg_test]
    ///This function is only a mock to bring up the test framewokr in test_delete_vacuum
    fn test_delete_mock_fn() -> spi::Result<()> {
        Ok(())
    }
}
