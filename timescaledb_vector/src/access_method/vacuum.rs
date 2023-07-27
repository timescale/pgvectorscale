use pgrx::*;

#[pg_guard]
pub extern "C" fn ambulkdelete(
    _info: *mut pg_sys::IndexVacuumInfo,
    stats: *mut pg_sys::IndexBulkDeleteResult,
    _callback: pg_sys::IndexBulkDeleteCallback,
    _callback_state: *mut ::std::os::raw::c_void,
) -> *mut pg_sys::IndexBulkDeleteResult {
    let results = if stats.is_null() {
        unsafe { PgBox::<pg_sys::IndexBulkDeleteResult>::alloc0().into_pg() }
    } else {
        stats
    };
    //TODO: actually optimize the deletes by removing index tuples.
    //for now just have the indexscan do the right thing by looking at the heap.
    results
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
mod tests {
    use pgrx::*;

    #[test]
    fn test_delete_vacuum() {
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

        client
            .batch_execute(
                "CREATE TABLE test(embedding vector(3));

        INSERT INTO test(embedding) VALUES ('[1,2,3]'), ('[4,5,6]'), ('[7,8,10]');

        CREATE INDEX idxtest
              ON test
           USING tsv(embedding)
            WITH (num_neighbors=30);
            ",
            )
            .unwrap();

        client.execute("set enable_seqscan = 0;", &[]).unwrap();
        let cnt: i64 = client.query_one("WITH cte as (select * from test order by embedding <=> '[0,0,0]') SELECT count(*) from cte;", &[]).unwrap().get(0);

        assert_eq!(cnt, 3);

        client
            .execute("DELETE FROM test WHERE embedding = '[1,2,3]';", &[])
            .unwrap();

        client.execute("VACUUM test", &[]).unwrap();
        client.execute("DROP INDEX idxtest", &[]).unwrap();

        client.execute("set enable_seqscan = 0;", &[]).unwrap();
        let cnt: i64 = client.query_one("WITH cte as (select * from test order by embedding <=> '[0,0,0]') SELECT count(*) from cte;", &[]).unwrap().get(0);
        assert_eq!(cnt, 2);

        client.execute("DROP TABLE test", &[]).unwrap();
    }

    #[pg_test]
    ///This function is only a mock to bring up the test framewokr in test_delete_vacuum
    fn test_delete_mock_fn() -> spi::Result<()> {
        Ok(())
    }

    #[pg_test]
    unsafe fn test_delete() -> spi::Result<()> {
        Spi::run(&format!(
            "CREATE TABLE test(embedding vector(3));

            INSERT INTO test(embedding) VALUES ('[1,2,3]'), ('[4,5,6]'), ('[7,8,10]');

            CREATE INDEX idxtest
                  ON test
               USING tsv(embedding)
                WITH (num_neighbors=30);

            DELETE FROM test WHERE embedding = '[1,2,3]';
            ",
        ))?;

        let res: Option<i64> = Spi::get_one(&format!(
            "   set enable_seqscan = 0;
                WITH cte as (select * from test order by embedding <=> '[0,0,0]') SELECT count(*) from cte;",
        ))?;
        assert_eq!(2, res.unwrap());

        //delete same thing again -- should be a no-op;
        Spi::run(&format!("DELETE FROM test WHERE embedding = '[1,2,3]';",))?;
        let res: Option<i64> = Spi::get_one(&format!(
            "   set enable_seqscan = 0;
                WITH cte as (select * from test order by embedding <=> '[0,0,0]') SELECT count(*) from cte;",
        ))?;
        assert_eq!(2, res.unwrap());

        Spi::run(&format!("drop index idxtest;",))?;

        Ok(())
    }
}
