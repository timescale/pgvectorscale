#[cfg(any(test, feature = "pg_test"))]
#[pgrx::pg_schema]
pub mod tests {
    use pgrx::prelude::*;
    use pgrx::spi;

    // Helper function to ensure clean test environment
    unsafe fn cleanup_test_tables() -> spi::Result<()> {
        // Drop tables if they exist
        Spi::run("DROP TABLE IF EXISTS test_null_labels;")?;
        Spi::run("DROP TABLE IF EXISTS test_nonempty;")?;
        Spi::run("DROP TABLE IF EXISTS test_mixed_labels;")?;
        Spi::run("DROP TABLE IF EXISTS test_update_labels;")?;
        Spi::run("DROP TABLE IF EXISTS test_labeled;")?;
        Spi::run("DROP TABLE IF EXISTS test_overlap;")?;
        Spi::run("DROP TABLE IF EXISTS test_unusual_order;")?;
        Spi::run("DROP TABLE IF EXISTS label_definitions;")?;
        Spi::run("DROP TABLE IF EXISTS test_complex_order_by;")?;
        Ok(())
    }

    #[pg_test]
    pub unsafe fn test_null_and_empty_labels() -> spi::Result<()> {
        // Ensure clean environment
        cleanup_test_tables()?;

        // Create test table and index
        Spi::run(
            "CREATE TABLE test_null_labels (
                    id SERIAL PRIMARY KEY,
                    embedding vector(3),
                    labels SMALLINT[]
                );

                CREATE INDEX idx_null_labels ON test_null_labels USING diskann (embedding, labels);

                -- Insert data with various label scenarios
                INSERT INTO test_null_labels (embedding, labels) VALUES
                ('[1,2,3]', '{1,2}'),           -- Normal case
                ('[4,5,6]', NULL),              -- NULL array
                ('[7,8,9]', '{}'),              -- Empty array
                ('[10,11,12]', '{1,NULL,3}');   -- Array with NULL element
                ",
        )?;

        // Test 1: Query with normal labels
        let res: Option<i64> = Spi::get_one(
            "
                SET enable_seqscan = 0;
                WITH cte AS (
                    SELECT * FROM test_null_labels
                    WHERE labels && '{1}'
                    ORDER BY embedding <=> '[0,0,0]'
                )
                SELECT COUNT(*) FROM cte;",
        )?;
        assert_eq!(2, res.unwrap(), "Should find 2 documents with label 1");

        // Test 2: Query with empty labels in WHERE clause
        let res: Option<i64> = Spi::get_one(
            "
                SET enable_seqscan = 0;
                WITH cte AS (
                    SELECT * FROM test_null_labels
                    WHERE labels && '{}'
                    ORDER BY embedding <=> '[0,0,0]'
                )
                SELECT COUNT(*) FROM cte;",
        )?;
        assert_eq!(
            0,
            res.unwrap(),
            "Should find 0 documents since `... && '{{}}'` is always false"
        );

        // Test 3: Query with array containing NULL element
        let res: Option<i64> = Spi::get_one(
            "
                SET enable_seqscan = 0;
                WITH cte AS (
                    SELECT * FROM test_null_labels
                    WHERE labels && '{3}'
                    ORDER BY embedding <=> '[0,0,0]'
                )
                SELECT COUNT(*) FROM cte;",
        )?;
        assert_eq!(
            1,
            res.unwrap(),
            "Should find 1 document with label 3 (array with NULL element)"
        );

        // Test 4: Query with no label filtering
        let res: Option<i64> = Spi::get_one(
            "
                SET enable_seqscan = 0;
                WITH cte AS (
                    SELECT * FROM test_null_labels
                    ORDER BY embedding <=> '[0,0,0]'
                )
                SELECT COUNT(*) FROM cte;",
        )?;
        assert_eq!(4, res.unwrap(), "Should find 4 documents");

        // Clean up
        cleanup_test_tables()?;

        Ok(())
    }

    #[pg_test]
    pub unsafe fn test_build_index_on_nonempty_table() -> spi::Result<()> {
        // Ensure clean environment
        cleanup_test_tables()?;

        // First create a table and populate it
        Spi::run(
            "CREATE TABLE test_nonempty (
                    id SERIAL PRIMARY KEY,
                    embedding vector(3),
                    labels SMALLINT[]
                );

                -- Insert data before creating the index
                INSERT INTO test_nonempty (embedding, labels) VALUES
                ('[1,2,3]', '{1,2}'),
                ('[4,5,6]', '{1,3}'),
                ('[7,8,9]', '{2,3}'),
                ('[10,11,12]', '{4,5}'),
                ('[13,14,15]', NULL);
                ",
        )?;

        // Now create the index on the non-empty table
        Spi::run("CREATE INDEX idx_nonempty ON test_nonempty USING diskann (embedding, labels);")?;

        // Test 1: Basic label filtering
        let res: Option<i64> = Spi::get_one(
            "
                SET enable_seqscan = 0;
                WITH cte AS (
                    SELECT * FROM test_nonempty
                    WHERE labels && '{1}'
                    ORDER BY embedding <=> '[0,0,0]'
                )
                SELECT COUNT(*) FROM cte;",
        )?;
        assert_eq!(2, res.unwrap(), "Should find 2 documents with label 1");

        // Test 2: Multiple label filtering
        let res: Option<i64> = Spi::get_one(
            "
                SET enable_seqscan = 0;
                WITH cte AS (
                    SELECT * FROM test_nonempty
                    WHERE labels && '{2,3}'
                    ORDER BY embedding <=> '[0,0,0]'
                )
                SELECT COUNT(*) FROM cte;",
        )?;
        assert_eq!(3, res.unwrap(), "Should find 3 documents with label 2 OR 3");

        // Clean up
        cleanup_test_tables()?;

        Ok(())
    }

    #[pg_test]
    pub unsafe fn test_mixed_filtering_with_null_labels() -> spi::Result<()> {
        // Ensure clean environment
        cleanup_test_tables()?;

        Spi::run(
                "CREATE TABLE test_mixed_labels (
                    id SERIAL PRIMARY KEY,
                    embedding vector(3),
                    labels SMALLINT[],
                    category TEXT
                );

                CREATE INDEX idx_mixed_labels ON test_mixed_labels USING diskann (embedding, labels);

                -- Insert data with mixed scenarios
                INSERT INTO test_mixed_labels (embedding, labels, category) VALUES
                ('[1,2,3]', '{1,2}', 'article'),
                ('[4,5,6]', NULL, 'blog'),
                ('[7,8,9]', '{}', 'article'),
                ('[10,11,12]', '{1,3}', 'blog'),
                ('[13,14,15]', '{2,3}', 'article'),
                ('[16,17,18]', '{NULL}', 'blog');
                ",
            )?;

        // Test 1: Combining label filtering with category
        let res: Option<i64> = Spi::get_one(
            "
                SET enable_seqscan = 0;
                WITH cte AS (
                    SELECT * FROM test_mixed_labels
                    WHERE labels && '{1}' AND category = 'blog'
                    ORDER BY embedding <=> '[0,0,0]'
                )
                SELECT COUNT(*) FROM cte;",
        )?;
        assert_eq!(1, res.unwrap(), "Should find 1 blog with label 1");

        // Clean up
        cleanup_test_tables()?;

        Ok(())
    }

    #[pg_test]
    pub unsafe fn test_update_labels_after_index_creation() -> spi::Result<()> {
        // Ensure clean environment
        cleanup_test_tables()?;

        // Create table and index
        Spi::run(
                "CREATE TABLE test_update_labels (
                    id SERIAL PRIMARY KEY,
                    embedding vector(3),
                    labels SMALLINT[]
                );

                -- Insert initial data
                INSERT INTO test_update_labels (embedding, labels) VALUES
                ('[1,2,3]', '{1,2}'),
                ('[4,5,6]', '{3,4}');

                -- Create index on non-empty table
                CREATE INDEX idx_update_labels ON test_update_labels USING diskann (embedding, labels);
                ",
            )?;

        // Test initial state
        let res: Option<i64> = Spi::get_one(
            "
                SET enable_seqscan = 0;
                WITH cte AS (
                    SELECT * FROM test_update_labels
                    WHERE labels && '{1}'
                    ORDER BY embedding <=> '[0,0,0]'
                )
                SELECT COUNT(*) FROM cte;",
        )?;
        assert_eq!(
            1,
            res.unwrap(),
            "Should find 1 document with label 1 initially"
        );

        // Update labels
        Spi::run(
            "
                -- Update existing row
                UPDATE test_update_labels SET labels = '{1,5}' WHERE id = 2;

                -- Insert new rows with edge cases
                INSERT INTO test_update_labels (embedding, labels) VALUES
                ('[7,8,9]', NULL),
                ('[10,11,12]', '{}');
                ",
        )?;

        // Test after updates
        let res: Option<i64> = Spi::get_one(
            "
                SET enable_seqscan = 0;
                WITH cte AS (
                    SELECT * FROM test_update_labels
                    WHERE labels && '{1}'
                    ORDER BY embedding <=> '[0,0,0]'
                )
                SELECT COUNT(*) FROM cte;",
        )?;
        assert_eq!(
            2,
            res.unwrap(),
            "Should find 2 documents with label 1 after update"
        );

        // Clean up
        cleanup_test_tables()?;

        Ok(())
    }

    #[pg_test]
    pub unsafe fn test_labeled_filtering_with_category() -> spi::Result<()> {
        // Ensure clean environment
        cleanup_test_tables()?;

        Spi::run(
            "CREATE TABLE test_labeled (
            id SERIAL PRIMARY KEY,
            embedding vector(3),
            labels SMALLINT[],
            category TEXT
        );

        CREATE INDEX idx_labeled_diskann ON test_labeled USING diskann (embedding, labels);

        INSERT INTO test_labeled (embedding, labels, category) VALUES
        ('[1,2,3]', '{1,2}', 'article'),
        ('[4,5,6]', '{1,3}', 'blog'),
        ('[7,8,9]', '{2,3}', 'article');",
        )?;

        // Test 1: Basic label filtering
        let res: Option<i64> = Spi::get_one(
            "
        SET enable_seqscan = 0;
        WITH cte AS (
            SELECT * FROM test_labeled
            WHERE labels && '{1}'
            ORDER BY embedding <=> '[0,0,0]'
        )
        SELECT COUNT(*) FROM cte;",
        )?;
        assert_eq!(2, res.unwrap(), "Should find 2 documents with label 1");

        // Test 2: Combining label filtering with category
        let res: Option<i64> = Spi::get_one(
            "
        SET enable_seqscan = 0;
        WITH cte AS (
            SELECT * FROM test_labeled
            WHERE labels && '{2}' AND category = 'article'
            ORDER BY embedding <=> '[0,0,0]'
        )
        SELECT COUNT(*) FROM cte;",
        )?;
        assert_eq!(2, res.unwrap(), "Should find 2 articles with label 2");

        // Clean up
        Spi::run("DROP TABLE test_labeled;")?;

        Ok(())
    }

    #[pg_test]
    pub unsafe fn test_unusual_column_order() -> spi::Result<()> {
        // Ensure clean environment
        cleanup_test_tables()?;

        Spi::run(
            "CREATE TABLE test_unusual_order (
            id SERIAL PRIMARY KEY,
            labels SMALLINT[],
            comments TEXT,
            embedding vector(3)
        );

        CREATE INDEX idx_unusual_order ON test_unusual_order USING diskann (embedding, labels);

        INSERT INTO test_unusual_order (embedding, labels, comments) VALUES
        ('[1,2,3]', '{1,2}', 'This is a comment'),
        ('[4,5,6]', '{1,3}', 'Another comment'),
        ('[7,8,9]', '{2,3}', 'Yet another comment');
        ",
        )?;

        let res: Option<i64> = Spi::get_one(
            "
        SET enable_seqscan = 0;
        WITH cte AS (
            SELECT * FROM test_unusual_order WHERE labels && '{1}' ORDER BY embedding <=> '[0,0,0]'
        )
        SELECT COUNT(*) FROM cte;",
        )?;
        assert_eq!(2, res.unwrap(), "Should find 2 documents with label 1");

        Spi::run("DROP TABLE test_unusual_order;")?;

        Ok(())
    }

    #[pg_test]
    pub unsafe fn test_complex_order_by() -> spi::Result<()> {
        // Ensure clean environment
        cleanup_test_tables()?;

        Spi::run("CREATE TABLE test_complex_order_by (
            id SERIAL PRIMARY KEY,
            embedding vector(3),
            labels SMALLINT[]
        );

        CREATE INDEX idx_complex_order_by ON test_complex_order_by USING diskann (embedding, labels);

        INSERT INTO test_complex_order_by (embedding, labels) VALUES
        ('[1,2,3]', '{1,2}'),
        ('[4,5,6]', '{1,3}'),
        ('[7,8,9]', '{2,3}');
        ")?;

        // Tests 1: order by distance, labels.  Index should be used.
        let res: Option<i64> = Spi::get_one(
            "
        SET enable_seqscan = 0;
        WITH cte AS (
            SELECT * FROM test_complex_order_by
            WHERE labels && '{1}'
            ORDER BY embedding <=> '[0,0,0]', labels
        )
        SELECT COUNT(*) FROM cte;",
        )?;
        assert_eq!(2, res.unwrap(), "Should find 2 documents with label 1");

        // Ensure that the index is used.  Test seems to be unreliable pre-pg17.
        #[cfg(feature = "pg17")]
        {
            let res = Spi::explain(
                "SELECT * FROM test_complex_order_by WHERE labels && '{1}' ORDER BY embedding <=> '[0,0,0]', labels;"
            )?;
            let res_str = format!("{:?}", res);
            assert!(
                res_str.contains("idx_complex_order_by"),
                "Index should be used"
            );
        }

        // Tests 2: order by labels, distance.  Index cannot be used.
        let res: Option<i64> = Spi::get_one(
            "
        SET enable_seqscan = 0;
        WITH cte AS (
            SELECT * FROM test_complex_order_by
            WHERE labels && '{1}'
            ORDER BY labels, embedding <=> '[0,0,0]'
        )
        SELECT COUNT(*) FROM cte;",
        )?;
        assert_eq!(2, res.unwrap(), "Should find 2 documents with label 1");

        // Ensure that the index is not used
        let res = Spi::explain(
            "SELECT * FROM test_complex_order_by WHERE labels && '{1}' ORDER BY labels, embedding <=> '[0,0,0]';"
        )?;
        let res_str = format!("{:?}", res);
        assert!(
            !res_str.contains("idx_complex_order_by"),
            "Index should not be used"
        );

        // Test 3: parameterize the vector and order by distance, labels.  Index should be used.
        let vector = vec![0, 0, 0];
        let res: Option<i64> = Spi::get_one_with_args(
            "
        SET enable_seqscan = 0;
        WITH cte AS (
            SELECT * FROM test_complex_order_by
            WHERE labels && '{1}'
            ORDER BY embedding <=> $1::vector, labels
        )
        SELECT COUNT(*) FROM cte;",
            vec![(
                pgrx::PgOid::Custom(pgrx::pg_sys::FLOAT4ARRAYOID),
                vector.clone().into_datum(),
            )],
        )?;
        assert_eq!(2, res.unwrap(), "Should find 2 documents with label 1");

        // Ensure that the index is used.  Test seems to be unreliable pre-pg17.
        #[cfg(feature = "pg17")]
        {
            let res = Spi::explain(
                "SELECT * FROM test_complex_order_by WHERE labels && '{1}' ORDER BY embedding <=> '[0,0,0]', labels;"
            )?;
            let res_str = format!("{:?}", res);
            assert!(
                res_str.contains("idx_complex_order_by"),
                "Index should be used"
            );
        }

        // Test 4: parameterize the vector and order by labels, distance.  Index cannot be used.
        let res: Option<i64> = Spi::get_one_with_args(
            "
        SET enable_seqscan = 0;
        WITH cte AS (
            SELECT * FROM test_complex_order_by
            WHERE labels && '{1}'
            ORDER BY labels, embedding <=> $1::vector
        )
        SELECT COUNT(*) FROM cte;",
            vec![(
                pgrx::PgOid::Custom(pgrx::pg_sys::FLOAT4ARRAYOID),
                vector.into_datum(),
            )],
        )?;
        assert_eq!(2, res.unwrap(), "Should find 2 documents with label 1");

        // Ensure that the index is not used
        let res = Spi::explain(
            "SELECT * FROM test_complex_order_by WHERE labels && '{1}' ORDER BY labels, embedding <=> '[0,0,0]';"
        )?;
        let res_str = format!("{:?}", res);
        assert!(
            !res_str.contains("idx_complex_order_by"),
            "Index should not be used"
        );

        Spi::run("DROP TABLE test_complex_order_by;")?;

        Ok(())
    }

    #[pg_test]
    pub unsafe fn test_labeled_filtering_with_label_definitions() -> spi::Result<()> {
        // Ensure clean environment
        cleanup_test_tables()?;

        Spi::run(
            "CREATE TABLE test_labeled (
            id SERIAL PRIMARY KEY,
            embedding vector(3),
            labels SMALLINT[],
            content TEXT
        );

        CREATE TABLE label_definitions (
            id INTEGER PRIMARY KEY,
            name TEXT,
            description TEXT
        );

        CREATE INDEX idx_labeled_diskann ON test_labeled USING diskann (embedding, labels);

        INSERT INTO label_definitions (id, name, description) VALUES
        (1, 'science', 'Scientific content'),
        (2, 'technology', 'Technology-related content'),
        (3, 'business', 'Business content');

        INSERT INTO test_labeled (embedding, labels, content) VALUES
        ('[1,2,3]', '{1,2}', 'Science and technology article'),
        ('[4,5,6]', '{1,3}', 'Science and business blog'),
        ('[7,8,9]', '{2,3}', 'Technology and business news');",
        )?;

        // Test 1: Filter by label name using subquery
        let res: Option<i64> = Spi::get_one(
            "
        SET enable_seqscan = 0;
        WITH cte AS (
            SELECT * FROM test_labeled
            WHERE labels && (
                SELECT array_agg(id::smallint) FROM label_definitions WHERE name = 'science'
            )
            ORDER BY embedding <=> '[0,0,0]'
        )
        SELECT COUNT(*) FROM cte;",
        )?;
        assert_eq!(
            2,
            res.unwrap(),
            "Should find 2 documents with science label"
        );

        // Test 2: Join with label definitions to get label names
        let res: Option<String> = Spi::get_one(
            "
        SET enable_seqscan = 0;
        WITH labeled_docs AS (
            SELECT t.id, t.content, array_agg(l.name) as label_names
            FROM test_labeled t
            JOIN label_definitions l ON l.id = ANY(t.labels)
            WHERE t.labels && '{1}'
            GROUP BY t.id, t.content
            ORDER BY t.embedding <=> '[0,0,0]'
            LIMIT 1
        )
        SELECT array_to_string(label_names, ',') FROM labeled_docs;",
        )?;

        // The result should contain 'science' since we're filtering for label 1
        assert!(
            res.unwrap().contains("science"),
            "Result should contain the science label"
        );

        // Clean up
        Spi::run("DROP TABLE test_labeled; DROP TABLE label_definitions;")?;

        Ok(())
    }

    #[pg_test]
    pub unsafe fn test_labeled_filtering_complex() -> spi::Result<()> {
        Spi::run(
            "CREATE TABLE test_labeled (
            id SERIAL PRIMARY KEY,
            embedding vector(3),
            labels SMALLINT[],
            category TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX idx_labeled_diskann ON test_labeled USING diskann (embedding, labels);

        INSERT INTO test_labeled (embedding, labels, category) VALUES
        ('[1,2,3]', '{1,2}', 'article'),
        ('[4,5,6]', '{1,3}', 'blog'),
        ('[7,8,9]', '{2,3}', 'article'),
        ('[10,11,12]', '{2,4}', 'blog'),
        ('[13,14,15]', '{3,4}', 'article'),
        ('[16,17,18]', '{1,2,3}', 'blog');",
        )?;

        // Test 1: Multiple label filtering (OR condition)
        let res: Option<i64> = Spi::get_one(
            "
        SET enable_seqscan = 0;
        WITH cte AS (
            SELECT * FROM test_labeled
            WHERE labels && '{1,4}'
            ORDER BY embedding <=> '[0,0,0]'
        )
        SELECT COUNT(*) FROM cte;",
        )?;
        assert_eq!(5, res.unwrap(), "Should find 5 documents with label 1 OR 4");

        // Test 2: Complex filtering with CTE and multiple conditions
        let res: Option<i64> = Spi::get_one(
            "
        SET enable_seqscan = 0;
        WITH filtered_docs AS (
            SELECT * FROM test_labeled
            WHERE labels && '{2,3}'
            ORDER BY embedding <=> '[0,0,0]'
        )
        SELECT COUNT(*) FROM filtered_docs WHERE category = 'article';",
        )?;
        assert_eq!(3, res.unwrap(), "Should find 3 articles with label 2 OR 3");

        // Test 3: Combining label filtering with array length check
        let res: Option<i64> = Spi::get_one(
            "
        SET enable_seqscan = 0;
        WITH filtered_docs AS (
            SELECT * FROM test_labeled
            WHERE labels && '{1}' AND array_length(labels, 1) > 2
            ORDER BY embedding <=> '[0,0,0]'
        )
        SELECT COUNT(*) FROM filtered_docs;",
        )?;
        assert_eq!(
            1,
            res.unwrap(),
            "Should find 1 document with label 1 and more than 2 labels"
        );

        // Clean up
        Spi::run("DROP TABLE test_labeled;")?;

        Ok(())
    }

    #[pg_test]
    pub unsafe fn test_tiny_labeled_index() -> spi::Result<()> {
        Spi::run(
            "CREATE TABLE test(embedding vector(3), labels smallint[]);

            CREATE INDEX idxtest
                  ON test
               USING diskann(embedding)
                WITH (num_neighbors=15, search_list_size=10);

            INSERT INTO test(embedding, labels) VALUES ('[1,2,3]', '{1,2}'), ('[4,5,6]', '{1,3}'), ('[7,8,10]', '{2,3}');
            ",
        )?;

        let res: Option<i64> = Spi::get_one("   set enable_seqscan = 0;
                WITH cte as (select * from test order by embedding <=> '[0,0,0]') SELECT count(*) from cte;")?;
        assert_eq!(3, res.unwrap());

        let res: Option<i64> = Spi::get_one("   set enable_seqscan = 0;
                WITH cte as (select * from test where labels && '{1}' order by embedding <=> '[0,0,0]') SELECT count(*) from cte;")?;
        assert_eq!(2, res.unwrap());

        let res: Option<i64> = Spi::get_one("   set enable_seqscan = 0;
                WITH cte as (select * from test where labels && '{2}' order by embedding <=> '[0,0,0]') SELECT count(*) from cte;")?;
        assert_eq!(2, res.unwrap());

        let res: Option<i64> = Spi::get_one("   set enable_seqscan = 0;
                WITH cte as (select * from test where labels && '{3}' order by embedding <=> '[0,0,0]') SELECT count(*) from cte;")?;
        assert_eq!(2, res.unwrap());

        let res: Option<i64> = Spi::get_one("   set enable_seqscan = 0;
        WITH cte as (select * from test where labels && '{1,3}' order by embedding <=> '[0,0,0]') SELECT count(*) from cte;")?;
        assert_eq!(3, res.unwrap());

        let res: Option<i64> = Spi::get_one("   set enable_seqscan = 0;
        WITH cte as (select * from test where labels && '{1,2,3}' order by embedding <=> '[0,0,0]') SELECT count(*) from cte;")?;
        assert_eq!(3, res.unwrap());

        let res: Option<i64> = Spi::get_one("   set enable_seqscan = 0;
        WITH cte as (select * from test where labels && '{4}' order by embedding <=> '[0,0,0]') SELECT count(*) from cte;")?;
        assert_eq!(0, res.unwrap());

        let res: Option<i64> = Spi::get_one("   set enable_seqscan = 0;
        WITH cte as (select * from test where labels && '{1,4}' order by embedding <=> '[0,0,0]') SELECT count(*) from cte;")?;
        assert_eq!(2, res.unwrap());

        let res: Option<i64> = Spi::get_one("   set enable_seqscan = 0;
        WITH cte as (select * from test where labels && '{4,1}' order by embedding <=> '[0,0,0]') SELECT count(*) from cte;")?;
        assert_eq!(2, res.unwrap());

        Spi::run("DROP TABLE test;")?;

        Ok(())
    }

    #[pg_test]
    pub unsafe fn test_label_size_bounds() -> spi::Result<()> {
        // Create a test table with vector and labels columns
        Spi::run(
            "CREATE TABLE test_label_bounds(embedding vector(3), labels smallint[]);

            CREATE INDEX idx_label_bounds
                  ON test_label_bounds
               USING diskann(embedding, labels);

            -- These inserts should succeed (labels within smallint bounds)
            INSERT INTO test_label_bounds(embedding, labels) VALUES ('[1,2,3]', '{0,32767}');
            ",
        )?;

        // Verify the valid labels were inserted correctly using the && operator
        let res: Option<i64> = Spi::get_one(
            "SELECT COUNT(*) FROM test_label_bounds
             WHERE labels && ARRAY[32767]::smallint[];",
        )?;
        assert_eq!(
            1,
            res.unwrap(),
            "Should find 1 document with label 32767 (max smallint value)"
        );

        // Test inserting a label that exceeds u16 bounds - this should fail
        // Use PL/pgSQL's exception handling to catch the error
        let error_captured: Option<bool> = Spi::get_one(
            "DO $$
            BEGIN
                -- This should fail because 32768 is outside smallint bounds
                INSERT INTO test_label_bounds(embedding, labels) VALUES ('[4,5,6]', '{32768}');
                -- If we get here, no error occurred
                RAISE NOTICE 'Test failed: expected error for out-of-bounds label but none occurred';
                PERFORM pg_catalog.set_config('pgrx.tests.failed', 'true', false);
            EXCEPTION WHEN OTHERS THEN
                -- We expect an error, so this is good
                RAISE NOTICE 'Got expected error: %', SQLERRM;
                -- Check if the error message contains what we expect
                IF SQLERRM LIKE '%out of range%' OR SQLERRM LIKE '%32768%' THEN
                    PERFORM pg_catalog.set_config('pgrx.tests.failed', 'false', false);
                ELSE
                    RAISE NOTICE 'Error message did not contain expected text: %', SQLERRM;
                    PERFORM pg_catalog.set_config('pgrx.tests.failed', 'true', false);
                END IF;
            END;
            $$ LANGUAGE plpgsql;

            -- Return whether the test failed
            SELECT current_setting('pgrx.tests.failed', true) = 'true';"
        )?;

        assert!(
            !error_captured.unwrap_or(true),
            "Test for out-of-bounds label (32768) failed"
        );

        // Test with negative label - this should now succeed since smallint allows negative values
        Spi::run("INSERT INTO test_label_bounds(embedding, labels) VALUES ('[7,8,9]', '{-1}');")?;

        // Verify the negative label was inserted correctly using the && operator
        let res: Option<i64> = Spi::get_one(
            "SELECT COUNT(*) FROM test_label_bounds
             WHERE labels && ARRAY[-1]::smallint[];",
        )?;
        assert_eq!(
            1,
            res.unwrap(),
            "Should find 1 document with negative label (-1)"
        );

        // Clean up
        Spi::run("DROP TABLE test_label_bounds;")?;

        Ok(())
    }

    #[pg_test]
    pub unsafe fn test_smallint_array_overlap() -> spi::Result<()> {
        // Test the smallint_array_overlap function directly
        Spi::run(
            "CREATE TABLE test_overlap (
                id SERIAL PRIMARY KEY,
                array1 SMALLINT[],
                array2 SMALLINT[]
            );

            INSERT INTO test_overlap (array1, array2) VALUES
            ('{1,2,3}', '{3,4,5}'),       -- Overlap: 3 (sorted)
            ('{-10,20,30}', '{40,50}'),   -- No overlap (sorted)
            ('{-3,-2,-1}', '{-5,-4,-3}'), -- Overlap: -3 (sorted)
            ('{0}', '{0}'),               -- Overlap: 0 (sorted)
            ('{32767}', '{32767}'),       -- Overlap: max smallint (sorted)
            ('{-32768}', '{-32768}'),     -- Overlap: min smallint (sorted)
            ('{}', '{1,2,3}'),            -- Empty array test
            ('{1,2,3}', '{}');            -- Empty array test
            ",
        )?;

        // Test cases where overlap exists using the && operator
        let res: Option<i64> =
            Spi::get_one("SELECT COUNT(*) FROM test_overlap WHERE array1 && array2;")?;
        assert_eq!(
            5,
            res.unwrap(),
            "Should find 5 rows with overlapping arrays"
        );

        // Test specific overlap cases using the && operator
        let res: Option<bool> =
            Spi::get_one("SELECT ARRAY[1,2,3]::smallint[] && ARRAY[3,4,5]::smallint[];")?;
        assert!(
            res.unwrap(),
            "Arrays {{1,2,3}} and {{3,4,5}} should overlap"
        );

        let res: Option<bool> =
            Spi::get_one("SELECT ARRAY[-10,20,30]::smallint[] && ARRAY[40,50]::smallint[];")?;
        assert!(
            !res.unwrap(),
            "Arrays {{-10,20,30}} and {{40,50}} should not overlap"
        );

        // Test with unsorted arrays
        let res: Option<bool> =
            Spi::get_one("SELECT ARRAY[3,1,2]::smallint[] && ARRAY[5,3,4]::smallint[];")?;
        assert!(
            res.unwrap(),
            "Arrays {{3,1,2}} and {{5,3,4}} should overlap"
        );

        // Test with empty arrays using the && operator
        let res: Option<bool> =
            Spi::get_one("SELECT ARRAY[]::smallint[] && ARRAY[1,2,3]::smallint[];")?;
        assert!(
            !res.unwrap(),
            "Empty array should not overlap with any array"
        );

        // Test with boundary values using the && operator
        let res: Option<bool> =
            Spi::get_one("SELECT ARRAY[32767]::smallint[] && ARRAY[32767]::smallint[];")?;
        assert!(
            res.unwrap(),
            "Arrays with max smallint value should overlap"
        );

        let res: Option<bool> =
            Spi::get_one("SELECT ARRAY[-32768]::smallint[] && ARRAY[-32768]::smallint[];")?;
        assert!(
            res.unwrap(),
            "Arrays with min smallint value should overlap"
        );

        // Clean up
        Spi::run("DROP TABLE test_overlap;")?;

        Ok(())
    }

    // For simplicity, only run this test on pg version 16 and above.  Otherwise, we have
    // to choose different seeds for different pg versions to get the test to pass.
    #[cfg(any(feature = "pg16", feature = "pg17"))]
    #[pg_test]
    pub unsafe fn test_labeled_recall() -> spi::Result<()> {
        // Ensure clean environment
        cleanup_test_tables()?;

        // Make it reproducible
        Spi::run("SELECT setseed(0.2);")?;

        // Create test table with 1K vectors and random labels
        Spi::run(
            "CREATE TABLE test_recall (
                id SERIAL PRIMARY KEY,
                embedding vector(128),
                labels SMALLINT[]
            );

            -- Generate 1K vectors with random labels from [1,32]
            INSERT INTO test_recall (embedding, labels)
            SELECT 
                ('[' || array_to_string(array_agg(random() * 2 - 1), ',') || ']')::vector,
                ARRAY[floor(random() * 32 + 1)::smallint]
            FROM generate_series(1, 128 * 1000) i
            GROUP BY i % 1000;

            -- Create a query vector for testing
            CREATE TEMP TABLE query_vector AS
            SELECT ('[' || array_to_string(array_agg(random() * 2 - 1), ',') || ']')::vector AS vec
            FROM generate_series(1, 128);",
        )?;

        // Compute ground truth results for different query types
        // 1. No filter
        let ground_truth_no_filter: Vec<String> = Spi::get_one(
            "WITH cte AS (
                SELECT id, embedding <=> (SELECT vec FROM query_vector) AS dist
                FROM test_recall
                ORDER BY dist
                LIMIT 10
            )
            SELECT array_agg(id::text) FROM cte;",
        )?
        .unwrap();

        // 2. Single label filter (pick a random label that exists)
        let ground_truth_single_label: Vec<String> = Spi::get_one(
            "WITH cte AS (
                SELECT id, embedding <=> (SELECT vec FROM query_vector) AS dist
                FROM test_recall
                WHERE labels && ARRAY[1]::smallint[]
                ORDER BY dist
                LIMIT 10
            )
            SELECT array_agg(id::text) FROM cte;",
        )?
        .unwrap();

        // 3. Two label filter (pick two random labels that exist)
        let ground_truth_two_labels: Vec<String> = Spi::get_one(
            "WITH cte AS (
                SELECT id, embedding <=> (SELECT vec FROM query_vector) AS dist
                FROM test_recall
                WHERE labels && ARRAY[1,2]::smallint[]
                ORDER BY dist
                LIMIT 10
            )
            SELECT array_agg(id::text) FROM cte;",
        )?
        .unwrap();

        // Create the index
        Spi::run("CREATE INDEX idx_recall ON test_recall USING diskann (embedding, labels);")?;

        // Run queries with index and compute recall
        let compute_recall = |ground_truth: &[String], query: &str| -> f64 {
            let indexed_results: Vec<String> = Spi::get_one(query).unwrap().unwrap();
            let ground_truth_set: std::collections::HashSet<_> = ground_truth.iter().collect();
            let matches = indexed_results
                .iter()
                .filter(|id| ground_truth_set.contains(id))
                .count();
            matches as f64 / ground_truth.len() as f64
        };

        // 1. No filter recall
        let recall_no_filter = compute_recall(
            &ground_truth_no_filter,
            "SET enable_seqscan = 0;
             WITH cte AS (
                 SELECT id, embedding <=> (SELECT vec FROM query_vector) AS dist
                 FROM test_recall
                 ORDER BY dist
                 LIMIT 10
             )
             SELECT array_agg(id::text) FROM cte;",
        );

        // 2. Single label recall
        let recall_single_label = compute_recall(
            &ground_truth_single_label,
            "SET enable_seqscan = 0;
             WITH cte AS (
                 SELECT id, embedding <=> (SELECT vec FROM query_vector) AS dist
                 FROM test_recall
                 WHERE labels && ARRAY[1]::smallint[]
                 ORDER BY dist
                 LIMIT 10
             )
             SELECT array_agg(id::text) FROM cte;",
        );

        // 3. Two label recall
        let recall_two_labels = compute_recall(
            &ground_truth_two_labels,
            "SET enable_seqscan = 0;
             WITH cte AS (
                 SELECT id, embedding <=> (SELECT vec FROM query_vector) AS dist
                 FROM test_recall
                 WHERE labels && ARRAY[1,2]::smallint[]
                 ORDER BY dist
                 LIMIT 10
             )
             SELECT array_agg(id::text) FROM cte;",
        );

        assert!(
            recall_no_filter >= 0.9,
            "Recall for no filter case is too low: {}",
            recall_no_filter
        );
        assert!(
            recall_single_label >= 0.9,
            "Recall for single label case is too low: {}",
            recall_single_label
        );
        assert!(
            recall_two_labels >= 0.9,
            "Recall for two label case is too low: {}",
            recall_two_labels
        );

        // Clean up
        cleanup_test_tables()?;

        Ok(())
    }
}
