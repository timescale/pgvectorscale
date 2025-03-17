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
        Spi::run("DROP TABLE IF EXISTS label_definitions;")?;
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
                SELECT array_agg(id) FROM label_definitions WHERE name = 'science'
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
}
