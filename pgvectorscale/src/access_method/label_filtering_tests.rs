#[cfg(any(test, feature = "pg_test"))]
#[pgrx::pg_schema]
pub mod tests {
    use pgrx::prelude::*;
    use pgrx::spi;

    #[pg_test]
    pub unsafe fn test_labeled_filtering_with_category() -> spi::Result<()> {
        Spi::run(
            "CREATE TABLE test_labeled (
            id SERIAL PRIMARY KEY,
            embedding vector(3),
            labels INTEGER[],
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
        Spi::run(
            "CREATE TABLE test_labeled (
            id SERIAL PRIMARY KEY,
            embedding vector(3),
            labels INTEGER[],
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
            labels INTEGER[],
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
            "CREATE TABLE test(embedding vector(3), labels integer[]);

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
            "CREATE TABLE test_label_bounds(embedding vector(3), labels integer[]);

            CREATE INDEX idx_label_bounds
                  ON test_label_bounds
               USING diskann(embedding, labels);

            -- These inserts should succeed (labels within u16 bounds)
            INSERT INTO test_label_bounds(embedding, labels) VALUES ('[1,2,3]', '{0,65535}');
            ",
        )?;

        // Verify the valid labels were inserted correctly
        let res: Option<i64> = Spi::get_one("SET enable_seqscan = 0;
                WITH cte as (select * from test_label_bounds where labels && '{65535}' order by embedding <=> '[0,0,0]') 
                SELECT count(*) from cte;")?;
        assert_eq!(
            1,
            res.unwrap(),
            "Should find 1 document with label 65535 (max u16 value)"
        );

        // Test inserting a label that exceeds u16 bounds - this should fail
        // Use PL/pgSQL's exception handling to catch the error
        let error_captured: Option<bool> = Spi::get_one(
            "DO $$
            BEGIN
                -- This should fail because 65536 is outside u16 bounds
                INSERT INTO test_label_bounds(embedding, labels) VALUES ('[4,5,6]', '{65536}');
                -- If we get here, no error occurred
                RAISE NOTICE 'Test failed: expected error for out-of-bounds label but none occurred';
                PERFORM pg_catalog.set_config('pgrx.tests.failed', 'true', false);
            EXCEPTION WHEN OTHERS THEN
                -- We expect an error, so this is good
                RAISE NOTICE 'Got expected error: %', SQLERRM;
                -- Check if the error message contains what we expect
                IF SQLERRM LIKE '%out of bounds%' OR SQLERRM LIKE '%65536%' THEN
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
            "Test for out-of-bounds label (65536) failed"
        );

        // Test with negative label - this should also fail
        let error_captured: Option<bool> = Spi::get_one(
            "DO $$
            BEGIN
                -- This should fail because -1 is outside u16 bounds
                INSERT INTO test_label_bounds(embedding, labels) VALUES ('[7,8,9]', '{-1}');
                -- If we get here, no error occurred
                RAISE NOTICE 'Test failed: expected error for negative label but none occurred';
                PERFORM pg_catalog.set_config('pgrx.tests.failed', 'true', false);
            EXCEPTION WHEN OTHERS THEN
                -- We expect an error, so this is good
                RAISE NOTICE 'Got expected error: %', SQLERRM;
                -- Check if the error message contains what we expect
                IF SQLERRM LIKE '%out of bounds%' OR SQLERRM LIKE '%-1%' THEN
                    PERFORM pg_catalog.set_config('pgrx.tests.failed', 'false', false);
                ELSE
                    RAISE NOTICE 'Error message did not contain expected text: %', SQLERRM;
                    PERFORM pg_catalog.set_config('pgrx.tests.failed', 'true', false);
                END IF;
            END;
            $$ LANGUAGE plpgsql;
            
            -- Return whether the test failed
            SELECT current_setting('pgrx.tests.failed', true) = 'true';",
        )?;

        assert!(
            !error_captured.unwrap_or(true),
            "Test for negative label (-1) failed"
        );

        // Clean up
        Spi::run("DROP TABLE test_label_bounds;")?;

        Ok(())
    }
}
