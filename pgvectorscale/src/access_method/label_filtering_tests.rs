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
        SELECT COUNT(*) FROM test_labeled 
        WHERE labels && '{1}'
        ORDER BY embedding <=> '[0,0,0]';",
    )?;
    assert_eq!(2, res.unwrap(), "Should find 2 documents with label 1");

    // Test 2: Combining label filtering with category
    let res: Option<i64> = Spi::get_one(
        "
        SET enable_seqscan = 0;
        SELECT COUNT(*) FROM test_labeled 
        WHERE labels && '{2}' AND category = 'article'
        ORDER BY embedding <=> '[0,0,0]';",
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
        SELECT COUNT(*) FROM test_labeled 
        WHERE labels && (
            SELECT array_agg(id) FROM label_definitions WHERE name = 'science'
        )
        ORDER BY embedding <=> '[0,0,0]';",
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
        SELECT COUNT(*) FROM test_labeled 
        WHERE labels && '{1,4}'
        ORDER BY embedding <=> '[0,0,0]';",
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
    assert_eq!(2, res.unwrap(), "Should find 2 articles with label 2 OR 3");

    // Test 3: Combining label filtering with array length check
    let res: Option<i64> = Spi::get_one(
        "
        SET enable_seqscan = 0;
        SELECT COUNT(*) FROM test_labeled 
        WHERE labels && '{1}' AND array_length(labels, 1) > 2
        ORDER BY embedding <=> '[0,0,0]';",
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
