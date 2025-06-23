"""
Basic integration tests for pgvectorscale functionality.

These tests verify that the extension is properly installed and
basic operations work correctly.
"""
import pytest
import numpy as np


@pytest.mark.integration
def test_extension_installation(db_conn):
    """Test that pgvectorscale extension is properly installed."""
    with db_conn.cursor() as cur:
        # Test that we can create the extension
        cur.execute("CREATE EXTENSION IF NOT EXISTS vectorscale")
        
        # Verify extension exists
        cur.execute("""
            SELECT extname, extversion 
            FROM pg_extension 
            WHERE extname = 'vectorscale'
        """)
        row = cur.fetchone()
        assert row is not None, "vectorscale extension should be installed"
        assert row[0] == 'vectorscale', "Extension name should be vectorscale"


@pytest.mark.integration  
def test_diskann_index_creation(db_conn, clean_db):
    """Test that diskann indexes can be created successfully."""
    dimensions = 5
    
    with db_conn.cursor() as cur:
        # Create table with vector column
        cur.execute(f"""
            CREATE TABLE test_embeddings (
                id SERIAL PRIMARY KEY,
                embedding vector({dimensions})
            )
        """)
        
        # Create diskann index
        cur.execute("""
            CREATE INDEX test_diskann_idx 
            ON test_embeddings 
            USING diskann (embedding vector_cosine_ops)
        """)
        
        # Verify index was created
        cur.execute("""
            SELECT indexname, indexdef
            FROM pg_indexes 
            WHERE indexname = 'test_diskann_idx'
        """)
        row = cur.fetchone()
        assert row is not None, "diskann index should be created"
        assert 'diskann' in row[1], "Index definition should contain 'diskann'"


@pytest.mark.integration
def test_vector_insert_and_query(db_conn, clean_db):
    """Test basic vector insertion and similarity queries."""
    dimensions = 4
    num_vectors = 50
    
    with db_conn.cursor() as cur:
        # Create table and index
        cur.execute(f"""
            CREATE TABLE test_embeddings (
                id SERIAL PRIMARY KEY,
                embedding vector({dimensions})
            )
        """)
        
        cur.execute("""
            CREATE INDEX test_diskann_idx 
            ON test_embeddings 
            USING diskann (embedding vector_cosine_ops)
        """)
        
        # Insert test vectors
        vectors = []
        for i in range(num_vectors):
            vector = np.random.rand(dimensions)
            vector_str = f"[{','.join(map(str, vector))}]"
            vectors.append((i, vector_str))
        
        cur.executemany(
            "INSERT INTO test_embeddings (id, embedding) VALUES (%s, %s)",
            vectors
        )
        
        # Test similarity query
        query_vector = np.random.rand(dimensions)
        query_str = f"[{','.join(map(str, query_vector))}]"
        
        cur.execute(f"""
            SELECT id, embedding <=> %s::vector as distance
            FROM test_embeddings
            ORDER BY embedding <=> %s::vector
            LIMIT 5
        """, (query_str, query_str))
        
        rows = cur.fetchall()
        assert len(rows) == 5, "Should return 5 nearest neighbors"
        
        # Verify that we got valid distances (should be non-negative for cosine distance)
        distances = [row[1] for row in rows]
        assert all(d >= 0 for d in distances), "All cosine distances should be non-negative"
        assert all(d <= 2 for d in distances), "All cosine distances should be <= 2"
        
        # Note: pgvectorscale uses "relaxed ordering" so distances may not be 
        # in perfectly ascending order, but should be approximately sorted


@pytest.mark.integration
def test_different_distance_metrics(db_conn, clean_db):
    """Test different distance metrics (cosine, L2, inner product)."""
    dimensions = 3
    
    with db_conn.cursor() as cur:
        # Test cosine distance
        cur.execute(f"""
            CREATE TABLE test_cosine (
                id SERIAL PRIMARY KEY,
                embedding vector({dimensions})
            );
            
            CREATE INDEX test_cosine_idx 
            ON test_cosine 
            USING diskann (embedding vector_cosine_ops);
        """)
        
        # Test L2 distance  
        cur.execute(f"""
            CREATE TABLE test_l2 (
                id SERIAL PRIMARY KEY,
                embedding vector({dimensions})
            );
            
            CREATE INDEX test_l2_idx 
            ON test_l2 
            USING diskann (embedding vector_l2_ops);
        """)
        
        # Test inner product
        cur.execute(f"""
            CREATE TABLE test_ip (
                id SERIAL PRIMARY KEY,
                embedding vector({dimensions})
            );
            
            CREATE INDEX test_ip_idx 
            ON test_ip 
            USING diskann (embedding vector_ip_ops);
        """)
        
        # Insert test data into all tables
        test_vector = "[1.0, 2.0, 3.0]"
        for table in ['test_cosine', 'test_l2', 'test_ip']:
            cur.execute(f"""
                INSERT INTO {table} (embedding) VALUES (%s)
            """, (test_vector,))
        
        # Test each distance metric
        query_vector = "[2.0, 3.0, 1.0]"
        
        # Cosine distance
        cur.execute(f"""
            SELECT embedding <=> %s::vector as distance
            FROM test_cosine
            LIMIT 1
        """, (query_vector,))
        cosine_dist = cur.fetchone()[0]
        assert cosine_dist is not None, "Cosine distance should be calculated"
        
        # L2 distance
        cur.execute(f"""
            SELECT embedding <-> %s::vector as distance
            FROM test_l2
            LIMIT 1
        """, (query_vector,))
        l2_dist = cur.fetchone()[0]
        assert l2_dist is not None, "L2 distance should be calculated"
        
        # Inner product
        cur.execute(f"""
            SELECT embedding <#> %s::vector as distance
            FROM test_ip
            LIMIT 1
        """, (query_vector,))
        ip_dist = cur.fetchone()[0]
        assert ip_dist is not None, "Inner product distance should be calculated"


@pytest.mark.integration
def test_index_options(db_conn, clean_db):
    """Test diskann index with different configuration options."""
    dimensions = 4
    
    with db_conn.cursor() as cur:
        # Create table
        cur.execute(f"""
            CREATE TABLE test_embeddings (
                id SERIAL PRIMARY KEY,
                embedding vector({dimensions})
            )
        """)
        
        # Test index with custom options
        cur.execute("""
            CREATE INDEX test_diskann_custom_idx 
            ON test_embeddings 
            USING diskann (embedding vector_cosine_ops)
            WITH (num_neighbors = 20, search_list_size = 40)
        """)
        
        # Insert some test data
        vectors = []
        for i in range(30):
            vector = np.random.rand(dimensions)
            vector_str = f"[{','.join(map(str, vector))}]"
            vectors.append((i, vector_str))
        
        cur.executemany(
            "INSERT INTO test_embeddings (id, embedding) VALUES (%s, %s)",
            vectors
        )
        
        # Test that queries work with custom index
        query_vector = np.random.rand(dimensions)
        query_str = f"[{','.join(map(str, query_vector))}]"
        
        # Test that we can perform similarity search with the custom index
        cur.execute("""
            SELECT id, embedding <=> %s::vector as distance
            FROM test_embeddings
            ORDER BY embedding <=> %s::vector
            LIMIT 10
        """, (query_str, query_str))
        
        results = cur.fetchall()
        assert len(results) == 10, "Should return 10 nearest neighbors"
        
        # Also test that we can count all vectors
        cur.execute("SELECT COUNT(*) FROM test_embeddings")
        count = cur.fetchone()[0]
        assert count == 30, "Should be able to query all vectors through index"