"""
Test for concurrent insert safety (GitHub issue #193).

This test reproduces the race condition that can occur when multiple
processes insert vectors concurrently into a diskann index.

The race condition occurs when multiple PostgreSQL backend processes
access the same page simultaneously during index operations, potentially
leading to page corruption and assertion failures.
"""
import pytest
import threading
import time
import numpy as np
import psycopg2
from concurrent.futures import ThreadPoolExecutor, as_completed


@pytest.mark.concurrency
def test_concurrent_insert_race_condition(db_setup, clean_db):
    """
    Test concurrent inserts don't cause page corruption.
    
    This test reproduces GitHub issue #193 by performing concurrent
    vector insertions that can trigger race conditions in the diskann
    index implementation.
    """
    # Test parameters - using smaller values for faster testing
    # but enough to potentially trigger race conditions
    batch_size = 50
    batches = 20
    parallelism = 4
    dimensions = 3
    
    # Setup test table with main connection
    setup_conn = psycopg2.connect(**db_setup)
    setup_conn.autocommit = True
    
    try:
        with setup_conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE embeddings (
                    id BIGSERIAL PRIMARY KEY,
                    embedding vector({dimensions})
                )
            """)
            cur.execute("""
                CREATE INDEX embeddings_embedding_diskann
                ON embeddings USING diskann (embedding vector_cosine_ops)
            """)
    finally:
        setup_conn.close()
    
    def insert_batch():
        """Insert a batch of random vectors using its own connection."""
        conn = psycopg2.connect(**db_setup)
        conn.autocommit = True
        
        try:
            with conn.cursor() as cur:
                # Generate random vectors
                vectors = []
                for _ in range(batch_size):
                    vector = np.random.rand(dimensions)
                    vectors.append(f"[{','.join(map(str, vector))}]")
                
                # Insert vectors
                cur.executemany(
                    "INSERT INTO embeddings (embedding) VALUES (%s)",
                    [(vector,) for vector in vectors]
                )
        finally:
            conn.close()
    
    # Run concurrent inserts using thread pool
    with ThreadPoolExecutor(max_workers=parallelism) as executor:
        # Submit all tasks
        futures = []
        for _ in range(batches):
            for _ in range(parallelism):
                future = executor.submit(insert_batch)
                futures.append(future)
        
        # Wait for all tasks to complete and collect any exceptions
        exceptions = []
        for future in as_completed(futures):
            try:
                future.result()  # This will raise any exception that occurred
            except Exception as e:
                exceptions.append(e)
        
        # If any exceptions occurred, the test should fail
        if exceptions:
            pytest.fail(f"Concurrent inserts failed with errors: {exceptions}")
    
    # Verify final count
    verify_conn = psycopg2.connect(**db_setup)
    verify_conn.autocommit = True
    
    try:
        with verify_conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM embeddings")
            count = cur.fetchone()[0]
            expected_count = batch_size * batches * parallelism
            assert count == expected_count, f"Expected {expected_count} rows, got {count}"
    finally:
        verify_conn.close()


@pytest.mark.concurrency
@pytest.mark.slow
def test_high_concurrency_inserts(db_setup, clean_db):
    """
    Test higher concurrency levels for stress testing.
    
    This test uses higher concurrency to increase the likelihood
    of triggering race conditions.
    """
    # Higher stress test parameters
    batch_size = 100
    batches = 50
    parallelism = 8
    dimensions = 3
    
    # Setup test table
    setup_conn = psycopg2.connect(**db_setup)
    setup_conn.autocommit = True
    
    try:
        with setup_conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE embeddings (
                    id BIGSERIAL PRIMARY KEY,
                    embedding vector({dimensions})
                )
            """)
            cur.execute("""
                CREATE INDEX embeddings_embedding_diskann
                ON embeddings USING diskann (embedding vector_cosine_ops)
            """)
    finally:
        setup_conn.close()
    
    def worker():
        """Worker that performs multiple insert batches."""
        conn = psycopg2.connect(**db_setup)
        conn.autocommit = True
        
        try:
            for _ in range(batches):
                with conn.cursor() as cur:
                    # Generate and insert vectors
                    vectors = []
                    for _ in range(batch_size):
                        vector = np.random.rand(dimensions)
                        vectors.append(f"[{','.join(map(str, vector))}]")
                    
                    cur.executemany(
                        "INSERT INTO embeddings (embedding) VALUES (%s)",
                        [(vector,) for vector in vectors]
                    )
        finally:
            conn.close()
    
    # Run concurrent workers
    with ThreadPoolExecutor(max_workers=parallelism) as executor:
        futures = [executor.submit(worker) for _ in range(parallelism)]
        
        # Wait for completion and handle exceptions
        exceptions = []
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                exceptions.append(e)
        
        if exceptions:
            pytest.fail(f"High concurrency test failed with errors: {exceptions}")
    
    # Verify final count
    verify_conn = psycopg2.connect(**db_setup)
    verify_conn.autocommit = True
    
    try:
        with verify_conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM embeddings")
            count = cur.fetchone()[0]
            expected_count = batch_size * batches * parallelism
            assert count == expected_count, f"Expected {expected_count} rows, got {count}"
    finally:
        verify_conn.close()


@pytest.mark.concurrency  
def test_concurrent_mixed_operations(db_setup, clean_db):
    """
    Test concurrent inserts and queries to test mixed workload race conditions.
    """
    dimensions = 3
    insert_batches = 10
    batch_size = 20
    query_count = 50
    
    # Setup test table with some initial data
    setup_conn = psycopg2.connect(**db_setup)
    setup_conn.autocommit = True
    
    try:
        with setup_conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE embeddings (
                    id BIGSERIAL PRIMARY KEY,
                    embedding vector({dimensions})
                )
            """)
            
            # Insert some initial vectors
            initial_vectors = []
            for i in range(100):
                vector = np.random.rand(dimensions)
                initial_vectors.append(f"[{','.join(map(str, vector))}]")
            
            cur.executemany(
                "INSERT INTO embeddings (embedding) VALUES (%s)",
                [(vector,) for vector in initial_vectors]
            )
            
            # Create index after initial data
            cur.execute("""
                CREATE INDEX embeddings_embedding_diskann
                ON embeddings USING diskann (embedding vector_cosine_ops)
            """)
    finally:
        setup_conn.close()
    
    def insert_worker():
        """Worker that inserts vectors."""
        conn = psycopg2.connect(**db_setup)
        conn.autocommit = True
        
        try:
            for _ in range(insert_batches):
                with conn.cursor() as cur:
                    vectors = []
                    for _ in range(batch_size):
                        vector = np.random.rand(dimensions)
                        vectors.append(f"[{','.join(map(str, vector))}]")
                    
                    cur.executemany(
                        "INSERT INTO embeddings (embedding) VALUES (%s)",
                        [(vector,) for vector in vectors]
                    )
        finally:
            conn.close()
    
    def query_worker():
        """Worker that performs similarity queries."""
        conn = psycopg2.connect(**db_setup)
        conn.autocommit = True
        
        try:
            for _ in range(query_count):
                with conn.cursor() as cur:
                    # Random query vector
                    query_vector = np.random.rand(dimensions)
                    query_str = f"[{','.join(map(str, query_vector))}]"
                    
                    # Perform similarity search
                    cur.execute("""
                        SELECT id, embedding <=> %s::vector as distance
                        FROM embeddings 
                        ORDER BY embedding <=> %s::vector
                        LIMIT 5
                    """, (query_str, query_str))
                    
                    rows = cur.fetchall()
                    assert len(rows) > 0, "Query should return results"
        finally:
            conn.close()
    
    # Run concurrent insert and query workers
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        
        # Submit insert workers
        for _ in range(2):
            futures.append(executor.submit(insert_worker))
        
        # Submit query workers
        for _ in range(3):
            futures.append(executor.submit(query_worker))
        
        # Wait for completion and handle exceptions
        exceptions = []
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                exceptions.append(e)
        
        if exceptions:
            pytest.fail(f"Mixed operations test failed with errors: {exceptions}")