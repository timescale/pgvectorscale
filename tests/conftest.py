"""
Pytest configuration and fixtures for pgvectorscale tests.

This module provides database connections, test setup/teardown,
and common utilities for all test modules.
"""
import pytest
import os
import psycopg2
import psycopg2.extensions
from contextlib import contextmanager


@pytest.fixture(scope="session")
def db_connection_params():
    """Database connection parameters."""
    return {
        'host': os.environ.get('DB_HOST', 'localhost'),
        'port': int(os.environ.get('DB_PORT', 5432)),
        'user': os.environ.get('DB_USER', os.environ.get('USER', 'postgres')),
        'database': os.environ.get('DB_NAME', 'postgres'),
        'password': os.environ.get('DB_PASSWORD', '')
    }


@pytest.fixture(scope="session")
def db_setup(db_connection_params):
    """Setup database extensions for the test session."""
    conn = psycopg2.connect(**db_connection_params)
    conn.autocommit = True
    
    try:
        with conn.cursor() as cur:
            # Install extensions
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute("CREATE EXTENSION IF NOT EXISTS vectorscale")
        
        yield db_connection_params
        
    except Exception as e:
        print(f"Warning: Could not install extensions: {e}")
        print("Please ensure pgvector and pgvectorscale extensions are available")
        yield db_connection_params
    finally:
        conn.close()


@pytest.fixture
def db_conn(db_setup):
    """Database connection for individual tests."""
    conn = psycopg2.connect(**db_setup)
    conn.autocommit = True
    yield conn
    conn.close()


@pytest.fixture
def clean_db(db_conn):
    """Fixture that provides a clean database state for each test."""
    # Clean up any existing test tables before the test
    with db_conn.cursor() as cur:
        cur.execute("""
            DROP TABLE IF EXISTS test_embeddings CASCADE;
            DROP TABLE IF EXISTS test_concurrent CASCADE;
            DROP TABLE IF EXISTS embeddings CASCADE;
            DROP TABLE IF EXISTS test_cosine CASCADE;
            DROP TABLE IF EXISTS test_l2 CASCADE;
            DROP TABLE IF EXISTS test_ip CASCADE;
            DROP TABLE IF EXISTS documents CASCADE;
        """)
    
    yield
    
    # Clean up after the test
    with db_conn.cursor() as cur:
        cur.execute("""
            DROP TABLE IF EXISTS test_embeddings CASCADE;
            DROP TABLE IF EXISTS test_concurrent CASCADE;
            DROP TABLE IF EXISTS embeddings CASCADE;
            DROP TABLE IF EXISTS test_cosine CASCADE;
            DROP TABLE IF EXISTS test_l2 CASCADE;
            DROP TABLE IF EXISTS test_ip CASCADE;
            DROP TABLE IF EXISTS documents CASCADE;
        """)


def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "concurrency: marks tests that test concurrent operations"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests that test full integration scenarios"
    )