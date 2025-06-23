# Testing Guide for pgvectorscale

This document describes the testing infrastructure and practices for the pgvectorscale PostgreSQL extension.

## Overview

pgvectorscale has two main types of tests:

1. **Rust Unit and Integration Tests** - Using PGRX's `#[pg_test]` framework
2. **Python Integration and Concurrency Tests** - Using pytest for multi-process scenarios

## Rust Tests (PGRX)

### Running Rust Tests

```bash
# Run all Rust tests
cd pgvectorscale && cargo pgrx test pg17

# Run specific test
cd pgvectorscale && cargo pgrx test pg17 test_name

# Run tests for different PostgreSQL versions
cd pgvectorscale && cargo pgrx test pg15
cd pgvectorscale && cargo pgrx test pg16
cd pgvectorscale && cargo pgrx test pg17
```

### Test Structure

Rust tests are located within the source code using the `#[pg_test]` macro:
- `src/access_method/build.rs` - Index building and functionality tests
- `src/access_method/sbq/tests.rs` - SBQ compression tests
- `src/access_method/plain/tests.rs` - Plain storage tests
- Various other `tests.rs` files throughout the codebase

## Python Tests

### Overview

Python tests provide comprehensive integration testing and multi-process concurrency testing that cannot be achieved with PGRX's single-process test framework.

### Directory Structure

```
tests/
├── __init__.py                    # Package initialization
├── conftest.py                    # pytest configuration and shared fixtures
├── requirements.txt               # Python dependencies
├── test_concurrent_inserts.py     # Concurrency and race condition tests (GitHub issue #193)
└── test_basic_operations.py       # Basic functionality and integration tests
```

### Quick Start

#### Prerequisites

1. **PostgreSQL with pgvector and pgvectorscale installed**:
   ```bash
   # For PGRX development
   cd pgvectorscale && cargo pgrx start pg17
   cargo pgrx install --features pg17
   ```

2. **Python dependencies**:
   ```bash
   pip install -r tests/requirements.txt
   ```

#### Running Tests

**Simple execution:**
```bash
# Install Python test dependencies
make test-python-setup

# Run all Python tests
pytest tests/ -v

# Run only concurrency tests
pytest tests/ -m concurrency -v

# Run only integration tests  
pytest tests/ -m integration -v

# Run specific test
pytest tests/test_concurrent_inserts.py::test_concurrent_insert_race_condition -v
```

**Using Makefile targets:**
```bash
make test-python              # Run all Python tests
make test-all                 # Run Rust + Python tests
```

**Using the test runner script:**
```bash
# Basic usage
./scripts/run-python-tests.sh

# With custom database (e.g., PGRX test database)
DB_PORT=28817 ./scripts/run-python-tests.sh

# With custom pytest arguments
PYTEST_ARGS="-v -k concurrent" ./scripts/run-python-tests.sh
```

### Configuration

#### Database Connection

Tests use the `DATABASE_URL` environment variable:

```bash
# Default (assumes local PostgreSQL)
DATABASE_URL="postgresql+asyncpg://postgres:postgres@localhost:5432/test_db"

# For PGRX development
DATABASE_URL="postgresql+asyncpg://$(whoami)@localhost:28817/postgres"

# Custom configuration
DATABASE_URL="postgresql+asyncpg://user:pass@host:port/database"
```

#### Test Markers

Tests use pytest markers for categorization:

- `@pytest.mark.concurrency` - Concurrency tests
- `@pytest.mark.integration` - Integration tests  
- `@pytest.mark.slow` - Slow-running tests

```bash
# Run only concurrency tests
pytest -m concurrency -v

# Skip slow tests
pytest -m "not slow" -v

# Run multiple marker types
pytest -m "concurrency or integration" -v
```

### Test Categories

#### Concurrency Tests (`test_concurrent_inserts.py`)

These tests verify that the extension handles concurrent operations correctly, particularly focusing on race conditions and multi-process scenarios.

**Key Tests:**
- `test_concurrent_insert_race_condition` - Tests GitHub issue #193 fix
- `test_high_concurrency_inserts` - Stress testing with higher concurrency
- `test_concurrent_mixed_operations` - Mixed insert/query workloads

**Note:** These tests require multiple database connections and cannot be replicated in PGRX's single-process test environment.

#### Integration Tests (`test_basic_operations.py`)

These tests verify basic functionality and integration between components.

**Key Tests:**
- `test_extension_installation` - Verifies extension installation
- `test_diskann_index_creation` - Tests index creation
- `test_vector_insert_and_query` - Basic CRUD operations
- `test_different_distance_metrics` - Tests cosine, L2, and inner product
- `test_index_options` - Tests index configuration options

### Development Workflow

#### Daily Development

```bash
# 1. Start PGRX environment (one-time)
cd pgvectorscale && cargo pgrx start pg17

# 2. Install/update extension
cargo pgrx install --features pg17

# 3. Run tests
make test-python

# Or with PGRX database directly
DATABASE_URL="postgresql+asyncpg://$(whoami)@localhost:28817/postgres" pytest tests/ -v
```

#### Testing Against Different PostgreSQL Versions

```bash
# Test against different PostgreSQL versions (if multiple PGRX instances)
DB_PORT=28815 pytest tests/  # pg15
DB_PORT=28816 pytest tests/  # pg16  
DB_PORT=28817 pytest tests/  # pg17
```

#### IDE Integration

**VSCode:** Add to `.vscode/settings.json`:
```json
{
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"],
    "python.testing.cwd": "${workspaceFolder}"
}
```

**Environment file (`.env`):**
```bash
DATABASE_URL=postgresql+asyncpg://$(whoami)@localhost:28817/postgres
```

### Continuous Integration

Python tests run automatically in GitHub Actions for:
- PostgreSQL versions: 15, 16, 17
- Python versions: 3.9, 3.11
- Multiple platforms: amd64, arm64

The workflow:
1. Sets up PostgreSQL with pgvector and pgvectorscale
2. Installs Python dependencies
3. Runs the full test suite
4. Uploads test results as artifacts

### Troubleshooting

#### Common Issues

**Extension not found:**
```bash
# Ensure extensions are installed
psql -c "CREATE EXTENSION IF NOT EXISTS vector;"
psql -c "CREATE EXTENSION IF NOT EXISTS vectorscale;"
```

**Connection errors:**
```bash
# Check PostgreSQL is running
pg_isready -h localhost -p 5432

# For PGRX development
pg_isready -h localhost -p 28817
```

**Dependency issues:**
```bash
# Reinstall Python dependencies
pip install -r tests/requirements.txt --force-reinstall
```

#### Debugging Tests

```bash
# Run with verbose output and stop on first failure
pytest tests/ -v -s -x

# Run with detailed traceback
pytest tests/ --tb=long

# Run specific test with debugging
pytest tests/test_concurrent_inserts.py::test_concurrent_insert_race_condition -v -s
```

#### Docker-based Testing

For isolated testing environments:

```bash
# Start PostgreSQL in Docker
docker run --rm -d \
    --name pgvectorscale-test \
    -e POSTGRES_PASSWORD=postgres \
    -e POSTGRES_DB=test_db \
    -p 5432:5432 \
    postgres:16-alpine

# Run tests against Docker instance
DATABASE_URL="postgresql+asyncpg://postgres:postgres@localhost:5432/test_db" \
    pytest tests/ -v

# Cleanup
docker stop pgvectorscale-test
```

### Writing New Tests

#### Adding Concurrency Tests

Add new concurrency tests to `tests/test_concurrent_inserts.py` or create new test files:

```python
import pytest
import asyncio
from sqlalchemy import text

@pytest.mark.asyncio
@pytest.mark.concurrency
async def test_your_concurrency_scenario(db_engine, clean_db):
    """Test description."""
    # Setup
    async with db_engine.begin() as conn:
        await conn.execute(text("CREATE TABLE ..."))
    
    # Test concurrent operations
    async def worker():
        # Your concurrent operation
        pass
    
    tasks = [worker() for _ in range(parallelism)]
    await asyncio.gather(*tasks)
    
    # Assertions
    assert ...
```

#### Adding Integration Tests

Add new integration tests to `tests/test_basic_operations.py` or create new test files:

```python
import pytest
from sqlalchemy import text

@pytest.mark.asyncio
@pytest.mark.integration
async def test_your_integration_scenario(db_engine, clean_db):
    """Test description."""
    async with db_engine.begin() as conn:
        # Your test logic
        result = await conn.execute(text("SELECT ..."))
        assert result.scalar() == expected_value
```

### Performance and Load Testing

For performance testing, use the `@pytest.mark.slow` marker:

```python
@pytest.mark.slow
@pytest.mark.asyncio
async def test_performance_scenario(db_engine, clean_db):
    """Performance test with large datasets."""
    # Test with larger datasets, higher concurrency, etc.
```

Run performance tests separately:
```bash
pytest -m slow -v
```

## Best Practices

### Test Organization

1. **Separate concerns**: Use different test files and markers for different test types
2. **Use descriptive names**: Test names should clearly indicate what they test
3. **Use appropriate markers**: Mark tests by type and characteristics
4. **Clean up**: Use fixtures for setup/teardown

### Database Testing

1. **Use transactions**: Tests should be isolated and not affect each other
2. **Use clean fixtures**: Start each test with a clean database state
3. **Test realistic scenarios**: Use realistic data sizes and access patterns
4. **Verify cleanup**: Ensure tests clean up after themselves

### Concurrency Testing

1. **Test real scenarios**: Use realistic concurrency levels
2. **Use proper synchronization**: Use asyncio properly for concurrent operations
3. **Verify race conditions**: Tests should fail without the fix
4. **Test different patterns**: Mix different types of operations

## Related Documentation

- [CLAUDE.md](CLAUDE.md) - Development commands and architecture
- [DEVELOPMENT.md](DEVELOPMENT.md) - Development setup and guidelines  
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines