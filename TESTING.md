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

## Related Documentation

- [CLAUDE.md](CLAUDE.md) - Development commands and architecture
- [DEVELOPMENT.md](DEVELOPMENT.md) - Development setup and guidelines  
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines