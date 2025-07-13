# Testing Guide for pgvectorscale

pgvectorscale has two main types of tests:

1. **Rust Tests** - Using PGRX's `#[pg_test]` framework (can be in any source file)
2. **Python Tests** - Using pytest for multi-process concurrency testing

## Rust Tests

```bash
# Run all Rust tests
cd pgvectorscale && cargo pgrx test pg16

# Run specific test
cd pgvectorscale && cargo pgrx test pg16 test_name
```

## Python Tests

```bash
# Setup (creates .venv virtual environment)
make test-python-setup

# Run all Python tests
make test-python

# Run specific categories
pytest tests/ -m concurrency -v    # Multi-process concurrency tests
pytest tests/ -m integration -v    # Basic integration tests

# For PGRX development (custom port)
DB_PORT=28817 ./scripts/run-python-tests.sh
```

### Test Markers

- `@pytest.mark.concurrency` - Multi-process concurrency tests
- `@pytest.mark.integration` - Basic integration tests

### Prerequisites

For PGRX development:
```bash
cd pgvectorscale && cargo pgrx start pg16
cargo pgrx install --features pg16
```
