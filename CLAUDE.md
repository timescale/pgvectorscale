# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Building and Installation
- **Build development version**: `cd pgvectorscale && cargo pgrx install --features pg17`
- **Build release version**: `cd pgvectorscale && cargo pgrx install --release --features pg17`
- **Package extension**: `cd pgvectorscale && cargo pgrx package --features pg17`
- **Initialize PGRX environment**: `cd pgvectorscale && cargo pgrx init --pg17 pg_config`

### Testing
- **Run Rust unit tests**: `cd pgvectorscale && cargo test`
- **Run PGRX integration tests**: `cd pgvectorscale && cargo pgrx test pg17`
- **Run specific test**: `cd pgvectorscale && cargo pgrx test pg17 test_name`
- **Run tests for specific PostgreSQL version**: `cd pgvectorscale && cargo pgrx test -- pg16` (or pg13, pg14, pg15, pg17)

### Code Quality
- **Format code**: `cd pgvectorscale && cargo fmt`
- **Check formatting**: `cd pgvectorscale && cargo fmt --check`  
- **Run linter**: `cd pgvectorscale && cargo clippy --all-targets --features pg17`
- **Format shell scripts**: `make shfmt`
- **Check shell scripts**: `make shellcheck`

### Makefile Commands
- **Format Rust code**: `make format`
- **Build debug**: `make build`
- **Install debug**: `make install-debug`
- **Install release**: `make install-release`

## Architecture Overview

pgvectorscale is a PostgreSQL extension written in Rust using the PGRX framework that provides high-performance vector indexing and search capabilities. It builds on pgvector with new index types and compression methods.

### Core Components

**Access Method Implementation** (`src/access_method/`):
- **StreamingDiskANN Index**: Main index algorithm based on Microsoft's DiskANN research
- **SBQ (Statistical Binary Quantization)**: Compression method for memory-efficient storage
- **Plain Storage**: Uncompressed vector storage option
- **Label-based Filtering**: Efficient filtered vector search using smallint arrays

**Key Modules**:
- `access_method/mod.rs`: Main access method registration and interface
- `access_method/build.rs`: Index building logic and construction algorithms
- `access_method/scan.rs`: Query execution and graph traversal during search
- `access_method/sbq/`: Statistical Binary Quantization implementation for compression
- `access_method/plain/`: Plain (uncompressed) storage implementation
- `access_method/labels/`: Label-based filtering system for efficient metadata filtering
- `access_method/distance/`: Optimized distance calculations with SIMD support

**Storage Architecture**:
- Uses PostgreSQL's access method API for integration
- Supports both compressed (SBQ) and uncompressed (plain) storage layouts
- Graph-based index structure stored across PostgreSQL pages
- Label arrays stored as smallint[] for efficient filtering

**Distance Support**:
- Cosine distance (`<=>`) with `vector_cosine_ops`
- L2 distance (`<->`) with `vector_l2_ops` 
- Inner product (`<#>`) with `vector_ip_ops`

### Build Configuration

The project uses a workspace structure with the main extension in `pgvectorscale/` and derives in `pgvectorscale_derive/`. PostgreSQL version support is controlled via Cargo features (pg13-pg17). The default feature is pg17, but you can build for other versions using `--features pg16`, `--features pg15`, etc.

**Version Dependencies**:
- PGRX version: 0.12.9 (must match cargo-pgrx version)
- Supports PostgreSQL 13, 14, 15, 16, and 17
- Requires pgvector extension as a dependency

### Testing Strategy

Tests are primarily Rust unit tests within modules and PGRX integration tests. The extension includes benchmark suites for distance calculations and graph operations.

**CI/CD Process**:
- Code formatting is checked via `cargo fmt --check` in CI
- Clippy linting runs on all PostgreSQL versions (pg13-pg17)
- Full test suite runs on both AMD64 and ARM64 platforms
- Tests run against multiple PostgreSQL versions and pgvector 0.7.4

### Development Notes

**Important Limitations**:
- Building on macOS X86 (Intel) is currently not supported (use ARM Mac, Linux, or Docker)
- Index creation on UNLOGGED tables is not yet implemented
- The StreamingDiskANN index uses relaxed ordering (results may be slightly out of order by distance)