# ANN Benchmarker Tool

This command-line tool is designed to work with ANN benchmark datasets from [ann-benchmarks](https://github.com/erikbern/ann-benchmarks) and test them against a PostgreSQL database with pgvector or pgvectorscale.

## Features

- Load training vectors from HDF5 files into PostgreSQL
- Run test queries against the database and calculate recall using ground truth
- Support for splitting data loading into multiple transactions
- Parallel loading of data using multiple PostgreSQL connections
- Performance timing statistics for all operations

## Prerequisites

- Rust and Cargo
- PostgreSQL with pgvector extension installed

No external HDF5 library installation is required as this tool uses a pure Rust implementation for reading HDF5 files.

## Building

```bash
cargo build --release --bin bench
```

The binary will be available at `target/release/bench`.

## Usage

### Loading Data

To load training vectors from an HDF5 file into PostgreSQL:

```bash
bench -c "host=localhost user=postgres" load \
    --file path/to/dataset.hdf5 \
    --dataset train \
    --table my_vectors \
    --create-table \
    --num-vectors 10000 \
    --transactions 10
```

Options:
- `-c, --connection-string`: PostgreSQL connection string (default: "host=localhost user=postgres")
- `-f, --file`: Path to HDF5 file
- `-d, --dataset`: Dataset name within the HDF5 file (default: "train")
- `-t, --table`: Table name to load vectors into
- `-c, --create-table`: Whether to create a new table
- `-n, --num-vectors`: Number of vectors to load (0 = all)
- `-t, --transactions`: Number of transactions to split the load into (default: 1)
- `-p, --parallel`: Number of parallel connections to use (default: 1)

### Running Tests

To run test queries and calculate recall:

```bash
bench -c "host=localhost user=postgres" test \
    --file path/to/dataset.hdf5 \
    --query-dataset test \
    --neighbors-dataset neighbors \
    --table my_vectors \
    --num-queries 100 \
    --k 10
```

Options:
- `-c, --connection-string`: PostgreSQL connection string (default: "host=localhost user=postgres")
- `-f, --file`: Path to HDF5 file
- `-q, --query-dataset`: Dataset name for test queries (default: "test")
- `-n, --neighbors-dataset`: Dataset name for ground truth (default: "neighbors")
- `-t, --table`: Table name to query against
- `-n, --num-queries`: Number of queries to run (0 = all)
- `-k, --k`: Number of nearest neighbors to retrieve (default: 10)

## Example Workflow

1. Download a dataset from ann-benchmarks:
   ```bash
   wget http://ann-benchmarks.com/glove-100-angular.hdf5
   ```

2. Load the training vectors into PostgreSQL:
   ```bash
   bench load \
       --file glove-100-angular.hdf5 \
       --table glove_vectors \
       --create-table
   ```

3. Run test queries and calculate recall:
   ```bash
   bench test \
       --file glove-100-angular.hdf5 \
       --table glove_vectors \
       --k 10
   ```

## Performance Statistics

The tool will output performance statistics for both loading and testing operations, including:
- Duration of the operation
- Number of items processed
- Items processed per second

## Notes on HDF5 Datasets

ANN benchmark datasets typically contain the following datasets:
- `train`: Vectors to be indexed
- `test`: Query vectors
- `neighbors`: Ground truth results for each query vector
- `distances`: Distances to ground truth results

This tool primarily works with the `train`, `test`, and `neighbors` datasets.
