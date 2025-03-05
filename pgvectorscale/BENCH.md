# ANN Benchmarker Tool

This command-line tool is designed to work with ANN benchmark datasets from [ann-benchmarks](https://github.com/erikbern/ann-benchmarks) and test them against a PostgreSQL database with pgvector or pgvectorscale.

## Features

- List available datasets from the ann-benchmarks repository
- Download datasets directly from the ann-benchmarks repository
- Load training vectors from HDF5 files into PostgreSQL
- Run test queries against the database and calculate recall using ground truth
- Support for splitting data loading into multiple transactions
- Parallel loading of data using multiple PostgreSQL connections
- Performance timing statistics for all operations
- Automatic inference of dimensions and distance metrics from dataset metadata

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

### Listing Available Datasets

To list all available datasets from the ann-benchmarks repository:

```bash
bench list-datasets
```

This will display a table of available datasets with their dimensions, sizes, and distance metrics.

### Downloading and Loading Data

To download a dataset directly from ann-benchmarks and load it into PostgreSQL:

```bash
bench download-and-load \
    --dataset glove-25-angular \
    --table my_vectors \
    --create-table \
    --create-index \
    --index-type diskann
```

Options:
- `--dataset`: Name of the dataset to download (use `list-datasets` to see available options)
- `-t, --table`: Table name to load vectors into
- `-c, --create-table`: Whether to create a new table
- `-i, --create-index`: Create an index after loading data
- `--index-type`: Type of index to create (diskann, hnsw, ivfflat) (default: diskann)

All other options from the `load` command are also available.

### Loading Data from Local File

To load training vectors from a local HDF5 file into PostgreSQL:

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
- `--transactions`: Number of transactions to split the load into (default: 1)
- `-p, --parallel`: Number of parallel connections to use (default: 1)
- `-i, --create-index`: Create an index after loading data
- `--index-type`: Type of index to create (diskann, hnsw, ivfflat) (default: diskann)
- `--distance-metric`: Distance metric to use (cosine, l2, inner_product) (default: cosine)

DiskANN Index Parameters:
- `--diskann-storage-layout`: Storage layout (memory_optimized or plain) (default: memory_optimized)
- `--diskann-num-neighbors`: Number of neighbors per node (default: 50)
- `--diskann-search-list-size`: Search list size for construction (default: 100)
- `--diskann-max-alpha`: Alpha parameter (default: 1.2)
- `--diskann-num-dimensions`: Number of dimensions to index (0 = all)
- `--diskann-num-bits-per-dimension`: Number of bits per dimension for SBQ

HNSW Index Parameters:
- `--hnsw-m`: Max number of connections per layer (default: 16)
- `--hnsw-ef-construction`: Size of dynamic candidate list for construction (default: 64)

IVFFlat Index Parameters:
- `--ivfflat-lists`: Number of lists (default depends on data size)

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
- `-g, --neighbors-dataset`: Dataset name for ground truth (default: "neighbors")
- `-t, --table`: Table name to query against
- `-n, --num-queries`: Number of queries to run (0 = all)
- `-k, --k`: Number of nearest neighbors to retrieve (default: 100)
- `--distance-metric`: Distance metric to use (cosine, l2, inner_product) (default: cosine)
- `-v, --verbose`: Show detailed recall information for each query

DiskANN Query Parameters:
- `--diskann-query-search-list-size`: Number of additional candidates during graph search (default: 100)
- `--diskann-query-rescore`: Number of elements to rescore (default: 50, 0 to disable)

HNSW Query Parameters:
- `--hnsw-ef-search`: Size of dynamic candidate list for search (default: 40)

IVFFlat Query Parameters:
- `--ivfflat-probes`: Number of lists to probe (default: 1)

## Example Workflows

### Using the Automated Download and Load

1. List available datasets:
   ```bash
   bench list-datasets
   ```

2. Download and load a dataset with automatic dimension and distance metric inference:
   ```bash
   bench download-and-load \
       --dataset glove-100-angular \
       --table glove_vectors \
       --create-table \
       --create-index \
       --index-type diskann \
       --diskann-num-neighbors 64 \
       --diskann-search-list-size 128
   ```

### Using a Local HDF5 File

1. Download a dataset from ann-benchmarks:
   ```bash
   wget http://ann-benchmarks.com/glove-100-angular.hdf5
   ```

2. Load the training vectors into PostgreSQL with a DiskANN index:
   ```bash
   bench load \
       --file glove-100-angular.hdf5 \
       --table glove_vectors \
       --create-table \
       --create-index \
       --index-type diskann \
       --distance-metric cosine \
       --diskann-num-neighbors 64 \
       --diskann-search-list-size 128
   ```

3. Run test queries and calculate recall with custom query parameters:
   ```bash
   bench test \
       --file glove-100-angular.hdf5 \
       --table glove_vectors \
       --k 10 \
       --distance-metric cosine \
       --diskann-query-search-list-size 150 \
       --diskann-query-rescore 100 \
       --verbose
   ```

4. Try with HNSW index instead:
   ```bash
   # Create table with HNSW index
   bench load \
       --file glove-100-angular.hdf5 \
       --table glove_hnsw \
       --create-table \
       --create-index \
       --index-type hnsw \
       --hnsw-m 16 \
       --hnsw-ef-construction 100
   
   # Run queries with custom ef_search
   bench test \
       --file glove-100-angular.hdf5 \
       --table glove_hnsw \
       --hnsw-ef-search 80
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
