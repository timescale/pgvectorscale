use std::fmt;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// Import the DistanceType from the vectorscale crate
use vectorscale::access_method::distance::DistanceType;

use bytes::Bytes;
use clap::{Parser, Subcommand, ValueEnum};
use csv;
use futures::{future::join_all, SinkExt, StreamExt};
use hdf5_metno::File;
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::{s, Array, Array2, Ix2};
use serde::{Deserialize, Serialize};
use tokio::task;
use tokio_postgres::{Client, Error as PgError, NoTls};
use hdf5_metno::{Selection, Hyperslab};

/// Cache key for ground truth results
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
struct GroundTruthCacheKey {
    table_name: String,
    query_vector_hash: String,
    labels: Option<Vec<i16>>,
    top_k: usize,
}

/// Cache value for ground truth results
#[derive(Debug, Clone, Serialize, Deserialize)]
struct GroundTruthCacheValue {
    results: Vec<(i32, f64)>,
    timestamp: chrono::DateTime<chrono::Utc>,
}

/// Get the cache directory for ground truth results
fn get_ground_truth_cache_dir() -> Result<PathBuf, Box<dyn std::error::Error>> {
    // Get the user's home directory
    let home_dir = dirs::home_dir().ok_or_else(|| "Could not find home directory".to_string())?;

    // Create the cache directory path
    let cache_dir = home_dir.join(".pgvectorscale").join("ground_truth");

    // Create the directory if it doesn't exist
    if !cache_dir.exists() {
        std::fs::create_dir_all(&cache_dir)?;
    }

    Ok(cache_dir)
}

/// Get the cache file path for a given cache key
fn get_cache_file_path(key: &GroundTruthCacheKey) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let cache_dir = get_ground_truth_cache_dir()?;
    let key_bytes = bincode::serialize(key)?;
    let key_hash = blake3::hash(&key_bytes);
    Ok(cache_dir.join(format!("{}.bin", key_hash.to_hex())))
}

/// Load ground truth results from cache
fn load_from_cache(
    key: &GroundTruthCacheKey,
) -> Result<Option<Vec<(i32, f64)>>, Box<dyn std::error::Error>> {
    let cache_file = get_cache_file_path(key)?;
    if !cache_file.exists() {
        return Ok(None);
    }

    let file = std::fs::File::open(cache_file)?;
    let value: GroundTruthCacheValue = bincode::deserialize_from(file)?;

    // Check if the cache is older than 7 days
    let now = chrono::Utc::now();
    let cache_age = now - value.timestamp;
    if cache_age > chrono::Duration::days(7) {
        return Ok(None);
    }

    Ok(Some(value.results))
}

/// Save ground truth results to cache
fn save_to_cache(
    key: &GroundTruthCacheKey,
    results: &[(i32, f64)],
) -> Result<(), Box<dyn std::error::Error>> {
    let cache_file = get_cache_file_path(key)?;
    let value = GroundTruthCacheValue {
        results: results.to_vec(),
        timestamp: chrono::Utc::now(),
    };
    let file = std::fs::File::create(cache_file)?;
    bincode::serialize_into(file, &value)?;
    Ok(())
}

// Add this function near the top of the file, after the imports
fn kebab_to_snake_case(s: &str) -> String {
    s.replace('-', "_")
}

// Dataset information struct
#[derive(Debug, Clone, Serialize, Deserialize)]
struct DatasetInfo {
    name: String,
    dimensions: usize,
    train_size: usize,
    test_size: usize,
    neighbors: usize,
    distance: String,
    url: String,
}

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    /// PostgreSQL connection string
    #[arg(short, long, default_value = "host=localhost user=postgres")]
    connection_string: String,

    #[command(subcommand)]
    command: Commands,
}

// DistanceType now implements ValueEnum in access_method/distance.rs

#[derive(Clone, Copy, Debug)]
enum IndexType {
    /// DiskANN index (default)
    DiskANN,
    /// HNSW index
    Hnsw,
    /// HNSW index with BQ
    HnswBq,
    /// IVFFlat index
    IVFFlat,
}

impl ValueEnum for IndexType {
    fn value_variants<'a>() -> &'a [Self] {
        &[
            IndexType::DiskANN,
            IndexType::Hnsw,
            IndexType::HnswBq,
            IndexType::IVFFlat,
        ]
    }

    fn to_possible_value(&self) -> Option<clap::builder::PossibleValue> {
        Some(match self {
            IndexType::DiskANN => {
                clap::builder::PossibleValue::new("diskann").help("DiskANN index (default)")
            }
            IndexType::Hnsw => clap::builder::PossibleValue::new("hnsw").help("HNSW index"),
            IndexType::HnswBq => {
                clap::builder::PossibleValue::new("hnsw_bq").help("HNSW index with BQ")
            }
            IndexType::IVFFlat => {
                clap::builder::PossibleValue::new("ivfflat").help("IVFFlat index")
            }
        })
    }
}

impl fmt::Display for IndexType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IndexType::DiskANN => write!(f, "diskann"),
            IndexType::Hnsw => write!(f, "hnsw"),
            IndexType::HnswBq => write!(f, "hnsw_bq"),
            IndexType::IVFFlat => write!(f, "ivfflat"),
        }
    }
}

#[derive(Subcommand)]
enum Commands {
    /// List available datasets from ann-benchmarks
    ListDatasets,

    /// Load training vectors from HDF5 file into PostgreSQL
    Load {
        /// Path to HDF5 file (either --file or --dataset must be specified)
        #[arg(short, long, group = "input_source")]
        file: Option<PathBuf>,

        /// Dataset name from ann-benchmarks (either --file or --dataset must be specified)
        #[arg(short, long, group = "input_source", required_unless_present = "file")]
        dataset: Option<String>,

        /// Dataset name within the HDF5 file (usually 'train')
        #[arg(short = 'd', long, default_value = "train")]
        dataset_name: String,

        /// Table name to load vectors into (defaults to input name + index type)
        #[arg(short, long)]
        table: Option<String>,

        /// Whether to create a new table
        #[arg(short, long, default_value_t = false)]
        create_table: bool,

        /// Number of vectors to load (0 = all)
        #[arg(short, long, default_value_t = 0)]
        num_vectors: usize,

        /// Number of transactions to split the load into
        #[arg(long, default_value_t = 1)]
        transactions: usize,

        /// Number of parallel connections to use
        #[arg(short, long, default_value_t = 1)]
        parallel: usize,

        /// Create an index after loading data
        #[arg(short = 'i', long, default_value_t = false)]
        create_index: bool,

        /// Type of index to create
        #[arg(long, default_value_t = IndexType::DiskANN)]
        index_type: IndexType,

        /// Distance metric to use for the index
        #[arg(long, default_value_t = DistanceType::Cosine)]
        distance_metric: DistanceType,

        // DiskANN index hyperparameters
        /// DiskANN: Storage layout (memory_optimized or plain)
        #[arg(long, default_value = "memory_optimized", requires = "create_index")]
        diskann_storage_layout: Option<String>,

        /// DiskANN: Number of neighbors per node (default: 50)
        #[arg(long, requires = "create_index")]
        diskann_num_neighbors: Option<usize>,

        /// DiskANN: Search list size for construction (default: 100)
        #[arg(long, requires = "create_index")]
        diskann_search_list_size: Option<usize>,

        /// DiskANN: Alpha parameter (default: 1.2)
        #[arg(long, requires = "create_index")]
        diskann_max_alpha: Option<f32>,

        /// DiskANN: Number of dimensions to index (0 = all)
        #[arg(long, requires = "create_index")]
        diskann_num_dimensions: Option<usize>,

        /// DiskANN: Number of bits per dimension for SBQ
        #[arg(long, requires = "create_index")]
        diskann_num_bits_per_dimension: Option<usize>,

        // HNSW index hyperparameters
        /// HNSW: Max number of connections per layer (default: 16)
        #[arg(long, requires = "create_index")]
        hnsw_m: Option<usize>,

        /// HNSW: Size of dynamic candidate list for construction (default: 64)
        #[arg(long, requires = "create_index")]
        hnsw_ef_construction: Option<usize>,

        // IVFFlat index hyperparameters
        /// IVFFlat: Number of lists (default depends on data size)
        #[arg(long, requires = "create_index")]
        ivfflat_lists: Option<usize>,

        /// Maximum label value (0 to disable labels)
        #[arg(long, default_value_t = 0)]
        max_label: u16,

        /// Number of labels per row (0 to disable labels)
        #[arg(long, default_value_t = 0)]
        num_labels: usize,

        /// Use normal distribution for labels instead of uniform
        #[arg(long, default_value_t = false)]
        normal: bool,
    },

    /// Run test queries and calculate recall
    Test {
        /// Path to HDF5 file (either --file or --dataset must be specified)
        #[arg(short, long, group = "input_source")]
        file: Option<PathBuf>,

        /// Dataset name from ann-benchmarks (either --file or --dataset must be specified)
        #[arg(short, long, group = "input_source")]
        dataset: Option<String>,

        /// Dataset name for test queries within the HDF5 file (usually 'test')
        #[arg(short = 'q', long, default_value = "test")]
        query_dataset: String,

        /// Dataset name for ground truth within the HDF5 file (usually 'neighbors')
        #[arg(short = 'g', long, default_value = "neighbors")]
        neighbors_dataset: String,

        /// Table name to query against
        #[arg(short, long)]
        table: String,

        /// Table name to use for ground truth results (if not specified, uses HDF5 ground truth)
        #[arg(long)]
        ground_truth_table: Option<String>,

        /// Number of queries to run (0 = all)
        #[arg(short, long, default_value_t = 0)]
        num_queries: usize,

        /// Number of nearest neighbors to retrieve
        #[arg(short, long, default_value_t = 10)]
        top_k: usize,

        /// Distance metric to use for querie
        #[arg(long, default_value_t = DistanceType::Cosine)]
        distance_metric: DistanceType,

        /// Show detailed recall information for each query
        #[arg(short, long)]
        verbose: bool,

        // DiskANN query-time parameters
        /// DiskANN: Number of additional candidates during graph search (default: 100)
        #[arg(long, default_value = "100")]
        diskann_query_search_list_size: Option<usize>,

        /// DiskANN: Number of elements to rescore (default: 115, 0 to disable)
        #[arg(long, default_value = "115")]
        diskann_query_rescore: Option<usize>,

        // HNSW query-time parameters
        /// HNSW: Size of dynamic candidate list for search (default: 40)
        #[arg(long)]
        hnsw_ef_search: Option<usize>,

        /// HNSW: Maximum number of tuples to scan during search
        #[arg(long)]
        hnsw_max_scan_tuples: Option<usize>,

        /// HNSW: Memory multiplier for scan operations
        #[arg(long)]
        hnsw_scan_mem_multiplier: Option<usize>,

        /// HNSW: Iterative scan order ("strict_order" or "relaxed_order")
        #[arg(long)]
        hnsw_iterative_scan: Option<String>,

        /// HNSW: Number of candidates to rerank
        #[arg(long)]
        hnsw_rerank_candidates: Option<usize>,

        // IVFFlat query-time parameters
        /// IVFFlat: Number of lists to probe (default: 1)
        #[arg(long)]
        ivfflat_probes: Option<usize>,

        /// Maximum label value (0 to disable labels)
        #[arg(long, default_value_t = 0)]
        max_label: u16,

        /// Number of labels per row (0 to disable labels)
        #[arg(long, default_value_t = 0)]
        num_labels: usize,

        /// Use normal distribution for labels instead of uniform
        #[arg(long, default_value_t = false)]
        normal: bool,

        /// Number of warmup runs before timing each query
        #[arg(long, default_value_t = 0)]
        warmup: usize,

        /// CSV file to append results to
        #[arg(long)]
        csv: Option<PathBuf>,
    },
}

/// Parameters for DiskANN index creation
struct DiskAnnIndexParams {
    storage_layout: Option<String>,
    num_neighbors: Option<usize>,
    search_list_size: Option<usize>,
    max_alpha: Option<f32>,
    num_dimensions: Option<usize>,
    num_bits_per_dimension: Option<usize>,
}

/// Parameters for HNSW index creation
struct HnswIndexParams {
    m: Option<usize>,
    ef_construction: Option<usize>,
}

/// Parameters for IVFFlat index creation
struct IvfFlatIndexParams {
    lists: Option<usize>,
}

/// Combined parameters for index creation
struct IndexParams {
    index_type: IndexType,
    distance_metric: DistanceType,
    diskann: DiskAnnIndexParams,
    hnsw: HnswIndexParams,
    ivfflat: IvfFlatIndexParams,
}

/// Parameters for DiskANN query
struct DiskAnnQueryParams {
    query_search_list_size: Option<usize>,
    query_rescore: Option<usize>,
}

/// Parameters for HNSW query
struct HnswQueryParams {
    ef_search: Option<usize>,
    max_scan_tuples: Option<usize>,
    scan_mem_multiplier: Option<usize>,
    iterative_scan: Option<String>,
    /// rerank candidates implies HNSW BQ
    rerank_candidates: Option<usize>,
}

/// Parameters for IVFFlat query
struct IvfFlatQueryParams {
    probes: Option<usize>,
}

/// Combined parameters for query execution
struct QueryParams {
    table_name: String,
    query_vector: Vec<f32>,
    top_k: usize,
    distance_metric: DistanceType,
    diskann: DiskAnnQueryParams,
    hnsw: HnswQueryParams,
    ivfflat: IvfFlatQueryParams,
    max_label: u16,
    num_labels: usize,
    normal: bool,
}

struct QueryStats {
    query_times: Vec<Duration>,
    total_duration: Duration,
    num_queries: usize,
}

impl QueryStats {
    fn new(capacity: usize) -> Self {
        Self {
            query_times: Vec::with_capacity(capacity),
            total_duration: Duration::new(0, 0),
            num_queries: 0,
        }
    }

    fn add_query_time(&mut self, duration: Duration) {
        self.query_times.push(duration);
        self.total_duration += duration;
        self.num_queries += 1;
    }

    fn mean(&self) -> Duration {
        if self.num_queries == 0 {
            return Duration::new(0, 0);
        }
        self.total_duration / self.num_queries as u32
    }

    fn percentile(&self, p: f64) -> Duration {
        if self.query_times.is_empty() {
            return Duration::new(0, 0);
        }

        let mut sorted_times = self.query_times.clone();
        sorted_times.sort();

        let idx = (p * (sorted_times.len() - 1) as f64).round() as usize;
        sorted_times[idx]
    }

    fn min(&self) -> Duration {
        self.query_times
            .iter()
            .min()
            .cloned()
            .unwrap_or_else(|| Duration::new(0, 0))
    }

    fn max(&self) -> Duration {
        self.query_times
            .iter()
            .max()
            .cloned()
            .unwrap_or_else(|| Duration::new(0, 0))
    }

    fn print_histogram(&self, num_bins: usize) {
        if self.query_times.is_empty() {
            return;
        }

        let min_time = self.min().as_secs_f64() * 1000.0; // Convert to ms
        let max_time = self.max().as_secs_f64() * 1000.0; // Convert to ms

        if min_time == max_time {
            println!("All queries took exactly {:.2} ms", min_time);
            return;
        }

        let bin_width = (max_time - min_time) / num_bins as f64;
        let mut bins = vec![0; num_bins];

        for &time in &self.query_times {
            let ms = time.as_secs_f64() * 1000.0;
            let bin = ((ms - min_time) / bin_width).min((num_bins - 1) as f64) as usize;
            bins[bin] += 1;
        }

        let max_count = *bins.iter().max().unwrap_or(&0);
        let scale = 40.0 / max_count as f64;

        println!("\nQuery Time Histogram (ms)");
        println!("-------------------------");

        for (i, &count) in bins.iter().enumerate() {
            let lower = min_time + i as f64 * bin_width;
            let upper = min_time + (i + 1) as f64 * bin_width;
            let bar_length = (count as f64 * scale).round() as usize;
            let bar = "#".repeat(bar_length);
            println!("{:6.2}-{:6.2} | {:4} | {}", lower, upper, count, bar);
        }
    }

    fn print(&self) {
        println!("\nQuery Time Statistics (ms)");
        println!("--------------------------");
        println!("Min:          {:8.2}", self.min().as_secs_f64() * 1000.0);
        println!(
            "p25:          {:8.2}",
            self.percentile(0.25).as_secs_f64() * 1000.0
        );
        println!(
            "p50/Median:   {:8.2}",
            self.percentile(0.5).as_secs_f64() * 1000.0
        );
        println!(
            "p75:          {:8.2}",
            self.percentile(0.75).as_secs_f64() * 1000.0
        );
        println!(
            "p90:          {:8.2}",
            self.percentile(0.9).as_secs_f64() * 1000.0
        );
        println!(
            "p95:          {:8.2}",
            self.percentile(0.95).as_secs_f64() * 1000.0
        );
        println!(
            "p99:          {:8.2}",
            self.percentile(0.99).as_secs_f64() * 1000.0
        );
        println!("Max:          {:8.2}", self.max().as_secs_f64() * 1000.0);
        println!("Mean:         {:8.2}", self.mean().as_secs_f64() * 1000.0);
        println!(
            "Total:        {:8.2}",
            self.total_duration.as_secs_f64() * 1000.0
        );
        println!("Queries:      {}", self.num_queries);
        println!(
            "QPS:          {:8.2}",
            self.num_queries as f64 / self.total_duration.as_secs_f64()
        );

        // Print histogram with 10 bins
        self.print_histogram(10);
    }
}

struct PerformanceStats {
    operation: String,
    duration: Duration,
    items_processed: usize,
    items_per_second: f64,
}

impl PerformanceStats {
    fn new(operation: &str, duration: Duration, items_processed: usize) -> Self {
        let items_per_second = if duration.as_secs_f64() > 0.0 {
            items_processed as f64 / duration.as_secs_f64()
        } else {
            0.0
        };

        Self {
            operation: operation.to_string(),
            duration,
            items_processed,
            items_per_second,
        }
    }

    fn print(&self) {
        println!("\nPerformance Statistics");
        println!("----------------------");
        println!("Operation: {}", self.operation);
        println!("Duration: {:.2} seconds", self.duration.as_secs_f64());
        println!("Items processed: {}", self.items_processed);
        println!("Items per second: {:.2}", self.items_per_second);
    }
}

async fn connect_to_postgres(connection_string: &str) -> Result<Client, PgError> {
    let (client, connection) = tokio_postgres::connect(connection_string, NoTls).await?;

    // Spawn the connection handler to run in the background
    tokio::spawn(async move {
        if let Err(e) = connection.await {
            eprintln!("Connection error: {}", e);
        }
    });

    Ok(client)
}

/// Create a vector index on the specified table
async fn create_vector_index(
    client: &Client,
    table_name: &str,
    vector_dim: usize,
    params: &IndexParams,
) -> Result<(), PgError> {
    // Get the operator class for the distance metric
    let operator_class = params.distance_metric.get_operator_class();

    // Create the index based on the index type
    match params.index_type {
        IndexType::DiskANN => {
            // Build DiskANN index parameters
            let mut index_params = Vec::new();

            if let Some(storage_layout) = &params.diskann.storage_layout {
                index_params.push(format!("storage_layout='{}'", storage_layout));
            }

            if let Some(num_neighbors) = params.diskann.num_neighbors {
                index_params.push(format!("num_neighbors={}", num_neighbors));
            }

            if let Some(search_list_size) = params.diskann.search_list_size {
                index_params.push(format!("search_list_size={}", search_list_size));
            }

            if let Some(max_alpha) = params.diskann.max_alpha {
                index_params.push(format!("max_alpha={}", max_alpha));
            }

            if let Some(num_dimensions) = params.diskann.num_dimensions {
                index_params.push(format!("num_dimensions={}", num_dimensions));
            }

            if let Some(num_bits) = params.diskann.num_bits_per_dimension {
                index_params.push(format!("num_bits_per_dimension={}", num_bits));
            }

            // Create the index with parameters
            let sql = if index_params.is_empty() {
                format!(
                    "CREATE INDEX ON {} USING diskann (embedding {});",
                    table_name, operator_class
                )
            } else {
                format!(
                    "CREATE INDEX ON {} USING diskann (embedding {}) WITH ({});",
                    table_name,
                    operator_class,
                    index_params.join(", ")
                )
            };

            client.execute(&sql, &[]).await?
        }
        IndexType::Hnsw => {
            // Build HNSW index parameters
            let mut index_params = Vec::new();

            if let Some(m) = params.hnsw.m {
                index_params.push(format!("m={}", m));
            }

            if let Some(ef_construction) = params.hnsw.ef_construction {
                index_params.push(format!("ef_construction={}", ef_construction));
            }

            // Create the index with parameters
            let sql = if index_params.is_empty() {
                format!(
                    "CREATE INDEX ON {} USING hnsw (embedding {});",
                    table_name, operator_class
                )
            } else {
                format!(
                    "CREATE INDEX ON {} USING hnsw (embedding {}) WITH ({});",
                    table_name,
                    operator_class,
                    index_params.join(", ")
                )
            };

            client.execute(&sql, &[]).await?
        }
        IndexType::HnswBq => {
            // Build HNSW index parameters
            let mut index_params = Vec::new();

            if let Some(m) = params.hnsw.m {
                index_params.push(format!("m={}", m));
            }

            if let Some(ef_construction) = params.hnsw.ef_construction {
                index_params.push(format!("ef_construction={}", ef_construction));
            }

            // Create the index with parameters
            let sql = if index_params.is_empty() {
                format!(
                    "CREATE INDEX ON {} USING hnsw (binary_quantize(embedding)::bit({}) {});",
                    table_name, vector_dim, operator_class
                )
            } else {
                format!(
                    "CREATE INDEX ON {} USING hnsw (binary_quantize(embedding)::bit({}) {}) WITH ({});",
                    table_name,
                    vector_dim,
                    operator_class,
                    index_params.join(", ")
                )
            };

            client.execute(&sql, &[]).await?
        }
        IndexType::IVFFlat => {
            // Build IVFFlat index parameters
            let mut index_params = Vec::new();

            if let Some(lists) = params.ivfflat.lists {
                index_params.push(format!("lists={}", lists));
            }

            // Create the index with parameters
            let sql = if index_params.is_empty() {
                format!(
                    "CREATE INDEX ON {} USING ivfflat (embedding {});",
                    table_name, operator_class
                )
            } else {
                format!(
                    "CREATE INDEX ON {} USING ivfflat (embedding {}) WITH ({});",
                    table_name,
                    operator_class,
                    index_params.join(", ")
                )
            };

            client.execute(&sql, &[]).await?
        }
    };

    Ok(())
}

async fn create_vector_table(
    client: &Client,
    table_name: &str,
    vector_dim: usize,
    should_drop_existing: bool,
    max_label: u16,
    num_labels: usize,
) -> Result<(), PgError> {
    // Check if vectorscale extension is installed
    let extension_exists = client
        .query_one(
            "SELECT COUNT(*) FROM pg_extension WHERE extname = 'vectorscale'",
            &[],
        )
        .await?
        .get::<_, i64>(0);

    if extension_exists == 0 {
        println!("Installing vectorscale extension...");
        client
            .execute("CREATE EXTENSION IF NOT EXISTS vectorscale", &[])
            .await?;
    }

    // Drop the existing table if requested
    if should_drop_existing {
        println!("Dropping table {} if it exists...", table_name);
        client
            .execute(&format!("DROP TABLE IF EXISTS {}", table_name), &[])
            .await?;
    }

    // Create the table with an id, vector column, and optional labels column
    println!("Creating table {}...", table_name);
    let mut create_table_sql = format!(
        "CREATE TABLE {} (
            id INTEGER PRIMARY KEY,
            embedding VECTOR({})",
        table_name, vector_dim
    );

    if max_label > 0 && num_labels > 0 {
        create_table_sql.push_str(&format!(",\n            labels smallint[{}]", num_labels));
    }

    create_table_sql.push_str("\n        )");

    client.execute(&create_table_sql, &[]).await?;

    Ok(())
}

// Add this new function to generate deterministic random labels
fn generate_random_labels(
    max_label: u16,
    num_labels: usize,
    use_normal: bool,
    query_vector: &[f32],
) -> Vec<i16> {
    use rand::seq::SliceRandom;
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal};
    use rand_pcg::Pcg64;

    // Create a deterministic seed from the query vector
    let seed = blake3::hash(&bincode::serialize(query_vector).unwrap()).as_bytes()[0..8]
        .iter()
        .fold(0u64, |acc, &b| (acc << 8) | b as u64);

    // Create a deterministic RNG using the seed
    let mut rng = Pcg64::seed_from_u64(seed);

    if use_normal {
        // Create a normal distribution centered at max_label/2 with standard deviation of max_label/8
        let mean = max_label as f64 / 2.0;
        let std_dev = max_label as f64 / 8.0;
        let normal = Normal::new(mean, std_dev).unwrap();

        let mut labels = Vec::with_capacity(num_labels);
        for _ in 0..num_labels {
            let mut value;
            loop {
                value = normal.sample(&mut rng).round() as i16;
                if value >= 0 && value <= max_label as i16 {
                    break;
                }
            }
            labels.push(value);
        }
        labels
    } else {
        // Original uniform distribution implementation
        let mut labels: Vec<i16> = (0..=max_label).map(|x| x as i16).collect();
        labels.shuffle(&mut rng);
        labels.truncate(num_labels);
        labels
    }
}

#[allow(clippy::too_many_arguments)]
async fn load_vectors(
    client: &mut Client,
    table_name: &str,
    vectors: &Array2<f32>,
    start_idx: usize,
    end_idx: usize,
    progress_bar: Option<Arc<Mutex<ProgressBar>>>,
    max_label: u16,
    num_labels: usize,
    normal: bool,
) -> Result<(), PgError> {
    // Define chunk size (1M vectors)
    const CHUNK_SIZE: usize = 1_000_000;

    // Calculate total vectors to process
    let total_vectors = end_idx - start_idx;

    // Process vectors in chunks
    let mut current_idx = start_idx;
    while current_idx < end_idx {
        // Mark that we have an active transaction
        ACTIVE_TRANSACTIONS.store(true, Ordering::SeqCst);

        // Calculate end of current chunk
        let chunk_end = std::cmp::min(current_idx + CHUNK_SIZE, end_idx);

        // Start a transaction for this chunk
        let transaction = client.transaction().await?;

        // Prepare the COPY command for binary format
        let copy_cmd = if max_label > 0 && num_labels > 0 {
            format!("COPY {} (id, embedding, labels) FROM STDIN", table_name)
        } else {
            format!("COPY {} (id, embedding) FROM STDIN", table_name)
        };

        // Start the COPY operation
        let sink = transaction.copy_in(&copy_cmd).await?;

        // Helper function to convert io::Error to PgError
        let to_pg_error = |_e: std::io::Error| -> PgError { PgError::__private_api_timeout() };

        // Use Box::pin to pin the sink properly
        let mut sink = Box::pin(sink);

        // Define batch size for sending data to avoid connection issues
        const BATCH_SIZE: usize = 10000;

        // Process vectors in batches within the current chunk
        let mut batch_start = current_idx;
        while batch_start < chunk_end {
            // Check if we should abort due to Ctrl+C
            if SHOULD_ABORT.load(Ordering::SeqCst) {
                println!("Aborting transaction due to user interrupt...");
                transaction.rollback().await?;
                ACTIVE_TRANSACTIONS.store(false, Ordering::SeqCst);
                return Err(PgError::__private_api_timeout());
            }

            // Determine end of current batch
            let batch_end = std::cmp::min(batch_start + BATCH_SIZE, chunk_end);

            // Create a buffer for this batch
            let mut buffer = Vec::with_capacity((batch_end - batch_start) * 64);

            // Process each vector in this batch
            for i in batch_start..batch_end {
                // Calculate the local index within the current chunk
                let local_idx = i - batch_start;
                let vector = vectors.row(local_idx);
                let vector_data = vector.as_slice().unwrap();

                // Format the vector as a PostgreSQL array string
                let vector_str = format_vector_for_postgres(vector_data);

                // Calculate the actual ID from the HDF5 file position
                let actual_id = current_idx + i;
                if actual_id == 0 {
                    println!("Debug: Writing first vector with ID={}", actual_id);
                }

                // Write the id (use the actual index from the HDF5 file)
                buffer
                    .write_all(actual_id.to_string().as_bytes())
                    .map_err(to_pg_error)?;

                // Write tab separator
                buffer.write_all(b"\t").map_err(to_pg_error)?;

                // Write the vector string
                buffer
                    .write_all(vector_str.as_bytes())
                    .map_err(to_pg_error)?;

                // Write labels if enabled
                if max_label > 0 && num_labels > 0 {
                    // Write tab separator
                    buffer.write_all(b"\t").map_err(to_pg_error)?;

                    // Generate and format random labels
                    let labels =
                        generate_random_labels(max_label, num_labels, normal, &vector_data);
                    let labels_str = format!(
                        "{{{}}}",
                        labels
                            .iter()
                            .map(|x| x.to_string())
                            .collect::<Vec<_>>()
                            .join(",")
                    );
                    buffer
                        .write_all(labels_str.as_bytes())
                        .map_err(to_pg_error)?;
                }

                // Write newline
                buffer.write_all(b"\n").map_err(to_pg_error)?;

                if let Some(pb) = &progress_bar {
                    pb.lock().unwrap().inc(1);
                }
            }

            // Send this batch to the sink
            sink.as_mut().feed(Bytes::from(buffer)).await?;

            // Move to the next batch
            batch_start = batch_end;
        }

        // Write the end-of-copy marker
        let mut end_marker = Vec::new();
        end_marker.write_all(b"\\.").map_err(to_pg_error)?;
        end_marker.write_all(b"\n").map_err(to_pg_error)?;
        sink.as_mut().feed(Bytes::from(end_marker)).await?;

        // Finish the COPY operation
        sink.as_mut().close().await?;

        // Check if we should abort due to Ctrl+C
        if SHOULD_ABORT.load(Ordering::SeqCst) {
            println!("Aborting transaction due to user interrupt...");
            transaction.rollback().await?;
            ACTIVE_TRANSACTIONS.store(false, Ordering::SeqCst);
            return Err(PgError::__private_api_timeout());
        }

        // Commit the transaction for this chunk
        transaction.commit().await?;

        // Mark that we no longer have an active transaction
        ACTIVE_TRANSACTIONS.store(false, Ordering::SeqCst);

        // Move to the next chunk
        current_idx = chunk_end;

        // Print progress for chunk completion
        println!(
            "Completed chunk: {}/{} vectors",
            current_idx - start_idx,
            total_vectors
        );
    }

    Ok(())
}

// Format vector specifically for PostgreSQL vector type
fn format_vector_for_postgres(vector: &[f32]) -> String {
    // The pgvector format is simply [val1,val2,val3,...]
    // No spaces after commas to avoid dimension parsing issues
    let mut vector_str = String::from("[");

    for (i, &val) in vector.iter().enumerate() {
        if i > 0 {
            vector_str.push(','); // No space after comma
        }
        vector_str.push_str(&val.to_string());
    }

    vector_str.push(']');
    vector_str
}

async fn run_query(
    client: &Client,
    params: &QueryParams,
    verbose: bool,
    label_predicate: &str,
    vector_dim: usize,
    is_ground_truth: bool,
    warmup: usize,
) -> Result<(Vec<(i32, f64)>, Duration), PgError> {
    // Set session GUCs to ensure the index is used
    let sql = "SET enable_seqscan = OFF";
    if verbose {
        println!("SQL: {}", sql);
    }
    client.execute(sql, &[]).await?;

    // Set DiskANN query parameters if provided
    if let Some(search_list_size) = params.diskann.query_search_list_size {
        let sql = format!("SET diskann.query_search_list_size = {}", search_list_size);
        if verbose {
            println!("SQL: {}", sql);
        }
        client.execute(&sql, &[]).await?;
    }

    if let Some(rescore) = params.diskann.query_rescore {
        let sql = format!("SET diskann.query_rescore = {}", rescore);
        if verbose {
            println!("SQL: {}", sql);
        }
        client.execute(&sql, &[]).await?;
    }

    // Set HNSW query parameters if provided
    if let Some(ef_search) = params.hnsw.ef_search {
        let sql = format!("SET hnsw.ef_search = {}", ef_search);
        if verbose {
            println!("SQL: {}", sql);
        }
        client.execute(&sql, &[]).await?;
    }

    if let Some(max_scan_tuples) = params.hnsw.max_scan_tuples {
        let sql = format!("SET hnsw.max_scan_tuples = {}", max_scan_tuples);
        if verbose {
            println!("SQL: {}", sql);
        }
        client.execute(&sql, &[]).await?;
    }

    if let Some(scan_mem_multiplier) = params.hnsw.scan_mem_multiplier {
        let sql = format!("SET hnsw.scan_mem_multiplier = {}", scan_mem_multiplier);
        if verbose {
            println!("SQL: {}", sql);
        }
        client.execute(&sql, &[]).await?;
    }

    if let Some(iterative_scan) = &params.hnsw.iterative_scan {
        let sql = format!("SET hnsw.iterative_scan = '{}'", iterative_scan);
        if verbose {
            println!("SQL: {}", sql);
        }
        client.execute(&sql, &[]).await?;
    }

    // Set IVFFlat query parameters if provided
    if let Some(probes) = params.ivfflat.probes {
        let sql = format!("SET ivfflat.probes = {}", probes);
        if verbose {
            println!("SQL: {}", sql);
        }
        client.execute(&sql, &[]).await?;
    }

    // Get the appropriate operator for the distance metric
    let distance_operator = params.distance_metric.get_operator();

    // Format the vector for PostgreSQL
    let vector_str = format_vector_for_postgres(&params.query_vector);

    // For ground truth queries, try to use the cache first
    if is_ground_truth {
        let labels = if params.max_label > 0 && params.num_labels > 0 {
            Some(generate_random_labels(
                params.max_label,
                params.num_labels,
                params.normal,
                &params.query_vector,
            ))
        } else {
            None
        };

        // Create a hash of the query vector
        let query_vector_hash = blake3::hash(&bincode::serialize(&params.query_vector).unwrap())
            .to_hex()
            .to_string();

        let cache_key = GroundTruthCacheKey {
            table_name: params.table_name.clone(),
            query_vector_hash,
            labels: labels.clone(),
            top_k: params.top_k,
        };

        if let Ok(Some(cached_results)) = load_from_cache(&cache_key) {
            if verbose {
                println!("Using cached ground truth results");
            }
            return Ok((cached_results, Duration::new(0, 0)));
        }
    }

    // Construct the SQL query with the vector literal directly in the query
    // Include the distance in the result
    let query = if params.hnsw.rerank_candidates.is_none() || is_ground_truth {
        // Standard query without re-ranking
        format!(
            "SELECT id, embedding {} '{}' as distance FROM {} {} ORDER BY distance LIMIT {}",
            distance_operator, vector_str, params.table_name, label_predicate, params.top_k
        )
    } else {
        // Use re-ranking pattern when rerank_candidates is specified and not ground truth
        let rerank_candidates = params.hnsw.rerank_candidates.unwrap();
        format!(
            "WITH candidates AS (
                SELECT id, embedding, binary_quantize(embedding)::bit({}) <~> binary_quantize('{}'::vector({}))::bit({}) as distance
                FROM {} {}
                ORDER BY distance
                LIMIT {}
            )
            SELECT id, embedding {} '{}' as distance
            FROM candidates
            ORDER BY distance
            LIMIT {}",
            vector_dim,
            vector_str,
            vector_dim,
            vector_dim,
            params.table_name,
            label_predicate,
            rerank_candidates,
            distance_operator,
            vector_str,
            params.top_k
        )
    };

    // Echo the SQL query if verbose is enabled
    if verbose {
        println!("SQL: {}", query);
    }

    // Run warmup queries if specified
    for i in 0..warmup {
        if verbose {
            println!("Warmup run {}/{}", i + 1, warmup);
        }
        client.query(&query, &[]).await?;
    }

    // Clear the query cache after warmup
    if warmup > 0 {
        let clear_cache_sql = "DISCARD ALL";
        if verbose {
            println!("SQL: {}", clear_cache_sql);
        }
        client.execute(clear_cache_sql, &[]).await?;
    }

    // Run the actual query and measure its time
    let start_time = Instant::now();
    let rows = client.query(&query, &[]).await?;
    let query_time = start_time.elapsed();

    let result: Vec<(i32, f64)> = rows
        .iter()
        .map(|row| (row.get::<_, i32>(0), row.get::<_, f64>(1)))
        .collect();

    // Cache the results if this is a ground truth query
    if is_ground_truth {
        let labels = if params.max_label > 0 && params.num_labels > 0 {
            Some(generate_random_labels(
                params.max_label,
                params.num_labels,
                params.normal,
                &params.query_vector,
            ))
        } else {
            None
        };

        // Create a hash of the query vector
        let query_vector_hash = blake3::hash(&bincode::serialize(&params.query_vector).unwrap())
            .to_hex()
            .to_string();

        let cache_key = GroundTruthCacheKey {
            table_name: params.table_name.clone(),
            query_vector_hash,
            labels,
            top_k: params.top_k,
        };

        if let Err(e) = save_to_cache(&cache_key, &result) {
            eprintln!("Warning: Failed to cache ground truth results: {}", e);
        }
    }

    Ok((result, query_time))
}

/// Generate random labels for the query if enabled
fn construct_label_predicate(params: &QueryParams) -> String {
    if params.max_label > 0 && params.num_labels > 0 {
        let labels = generate_random_labels(
            params.max_label,
            params.num_labels,
            params.normal,
            &params.query_vector,
        );
        let labels_str = format!(
            "{{{}}}",
            labels
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
        format!(" WHERE labels && '{}'", labels_str)
    } else {
        String::new()
    }
}

/// Cf. the recall measure `knn` in ann-benchmarks
fn calculate_recall(
    actual: &[(i32, f64)],
    expected: &[(i32, f64)],
    verbose: bool,
    epsilon: f64,
) -> f64 {
    if expected.is_empty() {
        return 1.0;
    }
    let threshold = expected[expected.len() - 1].1 + epsilon;
    let count = actual
        .iter()
        .filter(|&&(_, distance)| distance <= threshold)
        .count();
    if verbose {
        println!(
            "count={}, actual.len()={}, expected.len()={}",
            count,
            actual.len(),
            expected.len()
        );
    }
    count as f64 / expected.len() as f64
}

/// Get the list of available datasets from ann-benchmarks
async fn get_ann_benchmark_datasets() -> Result<Vec<DatasetInfo>, Box<dyn std::error::Error>> {
    // This is a hardcoded list based on the README.md from ann-benchmarks
    // In a real implementation, you might want to scrape this from the website or use an API
    let datasets = vec![
        DatasetInfo {
            name: "deep-image-96-angular".to_string(),
            dimensions: 96,
            train_size: 9990000,
            test_size: 10000,
            neighbors: 100,
            distance: "cosine".to_string(),
            url: "http://ann-benchmarks.com/deep-image-96-angular.hdf5".to_string(),
        },
        DatasetInfo {
            name: "fashion-mnist-784-euclidean".to_string(),
            dimensions: 784,
            train_size: 60000,
            test_size: 10000,
            neighbors: 100,
            distance: "euclidean".to_string(),
            url: "http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5".to_string(),
        },
        DatasetInfo {
            name: "gist-960-euclidean".to_string(),
            dimensions: 960,
            train_size: 1000000,
            test_size: 1000,
            neighbors: 100,
            distance: "euclidean".to_string(),
            url: "http://ann-benchmarks.com/gist-960-euclidean.hdf5".to_string(),
        },
        DatasetInfo {
            name: "glove-25-angular".to_string(),
            dimensions: 25,
            train_size: 1183514,
            test_size: 10000,
            neighbors: 100,
            distance: "cosine".to_string(),
            url: "http://ann-benchmarks.com/glove-25-angular.hdf5".to_string(),
        },
        DatasetInfo {
            name: "glove-50-angular".to_string(),
            dimensions: 50,
            train_size: 1183514,
            test_size: 10000,
            neighbors: 100,
            distance: "cosine".to_string(),
            url: "http://ann-benchmarks.com/glove-50-angular.hdf5".to_string(),
        },
        DatasetInfo {
            name: "glove-100-angular".to_string(),
            dimensions: 100,
            train_size: 1183514,
            test_size: 10000,
            neighbors: 100,
            distance: "cosine".to_string(),
            url: "http://ann-benchmarks.com/glove-100-angular.hdf5".to_string(),
        },
        DatasetInfo {
            name: "glove-200-angular".to_string(),
            dimensions: 200,
            train_size: 1183514,
            test_size: 10000,
            neighbors: 100,
            distance: "cosine".to_string(),
            url: "http://ann-benchmarks.com/glove-200-angular.hdf5".to_string(),
        },
        DatasetInfo {
            name: "mnist-784-euclidean".to_string(),
            dimensions: 784,
            train_size: 60000,
            test_size: 10000,
            neighbors: 100,
            distance: "euclidean".to_string(),
            url: "http://ann-benchmarks.com/mnist-784-euclidean.hdf5".to_string(),
        },
        DatasetInfo {
            name: "sift-128-euclidean".to_string(),
            dimensions: 128,
            train_size: 1000000,
            test_size: 10000,
            neighbors: 100,
            distance: "euclidean".to_string(),
            url: "http://ann-benchmarks.com/sift-128-euclidean.hdf5".to_string(),
        },
        DatasetInfo {
            name: "nytimes-256-angular".to_string(),
            dimensions: 256,
            train_size: 290000,
            test_size: 10000,
            neighbors: 100,
            distance: "cosine".to_string(),
            url: "http://ann-benchmarks.com/nytimes-256-angular.hdf5".to_string(),
        },
        DatasetInfo {
            name: "lastfm-64-dot".to_string(),
            dimensions: 65,
            train_size: 292385,
            test_size: 50000,
            neighbors: 100,
            distance: "inner_product".to_string(),
            url: "http://ann-benchmarks.com/lastfm-64-dot.hdf5".to_string(),
        },
        DatasetInfo {
            name: "cohere-wikipedia-22-12-1M-angular".to_string(),
            dimensions: 768,
            train_size: 1000000,
            test_size: 10000,
            neighbors: 100,
            distance: "cosine".to_string(),
            url: "s3://vector-datasets/1M/cohere-wikipedia-22-12-1M-euclidean.hdf5".to_string(),
        },
        DatasetInfo {
            name: "cohere-wikipedia-22-12-10M-angular".to_string(),
            dimensions: 768,
            train_size: 10000000,
            test_size: 5000,
            neighbors: 100,
            distance: "cosine".to_string(),
            url: "s3://vector-datasets/10M/cohere-wikipedia-22-12-10M-euclidean.hdf5".to_string(),
        },
        DatasetInfo {
            name: "cohere-wikipedia-22-12-50M-angular".to_string(),
            dimensions: 768,
            train_size: 10000000,
            test_size: 5000,
            neighbors: 100,
            distance: "cosine".to_string(),
            url: "s3://vector-datasets/10M/cohere-wikipedia-22-12-50M-euclidean.hdf5".to_string(),
        },
    ];

    Ok(datasets)
}

/// List available datasets from ann-benchmarks
async fn list_ann_benchmark_datasets() -> Result<(), Box<dyn std::error::Error>> {
    let datasets = get_ann_benchmark_datasets().await?;

    println!("Available datasets from ann-benchmarks:");
    println!(
        "{:<30} {:<10} {:<12} {:<10} {:<10} {:<10}",
        "Name", "Dimensions", "Train Size", "Test Size", "Neighbors", "Distance"
    );
    println!("{:-<86}", "");

    for dataset in datasets {
        println!(
            "{:<30} {:<10} {:<12} {:<10} {:<10} {:<10}",
            dataset.name,
            dataset.dimensions,
            dataset.train_size,
            dataset.test_size,
            dataset.neighbors,
            dataset.distance
        );
    }

    println!(
        "\nTo download and load a dataset, use the 'download-and-load' command with --dataset <name>"
    );

    Ok(())
}

/// Get the dataset cache directory
fn get_dataset_cache_dir() -> Result<PathBuf, Box<dyn std::error::Error>> {
    // Get the user's home directory
    let home_dir = dirs::home_dir().ok_or_else(|| "Could not find home directory".to_string())?;

    // Create the cache directory path
    let cache_dir = home_dir.join(".pgvectorscale").join("datasets");

    // Create the directory if it doesn't exist
    if !cache_dir.exists() {
        std::fs::create_dir_all(&cache_dir)?;
    }

    Ok(cache_dir)
}

/// Download a dataset from ann-benchmarks
async fn download_dataset(dataset: &DatasetInfo) -> Result<PathBuf, Box<dyn std::error::Error>> {
    // Get the cache directory
    let cache_dir = get_dataset_cache_dir()?;
    let file_path = cache_dir.join(format!("{}.hdf5", dataset.name));

    // Check if the file already exists
    if file_path.exists() {
        println!("Using cached dataset at: {}", file_path.display());
        return Ok(file_path);
    }

    println!("Downloading dataset from: {}", dataset.url);
    println!("Saving to: {}", file_path.display());

    // Create a progress bar for the download
    let progress_bar = ProgressBar::new(0);
    progress_bar.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {bytes}/{total_bytes} ({eta})")
            .unwrap()
            .progress_chars("##-"),
    );

    // Download the file
    let client = reqwest::Client::new();
    let response = client.get(&dataset.url).send().await?;
    let total_size = response.content_length().unwrap_or(0);
    progress_bar.set_length(total_size);

    let mut file = std::fs::File::create(&file_path)?;
    let mut downloaded: u64 = 0;
    let mut stream = response.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        file.write_all(&chunk)?;
        downloaded += chunk.len() as u64;
        progress_bar.set_position(downloaded);
    }

    progress_bar.finish_with_message("Download complete");

    // Return the path to the downloaded file
    Ok(file_path)
}

// Global variables to track active transactions
static ACTIVE_TRANSACTIONS: AtomicBool = AtomicBool::new(false);
static SHOULD_ABORT: AtomicBool = AtomicBool::new(false);

#[derive(Debug, Serialize)]
struct TestResult {
    timestamp: chrono::DateTime<chrono::Utc>,
    dataset: String,
    table: String,
    ground_truth_table: Option<String>,
    num_queries: usize,
    top_k: usize,
    distance_metric: String,
    avg_recall: f64,
    min_query_time_ms: f64,
    p25_query_time_ms: f64,
    p50_query_time_ms: f64,
    p75_query_time_ms: f64,
    p90_query_time_ms: f64,
    p95_query_time_ms: f64,
    p99_query_time_ms: f64,
    max_query_time_ms: f64,
    mean_query_time_ms: f64,
    qps: f64,
    warmup: usize,
    diskann_query_search_list_size: Option<usize>,
    diskann_query_rescore: Option<usize>,
    hnsw_ef_search: Option<usize>,
    hnsw_max_scan_tuples: Option<usize>,
    hnsw_scan_mem_multiplier: Option<usize>,
    hnsw_iterative_scan: Option<String>,
    hnsw_rerank_candidates: Option<usize>,
    ivfflat_probes: Option<usize>,
    max_label: u16,
    num_labels: usize,
    normal: bool,
    command_line: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Set up Ctrl+C handler
    let ctrl_c = tokio::signal::ctrl_c();
    tokio::spawn(async move {
        match ctrl_c.await {
            Ok(()) => {
                println!("\nReceived Ctrl+C, cleaning up...");

                // Check if there are active transactions
                if ACTIVE_TRANSACTIONS.load(Ordering::SeqCst) {
                    println!("Aborting active transactions...");
                    // Set the abort flag to true so that transactions can be aborted
                    SHOULD_ABORT.store(true, Ordering::SeqCst);

                    // Give a small delay to allow transactions to abort
                    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
                }

                println!("Exiting.");
                std::process::exit(130); // Standard exit code for SIGINT
            }
            Err(err) => {
                eprintln!("Error setting up Ctrl+C handler: {}", err);
            }
        }
    });
    let cli = Cli::parse();

    match &cli.command {
        Commands::ListDatasets => list_ann_benchmark_datasets().await?,

        Commands::Load {
            file,
            dataset,
            dataset_name,
            table,
            create_table,
            num_vectors,
            transactions: _,
            parallel,
            create_index,
            index_type,
            distance_metric,
            diskann_storage_layout,
            diskann_num_neighbors,
            diskann_search_list_size,
            diskann_max_alpha,
            diskann_num_dimensions,
            diskann_num_bits_per_dimension,
            hnsw_m,
            hnsw_ef_construction,
            ivfflat_lists,
            max_label,
            num_labels,
            normal,
        } => {
            // Determine the file path and base table name
            let (file_path, base_table_name) = if let Some(file_path) = file {
                let name = file_path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("vectors");
                (file_path.clone(), kebab_to_snake_case(name))
            } else if let Some(dataset_name) = dataset {
                // Get dataset information
                let datasets = get_ann_benchmark_datasets().await?;
                let dataset_info = datasets.iter().find(|d| d.name == *dataset_name);

                if let Some(dataset_info) = dataset_info {
                    println!(
                        "Found dataset: {} ({}D, {} distance)",
                        dataset_info.name, dataset_info.dimensions, dataset_info.distance
                    );

                    // Download the dataset
                    let file_path = download_dataset(dataset_info).await?;
                    (file_path, kebab_to_snake_case(&dataset_info.name))
                } else {
                    println!("Dataset '{}' not found. Use the ListDatasets command to see available datasets.", dataset_name);
                    return Ok(());
                }
            } else {
                println!("Error: Either --file or --dataset must be specified");
                return Ok(());
            };

            // Determine the final table name
            let table_name = if let Some(name) = table {
                name.clone()
            } else {
                let index_suffix = match index_type {
                    IndexType::DiskANN => "diskann",
                    IndexType::Hnsw => "hnsw",
                    IndexType::HnswBq => "hnsw_bq",
                    IndexType::IVFFlat => "ivfflat",
                };
                format!("{}_{}", base_table_name, index_suffix)
            };

            // Open the HDF5 file
            let h5_file = File::open(&file_path)?;

            // Get the dataset
            let dataset = h5_file.dataset(dataset_name)?;

            // Get the dimensions of the dataset
            let shape = dataset.shape();
            let total_vectors = shape[0];
            let vector_dim = shape[1];

            // Determine how many vectors to load
            let vectors_to_load = if *num_vectors == 0 || *num_vectors > total_vectors {
                total_vectors
            } else {
                *num_vectors
            };

            println!(
                "Loading {} vectors with dimension {}",
                vectors_to_load, vector_dim
            );

            // Connect to PostgreSQL
            let mut client = connect_to_postgres(&cli.connection_string).await?;

            // Create the table if requested
            if *create_table {
                create_vector_table(
                    &client,
                    &table_name,
                    vector_dim,
                    true,
                    *max_label,
                    *num_labels,
                )
                .await?;
            }

            // Set up progress bar
            let progress_bar = ProgressBar::new(vectors_to_load as u64);
            progress_bar.set_style(
                ProgressStyle::default_bar()
                    .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({eta})")
                    .unwrap()
                    .progress_chars("##-"),
            );
            let progress_bar = Arc::new(Mutex::new(progress_bar));

            let start_time = Instant::now();

            // Define chunk size for HDF5 reading (1M vectors)
            const HDF5_CHUNK_SIZE: usize = 1_000_000;

            if *parallel > 1 {
                // Parallel loading with multiple connections
                let vectors_per_connection = vectors_to_load / *parallel;
                let mut handles = Vec::new();

                for i in 0..*parallel {
                    let start_idx = i * vectors_per_connection;
                    let end_idx = if i == *parallel - 1 {
                        vectors_to_load
                    } else {
                        (i + 1) * vectors_per_connection
                    };

                    let connection_string = cli.connection_string.clone();
                    let table_name = table_name.clone();
                    let file_path = file_path.clone();
                    let dataset_name = dataset_name.clone();
                    let pb_clone = progress_bar.clone();
                    let max_label = *max_label;
                    let num_labels = *num_labels;
                    let normal = *normal;

                    let handle = task::spawn(async move {
                        let mut client = connect_to_postgres(&connection_string).await.unwrap();
                        let h5_file = File::open(&file_path).unwrap();
                        let dataset = h5_file.dataset(&dataset_name).unwrap();

                        // Process vectors in chunks
                        let mut current_idx = start_idx;
                        while current_idx < end_idx {
                            let chunk_end = std::cmp::min(current_idx + HDF5_CHUNK_SIZE, end_idx);

                            // Read only the current chunk from HDF5 using hyperslab selection
                            let hyperslab = Hyperslab::new((
                                current_idx..chunk_end,
                                0..vector_dim
                            ));
                            let selection = Selection::new(hyperslab);
                            let vectors = match dataset.read_slice::<f32, Selection, Ix2>(selection) {
                                Ok(arr) => {
                                    // Convert to the correct ndarray version
                                    let (data, _) = arr.into_raw_vec_and_offset();
                                    Array2::from_shape_vec((chunk_end - current_idx, vector_dim), data).unwrap()
                                },
                                Err(e) => {
                                    eprintln!("Error reading HDF5 data: {}", e);
                                    return;
                                }
                            };

                            if let Err(e) = load_vectors(
                                &mut client,
                                &table_name,
                                &vectors,
                                current_idx,  // Pass the actual start index
                                chunk_end,    // Pass the actual end index
                                Some(pb_clone.clone()),
                                max_label,
                                num_labels,
                                normal,
                            ).await {
                                eprintln!("Error loading vectors: {}", e);
                                return;
                            }

                            current_idx = chunk_end;
                        }
                    });

                    handles.push(handle);
                }

                // Wait for all parallel loads to complete
                join_all(handles).await;
            } else {
                // Single connection loading
                let mut current_idx = 0;
                while current_idx < vectors_to_load {
                    let chunk_end = std::cmp::min(current_idx + HDF5_CHUNK_SIZE, vectors_to_load);

                    // Read only the current chunk from HDF5 using hyperslab selection
                    let hyperslab = Hyperslab::new((
                        current_idx..chunk_end,
                        0..vector_dim
                    ));
                    let selection = Selection::new(hyperslab);
                    let vectors = match dataset.read_slice::<f32, Selection, Ix2>(selection) {
                        Ok(arr) => {
                            // Convert to the correct ndarray version
                            let (data, _) = arr.into_raw_vec_and_offset();
                            Array2::from_shape_vec((chunk_end - current_idx, vector_dim), data).unwrap()
                        },
                        Err(e) => {
                            eprintln!("Error reading HDF5 data: {}", e);
                            return Err(e.into());
                        }
                    };

                    load_vectors(
                        &mut client,
                        &table_name,
                        &vectors,
                        current_idx,  // Pass the actual start index
                        chunk_end,    // Pass the actual end index
                        Some(progress_bar.clone()),
                        *max_label,
                        *num_labels,
                        *normal,
                    )
                    .await?;

                    current_idx = chunk_end;
                }
            }

            let duration = start_time.elapsed();
            progress_bar
                .lock()
                .unwrap()
                .finish_with_message("Loading complete");

            // Report performance statistics
            let stats = PerformanceStats::new("Vector Loading", duration, vectors_to_load);
            stats.print();

            // Create index if requested
            if *create_index {
                println!(
                    "Creating {} index with {} distance metric...",
                    format!("{:?}", index_type).to_lowercase(),
                    format!("{:?}", distance_metric).to_lowercase()
                );
                let index_start_time = Instant::now();

                // Create index parameters struct
                let index_params = IndexParams {
                    index_type: *index_type,
                    distance_metric: *distance_metric,
                    diskann: DiskAnnIndexParams {
                        storage_layout: diskann_storage_layout.clone(),
                        num_neighbors: *diskann_num_neighbors,
                        search_list_size: *diskann_search_list_size,
                        max_alpha: *diskann_max_alpha,
                        num_dimensions: *diskann_num_dimensions,
                        num_bits_per_dimension: *diskann_num_bits_per_dimension,
                    },
                    hnsw: HnswIndexParams {
                        m: *hnsw_m,
                        ef_construction: *hnsw_ef_construction,
                    },
                    ivfflat: IvfFlatIndexParams {
                        lists: *ivfflat_lists,
                    },
                };

                create_vector_index(&client, &table_name, vector_dim, &index_params).await?;

                let index_duration = index_start_time.elapsed();
                let index_stats = PerformanceStats::new("Index Creation", index_duration, 1);
                index_stats.print();
            }
        }

        Commands::Test {
            file,
            dataset,
            query_dataset,
            neighbors_dataset,
            table,
            ground_truth_table,
            num_queries,
            top_k,
            distance_metric,
            verbose,
            diskann_query_search_list_size,
            diskann_query_rescore,
            hnsw_ef_search,
            hnsw_max_scan_tuples,
            hnsw_scan_mem_multiplier,
            hnsw_iterative_scan,
            hnsw_rerank_candidates,
            ivfflat_probes,
            max_label,
            num_labels,
            normal,
            warmup,
            csv,
        } => {
            // Determine the file path - either from direct file path or by downloading the dataset
            let file_path: PathBuf = if let Some(file_path) = file {
                file_path.clone()
            } else if let Some(dataset_name) = dataset {
                // Get dataset information
                let datasets = get_ann_benchmark_datasets().await?;
                let dataset_info = datasets.iter().find(|d| d.name == *dataset_name);

                if let Some(dataset_info) = dataset_info {
                    println!(
                        "Found dataset: {} ({}D, {} distance)",
                        dataset_info.name, dataset_info.dimensions, dataset_info.distance
                    );

                    // Download the dataset
                    download_dataset(dataset_info).await?
                } else {
                    println!("Dataset '{}' not found. Use the ListDatasets command to see available datasets.", dataset_name);
                    return Ok(());
                }
            } else {
                println!("Error: Either --file or --dataset must be specified");
                return Ok(());
            };

            // Open the HDF5 file
            let h5_file = File::open(&file_path)?;

            // Get the query dataset
            let query_dataset = h5_file.dataset(query_dataset)?;
            let query_shape = query_dataset.shape();
            let total_queries = query_shape[0];
            let vector_dim = query_shape[1];

            // Determine how many queries to run
            let queries_to_run = if *num_queries == 0 || *num_queries > total_queries {
                total_queries
            } else {
                *num_queries
            };

            println!(
                "Running {} queries with dimension {}",
                queries_to_run, vector_dim
            );

            // Read the query vectors
            let query_data = query_dataset.read_raw::<f32>()?;
            let query_vectors = Array::from_shape_vec((total_queries, vector_dim), query_data)?
                .slice(s![0..queries_to_run, ..])
                .to_owned();

            // Get the neighbors dataset
            let neighbors_dataset = h5_file.dataset(neighbors_dataset)?;

            // Read the ground truth neighbors
            let neighbors_data = neighbors_dataset.read_raw::<i32>()?;
            let neighbors_shape = neighbors_dataset.shape();

            // Make sure we don't request more neighbors than available in ground truth
            let top_k = if *top_k > neighbors_shape[1] {
                println!(
                    "Warning: Requested k={} but ground truth only has {} neighbors. Using k={}.",
                    *top_k, neighbors_shape[1], neighbors_shape[1]
                );
                neighbors_shape[1]
            } else {
                *top_k
            };

            let ground_truth_ids =
                Array::from_shape_vec((total_queries, neighbors_shape[1]), neighbors_data)?
                    .slice(s![0..queries_to_run, 0..top_k])
                    .to_owned();

            // Check if distances dataset exists
            let ground_truth_distances = if h5_file.dataset("distances").is_ok() {
                // Read the ground truth distances if available
                let distances_dataset = h5_file.dataset("distances")?;
                let distances_data = distances_dataset.read_raw::<f32>()?;
                Some(
                    Array::from_shape_vec((total_queries, neighbors_shape[1]), distances_data)?
                        .slice(s![0..queries_to_run, 0..top_k])
                        .to_owned(),
                )
            } else {
                if *verbose {
                    println!("No distances dataset found in HDF5 file. Will compute distances for ground truth.");
                }
                None
            };

            // Connect to PostgreSQL
            let client = connect_to_postgres(&cli.connection_string).await?;

            // Set up progress bar
            let progress_bar = ProgressBar::new(queries_to_run as u64);
            progress_bar.set_style(
                ProgressStyle::default_bar()
                    .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({eta})")
                    .unwrap()
                    .progress_chars("##-"),
            );

            let start_time = Instant::now();
            let mut total_recall = 0.0;
            let mut recall_values = Vec::with_capacity(queries_to_run);
            let mut query_stats = QueryStats::new(queries_to_run);

            // Run each query
            for i in 0..queries_to_run {
                let query_row = query_vectors.row(i);
                let query_vector_slice = query_row.as_slice().unwrap();

                // Create query parameters struct
                let mut query_params = QueryParams {
                    table_name: table.clone(),
                    query_vector: query_vector_slice.to_vec(),
                    top_k,
                    distance_metric: *distance_metric,
                    diskann: DiskAnnQueryParams {
                        query_search_list_size: *diskann_query_search_list_size,
                        query_rescore: *diskann_query_rescore,
                    },
                    hnsw: HnswQueryParams {
                        ef_search: *hnsw_ef_search,
                        max_scan_tuples: *hnsw_max_scan_tuples,
                        scan_mem_multiplier: *hnsw_scan_mem_multiplier,
                        iterative_scan: hnsw_iterative_scan.clone(),
                        rerank_candidates: *hnsw_rerank_candidates,
                    },
                    ivfflat: IvfFlatQueryParams {
                        probes: *ivfflat_probes,
                    },
                    max_label: *max_label,
                    num_labels: *num_labels,
                    normal: *normal,
                };

                let filter = construct_label_predicate(&query_params);

                // Run the query and get the result and query time
                let (result, query_duration) = run_query(
                    &client,
                    &query_params,
                    *verbose,
                    &filter,
                    vector_dim,
                    false,
                    *warmup,
                )
                .await?;
                query_stats.add_query_time(query_duration);

                if *verbose {
                    println!("Result: {:?}", result);
                }

                // Calculate recall for this query
                let expected_with_distances: Vec<(i32, f64)> =
                    if let Some(ground_truth_table) = ground_truth_table {
                        // Run the same query against the ground truth table
                        query_params.table_name = ground_truth_table.clone();
                        if *verbose {
                            println!(
                                "Running query against ground truth table: {}",
                                query_params.table_name
                            );
                        }
                        let (result, _) = run_query(
                            &client,
                            &query_params,
                            *verbose,
                            &filter,
                            vector_dim,
                            true,
                            *warmup,
                        )
                        .await?;
                        result
                    } else {
                        let ground_truth_ids_row = ground_truth_ids.row(i);
                        let expected_ids = ground_truth_ids_row.as_slice().unwrap();

                        // Create expected results with distances
                        if let Some(ref distances) = ground_truth_distances {
                            // Use distances from the HDF5 file
                            let distances_row = distances.row(i as usize);
                            let distances_slice = distances_row.as_slice().unwrap();
                            expected_ids
                                .iter()
                                .zip(distances_slice.iter())
                                .map(|(&id, &dist)| (id, dist as f64))
                                .collect()
                        } else {
                            panic!("No distances dataset found in HDF5 file");
                        }
                    };

                if *verbose {
                    println!("Expected with distances: {:?}", expected_with_distances);
                }

                // Use epsilon value of 0.001 (same as ann_benchmarks default)
                let epsilon = 0.001;
                let recall = calculate_recall(&result, &expected_with_distances, *verbose, epsilon);
                total_recall += recall;
                recall_values.push(recall);

                progress_bar.inc(1);
            }

            let duration = start_time.elapsed();
            progress_bar.finish_with_message("Testing complete");

            // Calculate average recall
            let avg_recall = total_recall / queries_to_run as f64;

            // Print detailed recall information if verbose flag is set
            if *verbose {
                println!("\nRecall Results:");
                println!("----------------");
                println!("Total queries: {}", queries_to_run);
                println!("K (nearest neighbors): {}", top_k);
                println!("Distance metric: {}", distance_metric);

                // Print individual recall values (limit to first 10 if there are many)
                if queries_to_run <= 10 {
                    for (i, recall) in recall_values.iter().enumerate() {
                        println!("Query {}: Recall@{} = {:.4}", i + 1, top_k, recall);
                    }
                } else {
                    for (i, &recall) in recall_values.iter().enumerate().take(5) {
                        println!("Query {}: Recall@{} = {:.4}", i + 1, top_k, recall);
                    }
                    println!("...");
                    // Get the last 5 queries
                    #[allow(clippy::needless_range_loop)]
                    for i in (queries_to_run - 5)..queries_to_run {
                        println!(
                            "Query {}: Recall@{} = {:.4}",
                            i + 1,
                            top_k,
                            recall_values[i]
                        );
                    }
                }

                println!();
            }

            println!("Average recall@{}: {:.4}", top_k, avg_recall);

            // Print detailed query time statistics
            query_stats.print();

            // Report overall performance statistics
            let stats = PerformanceStats::new("Query Testing", duration, queries_to_run);
            stats.print();

            // Write results to CSV if specified
            if let Some(csv_path) = csv {
                let result = TestResult {
                    timestamp: chrono::Utc::now(),
                    dataset: if let Some(dataset_name) = dataset {
                        dataset_name.clone()
                    } else if let Some(file_path) = file {
                        file_path
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .unwrap_or("unknown")
                            .to_string()
                    } else {
                        "unknown".to_string()
                    },
                    table: table.clone(),
                    ground_truth_table: ground_truth_table.clone(),
                    num_queries: queries_to_run,
                    top_k: top_k,
                    distance_metric: format!("{:?}", distance_metric),
                    avg_recall,
                    min_query_time_ms: query_stats.min().as_secs_f64() * 1000.0,
                    p25_query_time_ms: query_stats.percentile(0.25).as_secs_f64() * 1000.0,
                    p50_query_time_ms: query_stats.percentile(0.5).as_secs_f64() * 1000.0,
                    p75_query_time_ms: query_stats.percentile(0.75).as_secs_f64() * 1000.0,
                    p90_query_time_ms: query_stats.percentile(0.9).as_secs_f64() * 1000.0,
                    p95_query_time_ms: query_stats.percentile(0.95).as_secs_f64() * 1000.0,
                    p99_query_time_ms: query_stats.percentile(0.99).as_secs_f64() * 1000.0,
                    max_query_time_ms: query_stats.max().as_secs_f64() * 1000.0,
                    mean_query_time_ms: query_stats.mean().as_secs_f64() * 1000.0,
                    qps: queries_to_run as f64 / duration.as_secs_f64(),
                    warmup: *warmup,
                    diskann_query_search_list_size: *diskann_query_search_list_size,
                    diskann_query_rescore: *diskann_query_rescore,
                    hnsw_ef_search: *hnsw_ef_search,
                    hnsw_max_scan_tuples: *hnsw_max_scan_tuples,
                    hnsw_scan_mem_multiplier: *hnsw_scan_mem_multiplier,
                    hnsw_iterative_scan: hnsw_iterative_scan.clone(),
                    hnsw_rerank_candidates: *hnsw_rerank_candidates,
                    ivfflat_probes: *ivfflat_probes,
                    max_label: *max_label,
                    num_labels: *num_labels,
                    normal: *normal,
                    command_line: std::env::args().collect::<Vec<_>>().join(" "),
                };

                let file_exists = csv_path.exists();
                let mut writer = if !file_exists {
                    // Create new file and write header
                    let file = std::fs::File::create(csv_path)?;
                    let mut writer = csv::WriterBuilder::new()
                        .has_headers(true)
                        .from_writer(file);
                    writer.serialize(result)?;
                    writer
                } else {
                    // Open existing file in append mode
                    let file = std::fs::OpenOptions::new()
                        .write(true)
                        .append(true)
                        .open(csv_path)?;
                    let mut writer = csv::WriterBuilder::new()
                        .has_headers(false)
                        .from_writer(file);
                    writer.serialize(result)?;
                    writer
                };

                writer.flush()?;
            }
        }
    }

    Ok(())
}
