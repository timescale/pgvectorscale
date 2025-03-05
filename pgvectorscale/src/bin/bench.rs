use std::fmt;
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// Import the DistanceType from the vectorscale crate
use vectorscale::access_method::distance::DistanceType;

use clap::{Parser, Subcommand, ValueEnum};
use futures::{future::join_all, StreamExt};
use hdf5_metno::File;
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::{s, Array, Array2, ArrayView1};
use reqwest;
use serde::{Deserialize, Serialize};
use tokio::task;
use tokio_postgres::{Client, Error as PgError, NoTls};
use uuid::Uuid;

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
    /// IVFFlat index
    IVFFlat,
}

impl ValueEnum for IndexType {
    fn value_variants<'a>() -> &'a [Self] {
        &[IndexType::DiskANN, IndexType::Hnsw, IndexType::IVFFlat]
    }

    fn to_possible_value(&self) -> Option<clap::builder::PossibleValue> {
        Some(match self {
            IndexType::DiskANN => {
                clap::builder::PossibleValue::new("diskann").help("DiskANN index (default)")
            }
            IndexType::Hnsw => clap::builder::PossibleValue::new("hnsw").help("HNSW index"),
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
            IndexType::IVFFlat => write!(f, "ivfflat"),
        }
    }
}

#[derive(Subcommand)]
enum Commands {
    /// List available datasets from ann-benchmarks
    ListDatasets,

    /// List locally cached datasets
    ListCachedDatasets,

    /// Download and load a dataset from ann-benchmarks
    DownloadAndLoad {
        /// Dataset name to download and load
        #[arg(short, long)]
        dataset: String,

        /// Table name to load vectors into
        #[arg(short, long)]
        table: String,

        /// Whether to create a new table
        #[arg(short, long)]
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
        #[arg(short = 'i', long)]
        create_index: bool,

        /// Type of index to create
        #[arg(long, default_value_t = IndexType::DiskANN, requires = "create_index")]
        index_type: IndexType,

        /// Distance metric to use for the index (will be inferred if not provided)
        #[arg(long, requires = "create_index")]
        distance_metric: Option<DistanceType>,

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
    },

    /// Load training vectors from HDF5 file into PostgreSQL
    Load {
        /// Path to HDF5 file
        #[arg(short, long)]
        file: PathBuf,

        /// Dataset name within the HDF5 file (usually 'train')
        #[arg(short, long, default_value = "train")]
        dataset: String,

        /// Table name to load vectors into
        #[arg(short, long)]
        table: String,

        /// Whether to create a new table
        #[arg(short, long)]
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
        #[arg(short = 'i', long)]
        create_index: bool,

        /// Type of index to create
        #[arg(long, default_value_t = IndexType::DiskANN, requires = "create_index")]
        index_type: IndexType,

        /// Distance metric to use for the index
        #[arg(long, default_value_t = DistanceType::Cosine, requires = "create_index")]
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
    },

    /// Run test queries and calculate recall
    Test {
        /// Path to HDF5 file
        #[arg(short, long)]
        file: PathBuf,

        /// Dataset name for test queries within the HDF5 file (usually 'test')
        #[arg(short = 'q', long, default_value = "test")]
        query_dataset: String,

        /// Dataset name for ground truth within the HDF5 file (usually 'neighbors')
        #[arg(short = 'g', long, default_value = "neighbors")]
        neighbors_dataset: String,

        /// Table name to query against
        #[arg(short, long)]
        table: String,

        /// Number of queries to run (0 = all)
        #[arg(short, long, default_value_t = 0)]
        num_queries: usize,

        /// Number of nearest neighbors to retrieve
        #[arg(short, long, default_value_t = 100)]
        k: usize,

        /// Distance metric to use for queries
        #[arg(long, default_value_t = DistanceType::Cosine)]
        distance_metric: DistanceType,

        /// Show detailed recall information for each query
        #[arg(short, long)]
        verbose: bool,

        // DiskANN query-time parameters
        /// DiskANN: Number of additional candidates during graph search (default: 100)
        #[arg(long)]
        diskann_query_search_list_size: Option<usize>,

        /// DiskANN: Number of elements to rescore (default: 50, 0 to disable)
        #[arg(long)]
        diskann_query_rescore: Option<usize>,

        // HNSW query-time parameters
        /// HNSW: Size of dynamic candidate list for search (default: 40)
        #[arg(long)]
        hnsw_ef_search: Option<usize>,

        // IVFFlat query-time parameters
        /// IVFFlat: Number of lists to probe (default: 1)
        #[arg(long)]
        ivfflat_probes: Option<usize>,
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
}

/// Parameters for IVFFlat query
struct IvfFlatQueryParams {
    probes: Option<usize>,
}

/// Combined parameters for query execution
struct QueryParams {
    table_name: String,
    query_vector: Vec<f32>,
    k: usize,
    distance_metric: DistanceType,
    diskann: DiskAnnQueryParams,
    hnsw: HnswQueryParams,
    ivfflat: IvfFlatQueryParams,
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
    _vector_dim: usize, // Prefix with underscore to indicate intentionally unused
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

    // Create the table with an id and vector column
    println!("Creating table {}...", table_name);
    client
        .execute(
            &format!(
                "CREATE TABLE {} (
                    id INTEGER PRIMARY KEY,
                    embedding VECTOR({})
                )",
                table_name, vector_dim
            ),
            &[],
        )
        .await?;

    Ok(())
}

async fn load_vectors(
    client: &mut Client,
    table_name: &str,
    vectors: &Array2<f32>,
    start_idx: usize,
    end_idx: usize,
    progress_bar: Option<Arc<Mutex<ProgressBar>>>,
) -> Result<(), PgError> {
    use std::io::{Cursor, Write};
    use byteorder::{BigEndian, WriteBytesExt};
    
    // Start a transaction
    let transaction = client.transaction().await?;
    
    // Prepare the COPY command for binary format
    let copy_cmd = format!("COPY {} (id, embedding) FROM STDIN BINARY", table_name);
    
    // Start the COPY operation
    let sink = transaction.copy_in(&copy_cmd).await?;
    let mut writer = sink.writer();
    
    // Write the PostgreSQL binary format header
    // PGCOPY\n\377\r\n\0 - 11 bytes
    writer.write_all(b"PGCOPY\n\xff\r\n\0").await?;
    
    // Write the flags field (0 for no OIDs) - 4 bytes
    writer.write_u32::<BigEndian>(0).await?;
    
    // Write the header extension area length (0) - 4 bytes
    writer.write_u32::<BigEndian>(0).await?;
    
    // Process each vector
    for i in start_idx..end_idx {
        // Write tuple field count (2 fields: id and embedding) - 2 bytes
        writer.write_u16::<BigEndian>(2).await?;
        
        // Write id field
        // Field length - 4 bytes
        writer.write_u32::<BigEndian>(4).await?;
        // Field value - 4 bytes (i32)
        writer.write_i32::<BigEndian>(i as i32).await?;
        
        // Write embedding field
        let vector = vectors.row(i);
        let vector_data = vector.as_slice().unwrap();
        
        // Format the vector as a PostgreSQL array string
        let vector_str = format_vector_for_postgres(vector_data);
        let vector_bytes = vector_str.as_bytes();
        
        // Field length - 4 bytes
        writer.write_u32::<BigEndian>(vector_bytes.len() as u32).await?;
        // Field value - variable length
        writer.write_all(vector_bytes).await?;
        
        if let Some(pb) = &progress_bar {
            pb.lock().unwrap().inc(1);
        }
    }
    
    // Write file trailer - 2 bytes (indicates end of copy data)
    writer.write_i16::<BigEndian>(-1).await?;
    
    // Finish the COPY operation
    writer.flush().await?;
    sink.finish().await?;
    
    // Commit the transaction
    transaction.commit().await?;
    
    Ok(())
}

// Format vector specifically for CSV output
fn format_vector_for_csv(vector: &ArrayView1<f32>) -> String {
    let mut vector_str = String::from("\"[");

    for (i, &val) in vector.as_slice().unwrap().iter().enumerate() {
        if i > 0 {
            vector_str.push_str(", ");
        }
        vector_str.push_str(&val.to_string());
    }

    vector_str.push_str("]\"");
    vector_str
}

// Format vector specifically for PostgreSQL vector type
fn format_vector_for_postgres(vector: &[f32]) -> String {
    let mut vector_str = String::from("[");

    for (i, &val) in vector.iter().enumerate() {
        if i > 0 {
            vector_str.push_str(", ");
        }
        vector_str.push_str(&val.to_string());
    }

    vector_str.push(']');
    vector_str
}

async fn run_query(client: &Client, params: &QueryParams) -> Result<Vec<i32>, PgError> {
    // Set session GUCs to ensure the index is used
    client.execute("SET enable_seqscan = OFF", &[]).await?;
    client
        .execute("SET vectorscale.enable_diskann = ON", &[])
        .await?;
    client
        .execute("SET vectorscale.enable_hnsw = ON", &[])
        .await?;
    client
        .execute("SET vectorscale.enable_ivfflat = ON", &[])
        .await?;

    // Set DiskANN query parameters if provided
    if let Some(search_list_size) = params.diskann.query_search_list_size {
        client
            .execute(
                &format!("SET diskann.query_search_list_size = {}", search_list_size),
                &[],
            )
            .await?;
    }

    if let Some(rescore) = params.diskann.query_rescore {
        client
            .execute(&format!("SET diskann.query_rescore = {}", rescore), &[])
            .await?;
    }

    // Set HNSW query parameters if provided
    if let Some(ef_search) = params.hnsw.ef_search {
        client
            .execute(&format!("SET hnsw.ef_search = {}", ef_search), &[])
            .await?;
    }

    // Set IVFFlat query parameters if provided
    if let Some(probes) = params.ivfflat.probes {
        client
            .execute(&format!("SET ivfflat.probes = {}", probes), &[])
            .await?;
    }

    // Get the appropriate operator for the distance metric
    let distance_operator = params.distance_metric.get_operator();

    // Format the vector for PostgreSQL
    let vector_str = format_vector_for_postgres(&params.query_vector);

    // Construct the SQL query with the vector literal directly in the query
    let query = format!(
        "SELECT id FROM {} ORDER BY embedding {} '{}' LIMIT {}",
        params.table_name, distance_operator, vector_str, params.k
    );

    // Run the query
    let rows = client.query(&query, &[]).await?;

    let result = rows.iter().map(|row| row.get::<_, i32>(0)).collect();

    Ok(result)
}

fn calculate_recall(actual: &[i32], expected: &[i32], k: usize, verbose: bool) -> f64 {
    // Ensure we're not trying to take more items than available
    let effective_k = std::cmp::min(k, expected.len());

    if effective_k == 0 {
        // No ground truth data available
        return 0.0;
    }

    let actual_set: std::collections::HashSet<_> = actual.iter().cloned().collect();
    let expected_set: std::collections::HashSet<_> =
        expected.iter().take(effective_k).cloned().collect();

    // Debug output for first few queries if verbose is enabled
    if verbose {
        static DEBUG_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let debug_idx = DEBUG_COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if debug_idx < 3 {
            println!("Debug for query #{}", debug_idx + 1);
            println!(
                "  Actual results (first 5): {:?}",
                actual.iter().take(5).collect::<Vec<_>>()
            );
            println!(
                "  Expected results (first 5): {:?}",
                expected.iter().take(5).collect::<Vec<_>>()
            );
            println!(
                "  Intersection count: {}",
                expected_set.intersection(&actual_set).count()
            );
            println!("  Effective k: {}", effective_k);
        }
    }

    let intersection_count = expected_set.intersection(&actual_set).count();
    intersection_count as f64 / effective_k as f64
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
            distance: "angular".to_string(),
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
            distance: "angular".to_string(),
            url: "http://ann-benchmarks.com/glove-25-angular.hdf5".to_string(),
        },
        DatasetInfo {
            name: "glove-50-angular".to_string(),
            dimensions: 50,
            train_size: 1183514,
            test_size: 10000,
            neighbors: 100,
            distance: "angular".to_string(),
            url: "http://ann-benchmarks.com/glove-50-angular.hdf5".to_string(),
        },
        DatasetInfo {
            name: "glove-100-angular".to_string(),
            dimensions: 100,
            train_size: 1183514,
            test_size: 10000,
            neighbors: 100,
            distance: "angular".to_string(),
            url: "http://ann-benchmarks.com/glove-100-angular.hdf5".to_string(),
        },
        DatasetInfo {
            name: "glove-200-angular".to_string(),
            dimensions: 200,
            train_size: 1183514,
            test_size: 10000,
            neighbors: 100,
            distance: "angular".to_string(),
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
            distance: "angular".to_string(),
            url: "http://ann-benchmarks.com/nytimes-256-angular.hdf5".to_string(),
        },
        DatasetInfo {
            name: "lastfm-64-dot".to_string(),
            dimensions: 65,
            train_size: 292385,
            test_size: 50000,
            neighbors: 100,
            distance: "angular".to_string(),
            url: "http://ann-benchmarks.com/lastfm-64-dot.hdf5".to_string(),
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
    println!("{:-<80}", "");

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

/// List locally cached datasets
async fn list_cached_datasets() -> Result<(), Box<dyn std::error::Error>> {
    println!("Listing locally cached datasets:");

    // Get the cache directory where datasets are stored
    let cache_dir = get_dataset_cache_dir()?;

    // List all HDF5 files in the cache directory
    let mut found_datasets = false;

    if cache_dir.exists() {
        for entry in std::fs::read_dir(cache_dir)? {
            let entry = entry?;
            let path = entry.path();

            // Check if the file has the .hdf5 extension
            if let Some(extension) = path.extension() {
                if extension == "hdf5" {
                    if let Some(filename) = path.file_stem() {
                        if let Some(name) = filename.to_str() {
                            println!("  - {}", name);
                            found_datasets = true;
                        }
                    }
                }
            }
        }
    }

    if !found_datasets {
        println!("  No cached datasets found in ~/.pgvectorscale/datasets");
        println!("  Use the DownloadAndLoad command to download and cache datasets");
    }

    Ok(())
}

/// Download a dataset from ann-benchmarks
async fn download_dataset(dataset: &DatasetInfo) -> Result<PathBuf, Box<dyn std::error::Error>> {
    // Get the cache directory
    let cache_dir = get_dataset_cache_dir()?;
    let file_path = cache_dir.join(format!("{}.hdf5", dataset.name));

    // Check if the file already exists
    if file_path.exists() {
        println!("Dataset already exists at: {}", file_path.display());
        return Ok(file_path);
    }

    println!("Downloading dataset from: {}", dataset.url);

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

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match &cli.command {
        Commands::ListDatasets => list_ann_benchmark_datasets().await?,

        Commands::ListCachedDatasets => list_cached_datasets().await?,

        Commands::DownloadAndLoad {
            dataset,
            table,
            create_table,
            num_vectors,
            transactions,
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
        } => {
            // Get dataset information
            let datasets = get_ann_benchmark_datasets().await?;
            let dataset_info = datasets.iter().find(|d| d.name == *dataset);

            if let Some(dataset_info) = dataset_info {
                println!(
                    "Found dataset: {} ({}D, {} distance)",
                    dataset_info.name, dataset_info.dimensions, dataset_info.distance
                );

                // Download the dataset
                let file_path = download_dataset(dataset_info).await?;

                // Infer the distance metric
                let inferred_distance_metric = match dataset_info.distance.as_str() {
                    "angular" => DistanceType::Cosine,
                    "euclidean" => DistanceType::L2,
                    "dot" => DistanceType::InnerProduct,
                    _ => {
                        println!(
                            "Warning: Unknown distance metric '{}', defaulting to Cosine",
                            dataset_info.distance
                        );
                        DistanceType::Cosine
                    }
                };

                // Use the provided distance metric or the inferred one
                let effective_distance_metric = distance_metric.unwrap_or(inferred_distance_metric);

                println!("Using distance metric: {:?}", effective_distance_metric);

                // Open the HDF5 file
                let h5_file = File::open(&file_path)?;

                // Get the dataset
                let dataset = h5_file.dataset("train")?;

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

                // Read the vectors from the HDF5 file
                let vectors_data: Vec<f32> = dataset.read_raw::<f32>()?;
                let vectors = Array::from_shape_vec((total_vectors, vector_dim), vectors_data)?
                    .slice(s![0..vectors_to_load, ..])
                    .to_owned();

                // Connect to PostgreSQL
                let mut client = connect_to_postgres(&cli.connection_string).await?;

                // Create the table if requested
                if *create_table {
                    create_vector_table(&client, table, vector_dim, true).await?;
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
                        let table_name = table.clone();
                        let vectors_clone = vectors.clone();
                        let pb_clone = progress_bar.clone();

                        let handle = task::spawn(async move {
                            let mut client = connect_to_postgres(&connection_string).await.unwrap();

                            load_vectors(
                                &mut client,
                                &table_name,
                                &vectors_clone,
                                start_idx,
                                end_idx,
                                Some(pb_clone),
                            )
                            .await
                            .unwrap();
                        });

                        handles.push(handle);
                    }

                    // Wait for all tasks to complete
                    join_all(handles).await;
                } else if *transactions > 1 {
                    // Split loading into multiple transactions
                    let vectors_per_transaction = vectors_to_load / *transactions;

                    for i in 0..*transactions {
                        let start_idx = i * vectors_per_transaction;
                        let end_idx = if i == *transactions - 1 {
                            vectors_to_load
                        } else {
                            (i + 1) * vectors_per_transaction
                        };

                        load_vectors(
                            &mut client,
                            table,
                            &vectors,
                            start_idx,
                            end_idx,
                            Some(progress_bar.clone()),
                        )
                        .await?;
                    }
                } else {
                    // Single transaction loading
                    let res = load_vectors(
                        &mut client,
                        table,
                        &vectors,
                        0,
                        vectors_to_load,
                        Some(progress_bar.clone()),
                    )
                    .await;
                    if res.is_err() {
                        println!("Error loading vectors: {:?}", res);
                        res?;
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
                        format!("{:?}", effective_distance_metric).to_lowercase()
                    );
                    let index_start_time = Instant::now();

                    // Create index parameters struct
                    let index_params = IndexParams {
                        index_type: *index_type,
                        distance_metric: effective_distance_metric,
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

                    create_vector_index(&client, table, vector_dim, &index_params).await?;

                    let index_duration = index_start_time.elapsed();
                    let index_stats = PerformanceStats::new("Index Creation", index_duration, 1);
                    index_stats.print();
                }
            } else {
                println!("Dataset '{}' not found. Use the ListDatasets command to see available datasets.", dataset);
            }
        }
        Commands::Load {
            file,
            dataset,
            table,
            create_table,
            num_vectors,
            transactions,
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
        } => {
            // Open the HDF5 file
            let h5_file = File::open(file)?;

            // Get the dataset
            let dataset = h5_file.dataset(dataset)?;

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

            // Read the vectors from the HDF5 file
            let vectors_data: Vec<f32> = dataset.read_raw::<f32>()?;
            let vectors = Array::from_shape_vec((total_vectors, vector_dim), vectors_data)?
                .slice(s![0..vectors_to_load, ..])
                .to_owned();

            // Connect to PostgreSQL
            let mut client = connect_to_postgres(&cli.connection_string).await?;

            // Create the table if requested
            if *create_table {
                create_vector_table(&client, table, vector_dim, true).await?;
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
                    let table_name = table.clone();
                    let vectors_clone = vectors.clone();
                    let pb_clone = progress_bar.clone();

                    let handle = task::spawn(async move {
                        let mut client = connect_to_postgres(&connection_string).await.unwrap();

                        load_vectors(
                            &mut client,
                            &table_name,
                            &vectors_clone,
                            start_idx,
                            end_idx,
                            Some(pb_clone),
                        )
                        .await
                        .unwrap();
                    });

                    handles.push(handle);
                }

                // Wait for all tasks to complete
                join_all(handles).await;
            } else if *transactions > 1 {
                // Split loading into multiple transactions
                let vectors_per_transaction = vectors_to_load / *transactions;

                for i in 0..*transactions {
                    let start_idx = i * vectors_per_transaction;
                    let end_idx = if i == *transactions - 1 {
                        vectors_to_load
                    } else {
                        (i + 1) * vectors_per_transaction
                    };

                    load_vectors(
                        &mut client,
                        table,
                        &vectors,
                        start_idx,
                        end_idx,
                        Some(progress_bar.clone()),
                    )
                    .await?;
                }
            } else {
                // Single transaction loading
                let res = load_vectors(
                    &mut client,
                    table,
                    &vectors,
                    0,
                    vectors_to_load,
                    Some(progress_bar.clone()),
                )
                .await;
                if res.is_err() {
                    println!("Error loading vectors: {:?}", res);
                    res?;
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

                create_vector_index(&client, table, vector_dim, &index_params).await?;

                let index_duration = index_start_time.elapsed();
                let index_stats = PerformanceStats::new("Index Creation", index_duration, 1);
                index_stats.print();
            }
        }

        Commands::Test {
            file,
            query_dataset,
            neighbors_dataset,
            table,
            num_queries,
            k,
            distance_metric,
            verbose,
            diskann_query_search_list_size,
            diskann_query_rescore,
            hnsw_ef_search,
            ivfflat_probes,
        } => {
            // Open the HDF5 file
            let h5_file = File::open(file)?;

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
            let effective_k = std::cmp::min(*k, neighbors_shape[1]);

            // Print debug info about the ground truth data if verbose is enabled
            if *verbose {
                println!("Ground truth shape: {:?}", neighbors_shape);
                println!(
                    "Number of neighbors in ground truth: {}",
                    neighbors_shape[1]
                );
                println!("Requested k: {}", k);

                if effective_k < *k {
                    println!("Warning: Requested k={} but ground truth only has {} neighbors. Using k={}.", 
                             k, neighbors_shape[1], effective_k);
                }
            }

            let ground_truth =
                Array::from_shape_vec((total_queries, neighbors_shape[1]), neighbors_data)?
                    .slice(s![0..queries_to_run, 0..effective_k])
                    .to_owned();

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

                // Measure the query time
                let query_start = Instant::now();
                // Create query parameters struct
                let query_params = QueryParams {
                    table_name: table.clone(),
                    query_vector: query_vector_slice.to_vec(),
                    k: effective_k,
                    distance_metric: *distance_metric,
                    diskann: DiskAnnQueryParams {
                        query_search_list_size: *diskann_query_search_list_size,
                        query_rescore: *diskann_query_rescore,
                    },
                    hnsw: HnswQueryParams {
                        ef_search: *hnsw_ef_search,
                    },
                    ivfflat: IvfFlatQueryParams {
                        probes: *ivfflat_probes,
                    },
                };

                let result = run_query(&client, &query_params).await?;
                let query_duration = query_start.elapsed();
                query_stats.add_query_time(query_duration);

                // Calculate recall for this query
                let ground_truth_row = ground_truth.row(i);
                let expected_slice = ground_truth_row.as_slice().unwrap();
                let recall = calculate_recall(&result, expected_slice, effective_k, *verbose);
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
                println!("K (nearest neighbors): {} (effective: {})", k, effective_k);
                println!("Distance metric: {}", distance_metric);

                // Print individual recall values (limit to first 10 if there are many)
                if queries_to_run <= 10 {
                    for (i, recall) in recall_values.iter().enumerate() {
                        println!("Query {}: Recall@{} = {:.4}", i + 1, k, recall);
                    }
                } else {
                    for (i, &recall) in recall_values.iter().enumerate().take(5) {
                        println!("Query {}: Recall@{} = {:.4}", i + 1, k, recall);
                    }
                    println!("...");
                    // Get the last 5 queries
                    for i in (queries_to_run - 5)..queries_to_run {
                        println!("Query {}: Recall@{} = {:.4}", i + 1, k, recall_values[i]);
                    }
                }

                println!();
            }

            println!("Average recall@{}: {:.4}", k, avg_recall);

            // Print detailed query time statistics
            query_stats.print();

            // Report overall performance statistics
            let stats = PerformanceStats::new("Query Testing", duration, queries_to_run);
            stats.print();
        }
    }

    Ok(())
}
