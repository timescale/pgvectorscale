use std::fmt;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// Import the DistanceType from the vectorscale crate
use vectorscale::access_method::distance::DistanceType;

use clap::{Parser, Subcommand, ValueEnum};
use futures::future::join_all;
use hdf5_metno::File;
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::{s, Array, Array2, ArrayView1};
use tokio::task;
use tokio_postgres::{Client, Error as PgError, NoTls};
use uuid::Uuid;

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
    },
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

        println!("\nQuery Time Histogram (ms):");
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
        println!("\n=== Query Time Statistics (ms) ===");
        println!("Min:       {:8.2}", self.min().as_secs_f64() * 1000.0);
        println!(
            "p25:       {:8.2}",
            self.percentile(0.25).as_secs_f64() * 1000.0
        );
        println!(
            "p50/Median:{:8.2}",
            self.percentile(0.5).as_secs_f64() * 1000.0
        );
        println!(
            "p75:       {:8.2}",
            self.percentile(0.75).as_secs_f64() * 1000.0
        );
        println!(
            "p90:       {:8.2}",
            self.percentile(0.9).as_secs_f64() * 1000.0
        );
        println!(
            "p95:       {:8.2}",
            self.percentile(0.95).as_secs_f64() * 1000.0
        );
        println!(
            "p99:       {:8.2}",
            self.percentile(0.99).as_secs_f64() * 1000.0
        );
        println!("Max:       {:8.2}", self.max().as_secs_f64() * 1000.0);
        println!("Mean:      {:8.2}", self.mean().as_secs_f64() * 1000.0);
        println!(
            "Total:     {:8.2}",
            self.total_duration.as_secs_f64() * 1000.0
        );
        println!("Queries:   {}", self.num_queries);
        println!(
            "QPS:       {:8.2}",
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
        println!("\n=== Performance Statistics ===");
        println!("Operation: {}", self.operation);
        println!("Duration: {:.2} seconds", self.duration.as_secs_f64());
        println!("Items processed: {}", self.items_processed);
        println!("Items per second: {:.2}", self.items_per_second);
        println!("============================");
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
    index_type: IndexType,
    distance_metric: DistanceType,
) -> Result<(), PgError> {
    // Get the operator class for the distance metric
    let operator_class = distance_metric.get_operator_class();

    // Create the index based on the index type
    match index_type {
        IndexType::DiskANN => {
            // DiskANN index
            client
                .execute(
                    &format!(
                        "CREATE INDEX ON {} USING diskann (embedding {});",
                        table_name, operator_class
                    ),
                    &[],
                )
                .await?
        }
        IndexType::Hnsw => {
            // HNSW index
            client
                .execute(
                    &format!(
                        "CREATE INDEX ON {} USING hnsw (embedding {});",
                        table_name, operator_class
                    ),
                    &[],
                )
                .await?
        }
        IndexType::IVFFlat => {
            // IVFFlat index
            client
                .execute(
                    &format!(
                        "CREATE INDEX ON {} USING ivfflat (embedding {});",
                        table_name, operator_class
                    ),
                    &[],
                )
                .await?
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
    use std::fs::File;
    use std::io::{BufWriter, Write};
    use tokio::fs;

    // Generate a unique temporary file name
    let temp_file_name = format!("/tmp/pgvectorscale_copy_{}.csv", Uuid::new_v4());

    // Create and write to the temporary CSV file
    {
        let file = File::create(&temp_file_name).unwrap();
        let mut writer = BufWriter::new(file);

        for i in start_idx..end_idx {
            let vector = vectors.row(i);
            // Format vector for CSV - no need for quotes as COPY will handle it
            let vector_str = format_vector_for_csv(&vector);
            // Include the index as the ID
            writeln!(writer, "{},{}", i, vector_str).unwrap();

            if let Some(pb) = &progress_bar {
                pb.lock().unwrap().inc(1);
            }
        }
        writer.flush().unwrap();
    }

    // Start a transaction
    let transaction = client.transaction().await?;

    // Use COPY command to bulk load the data
    let copy_cmd = format!(
        "COPY {} (id, embedding) FROM '{}' CSV",
        table_name, temp_file_name
    );

    transaction.execute(&copy_cmd, &[]).await?;

    // Commit the transaction
    transaction.commit().await?;

    // Clean up the temporary file
    tokio::spawn(async move {
        let _ = fs::remove_file(temp_file_name).await;
    });

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

async fn run_query(
    client: &Client,
    table_name: &str,
    query_vector: &[f32],
    k: usize,
    distance_metric: DistanceType,
) -> Result<Vec<i32>, PgError> {
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

    // Get the appropriate operator for the distance metric
    let distance_operator = distance_metric.get_operator();

    // Format the vector for PostgreSQL
    let vector_str = format_vector_for_postgres(query_vector);

    // Construct the SQL query with the vector literal directly in the query
    let query = format!(
        "SELECT id FROM {} ORDER BY embedding {} '{}' LIMIT {}",
        table_name, distance_operator, vector_str, k
    );

    // Run the query
    let rows = client.query(&query, &[]).await?;

    let result = rows.iter().map(|row| row.get::<_, i32>(0)).collect();

    Ok(result)
}

fn calculate_recall(actual: &[i32], expected: &[i32], k: usize) -> f64 {
    let actual_set: std::collections::HashSet<_> = actual.iter().cloned().collect();
    let expected_set: std::collections::HashSet<_> = expected.iter().take(k).cloned().collect();

    let intersection_count = expected_set.intersection(&actual_set).count();
    intersection_count as f64 / k as f64
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match &cli.command {
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

                create_vector_index(&client, table, vector_dim, *index_type, *distance_metric)
                    .await?;

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
            let ground_truth =
                Array::from_shape_vec((total_queries, neighbors_shape[1]), neighbors_data)?
                    .slice(s![0..queries_to_run, 0..*k])
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
                let result =
                    run_query(&client, table, query_vector_slice, *k, *distance_metric).await?;
                let query_duration = query_start.elapsed();
                query_stats.add_query_time(query_duration);

                // Calculate recall for this query
                let ground_truth_row = ground_truth.row(i);
                let expected_slice = ground_truth_row.as_slice().unwrap();
                let recall = calculate_recall(&result, expected_slice, *k);
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
                println!("K (nearest neighbors): {}", k);
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
                    for (i, &recall) in recall_values
                        .iter()
                        .enumerate()
                        .skip(queries_to_run - 5)
                        .take(5)
                    {
                        println!(
                            "Query {}: Recall@{} = {:.4}",
                            i + queries_to_run - 5 + 1,
                            k,
                            recall
                        );
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
