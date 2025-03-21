<p></p>
<div align=center>

# pgvectorscale

<h3>pgvectorscale builds on pgvector with higher performance embedding search and cost-efficient storage for AI applications. </h3>

[![Discord](https://img.shields.io/badge/Join_us_on_Discord-black?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/KRdHVXAmkp)
[![Try Timescale for free](https://img.shields.io/badge/Try_Timescale_for_free-black?style=for-the-badge&logo=timescale&logoColor=white)](https://tsdb.co/gh-pgvector-signup)
</div>

pgvectorscale complements [pgvector][pgvector], the open-source vector data extension for PostgreSQL, and introduces the following key innovations for pgvector data:
- A new index type called StreamingDiskANN, inspired by the [DiskANN](https://github.com/microsoft/DiskANN) algorithm, based on research from Microsoft.
- Statistical Binary Quantization: developed by Timescale researchers, This compression method improves on standard Binary Quantization.
- Label-based filtered vector search: based on Microsoft's Filtered DiskANN research, this allows you to combine vector similarity search with label filtering for more precise and efficient results.

On a benchmark dataset of 50 million Cohere embeddings with 768 dimensions
each, PostgreSQL with `pgvector` and `pgvectorscale` achieves **28x lower p95
latency** and **16x higher query throughput** compared to Pinecone's storage
optimized (s1) index for approximate nearest neighbor queries at 99% recall,
all at 75% less cost when self-hosted on AWS EC2.

<div align=center>

![Benchmarks](https://assets.timescale.com/docs/images/benchmark-comparison-pgvectorscale-pinecone.png)

</div>

To learn more about the performance impact of pgvectorscale, and details about benchmark methodology and results, see the [pgvector vs Pinecone comparison blog post](http://www.timescale.com/blog/pgvector-vs-pinecone).

In contrast to pgvector, which is written in C, pgvectorscale is developed in [Rust][rust-language] using the [PGRX framework](https://github.com/pgcentralfoundation/pgrx),
offering the PostgreSQL community a new avenue for contributing to vector support.

**Application developers or DBAs** can use pgvectorscale with their PostgreSQL databases.
   * [Install pgvectorscale](#installation)
   * [Get started using pgvectorscale](#get-started-with-pgvectorscale)

If you **want to contribute** to this extension, see how to [build pgvectorscale from source in a developer environment](./DEVELOPMENT.md).

For production vector workloads, get **private beta access to vector-optimized databases** with pgvector and pgvectorscale on Timescale. [Sign up here for priority access](https://timescale.typeform.com/to/H7lQ10eQ).

## Installation

The fastest ways to run PostgreSQL with pgvectorscale are:

* [Using a pre-built Docker container](#using-a-pre-built-docker-container)
* [Installing from source](#installing-from-source)
* [Enable pgvectorscale in a Timescale Cloud service](#enable-pgai-in-a-timescale-cloud-service)

### Using a pre-built Docker container

1.  [Run the TimescaleDB Docker image](https://docs.timescale.com/self-hosted/latest/install/installation-docker/).

1. Connect to your database:
   ```bash
   psql -d "postgres://<username>:<password>@<host>:<port>/<database-name>"
   ```

1. Create the pgvectorscale extension:

    ```sql
    CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;
    ```

   The `CASCADE` automatically installs `pgvector`.

### Installing from source

You can install pgvectorscale from source and install it in an existing PostgreSQL server

> [!WARNING]
> Building pgvectorscale on macOS X86 (Intel) machines is currently not
> supported due to an [open issue][macos-x86-issue]. As alternatives, you can:
>
> - Use an ARM-based Mac.
> - Build using Linux.
> - Use our pre-built Docker containers.
>
> We welcome community contributions to resolve this limitation. If you're
> interested in helping, please check the issue for details.

1. Compile and install the extension

    ```bash
    # install rust
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

    # download pgvectorscale
    cd /tmp
    git clone --branch <version> https://github.com/timescale/pgvectorscale
    cd pgvectorscale/pgvectorscale
    # install cargo-pgrx with the same version as pgrx
    cargo install --locked cargo-pgrx --version $(cargo metadata --format-version 1 | jq -r '.packages[] | select(.name == "pgrx") | .version')
    cargo pgrx init --pg17 pg_config
    # build and install pgvectorscale
    cargo pgrx install --release
    ```

    You can also take a look at our [documentation for extension developers](./DEVELOPMENT.md) for more complete instructions.

1. Connect to your database:
   ```bash
   psql -d "postgres://<username>:<password>@<host>:<port>/<database-name>"
   ```

1. Ensure the pgvector extension is available:

   ```sql
   SELECT * FROM pg_available_extensions WHERE name = 'vector';
   ```

   If pgvector is not available, install it using the [pgvector installation
   instructions][pgvector-install].


1. Create the pgvectorscale extension:

    ```sql
    CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;
    ```

   The `CASCADE` automatically installs `pgvector`.

### Enable pgvectorscale in a Timescale Cloud service

Note: the instructions below are for Timescale's standard compute instance. For production vector workloads, we’re offering **private beta access to vector-optimized databases** with pgvector and pgvectorscale on Timescale. [Sign up here for priority access](https://timescale.typeform.com/to/H7lQ10eQ).

To enable pgvectorscale:

1. Create a new [Timescale Service](https://console.cloud.timescale.com/signup?utm_campaign=vectorlaunch).

   If you want to use an existing service, pgvectorscale is added as an available extension on the first maintenance window
   after the pgvectorscale release date.

1. Connect to your Timescale service:
   ```bash
   psql -d "postgres://<username>:<password>@<host>:<port>/<database-name>"
   ```

1. Create the pgvectorscale extension:

    ```postgresql
    CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;
    ```

   The `CASCADE` automatically installs `pgvector`.


## Get started with pgvectorscale


1. Create a table with an embedding column. For example:

    ```postgresql
    CREATE TABLE IF NOT EXISTS document_embedding  (
        id BIGINT PRIMARY KEY GENERATED BY DEFAULT AS IDENTITY,
        metadata JSONB,
        contents TEXT,
        embedding VECTOR(1536)
    )
    ```

1. Populate the table.

   For more information, see the [pgvector instructions](https://github.com/pgvector/pgvector/blob/master/README.md#storing) and [list of clients](https://github.com/pgvector/pgvector/blob/master/README.md#languages).
1. Create a StreamingDiskANN index on the embedding column:
    ```postgresql
    CREATE INDEX document_embedding_idx ON document_embedding
    USING diskann (embedding vector_cosine_ops);
    ```
1. Find the 10 closest embeddings using the index.

    ```postgresql
    SELECT *
    FROM document_embedding
    ORDER BY embedding <=> $1
    LIMIT 10;
    ```

    Note: pgvectorscale currently supports: cosine distance (`<=>`) queries, for indices created with `vector_cosine_ops`; L2 distance (`<->`) queries, for indices created with `vector_l2_ops`; and inner product (`<#>`) queries, for indices created with `vector_ip_ops`.  This is the same syntax used by `pgvector`.  If you would like additional distance types,
    [create an issue](https://github.com/timescale/pgvectorscale/issues).  (Note: inner product indices are not compatible with plain storage.)

## Filtered Vector Search

pgvectorscale supports combining vector similarity search with metadata filtering. There are two basic kinds of filtering, which can be combined in a single query:

1. **Label-based filtering with the diskann index**: This provides optimized performance for filtering by labels.
2. **Arbitrary WHERE clause filtering**: This uses post-filtering after the vector search.

The label-based filtering implementation is based on the [Filtered DiskANN](https://dl.acm.org/doi/10.1145/3543507.3583552) approach developed by Microsoft researchers, which enables efficient filtered vector search while maintaining high recall.

The post-filtering implementation, while slower, is streaming and correct, ensuring accurate results without requiring the entire result set to be loaded into memory.

### Label-based Filtering with diskann

For optimal performance with label filtering, you must specify the label column directly in the index creation:

1. Create a table with an embedding column and a labels array:

    ```postgresql
    CREATE TABLE documents (
        id SERIAL PRIMARY KEY,
        embedding VECTOR(1536),
        labels SMALLINT[],  -- Array of category labels
        status TEXT,
        created_at TIMESTAMPTZ
    );
    ```

2. Create a StreamingDiskANN index on the embedding column, including the labels column:

    ```postgresql
    CREATE INDEX ON documents USING diskann (embedding vector_cosine_ops, labels);
    ```

> **Note**: Label values must be within the PostgreSQL `smallint` range (-32768 to 32767). Using `smallint[]` for labels ensures that PostgreSQL's type system will automatically enforce these bounds.
> 
> pgvectorscale includes an implementation of the `&&` overlap operator for `smallint[]` arrays, which is used for efficient label-based filtering.

3. Perform label-filtered vector searches using the `&&` operator (array overlap):

    ```postgresql
    -- Find similar documents with specific labels
    SELECT * FROM documents
    WHERE labels && ARRAY[1, 3]  -- Documents with label 1 OR 3
    ORDER BY embedding <=> '[...]'
    LIMIT 10;
    ```

    The index directly supports this type of filtering, providing significantly lower latency results compared to post-filtering.

#### Giving Semantic Meaning to Labels

While the labels must be stored as integers in the array for the index to work efficiently, you can give them semantic meaning by relating them to a separate labels table:

1. Create a labels table with meaningful descriptions:

    ```postgresql
    CREATE TABLE label_definitions (
        id INTEGER PRIMARY KEY,
        name TEXT,
        description TEXT,
        attributes JSONB  -- Can store additional metadata about the label
    );

    -- Insert some label definitions
    INSERT INTO label_definitions (id, name, description, attributes) VALUES
    (1, 'science', 'Scientific content', '{"domain": "academic", "confidence": 0.95}'),
    (2, 'technology', 'Technology-related content', '{"domain": "technical", "confidence": 0.92}'),
    (3, 'business', 'Business and finance content', '{"domain": "commercial", "confidence": 0.88}');
    ```

2. When inserting documents, use the appropriate label IDs:

    ```postgresql
    -- Insert a document with science and technology labels
    INSERT INTO documents (embedding, labels)
    VALUES ('[...]', ARRAY[1, 2]);
    ```

3. When querying, you can join with the labels table to work with meaningful names:

    ```postgresql
    -- Find similar science documents and include label information
    SELECT d.*, array_agg(l.name) as label_names
    FROM documents d
    JOIN label_definitions l ON l.id = ANY(d.labels)
    WHERE d.labels && ARRAY[1]  -- Science label
    GROUP BY d.id, d.embedding, d.labels, d.status, d.created_at
    ORDER BY d.embedding <=> '[...]'
    LIMIT 10;
    ```

4. You can also convert between label names and IDs when filtering:

    ```postgresql
    -- Find documents with specific label names
    SELECT d.*
    FROM documents d
    WHERE d.labels && (
        SELECT array_agg(id)
        FROM label_definitions
        WHERE name IN ('science', 'business')
    )
    ORDER BY d.embedding <=> '[...]'
    LIMIT 10;
    ```

This approach gives you the performance benefits of integer-based label filtering while still allowing you to work with semantically meaningful labels in your application.

### Arbitrary WHERE Clause Filtering

You can also use any PostgreSQL WHERE clause with vector search, but these conditions will be applied as post-filtering:

```postgresql
-- Find similar documents with specific status and date range
SELECT * FROM documents
WHERE status = 'active' AND created_at > '2024-01-01'
ORDER BY embedding <=> '[...]'
LIMIT 10;
```

For these arbitrary conditions, the vector search happens first, and then the WHERE conditions are applied to the results. For best performance with frequently used filters, consider using the label-based approach described above.

## Tuning

The StreamingDiskANN index comes with **smart defaults** but also the ability to customize its behavior. There are two types of parameters: index build-time parameters that are specified when an index is created and query-time parameters that can be tuned when querying an index.

We suggest setting the index build-time paramers for major changes to index operations while query-time parameters can be used to tune the accuracy/performance tradeoff for individual queries.

 We expect most people to tune the query-time parameters (if any) and leave the index build time parameters set to default.

### StreamingDiskANN index build-time parameters

These parameters can be set when an index is created.

| Parameter name   | Description                                                                                                                                                    | Default value |
|------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------|
| `storage_layout` | `memory_optimized` which uses SBQ to compress vector data or `plain` which stores data uncompressed | memory_optimized
| `num_neighbors`    | Sets the maximum number of neighbors per node. Higher values increase accuracy but make the graph traversal slower.                                           | 50            |
| `search_list_size` | This is the S parameter used in the greedy search algorithm used during construction. Higher values improve graph quality at the cost of slower index builds. | 100           |
| `max_alpha`        | Is the alpha parameter in the algorithm. Higher values improve graph quality at the cost of slower index builds.                                              | 1.2           |
| `num_dimensions` | The number of dimensions to index. By default, all dimensions are indexed. But you can also index less dimensions to make use of [Matryoshka embeddings](https://huggingface.co/blog/matryoshka) | 0 (all dimensions)
| `num_bits_per_dimension` | Number of bits used to encode each dimension when using SBQ | 2 for less than 900 dimensions, 1 otherwise

An example of how to set the `num_neighbors` parameter is:

```sql
CREATE INDEX document_embedding_idx ON document_embedding
USING diskann (embedding) WITH(num_neighbors=50);
```

An example of creating an index with label-based filtering:

```sql
CREATE INDEX document_embedding_idx ON document_embedding
USING diskann (embedding vector_cosine_ops, labels);
```

#### StreamingDiskANN query-time parameters

You can also set two parameters to control the accuracy vs. query speed trade-off at query time. We suggest adjusting `diskann.query_rescore` to fine-tune accuracy.

| Parameter name   | Description                                                                                                                                                    | Default value |
|------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------|
| `diskann.query_search_list_size` | The number of additional candidates considered during the graph search. | 100
| `diskann.query_rescore` | The number of elements rescored (0 to disable rescoring) | 50


You can set the value by using `SET` before executing a query. For example:

```sql
SET diskann.query_rescore = 400;
```

Note the [SET command](https://www.postgresql.org/docs/current/sql-set.html) applies to the entire session (database connection) from the point of execution. You can use a transaction-local variant using `LOCAL` which will
be reset after the end of the transaction:

```sql
BEGIN;
SET LOCAL diskann.query_search_list_size= 10;
SELECT * FROM document_embedding ORDER BY embedding <=> $1 LIMIT 10
COMMIT;
```

## Null Value Handling

* Null vectors are not indexed
* Null labels are treated as empty arrays
* Null values in label arrays are ignored

## ORDER BY vector distance

pgvectorscale's diskann index uses relaxed ordering which allows results to be
slightly out of order by distance. This is analogous to using
[`iterative scan with relaxed ordering`][pgvector-iterative-index-scan] with
pgvector's ivfflat or hnsw indexes.

If you need strict ordering you can use a [materialized CTE][materialized-cte]:

```sql
WITH relaxed_results AS MATERIALIZED (
    SELECT id, embedding <=> '[1,2,3]' AS distance
    FROM items
    WHERE category_id = 123
    ORDER BY distance
    LIMIT 5
) SELECT * FROM relaxed_results ORDER BY distance;
```

## Index on an UNLOGGED table

Creating an index on an UNLOGGED table is currently not supported.
Trying will yield the error:

```
ERROR:  ambuildempty: not yet implemented
```

## Get involved

pgvectorscale is still at an early stage. Now is a great time to help shape the
direction of this project; we are currently deciding priorities. Have a look at the
list of features we're thinking of working on. Feel free to comment, expand
the list, or hop on the Discussions forum.

## About Timescale

Timescale is a PostgreSQL cloud company. To learn more visit the [timescale.com](https://www.timescale.com).

[Timescale Cloud](https://console.cloud.timescale.com/signup?utm_campaign=vectorlaunch) is a high-performance, developer focused, cloud platform that provides PostgreSQL services for the most demanding AI, time-series, analytics, and event workloads. Timescale Cloud is ideal for production applications and provides high availability, streaming backups, upgrades over time, roles and permissions, and great security.

[pgvector]: https://github.com/pgvector/pgvector/blob/master/README.md
[rust-language]: https://www.rust-lang.org/
[pgvector-install]: https://github.com/pgvector/pgvector?tab=readme-ov-file#installation
[pgvector-iterative-index-scan]: https://github.com/pgvector/pgvector?tab=readme-ov-file#iterative-index-scans
[materialized-cte]: https://www.postgresql.org/docs/current/queries-with.html#QUERIES-WITH-CTE-MATERIALIZATION
[macos-x86-issue]: https://github.com/timescale/pgvectorscale/issues/155
