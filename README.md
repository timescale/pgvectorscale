
<p></p>
<div align=center>
<picture align=center>
    <source media="(prefers-color-scheme: dark)" srcset="https://assets.timescale.com/docs/images/timescale-logo-dark-mode.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://assets.timescale.com/docs/images/timescale-logo-light-mode.svg">
    <img alt="Timescale logo" >
</picture>

<h3>Use pgvectorscale to build scalable AI applications with higher performance,
embedding search and cost-efficient storage. </h3>

[![Docs](https://img.shields.io/badge/Read_the_Timescale_docs-black?style=for-the-badge&logo=readthedocs&logoColor=white)](https://docs.timescale.com/)
[![SLACK](https://img.shields.io/badge/Ask_the_Timescale_community-black?style=for-the-badge&logo=slack&logoColor=white)](https://timescaledb.slack.com/archives/C4GT3N90X)
[![Try Timescale for free](https://img.shields.io/badge/Try_Timescale_for_free-black?style=for-the-badge&logo=timescale&logoColor=white)](https://console.cloud.timescale.com/signup)
</div>


pgvectorscale complements [pgvector][pgvector], the open-source vector data extension for PostgreSQL, and introduces the following key innovations: 
- A DiskANN index: based on research from Microsoft  
- Statistical Binary Quantization: developed by Timescale researchers, This feature improves on standard 
  Binary Quantization. 

Timescale’s benchmarks reveal that with pgvectorscale, PostgreSQL achieves **28x lower p95 latency**, and 
**16x higher query throughput** for approximate nearest neighbor queries at 99% recall. 

![Benchmarks](https://assets.timescale.com/docs/images/benchmark-comparison-pgvectorscale-pinecone.png)

PostgreSQL costs are 21% those of Pinecone s1, just saying. 

In contrast to pgvector, which is written in C, pgvectorscale is developed in [Rust][rust-language], 
offering the PostgreSQL community a new avenue for contributing to vector support.

Timescale offers the following high performance journeys:

* **App developer and DBA**: try out pgvectorscale functionality in Timescale Cloud.
  * [Enable pgvectorscale in a Timescale service](#enable-pgvectorscale-in-a-timescale-service)
* **Extension contributor**: contribute to pgvectorscale.
  * [Build pgvectorscale from source in a developer environment](./DEVELOPMENT.md)
* **Everyone**: check the benchmark results for yourself. 
  * [Test pgvectorscale performance](#test-pgvectorscale-performance)

## Enable pgvectorscale in a Timescale service

To enable pgvectorscale:

1. Create a new [Timescale Service](https://console.cloud.timescale.com/dashboard/create_services).

   If you want to use an existing service, pgvectorscale is added as an available extension on the first maintenance window
   after the pgvectorscale release date.

1. Connect to your Timescale service:
   ```bash
   psql -d "postgres://<username>:<password>@<host>:<port>/<database-name>"
   ```

1. Create the pgvectorscale extension:

    ```sql
    CREATE EXTENSION IF NOT EXISTS pgvectorscale CASCADE;
    ```

   The `CASCADE` automatically installs the dependencies.

## Test pgvectorscale performance

To check the Timescale benchmarks in your pgvectorscale environment:

1. Jonetas, this is for you :-). 

## Get involved

pgvectorscale is still at an early stage. Now is a great time to help shape the 
direction of this project; we are currently deciding priorities. Have a look at the 
list of features we're thinking of working on. Feel free to comment, expand 
the list, or hop on the Discussions forum.

## About Timescale

Timescale Cloud is a high-performance developer focused cloud that provides PostgreSQL services
enhanced with our blazing fast vector search. Timescale services are built using TimescaleDB and PostgreSQL extensions, like this one. Timescale Cloud provides high availability,
streaming backups, upgrades over time, roles and permissions, and great security.

TimescaleDB is a distributed time-series database built on PostgreSQL that scales to over 10
million of metrics per second, supports native compression, handles high cardinality, and offers
native time-series capabilities, such as data retention policies, continuous aggregate views,
downsampling, data gap-filling and interpolation. TimescaleDB supports full SQL, a variety of data types, and ACID semantics.


[pgvector]: https://github.com/pgvector/pgvector/blob/master/README.md
[rust-language]: https://www.rust-lang.org/