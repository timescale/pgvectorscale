# pgvectorscale

You use pgvectorscale to build scalable AI applications with higher performance 
embedding search, and cost-efficient storage. 

pgvectorscale complements [pgvector][pgvector], the open-source vector data extension for PostgreSQL, and introduces the following key innovations: 
- A DiskANN index: based on research from Microsoft  
- Statistical Binary Quantization: developed by Timescale researchers, This feature improves on standard 
  Binary Quantization. 

Timescale’s benchmarks reveal that with pgvectorscale, PostgreSQL achieves **28x lower p95 latency**, and 
**16x higher query throughput** for approximate nearest neighbor queries at 99% recall. 

In contrast to pgvector, which is written in C, pgvectorscale is developed in [Rust][rust-language], 
offering the PostgreSQL community a new avenue for contributing to vector support.

## pgvectorscale Prerequisites

* Create a [pgvectorscale developer environment](./DEVELOPMENT.md)

## Get involved

pgvectorscale is still at an early stage. Now is a great time to help shape the 
direction of this project; we are currently deciding priorities. Have a look at the 
list of features we're thinking of working on. Feel free to comment, expand 
the list, or hop on the Discussions forum.

## About Timescale

Timescale Cloud is a high-performance developer focused cloud that provides PostgreSQL services 
enhanced with our blazing fast vector search. Timescale services are built with TimescaleDB and
Timescale PostgreSQL extensions, like this one. Timescale Cloud provides high availability,
streaming backups, upgrades over time, roles and permissions, and security.

TimescaleDB is a distributed time-series database built on PostgreSQL that scales to over 10 
million of metrics per second, supports native compression, handles high cardinality, and offers 
native time-series capabilities, such as data retention policies, continuous aggregate views, 
downsampling, data gap-filling and interpolation.

TimescaleDB supports full SQL, a variety of data types, and ACID semantics.


[pgvector]: https://github.com/pgvector/pgvector/blob/master/README.md
[rust-language]: https://www.rust-lang.org/