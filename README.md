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

##  Testing

Want to see how vast pgvectorscale really is:

- Run tests against a postgres version pg16 using

  ```shell
  cargo pgrx test ${postgres_version}
  ```

- Run all tests:
  ```shell
  cargo test -- --ignored && cargo pgrx test ${postgres_version}
  ```


## Get involved

The pgvectorscale project is still in it's early stage. Now is a great time to help shape the 
directin of this project; we are currently deciding our priorities and what to implement.  
Have a look at the list of features we're thinking of working on. Feel free to comment, expand 
the list, or hop on the Discussions forum for more in-depth discussions.

## About Timescale

TimescaleDB is a distributed time-series database built on PostgreSQL that scales to over 10 million of metrics per second, supports native compression, handles high cardinality, and offers native time-series capabilities, such as data retention policies, continuous aggregate views, downsampling, data gap-filling and interpolation.

TimescaleDB also supports full SQL, a variety of data types (numerics, text, arrays, JSON, booleans), and ACID semantics. Operationally mature capabilities include high availability, streaming backups, upgrades over time, roles and permissions, and security.

TimescaleDB has a large and active user community (tens of millions of downloads, hundreds of thousands of active deployments, Slack channels with thousands of members).


[pgvector]: https://github.com/pgvector/pgvector/blob/master/README.md
[rust-language]: https://www.rust-lang.org/