# pgvectorscale

A vector index for speeding up ANN search in `pgvector`.

## 💾 Building and Installing pgvectorscale

### From source

#### Prerequisites

Building the extension requires valid rust, along with the postgres headers for whichever version of postgres you are running, and pgrx. We recommend installing rust using the official instructions:
```shell
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```
 
You should install the appropriate build tools and postgres headers in the preferred manner for your system. You may also need to install OpenSSL. For Ubuntu you can follow the postgres install instructions then run

```shell
sudo apt-get install make gcc pkg-config clang postgresql-server-dev-16 libssl-dev
```

Next you need cargo-pgrx, which can be installed with
```shell
cargo install --locked cargo-pgrx
```

You must reinstall cargo-pgrx whenever you update your Rust compiler, since cargo-pgrx needs to be built with the same compiler as pgvectorscale.

Finally, setup the pgrx development environment with
```shell
cargo pgrx init --pg16 pg_config
```

#### Building and installing the extension 

Download or clone this repository, and switch to the extension subdirectory, e.g.
```shell
git clone https://github.com/timescale/pgvectorscale && \
cd pgvectorscale/pgvectorscale
```

Then run
```shell
cargo pgrx install --release
```

To initialize the extension after installation, enter the following into psql:

```sql
CREATE EXTENSION vectorscale;
```

## ✏️ Get Involved

The pgvectorscale project is still in it's early stage as we decide our priorities and what to implement. As such, now is a great time to help shape the project's direction! Have a look at the list of features we're thinking of working on and feel free to comment on the features, expand the list, or hop on the Discussions forum for more in-depth discussions.

### 🔨 Testing
See above for prerequisites and installation instructions.

You can run tests against a postgres version pg16 using
```shell
cargo pgrx test ${postgres_version}
```

To run all tests run:
```shell
cargo test -- --ignored && cargo pgrx test ${postgres_version}
```

### 🐯 About Timescale

TimescaleDB is a distributed time-series database built on PostgreSQL that scales to over 10 million of metrics per second, supports native compression, handles high cardinality, and offers native time-series capabilities, such as data retention policies, continuous aggregate views, downsampling, data gap-filling and interpolation.

TimescaleDB also supports full SQL, a variety of data types (numerics, text, arrays, JSON, booleans), and ACID semantics. Operationally mature capabilities include high availability, streaming backups, upgrades over time, roles and permissions, and security.

TimescaleDB has a large and active user community (tens of millions of downloads, hundreds of thousands of active deployments, Slack channels with thousands of members).