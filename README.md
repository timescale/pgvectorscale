Timescale Vector

Say something chat gpt.

🔧 Tools Setup
Building the extension requires valid rust (we build and test on 1.65), rustfmt, and clang installs, along with the postgres headers for whichever version of postgres you are running, and pgx. We recommend installing rust using the official instructions:
```shell
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```
and build tools, the postgres headers, in the preferred manner for your system. You may also need to install OpenSSl. For Ubuntu you can follow the postgres install instructions then run

```shell
sudo apt-get install make gcc pkg-config clang postgresql-server-dev-15 libssl-dev
```

Next you need cargo-pgx, which can be installed with
```shell
cargo install --locked cargo-pgrx
```

You must reinstall cargo-pgx whenever you update your Rust compiler, since cargo-pgx needs to be built with the same compiler as Toolkit.

Finally, setup the pgx development environment with
```shell
cargo pgrx init --pg15 pg_config
```

Installing from source is also available on macOS and requires the same set of prerequisites and set up commands listed above.

💾 Building and Installing the extension
Download or clone this repository, and switch to the extension subdirectory, e.g.
```shell
git clone https://github.com/timescale/timescale-vector && \
cd timescale-vector/extension
```

Then run
```shell
cargo pgrx install --release
```

To initialize the extension after installation, enter the following into psql:

CREATE EXTENSION timescale_vector;
✏️ Get Involved
The Timescale Vecotr project is still in the initial planning stage as we decide our priorities and what to implement first. As such, now is a great time to help shape the project's direction! Have a look at the list of features we're thinking of working on and feel free to comment on the features, expand the list, or hop on the Discussions forum for more in-depth discussions.

🔨 Testing
See above for prerequisites and installation instructions.

You can run tests against a postgres version pg15 using
```shell
cargo pgrx test ${postgres_version}
```
🐯 About Timescale
TimescaleDB is a distributed time-series database built on PostgreSQL that scales to over 10 million of metrics per second, supports native compression, handles high cardinality, and offers native time-series capabilities, such as data retention policies, continuous aggregate views, downsampling, data gap-filling and interpolation.

TimescaleDB also supports full SQL, a variety of data types (numerics, text, arrays, JSON, booleans), and ACID semantics. Operationally mature capabilities include high availability, streaming backups, upgrades over time, roles and permissions, and security.

TimescaleDB has a large and active user community (tens of millions of downloads, hundreds of thousands of active deployments, Slack channels with thousands of members).