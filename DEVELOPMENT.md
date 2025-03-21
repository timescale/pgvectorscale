# Setup your pgvectorscale developer environment

You build pgvectorscale from source, then integrate the extension into each database in your PostgreSQL environment.

## pgvectorscale prerequisites

To create a pgvectorscale developer environment, you need the following on your local machine:

* [PostgreSQL v16](https://docs.timescale.com/self-hosted/latest/install/installation-linux/#install-and-configure-timescaledb-on-postgresql)
* [pgvector](https://github.com/pgvector/pgvector/blob/master/README.md)
* Development packages:
    ```
    sudo apt-get install make gcc pkg-config clang postgresql-server-dev-16 libssl-dev
    ```

* [Rust][rust-language]:
    ```shell
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    ```

## Build and install pgvectorscale on your database

1. In Terminal, clone this repository and switch to the extension subdirectory:

    ```shell
    git clone https://github.com/timescale/pgvectorscale && \
    cd pgvectorscale/pgvectorscale
    ```

1. Install [Cargo-pgrx][cargo-pgrx]:

   ```bash
   cargo install --locked cargo-pgrx --version $(cargo metadata --format-version 1 | jq -r '.packages[] | select(.name == "pgrx") | .version')
   ```
   You must reinstall cargo-pgrx whenever you update Rust, cargo-pgrx must be
   built with the same compiler as pgvectorscale.

1. Initialize the pgrx development environment:

  ```bash
  cargo pgrx init --pg16 pg_config
  ```

1. Build pgvectorscale:

   ```shell
   cargo pgrx install --release
   ```

   If the destination folder requires elevated permissions, use the `--sudo`
   flag:

   ```shell
   cargo pgrx install --sudo --release
   ```

1. Connect to the database:

   ```bash
   psql -d "postgres://<username>@<password>:<port>/<database-name>"
   ```

1. Add pgvectorscale to your database:

    ```postgresql
    CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;
    ```

# Troubleshooting

## MacOS: fatal error: 'stdio.h' file not found

Make sure you have the command line tools:

```bash
xcode-select --install
```

If the error persists try setting explicitly the `SDKROOT` environment
variable:

```bash
export SDKROOT=$(xcrun --sdk macosx --show-sdk-path)
```

Make sure the variable was set:


```bash
$ env | grep SDKROOT
SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk
```

## Permission denied while installing the extension artifacts

If the PostgreSQL directory where the extension needs to be installed requires
elevated permissions (for example, when using MacOS PostgreSQL that installs in the
`/Applications` folder, or when installing from a package manager that uses a system
directory), you need to use the pgrx `--sudo` flag:

```bash
cargo pgrx install --sudo --release
```

[pgvector]: https://github.com/pgvector/pgvector/blob/master/README.md
[rust-language]: https://www.rust-lang.org/
[cargo-pgrx]: https://lib.rs/crates/cargo-pgrx
