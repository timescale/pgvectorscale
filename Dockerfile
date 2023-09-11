# syntax=docker/dockerfile:1.3-labs
ARG PG_VERSION=15
ARG TIMESCALEDB_VERSION_MAJMIN=2.11
ARG PGRX_VERSION=0.9.8
ARG BASE_IMAGE=timescale/timescaledb-ha:pg${PG_VERSION}-ts${TIMESCALEDB_VERSION_MAJMIN}-all

FROM timescale/timescaledb-ha:pg${PG_VERSION}-ts${TIMESCALEDB_VERSION_MAJMIN}-all AS ha-build-tools
ARG PG_VERSION
ARG PGRX_VERSION

ENV DEBIAN_FRONTEND=noninteractive
USER root

RUN apt-get update
RUN apt-get install -y \
    clang \
    gcc \
    pkg-config \
    wget \
    lsb-release \
    libssl-dev \
    curl \
    gnupg2 \
    binutils \
    devscripts \
    equivs \
    git \
    libkrb5-dev \
    libopenblas-dev \
    libopenblas-base \
    libperl-dev \
    make \
    cmake

RUN wget -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | apt-key add -
RUN for t in deb deb-src; do \
    echo "$t [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/postgresql.keyring] http://apt.postgresql.org/pub/repos/apt/ $(lsb_release -s -c)-pgdg main" >> /etc/apt/sources.list.d/pgdg.list; \
    done

RUN apt-get update && apt-get install -y \
    postgresql-${PG_VERSION} \
    postgresql-server-dev-${PG_VERSION}

USER postgres
WORKDIR /build

ENV HOME=/build \
    PATH=/build/.cargo/bin:$PATH \
    CARGO_HOME=/build/.cargo \
    RUSTUP_HOME=/build/.rustup

RUN chown postgres:postgres /build

# if you need bleeding edge timescaledb
# RUN cd /build && git clone https://github.com/timescale/timescaledb.git /build/timescaledb \
#     && cd /build/timescaledb && rm -fr build \
#     && git checkout ${TS_VERSION} \
#     && ./bootstrap -DCMAKE_BUILD_TYPE=RelWithDebInfo -DREGRESS_CHECKS=OFF -DTAP_CHECKS=OFF -DGENERATE_DOWNGRADE_SCRIPT=OFF -DWARNINGS_AS_ERRORS=OFF \
#     && cd build && make install \
#     && cd ~

RUN curl https://sh.rustup.rs -sSf | bash -s -- -y --profile=minimal -c rustfmt
ENV PATH="${CARGO_HOME}/bin:${PATH}"

RUN set -ex \
    && mkdir /build/timescale-vector \
    && mkdir /build/timescale-vector/scripts \
    && mkdir /build/timescale-vector/timescale_vector

## Install pgrx taking into account selected rust toolchain version.
## Making this a separate step to improve layer caching
#COPY --chown=postgres:postgres timescale_vector/rust-toolchain.toml /build/timescale-vector/timescale_vector/rust-toolchain.toml
COPY --chown=postgres:postgres scripts /build/timescale-vector/scripts
USER postgres
WORKDIR /build/timescale-vector/timescale_vector
RUN set -ex \
    && rm -rf "${CARGO_HOME}/registry" "${CARGO_HOME}/git" \
    && chown postgres:postgres -R "${CARGO_HOME}" \
    && cargo install cargo-pgrx --version ${PGRX_VERSION} --config net.git-fetch-with-cli=true

## Copy and build Vector itself
USER postgres
COPY --chown=postgres:postgres timescale_vector /build/timescale-vector/timescale_vector
COPY --chown=postgres:postgres Makefile /build/timescale-vector/Makefile

WORKDIR /build/timescale-vector
RUN PG_CONFIG="/usr/lib/postgresql/${PG_VERSION}/bin/pg_config" make package

## COPY over the new files to the image. Done as a seperate stage so we don't
## ship the build tools. Fixed pg15 image is intentional. The image ships with
## PG 12, 13, 14 and 15 binaries. The PATH environment variable below is used
## to specify the runtime version.
FROM ${BASE_IMAGE}
ARG PG_VERSION

## Copy old versions and/or bleeding edge timescaledb if any were installed
COPY --from=ha-build-tools --chown=root:postgres /usr/share/postgresql /usr/share/postgresql
COPY --from=ha-build-tools --chown=root:postgres /usr/lib/postgresql /usr/lib/postgresql

## Copy freshly build current Vector version
COPY --from=ha-build-tools --chown=root:postgres /build/timescale-vector/timescale_vector/target/release/timescale_vector-pg${PG_VERSION}/usr/lib/postgresql /usr/lib/postgresql
COPY --from=ha-build-tools --chown=root:postgres /build/timescale-vector/timescale_vector/target/release/timescale_vector-pg${PG_VERSION}/usr/share/postgresql /usr/share/postgresql
ENV PATH="/usr/lib/postgresql/${PG_VERSION}/bin:${PATH}"

USER postgres
