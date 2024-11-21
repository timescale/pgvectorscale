#!/bin/bash

DEBHELPER_COMPAT=11

set -eux

OS_NAME="${3}"
BASEDIR="${2}"/pgvectorscale
DEBDIR="${PWD}"/pkgdump
PGVECTORSCALE_VERSION="${1}"
PG_VERSIONS="${4}"

echo "$BASEDIR"

if [ ! -d "$DEBDIR" ]; then
    mkdir -p "${DEBDIR}"
fi

DEB_VERSION=${PGVECTORSCALE_VERSION}-${OS_NAME}

# Show what we got to aid debugging.
git log -1
rm -rf "${BASEDIR}"/debian && mkdir -p "${BASEDIR}"/debian
ln -s /usr/bin/dh "${BASEDIR}"/debian/rules

date=$(TZ=Etc/UTC date -R)
maintainer='Timescale <hello@timescale.com>'

cd "${BASEDIR}"

# deb-changelog(5)
cat >"${BASEDIR}"/debian/changelog <<EOF
pgvectorscale (1:$DEB_VERSION) unused; urgency=medium
  * See https://github.com/timescale/pgvectorscale/releases/tag/$PGVECTORSCALE_VERSION
 -- $maintainer  $date
EOF
# deb-src-control(5)
cat >"${BASEDIR}"/debian/control <<EOF
Source: pgvectorscale
Maintainer: $maintainer
Homepage: https://github.com/timescale/pgvectorscale
Rules-Requires-Root: no
Section: vector
Priority: extra
Build-Depends: debhelper-compat (= $DEBHELPER_COMPAT)
EOF

libdir=$(pg_config --libdir)
sharedir=$(pg_config --sharedir)

base_PATH=$PATH

for pg in $PG_VERSIONS; do
    PATH=/usr/lib/postgresql/$pg/bin:$base_PATH
    #    cargo pgrx package
    cat >>"${BASEDIR}"/debian/control <<EOF

Package: pgvectorscale-postgresql-$pg
Architecture: any
Depends: postgresql-$pg
Description: pgvectorscale for speeding up ANN search
EOF

    echo "../target/release/vectorscale-pg$pg/$libdir/* usr/lib/postgresql/$pg/lib/" >"${BASEDIR}"/debian/pgvectorscale-postgresql-"$pg".install
    echo "../target/release/vectorscale-pg$pg/$sharedir/* usr/share/postgresql/$pg/" >>"${BASEDIR}"/debian/pgvectorscale-postgresql-"$pg".install
done

dpkg-buildpackage --build=binary --no-sign --post-clean

cd ..

# packagecloud.io doesn't support `.ddeb` files?  Like `.udeb`, they're just
# deb packages by another name, so:
for i in pgvectorscale*.ddeb; do
    # But it's only on Ubuntu that dpkg-buildpackage creates dbgsym packages
    # with the suffix `.ddeb`.  On Debian, 'pgvectorscale*.ddeb'
    # evaluates to 'pgvectorscale*.ddeb' so there's nothing to do.
    [ "$i" = 'pgvectorscale*.ddeb' ] || mv "$i" "${i%.ddeb}".deb
done

cp pgvectorscale*.deb "$DEBDIR"
