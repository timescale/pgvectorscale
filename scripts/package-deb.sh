#!/bin/bash

DEBHELPER_COMPAT=11

set -ex

OS_NAME="${3}"
BASEDIR="${2}"/timescale_vector
DEBDIR="${PWD}"/pkgdump
TIMESCALE_VECTOR_VERSION="${1}"
PG_VERSIONS="${4}"

echo "$BASEDIR"

if [ ! -d "$DEBDIR" ]; then
    mkdir -p "${DEBDIR}"
fi

DEB_VERSION=${TIMESCALE_VECTOR_VERSION}-${OS_NAME}

# Show what we got to aid debugging.
git log -1
rm -rf "${BASEDIR}"/debian && mkdir -p "${BASEDIR}"/debian
ln -s /usr/bin/dh "${BASEDIR}"/debian/rules

date=$(TZ=Etc/UTC date -R)
maintainer='Timescale <hello@timescale.com>'

cd "${BASEDIR}"

# deb-changelog(5)
cat >"${BASEDIR}"/debian/changelog <<EOF
timescale-vector (1:$DEB_VERSION) unused; urgency=medium
  * See https://github.com/timescale/timescale-vector/releases/tag/$TIMESCALE_VECTOR_VERSION
 -- $maintainer  $date
EOF
# deb-src-control(5)
cat >"${BASEDIR}"/debian/control <<EOF
Source: timescale-vector
Maintainer: $maintainer
Homepage: https://github.com/timescale/timescale-vector
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

Package: timescale-vector-postgresql-$pg
Architecture: any
Depends: postgresql-$pg
Description: Timescale Vector Extension for Cloud
EOF

    echo "target/release/timescale_vector-pg$pg/$libdir/* usr/lib/postgresql/$pg/lib/" >"${BASEDIR}"/debian/timescale-vector-postgresql-"$pg".install
    echo "target/release/timescale_vector-pg$pg/$sharedir/* usr/share/postgresql/$pg/" >>"${BASEDIR}"/debian/timescale-vector-postgresql-"$pg".install
done

dpkg-buildpackage --build=binary --no-sign --post-clean

cd ..

# packagecloud.io doesn't support `.ddeb` files?  Like `.udeb`, they're just
# deb packages by another name, so:
for i in timescale-vector*.ddeb; do
    # But it's only on Ubuntu that dpkg-buildpackage creates dbgsym packages
    # with the suffix `.ddeb`.  On Debian, 'timescale-vector*.ddeb'
    # evaluates to 'timescale-vector*.ddeb' so there's nothing to do.
    [ "$i" = 'timescale-vector*.ddeb' ] || mv "$i" "${i%.ddeb}".deb
done

cp timescale-vector*.deb "$DEBDIR"
