SHELL:=/bin/bash
ROOTDIR = $(realpath .)
RUST_SRCDIR =$(ROOTDIR)/timescale_vector

PG_CONFIG = $(shell which pg_config)
EXTENSION=timescale_vector

PG_VERSION = $(shell ${PG_CONFIG} --version | awk -F'[ \.]' '{print $$2}')
##TODO error out if this is not PG14???
PGRX_HOME?= ${HOME}/.pgrx
PGRX_VERSION=0.9.8
VECTOR_VERSION?=$(shell sed -n 's/^[[:space:]]*version[[:space:]]*=[[:space:]]*"\(.*\)"/\1/p' timescale_vector/Cargo.toml)
PG_DATA=${PGRX_HOME}/data-${PG_VERSION}

PG_PKGLIBDIR=$(shell ${PG_CONFIG} --pkglibdir)
PG_SHARELIBDIR=$(shell ${PG_CONFIG} --sharedir)
$(info pg_pkglib = $(PG_PKGLIBDIR) and pg_sharelib = $(PG_SHARELIBDIR) )

MODULE_big = $(EXTENSION)
PGXS := $(shell $(PG_CONFIG) --pgxs)

include $(PGXS)
PG_REGRESS='$(top_builddir)/src/test/regress/pg_regress'
PG_REGRESS_OPTS_EXTRA=--create-role=superuser,tsdbadmin,test_role_1  --launcher=./test/runner.sh
export TEST_OUTPUT_DIR:=$(ROOTDIR)/test_output
export PG_ABS_SRCDIR:=$(ROOTDIR)/test
export TEST_DBNAME:=regression

### default collation settings on Cloud is C.UTF-8
PG_DEFAULT_REGRESS_LOCALE=$(shell uname | grep -q 'Darwin' && echo 'en_US.UTF-8'  || echo 'C.UTF-8')
PG_REGRESS_LOCALE?=$(PG_DEFAULT_REGRESS_LOCALE)
PG_REGRESS_ENV=CONFDIR='$(CURDIR)/test' TESTDIR='$(CURDIR)' LC_COLLATE=$(PG_REGRESS_LOCALE) LC_CTYPE=$(PG_REGRESS_LOCALE)

#ifdef PGHOST
#USE_EXISTING_INSTANCE=0
#endif

ifdef USE_EXISTING_INSTANCE
$(info Use existing instance)
INSTANCE_OPTS=
else
$(info Use temp instance)
INSTANCE_OPTS=--temp-instance=$(ROOTDIR)/test_instance --temp-config=$(ROOTDIR)/test/postgres.conf
endif

.PHONY: format
format:
	cd $(RUST_SRCDIR)/src && rustfmt --edition 2021 *.rs

.PHONY: build
build:
	cd $(RUST_SRCDIR) && cargo build --features pg${PG_VERSION} $(EXTRA_RUST_ARGS)

.PHONY: install-pgrx
install-pgrx:
	cargo install cargo-pgrx --version ${PGRX_VERSION}

.PHONY: init-pgrx
init-pgrx: $(PG_DATA)

$(PG_DATA):
	cd $(RUST_SRCDIR) && cargo pgrx init --pg${PG_VERSION}=${PG_CONFIG}

.PHONY: install-debug
###pgxs.mk has a rule for install.So we need a different rule name
install-debug: init-pgrx
	cd $(RUST_SRCDIR) && cargo pgrx install --features pg${PG_VERSION}

.PHONY: install-release
install-release: init-pgrx
	cd $(RUST_SRCDIR) && cargo pgrx install --release --features pg${PG_VERSION}


.PHONY: package
package: init-pgrx
	cd $(RUST_SRCDIR) && cargo pgrx package --features pg${PG_VERSION}

.PHONY: shellcheck
shellcheck:
	find . -name '*.sh' -exec shellcheck '{}' +

.PHONY: shfmt
shfmt:
	shfmt -w -i 4 test scripts


.PHONY: release rust test prove install clean
