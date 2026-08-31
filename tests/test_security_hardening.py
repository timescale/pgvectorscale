"""Regression tests for DiskANN type and layout hardening."""

import os
import uuid
from contextlib import contextmanager

import psycopg2
import pytest


@contextmanager
def _temporary_database(db_connection_params):
    database = f"vectorscale_security_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    admin = psycopg2.connect(**db_connection_params)
    admin.autocommit = True

    try:
        with admin.cursor() as cur:
            cur.execute(f'CREATE DATABASE "{database}"')

        params = {**db_connection_params, "database": database}
        conn = psycopg2.connect(**params)
        conn.autocommit = True
        try:
            yield conn
        finally:
            conn.close()
    finally:
        with admin.cursor() as cur:
            cur.execute(
                """
                SELECT pg_catalog.pg_terminate_backend(pid)
                FROM pg_catalog.pg_stat_activity
                WHERE datname = %s
                """,
                (database,),
            )
            cur.execute(f'DROP DATABASE IF EXISTS "{database}"')
        admin.close()


def _assert_diskann_vector_bindings(cur):
    cur.execute(
        """
        WITH expected AS (
            SELECT e.extnamespace, t.oid AS vector_oid
            FROM pg_catalog.pg_extension e
            JOIN pg_catalog.pg_type t
              ON t.typnamespace = e.extnamespace
             AND t.typname = 'vector'
            WHERE e.extname = 'vector'
        )
        SELECT
            pg_catalog.bool_and(c.opcintype = expected.vector_oid),
            pg_catalog.bool_and(opr.oprnamespace = expected.extnamespace)
        FROM pg_catalog.pg_opclass c
        JOIN pg_catalog.pg_am am ON am.oid = c.opcmethod
        JOIN pg_catalog.pg_amop amop ON amop.amopfamily = c.opcfamily
        JOIN pg_catalog.pg_operator opr ON opr.oid = amop.amopopr
        CROSS JOIN expected
        WHERE am.amname = 'diskann'
          AND c.opcname IN (
              'vector_cosine_ops',
              'vector_l2_ops',
              'vector_ip_ops'
          )
        """
    )
    type_ok, operator_ok = cur.fetchone()
    assert type_ok is True
    assert operator_ok is True


@pytest.mark.integration
def test_typmodless_vector_index_is_rejected(db_conn):
    with db_conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS test_typmodless_vector CASCADE")
        cur.execute("CREATE TABLE test_typmodless_vector (embedding vector)")

        with pytest.raises(
            psycopg2.Error,
            match="indexed column has no valid vector dimension",
        ):
            cur.execute(
                "CREATE INDEX ON test_typmodless_vector USING diskann (embedding)"
            )

        cur.execute("DROP TABLE test_typmodless_vector CASCADE")


@pytest.mark.integration
def test_schema_shadow_cannot_capture_vector_opclasses(db_connection_params):
    with _temporary_database(db_connection_params) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE SCHEMA real_vector;
                CREATE EXTENSION vector WITH SCHEMA real_vector;
                CREATE SCHEMA evil;
                CREATE DOMAIN evil.vector AS pg_catalog.uuid;
                CREATE FUNCTION evil.fake_dist(evil.vector, evil.vector)
                RETURNS double precision
                LANGUAGE sql IMMUTABLE STRICT PARALLEL SAFE
                AS $$ SELECT 0::double precision $$;
                CREATE OPERATOR evil.<=> (
                    LEFTARG = evil.vector,
                    RIGHTARG = evil.vector,
                    FUNCTION = evil.fake_dist,
                    COMMUTATOR = OPERATOR(evil.<=>)
                );
                CREATE OPERATOR evil.<-> (
                    LEFTARG = evil.vector,
                    RIGHTARG = evil.vector,
                    FUNCTION = evil.fake_dist,
                    COMMUTATOR = OPERATOR(evil.<->)
                );
                CREATE OPERATOR evil.<#> (
                    LEFTARG = evil.vector,
                    RIGHTARG = evil.vector,
                    FUNCTION = evil.fake_dist,
                    COMMUTATOR = OPERATOR(evil.<#>)
                );
                CREATE EXTENSION vectorscale WITH SCHEMA evil;
                """
            )
            _assert_diskann_vector_bindings(cur)


@pytest.mark.integration
def test_schema_shadow_cannot_capture_vector_operators(db_connection_params):
    with _temporary_database(db_connection_params) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE SCHEMA real_vector;
                CREATE EXTENSION vector WITH SCHEMA real_vector;
                CREATE SCHEMA evil;
                CREATE FUNCTION evil.fake_dist(real_vector.vector, real_vector.vector)
                RETURNS double precision
                LANGUAGE sql IMMUTABLE STRICT PARALLEL SAFE
                AS $$ SELECT 0::double precision $$;
                CREATE OPERATOR evil.<=> (
                    LEFTARG = real_vector.vector,
                    RIGHTARG = real_vector.vector,
                    FUNCTION = evil.fake_dist
                );
                CREATE OPERATOR evil.<-> (
                    LEFTARG = real_vector.vector,
                    RIGHTARG = real_vector.vector,
                    FUNCTION = evil.fake_dist
                );
                CREATE OPERATOR evil.<#> (
                    LEFTARG = real_vector.vector,
                    RIGHTARG = real_vector.vector,
                    FUNCTION = evil.fake_dist
                );
                CREATE EXTENSION vectorscale WITH SCHEMA evil;
                """
            )
            _assert_diskann_vector_bindings(cur)

            cur.execute(
                """
                CREATE OPERATOR CLASS evil.poisoned_ops
                FOR TYPE real_vector.vector USING diskann AS
                    OPERATOR 1 evil.<=> (real_vector.vector, real_vector.vector)
                    FOR ORDER BY pg_catalog.float_ops
                """
            )
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM pg_catalog.pg_opclass c
                    JOIN pg_catalog.pg_am am ON am.oid = c.opcmethod
                    JOIN pg_catalog.pg_extension e ON e.extname = 'vector'
                    WHERE am.amname = 'diskann'
                      AND c.opcname = 'poisoned_ops'
                      AND EXISTS (
                          SELECT 1
                          FROM pg_catalog.pg_amop amop
                          JOIN pg_catalog.pg_operator opr ON opr.oid = amop.amopopr
                          WHERE amop.amopfamily = c.opcfamily
                            AND opr.oprnamespace IS DISTINCT FROM e.extnamespace
                      )
                )
                """
            )
            assert cur.fetchone()[0] is True
