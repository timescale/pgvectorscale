"""Regression tests for DiskANN type and layout hardening."""

import os
import uuid

import psycopg2
import pytest


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

                cur.execute(
                    """
                    WITH expected AS (
                        SELECT t.oid
                        FROM pg_catalog.pg_extension e
                        JOIN pg_catalog.pg_type t
                          ON t.typnamespace = e.extnamespace
                         AND t.typname = 'vector'
                        WHERE e.extname = 'vector'
                    )
                    SELECT pg_catalog.bool_and(c.opcintype = expected.oid)
                    FROM pg_catalog.pg_opclass c
                    JOIN pg_catalog.pg_am am ON am.oid = c.opcmethod
                    CROSS JOIN expected
                    WHERE am.amname = 'diskann'
                      AND c.opcname IN (
                          'vector_cosine_ops',
                          'vector_l2_ops',
                          'vector_ip_ops'
                      )
                    """
                )
                assert cur.fetchone()[0] is True
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
