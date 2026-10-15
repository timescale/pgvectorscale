"""
Tests for CREATE INDEX progress reporting via pg_stat_progress_create_index.

Verifies that the diskann access method populates tuples_total / tuples_done /
phase during index builds. Exercises the parallel build path (forced via GUCs)
so the leader's observer loop is hit, not just the sequential path.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import psycopg2

VEC_DIM = 32
ROW_COUNT = 30_000
POLL_INTERVAL_S = 0.05
MAX_POLL_S = 30.0


def _vec_literal(n):
    """SQL expression that produces a vector of n random floats."""
    return (
        "('[' || array_to_string("
        f"ARRAY(SELECT random()::float4 FROM generate_series(1, {n}))"
        ", ',', '') || ']')::vector"
    )


def _force_parallel_gucs(cur):
    """Lower the parallel-build threshold so the 30k-row test triggers it."""
    for stmt in [
        "SET diskann.min_vectors_for_parallel_build = 1000",
        "SET diskann.force_parallel_workers = 4",
        "SET max_parallel_maintenance_workers = 4",
        "SET max_parallel_workers = 8",
        "SET parallel_tuple_cost = 0",
        "SET parallel_setup_cost = 0",
    ]:
        cur.execute(stmt)


def test_progress_increases(db_setup, clean_db):
    """tuples_done in pg_stat_progress_create_index rises during a parallel build."""
    table = "test_progress_inc"
    index = "idx_progress_inc"

    with db_conn_for(db_setup) as setup_conn:
        with setup_conn.cursor() as cur:
            cur.execute(
                f"CREATE TABLE {table} (id INT, v VECTOR({VEC_DIM}))"
            )
            cur.execute(
                f"INSERT INTO {table} "
                f"SELECT i, ({_vec_literal(VEC_DIM)}) "
                f"FROM generate_series(1, {ROW_COUNT}) i"
            )
            cur.execute(f"ANALYZE {table}")
            _force_parallel_gucs(cur)

    samples = []
    stop = threading.Event()

    def poll(poll_conn):
        while not stop.is_set():
            with poll_conn.cursor() as cur:
                cur.execute(
                    "SELECT phase, tuples_total, tuples_done "
                    "FROM pg_stat_progress_create_index"
                )
                rows = cur.fetchall()
                if rows:
                    samples.append((time.monotonic(), *rows[0]))
            time.sleep(POLL_INTERVAL_S)

    def build(build_conn):
        with build_conn.cursor() as cur:
            cur.execute(
                f"CREATE INDEX {index} ON {table} "
                f"USING diskann (v) WITH (num_neighbors=20)"
            )

    with db_conn_for(db_setup) as poll_conn, \
         db_conn_for(db_setup) as build_conn:
        with ThreadPoolExecutor(max_workers=2) as ex:
            poller = ex.submit(poll, poll_conn)
            builder = ex.submit(build, build_conn)
            try:
                builder.result(timeout=MAX_POLL_S)
            finally:
                stop.set()
                poller.result(timeout=5)

    assert len(samples) >= 3, f"too few progress samples: {len(samples)}"
    done_values = [s[3] for s in samples]
    for a, b in zip(done_values, done_values[1:]):
        assert b >= a, f"tuples_done went backwards: {done_values}"
    assert done_values[-1] >= 1, f"final tuples_done too low: {done_values[-1]}"
    phases = [s[1] for s in samples if s[1]]
    assert any(
        "building graph" in p
        or "training quantizer" in p
        or "finalizing graph" in p
        for p in phases
    ), f"no build phase seen in {phases}"


def test_progress_reaches_completion(db_conn, clean_db):
    """After CREATE INDEX finishes, no stale row remains in pg_stat_progress_create_index."""
    table = "test_progress_done"
    index = "idx_progress_done"

    with db_conn.cursor() as cur:
        cur.execute(f"CREATE TABLE {table} (id INT, v VECTOR(16))")
        cur.execute(
            f"INSERT INTO {table} "
            f"SELECT i, ({_vec_literal(16)}) "
            f"FROM generate_series(1, {ROW_COUNT}) i"
        )
        cur.execute(f"ANALYZE {table}")
        _force_parallel_gucs(cur)

        cur.execute(
            f"CREATE INDEX {index} ON {table} "
            f"USING diskann (v) WITH (num_neighbors=20)"
        )

        cur.execute(
            "SELECT indisvalid FROM pg_index "
            "WHERE indexrelid = %s::regclass",
            (index,),
        )
        row = cur.fetchone()
        assert row is not None, f"index {index} not found"
        assert row[0] is True, f"index {index} not valid"

        cur.execute(
            "SELECT count(*) FROM pg_stat_progress_create_index"
        )
        assert cur.fetchone()[0] == 0, (
            "stale row in pg_stat_progress_create_index"
        )


class _ConnCtx:
    """Tiny context manager so `with db_conn_for(params) as conn:` closes cleanly."""
    def __init__(self, params):
        self._conn = psycopg2.connect(**params)
        self._conn.autocommit = True

    def __enter__(self):
        return self._conn

    def __exit__(self, exc_type, exc, tb):
        self._conn.close()


def db_conn_for(params):
    return _ConnCtx(params)
