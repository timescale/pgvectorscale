//! PostgreSQL advisory lock utilities for serializing index operations.

use pgrx::PgRelation;

/// Acquire a PostgreSQL transaction-level advisory lock for the given relation to serialize index
/// operations.  The lock is managed by Postgres, so no RAII/Drop implementation is needed.
pub fn acquire_index_lock(index: &PgRelation) {
    let oid = index.oid().as_u32();

    unsafe {
        // Use PostgreSQL's transaction-level advisory lock with relation OID as key
        // This will block until the lock is acquired and automatically release on transaction end
        pgrx::direct_function_call::<()>(
            pgrx::pg_sys::pg_advisory_xact_lock_int8,
            &[
                Some(pgrx::pg_sys::Datum::from(oid as i64)),
                Some(pgrx::pg_sys::Datum::from(1i64)), // Use 1 to distinguish from other lock types
            ],
        );
    }
}
