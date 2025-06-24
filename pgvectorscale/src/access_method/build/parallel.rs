use crate::util::ports;
use pgrx::pg_sys;

pub const SHM_TOC_SHARED_KEY: u64 = 0xD000000000000001;
pub const SHM_TOC_TABLESCANDESC_KEY: u64 = 0xD000000000000002;

/// Is a snapshop MVCC-safe? (This should really be a part of pgrx)
pub unsafe fn is_mvcc_snapshot(snapshot: *mut pg_sys::SnapshotData) -> bool {
    let typ = (*snapshot).snapshot_type;
    typ == pg_sys::SnapshotType::SNAPSHOT_MVCC
        || typ == pg_sys::SnapshotType::SNAPSHOT_HISTORIC_MVCC
}

/// Cleans up a parallel context when we're done with it.
pub unsafe fn cleanup_pcxt(
    pcxt: *mut pg_sys::ParallelContext,
    snapshot: *mut pg_sys::SnapshotData,
) {
    // need DSM segment to do parallel build
    if is_mvcc_snapshot(snapshot) {
        pg_sys::UnregisterSnapshot(snapshot);
    }
    pg_sys::DestroyParallelContext(pcxt);
    pg_sys::ExitParallelMode();
}

/// Estimate a single chunk in the shared memory TOC.
pub unsafe fn toc_estimate_single_chunk(pcxt: *mut pg_sys::ParallelContext, size: usize) {
    (*pcxt).estimator.space_for_chunks += ports::buffer_align(size);
    (*pcxt).estimator.number_of_keys += 1;
}
