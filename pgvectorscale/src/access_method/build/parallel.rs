use crate::util::ports;
use pgrx::pg_sys;

pub fn flush_rate() -> usize {
    crate::access_method::guc::TSV_PARALLEL_FLUSH_RATE.get() as usize
}

pub fn initial_start_nodes_count() -> usize {
    crate::access_method::guc::TSV_PARALLEL_INITIAL_START_NODES_COUNT.get() as usize
}

pub const SHM_TOC_SHARED_KEY: u64 = 0xD000000000000001;
pub const SHM_TOC_TABLESCANDESC_KEY: u64 = 0xD000000000000002;

/// Cleans up a parallel context when we're done with it.
pub unsafe fn cleanup_pcxt(
    pcxt: *mut pg_sys::ParallelContext,
    snapshot: *mut pg_sys::SnapshotData,
) {
    // need DSM segment to do parallel build
    if ports::is_mvcc_snapshot(snapshot) {
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
