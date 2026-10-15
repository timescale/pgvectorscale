use pgrx::pg_sys::{Datum, SnapshotData, TupleTableSlot};
use pgrx::{pg_sys, PgBox, PgRelation};

use crate::access_method::stats::StatsHeapNodeRead;
use crate::util::ports::slot_getattr;
use crate::util::HeapPointer;

pub struct TableSlot {
    slot: PgBox<TupleTableSlot>,
}

impl TableSlot {
    pub unsafe fn from_index_heap_pointer<S: StatsHeapNodeRead>(
        heap_rel: &PgRelation,
        heap_pointer: HeapPointer,
        snapshot: *mut SnapshotData,
        stats: &mut S,
    ) -> Option<Self> {
        let slot = PgBox::from_pg(pg_sys::table_slot_create(
            heap_rel.as_ptr(),
            std::ptr::null_mut(),
        ));

        // Arm the RAII wrapper IMMEDIATELY, before any fallible work.
        //
        // `table_slot_create` -> `MakeTupleTableSlot` calls `PinTupleDesc` on the
        // heap relation's rowtype descriptor, and that pin is released only by
        // `TableSlot::drop` -> `ExecDropSingleTupleTableSlot`. Constructing `Self`
        // here means every exit path below — the `!valid` early return, and any
        // unwind out of the fetch or the assert — drops the slot and releases the
        // descriptor pin. Previously the wrapper was built only on the success
        // path, so a heap pointer with no snapshot-visible tuple leaked both the
        // slot and its TupleDesc reference on every rescored candidate.
        let table_slot = Self { slot };

        let table_am = heap_rel.rd_tableam;
        let mut ctid: pg_sys::ItemPointerData = pg_sys::ItemPointerData {
            ..Default::default()
        };
        heap_pointer.to_item_pointer_data(&mut ctid);

        let scan = (*table_am).index_fetch_begin.unwrap()(heap_rel.as_ptr());
        let mut call_again = false;
        /* all_dead can be ignored, only used in optimizations we don't implement */
        let mut all_dead = false;
        let valid = (*table_am).index_fetch_tuple.unwrap()(
            scan,
            &mut ctid,
            snapshot,
            table_slot.slot.as_ptr(),
            &mut call_again,
            &mut all_dead,
        );
        (*table_am).index_fetch_end.unwrap()(scan);

        assert!(!call_again, "MVCC snapshots should not require call_again");
        stats.record_heap_read();

        if !valid {
            /* No visible tuple in the HOT chain (deleted / updated-away / not yet
             * vacuumed). Dropping `table_slot` here frees the slot and releases its
             * rowtype TupleDesc pin. */
            return None;
        }

        Some(table_slot)
    }

    pub unsafe fn get_attribute(&self, attribute_number: pg_sys::AttrNumber) -> Option<Datum> {
        slot_getattr(&self.slot, attribute_number)
    }
}

impl Drop for TableSlot {
    fn drop(&mut self) {
        unsafe { pg_sys::ExecDropSingleTupleTableSlot(self.slot.as_ptr()) };
    }
}

#[cfg(any(test, feature = "pg_test"))]
#[pgrx::pg_schema]
mod tests {
    use pgrx::{pg_sys, pg_test, Spi};

    use super::*;
    use crate::access_method::stats::GreedySearchStats;

    /// Parse a `ctid` in its text form, e.g. `"(0,1)"`.
    fn parse_ctid(ctid: &str) -> (pg_sys::BlockNumber, pg_sys::OffsetNumber) {
        let inner = ctid.trim_matches(|c| c == '(' || c == ')');
        let mut parts = inner.split(',');
        let block = parts
            .next()
            .expect("ctid should have a block number")
            .trim()
            .parse()
            .expect("block number should parse");
        let offset = parts
            .next()
            .expect("ctid should have an offset")
            .trim()
            .parse()
            .expect("offset number should parse");
        (block, offset)
    }

    /// Regression test for the per-scan `TupleDesc` + slot leak (issue #211).
    ///
    /// [`TableSlot::from_index_heap_pointer`] builds a `TupleTableSlot` with
    /// `table_slot_create`, which pins the heap relation's rowtype `TupleDesc`;
    /// that pin is released only by [`TableSlot`]'s `Drop`. When the heap pointer
    /// has no snapshot-visible tuple the function returns `None`. If the RAII
    /// wrapper is armed only on the success path, that early return leaks both the
    /// slot and the descriptor pin -- once per dead/invisible rescored candidate --
    /// which surfaces as `resource was not closed: TupleDesc ... (<oid>,-1)` at
    /// ResourceOwner release and as unbounded backend memory growth on tables with
    /// ongoing updates/deletes.
    ///
    /// The descriptor's reference count must therefore be conserved across a call
    /// that takes the no-visible-tuple path.
    #[pg_test]
    unsafe fn table_slot_releases_tupledesc_on_dead_tuple() {
        Spi::run(
            "CREATE TABLE slot_leak_test(encoding vector(3));
             INSERT INTO slot_leak_test(encoding) VALUES ('[1,2,3]');",
        )
        .unwrap();

        // Note where the row lives, then delete it, so that a snapshot taken
        // afterwards finds no visible version at that TID -- the path under test.
        let ctid = Spi::get_one::<String>("SELECT ctid::text FROM slot_leak_test LIMIT 1")
            .unwrap()
            .expect("the inserted row should exist");
        let (block, offset) = parse_ctid(&ctid);
        Spi::run("DELETE FROM slot_leak_test;").unwrap();

        let heap_oid = Spi::get_one::<pg_sys::Oid>("SELECT 'slot_leak_test'::regclass::oid")
            .unwrap()
            .expect("the relation should exist");
        // Open with a lock, exactly as the executor does before any table-AM
        // fetch. `PgRelation::with_lock` also closes and unlocks on drop.
        let heap_rel = PgRelation::with_lock(heap_oid, pg_sys::AccessShareLock as pg_sys::LOCKMODE);

        let tupdesc = heap_rel.rd_att;
        assert!(
            (*tupdesc).tdrefcount >= 0,
            "the relcache descriptor must be reference-counted for this test to be meaningful"
        );
        let refcount_before = (*tupdesc).tdrefcount;

        let mut stats = GreedySearchStats::default();
        let slot = TableSlot::from_index_heap_pointer(
            &heap_rel,
            HeapPointer::new(block, offset),
            // The statement's own registered snapshot. It was taken after the
            // DELETE above, so the row has no visible version at that TID.
            pg_sys::GetActiveSnapshot(),
            &mut stats,
        );

        assert!(
            slot.is_none(),
            "a deleted row must not yield a visible tuple"
        );
        assert_eq!(
            (*tupdesc).tdrefcount,
            refcount_before,
            "the rowtype TupleDesc pin must be released when there is no visible \
             tuple (leak: issue #211)"
        );
    }
}
