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
            slot.as_ptr(),
            &mut call_again,
            &mut all_dead,
        );
        (*table_am).index_fetch_end.unwrap()(scan);

        assert!(!call_again, "MVCC snapshots should not require call_again");
        stats.record_heap_read();

        if !valid {
            /* no valid tuples found in HOT-chain */
            return None;
        }

        Some(Self { slot })
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
