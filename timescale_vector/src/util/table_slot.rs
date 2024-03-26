use pgrx::pg_sys::{Datum, TupleTableSlot};
use pgrx::{pg_sys, PgBox, PgRelation};

use crate::access_method::pg_vector::PgVector;
use crate::access_method::stats::StatsHeapNodeRead;
use crate::util::ports::slot_getattr;
use crate::util::HeapPointer;

pub struct TableSlot {
    slot: PgBox<TupleTableSlot>,
    attribute_number: pg_sys::AttrNumber,
}

impl TableSlot {
    pub unsafe fn new<S: StatsHeapNodeRead>(
        heap_rel: &PgRelation,
        heap_pointer: HeapPointer,
        attribute_number: pg_sys::AttrNumber,
        stats: &mut S,
    ) -> Self {
        let slot = PgBox::from_pg(pg_sys::table_slot_create(
            heap_rel.as_ptr(),
            std::ptr::null_mut(),
        ));

        let table_am = heap_rel.rd_tableam;
        let fetch_row_version = (*table_am).tuple_fetch_row_version.unwrap();
        let mut ctid: pg_sys::ItemPointerData = pg_sys::ItemPointerData {
            ..Default::default()
        };
        heap_pointer.to_item_pointer_data(&mut ctid);
        fetch_row_version(
            heap_rel.as_ptr(),
            &mut ctid,
            &mut pg_sys::SnapshotAnyData,
            slot.as_ptr(),
        );
        stats.record_heap_read();

        Self {
            slot,
            attribute_number,
        }
    }

    unsafe fn get_attribute(&self, attribute_number: pg_sys::AttrNumber) -> Option<Datum> {
        slot_getattr(&self.slot, attribute_number)
    }

    pub unsafe fn get_pg_vector(&self) -> PgVector {
        let vector = PgVector::from_datum(self.get_attribute(self.attribute_number).unwrap());
        vector
    }
}

impl Drop for TableSlot {
    fn drop(&mut self) {
        unsafe { pg_sys::ExecDropSingleTupleTableSlot(self.slot.as_ptr()) };
    }
}
