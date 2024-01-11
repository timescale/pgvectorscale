use pgrx::PgRelation;

use crate::util::{table_slot::TableSlot, IndexPointer};

use super::{
    stats::{StatsDistanceComparison, StatsNodeRead},
    storage::{NodeFullDistanceMeasure, Storage, StorageFullDistanceFromHeap},
};

pub struct HeapFullDistanceMeasure<'a, S: Storage + StorageFullDistanceFromHeap> {
    table_slot: Option<TableSlot>,
    storage: &'a S,
}

impl<'a, S: Storage + StorageFullDistanceFromHeap> HeapFullDistanceMeasure<'a, S> {
    pub fn with_table_slot(slot: TableSlot, storage: &'a S) -> Self {
        Self {
            table_slot: Some(slot),
            storage: storage,
        }
    }
}

impl<'a, S: Storage + StorageFullDistanceFromHeap> NodeFullDistanceMeasure
    for HeapFullDistanceMeasure<'a, S>
{
    unsafe fn get_distance<T: StatsNodeRead + StatsDistanceComparison>(
        &self,
        index: &PgRelation,
        index_pointer: IndexPointer,
        stats: &mut T,
    ) -> f32 {
        let slot = self
            .storage
            .get_heap_table_slot(index, index_pointer, stats);
        stats.record_full_distance_comparison();
        let slice1 = slot.get_pg_vector();
        let slice2 = self.table_slot.as_ref().unwrap().get_pg_vector();
        (self.storage.get_distance_function())(slice1.to_slice(), slice2.to_slice())
    }
}
