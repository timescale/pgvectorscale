use pgrx::PgRelation;
use rkyv::AlignedVec;

use crate::util::{tape::Tape, ItemPointer};

use super::stats::{StatsNodeModify, StatsNodeRead, StatsNodeWrite};

pub trait ReadableNode {
    type Node<'a>;
    unsafe fn read<'a, S: StatsNodeRead>(
        index: &'a PgRelation,
        index_pointer: ItemPointer,
        stats: &mut S,
    ) -> Self::Node<'a>;
}

pub trait WriteableNode {
    type Node<'a>;

    unsafe fn modify<'a, S: StatsNodeModify>(
        index: &'a PgRelation,
        index_pointer: ItemPointer,
        stats: &mut S,
    ) -> Self::Node<'a>;

    fn write<S: StatsNodeWrite>(&self, tape: &mut Tape, stats: &mut S) -> ItemPointer;

    fn serialize_to_vec(&self) -> AlignedVec;
}
