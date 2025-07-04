pub mod node;
pub mod storage;
mod tests;

use super::{
    distance::DistanceFn,
    labels::LabeledVector,
    stats::{StatsDistanceComparison, StatsNodeRead},
    storage::{NodeDistanceMeasure, Storage},
};
use node::{PlainNode, ReadablePlainNode};
use storage::PlainStorage;

use crate::access_method::node::ReadableNode;
use crate::util::IndexPointer;

pub enum PlainDistanceMeasure {
    Full(LabeledVector),
}

impl PlainDistanceMeasure {
    pub fn calculate_distance<S: StatsDistanceComparison>(
        distance_fn: DistanceFn,
        query: &[f32],
        vector: &[f32],
        stats: &mut S,
    ) -> f32 {
        assert!(!vector.is_empty());
        assert!(vector.len() == query.len());
        stats.record_full_distance_comparison();
        (distance_fn)(query, vector)
    }
}

/* This is only applicable to plain, so keep here not in storage_common */
pub struct IndexFullDistanceMeasure<'a> {
    readable_node: ReadablePlainNode<'a>,
    storage: &'a PlainStorage<'a>,
}

impl<'a> IndexFullDistanceMeasure<'a> {
    /// # Safety
    ///
    /// The caller must ensure that:
    /// 1. The index_pointer is valid and points to a properly initialized PlainNode
    /// 2. The storage reference remains valid for the lifetime 'a
    /// 3. The node at index_pointer is not modified while this IndexFullDistanceMeasure exists
    pub unsafe fn with_index_pointer<T: StatsNodeRead>(
        storage: &'a PlainStorage<'a>,
        index_pointer: IndexPointer,
        stats: &mut T,
    ) -> Self {
        let rn = unsafe { PlainNode::read(storage.index, index_pointer, stats) };
        Self {
            readable_node: rn,
            storage,
        }
    }

    /// # Safety
    ///
    /// The caller must ensure that:
    /// 1. The readable_node is valid and points to a properly initialized PlainNode
    /// 2. The storage reference remains valid for the lifetime 'a
    /// 3. The node at readable_node is not modified while this IndexFullDistanceMeasure exists
    pub unsafe fn with_readable_node(
        storage: &'a PlainStorage<'a>,
        readable_node: ReadablePlainNode<'a>,
    ) -> Self {
        Self {
            readable_node,
            storage,
        }
    }
}

impl NodeDistanceMeasure for IndexFullDistanceMeasure<'_> {
    unsafe fn get_distance<T: StatsNodeRead + StatsDistanceComparison>(
        &self,
        index_pointer: IndexPointer,
        stats: &mut T,
    ) -> f32 {
        let rn1 = PlainNode::read(self.storage.index, index_pointer, stats);
        let rn2 = &self.readable_node;
        let node1 = rn1.get_archived_node();
        let node2 = rn2.get_archived_node();
        assert!(!node1.vector.is_empty());
        assert!(node1.vector.len() == node2.vector.len());
        let vec1 = node1.vector.as_slice();
        let vec2 = node2.vector.as_slice();
        (self.storage.get_distance_function())(vec1, vec2)
    }
}
