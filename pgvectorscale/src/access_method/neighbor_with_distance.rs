use std::cmp::Ordering;

use crate::util::{IndexPointer, ItemPointer};

//TODO is this right?
pub type Distance = f32;
#[derive(Clone, Debug)]
pub struct NeighborWithDistance {
    index_pointer: IndexPointer,
    distance: Distance,
    distance_tie_break: usize,
}

impl NeighborWithDistance {
    pub fn new(
        neighbor_index_pointer: ItemPointer,
        distance: Distance,
        distance_tie_break: usize,
    ) -> Self {
        assert!(!distance.is_nan());
        assert!(distance >= 0.0);
        Self {
            index_pointer: neighbor_index_pointer,
            distance,
            distance_tie_break,
        }
    }

    pub fn get_index_pointer_to_neighbor(&self) -> ItemPointer {
        self.index_pointer
    }
    pub fn get_distance(&self) -> Distance {
        self.distance
    }
    pub fn get_distance_tie_break(&self) -> usize {
        return self.distance_tie_break;
    }
}

impl PartialOrd for NeighborWithDistance {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for NeighborWithDistance {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.distance == 0.0 && other.distance == 0.0 {
            return self.distance_tie_break.cmp(&other.distance_tie_break);
        }
        self.distance.total_cmp(&other.distance)
    }
}

impl PartialEq for NeighborWithDistance {
    fn eq(&self, other: &Self) -> bool {
        self.index_pointer == other.index_pointer
    }
}

//promise that PartialEq is reflexive
impl Eq for NeighborWithDistance {}

impl std::hash::Hash for NeighborWithDistance {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.index_pointer.hash(state);
    }
}
