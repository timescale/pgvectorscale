use std::cmp::Ordering;

use crate::util::{IndexPointer, ItemPointer};

//TODO is this right?
pub type Distance = f32;
#[derive(Clone, Debug)]
pub struct NeighborWithDistance {
    index_pointer: IndexPointer,
    distance: Distance,
}

impl NeighborWithDistance {
    pub fn new(neighbor_index_pointer: ItemPointer, distance: Distance) -> Self {
        assert!(!distance.is_nan());
        assert!(distance >= 0.0);
        Self {
            index_pointer: neighbor_index_pointer,
            distance,
        }
    }

    pub fn get_index_pointer_to_neighbor(&self) -> ItemPointer {
        return self.index_pointer;
    }
    pub fn get_distance(&self) -> Distance {
        return self.distance;
    }
}

impl PartialOrd for NeighborWithDistance {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}

impl Ord for NeighborWithDistance {
    fn cmp(&self, other: &Self) -> Ordering {
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
