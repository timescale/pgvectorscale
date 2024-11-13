use std::{cell::OnceCell, cmp::Ordering};

use crate::util::{IndexPointer, ItemPointer};

//TODO is this right?
pub type Distance = f32;

// implements a distance with a lazy tie break
#[derive(Clone, Debug)]
pub struct DistanceWithTieBreak {
    distance: Distance,
    from: IndexPointer,
    to: IndexPointer,
    distance_tie_break: OnceCell<usize>,
}

impl DistanceWithTieBreak {
    pub fn new(distance: Distance, from: IndexPointer, to: IndexPointer) -> Self {
        assert!(!distance.is_nan());
        assert!(distance >= 0.0);
        DistanceWithTieBreak {
            distance,
            from,
            to,
            distance_tie_break: OnceCell::new(),
        }
    }

    pub fn with_query(distance: Distance, to: IndexPointer) -> Self {
        //this is the distance from the query to a index node.
        //make the distance_tie_break = 0
        let distance_tie_break = OnceCell::new();
        //explicitly set the distance_tie_break to 0 to avoid the cost of computing it
        distance_tie_break.set(0).unwrap();
        DistanceWithTieBreak {
            distance,
            from: to,
            to,
            distance_tie_break,
        }
    }

    fn get_distance_tie_break(&self) -> usize {
        *self
            .distance_tie_break
            .get_or_init(|| self.from.ip_distance(self.to))
    }

    fn get_distance(&self) -> Distance {
        self.distance
    }

    pub fn get_factor(&self, divisor: &Self) -> f64 {
        if divisor.get_distance() < 0.0 + f32::EPSILON {
            if self.get_distance() < 0.0 + f32::EPSILON {
                self.get_distance_tie_break() as f64 / divisor.get_distance_tie_break() as f64
            } else {
                f64::MAX
            }
        } else {
            self.get_distance() as f64 / divisor.get_distance() as f64
        }
    }
}

impl PartialOrd for DistanceWithTieBreak {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DistanceWithTieBreak {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.distance == 0.0 && other.distance == 0.0 {
            return self
                .get_distance_tie_break()
                .cmp(&other.get_distance_tie_break());
        }
        self.distance.total_cmp(&other.distance)
    }
}

impl PartialEq for DistanceWithTieBreak {
    fn eq(&self, other: &Self) -> bool {
        if self.distance == 0.0 && other.distance == 0.0 {
            return self.get_distance_tie_break() == other.get_distance_tie_break();
        }
        self.distance == other.distance
    }
}

//promise that PartialEq is reflexive
impl Eq for DistanceWithTieBreak {}

#[derive(Clone, Debug)]
pub struct NeighborWithDistance {
    index_pointer: IndexPointer,
    distance: DistanceWithTieBreak,
}

impl NeighborWithDistance {
    pub fn new(neighbor_index_pointer: ItemPointer, distance: DistanceWithTieBreak) -> Self {
        Self {
            index_pointer: neighbor_index_pointer,
            distance,
        }
    }

    pub fn get_index_pointer_to_neighbor(&self) -> ItemPointer {
        self.index_pointer
    }

    pub fn get_distance_with_tie_break(&self) -> &DistanceWithTieBreak {
        &self.distance
    }
}

impl PartialOrd for NeighborWithDistance {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for NeighborWithDistance {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance.cmp(&other.distance)
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
