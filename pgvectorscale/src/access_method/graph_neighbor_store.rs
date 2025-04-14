use std::collections::BTreeMap;

use crate::util::{IndexPointer, ItemPointer};

use super::labels::LabelSet;
use super::stats::{StatsDistanceComparison, StatsNodeModify, StatsNodeRead};

use super::meta_page::MetaPage;
use super::neighbor_with_distance::*;
use super::storage::Storage;

/// A builderGraph is a graph that keep the neighbors in-memory in the neighbor_map below
/// The idea is that during the index build, you don't want to update the actual Postgres
/// pages every time you change the neighbors. Instead you change the neighbors in memory
/// until the build is done. Afterwards, calling the `write` method, will write out all
/// the neighbors to the right pages.
pub struct BuilderNeighborCache {
    //maps node's pointer to the representation on disk
    //use a btree to provide ordering on the item pointers in iter().
    //this ensures the write in finalize_node_at_end_of_build() is ordered, not random.
    neighbor_map: BTreeMap<ItemPointer, (Option<LabelSet>, Vec<NeighborWithDistance>)>,
}

impl BuilderNeighborCache {
    pub fn new() -> Self {
        Self {
            neighbor_map: BTreeMap::new(),
        }
    }

    pub fn iter(
        &self,
    ) -> impl Iterator<
        Item = (
            &ItemPointer,
            (Option<&LabelSet>, &Vec<NeighborWithDistance>),
        ),
    > {
        self.neighbor_map
            .iter()
            .map(|(k, (v1, v2))| (k, (v1.as_ref(), v2)))
    }

    pub fn get_neighbors(&self, neighbors_of: ItemPointer) -> Vec<IndexPointer> {
        let neighbors = self.neighbor_map.get(&neighbors_of);
        match neighbors {
            Some((_, n)) => n
                .iter()
                .map(|n| n.get_index_pointer_to_neighbor())
                .collect(),
            None => vec![],
        }
    }

    pub fn get_neighbors_with_full_vector_distances(
        &self,
        neighbors_of: ItemPointer,
        result: &mut Vec<NeighborWithDistance>,
    ) {
        let neighbors = self.neighbor_map.get(&neighbors_of);
        if let Some((_, n)) = neighbors {
            for nwd in n {
                result.push(nwd.clone());
            }
        }
    }

    pub fn set_neighbors(
        &mut self,
        neighbors_of: ItemPointer,
        labels: Option<LabelSet>,
        new_neighbors: Vec<NeighborWithDistance>,
    ) {
        self.neighbor_map
            .insert(neighbors_of, (labels, new_neighbors));
    }

    pub fn max_neighbors(&self, meta_page: &MetaPage) -> usize {
        meta_page.get_max_neighbors_during_build()
    }
}

pub enum GraphNeighborStore {
    Builder(BuilderNeighborCache),
    Disk,
}

impl GraphNeighborStore {
    pub fn get_neighbors_with_full_vector_distances<
        S: Storage,
        T: StatsNodeRead + StatsDistanceComparison,
    >(
        &self,
        neighbors_of: ItemPointer,
        storage: &S,
        result: &mut Vec<NeighborWithDistance>,
        stats: &mut T,
    ) {
        match self {
            GraphNeighborStore::Builder(b) => {
                b.get_neighbors_with_full_vector_distances(neighbors_of, result)
            }
            GraphNeighborStore::Disk => {
                storage.get_neighbors_with_distances_from_disk(neighbors_of, result, stats)
            }
        };
    }

    pub fn set_neighbors<S: Storage, T: StatsNodeModify + StatsNodeRead>(
        &mut self,
        storage: &S,
        meta_page: &MetaPage,
        neighbors_of: ItemPointer,
        labels: Option<LabelSet>,
        new_neighbors: Vec<NeighborWithDistance>,
        stats: &mut T,
    ) {
        match self {
            GraphNeighborStore::Builder(b) => b.set_neighbors(neighbors_of, labels, new_neighbors),
            GraphNeighborStore::Disk => storage.set_neighbors_on_disk(
                meta_page,
                neighbors_of,
                new_neighbors.as_slice(),
                stats,
            ),
        }
    }

    pub fn max_neighbors(&self, meta_page: &MetaPage) -> usize {
        match self {
            GraphNeighborStore::Builder(b) => b.max_neighbors(meta_page),
            GraphNeighborStore::Disk => meta_page.get_num_neighbors() as _,
        }
    }
}
