use std::collections::HashMap;

use pgrx::*;

use crate::util::{IndexPointer, ItemPointer};

use super::build::TsvMetaPage;
use super::graph::Graph;
use super::model::{Distance, NeighborWithDistance, Node, ReadableNode};

/// A builderGraph is a graph that keep the neighbors in-memory in the neighbor_map below
/// The idea is that during the index build, you don't want to update the actual Postgres
/// pages every time you change the neighbors. Instead you change the neighbors in memory
/// until the build is done. Afterwards, calling the `write` method, will write out all
/// the neighbors to the right pages.
pub struct BuilderGraph {
    //maps node's pointer to the representation on disk
    neighbor_map: std::collections::HashMap<ItemPointer, Vec<NeighborWithDistance>>,
    first: Option<ItemPointer>,
    meta_page: super::build::TsvMetaPage,
}

impl BuilderGraph {
    pub fn new(meta_page: super::build::TsvMetaPage) -> Self {
        Self {
            neighbor_map: HashMap::with_capacity(200),
            first: None,
            meta_page: meta_page,
        }
    }

    pub fn insert(&mut self, index: &PgRelation, index_pointer: IndexPointer, vec: &[f32]) {
        if self.neighbor_map.len() == 0 {
            self.neighbor_map.insert(
                index_pointer,
                Vec::<NeighborWithDistance>::with_capacity(self.meta_page.num_neighbors as _),
            );
            return;
        }

        //TODO: make configurable?
        let search_list_size = 100;
        let (_, v) = self.greedy_search(index, vec, search_list_size);
        let neighbor_list =
            self.prune_neighbors(index, index_pointer, v.unwrap().into_iter().collect());

        //set forward pointers
        self.neighbor_map
            .insert(index_pointer, neighbor_list.clone());

        //update back pointers
        for neighbor in neighbor_list {
            self.update_back_pointer(
                index,
                neighbor.get_index_pointer_to_neigbor(),
                index_pointer,
                neighbor.get_distance(),
            )
        }
    }

    fn update_back_pointer(
        &mut self,
        index: &PgRelation,
        from: IndexPointer,
        to: IndexPointer,
        distance: f32,
    ) {
        let current_links = self.neighbor_map.get_mut(&from).unwrap();
        if current_links.len() + 1 < self.meta_page.num_neighbors as _ {
            current_links.push(NeighborWithDistance::new(to, distance));
        } else {
            let new_list =
                self.prune_neighbors(index, from, vec![NeighborWithDistance::new(to, distance)]);
            self.neighbor_map.insert(from, new_list);
        }
    }

    pub unsafe fn write(&self, index: &PgRelation) {
        //TODO: OPT: do this in order of item pointers
        for (index_pointer, neighbors) in &self.neighbor_map {
            let node = Node::modify(index, index_pointer);
            let mut archived = node.get_archived_node();
            for (i, new_neighbor) in neighbors.iter().enumerate() {
                //TODO: why do we need to recreate the archive?
                let mut a_index_pointer = archived.as_mut().neighbor_index_pointer().index_pin(i);
                //TODO hate that we have to set each field like this
                a_index_pointer.block_number =
                    new_neighbor.get_index_pointer_to_neigbor().block_number;
                a_index_pointer.offset = new_neighbor.get_index_pointer_to_neigbor().offset;

                let mut a_distance = archived.as_mut().neighbor_distances().index_pin(i);
                *a_distance = new_neighbor.get_distance() as Distance;
            }
            //set the marker that the list ended
            if neighbors.len() < self.meta_page.num_neighbors as _ {
                //TODO: why do we need to recreate the archive?
                let archived = node.get_archived_node();
                let mut past_last_distance =
                    archived.neighbor_distances().index_pin(neighbors.len());
                *past_last_distance = Distance::NAN;
            }
            node.commit()
        }
    }
}

impl Graph for BuilderGraph {
    fn get_init_ids(&mut self) -> Option<Vec<ItemPointer>> {
        //TODO make this based on centroid. For now, just first node.
        //returns a vector for generality
        match &self.first {
            Some(item) => Some(vec![*item]),
            None => match self.neighbor_map.keys().next() {
                Some(item) => {
                    self.first = Some(*item);
                    Some(vec![*item])
                }
                None => None,
            },
        }
    }

    fn read(&self, index: &PgRelation, index_pointer: ItemPointer) -> ReadableNode {
        unsafe { Node::read(index, &index_pointer) }
    }

    fn get_neighbors(
        &self,
        _index: &PgRelation,
        neighbors_of: ItemPointer,
    ) -> Option<Vec<NeighborWithDistance>> {
        let neighbors = self.neighbor_map.get(&neighbors_of);
        match neighbors {
            Some(n) => Some(n.iter().map(|v| v.clone()).collect()),
            None => None,
        }
    }

    fn get_meta_page(&self, _index: &PgRelation) -> &TsvMetaPage {
        &self.meta_page
    }
}
