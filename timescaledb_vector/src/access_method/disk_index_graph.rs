use pgrx::PgRelation;

use crate::util::ItemPointer;

use super::{
    build::{read_meta_page, TsvMetaPage},
    graph::Graph,
    model::{NeighborWithDistance, Node, ReadableNode},
};

pub struct DiskIndexGraph {
    meta_page: TsvMetaPage,
}

impl DiskIndexGraph {
    pub fn new(index: &PgRelation) -> Self {
        let meta = read_meta_page(index);
        Self { meta_page: meta }
    }
}

impl Graph for DiskIndexGraph {
    fn read(&self, index: &PgRelation, index_pointer: ItemPointer) -> ReadableNode {
        unsafe { Node::read(index, &index_pointer) }
    }

    fn get_init_ids(&mut self) -> Option<Vec<ItemPointer>> {
        self.meta_page.get_init_ids()
    }

    fn get_neighbors(
        &self,
        index: &PgRelation,
        neighbors_of: ItemPointer,
        result: &mut Vec<NeighborWithDistance>,
    ) -> bool {
        let rn = self.read(index, neighbors_of);
        rn.get_archived_node().apply_to_neighbors(|dist, n| {
            result.push(NeighborWithDistance::new(
                n.deserialize_item_pointer(),
                dist,
            ))
        });
        true
    }

    fn is_empty(&self) -> bool {
        self.meta_page.get_init_ids().is_none()
    }

    fn set_neighbors(
        &mut self,
        index: &PgRelation,
        neighbors_of: ItemPointer,
        new_neighbors: Vec<NeighborWithDistance>,
    ) {
        if self.meta_page.get_init_ids().is_none() {
            super::build::update_meta_page_init_ids(index, vec![neighbors_of]);
            self.meta_page = read_meta_page(index);
        }
        unsafe {
            Node::update_neighbors_and_pq(
                index,
                neighbors_of,
                &new_neighbors,
                self.get_meta_page(index),
                None,
            );
        }
    }

    fn get_meta_page(&self, _index: &PgRelation) -> &TsvMetaPage {
        &self.meta_page
    }
}
