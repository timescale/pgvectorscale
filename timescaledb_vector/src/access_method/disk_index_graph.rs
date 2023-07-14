use pgrx::PgRelation;


use crate::util::ItemPointer;

use super::{
    build::{read_meta_page, TsvMetaPage},
    graph::Graph,
    model::{NeighborWithDistance, Node, ReadableNode},
};

pub struct DiskIndexGraph {
    init_ids: Vec<ItemPointer>,
    meta_page: super::build::TsvMetaPage,
}

impl DiskIndexGraph {
    pub fn new(index: &PgRelation, init_ids: Vec<ItemPointer>) -> Self {
        let meta = unsafe { read_meta_page(index) };
        Self {
            init_ids: init_ids,
            meta_page: meta,
        }
    }
}

impl Graph for DiskIndexGraph {
    fn get_init_ids(&mut self) -> Option<Vec<ItemPointer>> {
        Some(self.init_ids.clone())
    }

    fn read(&self, index: &PgRelation, index_pointer: ItemPointer) -> ReadableNode {
        unsafe { Node::read(index, &index_pointer) }
    }

    fn get_neighbors(
        &self,
        index: &PgRelation,
        neighbors_of: ItemPointer,
    ) -> Option<Vec<NeighborWithDistance>> {
        let rn = self.read(index, neighbors_of);
        let mut ns = Vec::<NeighborWithDistance>::new();
        rn.get_archived_node().apply_to_neightbors(|dist, n| {
            ns.push(NeighborWithDistance::new(
                n.deserialize_item_pointer(),
                dist,
            ))
        });
        Some(ns)
    }

    fn get_meta_page(&self, _index: &PgRelation) -> &TsvMetaPage {
        &self.meta_page
    }
}
