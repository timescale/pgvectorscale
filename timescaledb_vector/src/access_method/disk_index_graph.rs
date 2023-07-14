use pgrx::PgRelation;
use rkyv::Deserialize;

use crate::util::ItemPointer;

use super::{
    build::{read_meta_page, TsvMetaPage},
    graph::Graph,
    model::{NeighborWithDistance, Node, ReadableNode},
};

pub struct DiskIndexGraph {
    index: PgRelation,
    init_ids: Vec<ItemPointer>,
    meta_page: super::build::TsvMetaPage,
}

impl DiskIndexGraph {
    pub fn new(index: PgRelation, init_ids: Vec<ItemPointer>) -> Self {
        let meta = unsafe { read_meta_page(index.clone()) };
        Self {
            index: index,
            init_ids: init_ids,
            meta_page: meta,
        }
    }
}

impl Graph for DiskIndexGraph {
    fn get_init_ids(&mut self) -> Option<Vec<ItemPointer>> {
        Some(self.init_ids.clone())
    }

    fn read<'b, 'd>(&'b self, index_pointer: ItemPointer) -> ReadableNode<'d> {
        unsafe { Node::read(&self.index, &index_pointer) }
    }

    fn get_neighbors(&self, neighbors_of: ItemPointer) -> Option<Vec<NeighborWithDistance>> {
        let rn = self.read(neighbors_of);
        let mut ns = Vec::<NeighborWithDistance>::new();
        rn.node.apply_to_neightbors(|dist, n| {
            ns.push(NeighborWithDistance::new(
                n.deserialize_item_pointer(),
                dist,
            ))
        });
        Some(ns)
    }

    fn get_meta_page(&self) -> &TsvMetaPage {
        &self.meta_page
    }
}
