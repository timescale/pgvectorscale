use pgrx::PgRelation;

use crate::util::ItemPointer;

use super::{
    graph::{Graph, LsrPrivateData, NodeNeighbor},
    meta_page::MetaPage,
    model::{ArchivedNode, NeighborWithDistance, Node, ReadableNode},
    quantizer::{self, Quantizer},
};

pub struct DiskIndexGraph {
    meta_page: MetaPage,
}

impl DiskIndexGraph {
    pub fn new(index: &PgRelation) -> Self {
        let meta = MetaPage::read(index);
        Self { meta_page: meta }
    }

    fn read<'b>(&self, index: &'b PgRelation, index_pointer: ItemPointer) -> ReadableNode<'b> {
        unsafe { Node::read(index, index_pointer) }
    }
}

impl Graph for DiskIndexGraph {
    fn get_init_ids(&self) -> Option<Vec<ItemPointer>> {
        self.meta_page.get_init_ids()
    }

    fn get_neighbors<N: NodeNeighbor>(
        &self,
        node: &N,
        _neighbors_of: ItemPointer,
    ) -> Vec<ItemPointer> {
        node.get_index_pointer_to_neighbors()
    }

    fn get_neighbors_with_distances(
        &self,
        index: &PgRelation,
        neighbors_of: ItemPointer,
        quantizer: &Quantizer,
        result: &mut Vec<NeighborWithDistance>,
    ) -> bool {
        quantizer.get_neighbors_with_distances(index, neighbors_of, result)
    }

    fn is_empty(&self) -> bool {
        self.meta_page.get_init_ids().is_none()
    }

    fn get_meta_page(&self, _index: &PgRelation) -> &MetaPage {
        &self.meta_page
    }

    fn set_neighbors(
        &mut self,
        index: &PgRelation,
        neighbors_of: ItemPointer,
        new_neighbors: Vec<NeighborWithDistance>,
    ) {
        pgrx::error!("disk index graph set neighbor not implemented")
        /*if self.meta_page.get_init_ids().is_none() {
            MetaPage::update_init_ids(index, vec![neighbors_of]);
            self.meta_page = MetaPage::read(index);
        }

        unsafe {
            let node = Node::modify(index, neighbors_of);
            let archived = node.get_archived_node();
            archived.set_neighbors(&new_neighbors, &self.meta_page);
            node.commit();
        }*/
    }
}
