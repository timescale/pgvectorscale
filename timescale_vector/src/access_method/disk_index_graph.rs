use pgrx::PgRelation;

use crate::util::ItemPointer;

use super::{
    graph::NodeNeighbor, meta_page::MetaPage, model::NeighborWithDistance, quantizer::Quantizer,
};

pub struct DiskIndexGraph {}

impl DiskIndexGraph {
    pub fn new() -> Self {
        Self {}
    }

    pub fn get_neighbors<N: NodeNeighbor>(&self, node: &N) -> Vec<ItemPointer> {
        node.get_index_pointer_to_neighbors()
    }

    pub fn get_neighbors_with_distances(
        &self,
        index: &PgRelation,
        neighbors_of: ItemPointer,
        quantizer: &Quantizer,
        result: &mut Vec<NeighborWithDistance>,
    ) -> bool {
        quantizer.get_neighbors_with_distances(index, neighbors_of, result)
    }

    pub fn is_empty(&self, meta_page: &MetaPage) -> bool {
        meta_page.get_init_ids().is_none()
    }

    pub fn set_neighbors(
        &mut self,
        _index: &PgRelation,
        _neighbors_of: ItemPointer,
        _new_neighbors: Vec<NeighborWithDistance>,
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
