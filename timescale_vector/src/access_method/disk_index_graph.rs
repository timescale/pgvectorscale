use pgrx::PgRelation;

use crate::util::ItemPointer;

use super::{
    graph::{Graph, VectorProvider},
    meta_page::MetaPage,
    model::{ArchivedNode, NeighborWithDistance, Node, ReadableNode},
};

pub struct DiskIndexGraph<'a> {
    meta_page: MetaPage,
    vector_provider: VectorProvider<'a>,
}

impl<'a> DiskIndexGraph<'a> {
    pub fn new(index: &PgRelation, vp: VectorProvider<'a>) -> Self {
        let meta = MetaPage::read(index);
        Self {
            meta_page: meta,
            vector_provider: vp,
        }
    }
}

impl<'h> Graph for DiskIndexGraph<'h> {
    fn get_vector_provider(&self) -> VectorProvider {
        return self.vector_provider.clone();
    }

    fn read<'a>(&self, index: &'a PgRelation, index_pointer: ItemPointer) -> ReadableNode<'a> {
        unsafe { Node::read(index, index_pointer) }
    }

    fn get_init_ids(&mut self) -> Option<Vec<ItemPointer>> {
        self.meta_page.get_init_ids()
    }

    fn get_neighbors(&self, node: &ArchivedNode, _neighbors_of: ItemPointer) -> Vec<ItemPointer> {
        let mut result = Vec::with_capacity(node.num_neighbors());
        node.apply_to_neighbors(|n| {
            let n = n.deserialize_item_pointer();
            result.push(n)
        });
        result
    }

    fn get_neighbors_with_distances(
        &self,
        index: &PgRelation,
        neighbors_of: ItemPointer,
        result: &mut Vec<NeighborWithDistance>,
    ) -> bool {
        let rn = self.read(index, neighbors_of);
        let vp = self.get_vector_provider();
        let dist_state = unsafe { vp.get_full_vector_distance_state(index, neighbors_of) };
        rn.get_archived_node().apply_to_neighbors(|n| {
            let n = n.deserialize_item_pointer();
            let dist =
                unsafe { vp.get_distance_pair_for_full_vectors_from_state(&dist_state, index, n) };
            result.push(NeighborWithDistance::new(n, dist))
        });
        true
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
        if self.meta_page.get_init_ids().is_none() {
            MetaPage::update_init_ids(index, vec![neighbors_of]);
            self.meta_page = MetaPage::read(index);
        }

        unsafe {
            let node = Node::modify(index, neighbors_of);
            let archived = node.get_archived_node();
            archived.set_neighbors(&new_neighbors, &self.meta_page);
            node.commit();
        }
    }
}
