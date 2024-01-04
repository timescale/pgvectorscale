use std::collections::HashMap;
use std::time::Instant;

use pgrx::*;

use crate::util::{IndexPointer, ItemPointer};

use super::graph::{Graph, LsrPrivateData, NodeNeighbor};
use super::meta_page::MetaPage;
use super::model::*;
use super::quantizer::Quantizer;

/// A builderGraph is a graph that keep the neighbors in-memory in the neighbor_map below
/// The idea is that during the index build, you don't want to update the actual Postgres
/// pages every time you change the neighbors. Instead you change the neighbors in memory
/// until the build is done. Afterwards, calling the `write` method, will write out all
/// the neighbors to the right pages.
pub struct BuilderGraph {
    //maps node's pointer to the representation on disk
    neighbor_map: HashMap<ItemPointer, Vec<NeighborWithDistance>>,
    meta_page: MetaPage,
}

impl BuilderGraph {
    pub fn new(meta_page: MetaPage) -> Self {
        Self {
            neighbor_map: HashMap::with_capacity(200),
            meta_page,
        }
    }

    pub unsafe fn write(&self, index: &PgRelation, quantizer: &Quantizer) -> WriteStats {
        let mut stats = WriteStats::new();
        let meta = self.get_meta_page(index);

        //TODO: OPT: do this in order of item pointers
        for (index_pointer, neighbors) in &self.neighbor_map {
            stats.num_nodes += 1;
            let prune_neighbors;
            let neighbors = if neighbors.len() > self.meta_page.get_num_neighbors() as _ {
                stats.num_prunes += 1;
                stats.num_neighbors_before_prune += neighbors.len();
                (prune_neighbors, _) =
                    self.prune_neighbors(index, *index_pointer, vec![], quantizer);
                stats.num_neighbors_after_prune += prune_neighbors.len();
                &prune_neighbors
            } else {
                neighbors
            };
            stats.num_neighbors += neighbors.len();

            match quantizer {
                Quantizer::None => {
                    error!("Quantizer::None not implemented")
                    /* need to update the neighbors */
                }
                Quantizer::PQ(pq) => {
                    error!("Quantizer::None not implemented");
                    //pq.update_node_after_traing(index, &meta, *index_pointer, neighbors);
                }
                Quantizer::BQ(bq) => {
                    //TODO: OPT: this may not be needed
                    bq.update_node_after_traing(index, &meta, *index_pointer, neighbors);
                }
            };
        }
        stats
    }
}

impl Graph for BuilderGraph {
    fn get_init_ids(&self) -> Option<Vec<ItemPointer>> {
        //returns a vector for generality
        self.meta_page.get_init_ids()
    }

    fn get_neighbors<N: NodeNeighbor>(
        &self,
        _node: &N,
        neighbors_of: ItemPointer,
    ) -> Vec<IndexPointer> {
        let neighbors = self.neighbor_map.get(&neighbors_of);
        match neighbors {
            Some(n) => n
                .iter()
                .map(|n| n.get_index_pointer_to_neighbor())
                .collect(),
            None => vec![],
        }
    }

    fn get_neighbors_with_distances(
        &self,
        _index: &PgRelation,
        neighbors_of: ItemPointer,
        _quantizer: &Quantizer,
        result: &mut Vec<NeighborWithDistance>,
    ) -> bool {
        let neighbors = self.neighbor_map.get(&neighbors_of);
        match neighbors {
            Some(n) => {
                for nwd in n {
                    result.push(nwd.clone());
                }
                true
            }
            None => false,
        }
    }

    fn is_empty(&self) -> bool {
        self.neighbor_map.len() == 0
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
            //TODO probably better set off of centeroids
            MetaPage::update_init_ids(index, vec![neighbors_of]);
            self.meta_page = MetaPage::read(index);
        }
        self.neighbor_map.insert(neighbors_of, new_neighbors);
    }
}

pub struct WriteStats {
    pub started: Instant,
    pub num_nodes: usize,
    pub num_prunes: usize,
    pub num_neighbors_before_prune: usize,
    pub num_neighbors_after_prune: usize,
    pub num_neighbors: usize,
}

impl WriteStats {
    pub fn new() -> Self {
        Self {
            started: Instant::now(),
            num_nodes: 0,
            num_prunes: 0,
            num_neighbors_before_prune: 0,
            num_neighbors_after_prune: 0,
            num_neighbors: 0,
        }
    }
}
