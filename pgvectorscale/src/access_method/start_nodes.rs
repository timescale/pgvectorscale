use pgvectorscale_derive::{Readable, Writeable};
use rkyv::{Archive, Deserialize, Serialize};

use crate::access_method::labels::Label;
use crate::util::{ItemPointer, ReadableBuffer, WritableBuffer};
use pgrx::PgRelation;
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq, Archive, Deserialize, Serialize, Readable, Writeable)]
#[archive(check_bytes)]
pub struct StartNodes {
    /// Default starting node for the graph.
    default_node: ItemPointer,
    /// Labeled starting nodes for the graph
    labeled_nodes: HashMap<Label, ItemPointer>,
}

impl StartNodes {
    pub fn new(default_node: ItemPointer) -> Self {
        Self {
            default_node,
            labeled_nodes: HashMap::new(),
        }
    }

    pub fn add_node(&mut self, labels: Option<&[Label]>, node: ItemPointer) -> bool {
        let mut added = false;
        if let Some(labels) = labels {
            for label in labels {
                if !self.labeled_nodes.contains_key(label) {
                    added = true;
                    self.labeled_nodes.insert(*label, node);
                }
            }
        }
        added
    }

    pub fn contains(&self, labels: Option<&[Label]>) -> bool {
        match labels {
            Some(labels) => labels
                .iter()
                .all(|label| self.labeled_nodes.contains_key(label)),
            None => true,
        }
    }

    pub fn contains_one(&self, label: Label) -> bool {
        self.labeled_nodes.contains_key(&label)
    }

    pub fn get_for_node(&self, labels: Option<&[Label]>) -> Vec<ItemPointer> {
        match labels {
            Some(labels) => labels
                .iter()
                .filter_map(|label| self.labeled_nodes.get(label).copied())
                .collect(),
            None => vec![self.default_node],
        }
    }

    pub fn get_default_node(&self) -> ItemPointer {
        self.default_node
    }
}
