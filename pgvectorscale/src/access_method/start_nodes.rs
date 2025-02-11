use pgvectorscale_derive::{Readable, Writeable};
use rkyv::{Archive, Deserialize, Serialize};

use crate::access_method::labels::{Label, LabelSet};
use crate::util::{ItemPointer, ReadableBuffer, WritableBuffer};
use pgrx::PgRelation;
use std::collections::HashMap;

use super::labels::LabelSetView;

const OVERLOAD_THRESHHOLD: usize = 10;

/// Start nodes for the graph.  For unlabeled vectorsets, this is a single node.  For
/// labeled vectorsets, this is a map of labels to nodes.
#[derive(Clone, Debug, PartialEq, Archive, Deserialize, Serialize, Readable, Writeable)]
#[archive(check_bytes)]
pub struct StartNodes {
    /// Default starting node for the graph.
    default_node: ItemPointer,
    /// Labeled starting nodes for the graph
    labeled_nodes: HashMap<Label, ItemPointer>,
    /// Load for a start node
    node_count: HashMap<ItemPointer, usize>,
}

impl StartNodes {
    pub fn new(default_node: ItemPointer) -> Self {
        Self {
            default_node,
            labeled_nodes: HashMap::new(),
            node_count: HashMap::new(),
        }
    }

    pub fn is_overloaded(&self, label: ItemPointer) -> bool {
        self.node_count
            .get(&label)
            .is_some_and(|count| *count > OVERLOAD_THRESHHOLD)
    }

    pub fn insert(&mut self, label: Label, node: ItemPointer) {
        self.labeled_nodes.insert(label, node);
        *self.node_count.entry(node).or_insert(0) += 1;
    }

    pub fn count_for_node(&self, node: ItemPointer) -> usize {
        *self.node_count.get(&node).unwrap_or(&0)
    }

    pub fn default_node(&self) -> ItemPointer {
        self.default_node
    }

    pub fn node_for_label(&self, label: Label) -> Option<ItemPointer> {
        self.labeled_nodes.get(&label).copied()
    }

    pub fn node_for_labels(&self, labels: &LabelSet) -> Vec<ItemPointer> {
        if labels.is_empty() {
            vec![self.default_node]
        } else {
            labels
                .iter()
                .filter_map(|label| self.labeled_nodes.get(label).copied())
                .collect()
        }
    }
}
