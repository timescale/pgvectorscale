use pgvectorscale_derive::{Readable, Writeable};
use rkyv::{Archive, Deserialize, Serialize};

use crate::access_method::labels::{Label, LabelSet};
use crate::access_method::node::{ReadableNode, WriteableNode};
use crate::util::{ItemPointer, ReadableBuffer, WritableBuffer};
use pgrx::PgRelation;
use std::collections::BTreeMap;

use super::labels::LabelSetView;

/// Start nodes for the graph.  For unlabeled vectorsets, this is a single node.  For
/// labeled vectorsets, this is a map of labels to nodes.
#[derive(Clone, Debug, PartialEq, Eq, Archive, Deserialize, Serialize, Readable, Writeable)]
#[archive(check_bytes)]
pub struct StartNodes {
    /// Default starting node for the graph.
    default_node: ItemPointer,
    /// Labeled starting nodes for the graph
    labeled_nodes: BTreeMap<Label, ItemPointer>,
}

impl StartNodes {
    pub fn new(default_node: ItemPointer) -> Self {
        Self {
            default_node,
            labeled_nodes: BTreeMap::new(),
        }
    }

    pub fn upsert(&mut self, label: Label, node: ItemPointer) -> Option<ItemPointer> {
        self.labeled_nodes.insert(label, node)
    }

    pub fn default_node(&self) -> ItemPointer {
        self.default_node
    }

    pub fn get_for_node(&self, labels: Option<&LabelSet>) -> Vec<ItemPointer> {
        if let Some(labels) = labels {
            labels
                .iter()
                .filter_map(|label| self.labeled_nodes.get(label).copied())
                .collect()
        } else {
            vec![self.default_node]
        }
    }

    pub fn contains(&self, label: Label) -> bool {
        self.labeled_nodes.contains_key(&label)
    }

    pub fn contains_all(&self, labels: Option<&LabelSet>) -> bool {
        match labels {
            Some(labels) => labels
                .iter()
                .all(|label| self.labeled_nodes.contains_key(label)),
            None => true,
        }
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

    pub fn get_all_labeled_nodes(&self) -> Vec<(Option<Label>, ItemPointer)> {
        let mut nodes = vec![(None, self.default_node)];
        nodes.extend(
            self.labeled_nodes
                .iter()
                .map(|(label, node)| (Some(*label), *node)),
        );
        nodes
    }

    pub fn get_all_nodes(&self) -> Vec<ItemPointer> {
        let mut nodes = vec![self.default_node];
        nodes.extend(self.labeled_nodes.values().copied());
        nodes
    }
}
