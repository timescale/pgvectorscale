//! Debugging methods

use std::collections::HashMap;

use pgrx::PgRelation;
use rkyv::Deserialize;

use crate::util::ItemPointer;

use super::{plain_node::Node, stats::GreedySearchStats};

#[allow(dead_code)]
pub fn print_graph_from_disk(index: &PgRelation, init_id: ItemPointer) {
    let mut map = HashMap::<ItemPointer, Vec<f32>>::new();
    let mut sb = String::new();
    unsafe {
        print_graph_from_disk_visitor(&index, init_id, &mut map, &mut sb, 0);
    }
    panic!("{}", sb.as_str())
}

unsafe fn print_graph_from_disk_visitor(
    index: &PgRelation,
    index_pointer: ItemPointer,
    map: &mut HashMap<ItemPointer, Vec<f32>>,
    sb: &mut String,
    level: usize,
) {
    let mut stats = GreedySearchStats::new();
    let data_node = Node::read(&index, index_pointer, &mut stats);
    let node = data_node.get_archived_node();
    let v = node.vector.as_slice();
    let copy: Vec<f32> = v.iter().map(|f| *f).collect();
    let name = format!("node {:?}", &copy);

    map.insert(index_pointer, copy);

    for neighbor_pointer in node.iter_neighbors() {
        let p = neighbor_pointer;
        if !map.contains_key(&p) {
            print_graph_from_disk_visitor(index, p, map, sb, level + 1);
        }
    }
    sb.push_str(&name);
    sb.push_str("\n");

    for neighbor_pointer in node.iter_neighbors() {
        let neighbor = map.get(&neighbor_pointer).unwrap();
        sb.push_str(&format!("->{:?}\n", neighbor))
    }
    sb.push_str("\n")
}
