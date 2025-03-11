//! Debugging methods

use std::collections::HashMap;

use pgrx::PgRelation;

use crate::access_method::node::ReadableNode;
use crate::util::ItemPointer;

use super::{plain_node::PlainNode, stats::GreedySearchStats};

#[allow(dead_code)]
pub fn print_graph_from_disk(index: &PgRelation, init_id: ItemPointer) {
    let mut map = HashMap::<ItemPointer, Vec<f32>>::new();
    let mut sb = String::new();
    unsafe {
        print_graph_from_disk_visitor(index, init_id, &mut map, &mut sb);
    }
    panic!("{}", sb.as_str())
}

unsafe fn print_graph_from_disk_visitor(
    index: &PgRelation,
    index_pointer: ItemPointer,
    map: &mut HashMap<ItemPointer, Vec<f32>>,
    sb: &mut String,
) {
    let mut stats = GreedySearchStats::new();
    let data_node = PlainNode::read(index, index_pointer, &mut stats);
    let node = data_node.get_archived_node();
    let v = node.vector.as_slice();
    let copy: Vec<f32> = v.to_vec();
    let name = format!("node {:?}", &copy);

    map.insert(index_pointer, copy);

    for neighbor_pointer in node.iter_neighbors() {
        let p = neighbor_pointer;
        if !map.contains_key(&p) {
            print_graph_from_disk_visitor(index, p, map, sb);
        }
    }
    sb.push_str(&name);
    sb.push('\n');

    for neighbor_pointer in node.iter_neighbors() {
        let neighbor = map.get(&neighbor_pointer).unwrap();
        sb.push_str(&format!("->{:?}\n", neighbor))
    }
    sb.push('\n')
}
