//! Debugging methods

use std::collections::HashMap;

use pgrx::PgRelation;
use rkyv::Deserialize;

use crate::util::ItemPointer;

use super::model::Node;

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
    let data_node = Node::read(&index, &index_pointer);
    let node = data_node.get_archived_node();
    let v = node.vector.as_slice();
    let copy: Vec<f32> = v.iter().map(|f| *f).collect();
    let name = format!("node {:?}", &copy);

    map.insert(index_pointer, copy);

    node.apply_to_neightbors(|_dist, neighbor_pointer| {
        let p = neighbor_pointer.deserialize_item_pointer();
        if !map.contains_key(&p) {
            print_graph_from_disk_visitor(index, p, map, sb, level + 1);
        }
    });
    sb.push_str(&name);
    sb.push_str("\n");
    node.apply_to_neightbors(|dist, neighbor_pointer| {
        let ip: ItemPointer = (neighbor_pointer)
            .deserialize(&mut rkyv::Infallible)
            .unwrap();
        let neighbor = map.get(&ip).unwrap();
        sb.push_str(&format!("->{:?} dist({})\n", neighbor, dist))
    });
    sb.push_str("\n")
}
