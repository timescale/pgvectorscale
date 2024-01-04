use super::{
    distance::distance_cosine as default_distance,
    graph::{
        self, Graph, GraphNeighborStore, GreedySearchStats, ListSearchNeighbor, ListSearchResult,
        NodeNeighbor, SearchDistanceMeasure,
    },
};
use std::{pin::Pin, thread::Builder};

use pgrx::{
    pg_sys::{InvalidBlockNumber, InvalidOffsetNumber, Node},
    PgRelation,
};
use rkyv::{vec::ArchivedVec, Archive, Archived, Deserialize, Serialize};

use crate::util::{
    page::{self, PageType},
    tape::{self, Tape},
    ArchivedItemPointer, HeapPointer, IndexPointer, ItemPointer, ReadableBuffer,
};

use super::{
    graph::FullVectorDistanceState,
    graph::TableSlot,
    meta_page::{self, MetaPage},
    model::{NeighborWithDistance, PgVector},
    quantizer,
};
use crate::util::WritableBuffer;

type BqVectorElement = u8;
const BITS_STORE_TYPE_SIZE: usize = 8;

#[derive(Archive, Deserialize, Serialize)]
#[archive(check_bytes)]
#[repr(C)]
pub struct BqMeans {
    count: u64,
    means: Vec<f32>,
}

impl BqMeans {
    pub unsafe fn write(&self, tape: &mut Tape) -> ItemPointer {
        let bytes = rkyv::to_bytes::<_, 8192>(self).unwrap();
        tape.write(&bytes)
    }
    pub unsafe fn read<'a>(
        index: &'a PgRelation,
        index_pointer: &ItemPointer,
    ) -> ReadableBqMeans<'a> {
        let rb = index_pointer.read_bytes(index);
        ReadableBqMeans { _rb: rb }
    }
}

//ReadablePqNode ties an archive node to it's underlying buffer
pub struct ReadableBqMeans<'a> {
    _rb: ReadableBuffer<'a>,
}

impl<'a> ReadableBqMeans<'a> {
    pub fn get_archived_node(&self) -> &ArchivedBqMeans {
        // checking the code here is expensive during build, so skip it.
        // TODO: should we check the data during queries?
        //rkyv::check_archived_root::<Node>(self._rb.get_data_slice()).unwrap()
        unsafe { rkyv::archived_root::<BqMeans>(self._rb.get_data_slice()) }
    }
}

pub unsafe fn read_bq(index: &PgRelation, index_pointer: &IndexPointer) -> (u64, Vec<f32>) {
    let rpq = BqMeans::read(index, &index_pointer);
    let rpn = rpq.get_archived_node();
    (rpn.count, rpn.means.as_slice().to_vec())
}

pub unsafe fn write_bq(index: &PgRelation, count: u64, means: &[f32]) -> ItemPointer {
    let mut tape = Tape::new(index, PageType::BqMeans);
    let node = BqMeans {
        count,
        means: means.to_vec(),
    };
    let ptr = node.write(&mut tape);
    tape.close();
    ptr
}

pub struct BqQuantizer<'a> {
    use_mean: bool,
    training: bool,
    count: u64,
    mean: Vec<f32>,
    distance_fn: fn(&[f32], &[f32]) -> f32,
    heap_rel: Option<&'a PgRelation>,
    heap_attr: Option<pgrx::pg_sys::AttrNumber>,
}

impl<'a> BqQuantizer<'a> {
    pub fn new(
        heap_rel: Option<&'a PgRelation>,
        heap_attr: Option<pgrx::pg_sys::AttrNumber>,
    ) -> BqQuantizer<'a> {
        Self {
            use_mean: true,
            training: false,
            count: 0,
            mean: vec![],
            distance_fn: default_distance,
            heap_rel: heap_rel,
            heap_attr: heap_attr,
        }
    }

    pub fn load(&mut self, index_relation: &PgRelation, meta_page: &super::meta_page::MetaPage) {
        if self.use_mean {
            if meta_page.get_pq_pointer().is_none() {
                pgrx::error!("No PQ pointer found in meta page");
            }
            let pq_item_pointer = meta_page.get_pq_pointer().unwrap();
            (self.count, self.mean) = unsafe { read_bq(&index_relation, &pq_item_pointer) };
        }
    }

    /* get_lsn and visit_lsn are different because the distance
    comparisons for BQ get the vector from different places */
    pub fn get_lsn(
        &self,
        lsr: &mut ListSearchResult,
        index: &PgRelation,
        index_pointer: ItemPointer,
        query: &[f32],
    ) -> ListSearchNeighbor {
        let rn = unsafe { BqNode::read(index, index_pointer) };
        lsr.stats.node_reads += 1;
        let node = rn.get_archived_node();

        let distance = match &lsr.sdm {
            SearchDistanceMeasure::Full(distance_fn) => {
                let heap_pointer = node.heap_item_pointer.deserialize_item_pointer();
                let slot = unsafe {
                    TableSlot::new(
                        self.heap_rel.unwrap(),
                        heap_pointer,
                        self.heap_attr.unwrap(),
                    )
                };
                let slice = unsafe { slot.get_slice() };
                let distance = distance_fn(slice, query);
                lsr.stats.distance_comparisons += 1;
                distance
            }
            SearchDistanceMeasure::Bq(table) => {
                assert!(node.bq_vector.len() > 0);
                let vec = node.bq_vector.as_slice();
                let distance = table.distance(vec);
                lsr.stats.pq_distance_comparisons += 1;
                lsr.stats.distance_comparisons += 1;
                distance
            }
            _ => {
                pgrx::error!("wrong distance measure");
            }
        };

        let lsn =
            ListSearchNeighbor::new(index_pointer, distance, super::graph::LsrPrivateData::None);

        lsn
    }

    pub fn visit_lsn(
        &self,
        index: &PgRelation,
        lsr: &mut ListSearchResult,
        lsn_idx: usize,
        query: &[f32],
        gns: &GraphNeighborStore,
    ) {
        //Opt shouldn't need to read the node in the builder graph case.
        let lsn = &lsr.candidate_storage[lsn_idx];
        let rn = unsafe { BqNode::read(index, lsn.index_pointer) };
        lsr.stats.node_reads += 1;
        let node = rn.get_archived_node();

        let neighbors = match gns {
            GraphNeighborStore::Disk(d) => d.get_neighbors(node),
            GraphNeighborStore::Builder(b) => b.get_neighbors(lsn.index_pointer),
        };

        for (i, &neighbor_index_pointer) in neighbors.iter().enumerate() {
            if !lsr.prepare_insert(neighbor_index_pointer) {
                continue;
            }

            let dist = match &lsr.sdm {
                SearchDistanceMeasure::Full(distance_fn) => {
                    let rn = unsafe { BqNode::read(index, neighbor_index_pointer) };
                    lsr.stats.node_reads += 1;
                    let node = rn.get_archived_node();
                    let heap_pointer = node.heap_item_pointer.deserialize_item_pointer();
                    let slot = unsafe {
                        TableSlot::new(
                            self.heap_rel.unwrap(),
                            heap_pointer,
                            self.heap_attr.unwrap(),
                        )
                    };
                    let slice = unsafe { slot.get_slice() };
                    let distance = distance_fn(slice, query);
                    lsr.stats.distance_comparisons += 1;
                    distance
                }
                SearchDistanceMeasure::Bq(table) => {
                    if let GraphNeighborStore::Builder(_) = gns {
                        assert!(
                            false,
                            "BQ distance should not be used with the builder graph store"
                        )
                    }
                    let bq_vector = node.neighbor_vectors[i].as_slice();
                    assert!(bq_vector.len() > 0);
                    let vec = bq_vector;
                    let distance = table.distance(vec);
                    lsr.stats.pq_distance_comparisons += 1;
                    lsr.stats.distance_comparisons += 1;
                    distance
                }
                _ => {
                    pgrx::error!("wrong distance measure");
                }
            };
            let lsn = ListSearchNeighbor::new(
                neighbor_index_pointer,
                dist,
                super::graph::LsrPrivateData::None,
            );

            lsr.insert_neighbor(lsn);
        }

        /*
        //Opt shouldn't need to read the node in the builder graph case.
        let rn = unsafe { BqNode::read(index, lsn.index_pointer) };
        stats.node_reads += 1;
        let node = rn.get_archived_node();

        let neighbors = graph.get_neighbors(node, lsn.index_pointer);
        for neighbor_index_pointer in neighbors.iter() {
            //todo : check if the already visited lsn
            let lsn = self.get_lsn(stats, dm, index, *neighbor_index_pointer, bq_vector, query)

        }

        stats.node_reads += 1;
        */
    }

    pub fn return_lsn(
        &self,
        index: &PgRelation,
        lsr: &mut ListSearchResult,
        idx: usize,
    ) -> (HeapPointer, IndexPointer) {
        let lsn = &lsr.candidate_storage[idx];
        let rn = unsafe { BqNode::read(index, lsn.index_pointer) };
        lsr.stats.node_reads += 1;
        let node = rn.get_archived_node();
        let heap_pointer = node.heap_item_pointer.deserialize_item_pointer();
        (heap_pointer, lsn.index_pointer)
    }

    pub fn update_node_after_traing(
        &self,
        index: &PgRelation,
        meta: &MetaPage,
        index_pointer: IndexPointer,
        neighbors: &Vec<NeighborWithDistance>,
    ) {
        let node = unsafe { BqNode::modify(index, index_pointer) };
        let mut archived = node.get_archived_node();
        archived
            .as_mut()
            .set_neighbors(neighbors, &meta, index, self);

        if self.use_mean {
            let heap_pointer = node
                .get_archived_node()
                .heap_item_pointer
                .deserialize_item_pointer();
            let bq_vector = self.get_quantized_vector_from_heap_pointer(heap_pointer);

            assert!(bq_vector.len() == archived.bq_vector.len());
            for i in 0..=bq_vector.len() - 1 {
                let mut pgv = archived.as_mut().bq_vectors().index_pin(i);
                *pgv = bq_vector[i];
            }
        }
        node.commit();
    }

    pub fn create_node(
        &self,
        index_relation: &PgRelation,
        full_vector: &[f32],
        heap_pointer: HeapPointer,
        meta_page: &MetaPage,
        tape: &mut Tape,
    ) -> ItemPointer {
        let bq_vector = if self.use_mean && self.training {
            vec![0; Self::quantized_size(meta_page.get_num_dimensions() as _)]
        } else {
            self.quantize(&full_vector)
        };

        let node = BqNode::new(heap_pointer, &meta_page, bq_vector.as_slice());

        let index_pointer: IndexPointer = node.write(tape);
        index_pointer
    }

    pub fn start_training(&mut self, meta_page: &super::meta_page::MetaPage) {
        self.training = true;
        if self.use_mean {
            self.count = 0;
            self.mean = vec![0.0; meta_page.get_num_dimensions() as _];
        }
    }

    pub fn add_sample(&mut self, sample: &[f32]) {
        if self.use_mean {
            self.count += 1;
            assert!(self.mean.len() == sample.len());

            self.mean
                .iter_mut()
                .zip(sample.iter())
                .for_each(|(m, s)| *m += ((s - *m) / self.count as f32));
        }
    }

    pub fn finish_training(&mut self) {
        self.training = false;
    }

    pub fn write_metadata(&self, index: &PgRelation) {
        if self.use_mean {
            let index_pointer = unsafe { write_bq(&index, self.count, &self.mean) };
            super::meta_page::MetaPage::update_pq_pointer(&index, index_pointer);
        }
    }

    fn quantized_size(full_vector_size: usize) -> usize {
        if full_vector_size % BITS_STORE_TYPE_SIZE == 0 {
            full_vector_size / BITS_STORE_TYPE_SIZE
        } else {
            (full_vector_size / BITS_STORE_TYPE_SIZE) + 1
        }
    }

    pub fn quantize(&self, full_vector: &[f32]) -> Vec<BqVectorElement> {
        if self.use_mean {
            let mut res_vector = vec![0; Self::quantized_size(full_vector.len())];

            for (i, &v) in full_vector.iter().enumerate() {
                if v > self.mean[i] {
                    res_vector[i / BITS_STORE_TYPE_SIZE] |= 1 << (i % BITS_STORE_TYPE_SIZE);
                }
            }

            res_vector
        } else {
            let mut res_vector = vec![0; Self::quantized_size(full_vector.len())];

            for (i, &v) in full_vector.iter().enumerate() {
                if v > 0.0 {
                    res_vector[i / BITS_STORE_TYPE_SIZE] |= 1 << (i % BITS_STORE_TYPE_SIZE);
                }
            }

            res_vector
        }
    }

    fn get_distance_table(
        &self,
        query: &[f32],
        _distance_fn: fn(&[f32], &[f32]) -> f32,
    ) -> BqDistanceTable {
        BqDistanceTable::new(self.quantize(query))
    }

    fn get_heap_pointer(&self, index: &PgRelation, index_pointer: IndexPointer) -> HeapPointer {
        let rn = unsafe { BqNode::read(index, index_pointer) };
        let node = rn.get_archived_node();
        node.heap_item_pointer.deserialize_item_pointer()
    }

    pub unsafe fn get_full_vector_distance_state<'i>(
        &self,
        index: &'i PgRelation,
        index_pointer: IndexPointer,
    ) -> FullVectorDistanceState<'i> {
        let heap_pointer = self.get_heap_pointer(index, index_pointer);
        let slot = TableSlot::new(
            self.heap_rel.unwrap(),
            heap_pointer,
            self.heap_attr.unwrap(),
        );
        FullVectorDistanceState::with_table_slot(slot)
    }

    pub unsafe fn get_distance_pair_for_full_vectors_from_state(
        &self,
        state: &FullVectorDistanceState,
        index: &PgRelation,
        index_pointer: IndexPointer,
    ) -> f32 {
        let heap_pointer = self.get_heap_pointer(index, index_pointer);
        let slot = TableSlot::new(
            self.heap_rel.unwrap(),
            heap_pointer,
            self.heap_attr.unwrap(),
        );
        let slice1 = slot.get_slice();
        let slice2 = state.get_table_slot().get_slice();
        (self.distance_fn)(slice1, slice2)
    }

    fn get_quantized_vector_from_heap_pointer(
        &self,
        heap_pointer: HeapPointer,
    ) -> Vec<BqVectorElement> {
        let slot = unsafe {
            TableSlot::new(
                self.heap_rel.unwrap(),
                heap_pointer,
                self.heap_attr.unwrap(),
            )
        };

        let slice = unsafe { slot.get_slice() };
        self.quantize(slice)
    }

    pub fn get_search_distance_measure(
        &self,
        query: &[f32],
        distance_fn: fn(&[f32], &[f32]) -> f32,
        calc_distance_with_quantizer: bool,
    ) -> SearchDistanceMeasure {
        if !calc_distance_with_quantizer {
            return SearchDistanceMeasure::Full(distance_fn);
        } else {
            return SearchDistanceMeasure::Bq(self.get_distance_table(query, distance_fn));
        }
    }

    pub fn get_neighbors_with_distances(
        &self,
        index: &PgRelation,
        neighbors_of: ItemPointer,
        result: &mut Vec<NeighborWithDistance>,
    ) -> bool {
        let rn = unsafe { BqNode::read(index, neighbors_of) };
        let dist_state = unsafe { self.get_full_vector_distance_state(index, neighbors_of) };
        rn.get_archived_node().apply_to_neighbors(|n| {
            let n = n.deserialize_item_pointer();
            let dist = unsafe {
                self.get_distance_pair_for_full_vectors_from_state(&dist_state, index, n)
            };
            result.push(NeighborWithDistance::new(n, dist))
        });
        true
    }
}

/// DistanceCalculator encapsulates the code to generate distances between a PQ vector and a query.
pub struct BqDistanceTable {
    quantized_vector: Vec<BqVectorElement>,
}

fn xor_unoptimized(v1: &[BqVectorElement], v2: &[BqVectorElement]) -> usize {
    let mut result = 0;
    for (b1, b2) in v1.iter().zip(v2.iter()) {
        result += (b1 ^ b2).count_ones() as usize;
    }
    result
}

impl BqDistanceTable {
    pub fn new(query: Vec<BqVectorElement>) -> BqDistanceTable {
        BqDistanceTable {
            quantized_vector: query,
        }
    }

    /// distance emits the sum of distances between each centroid in the quantized vector.
    pub fn distance(&self, bq_vector: &[BqVectorElement]) -> f32 {
        let count_ones = xor_unoptimized(&self.quantized_vector, bq_vector);
        //dot product is LOWER the more xors that lead to 1 becaues that means a negative times a positive = negative component
        //but the distance is 1 - dot product, so the more count_ones the higher the distance.
        // one other check for distance(a,a), xor=0, count_ones=0, distance=0
        count_ones as f32
    }
}

use timescale_vector_derive::{Readable, Writeable};

#[derive(Archive, Deserialize, Serialize, Readable, Writeable)]
#[archive(check_bytes)]
pub struct BqNode {
    pub heap_item_pointer: HeapPointer,
    pub bq_vector: Vec<BqVectorElement>,
    neighbor_index_pointers: Vec<ItemPointer>,
    neighbor_vectors: Vec<Vec<BqVectorElement>>,
}

struct BqNodeLsrPrivateData {}

impl BqNode {
    pub fn new(
        heap_pointer: HeapPointer,
        meta_page: &MetaPage,
        bq_vector: &[BqVectorElement],
    ) -> Self {
        let num_neighbors = meta_page.get_num_neighbors();
        // always use vectors of num_neighbors in length because we never want the serialized size of a Node to change
        let neighbor_index_pointers: Vec<_> = (0..num_neighbors)
            .map(|_| ItemPointer::new(InvalidBlockNumber, InvalidOffsetNumber))
            .collect();

        let neighbor_vectors: Vec<_> = (0..num_neighbors)
            .map(|_| vec![0; BqQuantizer::quantized_size(meta_page.get_num_dimensions() as _)])
            .collect();

        Self {
            heap_item_pointer: heap_pointer,
            bq_vector: bq_vector.to_vec(),
            neighbor_index_pointers: neighbor_index_pointers,
            neighbor_vectors: neighbor_vectors,
        }
    }
}

impl ArchivedBqNode {
    pub fn neighbor_index_pointer(
        self: Pin<&mut Self>,
    ) -> Pin<&mut ArchivedVec<ArchivedItemPointer>> {
        unsafe { self.map_unchecked_mut(|s| &mut s.neighbor_index_pointers) }
    }

    pub fn neighbor_vector(self: Pin<&mut Self>) -> Pin<&mut ArchivedVec<ArchivedVec<u8>>> {
        unsafe { self.map_unchecked_mut(|s| &mut s.neighbor_vectors) }
    }

    pub fn bq_vectors(self: Pin<&mut Self>) -> Pin<&mut Archived<Vec<u8>>> {
        unsafe { self.map_unchecked_mut(|s| &mut s.bq_vector) }
    }

    pub fn set_neighbors(
        mut self: Pin<&mut Self>,
        neighbors: &Vec<NeighborWithDistance>,
        meta_page: &MetaPage,
        index: &PgRelation,
        quantizer: &BqQuantizer,
    ) {
        for (i, new_neighbor) in neighbors.iter().enumerate() {
            let mut a_index_pointer = self.as_mut().neighbor_index_pointer().index_pin(i);
            let ip = new_neighbor.get_index_pointer_to_neighbor();
            //TODO hate that we have to set each field like this
            a_index_pointer.block_number = ip.block_number;
            a_index_pointer.offset = ip.offset;

            //FIXME: REALLY this is a hack to get around the fact that we don't have the neighbor vectors set anywhere yet
            //really slow and stupid to do it this way.
            //FIXME: also broken deadlock if ip uses same block as as self. Because get_heap_pointer will try to read the block.
            let heap_pointer = quantizer.get_heap_pointer(index, ip);
            let quantized = quantizer.get_quantized_vector_from_heap_pointer(heap_pointer);

            let mut neighbor_vector = self.as_mut().neighbor_vector().index_pin(i);
            for (index_in_q_vec, val) in quantized.iter().enumerate() {
                let mut x = neighbor_vector.as_mut().index_pin(index_in_q_vec);
                *x = *val;
            }
        }
        //set the marker that the list ended
        if neighbors.len() < meta_page.get_num_neighbors() as _ {
            let mut past_last_index_pointers =
                self.neighbor_index_pointer().index_pin(neighbors.len());
            past_last_index_pointers.block_number = InvalidBlockNumber;
            past_last_index_pointers.offset = InvalidOffsetNumber;
        }
    }

    pub fn num_neighbors(&self) -> usize {
        self.neighbor_index_pointers
            .iter()
            .position(|f| f.block_number == InvalidBlockNumber)
            .unwrap_or(self.neighbor_index_pointers.len())
    }

    pub fn apply_to_neighbors<F>(&self, mut f: F)
    where
        F: FnMut(&ArchivedItemPointer),
    {
        for i in 0..self.num_neighbors() {
            let neighbor = &self.neighbor_index_pointers[i];
            f(neighbor);
        }
    }
}

impl NodeNeighbor for ArchivedBqNode {
    fn get_index_pointer_to_neighbors(&self) -> Vec<ItemPointer> {
        let mut result = vec![];
        self.apply_to_neighbors(|n| result.push(n.deserialize_item_pointer()));
        result
    }
}
