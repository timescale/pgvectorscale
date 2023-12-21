use std::pin::Pin;

use pgrx::PgRelation;

use super::meta_page::MetaPage;

const BITS_STORE_TYPE_SIZE: usize = 8;

pub struct BqQuantizer {}

impl BqQuantizer {
    pub fn new() -> BqQuantizer {
        Self {}
    }

    pub fn load(&mut self, index_relation: &PgRelation, meta_page: &super::meta_page::MetaPage) {}

    pub fn initialize_node(
        &self,
        node: &mut super::model::Node,
        meta_page: &MetaPage,
        full_vector: Vec<f32>,
    ) {
        node.pq_vector = self.quantize(&full_vector);
    }

    pub fn update_node_after_traing(
        &self,
        archived: &mut Pin<&mut super::model::ArchivedNode>,
        full_vector: Vec<f32>,
    ) {
    }

    pub fn start_training(&mut self, meta_page: &super::meta_page::MetaPage) {}

    pub fn add_sample(&mut self, sample: &[f32]) {}

    pub fn finish_training(&mut self) {}

    pub fn write_metadata(&self, index: &PgRelation) {}

    fn quantized_size(full_vector_size: usize) -> usize {
        if full_vector_size % BITS_STORE_TYPE_SIZE == 0 {
            full_vector_size / BITS_STORE_TYPE_SIZE
        } else {
            (full_vector_size / BITS_STORE_TYPE_SIZE) + 1
        }
    }

    pub fn quantize(&self, full_vector: &[f32]) -> Vec<u8> {
        let mut res_vector = vec![0; Self::quantized_size(full_vector.len())];

        for (i, &v) in full_vector.iter().enumerate() {
            if v > 0.0 {
                res_vector[i / BITS_STORE_TYPE_SIZE] |= 1 << (i % BITS_STORE_TYPE_SIZE);
            }
        }

        res_vector
    }

    pub fn get_distance_table(
        &self,
        query: &[f32],
        distance_fn: fn(&[f32], &[f32]) -> f32,
    ) -> BqDistanceTable {
        BqDistanceTable::new(self.quantize(query))
    }
}

/// DistanceCalculator encapsulates the code to generate distances between a PQ vector and a query.
pub struct BqDistanceTable {
    quantized_vector: Vec<u8>,
}

fn xor_unoptimized(v1: &[u8], v2: &[u8]) -> usize {
    let mut result = 0;
    for (b1, b2) in v1.iter().zip(v2.iter()) {
        result += (b1 ^ b2).count_ones() as usize;
    }
    result
}

impl BqDistanceTable {
    pub fn new(query: Vec<u8>) -> BqDistanceTable {
        BqDistanceTable {
            quantized_vector: query,
        }
    }

    /// distance emits the sum of distances between each centroid in the quantized vector.
    pub fn distance(&self, bq_vector: &[u8]) -> f32 {
        let count_ones = xor_unoptimized(&self.quantized_vector, bq_vector);
        //dot product is LOWER the more xors that lead to 1 becaues that means a negative times a positive = negative component
        //but the distance is 1 - dot product, so the more count_ones the higher the distance.
        // one other check for distance(a,a), xor=0, count_ones=0, distance=0
        count_ones as f32
    }
}
