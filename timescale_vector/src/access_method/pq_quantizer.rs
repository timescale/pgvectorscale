use ndarray::{Array1, Array2, Axis};
use pgrx::{error, notice, PgRelation};
use rand::Rng;
use reductive::pq::{Pq, QuantizeVector, TrainPq};

use crate::access_method::{
    distance::distance_l2_optimized_for_few_dimensions, pq_quantizer_storage::read_pq,
};

use super::{
    meta_page::MetaPage,
    stats::{StatsDistanceComparison, StatsNodeRead},
};

/// pq aka Product quantization (PQ) is one of the most widely used algorithms for memory-efficient approximated nearest neighbor search,
/// This module encapsulates a vanilla implementation of PQ that we use for the vector index.
/// More details: https://lear.inrialpes.fr/pubs/2011/JDS11/jegou_searching_with_quantization.pdf

/// PQ_TRAINING_ITERATIONS is the number of times we train each independent Kmeans cluster.
/// 20 - 40 iterations is considered an industry best practice
/// https://github.com/matsui528/nanopq/blob/main/nanopq/pq.py#L60
const PQ_TRAINING_ITERATIONS: usize = 20;

/// NUM_SUBQUANTIZER_BITS is the number of code words used for quantization. We pin it to 8 so we can
/// use u8 to represent a subspace.
const NUM_SUBQUANTIZER_BITS: u32 = 8;

/// NUM_TRAINING_ATTEMPTS is the number of times we'll attempt to train the quantizer.
const NUM_TRAINING_ATTEMPTS: usize = 1;

/// NUM_TRAINING_SET_SIZE is the maximum number of vectors we want to consider for the quantizer training set.
/// We pick a value used by DiskANN implementations.
const NUM_TRAINING_SET_SIZE: usize = 256000;

pub type PqVectorElement = u8;
/// PqTrainer is a utility that produces a product quantizer from training with sample vectors.

#[derive(Clone)]
pub struct PqTrainer {
    /// training_set contains the vectors we'll use to train PQ.
    training_set: Vec<Vec<f32>>,
    /// considered_samples is the number of samples we considered for the training set.
    /// It is useful for reservoir sampling as we add samples.
    considered_samples: usize,
    /// num_subquantizers is the number of independent kmeans we want to partition the vectors into.
    /// the more we have the more accurate the PQ, but the more space we use in memory.
    num_subquantizers: usize,
    // rng is the random number generator for reservoir sampling
    //rng: ThreadRng,
}

impl PqTrainer {
    pub fn new(meta_page: &super::meta_page::MetaPage) -> PqTrainer {
        PqTrainer {
            training_set: Vec::with_capacity(NUM_TRAINING_SET_SIZE),
            num_subquantizers: meta_page.get_pq_vector_length(),
            considered_samples: 0,
        }
    }

    /// add_sample adds vectors to the training set via uniform reservoir sampling to keep the
    /// number of vectors within a reasonable memory limit.
    pub fn add_sample(&mut self, sample: &[f32]) {
        if self.training_set.len() >= NUM_TRAINING_SET_SIZE {
            // TODO: Cache this somehow.
            let mut rng = rand::thread_rng();
            let index = rng.gen_range(0..self.considered_samples + 1);
            if index < NUM_TRAINING_SET_SIZE {
                self.training_set[index] = sample.to_vec();
            }
        } else {
            self.training_set.push(sample.to_vec());
        }
        self.considered_samples += 1;
    }

    pub fn train_pq(self) -> Pq<f32> {
        notice!(
            "Training Product Quantization with {} vectors",
            self.training_set.len()
        );
        if (self.training_set.len() as i32) < (2_i32.pow(NUM_SUBQUANTIZER_BITS)) {
            error!("training set is too small, please run with use_pq as false.")
        }
        let training_set = self
            .training_set
            .iter()
            .map(|x| x.to_vec())
            .flatten()
            .collect();
        let shape = (self.training_set.len(), self.training_set[0].len());
        let instances = Array2::<f32>::from_shape_vec(shape, training_set).unwrap();
        Pq::train_pq(
            self.num_subquantizers,
            NUM_SUBQUANTIZER_BITS,
            PQ_TRAINING_ITERATIONS,
            NUM_TRAINING_ATTEMPTS,
            instances,
        )
        .unwrap()
    }
}

/// build_distance_table produces an Asymmetric Distance Table to quickly compute distances.
/// We compute the distance from every centroid and cache that so actual distance calculations
/// can be fast.
// TODO: This function could return a table that fits in SIMD registers.
fn build_distance_table(
    pq: &Pq<f32>,
    query: &[f32],
    _distance_fn: fn(&[f32], &[f32]) -> f32,
) -> Vec<f32> {
    let sq = pq.subquantizers();
    let num_centroids = pq.n_quantizer_centroids();
    let num_subquantizers = sq.len_of(Axis(0));
    let dt_size = num_subquantizers * num_centroids;
    let mut distance_table = vec![0.0; dt_size];

    let ds = query.len() / num_subquantizers;
    let mut elements_for_assert = 0;
    for (subquantizer_index, subquantizer) in sq.outer_iter().enumerate() {
        let sl = &query[subquantizer_index * ds..(subquantizer_index + 1) * ds];
        for (centroid_index, c) in subquantizer.outer_iter().enumerate() {
            /* always use l2 for pq measurements since centeroids use k-means (which uses euclidean/l2 distance)
             * The quantization also uses euclidean distance too. In the future we can experiment with k-mediods
             * using a different distance measure, but this may make little difference. */
            let dist = distance_l2_optimized_for_few_dimensions(sl, c.to_slice().unwrap());
            assert!(subquantizer_index < num_subquantizers);
            assert!(centroid_index * num_subquantizers + subquantizer_index < dt_size);
            distance_table[centroid_index * num_subquantizers + subquantizer_index] = dist;
            elements_for_assert += 1;
        }
    }
    assert_eq!(dt_size, elements_for_assert);
    distance_table
}

fn build_distance_table_pq_query(pq: &Pq<f32>, query: &[u8]) -> Vec<f32> {
    let sq = pq.subquantizers();
    let num_centroids = pq.n_quantizer_centroids();
    let num_subquantizers = sq.len_of(Axis(0));
    let dt_size = num_subquantizers * num_centroids;
    let mut distance_table = vec![0.0; dt_size];

    let mut elements_for_assert = 0;
    for (subquantizer_index, subquantizer) in sq.outer_iter().enumerate() {
        let query_centroid_index = query[subquantizer_index] as usize;
        let query_slice = subquantizer.index_axis(Axis(0), query_centroid_index);
        for (centroid_index, c) in subquantizer.outer_iter().enumerate() {
            /* always use l2 for pq measurements since centeroids use k-means (which uses euclidean/l2 distance)
             * The quantization also uses euclidean distance too. In the future we can experiment with k-mediods
             * using a different distance measure, but this may make little difference. */
            let dist = if query_centroid_index == centroid_index {
                debug_assert!(query_slice.as_slice().unwrap() == c.as_slice().unwrap());
                0.0
            } else {
                distance_l2_optimized_for_few_dimensions(
                    query_slice.as_slice().unwrap(),
                    c.as_slice().unwrap(),
                )
            };
            assert!(subquantizer_index < num_subquantizers);
            assert!(centroid_index * num_subquantizers + subquantizer_index < dt_size);
            distance_table[centroid_index * num_subquantizers + subquantizer_index] = dist;
            elements_for_assert += 1;
        }
    }
    assert_eq!(dt_size, elements_for_assert);
    distance_table
}

#[derive(Clone)]
pub struct PqQuantizer {
    pq_trainer: Option<PqTrainer>,
    pq: Option<Pq<f32>>,
}

impl PqQuantizer {
    pub fn new() -> PqQuantizer {
        Self {
            pq_trainer: None,
            pq: None,
        }
    }

    pub fn load<S: StatsNodeRead>(
        index_relation: &PgRelation,
        meta_page: &super::meta_page::MetaPage,
        stats: &mut S,
    ) -> Self {
        let pq_item_pointer = meta_page.get_pq_pointer().unwrap();
        let pq = unsafe { Some(read_pq(&index_relation, pq_item_pointer, stats)) };

        Self {
            pq_trainer: None,
            pq: pq,
        }
    }

    pub fn must_get_pq(&self) -> &Pq<f32> {
        self.pq.as_ref().unwrap()
    }

    pub fn quantize(&self, full_vector: &[f32]) -> Vec<PqVectorElement> {
        assert!(self.pq.is_some());
        let pq = self.pq.as_ref().unwrap();
        //OPT is the copy really necessary?
        let array_vec = Array1::from(full_vector.to_vec());
        pq.quantize_vector(array_vec).to_vec()
    }

    pub fn start_training(&mut self, meta_page: &super::meta_page::MetaPage) {
        self.pq_trainer = Some(PqTrainer::new(meta_page));
    }

    pub fn add_sample(&mut self, sample: &[f32]) {
        self.pq_trainer.as_mut().unwrap().add_sample(sample);
    }

    pub fn finish_training(&mut self) {
        self.pq = Some(self.pq_trainer.take().unwrap().train_pq());
    }

    pub fn vector_for_new_node(
        &self,
        meta_page: &MetaPage,
        full_vector: &[f32],
    ) -> Vec<PqVectorElement> {
        assert!(self.pq_trainer.is_none() && self.pq.is_some());
        let pq_vec_len = meta_page.get_pq_vector_length();
        let res = self.quantize(full_vector);
        assert!(res.len() == pq_vec_len);
        res
    }

    pub fn get_distance_table_full_query(
        &self,
        query: &[f32],
        distance_fn: fn(&[f32], &[f32]) -> f32,
    ) -> PqDistanceTable {
        PqDistanceTable::new_for_full_query(&self.pq.as_ref().unwrap(), distance_fn, query)
    }

    pub fn get_distance_table_pq_query(&self, pq_vector: &[PqVectorElement]) -> PqDistanceTable {
        PqDistanceTable::new_for_pq_query(&self.pq.as_ref().unwrap(), pq_vector)
    }

    pub fn get_distance_directly(
        &self,
        left: &[PqVectorElement],
        right: &[PqVectorElement],
    ) -> f32 {
        let pq = self.pq.as_ref().unwrap();
        let sq = pq.subquantizers();

        let mut dist = 0.0;
        for (subquantizer_index, subquantizer) in sq.outer_iter().enumerate() {
            let left_slice = subquantizer.index_axis(Axis(0), left[subquantizer_index] as usize);
            let right_slice = subquantizer.index_axis(Axis(0), right[subquantizer_index] as usize);
            assert!(left_slice.len() == right_slice.len());
            dist += distance_l2_optimized_for_few_dimensions(
                left_slice.as_slice().unwrap(),
                right_slice.as_slice().unwrap(),
            );
        }
        dist
    }
}

/// DistanceCalculator encapsulates the code to generate distances between a PQ vector and a query.
pub struct PqDistanceTable {
    distance_table: Vec<f32>,
}

impl PqDistanceTable {
    pub fn new_for_full_query(
        pq: &Pq<f32>,
        distance_fn: fn(&[f32], &[f32]) -> f32,
        query: &[f32],
    ) -> PqDistanceTable {
        PqDistanceTable {
            distance_table: build_distance_table(pq, query, distance_fn),
        }
    }

    pub fn new_for_pq_query(pq: &Pq<f32>, pq_vector: &[u8]) -> PqDistanceTable {
        PqDistanceTable {
            distance_table: build_distance_table_pq_query(pq, pq_vector),
        }
    }

    /// distance emits the sum of distances between each centroid in the quantized vector.
    pub fn distance(&self, pq_vector: &[u8]) -> f32 {
        let mut d = 0.0;
        let num_subquantizers = pq_vector.len();
        // maybe we should unroll this loop?
        for subquantizer_index in 0..num_subquantizers {
            let centroid_index = pq_vector[subquantizer_index] as usize;
            d += self.distance_table[centroid_index * num_subquantizers + subquantizer_index]
            //d += self.distance_table[m][pq_vector[m] as usize];
        }
        d
    }
}

pub enum PqSearchDistanceMeasure {
    Pq(PqDistanceTable),
}

impl PqSearchDistanceMeasure {
    pub fn calculate_pq_distance<S: StatsDistanceComparison>(
        table: &PqDistanceTable,
        pq_vector: &[PqVectorElement],
        stats: &mut S,
    ) -> f32 {
        assert!(pq_vector.len() > 0);
        let vec = pq_vector;
        stats.record_quantized_distance_comparison();
        table.distance(vec)
    }
}
