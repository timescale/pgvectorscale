use ndarray::{Array1, Array2};
use pgrx::{error, notice, PgRelation};
use rand::Rng;
use reductive::pq::{Pq, QuantizeVector, TrainPq};

use crate::access_method::model::read_pq;

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

/// PqTrainer is a utility that produces a product quantizer from training with sample vectors.
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
    pub fn new(meta_page: &super::build::TsvMetaPage) -> PqTrainer {
        PqTrainer {
            training_set: Vec::with_capacity(NUM_TRAINING_SET_SIZE),
            num_subquantizers: meta_page.get_pq_vector_length(),
            considered_samples: 0,
        }
    }

    /// add_sample adds vectors to the training set via uniform reservoir sampling to keep the
    /// number of vectors within a reasonable memory limit.
    pub fn add_sample(&mut self, sample: Vec<f32>) {
        if self.training_set.len() >= NUM_TRAINING_SET_SIZE {
            // TODO: Cache this somehow.
            let mut rng = rand::thread_rng();
            let index = rng.gen_range(0..self.considered_samples + 1);
            if index < NUM_TRAINING_SET_SIZE {
                self.training_set[index] = sample;
            }
        } else {
            self.training_set.push(sample);
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

/// PgPq encapsulates functions to work with PQ.
pub struct PgPq {
    pq: Pq<f32>,
}

impl PgPq {
    pub fn new(meta_page: &super::build::TsvMetaPage, index_relation: &PgRelation) -> Option<PgPq> {
        if !meta_page.get_use_pq() {
            return None;
        }
        let pq_id = meta_page.get_pq_pointer();
        match pq_id {
            None => None,
            Some(pq_id) => {
                let pq = unsafe { read_pq(&index_relation, &pq_id) };
                Some(PgPq { pq })
            }
        }
    }
    /// quantize produces a quantized vector from the raw pg vector.
    pub fn quantize(self, vector: Vec<f32>) -> Vec<u8> {
        let og_vec = Array1::from(vector.to_vec());
        self.pq.quantize_vector(og_vec).to_vec()
    }
    pub fn distance_calculator(
        self,
        query: Vec<f32>,
        distance_fn: fn(&[f32], &[f32]) -> f32,
    ) -> DistanceCalculator {
        DistanceCalculator::new(self.pq, distance_fn, query)
    }
}

/// build_distance_table produces an Asymmetric Distance Table to quickly compute distances.
/// We compute the distance from every centroid and cache that so actual distance calculations
/// can be fast.
// TODO: This function could return a table that fits in SIMD registers.
fn build_distance_table(
    pq: Pq<f32>,
    query: &[f32],
    distance_fn: fn(&[f32], &[f32]) -> f32,
) -> Vec<Vec<f32>> {
    let sq = pq.subquantizers();
    let mut distance_table: Vec<Vec<f32>> = Vec::new();
    let clusters: Vec<_> = sq.outer_iter().collect();
    let ds = query.len() / clusters.len();
    for m in 0..clusters.len() {
        let mut res: Vec<f32> = Vec::new();
        let sl = &query[m * ds..(m + 1) * ds];
        for i in 0..clusters[m].nrows() {
            let c = clusters[m].row(i).to_vec();
            let p = distance_fn(sl, c.as_slice());
            res.push(p);
        }
        distance_table.push(res);
    }
    distance_table
}

/// DistanceCalculator encapsulates the code to generate distances between a PQ vector and a query.
pub struct DistanceCalculator {
    distance_table: Vec<Vec<f32>>,
}

impl DistanceCalculator {
    pub fn new(
        pq: Pq<f32>,
        distance_fn: fn(&[f32], &[f32]) -> f32,
        query: Vec<f32>,
    ) -> DistanceCalculator {
        DistanceCalculator {
            distance_table: build_distance_table(pq, query.as_slice(), distance_fn),
        }
    }

    /// distance emits the sum of distances between each centroid in the quantized vector.
    pub fn distance(&self, pq_vector: Vec<u8>) -> f32 {
        let mut d = 0.0;
        // maybe we should unroll this loop?
        for m in 0..pq_vector.len() {
            d += self.distance_table[m][pq_vector[m] as usize];
        }
        d
    }
}
