use crate::access_method::meta_page::MetaPage;

use super::{SbqVectorElement, BITS_STORE_TYPE_SIZE};

#[derive(Clone)]
pub struct SbqQuantizer {
    pub use_mean: bool,
    training: bool,
    pub count: u64,
    pub mean: Vec<f32>,
    pub m2: Vec<f32>,
    pub num_bits_per_dimension: u8,
}

impl SbqQuantizer {
    pub fn new(meta_page: &MetaPage) -> SbqQuantizer {
        Self {
            use_mean: true,
            training: false,
            count: 0,
            mean: vec![],
            m2: vec![],
            num_bits_per_dimension: meta_page.get_bq_num_bits_per_dimension(),
        }
    }

    pub fn load(&mut self, count: u64, mean: Vec<f32>, m2: Vec<f32>) {
        self.count = count;
        self.mean = mean;
        self.m2 = m2
    }

    pub fn quantized_size(&self, full_vector_size: usize) -> usize {
        Self::quantized_size_internal(full_vector_size, self.num_bits_per_dimension)
    }

    pub fn quantized_size_internal(full_vector_size: usize, num_bits_per_dimension: u8) -> usize {
        let num_bits = full_vector_size * num_bits_per_dimension as usize;

        if num_bits.is_multiple_of(BITS_STORE_TYPE_SIZE) {
            num_bits / BITS_STORE_TYPE_SIZE
        } else {
            (num_bits / BITS_STORE_TYPE_SIZE) + 1
        }
    }

    pub fn quantized_size_bytes(num_dimensions: usize, num_bits_per_dimension: u8) -> usize {
        Self::quantized_size_internal(num_dimensions, num_bits_per_dimension)
            * std::mem::size_of::<SbqVectorElement>()
    }

    pub fn quantize(&self, full_vector: &[f32]) -> Vec<SbqVectorElement> {
        assert!(!self.training);
        if self.use_mean {
            let mut res_vector = vec![0; self.quantized_size(full_vector.len())];

            if self.num_bits_per_dimension == 1 {
                for (i, &v) in full_vector.iter().enumerate() {
                    if v > self.mean[i] {
                        res_vector[i / BITS_STORE_TYPE_SIZE] |= 1 << (i % BITS_STORE_TYPE_SIZE);
                    }
                }
            } else {
                for (i, &v) in full_vector.iter().enumerate() {
                    let mean = self.mean[i];
                    let variance = self.m2[i] / self.count as f32;
                    let std_dev = variance.sqrt();
                    let ranges = self.num_bits_per_dimension + 1;

                    let v_z_score = (v - mean) / std_dev;
                    let index = (v_z_score + 2.0) / (4.0 / ranges as f32); //we consider z scores between -2 and 2 and divide them into {ranges} ranges

                    let bit_position = i * self.num_bits_per_dimension as usize;
                    if index < 1.0 {
                        //all zeros
                    } else {
                        let count_ones =
                            (index.floor() as usize).min(self.num_bits_per_dimension as usize);
                        //fill in count_ones bits from the left
                        // ex count_ones=1: 100
                        // ex count_ones=2: 110
                        // ex count_ones=3: 111
                        for j in 0..count_ones {
                            res_vector[(bit_position + j) / BITS_STORE_TYPE_SIZE] |=
                                1 << ((bit_position + j) % BITS_STORE_TYPE_SIZE);
                        }
                    }
                }
            }
            res_vector
        } else {
            let mut res_vector = vec![0; self.quantized_size(full_vector.len())];

            for (i, &v) in full_vector.iter().enumerate() {
                if v > 0.0 {
                    res_vector[i / BITS_STORE_TYPE_SIZE] |= 1 << (i % BITS_STORE_TYPE_SIZE);
                }
            }

            res_vector
        }
    }

    pub fn start_training(&mut self, meta_page: &MetaPage) {
        self.training = true;
        if self.use_mean {
            self.count = 0;
            self.mean = vec![0.0; meta_page.get_num_dimensions_to_index() as _];
            if self.num_bits_per_dimension > 1 {
                self.m2 = vec![0.0; meta_page.get_num_dimensions_to_index() as _];
            }
        }
    }

    pub fn add_sample(&mut self, sample: &[f32]) {
        if self.use_mean {
            self.count += 1;
            assert!(self.mean.len() == sample.len());

            if self.num_bits_per_dimension > 1 {
                assert!(self.m2.len() == sample.len());
                let delta: Vec<_> = self
                    .mean
                    .iter()
                    .zip(sample.iter())
                    .map(|(m, s)| s - *m)
                    .collect();

                self.mean
                    .iter_mut()
                    .zip(sample.iter())
                    .for_each(|(m, s)| *m += (s - *m) / self.count as f32);

                let delta2 = self.mean.iter().zip(sample.iter()).map(|(m, s)| s - *m);

                self.m2
                    .iter_mut()
                    .zip(delta.iter())
                    .zip(delta2)
                    .for_each(|((m2, d), d2)| *m2 += d * d2);
            } else {
                self.mean
                    .iter_mut()
                    .zip(sample.iter())
                    .for_each(|(m, s)| *m += (s - *m) / self.count as f32);
            }
        }
    }

    pub fn finish_training(&mut self) {
        self.training = false;
    }

    pub fn vector_for_new_node(
        &self,
        _meta_page: &MetaPage,
        full_vector: &[f32],
    ) -> Vec<SbqVectorElement> {
        self.quantize(full_vector)
    }
}
