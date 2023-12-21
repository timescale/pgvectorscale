use std::pin::Pin;

use pgrx::PgRelation;
use rkyv::{Archive, Deserialize, Serialize};

use crate::util::{page::PageType, tape::Tape, IndexPointer, ItemPointer, ReadableBuffer};

use super::meta_page::MetaPage;

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

pub struct BqQuantizer {
    use_mean: bool,
    count: u64,
    mean: Vec<f32>,
}

impl BqQuantizer {
    pub fn new() -> BqQuantizer {
        Self {
            use_mean: true,
            count: 0,
            mean: vec![],
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

    pub fn initialize_node(
        &self,
        node: &mut super::model::Node,
        meta_page: &MetaPage,
        full_vector: Vec<f32>,
    ) {
        if self.use_mean {
            node.pq_vector = vec![0; Self::quantized_size(meta_page.get_num_dimensions() as _)];
        } else {
            node.pq_vector = self.quantize(&full_vector);
        }
    }

    pub fn update_node_after_traing(
        &self,
        archived: &mut Pin<&mut super::model::ArchivedNode>,
        full_vector: Vec<f32>,
    ) {
        if self.use_mean {
            let bq_vector = self.quantize(&full_vector);

            assert!(bq_vector.len() == archived.pq_vector.len());
            for i in 0..=bq_vector.len() - 1 {
                let mut pgv = archived.as_mut().pq_vectors().index_pin(i);
                *pgv = bq_vector[i];
            }
        }
    }

    pub fn start_training(&mut self, meta_page: &super::meta_page::MetaPage) {
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

    pub fn finish_training(&mut self) {}

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

    pub fn quantize(&self, full_vector: &[f32]) -> Vec<u8> {
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
