use std::io::Write;
use std::pin::Pin;

use pgrx::pg_sys::{InvalidBlockNumber, InvalidOffsetNumber};
use pgrx::*;
use rayon::vec;

use crate::access_method::model::Node;
use crate::util::multiblock_tape::MultiblockTape;
use crate::util::tape::Tape;
use crate::util::{page, IndexPointer, ItemPointer, WritableBuffer};
use rkyv::vec::ArchivedVec;
use rkyv::{Archive, Archived, Deserialize, Serialize};

#[derive(Archive, Deserialize, Serialize)]
#[archive(check_bytes)]
pub struct StartingIds {
    count: usize,
    mean: Vec<f32>,
    m2: Vec<f32>,
    init_index_pointers: Vec<ItemPointer>,
    init_index_scores: Vec<f32>,
    init_index_values: Vec<f32>,
}

/*
//WritableStartingIds ties an archive node to it's underlying buffer that can be modified
pub struct WritableStartingIds<'a> {
    wb: WritableBuffer<'a>,
}

impl ArchivedStartingIds {
    pub fn with_data(data: &mut [u8]) -> Pin<&mut ArchivedStartingIds> {
        let pinned_bytes = Pin::new(data);
        unsafe { rkyv::archived_root_mut::<StartingIds>(pinned_bytes) }
    }
}

impl<'a> WritableStartingIds<'a> {
    pub fn get_archived_node(&self) -> Pin<&mut ArchivedStartingIds> {
        ArchivedStartingIds::with_data(self.wb.get_data_slice())
    }

    pub fn commit(self) {
        self.wb.commit()
    }
}
*/

fn distance(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());

    let norm: f32 = a
        .iter()
        .zip(b.iter())
        .map(|t| (*t.0 as f32 - *t.1 as f32) * (*t.0 as f32 - *t.1 as f32))
        .sum();
    assert!(norm >= 0.);
    norm.sqrt()
}

impl StartingIds {
    pub fn new(index: &PgRelation, num_dims: u32) -> Self {
        let item = Self {
            count: 0,
            mean: vec![0.0; num_dims as usize],
            m2: vec![0.0; num_dims as usize],
            init_index_pointers: vec![
                ItemPointer::new(InvalidBlockNumber, InvalidOffsetNumber);
                num_dims as usize
            ],
            init_index_scores: vec![0.0; num_dims as usize],
            init_index_values: vec![0.0; num_dims as usize],
        };

        //reserve the position
        let mut tape = unsafe { MultiblockTape::new(&index, page::PageType::StartingIds) };
        let bytes = rkyv::to_bytes::<_, 8096>(&item).unwrap();
        let ip = unsafe { tape.add(&bytes) };
        assert!(ip.block_number == 1);
        assert!(ip.offset == 1);
        item
    }

    pub fn read(index: &PgRelation) -> Self {
        let ip = ItemPointer::new(1, 1);
        let mut tape = unsafe { MultiblockTape::new(&index, page::PageType::StartingIds) };
        let bytes = unsafe { tape.read(ip) };
        let archived = unsafe { rkyv::archived_root::<StartingIds>(&bytes) };
        archived.deserialize(&mut rkyv::Infallible).unwrap()
    }

    pub fn write(&self, index: &PgRelation) {
        let ip = ItemPointer::new(1, 1);
        let mut tape = unsafe { MultiblockTape::new(&index, page::PageType::StartingIds) };
        let bytes = rkyv::to_bytes::<_, 8096>(self).unwrap();
        unsafe { tape.overwrite(ip, &bytes) };
    }

    fn add_to_stats(&mut self, vector: &[f32]) {
        //https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
        self.count += 1;
        if self.mean.is_empty() {
            self.mean = vec![0.0; vector.len()];
            self.m2 = vec![0.0; vector.len()];
        }
        let delta: Vec<f32> = vector
            .iter()
            .zip(self.mean.iter())
            .map(|(v, m)| v - m)
            .collect();

        self.mean
            .iter_mut()
            .zip(delta.iter())
            .for_each(|(m, d)| *m += d / self.count as f32);

        let delta2 = vector.iter().zip(self.mean.iter()).map(|(v, m)| v - m);
        let delta2d: Vec<_> = vector
            .iter()
            .zip(self.mean.iter())
            .map(|(v, m)| v - m)
            .collect();

        self.m2
            .iter_mut()
            .zip(delta)
            .zip(delta2)
            .for_each(|((m2, d), d2)| *m2 += d * d2);
    }

    pub fn get_count(&self) -> usize {
        self.count
    }
    pub fn get_starting_ids(&mut self, index: &PgRelation, vector: &[f32]) -> Vec<ItemPointer> {
        let count = 10;
        let indexes = self.get_indexes(vector);
        let indexes = indexes
            .iter()
            //first filter out the empty starting_ids
            .filter(|(idx, _)| self.init_index_pointers[*idx].block_number != InvalidBlockNumber)
            //then take count -- this guarantees 10 valid ids
            .take(count);

        let d: Vec<_> = indexes
            .clone()
            .map(|(idx, score)| {
                let rn = unsafe { Node::read(&index, self.init_index_pointers[*idx]) };
                let node = rn.get_archived_node();
                (
                    self.init_index_scores[*idx],
                    *score,
                    self.init_index_values[*idx],
                    vector[*idx],
                    distance(vector, node.vector.as_slice()),
                    //self.mean[*idx],
                    //self.m2[*idx] / self.count as f32,
                )
            })
            .collect();

        debug1!("starting_ids debug: {:?}", d);

        indexes
            .map(|(idx, _)| self.init_index_pointers[*idx])
            .collect()
    }

    pub fn get_or_init_starting_ids(
        &mut self,
        index_pointer: IndexPointer,
        vector: &[f32],
    ) -> Vec<ItemPointer> {
        let count = 10;
        self.add_to_stats(vector);
        let indexes = self.get_indexes(vector);
        let mut result = vec![];
        for (idx, score) in indexes.iter().take(count) {
            if self.init_index_pointers[*idx].block_number != InvalidBlockNumber {
                result.push(self.init_index_pointers[*idx]);
                if *score > (self.init_index_scores[*idx] * 1.2) {
                    self.init_index_pointers[*idx] = index_pointer;
                    self.init_index_scores[*idx] = *score;
                    self.init_index_values[*idx] = vector[*idx];
                }
            } else {
                self.init_index_pointers[*idx] = index_pointer;
                self.init_index_scores[*idx] = *score;
                self.init_index_values[*idx] = vector[*idx];
            }
        }
        result
    }

    fn get_indexes(&mut self, vector: &[f32]) -> Vec<(usize, f32)> {
        //like a z-score, but without the sqrt
        let variance_scores: Vec<f32> = vector
            .iter()
            .zip(self.mean.iter())
            .zip(self.m2.iter())
            .map(|((v, m), m2)| {
                if *m2 != 0.0 {
                    (*v - *m) / (*m2 / self.count as f32)
                } else {
                    *v - *m
                }
            })
            .collect();

        let mut indexed_scores: Vec<(usize, f32)> = variance_scores
            .iter()
            .enumerate()
            .map(|(index, &score)| (index, score))
            .collect();
        indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed_scores
        //  .iter()
        //.map(|(index, _score)| *index)
        //      .collect()
    }
}
