use std::{
    cmp::{Ordering, Reverse},
    collections::BinaryHeap,
};

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::Rng;

pub struct ListSearchNeighbor {
    pub index_pointer: u64,
    distance: f32,
    visited: bool,
    _private_data: u64,
}

impl PartialOrd for ListSearchNeighbor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for ListSearchNeighbor {
    fn eq(&self, other: &Self) -> bool {
        self.index_pointer == other.index_pointer
    }
}

impl Eq for ListSearchNeighbor {}

impl Ord for ListSearchNeighbor {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance.partial_cmp(&other.distance).unwrap()
    }
}

pub struct ListSearchResult {
    candidate_storage: Vec<ListSearchNeighbor>, //plain storage
    best_candidate: Vec<usize>,                 //pos in candidate storage, sorted by distance
}

impl ListSearchResult {
    pub fn get_lsn_by_idx(&self, idx: usize) -> &ListSearchNeighbor {
        &self.candidate_storage[idx]
    }

    pub fn insert_neighbor(&mut self, n: ListSearchNeighbor) {
        //insert while preserving sort order.
        let idx = self
            .best_candidate
            .partition_point(|x| self.candidate_storage[*x] < n);
        self.candidate_storage.push(n);
        let pos = self.candidate_storage.len() - 1;
        self.best_candidate.insert(idx, pos)
    }

    fn visit_closest(&mut self, pos_limit: usize) -> Option<usize> {
        //OPT: should we optimize this not to do a linear search each time?
        let neighbor_position = self
            .best_candidate
            .iter()
            .position(|n| !self.candidate_storage[*n].visited);
        match neighbor_position {
            Some(pos) => {
                if pos > pos_limit {
                    return None;
                }
                let n = &mut self.candidate_storage[self.best_candidate[pos]];
                n.visited = true;
                Some(self.best_candidate[pos])
            }
            None => None,
        }
    }
}

pub struct ListSearchResultMinHeap {
    candidates: BinaryHeap<Reverse<ListSearchNeighbor>>,
    visited: Vec<ListSearchNeighbor>,
}

impl ListSearchResultMinHeap {
    pub fn insert_neighbor(&mut self, n: ListSearchNeighbor) {
        //insert while preserving sort order.
        //        self.candidate_storage.push(n);
        //      let pos = self.candidate_storage.len() - 1;
        self.candidates.push(Reverse(n));

        /*let idx = self
            .best_candidate
            .partition_point(|x| self.candidate_storage[*x].distance < n.distance);
        self.candidate_storage.push(n);
        let pos = self.candidate_storage.len() - 1;
        self.best_candidate.insert(idx, pos)*/
    }

    fn visit_closest(&mut self, pos_limit: usize) -> Option<&ListSearchNeighbor> {
        //OPT: should we optimize this not to do a linear search each time?
        if self.candidates.is_empty() {
            panic!("no candidates left");
            //return None;
        }

        if self.visited.len() > pos_limit {
            let node_at_pos = &self.visited[pos_limit - 1];
            let head = self.candidates.peek().unwrap();
            if head.0.distance >= node_at_pos.distance {
                return None;
            }
        }

        let head = self.candidates.pop().unwrap();
        let idx = self
            .visited
            .partition_point(|x| x.distance < head.0.distance);
        self.visited.insert(idx, head.0);
        Some(&self.visited[idx])
    }
}

fn run_lsr_min_heap(lsr: &mut ListSearchResultMinHeap) {
    let item = lsr.visit_closest(100000000);
    let lsn = item.unwrap();

    let mut rng = rand::thread_rng();
    let delta: f64 = rng.gen(); // generates a float between 0 and 1
    let distance = lsn.distance + ((delta * 5.0) as f32);

    for _ in 0..20 {
        lsr.insert_neighbor(ListSearchNeighbor {
            index_pointer: 0,
            distance,
            visited: false,
            _private_data: 2,
        })
    }
}

fn run_lsr(lsr: &mut ListSearchResult) {
    let item_idx = lsr.visit_closest(1000000);
    let lsn = lsr.get_lsn_by_idx(item_idx.unwrap());

    let mut rng = rand::thread_rng();
    let delta: f64 = rng.gen(); // generates a float between 0 and 1
    let distance = lsn.distance + ((delta * 5.0) as f32);

    for _ in 0..20 {
        lsr.insert_neighbor(ListSearchNeighbor {
            index_pointer: 0,
            distance,
            visited: false,
            _private_data: 2,
        })
    }
}

pub fn benchmark_lsr(c: &mut Criterion) {
    let mut lsr = ListSearchResult {
        candidate_storage: Vec::new(),
        best_candidate: Vec::new(),
    };

    lsr.insert_neighbor(ListSearchNeighbor {
        index_pointer: 0,
        distance: 100.0,
        visited: false,
        _private_data: 1,
    });

    c.bench_function("lsr OG", |b| b.iter(|| run_lsr(black_box(&mut lsr))));
}

pub fn benchmark_lsr_min_heap(c: &mut Criterion) {
    let mut lsr = ListSearchResultMinHeap {
        candidates: BinaryHeap::new(),
        visited: Vec::new(),
    };

    lsr.insert_neighbor(ListSearchNeighbor {
        index_pointer: 0,
        distance: 100.0,
        visited: false,
        _private_data: 1,
    });

    c.bench_function("lsr min heap", |b| {
        b.iter(|| run_lsr_min_heap(black_box(&mut lsr)))
    });
}

criterion_group!(benches_lsr, benchmark_lsr, benchmark_lsr_min_heap);

criterion_main!(benches_lsr);
/*
fn fibonacci(n: u64) -> u64 {
    match n {
        0 => 1,
        1 => 1,
        n => fibonacci(n - 1) + fibonacci(n - 2),
    }
}
pub fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("fib 20", |b| b.iter(|| fibonacci(black_box(20))));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);*/
