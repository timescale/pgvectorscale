use std::num::NonZero;

use lru::LruCache;

use crate::{
    access_method::stats::{StatsNodeModify, StatsNodeRead, StatsNodeWrite},
    util::{IndexPointer, ItemPointer},
};

use super::{node::SbqNode, SbqSpeedupStorage, SbqVectorElement};

pub struct QuantizedVectorCache {
    cache: LruCache<ItemPointer, Vec<SbqVectorElement>>,
}

impl QuantizedVectorCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            cache: LruCache::new(NonZero::new(capacity).unwrap()),
        }
    }

    pub fn get<S: StatsNodeRead + StatsNodeWrite + StatsNodeModify>(
        &mut self,
        index_pointer: IndexPointer,
        storage: &SbqSpeedupStorage,
        stats: &mut S,
    ) -> &[SbqVectorElement] {
        // TODO this probes the cache twice in the case of a hit, figure out
        // how to do this in a single probe without running afould of the Rust
        // borrow checker
        if !self.cache.contains(&index_pointer) {
            // Not in cache, need to read from storage
            let node = unsafe { SbqNode::read(storage.index, index_pointer, true, stats) };
            let vector = node.get_archived_node().get_bq_vector().to_vec();

            // Insert into cache and handle evicted item
            let evicted = self.cache.push(index_pointer, vector);
            if let Some((evicted_pointer, evicted_vector)) = evicted {
                storage.write_quantized_vector(evicted_pointer, evicted_vector.as_slice(), stats);
            }
        }

        self.cache.get(&index_pointer).unwrap()
    }

    pub fn preload<I: Iterator<Item = IndexPointer>, S: StatsNodeRead>(
        &mut self,
        index_pointers: I,
        storage: &SbqSpeedupStorage,
        stats: &mut S,
    ) {
        for index_pointer in index_pointers {
            let item_pointer = ItemPointer::new(index_pointer.block_number, index_pointer.offset);
            // Only load if not already in cache
            if !self.cache.contains(&item_pointer) {
                let node = unsafe { SbqNode::read(storage.index, item_pointer, true, stats) };
                let vector = node.get_archived_node().get_bq_vector().to_vec();
                self.cache.push(item_pointer, vector);
            }
        }
    }
}
