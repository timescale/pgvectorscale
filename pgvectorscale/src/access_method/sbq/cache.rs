use std::collections::HashMap;

use crate::{
    access_method::stats::StatsNodeRead,
    util::{IndexPointer, ItemPointer},
};

use super::{SbqSpeedupStorage, SbqVectorElement};

pub struct QuantizedVectorCache {
    quantized_vector_map: HashMap<ItemPointer, Vec<SbqVectorElement>>,
}

/* should be a LRU cache for quantized vector. For now cheat and never evict
    TODO: implement LRU cache
*/
impl QuantizedVectorCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            quantized_vector_map: HashMap::with_capacity(capacity),
        }
    }

    pub fn get<S: StatsNodeRead>(
        &mut self,
        index_pointer: IndexPointer,
        storage: &SbqSpeedupStorage,
        stats: &mut S,
    ) -> &[SbqVectorElement] {
        let vec = self
            .quantized_vector_map
            .entry(index_pointer)
            .or_insert_with(|| {
                storage.get_quantized_vector_from_index_pointer(index_pointer, stats)
            });
        vec.as_slice()
    }

    /* Ensure that all these elements are in the cache. If the capacity isn't big enough throw an error.
    must_get must succeed on all the elements after this call prior to another get or preload call */

    pub fn preload<I: Iterator<Item = IndexPointer>, S: StatsNodeRead>(
        &mut self,
        index_pointers: I,
        storage: &SbqSpeedupStorage,
        stats: &mut S,
    ) {
        for index_pointer in index_pointers {
            self.get(index_pointer, storage, stats);
        }
    }
}
