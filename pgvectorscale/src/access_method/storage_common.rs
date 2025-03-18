use pgrx::{pg_sys::AttrNumber, PgRelation};

pub fn get_num_index_attributes(index: &PgRelation) -> usize {
    let natts = unsafe { (*index.rd_index).indnatts as usize };
    assert!(natts <= 2);
    natts
}

pub fn get_index_vector_attribute(index: &PgRelation) -> AttrNumber {
    unsafe {
        let a = index.rd_index;
        let natts = (*a).indnatts;
        assert!(natts <= 2);
        (*a).indkey.values.as_slice(natts as _)[0]
    }
}
