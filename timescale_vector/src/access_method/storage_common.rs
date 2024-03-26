use pgrx::{pg_sys, PgRelation};

pub fn get_attribute_number_from_index(index: &PgRelation) -> pg_sys::AttrNumber {
    unsafe {
        let a = index.rd_index;
        let natts = (*a).indnatts;
        assert!(natts == 1);
        (*a).indkey.values.as_slice(natts as _)[0]
    }
}
