use super::{meta_page::MetaPage, pg_vector::PgVector};
use pgrx::{
    pg_sys::{Datum, ScanKeyData},
    Array, FromDatum,
};

pub type Labels = Vec<u16>;

pub struct LabeledVector {
    vec: PgVector,
    labels: Option<Labels>,
}

impl LabeledVector {
    pub fn new(vec: PgVector, labels: Option<Labels>) -> Self {
        Self { vec, labels }
    }

    pub unsafe fn from_datums(
        values: *mut Datum,
        isnull: *mut bool,
        meta_page: &MetaPage,
    ) -> Option<Self> {
        let vec = PgVector::from_pg_parts(values, isnull, 0, meta_page, true, false)?;

        let labels = if meta_page.has_labels() {
            let arr = Array::<i32>::from_datum(*values.add(1), *isnull.add(1));
            arr.map(|arr| arr.into_iter().flatten().map(|x| x as u16).collect())
        } else {
            None
        };

        Some(Self::new(vec, labels))
    }

    pub unsafe fn from_scan_key_data(
        keys: &[ScanKeyData],
        orderbys: &[ScanKeyData],
        meta_page: &MetaPage,
    ) -> Self {
        let query = unsafe {
            PgVector::from_datum(
                orderbys[0].sk_argument,
                meta_page,
                true, /* needed for search */
                true, /* needed for resort */
            )
        };

        let labels: Option<Vec<u16>> = (!keys.is_empty()).then(|| {
            let arr = unsafe { Array::<i32>::from_datum(keys[0].sk_argument, false).unwrap() };
            arr.into_iter().flatten().map(|i| i as u16).collect()
        });

        Self::new(query, labels)
    }

    pub fn vec(&self) -> &PgVector {
        &self.vec
    }

    pub fn labels(&self) -> Option<&Vec<u16>> {
        self.labels.as_ref()
    }
}
