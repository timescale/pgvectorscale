use super::{meta_page::MetaPage, pg_vector::PgVector};
use pgrx::{
    pg_sys::{Datum, ScanKeyData},
    Array, FromDatum,
};

pub type Label = u8;
pub type LabelSet = [Label; MAX_LABELS_PER_NODE];
pub type LabelVec = Vec<Label>;

pub const INVALID_LABEL: u8 = 0;
pub const MAX_LABELS_PER_NODE: usize = 8;

/// Returns true if the two label sets overlap.  Assumes labels are sorted.
pub fn test_overlap(labels1: &[Label], labels2: &[Label]) -> bool {
    debug_assert!(labels1.is_sorted());
    debug_assert!(labels2.is_sorted());

    let mut i = 0;
    let mut j = 0;
    while i < labels1.len() && j < labels2.len() {
        if labels1[i] == labels2[j] {
            return true;
        } else if labels1[i] < labels2[j] {
            i += 1;
        } else {
            j += 1;
        }
    }
    false
}

/// Convert a label vector to a label set.
pub fn label_vec_to_set(labels: Option<&[Label]>) -> LabelSet {
    let mut set = [INVALID_LABEL; MAX_LABELS_PER_NODE];
    if let Some(labels) = labels {
        debug_assert!(labels.len() <= MAX_LABELS_PER_NODE);
        debug_assert!(labels.is_sorted());

        for (i, &label) in labels.iter().enumerate() {
            debug_assert!(label != INVALID_LABEL);
            set[i] = label;
        }
    }
    set
}

pub struct LabeledVector {
    vec: PgVector,
    labels: Option<LabelVec>,
}

impl LabeledVector {
    pub fn new(vec: PgVector, labels: Option<LabelVec>) -> Self {
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
            arr.map(|arr| arr.into_iter().flatten().map(|x| x as Label).collect())
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

        let labels: Option<Vec<Label>> = (!keys.is_empty()).then(|| {
            let arr = unsafe { Array::<i32>::from_datum(keys[0].sk_argument, false).unwrap() };
            arr.into_iter().flatten().map(|i| i as Label).collect()
        });

        Self::new(query, labels)
    }

    pub fn vec(&self) -> &PgVector {
        &self.vec
    }

    pub fn labels(&self) -> Option<&[Label]> {
        self.labels.as_deref()
    }
}

/// Test cases for test_overlap
#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_test_overlap() {
        assert_eq!(test_overlap(&[], &[]), false);
        assert_eq!(test_overlap(&[1], &[]), false);
        assert_eq!(test_overlap(&[], &[1]), false);
        assert_eq!(test_overlap(&[1], &[1]), true);
        assert_eq!(test_overlap(&[1], &[2]), false);
        assert_eq!(test_overlap(&[1, 2], &[2]), true);
        assert_eq!(test_overlap(&[1, 2], &[3]), false);
        assert_eq!(test_overlap(&[1, 2], &[2, 3]), true);
        assert_eq!(test_overlap(&[1, 2], &[3, 4]), false);
        assert_eq!(test_overlap(&[1, 2], &[2, 3]), true);
        assert_eq!(test_overlap(&[1, 2], &[2, 3, 4]), true);
        assert_eq!(test_overlap(&[1, 2], &[3, 4, 5]), false);
    }

    /// Test label_vec_to_set
    #[test]
    fn test_label_vec_to_set() {
        assert_eq!(label_vec_to_set(None), [INVALID_LABEL; MAX_LABELS_PER_NODE]);
        assert_eq!(
            label_vec_to_set(Some(&[])),
            [INVALID_LABEL; MAX_LABELS_PER_NODE]
        );
        assert_eq!(
            label_vec_to_set(Some(&[1])),
            [
                1,
                INVALID_LABEL,
                INVALID_LABEL,
                INVALID_LABEL,
                INVALID_LABEL,
                INVALID_LABEL,
                INVALID_LABEL,
                INVALID_LABEL
            ]
        );
        assert_eq!(
            label_vec_to_set(Some(&[1, 2])),
            [
                1,
                2,
                INVALID_LABEL,
                INVALID_LABEL,
                INVALID_LABEL,
                INVALID_LABEL,
                INVALID_LABEL,
                INVALID_LABEL
            ]
        );
        assert_eq!(
            label_vec_to_set(Some(&[1, 2, 3, 4, 5, 6, 7])),
            [1, 2, 3, 4, 5, 6, 7, INVALID_LABEL]
        );
        assert_eq!(
            label_vec_to_set(Some(&[1, 2, 3, 4, 5, 6, 7, 8])),
            [1, 2, 3, 4, 5, 6, 7, 8]
        );
    }
}
