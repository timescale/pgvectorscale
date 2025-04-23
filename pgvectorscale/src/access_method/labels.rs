use super::{meta_page::MetaPage, pg_vector::PgVector};
use pgrx::{
    pg_sys::{Datum, ScanKeyData},
    Array, FromDatum,
};
use rkyv::{Archive, Deserialize, Serialize};
use std::fmt::Debug;

pub type Label = i16;

/// LabelSet is a set of labels.  It is stored as a sorted array of labels.
/// LabelSets can be of varying lengths, but once constructed, they are immutable.
///
/// Note: indices that do not have labels are represented as an empty labelset.
#[derive(Archive, Deserialize, Serialize, Debug, Clone, Default)]
#[archive(check_bytes)]
#[repr(C)]
pub struct LabelSet {
    labels: Vec<Label>,
}

impl From<LabelSet> for Vec<Label> {
    fn from(set: LabelSet) -> Self {
        set.labels
    }
}

impl From<Vec<Label>> for LabelSet {
    fn from(mut labels: Vec<Label>) -> Self {
        labels.sort_unstable();
        labels.dedup();

        Self { labels }
    }
}

impl From<&Vec<Label>> for LabelSet {
    fn from(labels: &Vec<Label>) -> Self {
        let mut labels = labels.clone();
        labels.sort_unstable();
        labels.dedup();

        Self { labels }
    }
}

impl From<&ArchivedLabelSet> for LabelSet {
    fn from(set: &ArchivedLabelSet) -> Self {
        Self {
            labels: set.labels.to_vec(),
        }
    }
}

impl From<ArchivedLabelSet> for LabelSet {
    fn from(set: ArchivedLabelSet) -> Self {
        Self {
            labels: set.labels.to_vec(),
        }
    }
}

impl From<Label> for LabelSet {
    fn from(label: Label) -> Self {
        Self {
            labels: vec![label],
        }
    }
}

impl FromIterator<Label> for LabelSet {
    fn from_iter<T: IntoIterator<Item = Label>>(iter: T) -> Self {
        let mut labels: Vec<Label> = iter.into_iter().collect();
        labels.sort_unstable();
        labels.dedup();

        Self { labels }
    }
}

impl LabelSet {
    /// Given sorted arrays of labels, check: Is $a \cap b \subseteq self$?
    pub fn contains_intersection(&self, a: &LabelSet, b: &LabelSet) -> bool {
        let a = a.labels();
        let b = b.labels();
        let c = self.labels();

        let mut i = 0;
        let mut j = 0;
        let mut k = 0;

        while i < a.len() && j < b.len() {
            match a[i].cmp(&b[j]) {
                std::cmp::Ordering::Equal => {
                    while k < c.len() && c[k] < a[i] {
                        k += 1;
                    }
                    if k == c.len() || c[k] > a[i] {
                        return false;
                    }
                    i += 1;
                    j += 1;
                }
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
            }
        }
        true
    }
}

/// Accessor trait to allow common logic between `LabelSet` and `ArchivedLabelSet`.
pub trait LabelSetView {
    fn labels(&self) -> &[Label];

    /// Any labels in the labelset?
    fn is_empty(&self) -> bool {
        self.labels().is_empty()
    }

    /// Do the labelsets share any labels?
    fn overlaps<T: LabelSetView>(&self, other: &T) -> bool {
        let a = self.labels();
        let b = other.labels();

        debug_assert!(a.is_sorted());
        debug_assert!(b.is_sorted());

        let mut i = 0;
        let mut j = 0;

        while i < a.len() && j < b.len() {
            match a[i].cmp(&b[j]) {
                std::cmp::Ordering::Equal => return true,
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
            }
        }
        false
    }

    fn iter(&self) -> std::slice::Iter<Label> {
        self.labels().iter()
    }
}

impl LabelSetView for LabelSet {
    fn labels(&self) -> &[Label] {
        &self.labels
    }
}

impl LabelSetView for ArchivedLabelSet {
    fn labels(&self) -> &[Label] {
        &self.labels
    }
}

impl Debug for ArchivedLabelSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("ArchivedLabelSet")
            .field(&self.labels.to_vec())
            .finish()
    }
}

/// A labeled vector is a vector with an optional set of labels.
#[derive(Debug)]
pub struct LabeledVector {
    vec: PgVector,
    labels: Option<LabelSet>,
}

impl LabeledVector {
    pub fn new(vec: PgVector, labels: Option<LabelSet>) -> Self {
        Self { vec, labels }
    }

    pub unsafe fn from_datums(
        values: *mut Datum,
        isnull: *mut bool,
        meta_page: &MetaPage,
    ) -> Option<Self> {
        let vec = PgVector::from_pg_parts(values, isnull, 0, meta_page, true, false)?;

        let labels: Option<LabelSet> = if meta_page.has_labels() {
            if *isnull.add(1) {
                Some(LabelSet::default())
            } else {
                let arr = Array::<i16>::from_datum(*values.add(1), false);
                Some(arr.map_or_else(LabelSet::default, |arr| {
                    // Special case to work around apparent bug in pgrx
                    if arr.is_empty() || arr.iter().all(|x| x.is_none()) {
                        return LabelSet::default();
                    }
                    let labels_iter = arr.into_iter().flatten();
                    labels_iter.collect()
                }))
            }
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

        let labels: Option<LabelSet> = if keys.is_empty() {
            None
        } else {
            let arr = unsafe { Array::<i16>::from_datum(keys[0].sk_argument, false).unwrap() };
            // smallint already enforces the bounds, so we can just collect
            let labels: Vec<Label> = arr.into_iter().flatten().collect();

            Some(labels.into())
        };

        Self::new(query, labels)
    }

    pub fn vec(&self) -> &PgVector {
        &self.vec
    }

    pub fn labels(&self) -> Option<&LabelSet> {
        self.labels.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_overlaps_empty() {
        let a: LabelSet = vec![].into();
        let b: LabelSet = vec![1, 2, 3].into();
        assert!(!a.overlaps(&b));
        assert!(!b.overlaps(&a));
    }

    #[test]
    fn test_overlaps_non_empty() {
        let a: LabelSet = vec![1, 2].into();
        let b: LabelSet = vec![2, 3].into();
        assert!(a.overlaps(&b));
        assert!(b.overlaps(&a));
    }

    #[test]
    fn test_overlaps_no_overlap() {
        let a: LabelSet = vec![1, 2].into();
        let b: LabelSet = vec![3, 4].into();
        assert!(!a.overlaps(&b));
        assert!(!b.overlaps(&a));
    }

    #[test]
    fn test_overlaps_longer() {
        let a: LabelSet = vec![1, 2, 3, 4, 5].into();
        let b: LabelSet = vec![1, 2, 3, 4].into();
        assert!(a.overlaps(&b));
        assert!(b.overlaps(&a));
    }

    #[test]
    fn test_overlaps_non_empty_no_overlap() {
        let a: LabelSet = vec![1, 2, 3, 4, 5].into();
        let b: LabelSet = vec![6, 7, 8, 9, 10].into();
        assert!(!a.overlaps(&b));
        assert!(!b.overlaps(&a));
    }

    #[test]
    fn test_overlaps_non_empty_overlap() {
        let a: LabelSet = vec![1, 2, 3, 4, 5].into();
        let b: LabelSet = vec![2, 3, 4, 5, 6].into();
        assert!(a.overlaps(&b));
        assert!(b.overlaps(&a));
    }

    #[test]
    fn test_overlaps_interleavings() {
        let a: LabelSet = vec![1, 3, 5, 10, 11].into();
        let b: LabelSet = vec![2, 4, 6, 8, 11].into();
        assert!(a.overlaps(&b));
        assert!(b.overlaps(&a));
    }

    #[test]
    fn test_contains_intersection() {
        let a: LabelSet = vec![1, 3, 5, 10, 11].into();
        let b: LabelSet = vec![2, 4, 6, 8, 11].into();
        let c: LabelSet = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11].into();
        assert!(c.contains_intersection(&a, &b));
        assert!(c.contains_intersection(&b, &a));
    }

    #[test]
    fn test_contains_intersection_empty_sets() {
        let a: LabelSet = vec![].into();
        let b: LabelSet = vec![1, 2, 3].into();
        let c: LabelSet = vec![1, 2, 3].into();
        assert!(c.contains_intersection(&a, &b));
        assert!(c.contains_intersection(&b, &a));
    }

    #[test]
    fn test_contains_intersection_no_intersection() {
        let a: LabelSet = vec![1, 2, 3].into();
        let b: LabelSet = vec![4, 5, 6].into();
        let c: LabelSet = vec![1, 2, 3, 4, 5, 6].into();
        assert!(c.contains_intersection(&a, &b));
        assert!(c.contains_intersection(&b, &a));
    }

    #[test]
    fn test_contains_intersection_missing_element() {
        let a: LabelSet = vec![1, 2, 3].into();
        let b: LabelSet = vec![2, 3, 4].into();
        let c: LabelSet = vec![1, 3, 4].into(); // Missing 2
        assert!(!c.contains_intersection(&a, &b));
        assert!(!c.contains_intersection(&b, &a));
    }

    #[test]
    fn test_contains_intersection_empty_intersection() {
        let a: LabelSet = vec![1, 2, 3].into();
        let b: LabelSet = vec![4, 5, 6].into();
        let c: LabelSet = vec![1, 2, 3, 4, 5, 6].into();
        assert!(c.contains_intersection(&a, &b));
        assert!(c.contains_intersection(&b, &a));
    }

    #[test]
    fn test_contains_intersection_missing_intersection_element() {
        let a: LabelSet = vec![1, 2, 3, 4].into();
        let b: LabelSet = vec![2, 3, 4, 5].into();
        let c: LabelSet = vec![1, 2, 4, 5].into(); // Missing 3 which is in intersection
        assert!(!c.contains_intersection(&a, &b));
        assert!(!c.contains_intersection(&b, &a));
    }

    #[test]
    fn test_contains_intersection_empty_c() {
        let a: LabelSet = vec![1, 2, 3].into();
        let b: LabelSet = vec![2, 3, 4].into();
        let c: LabelSet = vec![].into();
        assert!(!c.contains_intersection(&a, &b));
        assert!(!c.contains_intersection(&b, &a));
    }

    #[test]
    fn test_contains_intersection_single_missing() {
        let a: LabelSet = vec![1, 2, 3].into();
        let b: LabelSet = vec![2, 3, 4].into();
        let c: LabelSet = vec![2, 4].into(); // Missing 3 which is in intersection
        assert!(!c.contains_intersection(&a, &b));
        assert!(!c.contains_intersection(&b, &a));
    }

    #[test]
    fn test_contains_intersection_single_element() {
        let a: LabelSet = vec![1].into();
        let b: LabelSet = vec![1].into();
        let c: LabelSet = vec![1].into();
        assert!(c.contains_intersection(&a, &b));
        assert!(c.contains_intersection(&b, &a));
    }

    #[test]
    fn test_contains_intersection_all_empty() {
        let a: LabelSet = vec![].into();
        let b: LabelSet = vec![].into();
        let c: LabelSet = vec![].into();
        assert!(c.contains_intersection(&a, &b));
        assert!(c.contains_intersection(&b, &a));
    }

    #[test]
    fn test_contains_intersection_large_sets() {
        let a: LabelSet = (1..=100).collect();
        let b: LabelSet = (50..=150).collect();
        let c: LabelSet = (1..=200).collect();
        assert!(c.contains_intersection(&a, &b));
        assert!(c.contains_intersection(&b, &a));
    }

    #[test]
    fn test_contains_intersection_duplicate_elements() {
        let a: LabelSet = vec![1, 1, 2, 2, 3, 3].into();
        let b: LabelSet = vec![2, 2, 3, 3, 4, 4].into();
        let c: LabelSet = vec![1, 2, 3, 4].into();
        assert!(c.contains_intersection(&a, &b));
        assert!(c.contains_intersection(&b, &a));
    }

    #[test]
    fn test_contains_intersection_negative_numbers() {
        let a: LabelSet = vec![-3, -2, -1, 0].into();
        let b: LabelSet = vec![-2, -1, 0, 1].into();
        let c: LabelSet = vec![-3, -2, -1, 0, 1].into();
        assert!(c.contains_intersection(&a, &b));
        assert!(c.contains_intersection(&b, &a));
    }
}
