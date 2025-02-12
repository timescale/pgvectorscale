use super::{meta_page::MetaPage, pg_vector::PgVector};
use pgrx::{
    pg_sys::{Datum, ScanKeyData},
    Array, FromDatum,
};
use rkyv::{Archive, Deserialize, Serialize};

pub type Label = u16;

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

impl FromIterator<Label> for LabelSet {
    fn from_iter<T: IntoIterator<Item = Label>>(iter: T) -> Self {
        let mut labels: Vec<Label> = iter.into_iter().collect();
        labels.sort_unstable();
        labels.dedup();

        Self { labels }
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
    fn matches<T: LabelSetView>(&self, other: &T) -> bool {
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

/// A labeled vector is a vector with a (possibly empty) set of labels.  For
/// uniformity, we use labeled vectors even in indices that do not have
/// labels.
pub struct LabeledVector {
    vec: PgVector,
    labels: LabelSet,
}

impl LabeledVector {
    pub fn new(vec: PgVector, labels: LabelSet) -> Self {
        Self { vec, labels }
    }

    pub unsafe fn from_datums(
        values: *mut Datum,
        isnull: *mut bool,
        meta_page: &MetaPage,
    ) -> Option<Self> {
        let vec = PgVector::from_pg_parts(values, isnull, 0, meta_page, true, false)?;

        let labels: LabelSet = if meta_page.has_labels() {
            let arr = Array::<i32>::from_datum(*values.add(1), *isnull.add(1));
            if let Some(arr) = arr {
                arr.into_iter().flatten().map(|x| x as Label).collect()
            } else {
                LabelSet::default()
            }
        } else {
            LabelSet::default()
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

        let labels = if keys.is_empty() {
            vec![]
        } else {
            let arr = unsafe { Array::<i32>::from_datum(keys[0].sk_argument, false).unwrap() };
            arr.into_iter().flatten().map(|i| i as Label).collect()
        };

        Self::new(query, labels.into())
    }

    pub fn vec(&self) -> &PgVector {
        &self.vec
    }

    pub fn labels(&self) -> &LabelSet {
        &self.labels
    }
}
