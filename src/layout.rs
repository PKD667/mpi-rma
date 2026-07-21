use std::marker::PhantomData;

use crate::{Error, RmaElement};

/// A validated scatter target in a homogeneous `Window<T>`.
///
/// Values are associated in order: origin value `i` is put into
/// `indices[i]`. Indices must be distinct and in bounds. This is the useful
/// extent of MPI's indexed datatypes: fixed, precomputed placement, not
/// conditional insertion or sorting.
pub struct Indexed<T: RmaElement> {
    pub(crate) displacements: Vec<mpi::Address>,
    pub(crate) len: usize,
    pub(crate) bound: usize,
    _element: PhantomData<T>,
}

impl<T: RmaElement> Indexed<T> {
    pub fn new(indices: &[usize], window_len: usize) -> Result<Self, Error> {
        if indices.is_empty() {
            return Err(Error::Layout("layout is empty"));
        }
        let mut sorted = indices.to_vec();
        sorted.sort_unstable();
        if sorted.windows(2).any(|p| p[0] == p[1]) {
            return Err(Error::Layout("target entries overlap"));
        }
        let bound = sorted.last().copied().unwrap() + 1;
        if bound > window_len {
            return Err(Error::Range {
                start: bound - 1,
                len: 1,
                window: window_len,
            });
        }
        let width = std::mem::size_of::<T>();
        let displacements = indices
            .iter()
            .map(|&i| {
                i.checked_mul(width)
                    .and_then(|d| mpi::Address::try_from(d).ok())
                    .ok_or(Error::SizeOverflow)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Indexed {
            displacements,
            len: indices.len(),
            bound,
            _element: PhantomData,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::Indexed;
    use crate::Error;

    #[test]
    fn validates_indices() {
        let layout = Indexed::<u64>::new(&[4, 1, 7], 8).unwrap();
        assert_eq!(layout.len(), 3);
        assert!(matches!(
            Indexed::<u64>::new(&[1, 1], 8),
            Err(Error::Layout("target entries overlap"))
        ));
        assert!(matches!(
            Indexed::<u64>::new(&[8], 8),
            Err(Error::Range { .. })
        ));
        assert!(matches!(
            Indexed::<u64>::new(&[], 8),
            Err(Error::Layout("layout is empty"))
        ));
    }
}
