/// A set of small integers, as bits. Iterates in increasing order, so whatever walks one
/// walks it the same way every time.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Bits(Vec<u64>);

impl Bits {
    /// Room for `0..n`, all absent.
    pub(crate) fn new(n: usize) -> Self {
        Bits(vec![0; n.div_ceil(64)])
    }

    pub(crate) fn insert(&mut self, i: usize) {
        self.0[i / 64] |= 1 << (i % 64);
    }

    pub(crate) fn remove(&mut self, i: usize) {
        self.0[i / 64] &= !(1 << (i % 64));
    }

    pub(crate) fn contains(&self, i: usize) -> bool {
        self.0[i / 64] >> (i % 64) & 1 == 1
    }

    /// Adds every member of `other`.
    pub(crate) fn union_with(&mut self, other: &Bits) {
        for (a, b) in self.0.iter_mut().zip(&other.0) {
            *a |= b;
        }
    }

    /// Removes every member of `other`.
    pub(crate) fn subtract(&mut self, other: &Bits) {
        for (a, b) in self.0.iter_mut().zip(&other.0) {
            *a &= !b;
        }
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.0.iter().enumerate().flat_map(|(w, &word)| {
            let mut rest = word;
            std::iter::from_fn(move || {
                (rest != 0).then(|| {
                    let bit = rest.trailing_zeros() as usize;
                    rest &= rest - 1;
                    w * 64 + bit
                })
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::Bits;

    #[test]
    fn keeps_members_in_order() {
        let mut b = Bits::new(200);
        for i in [130, 3, 64, 63, 199] {
            b.insert(i);
        }
        b.remove(64);
        assert_eq!(b.iter().collect::<Vec<_>>(), [3, 63, 130, 199]);
        assert!(b.contains(199) && !b.contains(64));
    }
}
