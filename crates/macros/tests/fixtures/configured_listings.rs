//! `LISTINGS` holds the listing of each x86 kernel `cfg` keeps, and none of one it leaves out: of
//! each instance of a kernel under `cfg`, of a kernel under a `cfg_attr(..)` that gives it a `cfg`,
//! and of none under a `cfg` that never holds. Rust checks, as it builds, which listings the crate
//! has with the feature `more` and without it.

#![deny(warnings)]

#[kernite_macros::kernels]
pub mod listed {
    /// Left out whatever the features are.
    #[cfg(any())]
    pub unsafe extern "sysv64" fn gone(p: *mut u64) {
        mov(qword[p], 1);
    }

    /// Always there.
    pub unsafe extern "sysv64" fn always(p: *mut u64) {
        mov(qword[p], 2);
    }

    /// There with the feature, an instance each.
    #[cfg(feature = "more")]
    #[instances(3, 4)]
    pub unsafe extern "sysv64" fn more<const N: u64>(p: *mut u64) {
        mov(qword[p], N);
    }

    /// There without it, under a `cfg_attr(..)`.
    #[cfg_attr(feature = "more", cfg(any()))]
    pub unsafe extern "sysv64" fn less(p: *mut u64) {
        mov(qword[p], 5);
    }
}

/// Every kernel's `cfg` leaves out.
#[kernite_macros::kernels]
pub mod none {
    #[cfg(any())]
    pub unsafe extern "sysv64" fn gone(p: *mut u64) {
        mov(qword[p], 1);
    }
}

// Whether a listing has the name, compared byte by byte: `==` of `str` is not `const`, and Rust
// checks the listings as it builds.
const fn is(listing: (&str, &str), name: &[u8]) -> bool {
    let (found, want) = (listing.0.as_bytes(), name);
    if found.len() != want.len() {
        return false;
    }
    let mut i = 0;
    while i < found.len() {
        if found[i] != want[i] {
            return false;
        }
        i += 1;
    }
    true
}

const _: () = {
    let kept = listed::LISTINGS;
    if cfg!(feature = "more") {
        assert!(kept.len() == 3);
        assert!(is(kept[0], b"always") && is(kept[1], b"more::<3>") && is(kept[2], b"more::<4>"));
    } else {
        assert!(kept.len() == 2);
        assert!(is(kept[0], b"always") && is(kept[1], b"less"));
    }
    assert!(none::LISTINGS.is_empty());
};
