//! GPU kernels that name a deprecated const, in a crate that denies `deprecated`, built with `kernite-cuda` and without
//! it: Rust tells of each place at the level its kernel sets in both, and in the second beside kernite's error that
//! says the crate needs `kernite-cuda`. `warns` lowers the lint, `allows` quiets it, `expects` meets its `expect`, and
//! `crate_level` sets no level. `names_nothing` forbids it, and names no item: nothing is written for it. `gone` sets
//! the level of a lint Rust does not know, and `cfg` leaves it out where it keeps the others: Rust says nothing of it.
#![deny(deprecated)]

#[kernite_macros::kernels]
pub mod gpu {
    #[deprecated]
    const K: u32 = 5;

    #[forbid(deprecated)]
    pub unsafe extern "ptx-kernel" fn names_nothing(out: *mut u32) {
        st.global.u32([out], 1);
    }

    #[warn(deprecated)]
    pub unsafe extern "ptx-kernel" fn warns(out: *mut u32) {
        st.global.u32([out], K);
    }

    #[allow(deprecated)]
    pub unsafe extern "ptx-kernel" fn allows(out: *mut u32) {
        st.global.u32([out], K);
    }

    #[expect(deprecated)]
    pub unsafe extern "ptx-kernel" fn expects(out: *mut u32) {
        st.global.u32([out], K);
    }

    pub unsafe extern "ptx-kernel" fn crate_level(out: *mut u32) {
        st.global.u32([out], K);
    }

    #[cfg(test)]
    #[warn(no_such_lint)]
    pub unsafe extern "ptx-kernel" fn gone(out: *mut u32) {
        st.global.u32([out], K);
    }
}
