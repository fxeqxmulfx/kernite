//! A module of GPU kernels that `#[kernels]` refuses at its attribute, in a crate without `kernite-cuda`: the
//! attribute is told once, and so is the dependency the module needs, where `cfg` keeps a GPU item of it, as for a
//! module kernite does not refuse.

/// A kernel under the feature `a`: the module needs the crate where `a` holds.
#[kernite_macros::kernels(consts = 5)]
pub mod gpu {
    #[cfg(feature = "a")]
    pub unsafe extern "ptx-kernel" fn one(p: *mut u32) {
        st.global.u32([p], 1);
    }
}
