//! A workspace its manifest names with a number, `name = 5`, is refused where the module is, and
//! the error names the manifest. This crate is a workspace of its own, in a directory `project`,
//! and `nonstring.toml` gives the workspace that name.

#[kernite_macros::kernels]
mod gpu {
    /// Writes 1 to `x`.
    pub unsafe extern "ptx-kernel" fn k(x: *mut u32) {
        st.global.u32([x], 1);
    }
}
