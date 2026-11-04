//! A workspace its manifest names with a number, `name = 5`, is refused where the module is, and
//! the error names the manifest: the crate is a workspace of its own, in a directory `project`,
//! named by `nonstring.toml`. A module refused at its attribute, beside it, is not told the name.

#[kernite_macros::kernels]
mod gpu {
    /// Writes 1 to `x`.
    pub unsafe extern "ptx-kernel" fn k(x: *mut u32) {
        st.global.u32([x], 1);
    }
}

/// The same module, given an argument that is not consts: told that, and nothing of the workspace.
#[kernite_macros::kernels(consts = 5)]
mod refused {
    /// Writes 1 to `x`.
    pub unsafe extern "ptx-kernel" fn k(x: *mut u32) {
        st.global.u32([x], 1);
    }
}
