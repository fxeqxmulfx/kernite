//! A helper whose signature names `Ret`, a type of the crate's that its module imports, in a module that is refused
//! for a workspace named by what is not a string, `name = 5` in `helper_imports_workspace.toml`: the error is the
//! module's, and none of its items'. No kernel calls the helpers, so kernite says nothing of their types; it shows
//! them to Rust, so Rust does not call the import unused beside the error, and says nothing of the language's
//! types, `gpr` or `local<[u32; 4], 16>`. This crate is a workspace of its own, in a directory `project`.

pub struct Ret(pub u64);

#[kernite_macros::kernels]
mod gpu {
    use super::Ret;

    fn h(x: Ret) {
        nop();
    }

    fn language(a: gpr, b: ymm, c: local<[u32; 4], 16>) -> r32 {
        nop();
    }

    /// Writes 1 to `x`.
    pub unsafe extern "ptx-kernel" fn k(x: *mut u32) {
        st.global.u32([x], 1);
    }
}
