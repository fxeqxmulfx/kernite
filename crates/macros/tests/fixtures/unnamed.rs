//! A workspace its manifest names `""` is left out of its kernels' names. This crate is a
//! workspace of its own, in a directory `project`, and `unnamed.toml` names the workspace `''`, so
//! its GPU kernel is `unnamed::gpu::k`, as Rust checks as it builds.

#[kgen_macros::kernels]
mod gpu {
    /// Writes 1 to `x`.
    pub unsafe extern "ptx-kernel" fn k(x: *mut u32) {
        st.global.u32([x], 1);
    }
}

const _: () = {
    let mut ptx = gpu::PTX.as_bytes();
    while !matches!(ptx.split_first_chunk(), Some((b".entry _ZN7unnamed3gpu1kE(", _))) {
        let [_, rest @ ..] = ptx else { panic!("the kernel is not named unnamed::gpu::k") };
        ptx = rest;
    }
};
