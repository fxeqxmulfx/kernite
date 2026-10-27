//! A workspace is named as the directory of its `Cargo.toml` is. This crate is a workspace of its
//! own, in a directory `project`, so its GPU kernel is `project::named::gpu::k`, as Rust checks
//! as it builds.

#[kgen_macros::kernels]
mod gpu {
    /// Writes 1 to `x`.
    pub unsafe extern "ptx-kernel" fn k(x: *mut u32) {
        st.global.u32([x], 1);
    }
}

const _: () = {
    let mut ptx = gpu::PTX.as_bytes();
    while !matches!(ptx.split_first_chunk(), Some((b".entry _ZN7project5named3gpu1kE(", _))) {
        let [_, rest @ ..] = ptx else { panic!("the kernel is not named project::named::gpu::k") };
        ptx = rest;
    }
};
