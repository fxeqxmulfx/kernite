//! A crate is built again when its workspace is renamed. This crate is a member of a workspace one
//! directory up, as a user's is. Its test writes the workspace's manifest, naming it `before`: its
//! GPU kernel is `before::rebuilt::gpu::k`, as Rust checks as it builds. The test then names the
//! workspace `after`, in the workspace's manifest and not the crate's, and builds the crate again:
//! Cargo builds it again, and the check fails.

#[kgen_macros::kernels]
mod gpu {
    /// Writes 1 to `x`.
    pub unsafe extern "ptx-kernel" fn k(x: *mut u32) {
        st.global.u32([x], 1);
    }
}

const _: () = {
    let mut ptx = gpu::PTX.as_bytes();
    while !matches!(ptx.split_first_chunk(), Some((b".entry _ZN6before7rebuilt3gpu1kE(", _))) {
        let [_, rest @ ..] = ptx else { panic!("the kernel is not named before::rebuilt::gpu::k") };
        ptx = rest;
    }
};
