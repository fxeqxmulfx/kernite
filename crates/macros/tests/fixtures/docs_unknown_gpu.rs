//! A module of GPU kernels, one of which names an instruction the GPU does not have, next to ones it has: the error is
//! that name's alone, as Rust has no doc of it to find.

#[kernite_macros::kernels]
pub mod gpu {
    /// Adds one to the float at `a`.
    pub unsafe extern "ptx-kernel" fn add_one(a: *mut f32) {
        let v: f32;
        ld.globl.f32(v, [a]);
        add.rn.f32(v, v, 1.0);
        st.global.f32([a], v);
    }

    /// Subtracts one from the float at `a`.
    pub unsafe extern "ptx-kernel" fn sub_one(a: *mut f32) {
        let v: f32;
        ld.global.f32(v, [a]);
        sub.rn.f32(v, v, 1.0);
        st.global.f32([a], v);
    }
}
