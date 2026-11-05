//! A module of consts, which Rust reads too, takes the paths Rust refuses as a module of kernels does: a path without
//! its root, `f32::consts::LOG2_E`, one through the primitive, `core::primitive::f32::consts::PI`, and an unstable
//! constant, `std::f32::consts::SQRT_3`. Rust says what is wrong with each, once, at the path, and kernite says
//! nothing, of the consts or of the kernels that take them, on x86 and on a GPU.

#[kernite_macros::consts]
pub mod given {
    pub const ROOTLESS: u64 = f32::consts::LOG2_E.to_bits() as u64;
    pub const THROUGH: u64 = core::primitive::f32::consts::PI.to_bits() as u64;
    pub const UNSTABLE: u64 = std::f32::consts::SQRT_3.to_bits() as u64;
}

/// An x86 module that reads each.
#[kernite_macros::kernels(consts = crate::given)]
pub mod cpu {
    pub unsafe extern "sysv64" fn k(out: *mut u64) {
        let g: gpr;
        mov(g, ROOTLESS);
        mov(qword[out], g);
        mov(g, THROUGH);
        mov(qword[out + 8], g);
        mov(g, UNSTABLE);
        mov(qword[out + 16], g);
    }
}

/// A GPU module that reads each.
#[kernite_macros::kernels(consts = crate::given)]
pub mod gpu {
    pub unsafe extern "ptx-kernel" fn k(out: *mut u64) {
        let x: u64;
        mov.u64(x, ROOTLESS);
        st.global.u64([out], x);
        mov.u64(x, THROUGH);
        st.global.u64([out + 8], x);
        mov.u64(x, UNSTABLE);
        st.global.u64([out + 16], x);
    }
}
