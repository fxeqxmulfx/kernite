//! A path through the primitive, `core::primitive::f32::consts::PI`, is no path of Rust's: `core::primitive::f32` is
//! the type, which has no `consts`, however the module names `f32`. Rust says "ambiguous associated type" of it, at the
//! path, as it says of the path in a const of its own, in a module that has `use std::f32;` and in one that has not:
//! in a const, a static's data and a kernel's operand on x86 and on a GPU. kernite takes the path as it is written,
//! for Rust to refuse it, and says nothing.

/// Rust's own word for the path, in a const that is not a kernel's.
pub const PLAIN: f32 = core::primitive::f32::consts::PI;

/// An x86 module that has `use std::f32;`, which Rust's path `f32::consts::PI` uses, and the path through the primitive
/// does not.
#[kernite_macros::kernels]
pub mod with_use {
    use std::f32;

    pub const BITS: u64 = core::primitive::f32::consts::PI.to_bits() as u64;
    pub static TABLE: global<[f32; 2]> = [::std::primitive::f32::consts::E, 1.0];

    pub unsafe extern "sysv64" fn cpu(out: *mut u64) {
        let g: gpr;
        mov(g, BITS);
        mov(qword[out], g);
        let v: ymm;
        vmovups(v, [[::core::primitive::f32::consts::TAU; 8]]);
        vmovups(v, [[f32::consts::PI; 8]]);
    }
}

/// A GPU module that has no such `use`.
#[kernite_macros::kernels]
pub mod without_use {
    pub static SCALES: constant<[f64; 2]> = [core::primitive::f64::consts::E, 1.0];

    pub unsafe extern "ptx-kernel" fn gpu(out: *mut u64) {
        let x: f64;
        mov.f64(x, ::std::primitive::f64::consts::PI);
        st.global.f64([out], x);
    }
}
