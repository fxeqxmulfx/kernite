//! A module that gives `std` another meaning, `use crate::fake as std;`, has Rust say that it cannot find `f32` in
//! `std`, at each path that begins with `std`, in a const, a static's data and a kernel's operand on x86 and on a GPU,
//! and at none that begins with `::std`, which is the crate whatever the module gives. kernite says nothing: it takes
//! the path for Rust to find.

/// A module of the crate's own, with nothing in it.
pub mod fake {}

/// An x86 module that gives `std` another meaning.
#[kernite_macros::kernels]
pub mod fake_x86 {
    use crate::fake as std;

    pub const LOG2_E: u64 = std::f32::consts::LOG2_E.to_bits() as u64;
    pub static TABLE: global<[f32; 2]> = [::std::f32::consts::E, std::f32::consts::E];

    pub unsafe extern "sysv64" fn cpu(out: *mut u64) {
        let g: gpr;
        mov(g, LOG2_E);
        mov(qword[out], g);
        let v: ymm;
        vmovups(v, [[std::f32::consts::PI; 8]]);
        vmovups(v, [[::std::f32::consts::PI; 8]]);
    }
}

/// A GPU module that gives `std` another meaning.
#[kernite_macros::kernels]
pub mod fake_gpu {
    use crate::fake as std;

    pub unsafe extern "ptx-kernel" fn gpu(out: *mut f32) {
        let x: f32;
        mov.f32(x, ::std::f32::consts::LOG2_E);
        mov.f32(x, std::f32::consts::E);
        st.global.f32([out], x);
    }
}
