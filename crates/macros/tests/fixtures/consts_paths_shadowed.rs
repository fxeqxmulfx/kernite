//! A module that gives a name in the path of a constant of Rust's a meaning of its own, `use crate::fake_std as std;`,
//! where that meaning has a constant of its own at the path, has Rust say, at the path, that it is not the constant
//! kernite takes: Rust reads the path as the module gives it, and its bits are not those of the constant the path
//! names where nothing shadows it. It says so of a `std`, a `core`, an `f32` and an `f64`, in a const, a static's data
//! and a kernel's operand on x86, and in an operand on a GPU, and says nothing of a kernel that `cfg` leaves out.
//! kernite says nothing else: it takes the constant for Rust to hold to its bits.

/// What `std` is in a module that says `use crate::fake_std as std;`, or `core` in one that says `as core;`: it has
/// an `f32` and an `f64` of its own, with a `PI` and an `E` of their own.
pub mod fake_std {
    pub mod f32 {
        pub mod consts {
            pub const PI: f32 = 3.0;
        }
    }
    pub mod f64 {
        pub mod consts {
            pub const E: f64 = 2.5;
        }
    }
}

/// What `f32` is in a module that says `use crate::fake_f32 as f32;`: it has `consts` of its own, with a `PI` of its
/// own.
pub mod fake_f32 {
    pub mod consts {
        pub const PI: f32 = 3.0;
    }
}

/// An x86 module that gives `std` another meaning.
#[kernite_macros::kernels]
pub mod std_x86 {
    use crate::fake_std as std;

    pub const BITS: u64 = std::f32::consts::PI.to_bits() as u64;
    pub static TABLE: global<[f64; 2]> = [std::f64::consts::E, 1.0];

    pub unsafe extern "sysv64" fn cpu(out: *mut u64) {
        let g: gpr;
        mov(g, BITS);
        mov(qword[out], g);
        let v: ymm;
        vmovups(v, [[std::f32::consts::PI; 8]]);
    }
}

/// A GPU module that gives `std` another meaning.
#[kernite_macros::kernels]
pub mod std_gpu {
    use crate::fake_std as std;

    pub static SCALES: constant<[f64; 2]> = [std::f64::consts::E, 1.0];

    pub unsafe extern "ptx-kernel" fn gpu(out: *mut u64) {
        let x: f32;
        mov.f32(x, std::f32::consts::PI);
        st.global.f32([out], x);
    }
}

/// An x86 module that gives `f32` another meaning, and names the constant without its root, as Rust reads it after
/// `use std::f32;`.
#[kernite_macros::kernels]
pub mod f32_x86 {
    use crate::fake_f32 as f32;

    pub const BITS: u64 = f32::consts::PI.to_bits() as u64;

    pub unsafe extern "sysv64" fn cpu(out: *mut u64) {
        let g: gpr;
        mov(g, BITS);
        mov(qword[out], g);
        let v: ymm;
        vmovups(v, [[f32::consts::PI; 8]]);
    }
}

/// A GPU module that gives `f32` another meaning.
#[kernite_macros::kernels]
pub mod f32_gpu {
    use crate::fake_f32 as f32;

    pub unsafe extern "ptx-kernel" fn gpu(out: *mut u64) {
        let x: f32;
        mov.f32(x, f32::consts::PI);
        st.global.f32([out], x);
    }
}

/// What `f64` is in a module that says `use crate::fake_f64 as f64;`: it has `consts` of its own, with an `E` of its
/// own.
pub mod fake_f64 {
    pub mod consts {
        pub const E: f64 = 2.5;
    }
}

/// An x86 module that gives `core` another meaning: a const and an operand.
#[kernite_macros::kernels]
pub mod core_x86 {
    use crate::fake_std as core;

    pub const BITS: u64 = core::f64::consts::E.to_bits();

    pub unsafe extern "sysv64" fn cpu(out: *mut u64) {
        let g: gpr;
        mov(g, BITS);
        mov(qword[out], g);
        let v: ymm;
        vmovupd(v, [[core::f64::consts::E; 4]]);
    }
}

/// A GPU module that gives `f64` another meaning, and a kernel that `cfg` leaves out, of which Rust says nothing: its
/// path is as wrong as the other's.
#[kernite_macros::kernels]
pub mod f64_gpu {
    use crate::fake_f64 as f64;

    pub unsafe extern "ptx-kernel" fn gpu(out: *mut u64) {
        let x: f64;
        mov.f64(x, f64::consts::E);
        st.global.f64([out], x);
    }

    #[cfg(any())]
    pub unsafe extern "ptx-kernel" fn left_out(out: *mut u64) {
        let x: f64;
        mov.f64(x, f64::consts::E);
        st.global.f64([out], x);
    }
}
