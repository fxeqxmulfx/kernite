//! A path that names no constant kernite knows is kernite's error alone, at the path, and Rust is not asked of it:
//! a name the modules have not, `::core::f64::consts::NOPE`; a type that has none, `f16::consts::PI`; one that Rust
//! has, deprecated, in a module of the type, `std::u32::MAX`; and a number's own constant that kernite has no value
//! for, `f32::NAN`. Each is in a kernel of its own, for x86 and for a GPU, as a kernel is refused at the first it
//! cannot work out.

/// A path that names no constant kernite knows, in an x86 kernel each.
#[kernite_macros::kernels]
pub mod unknown_x86 {
    pub unsafe extern "sysv64" fn name() -> u64 {
        let g: gpr;
        mov(g, ::core::f64::consts::NOPE);
        ret(g);
    }

    pub unsafe extern "sysv64" fn halves() -> u64 {
        let g: gpr;
        mov(g, f16::consts::PI);
        ret(g);
    }

    pub unsafe extern "sysv64" fn legacy() -> u64 {
        let g: gpr;
        mov(g, std::u32::MAX);
        ret(g);
    }

    pub unsafe extern "sysv64" fn nan() -> u64 {
        let g: gpr;
        mov(g, f32::NAN);
        ret(g);
    }
}

/// The same for a GPU, with a static's data and a helper's.
#[kernite_macros::kernels]
pub mod unknown_gpu {
    pub static SCALES: constant<[f64; 1]> = [::std::f64::consts::PHI];

    /// A path that names no constant kernite knows, in a helper.
    fn scaled(x: f32) {
        mul.rn.f32(x, x, core::f32::consts::INFINITY);
    }

    pub unsafe extern "ptx-kernel" fn first(out: *mut f32) {
        let x: f32;
        mov.f32(x, std::u32::MAX);
        st.global.f32([out], x);
    }

    pub unsafe extern "ptx-kernel" fn second(out: *mut f32) {
        let x: f32;
        mov.f32(x, 1.0);
        scaled(x);
        st.global.f32([out], x);
    }
}
