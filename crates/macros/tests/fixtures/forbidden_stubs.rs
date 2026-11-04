//! Kernels that do not compile, in a crate that forbids each lint the macro used to `allow`: each
//! is said where it does not compile, beside a stub of its signature or its launcher, a module
//! given consts as well, and nothing else is said of them or of the pointers they take or give back.

#![forbid(
    dead_code,
    non_snake_case,
    unused_variables,
    unused_imports,
    unused_macros,
    unreachable_patterns
)]

#[kernite_macros::consts]
pub mod sizes {
    /// How far a value moves.
    pub const STEP: u64 = 3;
}

#[kernite_macros::kernels]
pub mod cpu {
    pub unsafe extern "sysv64" fn broken(x: u64) -> u64 {
        ret(y);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tile<const N: u64>(x: u64) -> u64 {
        ret(y);
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn together(x: *mut u64) {
        mov([x], y);
    }
}

#[kernite_macros::kernels]
pub mod gpu {
    pub unsafe extern "ptx-kernel" fn broken(x: *mut u32) {
        st.global.u32([x], y);
    }

    #[cooperative]
    pub unsafe extern "ptx-kernel" fn together(x: *mut u32) {
        st.global.u32([x], y);
    }
}

#[kernite_macros::kernels(consts = crate::sizes)]
pub mod cpu_taking {
    pub unsafe extern "sysv64" fn broken(x: u64) -> u64 {
        ret(y);
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn together(x: *mut u64) {
        mov([x], y);
    }
}

#[kernite_macros::kernels(consts = crate::sizes)]
pub mod gpu_taking {
    pub unsafe extern "ptx-kernel" fn broken(x: *mut u32) {
        st.global.u32([x], y);
    }

    #[cooperative]
    pub unsafe extern "ptx-kernel" fn together(x: *mut u32) {
        st.global.u32([x], y);
    }
}

#[kernite_macros::kernels]
pub mod cpu_pointers {
    pub unsafe extern "sysv64" fn broken(x: *mut u64) -> *const u64 {
        mov([x], y);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tile<const N: u64>(x: *mut u64) -> *mut u64 {
        mov([x], y);
    }
}
