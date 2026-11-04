//! Modules of GPU items under `cfg`, in a crate that has no `kernite-cuda`: each says it needs one,
//! once, where it is, and only where one of its GPU items is kept, whichever features keep it.
//! Each module's items are under the features `a` and `b`, in each form `cfg` takes them by.

/// One kernel, under `a`.
#[kernite_macros::kernels]
pub mod just_a {
    #[cfg(feature = "a")]
    pub unsafe extern "ptx-kernel" fn one(p: *mut u32) {
        st.global.u32([p], 1);
    }
}

/// A kernel under `a`, written raw, and a function under `b`: kept where either is.
#[kernite_macros::kernels]
pub mod a_or_b {
    #[r#cfg(feature = "a")]
    pub unsafe extern "ptx-kernel" fn one(p: *mut u32) {
        st.global.u32([p], 1);
    }

    #[cfg(feature = "b")]
    extern "ptx-func" fn two() -> u32 {
        let v: u32;
        mov.u32(v, 2);
        ret(v);
    }
}

/// A kernel under two `cfg`s: kept where both hold.
#[kernite_macros::kernels]
pub mod a_and_b {
    #[cfg(feature = "a")]
    #[cfg(feature = "b")]
    pub unsafe extern "ptx-kernel" fn one(p: *mut u32) {
        st.global.u32([p], 1);
    }
}

/// A kernel under `b` where `a` holds, and kept where `a` does not.
#[kernite_macros::kernels]
pub mod a_then_b {
    #[cfg_attr(feature = "a", cfg(feature = "b"))]
    pub unsafe extern "ptx-kernel" fn one(p: *mut u32) {
        st.global.u32([p], 1);
    }
}

/// A kernel left out where `a` holds and `b` does, by a `cfg_attr(..)` in another, after a lint
/// that is no condition.
#[kernite_macros::kernels]
pub mod nested {
    #[cfg_attr(feature = "a", allow(dead_code), cfg_attr(feature = "b", cfg(any())))]
    pub unsafe extern "ptx-kernel" fn one(p: *mut u32) {
        st.global.u32([p], 1);
    }
}

/// A static under `b`.
#[kernite_macros::kernels]
pub mod static_b {
    #[cfg(feature = "b")]
    pub static COUNTS: global<[u32; 4]>;
}

/// A kernel that no `cfg` touches, and one under `a`: the module needs `kernite-cuda` whatever the
/// features say.
#[kernite_macros::kernels]
pub mod always {
    pub unsafe extern "ptx-kernel" fn one(p: *mut u32) {
        st.global.u32([p], 1);
    }

    #[cfg(feature = "a")]
    pub unsafe extern "ptx-kernel" fn two(p: *mut u32) {
        st.global.u32([p], 2);
    }
}

/// A GPU kernel under `a` and an x86 kernel under `b`, which needs no crate: kept where `a` is.
#[kernite_macros::kernels]
pub mod mixed {
    #[cfg(feature = "a")]
    pub unsafe extern "ptx-kernel" fn gpu(p: *mut u32) {
        st.global.u32([p], 1);
    }

    #[cfg(feature = "b")]
    pub unsafe extern "sysv64" fn host(p: *mut u64) {
        mov(qword[p], 1);
    }
}
