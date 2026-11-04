//! Kernels, helpers and statics with each error kernite says of one, in each form `cfg` takes them
//! out by. Built without the feature `cpu`, Rust leaves them out and the crate builds without a
//! word, under `deny(warnings)`, as it does a crate that never had them; built with it, kernite
//! says each error where it is.

#![deny(warnings)]

#[kernite_macros::kernels]
pub mod cpu {
    /// A kernel that compiles, which `cfg` does not touch.
    pub unsafe extern "sysv64" fn one(p: *mut u64) {
        mov(qword[p], 1);
    }

    /// A parameter kernite does not take.
    #[cfg(feature = "cpu")]
    pub unsafe extern "sysv64" fn narrow(x: u32) {
        let y: gpr;
        mov(y, x);
    }

    /// A name that is not defined, in a kernel that is deprecated and hidden from the docs: the
    /// error is under its `cfg` alone.
    #[deprecated]
    #[doc(hidden)]
    #[cfg(feature = "cpu")]
    pub unsafe extern "sysv64" fn undefined(p: *mut u64) {
        mov(qword[p], y);
    }

    /// An instruction that is not x86's, under a `cfg` written raw.
    #[r#cfg(feature = "cpu")]
    pub unsafe extern "sysv64" fn unknown(p: *mut u64) {
        frobnicate(qword[p]);
    }

    /// An ABI kernite does not compile.
    #[cfg(feature = "cpu")]
    pub unsafe extern "C" fn abi(p: *mut u64) {
        mov(qword[p], 1);
    }

    /// A kernel that is not `unsafe`.
    #[cfg(feature = "cpu")]
    pub extern "sysv64" fn safe(p: *mut u64) {
        mov(qword[p], 1);
    }

    /// A kernel of instances, whose errors are one for each.
    #[cfg(feature = "cpu")]
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tile<const N: u64>(x: u64) -> u64 {
        ret(y);
    }

    /// A kernel with `#[instances(..)]` and no compile-time argument to list them for.
    #[cfg(feature = "cpu")]
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn plain(x: u64) -> u64 {
        ret(x);
    }

    /// A cooperative kernel.
    #[cfg(feature = "cpu")]
    #[cooperative]
    pub unsafe extern "sysv64" fn together(p: *mut u64) {
        mov(qword[p], y);
    }

    /// A cooperative kernel with a compile-time argument named as a parameter of its launcher.
    #[cfg(feature = "cpu")]
    #[cooperative]
    #[instances(1, 2)]
    #[allow(non_upper_case_globals)]
    pub unsafe extern "sysv64" fn taken<const grid: u64>(p: *mut u64) {
        mov(qword[p], 1);
    }

    /// A helper that does not compile, which no kernel calls.
    #[cfg(feature = "cpu")]
    fn help() {
        while true {}
    }

    /// A static that holds nothing.
    #[cfg(feature = "cpu")]
    pub static EMPTY: global<[u32; 0]>;

    /// Out where the feature is not there: the `cfg_attr(..)` is a `cfg(any())` without it.
    #[cfg_attr(not(feature = "cpu"), cfg(any()))]
    pub unsafe extern "sysv64" fn conditional(p: *mut u64) {
        mov(qword[p], y);
    }

    /// The same in a `cfg_attr(..)` inside another, and after a lint that the error does not keep.
    #[cfg_attr(all(), cfg_attr(not(feature = "cpu"), allow(dead_code), cfg(any())))]
    pub unsafe extern "sysv64" fn nested(p: *mut u64) {
        mov(qword[p], y);
    }

    /// Out whatever the feature says, for each `cfg` must hold and the second never does.
    #[cfg(feature = "cpu")]
    #[cfg(any())]
    pub unsafe extern "sysv64" fn never(p: *mut u64) {
        mov(qword[p], y);
    }
}

#[kernite_macros::kernels]
pub mod gpu {
    /// A kernel that compiles, which `cfg` does not touch.
    pub unsafe extern "ptx-kernel" fn one(p: *mut u32) {
        st.global.u32([p], 1);
    }

    /// A name that is not defined.
    #[cfg(feature = "cpu")]
    pub unsafe extern "ptx-kernel" fn undefined(p: *mut u32) {
        st.global.u32([p], y);
    }

    /// An instruction that is not PTX's.
    #[cfg(feature = "cpu")]
    pub unsafe extern "ptx-kernel" fn unknown(p: *mut u32) {
        frobnicate.u32([p]);
    }

    /// A kernel that is not `unsafe`.
    #[cfg(feature = "cpu")]
    pub extern "ptx-kernel" fn safe(p: *mut u32) {
        st.global.u32([p], 1);
    }

    /// A kernel of instances, whose errors are one for each.
    #[cfg(feature = "cpu")]
    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn tile<const N: u32>(p: *mut u32) {
        st.global.u32([p], y);
    }

    /// A kernel with `#[instances(..)]` and no compile-time argument to list them for.
    #[cfg(feature = "cpu")]
    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn plain(p: *mut u32) {
        st.global.u32([p], 1);
    }

    /// A cooperative kernel.
    #[cfg(feature = "cpu")]
    #[cooperative]
    pub unsafe extern "ptx-kernel" fn together(p: *mut u32) {
        st.global.u32([p], y);
    }

    /// A compile-time argument named as a parameter of the launcher.
    #[cfg(feature = "cpu")]
    #[instances(1, 2)]
    #[allow(non_upper_case_globals)]
    pub unsafe extern "ptx-kernel" fn taken<const grid: u32>(p: *mut u32) {
        st.global.u32([p], 1);
    }

    /// A compile-time argument of a cooperative kernel named as a parameter of the method that
    /// says how many blocks its grid has at most.
    #[cfg(feature = "cpu")]
    #[cooperative]
    #[instances(1, 2)]
    #[allow(non_upper_case_globals)]
    pub unsafe extern "ptx-kernel" fn counted<const threads: u32>(p: *mut u32) {
        st.global.u32([p], 1);
    }

    /// A function that kernels call.
    #[cfg(feature = "cpu")]
    pub unsafe extern "ptx-func" fn func(p: u64) {
        st.global.u32([p], y);
    }

    /// A static that holds nothing.
    #[cfg(feature = "cpu")]
    pub static EMPTY: global<[u32; 0]>;

    /// Out where the feature is not there.
    #[cfg_attr(not(feature = "cpu"), cfg(any()))]
    pub unsafe extern "ptx-kernel" fn conditional(p: *mut u32) {
        st.global.u32([p], y);
    }
}

/// A cooperative kernel under a `cfg` and a kernel without one, whose names make one method of
/// `Kernels` twice where the first is there.
#[kernite_macros::kernels]
pub mod clash {
    #[cfg(feature = "cpu")]
    #[cooperative]
    pub unsafe extern "ptx-kernel" fn k() {}

    pub unsafe extern "ptx-kernel" fn k_blocks() {}
}

/// Consts under the feature, and what reads them, under it too: an x86 kernel and the length of a
/// static. A helper that the kernel calls has no `cfg` of its own, for it leaves nothing behind:
/// what it reads is named under the kernel's. Rust never sees a kernel run, so kernite names each
/// const where the items that read it are, and Rust says nothing of the consts without the
/// feature, nor of them with it.
#[kernite_macros::kernels]
pub mod x86_reads {
    #[cfg(feature = "cpu")]
    const FIRST: u64 = 1;
    #[cfg(feature = "cpu")]
    const SECOND: u64 = 2;
    #[cfg(feature = "cpu")]
    const LEN: usize = 4;

    fn put(p: gpr) {
        mov(qword[p], SECOND);
    }

    #[cfg(feature = "cpu")]
    pub unsafe extern "sysv64" fn reads(p: *mut u64) {
        mov(qword[p], FIRST);
        put(p);
    }

    #[cfg(feature = "cpu")]
    pub static COUNTS: global<[u32; LEN]>;
}

/// The same for GPU kernels: a kernel, a function that a kernel calls, and the length of a static.
#[kernite_macros::kernels]
pub mod gpu_reads {
    #[cfg(feature = "cpu")]
    const FIRST: u32 = 1;
    #[cfg(feature = "cpu")]
    const SECOND: u32 = 2;
    #[cfg(feature = "cpu")]
    const LEN: usize = 4;

    #[cfg(feature = "cpu")]
    extern "ptx-func" fn second() -> u32 {
        let v: u32;
        mov.u32(v, SECOND);
        ret(v);
    }

    #[cfg(feature = "cpu")]
    pub unsafe extern "ptx-kernel" fn reads(p: *mut u32) {
        let g: u32;
        call(g, second, ());
        st.global.u32([p], g);
        st.global.u32([p], FIRST);
    }

    #[cfg(feature = "cpu")]
    pub static COUNTS: global<[u32; LEN]>;
}

/// A module of consts of its own, whose const only a kernel that `cfg` leaves out reads: the `use`
/// that gives the module of kernels its consts is as quiet as the kernel.
#[kernite_macros::consts]
pub mod sizes {
    /// What the kernel reads.
    pub const ONLY: u64 = 3;
}

#[kernite_macros::kernels(consts = crate::sizes)]
pub mod given {
    #[cfg(feature = "cpu")]
    pub unsafe extern "sysv64" fn reads(p: *mut u64) {
        mov(qword[p], ONLY);
    }
}
