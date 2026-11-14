//! What a module says when its parts fail: which of a part's errors is said.

#[kernite_macros::kernels]
pub mod gpu {
    // Its instances' error.
    #[instances(1)]
    pub unsafe extern "ptx-kernel" fn bad() {}

    // Its `#[cooperative]`'s error alone, and not its instances' as well.
    #[cooperative]
    #[instances(1)]
    extern "ptx-func" fn f() {}

    // It is not `unsafe`.
    pub extern "ptx-kernel" fn old() {}
}

#[kernite_macros::consts]
pub mod sizes {
    pub const STEP: u64 = 3;
}

// One error, at the static: what comes after reads the module without the const given.
#[kernite_macros::kernels(consts = crate::sizes)]
pub mod a_static_named_as_a_const_given {
    pub static STEP: global<[u32; 4]>;
}

// The error of a static in a module of both kinds alone: x86 does not lay `A` out, so it does not
// refuse its `f16`.
#[kernite_macros::kernels]
pub mod both {
    pub unsafe extern "sysv64" fn x() {}

    pub unsafe extern "ptx-kernel" fn g() {}

    static A: global<[f16; 4]>;
}

// The ABI's error alone: `c` is no kernel, so `#[cooperative]` says nothing, and its stub is the
// function as written, which `outside` calls.
#[kernite_macros::kernels]
pub mod an_abi_kernite_does_not_know {
    #[cooperative]
    pub unsafe extern "C" fn c() {}
}

/// Calls `c` as it is written.
pub fn outside() {
    unsafe { an_abi_kernite_does_not_know::c() }
}
