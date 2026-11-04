//! Kernels with a result that none of their kind may have, whose type is a macro in braces, `byte!{ _ }`: a
//! cooperative kernel and a GPU kernel, which give back nothing, each plain and of compile-time arguments, and an
//! x86 kernel whose result is no number or pointer. kernite refuses each at the result, in the words it has for
//! that kind of kernel, as it refuses a result with no macro in it.

/// A macro that gives a type, and drops what it is given: `byte!{ _ }` is `u8`.
macro_rules! byte {
    ($x:tt) => {
        u8
    };
}

/// x86 kernels.
#[kernite_macros::kernels]
pub mod cpu {
    #[cooperative]
    pub unsafe extern "sysv64" fn cooperative(p: *const u8) -> *const byte!{ _ } {
        ret(p);
    }

    #[cooperative]
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn cooperative_tiles<const N: u64>(p: *const u8) -> *const byte!{ _ } {
        add(p, N);
        ret(p);
    }

    pub unsafe extern "sysv64" fn number(p: *const u8) -> byte!{ _ } {
        ret(p);
    }
}

/// GPU kernels.
#[kernite_macros::kernels]
pub mod gpu {
    pub unsafe extern "ptx-kernel" fn plain(p: *const u8) -> *const byte!{ _ } {
        ret();
    }

    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn tiles<const N: u64>(p: *const u8) -> *const byte!{ _ } {
        ret();
    }
}
