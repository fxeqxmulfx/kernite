//! Kernels with a result that none of their kind may have, whose type holds a block among its arguments,
//! `Arr<{ 3 }>`: a cooperative kernel and a GPU kernel, which give back nothing, each plain and of compile-time
//! arguments, and an x86 kernel whose result is no number or pointer. kernite refuses each at the result, as it
//! refuses a result with no block in it. The crate allows Rust's lint on the braces, which these kernels are not
//! about.

#![allow(unused_braces)]

/// A type with a number as its argument.
#[repr(C)]
pub struct Arr<const N: usize>(pub [u8; N]);

/// x86 kernels.
#[kernite_macros::kernels]
pub mod cpu {
    #[cooperative]
    pub unsafe extern "sysv64" fn cooperative(p: *const u8) -> *const crate::Arr<{ 3 }> {
        ret(p);
    }

    #[cooperative]
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn cooperative_tiles<const N: u64>(p: *const u8) -> *const crate::Arr<{ 3 }> {
        add(p, N);
        ret(p);
    }

    pub unsafe extern "sysv64" fn number(p: *const u8) -> crate::Arr<{ 3 }> {
        ret(p);
    }
}

/// GPU kernels.
#[kernite_macros::kernels]
pub mod gpu {
    pub unsafe extern "ptx-kernel" fn plain(p: *const u8) -> *const crate::Arr<{ 3 }> {
        ret();
    }

    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn tiles<const N: u64>(p: *const u8) -> *const crate::Arr<{ 3 }> {
        ret();
    }
}
