//! A static that raises `deprecated`, by `warn`, `deny` or `forbid`, is told of at each place its type or its data
//! names a deprecated item, as a warning or an error though the crate allows the lint, as a static of Rust's is: on
//! x86, a static in reach of `rip` and one far from it, and on the GPU, one in constant memory and one in global
//! memory. `CRATE` sets no level, and Rust says nothing of it.
#![allow(deprecated)]

#[kernite_macros::kernels]
pub mod cpu {
    #[deprecated]
    const K: u8 = 5;

    #[deprecated]
    const N: usize = 2;

    #[deprecated]
    const FAR: usize = 3221225472;

    /// Names `N` in its type and `K` twice in its data.
    #[warn(deprecated)]
    pub static WARNS: global<[u8; N]> = [K, K];

    #[deny(deprecated)]
    pub static DENIES: global<[u8; N]> = [K, K];

    #[forbid(deprecated)]
    pub static FORBIDS: global<[u8; N]> = [K, K];

    pub static CRATE: global<[u8; N]> = [K, K];

    /// Is 3 GiB, far from `rip`, and names `FAR` in its type.
    #[deny(deprecated)]
    pub static LONG: global<[u8; FAR]>;

    /// Writes 1 to the byte at `p`, and names no item. A module with an x86 kernel has its statics in the process's
    /// memory; with none, in the GPU's.
    pub unsafe extern "sysv64" fn one(p: *mut u8) {
        mov(byte[p], 1);
    }
}

#[kernite_macros::kernels]
pub mod gpu {
    #[deprecated]
    const K: u32 = 5;

    #[deprecated]
    const N: usize = 2;

    /// Names `N` in its type and `K` twice in its data.
    #[warn(deprecated)]
    pub static WARNS: constant<[u32; N]> = [K, K];

    #[deny(deprecated)]
    pub static DENIES: global<[u32; N]> = [K, K];

    #[forbid(deprecated)]
    pub static FORBIDS: constant<[u32; N]> = [K, K];

    pub static CRATE: constant<[u32; N]> = [K, K];
}
