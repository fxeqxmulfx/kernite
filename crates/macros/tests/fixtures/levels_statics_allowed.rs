//! A static that allows `deprecated` is told of nothing by Rust at any place its type or its data names a deprecated
//! item, as a static of Rust's that allows it is told of nothing, though the crate denies the lint, which would make
//! each place an error: on x86, a static in reach of `rip` and one far from it, and on the GPU, one in constant memory
//! and one in global memory. So it is of a static whose `allow` is in a `cfg_attr(..)`.
#![deny(deprecated)]

#[kernite_macros::kernels]
pub mod cpu {
    #[deprecated]
    const K: u8 = 5;

    #[deprecated]
    const N: usize = 2;

    #[deprecated]
    const FAR: usize = 3221225472;

    /// Names `K` twice, in its data.
    #[allow(deprecated)]
    pub static DATA: global<[u8; 2]> = [K, K];

    /// Names `N` in its type.
    #[allow(deprecated)]
    pub static TYPE: global<[u8; N]>;

    /// Names `N` in its type and `K` twice in its data.
    #[allow(deprecated)]
    pub static BOTH: global<[u8; N]> = [K, K];

    /// Allows it where `cfg` keeps the attribute, which it does.
    #[cfg_attr(all(), allow(deprecated))]
    pub static CONDITIONAL: global<[u8; N]> = [K, K];

    /// Is 3 GiB, far from `rip`, and names `FAR` in its type.
    #[allow(deprecated)]
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

    /// Names `K` twice, in its data.
    #[allow(deprecated)]
    pub static DATA: constant<[u32; 2]> = [K, K];

    /// Names `N` in its type.
    #[allow(deprecated)]
    pub static TYPE: global<[u32; N]>;

    /// Names `N` in its type and `K` twice in its data.
    #[allow(deprecated)]
    pub static BOTH: global<[u32; N]> = [K, K];

    /// Allows it where `cfg` keeps the attribute, which it does.
    #[cfg_attr(all(), allow(deprecated))]
    pub static CONDITIONAL: constant<[u32; N]> = [K, K];
}
