//! A static that lowers `deprecated` by `warn`, or expects it, is told of at the level it sets at each place its type
//! or its data names a deprecated item, where the crate denies the lint, as a static of Rust's is: `WARNS` as a warning
//! at each place, and `IN_DATA`, `IN_TYPE` and `LONG` nothing, as each meets its expectation: the data of the first
//! names the item, and the type of each of the others, of which `LONG` is far from `rip`. Of `UNMET`, whose type and
//! data name a const that is not deprecated, Rust says that its expectation is unfulfilled, once, at the attribute; and
//! `GONE`, which `cfg` leaves out, has none. `CRATE` sets no level, and each place of it is an error. So it is on x86
//! and on the GPU, in constant memory and in global memory.
#![deny(deprecated)]

#[kernite_macros::kernels]
pub mod cpu {
    #[deprecated]
    const K: u8 = 5;

    #[deprecated]
    const N: usize = 2;

    #[deprecated]
    const FAR: usize = 3221225472;

    const PLAIN: u8 = 1;

    /// Names `N` in its type and `K` twice in its data.
    #[warn(deprecated)]
    pub static WARNS: global<[u8; N]> = [K, K];

    #[expect(deprecated)]
    pub static IN_DATA: global<[u8; 2]> = [K, K];

    #[expect(deprecated)]
    pub static IN_TYPE: global<[u8; N]>;

    #[expect(deprecated)]
    pub static UNMET: global<[u8; 2]> = [PLAIN, PLAIN];

    #[cfg(any())]
    #[expect(deprecated)]
    pub static GONE: global<[u8; 2]> = [PLAIN, PLAIN];

    pub static CRATE: global<[u8; N]> = [K, K];

    /// Is 3 GiB, far from `rip`, and names `FAR` in its type.
    #[expect(deprecated)]
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

    const PLAIN: u32 = 1;

    /// Names `N` in its type and `K` twice in its data.
    #[warn(deprecated)]
    pub static WARNS: constant<[u32; N]> = [K, K];

    #[expect(deprecated)]
    pub static IN_DATA: global<[u32; 2]> = [K, K];

    #[expect(deprecated)]
    pub static IN_TYPE: constant<[u32; N]>;

    #[expect(deprecated)]
    pub static UNMET: constant<[u32; 2]> = [PLAIN, PLAIN];

    #[cfg(any())]
    #[expect(deprecated)]
    pub static GONE: constant<[u32; 2]> = [PLAIN, PLAIN];

    pub static CRATE: constant<[u32; N]> = [K, K];
}
