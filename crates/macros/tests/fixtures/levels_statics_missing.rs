//! A crate that depends on neither `kernite-cpu` nor `kernite-cuda`, with a static of x86 kernels and one in the GPU's
//! memory that warn of `deprecated` where the crate allows it: Rust tells of each place their data names a deprecated
//! item, as a warning, beside kernite's errors that say the crate needs them, as it does in a crate that has them.
#![allow(deprecated)]

#[kernite_macros::kernels]
pub mod cpu {
    #[deprecated]
    const K: u8 = 5;

    #[warn(deprecated)]
    pub static WARNS: global<[u8; 2]> = [K, K];

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

    #[warn(deprecated)]
    pub static WARNS: constant<[u32; 2]> = [K, K];
}
