//! A const that `cfg` keeps and that only a kernel `cfg` leaves out reads: Rust finds it unused, as it
//! would in a crate that never had the kernel, and says so; with the kernel there, it does not.

#[kernite_macros::kernels]
pub mod kept {
    const N: u64 = 4;

    #[cfg(feature = "cpu")]
    pub unsafe extern "sysv64" fn reads(p: *mut u64) {
        mov(qword[p], N);
    }
}
