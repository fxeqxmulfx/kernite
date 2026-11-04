//! What rests on all of a module's items together is said as it is for a module that has no `cfg`,
//! and so is an item kernite cannot read, which rests on no other: kernite reads every item before
//! it knows what `cfg` leaves out. Each case is a module of its own, for each stops its module.

/// The head of a function that kernite cannot read, a generic that is no const, left out.
#[kernite_macros::kernels]
pub mod head {
    #[cfg(any())]
    pub unsafe extern "sysv64" fn generic<T>(p: *mut u64) {
        mov(qword[p], 1);
    }
}

/// Two functions of a name, each under a `cfg` the other is not.
#[kernite_macros::kernels]
pub mod twice {
    #[cfg(all())]
    pub unsafe extern "sysv64" fn k(p: *mut u64) {
        mov(qword[p], 1);
    }

    #[cfg(not(all()))]
    pub unsafe extern "sysv64" fn k(p: *mut u64) {
        mov(qword[p], 2);
    }
}

/// A cooperative kernel, and a kernel named as its method `k_blocks`, left out.
#[kernite_macros::kernels]
pub mod blocks {
    #[cooperative]
    pub unsafe extern "ptx-kernel" fn k() {}

    #[cfg(any())]
    pub unsafe extern "ptx-kernel" fn k_blocks() {}
}
