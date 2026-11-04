//! Kernels that `big` leaves out with all they name: one of compile-time arguments, which names a static and calls
//! an x86 kernel, and a cooperative one, which names a static. With no feature, the crate says nothing; with `big`,
//! it says nothing, and each kernel runs.
#![deny(warnings)]

#[kernite_macros::kernels]
pub mod cpu {
    #[cfg(feature = "big")]
    pub static COUNT: global<[u64; 1]>;

    #[cfg(feature = "big")]
    pub unsafe extern "sysv64" fn eight() -> u64 {
        let v: gpr;
        mov(v, 8);
        ret(v);
    }

    #[cfg(feature = "big")]
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn inst<const N: u64>(p: *mut u64) {
        let v: gpr;
        call(v, eight, ());
        add(v, qword[COUNT]);
        add(v, N);
        mov(qword[p], v);
    }

    #[cfg(feature = "big")]
    #[cooperative]
    pub unsafe extern "sysv64" fn together(p: *mut u64) {
        let v: gpr;
        mov(v, qword[COUNT]);
        mov(qword[p], v);
    }
}

#[cfg(all(test, feature = "big"))]
mod tests {
    /// Each kernel runs, and writes through its pointer what it names: `eight` and `COUNT`, which is 0.
    #[test]
    fn the_kernels_run_where_the_feature_keeps_them() {
        let p = &mut 9u64;
        // SAFETY: `p` is the u64 each kernel writes, and nothing else touches `COUNT`.
        unsafe {
            super::cpu::inst::<2>(p);
            assert_eq!(*p, 10);
            *p = 9;
            super::cpu::together(kernite_cpu::Grid::new(1), p).unwrap();
            assert_eq!(*p, 0);
        }
    }
}
