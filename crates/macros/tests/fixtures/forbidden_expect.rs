//! A cooperative GPU kernel, an x86 kernel of instances and a cooperative one, whose
//! `#[expect(non_snake_case)]` each meets, in a crate that forbids that lint: Rust refuses each
//! `expect` once, where the kernel has it, for `k_blocks` keeps no lint and an instance has none.

#![forbid(non_snake_case)]

#[kernite_macros::kernels]
pub mod gpu {
    #[cooperative]
    #[expect(non_snake_case)]
    pub unsafe extern "ptx-kernel" fn k(xsPtr: *mut u32) {
        st.global.u32([xsPtr], 1);
    }
}

#[kernite_macros::kernels]
pub mod cpu {
    #[expect(non_snake_case)]
    #[instances(2, 3)]
    pub unsafe extern "sysv64" fn k<const N: usize>(xsPtr: *mut u8) {
        mov(byte[xsPtr], N);
    }

    #[expect(non_snake_case)]
    #[cooperative]
    #[instances(2, 3)]
    pub unsafe extern "sysv64" fn t<const N: usize>(xsPtr: *mut u8) {
        mov(byte[xsPtr], N);
    }
}
