//! A cooperative GPU kernel, an x86 kernel of instances and a cooperative one, whose `#[expect(non_snake_case)]` each
//! meets, in a crate that forbids that lint: Rust refuses each `expect` where the kernel has it, as `k_blocks` keeps no
//! lint, and on x86 the `allow` that the function of each instance has for the `expect`, at the same place.

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
