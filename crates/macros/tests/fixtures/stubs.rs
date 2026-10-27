//! Kernels that do not compile, each said where it does not, beside a stub of its signature or
//! its launcher, which Rust and the module's kernels call as they would the kernel: nothing
//! else is said of them.

#[kgen_macros::kernels]
pub mod cpu {
    pub unsafe extern "sysv64" fn broken(x: u64) -> u64 {
        ret(y);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tile<const N: u64>(x: u64) -> u64 {
        ret(y);
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn together(x: *mut u64) {
        mov([x], y);
    }

    pub unsafe extern "sysv64" fn caller(x: u64) -> u64 {
        let r: gpr;
        call(r, broken, x);
        call(r, tile::<1>, r);
        ret(r);
    }
}

#[kgen_macros::kernels]
pub mod gpu {
    pub unsafe extern "ptx-kernel" fn broken(x: *mut u32) {
        st.global.u32([x], y);
    }

    #[cooperative]
    pub unsafe extern "ptx-kernel" fn together(x: *mut u32) {
        st.global.u32([x], y);
    }
}

/// Each x86 kernel, called as Rust calls it.
pub fn on_the_cpu(x: *mut u64) -> Result<u64, kgen_cpu::Error> {
    unsafe {
        cpu::together(kgen_cpu::Grid::new(1), x)?;
        Ok(cpu::broken(1) + cpu::tile::<1>(1) + cpu::caller(1))
    }
}

/// Each GPU kernel, launched as Rust launches it.
pub fn on_the_gpu(kernels: &gpu::Kernels, x: kgen_cuda::Ptr<u32>) -> Result<u32, kgen_cuda::Error> {
    unsafe {
        kernels.broken(kgen_cuda::Grid::new(1, 1), x)?;
        kernels.together(kgen_cuda::Grid::new(1, 1), x)?;
    }
    kernels.together_blocks(32, 0)
}
