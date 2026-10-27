//! Kernels of each kind that compile, in private modules, where Rust says what is never used:
//! it builds them without a word.

/// Each x86 kernel, called as Rust calls it.
pub fn on_the_cpu(x: *mut u32) -> Result<u64, kgen_cpu::Error> {
    unsafe {
        cpu::together(kgen_cpu::Grid::new(1), x)?;
        Ok(cpu::step(1) + cpu::tile::<2>(1) + cpu::COUNTS.read()[0] as u64)
    }
}

/// Each GPU kernel, launched as Rust launches it.
pub fn on_the_gpu(kernels: &gpu::Kernels, x: kgen_cuda::Ptr<f32>) -> Result<u32, kgen_cuda::Error> {
    unsafe {
        kernels.scale(kgen_cuda::Grid::new(1, 1), x, 2.0)?;
        kernels.tile::<2>(kgen_cuda::Grid::new(1, 1), x)?;
        kernels.together(kgen_cuda::Grid::new(1, 1), x)?;
    }
    kernels.COUNTS().write(&[0; 4])?;
    kernels.together_blocks(32, 0)
}

#[kgen_macros::kernels]
mod cpu {
    /// Read by a kernel alone.
    const STEP: u64 = 3;

    pub static COUNTS: global<[u32; 4]>;

    pub unsafe extern "sysv64" fn step(x: u64) -> u64 {
        add(x, STEP);
        ret(x);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tile<const N: u64>(x: u64) -> u64 {
        add(x, N);
        ret(x);
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn together(x: *mut u32) {
        grid.sync();
        let counts: gpr;
        lea(counts, [COUNTS]);
        mov(dword[x], 1);
    }
}

#[kgen_macros::kernels]
mod gpu {
    /// Read by a kernel alone.
    const SCALE: f32 = 2.0;

    pub static COUNTS: global<[u32; 4]>;

    pub unsafe extern "ptx-kernel" fn scale(x: *mut f32, a: f32) {
        let v: f32;
        ld.global.f32(v, [x]);
        mul.rn.f32(v, v, a);
        mul.rn.f32(v, v, SCALE);
        st.global.f32([x], v);
    }

    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn tile<const N: u32>(x: *mut f32) {
        let at: u64;
        mov.u64(at, COUNTS);
        red.global.add.u32([at], N);
    }

    #[cooperative]
    pub unsafe extern "ptx-kernel" fn together(x: *mut f32) {
        grid.sync();
    }
}
