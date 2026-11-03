//! A crate that forbids each lint the macro used to `allow` builds its kernels without a word: a
//! module of each kind of kernel, x86 and GPU, of instances and not, cooperative, with a static,
//! and with consts given, in private modules, where Rust says what is never used. `k_` is a
//! cooperative kernel whose `_blocks` method has two underscores. The `forbid` is of every lint
//! but Clippy's, which Rust does not check.

#![forbid(
    dead_code,
    non_snake_case,
    unused_variables,
    unused_imports,
    unused_macros,
    unreachable_patterns
)]

/// Each x86 kernel, called as Rust calls it.
pub fn on_the_cpu(x: &mut u32) -> Result<u64, kernite_cpu::Error> {
    let grid = kernite_cpu::Grid::new(1);
    unsafe {
        cpu::together(grid, x)?;
        cpu::rounds::<2>(grid, x)?;
        Ok(cpu::step(1) + cpu::tile::<2>(1) + cpu::flag::<true>(1) + cpu::COUNTS.read()[0] as u64)
    }
}

/// Each GPU kernel, launched as Rust launches it.
pub fn on_the_gpu(kernels: &gpu::Kernels, x: kernite_cuda::Ptr<f32>) -> Result<u32, kernite_cuda::Error> {
    let grid = kernite_cuda::Grid::new(1, 1);
    unsafe {
        kernels.scale(grid, x, 2.0)?;
        kernels.tile::<2>(grid, x)?;
        kernels.flag::<true>(grid, x)?;
        kernels.together(grid, x)?;
        kernels.rounds::<2>(grid, x)?;
        kernels.k_(grid, x)?;
    }
    kernels.COUNTS().write(&[0; 4])?;
    kernels.together_blocks(32, 0)?;
    kernels.rounds_blocks::<2>(32, 0)?;
    kernels.k__blocks(32, 0)
}

/// Each kernel of the modules that take consts, called as Rust calls it.
pub fn with_consts(
    kernels: &gpu_taking::Kernels,
    x: kernite_cuda::Ptr<f32>,
    y: &mut u32,
) -> Result<u64, Box<dyn std::error::Error>> {
    let grid = kernite_cuda::Grid::new(1, 1);
    unsafe {
        kernels.scale(grid, x)?;
        kernels.tile::<2>(grid, x)?;
        kernels.flag::<true>(grid, x)?;
        kernels.together(grid, x)?;
        kernels.rounds::<2>(grid, x)?;
        kernels.k_(grid, x)?;
    }
    kernels.COUNTS().write(&[0; 4])?;
    kernels.together_blocks(32, 0)?;
    kernels.rounds_blocks::<2>(32, 0)?;
    kernels.k__blocks(32, 0)?;
    let grid = kernite_cpu::Grid::new(1);
    unsafe {
        cpu_taking::together(grid, y)?;
        cpu_taking::rounds::<2>(grid, y)?;
        let step = cpu_taking::step(1) + cpu_taking::tile::<2>(1) + cpu_taking::flag::<true>(1);
        Ok(step + cpu_taking::COUNTS.read()[0] as u64)
    }
}

/// Consts that kernels share, one of which no kernel reads.
#[kernite_macros::consts]
pub mod sizes {
    /// The length of a static.
    pub const LEN: usize = 4;

    /// How far a value moves.
    pub const STEP: u64 = 3;

    /// Read by no kernel, as a module of kernels that takes `sizes` need not read each const.
    pub const UNREAD: u32 = 1;
}

/// Consts that no module of kernels takes.
#[kernite_macros::consts]
pub mod spare {
    /// Read by no kernel.
    pub const NOBODY: u32 = 1;
}

#[kernite_macros::kernels]
mod cpu {
    /// Read by a kernel alone.
    const STEP: u64 = 3;

    /// Read by a static alone.
    const LEN: usize = 4;

    pub static COUNTS: global<[u32; LEN]>;

    pub unsafe extern "sysv64" fn step(x: u64) -> u64 {
        add(x, STEP);
        ret(x);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tile<const N: u64>(x: u64) -> u64 {
        add(x, N);
        ret(x);
    }

    /// Every value of a `bool` is an instance.
    #[instances(false, true)]
    pub unsafe extern "sysv64" fn flag<const ON: bool>(x: u64) -> u64 {
        if ON {
            add(x, 1);
        }
        ret(x);
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn together(x: *mut u32) {
        grid.sync();
        let counts: gpr;
        lea(counts, [COUNTS]);
        mov(dword[x], 1);
    }

    #[instances(1, 2)]
    #[cooperative]
    pub unsafe extern "sysv64" fn rounds<const N: u64>(x: *mut u32) {
        grid.sync();
        mov(dword[x], N);
    }
}

#[kernite_macros::kernels]
mod gpu {
    /// Read by a kernel alone.
    const SCALE: f32 = 2.0;

    /// Read by a static alone.
    const LEN: usize = 4;

    pub static COUNTS: global<[u32; LEN]>;

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

    /// Every value of a `bool` is an instance.
    #[instances(false, true)]
    pub unsafe extern "ptx-kernel" fn flag<const ON: bool>(x: *mut f32) {
        let at: u64;
        mov.u64(at, COUNTS);
        if ON {
            red.global.add.u32([at], 1);
        }
    }

    #[cooperative]
    pub unsafe extern "ptx-kernel" fn together(x: *mut f32) {
        grid.sync();
    }

    #[instances(1, 2)]
    #[cooperative]
    pub unsafe extern "ptx-kernel" fn rounds<const N: u32>(x: *mut f32) {
        let at: u64;
        mov.u64(at, COUNTS);
        red.global.add.u32([at], N);
        grid.sync();
    }

    /// A name that ends in an underscore, whose `_blocks` method has two.
    #[cooperative]
    pub unsafe extern "ptx-kernel" fn k_(x: *mut f32) {
        grid.sync();
    }
}

#[kernite_macros::kernels(consts = crate::sizes)]
mod cpu_taking {
    pub static COUNTS: global<[u32; LEN]>;

    pub unsafe extern "sysv64" fn step(x: u64) -> u64 {
        add(x, STEP);
        ret(x);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tile<const N: u64>(x: u64) -> u64 {
        add(x, N);
        ret(x);
    }

    /// Every value of a `bool` is an instance.
    #[instances(false, true)]
    pub unsafe extern "sysv64" fn flag<const ON: bool>(x: u64) -> u64 {
        if ON {
            add(x, 1);
        }
        ret(x);
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn together(x: *mut u32) {
        grid.sync();
        let counts: gpr;
        lea(counts, [COUNTS]);
        mov(dword[x], 1);
    }

    #[instances(1, 2)]
    #[cooperative]
    pub unsafe extern "sysv64" fn rounds<const N: u64>(x: *mut u32) {
        grid.sync();
        mov(dword[x], N);
    }
}

#[kernite_macros::kernels(consts = crate::sizes)]
mod gpu_taking {
    pub static COUNTS: global<[u32; LEN]>;

    pub unsafe extern "ptx-kernel" fn scale(x: *mut f32) {
        let at: u64;
        mov.u64(at, COUNTS);
        red.global.add.u32([at], STEP);
    }

    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn tile<const N: u32>(x: *mut f32) {
        let at: u64;
        mov.u64(at, COUNTS);
        red.global.add.u32([at], N);
    }

    /// Every value of a `bool` is an instance.
    #[instances(false, true)]
    pub unsafe extern "ptx-kernel" fn flag<const ON: bool>(x: *mut f32) {
        let at: u64;
        mov.u64(at, COUNTS);
        if ON {
            red.global.add.u32([at], 1);
        }
    }

    #[cooperative]
    pub unsafe extern "ptx-kernel" fn together(x: *mut f32) {
        grid.sync();
    }

    #[instances(1, 2)]
    #[cooperative]
    pub unsafe extern "ptx-kernel" fn rounds<const N: u32>(x: *mut f32) {
        let at: u64;
        mov.u64(at, COUNTS);
        red.global.add.u32([at], N);
        grid.sync();
    }

    /// A name that ends in an underscore, whose `_blocks` method has two.
    #[cooperative]
    pub unsafe extern "ptx-kernel" fn k_(x: *mut f32) {
        grid.sync();
    }
}
