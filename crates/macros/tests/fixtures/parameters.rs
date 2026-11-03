//! Kernels with a name that a local the macro's Rust binds for itself had: `index`, which the
//! launcher of a GPU kernel of instances bound for the instance; `grid`, the launcher's; and
//! `threads` and `shared`, the parameters of the method that says how many blocks a cooperative
//! GPU kernel's grid has at most. A parameter of that name is the kernel's own, and reaches it. So
//! is a compile-time argument, but for `grid`, `threads` and `shared` where the Rust the macro
//! writes binds them (`arguments.rs` holds those). Rust builds each module as written, without a
//! word.

/// Each GPU kernel, launched as Rust launches it, with the names it takes.
pub fn on_the_gpu(kernels: &gpu::Kernels, xs: kgen_cuda::Ptr<u32>) -> Result<u32, kgen_cuda::Error> {
    let grid = kgen_cuda::Grid::new(1, 1);
    unsafe {
        kernels.indexed::<2>(grid, xs, 40)?;
        kernels.gridded(grid, xs, 7)?;
        kernels.together(grid, xs, 32, 0)?;
        kernels.counted::<2>(grid, xs)?;
        kernels.sized::<1, 2>(grid, xs)?;
    }
    kernels.together_blocks(32, 0)
}

/// Each x86 kernel, called as Rust calls it, with the names it takes.
pub fn on_the_cpu(xs: *mut u64) -> Result<u64, kgen_cpu::Error> {
    unsafe {
        cpu::together(kgen_cpu::Grid::new(1), xs, 1, 2, 3)?;
        Ok(cpu::indexed::<2>(40) + cpu::gridded::<3>(4))
    }
}

/// GPU kernels. `indexed` stores `index + N`; `counted` is of instances picked by an `index`, and
/// `sized` of `threads` and `shared`, which it has no method to bind, as it is not cooperative.
#[kgen_macros::kernels]
pub mod gpu {
    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn indexed<const N: u32>(xs: *mut u32, index: u32) {
        let y: u32;
        add.u32(y, index, N);
        st.global.u32([xs], y);
    }

    pub unsafe extern "ptx-kernel" fn gridded(xs: *mut u32, grid: u32) {
        st.global.u32([xs], grid);
    }

    #[cooperative]
    pub unsafe extern "ptx-kernel" fn together(xs: *mut u32, threads: u32, shared: u32) {
        let y: u32;
        add.u32(y, threads, shared);
        st.global.u32([xs], y);
    }

    #[instances(1, 2)]
    #[allow(non_upper_case_globals)]
    pub unsafe extern "ptx-kernel" fn counted<const index: u32>(xs: *mut u32) {
        st.global.u32([xs], index);
    }

    #[instances((1, 2))]
    #[allow(non_upper_case_globals)]
    pub unsafe extern "ptx-kernel" fn sized<const threads: u32, const shared: u32>(xs: *mut u32) {
        let y: u32;
        add.u32(y, threads, shared);
        st.global.u32([xs], y);
    }
}

/// x86 kernels. `indexed` is of instances, with an `index`; `together` is cooperative, and its
/// launcher passes each parameter on, the pointer `grid` as the address it is, which the thread
/// makes a pointer again; `gridded` has a compile-time argument `grid`, and no launcher to bind it,
/// as it is not cooperative.
#[kgen_macros::kernels]
pub mod cpu {
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn indexed<const N: u64>(index: u64) -> u64 {
        add(index, N);
        ret(index);
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn together(grid: *mut u64, threads: u64, shared: u64, index: u64) {
        add(threads, shared);
        add(threads, index);
        mov(qword[grid], threads);
    }

    #[instances(3)]
    #[allow(non_upper_case_globals)]
    pub unsafe extern "sysv64" fn gridded<const grid: u64>(x: u64) -> u64 {
        add(x, grid);
        ret(x);
    }
}
