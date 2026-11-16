//! Kernels with a name that a local the macro's Rust binds for itself has or had: `index`, which the launcher of a GPU
//! kernel of instances bound for the instance; `__kernite_grid`, the launcher's, which was `grid`; and `threads` and
//! `shared`, which the parameters of the method that says how many blocks a cooperative GPU kernel's grid has at most
//! were. A parameter of that name is the kernel's own, and reaches it. So is a compile-time argument, but for
//! `__kernite_grid`, `__kernite_threads` and `__kernite_shared` (`arguments.rs` holds those). Rust builds each module
//! as written, without a word.

/// Each GPU kernel, launched as Rust launches it, with the names it takes.
pub fn on_the_gpu(kernels: &gpu::Kernels, xs: kernite_cuda::Ptr<u32>) -> Result<u32, kernite_cuda::Error> {
    let grid = kernite_cuda::Grid::new(1, 1);
    unsafe {
        kernels.indexed::<2>(grid, xs, 40)?;
        kernels.gridded(grid, xs, 7)?;
        kernels.together(grid, xs, 32, 0)?;
        kernels.counted::<2>(grid, xs)?;
        kernels.sized::<1, 2, 3>(grid, xs)?;
    }
    Ok(kernels.together_blocks(32, 0)? + kernels.sized_blocks::<1, 2, 3>(32, 0)?)
}

/// Each x86 kernel, called as Rust calls it, with the names it takes.
pub fn on_the_cpu(xs: *mut u64) -> Result<u64, kernite_cpu::Error> {
    unsafe {
        cpu::together(kernite_cpu::Grid::new(1), xs, 1, 2, 3)?;
        cpu::pooled::<3>(kernite_cpu::Grid::new(1), xs)?;
        Ok(cpu::indexed::<2>(40) + cpu::gridded::<3>(4))
    }
}

/// GPU kernels. `indexed` stores `index + N`; `gridded` takes a parameter named as the launcher's own grid; `counted`
/// is of instances picked by an `index`; and `sized`, which is cooperative, of `grid`, `threads` and `shared`, which
/// its methods do not bind.
#[kernite_macros::kernels]
pub mod gpu {
    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn indexed<const N: u32>(xs: *mut u32, index: u32) {
        let y: u32;
        add.u32(y, index, N);
        st.global.u32([xs], y);
    }

    pub unsafe extern "ptx-kernel" fn gridded(xs: *mut u32, __kernite_grid: u32) {
        st.global.u32([xs], __kernite_grid);
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

    #[cooperative]
    #[instances((1, 2, 3))]
    #[allow(non_upper_case_globals)]
    pub unsafe extern "ptx-kernel" fn sized<const grid: u32, const threads: u32, const shared: u32>(xs: *mut u32) {
        let y: u32;
        add.u32(y, threads, shared);
        add.u32(y, y, grid);
        st.global.u32([xs], y);
    }
}

/// x86 kernels. `indexed` is of instances, with an `index`; `together` is cooperative, and its
/// launcher passes each parameter on, the pointer `__kernite_grid`, named as the launcher's own grid, as the address it
/// is, which the thread makes a pointer again; `gridded` has a compile-time argument `grid`, and so has `pooled`, which
/// is cooperative, for its launcher binds no `grid`.
#[kernite_macros::kernels]
pub mod cpu {
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn indexed<const N: u64>(index: u64) -> u64 {
        add(index, N);
        ret(index);
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn together(__kernite_grid: *mut u64, threads: u64, shared: u64, index: u64) {
        add(threads, shared);
        add(threads, index);
        mov(qword[__kernite_grid], threads);
    }

    #[cooperative]
    #[instances(3)]
    #[allow(non_upper_case_globals)]
    pub unsafe extern "sysv64" fn pooled<const grid: u64>(x: *mut u64) {
        mov(qword[x], grid);
    }

    #[instances(3)]
    #[allow(non_upper_case_globals)]
    pub unsafe extern "sysv64" fn gridded<const grid: u64>(x: u64) -> u64 {
        add(x, grid);
        ret(x);
    }
}
