//! Compile-time arguments named as a local the Rust the macro writes binds for itself, which no
//! span of stable Rust keeps apart: `__kernite_grid`, the launcher's, and `__kernite_threads` and `__kernite_shared`,
//! of the method that says how many blocks a cooperative GPU kernel's grid has at most. Each is refused at its name,
//! and the launcher or the method never runs, so that Rust says no more, and the code that calls it is no error.

/// Each launcher and method, called as written.
pub fn callers(
    grid: &gpu_grid::Kernels,
    threads: &gpu_threads::Kernels,
    shared: &gpu_shared::Kernels,
    xs: kernite_cuda::Ptr<u32>,
    x: *mut u64,
) -> Result<u32, Box<dyn std::error::Error>> {
    unsafe {
        grid.k::<1>(kernite_cuda::Grid::new(1, 1), xs)?;
        cpu_grid::k::<1>(kernite_cpu::Grid::new(1), x)?;
    }
    Ok(threads.k_blocks::<1>(32, 0)? + shared.k_blocks::<2>(32, 0)?)
}

/// A GPU kernel whose compile-time argument is the launcher's `__kernite_grid`.
#[kernite_macros::kernels]
pub mod gpu_grid {
    #[instances(1, 2)]
    #[allow(non_upper_case_globals)]
    pub unsafe extern "ptx-kernel" fn k<const __kernite_grid: u32>(xs: *mut u32) {
        st.global.u32([xs], 1);
    }
}

/// A cooperative GPU kernel whose compile-time argument is `__kernite_threads`, a parameter of `k_blocks`.
#[kernite_macros::kernels]
pub mod gpu_threads {
    #[instances(1, 2)]
    #[cooperative]
    #[allow(non_upper_case_globals)]
    pub unsafe extern "ptx-kernel" fn k<const __kernite_threads: u32>(xs: *mut u32) {
        st.global.u32([xs], 1);
    }
}

/// The same, with `__kernite_shared`, the other parameter of `k_blocks`.
#[kernite_macros::kernels]
pub mod gpu_shared {
    #[instances(1, 2)]
    #[cooperative]
    #[allow(non_upper_case_globals)]
    pub unsafe extern "ptx-kernel" fn k<const __kernite_shared: u32>(xs: *mut u32) {
        st.global.u32([xs], 1);
    }
}

/// A cooperative x86 kernel whose compile-time argument is the launcher's `__kernite_grid`.
#[kernite_macros::kernels]
pub mod cpu_grid {
    #[instances(1, 2)]
    #[cooperative]
    #[allow(non_upper_case_globals)]
    pub unsafe extern "sysv64" fn k<const __kernite_grid: u64>(x: *mut u64) {
        mov(qword[x], 1);
    }
}

/// The same, with the name written raw, `r#__kernite_grid`, which Rust reads as `__kernite_grid`.
#[kernite_macros::kernels]
pub mod cpu_raw_grid {
    #[instances(1, 2)]
    #[cooperative]
    #[allow(non_upper_case_globals)]
    pub unsafe extern "sysv64" fn k<const r#__kernite_grid: u64>(x: *mut u64) {
        mov(qword[x], 1);
    }
}
