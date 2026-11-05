//! Kernels that take or give back a type kernite refuses, written raw: `r#f16`, `[r#f16; 4]`, `r#f128`, and a
//! pointer to a type with no size, `*const r#str` and `*mut [r#u64]`, which Rust reads as the names without
//! the `r#`. kernite says each at the type, once for each instance, as it says the same type written plain, and
//! Rust nothing more: not that the type is unstable, though the macros leave a stub for the kernel, nor
//! anything at a call of the kernel, which the crate's own function makes as it makes any call.

#[kernite_macros::kernels]
pub mod half {
    pub unsafe extern "sysv64" fn plain(x: r#f16) {
        ret();
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tiles<const N: u64>(x: r#f16) {
        ret();
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn together(x: r#f16) {
        ret();
    }

    #[cooperative]
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn together_tiles<const N: u64>(x: r#f16) {
        ret();
    }

    pub unsafe extern "sysv64" fn result(x: u64) -> r#f16 {
        ret(x);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tiles_result<const N: u64>(x: u64) -> r#f16 {
        ret(x);
    }
}

#[kernite_macros::kernels]
pub mod array {
    pub unsafe extern "sysv64" fn plain(x: [r#f16; 4]) {
        ret();
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tiles<const N: u64>(x: [r#f16; 4]) {
        ret();
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn together(x: [r#f16; 4]) {
        ret();
    }

    #[cooperative]
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn together_tiles<const N: u64>(x: [r#f16; 4]) {
        ret();
    }
}

#[kernite_macros::kernels]
pub mod wide {
    pub unsafe extern "sysv64" fn plain(x: r#f128) {
        ret();
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tiles<const N: u64>(x: r#f128) {
        ret();
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn together(x: r#f128) {
        ret();
    }

    #[cooperative]
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn together_tiles<const N: u64>(x: r#f128) {
        ret();
    }
}

#[kernite_macros::kernels]
pub mod slices {
    pub unsafe extern "sysv64" fn text(p: *const r#str) {
        ret();
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn words<const N: u64>(p: *mut [r#u64]) {
        ret();
    }
}

#[kernite_macros::kernels]
pub mod gpu_wide {
    pub unsafe extern "ptx-kernel" fn plain(x: r#f128) {
        ret();
    }

    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn tiles<const N: u32>(x: r#f128) {
        ret();
    }

    #[cooperative]
    pub unsafe extern "ptx-kernel" fn together(x: r#f128) {
        ret();
    }

    #[cooperative]
    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn together_tiles<const N: u32>(x: r#f128) {
        ret();
    }
}

#[kernite_macros::kernels]
pub mod gpu_array {
    pub unsafe extern "ptx-kernel" fn plain(x: [r#f16; 4]) {
        ret();
    }

    pub unsafe extern "ptx-kernel" fn text(p: *const r#str) {
        ret();
    }
}

#[kernite_macros::kernels]
pub mod results {
    #[cooperative]
    pub unsafe extern "sysv64" fn together(x: u64) -> r#f16 {
        ret(x);
    }

    #[cooperative]
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn together_tiles<const N: u64>(x: u64) -> r#f128 {
        ret(x);
    }

    pub unsafe extern "ptx-kernel" fn gpu(p: *mut u32, n: u32) -> r#f16 {
        st.global.u32([p], n);
    }

    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn gpu_tiles<const N: u32>(p: *mut u32) -> r#f128 {
        st.global.u32([p], N);
    }

    #[cooperative]
    pub unsafe extern "ptx-kernel" fn gpu_together(p: *mut u32, n: u32) -> r#u64 {
        st.global.u32([p], n);
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn together_number(x: u64) -> r#u64 {
        ret(x);
    }
}

/// Each x86 kernel, called as Rust calls it: with a value of any type where the kernel takes one of a type it
/// refuses, and its result read as a `u64` where it gives one back.
pub fn on_the_cpu() -> Result<(), kernite_cpu::Error> {
    unsafe {
        half::plain(1.0);
        half::tiles::<1>(1.0);
        half::together(kernite_cpu::Grid::new(1), 1.0)?;
        half::together_tiles::<2>(kernite_cpu::Grid::new(1), 1.0)?;
        let _: u64 = half::result(1);
        let _: u64 = half::tiles_result::<1>(1);
        array::plain([0.0; 4]);
        array::tiles::<1>([0.0; 4]);
        array::together(kernite_cpu::Grid::new(1), [0.0; 4])?;
        array::together_tiles::<2>(kernite_cpu::Grid::new(1), [0.0; 4])?;
        wide::plain(1.0);
        wide::tiles::<1>(1.0);
        wide::together(kernite_cpu::Grid::new(1), 1.0)?;
        wide::together_tiles::<2>(kernite_cpu::Grid::new(1), 1.0)?;
        slices::text("" as *const str);
        slices::words::<1>(core::ptr::slice_from_raw_parts_mut(core::ptr::null_mut::<u64>(), 0));
        results::together(kernite_cpu::Grid::new(1), 1)?;
        results::together_tiles::<2>(kernite_cpu::Grid::new(1), 1)?;
        results::together_number(kernite_cpu::Grid::new(1), 1)?;
    }
    Ok(())
}

/// Each GPU kernel, launched as Rust launches it: with a value of any type where the kernel takes one of a type
/// it refuses.
pub fn on_the_gpu(
    gpu_wide: &gpu_wide::Kernels,
    gpu_array: &gpu_array::Kernels,
    results: &results::Kernels,
    out: kernite_cuda::Ptr<u32>,
    text: kernite_cuda::Ptr<str>,
) -> Result<(), kernite_cuda::Error> {
    let grid = || kernite_cuda::Grid::new(1, 1);
    unsafe {
        gpu_wide.plain(grid(), 1.0)?;
        gpu_wide.tiles::<1>(grid(), 1.0)?;
        gpu_wide.together(grid(), 1.0)?;
        gpu_wide.together_tiles::<2>(grid(), 1.0)?;
        gpu_array.plain(grid(), [0.0; 4])?;
        gpu_array.text(grid(), text)?;
        results.gpu(grid(), out, 1)?;
        results.gpu_tiles::<1>(grid(), out)?;
        results.gpu_together(grid(), out, 1)?;
    }
    Ok(())
}
