//! x86 kernels that take or give back an `f16` or a `bf16` by value, which only a GPU has: kernite
//! refuses each at the type, once for each instance, and Rust says nothing more. Rust has no `bf16`,
//! and its `f16` is unstable; Rust says neither of the stub the macros leave for the kernel, which the
//! crate's own functions call as they would the kernel.

#[kernite_macros::kernels]
pub mod halves {
    pub unsafe extern "sysv64" fn plain(x: f16) {
        ret();
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tiles<const N: u64>(x: f16) {
        ret();
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn together(x: f16) {
        ret();
    }

    #[cooperative]
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn together_tiles<const N: u64>(x: f16) {
        ret();
    }

    pub unsafe extern "sysv64" fn result(x: u64) -> f16 {
        ret(x);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tiles_result<const N: u64>(x: u64) -> f16 {
        ret(x);
    }
}

#[kernite_macros::kernels]
pub mod brains {
    pub unsafe extern "sysv64" fn plain(x: bf16) {
        ret();
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tiles<const N: u64>(x: bf16) {
        ret();
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn together(x: bf16) {
        ret();
    }

    #[cooperative]
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn together_tiles<const N: u64>(x: bf16) {
        ret();
    }

    pub unsafe extern "sysv64" fn result(x: u64) -> bf16 {
        ret(x);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tiles_result<const N: u64>(x: u64) -> bf16 {
        ret(x);
    }
}

/// Each kernel, called as Rust calls it: a float where the kernel takes one, and its result read as a
/// `u64` where it gives one back.
pub fn on_the_cpu() -> Result<(), kernite_cpu::Error> {
    unsafe {
        halves::plain(1.0);
        halves::tiles::<1>(1.0);
        halves::together(kernite_cpu::Grid::new(1), 1.0)?;
        halves::together_tiles::<2>(kernite_cpu::Grid::new(1), 1.0)?;
        let _: u64 = halves::result(1);
        let _: u64 = halves::tiles_result::<2>(1);
        brains::plain(1.0);
        brains::tiles::<1>(1.0);
        brains::together(kernite_cpu::Grid::new(1), 1.0)?;
        brains::together_tiles::<2>(kernite_cpu::Grid::new(1), 1.0)?;
        let _: u64 = brains::result(1);
        let _: u64 = brains::tiles_result::<2>(1);
    }
    Ok(())
}
