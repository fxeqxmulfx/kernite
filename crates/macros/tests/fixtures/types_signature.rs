//! Kernels that take or give back a type kernite refuses, in which Rust finds what it finds in any function's
//! signature: a lifetime it cannot find, a `_` in a result, a module where a type is written, and an `f16` or an
//! `f128` kernite does not see as a type. kernite says each type, once for each instance, and Rust says its own
//! word once, at the fault. The crate's functions call each kernel as Rust calls it, and Rust says nothing there.

#[kernite_macros::kernels]
pub mod result_reference {
    /// A result `&u64` has no lifetime to take: Rust says so, at the `&`.
    pub unsafe extern "sysv64" fn plain(x: u64) -> &u64 {
        ret(x);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tiles<const N: u64>(x: u64) -> &u64 {
        ret(x);
    }

    /// The parameter `&u64` has a lifetime, which the result takes: Rust says nothing of the result.
    pub unsafe extern "sysv64" fn borrows(x: &u64) -> &u64 {
        ret(x);
    }
}

#[kernite_macros::kernels]
pub mod result_blank {
    /// A `_` is no result: Rust says so, at the `_`.
    pub unsafe extern "sysv64" fn plain(x: u64) -> _ {
        ret(x);
    }
}

#[kernite_macros::kernels]
pub mod param_lifetime {
    /// There is no `'a`: Rust says so, at the lifetime.
    pub unsafe extern "sysv64" fn plain(x: &'a u64) {
        ret();
    }
}

#[kernite_macros::kernels]
pub mod gpu_param_lifetime {
    /// There is no `'a`: Rust says so, at the lifetime.
    pub unsafe extern "ptx-kernel" fn plain(x: &'a u64) {
        ret();
    }
}

#[kernite_macros::kernels]
pub mod name_lifetime {
    /// There is no `'f16`: Rust says so, at the lifetime, as it does of an `'a`. The name is no type.
    pub unsafe extern "sysv64" fn plain(x: &'f16 u64) {
        ret();
    }
}

#[kernite_macros::kernels]
pub mod gpu_name_lifetime {
    /// There is no `'f128`: Rust says so, at the lifetime, as it does of an `'a`. The name is no type.
    pub unsafe extern "ptx-kernel" fn plain(x: &'f128 u64) {
        ret();
    }
}

macro_rules! half {
    () => {
        f16
    };
}

macro_rules! wide {
    () => {
        f128
    };
}

#[kernite_macros::kernels]
pub mod macro_type {
    /// An `f16` that a macro writes: kernite does not see a type, and Rust says it is unstable.
    pub unsafe extern "sysv64" fn plain(x: half!()) {
        ret();
    }
}

#[kernite_macros::kernels]
pub mod length_type {
    /// An `f16` in an array's length: kernite does not see a type, and Rust says it is unstable.
    pub unsafe extern "sysv64" fn plain(x: [u8; size_of::<f16>()]) {
        ret();
    }
}

#[kernite_macros::kernels]
pub mod gpu_macro_type {
    /// An `f128` that a macro writes: kernite does not see a type, and Rust says it is unstable.
    pub unsafe extern "ptx-kernel" fn plain(x: wide!()) {
        ret();
    }
}

#[kernite_macros::kernels]
pub mod gpu_length_type {
    /// An `f128` in an array's length: kernite does not see a type, and Rust says it is unstable.
    pub unsafe extern "ptx-kernel" fn plain(x: [u8; size_of::<f128>()]) {
        ret();
    }
}

/// A module of the crate, named as a half is.
pub mod f16 {}

/// A module of the crate, named as a wide float is.
pub mod f128 {}

#[kernite_macros::kernels]
pub mod core_module {
    /// `core::f16` is a module, no type: Rust says so, at the path.
    pub unsafe extern "sysv64" fn plain(x: core::f16) {
        ret();
    }
}

#[kernite_macros::kernels]
pub mod module_type {
    use crate::f16;

    /// An `f16` where the module imports a module so named: Rust reads it as the module, which is no type, and
    /// says so, at the `f16`.
    pub unsafe extern "sysv64" fn plain(x: f16) {
        ret();
    }
}

#[kernite_macros::kernels]
pub mod module_array {
    use crate::f16;

    /// An `f16` in an array, where the module imports a module so named: Rust reads it as the module, which is
    /// no type, and says so, at the `f16`.
    pub unsafe extern "sysv64" fn plain(x: [f16; 4]) {
        ret();
    }
}

#[kernite_macros::kernels]
pub mod gpu_core_module {
    /// `core::f128` is a module, no type: Rust says so, at the path.
    pub unsafe extern "ptx-kernel" fn plain(x: [core::f128; 2]) {
        ret();
    }
}

#[kernite_macros::kernels]
pub mod gpu_module_type {
    use crate::f128;

    /// An `f128` where the module imports a module so named: Rust reads it as the module, which is no type, and
    /// says so, at the `f128`.
    pub unsafe extern "ptx-kernel" fn plain(x: f128) {
        ret();
    }
}

#[kernite_macros::kernels]
pub mod gpu_module_array {
    use crate::f16;

    /// An `f16` in an array, where the module imports a module so named: Rust reads it as the module, which is
    /// no type, and says so, at the `f16`.
    pub unsafe extern "ptx-kernel" fn plain(x: [f16; 4]) {
        ret();
    }
}

/// Each x86 kernel, called as Rust calls it.
pub fn on_the_cpu() {
    unsafe {
        let _: u64 = result_reference::plain(1);
        let _: u64 = result_reference::tiles::<1>(1);
        let _: u64 = result_reference::borrows(&1);
        let _: u64 = result_blank::plain(1);
        param_lifetime::plain(&1);
        name_lifetime::plain(&1);
        macro_type::plain(1);
        length_type::plain([0; 2]);
        core_module::plain(1);
        module_type::plain(1);
        module_array::plain([0; 4]);
    }
}

/// Each GPU kernel, launched as Rust launches it.
pub fn on_the_gpu(
    param_lifetime: &gpu_param_lifetime::Kernels,
    name_lifetime: &gpu_name_lifetime::Kernels,
    macro_type: &gpu_macro_type::Kernels,
    length_type: &gpu_length_type::Kernels,
    core_module: &gpu_core_module::Kernels,
    module_type: &gpu_module_type::Kernels,
    module_array: &gpu_module_array::Kernels,
) -> Result<(), kernite_cuda::Error> {
    unsafe {
        param_lifetime.plain(kernite_cuda::Grid::new(1, 1), &1)?;
        name_lifetime.plain(kernite_cuda::Grid::new(1, 1), &1)?;
        macro_type.plain(kernite_cuda::Grid::new(1, 1), 1)?;
        length_type.plain(kernite_cuda::Grid::new(1, 1), [0; 16])?;
        core_module.plain(kernite_cuda::Grid::new(1, 1), [0; 2])?;
        module_type.plain(kernite_cuda::Grid::new(1, 1), 1)?;
        module_array.plain(kernite_cuda::Grid::new(1, 1), [0; 4])
    }
}
