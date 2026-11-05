//! Kernels that take or give back a type kernite refuses, whose name is a keyword written raw: `r#dyn::Foo`, `r#mut`,
//! `r#for<u8>`, and a pointer to a type with no size, `*const [r#mut]`, which Rust reads as names. kernite says each
//! at the type, once for each instance, and writes the keyword as it is written, `r#mut` and not `mut`, which is
//! another word; and Rust nothing more: not that the type is unstable, though the macros leave a stub for the
//! kernel, nor anything at a call of the kernel, which the crate's own function makes as it makes any call. A
//! pointer to `r#str` is two words, as `str` is: `str` is no keyword, so kernite writes it without the `r#`.

/// A module named `dyn`.
pub mod r#dyn {
    #[repr(C)]
    pub struct Foo(pub u64);
}

/// A struct named `mut`.
#[repr(C)]
#[allow(non_camel_case_types)]
pub struct r#mut(pub u64);

/// A generic struct named `for`.
#[repr(C)]
#[allow(non_camel_case_types)]
pub struct r#for<T>(pub T);

#[kernite_macros::kernels]
pub mod by_value {
    use super::{r#dyn, r#for, r#mut};

    pub unsafe extern "sysv64" fn path(x: r#dyn::Foo) {
        ret();
    }

    pub unsafe extern "sysv64" fn name(x: r#mut) {
        ret();
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tiles<const N: u64>(x: r#for<u8>) {
        ret();
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn together(x: r#mut) {
        ret();
    }

    pub unsafe extern "sysv64" fn result(x: u64) -> r#mut {
        ret(x);
    }

    pub unsafe extern "sysv64" fn variable() {
        let v: [r#mut; 2];
    }
}

#[kernite_macros::kernels]
pub mod slices {
    use super::r#mut;

    pub unsafe extern "sysv64" fn many(p: *const [r#mut]) {
        ret();
    }

    pub unsafe extern "sysv64" fn text(p: *const r#str) {
        ret();
    }
}

#[kernite_macros::kernels]
pub mod gpu {
    use super::{r#dyn, r#mut};

    pub unsafe extern "ptx-kernel" fn path(x: r#dyn::Foo) {
        ret();
    }

    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn tiles<const N: u32>(x: r#mut) {
        ret();
    }

    pub unsafe extern "ptx-kernel" fn many(p: *const [r#mut]) {
        ret();
    }
}

/// Each x86 kernel, called as Rust calls it: with a value of the type where the kernel takes one of a type it
/// refuses, and its result read as a `u64` where it gives one back.
pub fn on_the_cpu() -> Result<(), kernite_cpu::Error> {
    unsafe {
        by_value::path(r#dyn::Foo(1));
        by_value::name(r#mut(1));
        by_value::tiles::<1>(r#for(1u8));
        by_value::together(kernite_cpu::Grid::new(1), r#mut(1))?;
        let _: u64 = by_value::result(1);
        by_value::variable();
        slices::many(core::ptr::slice_from_raw_parts(core::ptr::null::<r#mut>(), 0));
        slices::text("" as *const str);
    }
    Ok(())
}

/// Each GPU kernel, launched as Rust launches it: with a value of the type where the kernel takes one of a type it
/// refuses.
pub fn on_the_gpu(
    kernels: &gpu::Kernels,
    many: kernite_cuda::Ptr<[r#mut]>,
) -> Result<(), kernite_cuda::Error> {
    let grid = kernite_cuda::Grid::new(1, 1);
    unsafe {
        kernels.path(grid, r#dyn::Foo(1))?;
        kernels.tiles::<1>(grid, r#mut(1))?;
        kernels.many(grid, many)?;
    }
    Ok(())
}
