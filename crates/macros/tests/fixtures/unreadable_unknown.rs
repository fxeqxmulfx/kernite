//! Modules kernite cannot read, for the head of a kernel it does not take, `bad<T>`: kernite knows the kernels beside
//! it, `good` and `scale` before it and `late` after it, and not `bad`. Rust says nothing of code that names `bad` by
//! the module's path or by `use`, and cannot find it by a glob, nor, of a GPU kernel, as a method of `Kernels`.

#[kernite_macros::kernels]
pub mod cpu {
    pub unsafe extern "sysv64" fn good(x: u64) -> u64 {
        ret(x);
    }

    pub unsafe extern "sysv64" fn bad<T>(x: u64) -> u64 {
        ret(x);
    }

    pub unsafe extern "sysv64" fn late(x: u64) -> u64 {
        ret(x);
    }
}

#[kernite_macros::kernels]
pub mod gpu {
    pub unsafe extern "ptx-kernel" fn scale(x: *mut f32, a: f32) {
        let v: f32;
        ld.global.f32(v, [x]);
        mul.rn.f32(v, v, a);
        st.global.f32([x], v);
    }

    pub unsafe extern "ptx-kernel" fn bad<T>(x: *mut f32) {
        let v: f32;
        ld.global.f32(v, [x]);
        st.global.f32([x], v);
    }

    pub unsafe extern "ptx-kernel" fn late(x: *mut f32) {
        let v: f32;
        ld.global.f32(v, [x]);
        st.global.f32([x], v);
    }
}

/// `scale` and `late` are found, as methods; `bad` is not.
pub fn by_method(kernels: &gpu::Kernels, p: kernite_cuda::Ptr<f32>) -> Result<(), kernite_cuda::Error> {
    let grid = kernite_cuda::Grid::new(1, 1);
    unsafe {
        kernels.scale(grid, p, 2.0)?;
        kernels.late(grid, p)?;
        kernels.bad(grid, p)?;
    }
    Ok(())
}

pub fn by_path() -> u64 {
    unsafe { crate::cpu::good(1) + crate::cpu::bad(1) + crate::cpu::late(1) }
}

pub mod by_use {
    use crate::cpu::{bad, good, late};

    pub fn f() -> u64 {
        unsafe { good(1) + bad(1) + late(1) }
    }
}

/// `good` and `late` are found, so the import is used; `bad` is not.
pub mod by_glob {
    use crate::cpu::*;

    pub fn f() -> u64 {
        unsafe { good(1) + bad(1) + late(1) }
    }
}

/// Nothing it names is found, so the import is unused as well.
pub mod by_glob_unfound {
    use crate::cpu::*;

    pub fn f() -> u64 {
        unsafe { bad(1) }
    }
}
