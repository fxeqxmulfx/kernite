//! Modules kernite cannot read, for two items of a name: each says so once, and Rust says nothing of the code that
//! names the rest.

#[kernite_macros::kernels]
pub mod cpu {
    pub unsafe extern "sysv64" fn same(x: u64) -> u64 {
        ret(x);
    }

    pub unsafe extern "sysv64" fn same(x: u64) -> u64 {
        ret(x);
    }

    pub const STEP: u64 = 3;

    pub struct Pair(pub u64, pub u64);

    pub static COUNTS: global<[u32; 4]>;

    pub unsafe extern "sysv64" fn step(x: u64) -> u64 {
        add(x, STEP);
        ret(x);
    }
}

#[kernite_macros::kernels]
pub mod gpu {
    pub static COUNTS: global<[u32; 4]>;

    pub unsafe extern "ptx-kernel" fn scale(x: *mut f32, a: f32) {
        let v: f32;
        ld.global.f32(v, [x]);
        mul.rn.f32(v, v, a);
        st.global.f32([x], v);
    }

    pub unsafe extern "ptx-kernel" fn scale(x: *mut f32, a: f32) {
        let v: f32;
        ld.global.f32(v, [x]);
        st.global.f32([x], v);
    }

    pub unsafe extern "ptx-kernel" fn halve(x: *mut f32) {
        let v: f32;
        ld.global.f32(v, [x]);
        mul.rn.f32(v, v, 0.5);
        st.global.f32([x], v);
    }
}

/// From the crate's root, by the modules' paths.
pub fn from_the_root(kernels: &gpu::Kernels, p: kernite_cuda::Ptr<f32>) -> Result<u64, kernite_cuda::Error> {
    let grid = kernite_cuda::Grid::new(1, 1);
    unsafe {
        kernels.scale(grid, p, 2.0)?;
        kernels.halve(grid, p)?;
        Ok(cpu::same(cpu::STEP) + cpu::step(1) + cpu::Pair(1, 2).0 + cpu::COUNTS.len() as u64)
    }
}

/// From another module, by `use`.
pub mod by_use {
    use crate::cpu::{COUNTS, Pair, STEP, same, step};
    use crate::gpu::Kernels;

    pub fn f(kernels: &Kernels, p: kernite_cuda::Ptr<f32>) -> Result<u64, kernite_cuda::Error> {
        unsafe { kernels.halve(kernite_cuda::Grid::new(1, 1), p)? };
        Ok(unsafe { same(STEP) + step(1) + Pair(1, 2).0 + COUNTS.len() as u64 })
    }
}

/// By the names a glob brings in.
pub mod by_glob {
    use crate::cpu::*;

    pub fn f() -> u64 {
        unsafe { same(STEP) + step(1) + Pair(1, 2).0 + COUNTS.len() as u64 }
    }
}

/// By the names a glob brings in.
pub mod gpu_globbed {
    use crate::gpu::*;

    pub fn f(kernels: &Kernels, p: kernite_cuda::Ptr<f32>) -> Result<(), kernite_cuda::Error> {
        let grid = kernite_cuda::Grid::new(1, 1);
        unsafe {
            kernels.scale(grid, p, 2.0)?;
            kernels.halve(grid, p)?;
        }
        kernels.COUNTS().write(&[0; 4])?;
        let _ = (PTX, Kernels::load);
        Ok(())
    }
}
