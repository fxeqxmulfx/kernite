//! Modules of kernels, x86 and GPU, whose consts' path Rust cannot find, `crate::nope`: Rust says so once, at the
//! path, and nothing of the code that names the modules' items, wherever it is. The modules are there, empty, until
//! the path is mended.

#[kernite_macros::kernels(consts = crate::nope)]
pub mod cpu {
    pub static COUNTS: global<[u32; SIZE]>;

    pub unsafe extern "sysv64" fn step(x: u64) -> u64 {
        add(x, STEP);
        ret(x);
    }
}

#[kernite_macros::kernels(consts = crate::nope)]
pub mod gpu {
    pub unsafe extern "ptx-kernel" fn scale(x: *mut f32, a: f32) {
        let v: f32;
        ld.global.f32(v, [x]);
        mul.rn.f32(v, v, a);
        st.global.f32([x], v);
    }
}

/// From the crate's root.
pub fn from_the_root(kernels: &gpu::Kernels, p: kernite_cuda::Ptr<f32>) -> Result<u64, kernite_cuda::Error> {
    unsafe {
        kernels.scale(kernite_cuda::Grid::new(1, 1), p, 2.0)?;
        Ok(cpu::step(1) + cpu::COUNTS.read()[0] as u64)
    }
}

/// From another module, by path.
pub mod other {
    pub fn by_path(kernels: &crate::gpu::Kernels, p: kernite_cuda::Ptr<f32>) -> Result<u64, kernite_cuda::Error> {
        unsafe {
            kernels.scale(kernite_cuda::Grid::new(1, 1), p, 2.0)?;
            Ok(crate::cpu::step(1) + crate::cpu::COUNTS.read()[0] as u64)
        }
    }
}

/// By the names a glob brings in. Rust warns of each glob that it brings in none.
pub mod cpu_globbed {
    use crate::cpu::*;

    pub fn all() -> u64 {
        unsafe { step(1) + COUNTS.read()[0] as u64 }
    }
}

pub mod gpu_globbed {
    use crate::gpu::*;

    pub fn all(kernels: &Kernels, p: kernite_cuda::Ptr<f32>) -> Result<(), kernite_cuda::Error> {
        unsafe { kernels.scale(kernite_cuda::Grid::new(1, 1), p, 2.0) }
    }
}

/// A method of the crate's own on a type of the module.
impl gpu::Kernels {
    pub fn callee(&self) -> u32 {
        7
    }
}
