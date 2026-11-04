//! A crate that depends on `kernite-macros` under another name, `km`, in its manifest, with a module of x86
//! kernels and one of GPU kernels given consts: each is told once, at its attribute, that the crate has no
//! dependency named `kernite-macros`, and Rust says nothing of the code that names their items, wherever it is,
//! for each module is still there, each kernel as one that does not compile is.

#[km::consts]
pub mod sizes {
    pub const N: u32 = 4;
}

#[km::kernels(consts = crate::sizes)]
pub mod cpu {
    pub unsafe extern "sysv64" fn k(x: *mut u32) {
        mov(dword[x], N);
        ret();
    }
}

#[km::kernels(consts = crate::sizes)]
pub mod gpu {
    /// No room of it is counted, so the module is not told again that it needs `kernite-macros`.
    pub static TABLE: constant<[u32; 4]>;

    pub unsafe extern "ptx-kernel" fn g(x: *mut u32) {
        st.global.u32([x], N);
    }
}

/// From the crate's root, by the modules' paths.
pub fn from_the_root(
    x: *mut u32,
    kernels: &gpu::Kernels,
    p: kernite_cuda::Ptr<u32>,
) -> Result<(), kernite_cuda::Error> {
    unsafe { cpu::k(x) };
    unsafe { kernels.g(kernite_cuda::Grid::new(1, 1), p) }
}

/// From another module, by path and by `use`.
pub mod other {
    use crate::cpu::k;

    pub fn by_path_and_use(x: *mut u32) {
        unsafe { crate::cpu::k(x) };
        unsafe { k(x) };
    }
}

/// By the names a glob brings in.
pub mod globbed {
    use crate::cpu::*;
    use crate::gpu::*;

    pub fn all(x: *mut u32, kernels: &Kernels, p: kernite_cuda::Ptr<u32>) -> Result<(), kernite_cuda::Error> {
        unsafe { k(x) };
        let _ = PTX;
        unsafe { kernels.g(kernite_cuda::Grid::new(1, 1), p) }
    }
}
