//! A module of x86 kernels, one of GPU kernels and one kernite cannot read, which `#[kernels]` refuses at its
//! attribute, `consts = 5`: the error is said once, there, and Rust says nothing of the code that names their
//! items, wherever it is. The two kernite can read are still there, with a stub for each kernel, as when it does not
//! compile, and each static as its type has it; the other holds its refusal alone.

#[kernite_macros::kernels(consts = 5)]
pub mod cpu {
    #![allow(dead_code)]

    pub const STEP: u64 = 3;

    pub struct Pair(pub u64, pub u64);

    pub static COUNTS: global<[u32; 4]>;

    /// A helper is none of Rust's.
    fn twice(x: u64) -> u64 {
        add(x, x);
        ret(x);
    }

    pub unsafe extern "sysv64" fn step(x: u64) -> u64 {
        add(x, STEP);
        ret(x);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tile<const N: u64>(x: u64) -> u64 {
        add(x, N);
        ret(x);
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn together(x: *mut u32) {
        grid.sync();
        mov(dword[x], 1);
    }

    #[cfg(any())]
    pub unsafe extern "sysv64" fn gone(x: u64) -> u64 {
        ret(x);
    }

    pub unsafe extern "C" fn foreign(x: u64) -> u64 {
        ret(x);
    }
}

#[kernite_macros::kernels(consts = 5)]
pub mod gpu {
    pub static COUNTS: global<[u32; 4]>;

    pub unsafe extern "ptx-kernel" fn scale(x: *mut f32, a: f32) {
        let v: f32;
        ld.global.f32(v, [x]);
        mul.rn.f32(v, v, a);
        st.global.f32([x], v);
    }

    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn tile<const N: u32>(x: *mut f32) {
        let at: u64;
        mov.u64(at, COUNTS);
        red.global.add.u32([at], N);
    }

    #[cooperative]
    pub unsafe extern "ptx-kernel" fn together(x: *mut f32) {
        grid.sync();
    }

    pub extern "ptx-func" fn callee(x: u32) -> u32 {
        ret(x);
    }
}

/// Two functions share a name, so kernite cannot read it: only its attribute's error is said.
#[kernite_macros::kernels(consts = 5)]
pub mod unreadable {
    pub unsafe extern "sysv64" fn same(x: u64) -> u64 {
        ret(x);
    }

    pub unsafe extern "sysv64" fn same(x: u64) -> u64 {
        ret(x);
    }
}

/// From the crate's root, by the modules' paths.
pub fn from_the_root(
    x: *mut u32,
    kernels: &gpu::Kernels,
    p: kernite_cuda::Ptr<f32>,
) -> Result<u64, Box<dyn std::error::Error>> {
    let grid = kernite_cuda::Grid::new(1, 1);
    unsafe {
        cpu::together(kernite_cpu::Grid::new(1), x)?;
        kernels.scale(grid, p, 2.0)?;
        Ok(cpu::step(cpu::STEP) + cpu::tile::<2>(1) + cpu::foreign(1) + cpu::Pair(1, 2).0 + unreadable::same(1))
    }
}

/// From another module, by path and by `use`.
pub mod other {
    use crate::cpu::step;
    use crate::gpu::Kernels;
    use crate::unreadable::same;

    pub fn by_path(x: *mut u32) -> usize {
        unsafe { crate::cpu::together(kernite_cpu::Grid::new(1), x).unwrap() };
        crate::cpu::COUNTS.len() + crate::cpu::LISTINGS.len() + unsafe { crate::cpu::step(crate::cpu::STEP) } as usize
    }

    pub fn by_use(kernels: &Kernels) -> u32 {
        kernels.together_blocks(32, 0).unwrap() + unsafe { step(1) + same(1) } as u32
    }
}

/// By the names a glob brings in.
pub mod cpu_globbed {
    use crate::cpu::*;

    pub fn all(x: *mut u32) -> u64 {
        unsafe { together(kernite_cpu::Grid::new(1), x).unwrap() };
        COUNTS.zero();
        unsafe { step(STEP) + tile::<2>(1) + foreign(1) + Pair(1, 2).0 }
    }
}

/// By the names a glob brings in.
pub mod gpu_globbed {
    use crate::gpu::*;

    pub fn all(kernels: &Kernels, p: kernite_cuda::Ptr<f32>) -> Result<u32, kernite_cuda::Error> {
        let grid = kernite_cuda::Grid::new(1, 1);
        unsafe {
            kernels.scale(grid, p, 2.0)?;
            kernels.tile::<2>(grid, p)?;
            kernels.together(grid, p)?;
        }
        kernels.COUNTS().write(&[0; 4])?;
        let _ = (PTX, Kernels::load);
        kernels.together_blocks(32, 0)
    }
}

/// A function of `ptx-func` is no kernel, so `Kernels` has no method of its name, and the crate's own may have one.
impl gpu::Kernels {
    pub fn callee(&self) -> u32 {
        7
    }
}
