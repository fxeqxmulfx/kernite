//! A crate that does not depend on `kernite-cuda`, with two modules of GPU kernels, the first of a
//! kernel of a pointer and an `f16`, one of a compile-time argument, a cooperative one and a static:
//! each module is said once, where it is, and nothing else is said of them, nor of the x86 kernels
//! that need no crate, nor of the code that loads a module and launches its kernels as Rust does.

#[kernite_macros::kernels]
pub mod gpu {
    pub static COUNTS: global<[u32; 4]>;

    pub unsafe extern "ptx-kernel" fn scale(x: *mut f16, a: f16) {
        let v: f16;
        ld.global.b16(v, [x]);
        mul.rn.f16(v, v, a);
        st.global.b16([x], v);
    }

    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn tile<const N: u32>(x: *mut u32) {
        let at: u64;
        mov.u64(at, COUNTS);
        red.global.add.u32([at], N);
    }

    #[cooperative]
    pub unsafe extern "ptx-kernel" fn together(x: *mut u32) {
        grid.sync();
    }
}

#[kernite_macros::kernels]
pub mod more {
    pub unsafe extern "ptx-kernel" fn nothing() {
        ret();
    }
}

#[kernite_macros::kernels]
pub mod cpu {
    pub unsafe extern "sysv64" fn step(x: u64) -> u64 {
        add(x, 3);
        ret(x);
    }
}

/// Each GPU kernel, launched as Rust launches it, of the handles a GPU gives, whatever they are.
pub fn on_the_gpu<C, G: Copy, P: Copy, H>(cuda: &C, grid: G, x: P, a: H) -> Result<u32, Box<dyn std::error::Error>> {
    let kernels = gpu::Kernels::load(cuda)?;
    unsafe {
        kernels.scale(grid, x, a)?;
        kernels.tile::<2>(grid, x)?;
        kernels.together(grid, x)?;
        more::Kernels::load(cuda)?.nothing(grid)?;
    }
    kernels.COUNTS().write(&[0; 4])?;
    let _ = (gpu::PTX, more::PTX);
    Ok(kernels.together_blocks(32, 0)? + unsafe { cpu::step(1) } as u32)
}
