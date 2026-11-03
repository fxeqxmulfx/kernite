//! A crate that depends on `kernite-cuda` and `kernite-cpu` under other names, `cuda` and `cpu`, in its
//! manifest, with a module of GPU kernels, and a static and a cooperative kernel in a module of x86
//! kernels: each is said once, where it is, as for a crate that has neither, and nothing else is said
//! of them, nor of the code that uses them as Rust does.

#[kernite_macros::kernels]
pub mod gpu {
    pub unsafe extern "ptx-kernel" fn nothing() {
        ret();
    }
}

#[kernite_macros::kernels]
pub mod x86 {
    pub static COUNTS: global<[u32; 4]>;

    #[cooperative]
    pub unsafe extern "sysv64" fn together(x: *mut u32) {
        grid.sync();
        mov(dword[x], 1);
    }
}

/// Each kernel, launched or called as Rust does, of the handles a GPU gives, whatever they are.
pub fn on_both<C, G: Copy>(cuda: &C, grid: G, x: *mut u32) -> Result<(), Box<dyn std::error::Error>> {
    let _ = x86::COUNTS.read();
    unsafe {
        gpu::Kernels::load(cuda)?.nothing(grid)?;
        x86::together((), x)?;
    }
    Ok(())
}
