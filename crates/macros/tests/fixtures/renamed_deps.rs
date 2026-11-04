//! A crate that depends on `kernite-macros`, `kernite-cuda` and `kernite-cpu` under other names, `km`,
//! `cuda` and `cpu`, in its manifest, with a module of GPU kernels and a static in it, and a static and
//! a cooperative kernel in a module of x86 kernels: each is said once, where it is, as for a crate that
//! has none of them, and nothing else is said of them, nor of the code that uses them as Rust does. A
//! module whose statics `cfg` leaves out says nothing, and one with a static there only with the feature
//! `more` says what it says with it.

#[km::kernels]
pub mod gpu {
    pub static TABLE: constant<[u32; 4]>;

    pub unsafe extern "ptx-kernel" fn nothing() {
        ret();
    }
}

#[km::kernels]
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

/// Its static is never there, so the module has nothing the crate needs `kernite-macros` for.
#[km::kernels]
pub mod gone {
    #[cfg(any())]
    pub static TABLE: constant<[u32; 4]>;
}

/// One of its statics is there with the feature, and the other never: the module needs what it needs
/// with the feature alone.
#[km::kernels]
pub mod some {
    #[cfg(feature = "more")]
    pub static TABLE: constant<[u32; 4]>;

    #[cfg(any())]
    pub static OLD: constant<[u32; 4]>;
}
