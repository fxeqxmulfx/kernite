//! A crate that does not depend on `kernite-cpu`, with a static in a module of x86 kernels and three
//! cooperative kernels, one of a compile-time argument and one with a result: each is said where it
//! is, and nothing else is said of them, but Rust's word that the result's type has no size, nor of
//! the kernels that need no crate, nor of the code that calls them as Rust calls them.

#[kernite_macros::kernels]
pub mod cpu {
    pub static COUNTS: global<[u32; 16]>;

    pub unsafe extern "sysv64" fn step(x: u64) -> u64 {
        add(x, 3);
        ret(x);
    }

    pub unsafe extern "sysv64" fn count() {
        let at: gpr;
        lea(at, [COUNTS]);
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn together(x: *mut u32) {
        grid.sync();
        mov(dword[x], 1);
    }

    #[instances(1, 2)]
    #[cooperative]
    pub unsafe extern "sysv64" fn tile<const N: u64>(x: *mut u32) {
        grid.sync();
        mov(dword[x], N);
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn counts(x: *mut u32) -> *const core::ffi::CStr {
        ret(x);
    }
}

#[kernite_macros::kernels]
pub mod plain {
    pub unsafe extern "sysv64" fn step(x: u64) -> u64 {
        add(x, 3);
        ret(x);
    }
}

/// Each x86 kernel, called as Rust calls it.
pub fn on_the_cpu(x: *mut u32) -> Result<u64, Box<dyn std::error::Error>> {
    let _ = cpu::COUNTS.read();
    unsafe {
        cpu::together((), x)?;
        cpu::tile::<2>((), x).unwrap();
        let _ = cpu::counts((), x)?;
        cpu::count();
        Ok(cpu::step(1) + plain::step(1))
    }
}
