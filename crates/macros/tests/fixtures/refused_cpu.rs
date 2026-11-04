//! A module of x86 kernels that `#[kernels]` refuses at its attribute, in a crate without `kernite-cpu`: the
//! attribute is told once, and so is the dependency that its static and its cooperative kernel need, as for a
//! module kernite does not refuse. Rust says nothing of the code that names them.

#[kernite_macros::kernels(consts = 5)]
pub mod cpu {
    pub static COUNTS: global<[u32; 4]>;

    #[cooperative]
    pub unsafe extern "sysv64" fn together(x: *mut u32) {
        grid.sync();
        mov(dword[x], 1);
    }

    pub unsafe extern "sysv64" fn step(x: u64) -> u64 {
        ret(x);
    }
}

/// From the crate's root, by the module's path.
pub fn from_the_root() -> usize {
    unsafe { cpu::step(1) };
    cpu::COUNTS.len()
}
