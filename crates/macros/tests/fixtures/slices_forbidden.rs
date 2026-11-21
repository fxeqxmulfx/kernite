//! A crate that forbids `improper_ctypes_definitions` and has a safe kernel that takes a slice, which the function
//! the macro writes for it allows: Rust refuses the crate.

#![forbid(improper_ctypes_definitions)]

#[kernite_macros::kernels]
pub mod cpu {
    /// The first integer of `xs`, or 0.
    pub extern "sysv64" fn first(xs: &[u64]) -> u64 {
        let x: gpr;
        zero(x);
        cmp(xs.len, 1);
        jb(done);
        mov(x, [xs]);
        label(done);
        ret(x);
    }
}
