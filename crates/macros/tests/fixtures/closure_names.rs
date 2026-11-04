//! A cooperative x86 kernel that takes numbers, in a library whose symbols are read: the closure that
//! the kernel's launcher runs is the first the launcher has, `closure_names::cpu::together::{closure#0}`.

#[kernite_macros::kernels]
pub mod cpu {
    #[cooperative]
    pub unsafe extern "sysv64" fn together(a: u64, b: f64) {
        ret();
    }
}
