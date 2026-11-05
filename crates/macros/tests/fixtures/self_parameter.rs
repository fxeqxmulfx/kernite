//! A kernel's parameter named `self`, written plain, on an x86 kernel. A name written plain is no name Rust refuses at
//! its lexer, and the macros hand it on as it is: Rust says what it says of any function that is no method,
//! "`self` parameter is only allowed in associated functions", at the name.

#[kernite_macros::kernels]
pub mod cpu {
    pub unsafe extern "sysv64" fn plain(self: u64) -> u64 {
        ret(self);
    }
}
