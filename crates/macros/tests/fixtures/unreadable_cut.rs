//! Modules kernite cannot read, for a const and a static whose value holds a brace group with more after it: a block
//! added to, `{ 1 } + 2`, a struct literal read for a field, `Pair { a: 1 }.a`, and a branch, `if .. { 1 } else { 2 }`.
//! Each ends at its `;`, so that Rust says nothing of what would be left of it as an item of its own, nor of the code
//! that names the kernel beside it.

#[kernite_macros::kernels]
pub mod block {
    pub const A: u64 = { 1 } + 2;

    pub unsafe extern "sysv64" fn good(x: u64) -> u64 {
        ret(x);
    }
}

#[kernite_macros::kernels]
pub mod literal {
    pub static S: global<[u32; 4]> = Pair { a: 1 }.a;

    pub unsafe extern "sysv64" fn good(x: u64) -> u64 {
        ret(x);
    }
}

#[kernite_macros::kernels]
pub mod branch {
    pub const B: u64 = if true { 1 } else { 2 };

    pub unsafe extern "sysv64" fn good(x: u64) -> u64 {
        ret(x);
    }
}

pub mod by_glob {
    use crate::block::*;
    use crate::branch::good as third;
    use crate::literal::good as other;

    pub fn f() -> u64 {
        unsafe { good(1) + other(1) + third(1) }
    }
}
