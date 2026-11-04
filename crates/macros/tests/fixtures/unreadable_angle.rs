//! A module kernite cannot read, for the heads of three helpers it does not take, `by_result<T>`, `by_bound<T>` and
//! `by_where<T>`, each of which holds a brace group in `<..>`, a const argument, before its body. Beside them is a
//! kernel it takes, `good`, that a glob names.

pub struct Foo<const N: usize>;

pub trait Tr<const N: usize> {}

#[kernite_macros::kernels]
pub mod cpu {
    pub fn by_result<T>() -> super::Foo<{ 3 }> {
        nop();
    }

    pub fn by_bound<T: super::Tr<{ 3 }>>() {
        nop();
    }

    pub fn by_where<T>()
    where
        super::Foo<{ 3 }>: Sized,
    {
        nop();
    }

    pub unsafe extern "sysv64" fn good(x: u64) -> u64 {
        ret(x);
    }
}

pub mod by_glob {
    use crate::cpu::*;

    pub fn f() -> u64 {
        unsafe { good(1) }
    }
}
