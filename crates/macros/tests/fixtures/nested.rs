//! Parentheses nested 64 deep, as the macro is given them: the 64th is refused where it opens, as it is in a string of
//! the same tokens. kernite reads none of the module, so Rust cannot find `f` by a glob of it, but says nothing by path
//! or `use`.

#[kernite_macros::kernels]
pub mod brackets {
    pub unsafe extern "sysv64" fn f() {
        let y = ((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((1))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))));
    }
}

pub fn by_path() {
    unsafe { crate::brackets::f() }
}

pub mod by_use {
    use crate::brackets::f;

    pub fn g() {
        unsafe { f() }
    }
}

pub mod by_glob {
    use crate::brackets::*;

    pub fn g() {
        unsafe { f() }
    }
}
