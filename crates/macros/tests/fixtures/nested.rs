//! Parentheses nested 64 deep, as the macro is given them: the 64th is refused where it opens,
//! as it is in a string of the same tokens.

#[kernite_macros::kernels]
pub mod brackets {
    pub unsafe extern "sysv64" fn f() {
        let y = ((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((1))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))));
    }
}
