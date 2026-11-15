//! A body that the call of a macro gives, a `$b:block`, after a pointer to a reference that names a lifetime,
//! `*const &'static u8`, is the kernel's body. The kernel is refused at its type, as the same kernel written out is, in
//! the lexer's words: a lifetime has no place in a kernel.

macro_rules! bodies {
    ($module:ident, $b:block) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "sysv64" fn only(x: *const &'static u8) -> *const &'static u8 $b
        }
    };
}

bodies!(bodied, { ret(x); });

#[kernite_macros::kernels]
pub mod written {
    pub unsafe extern "sysv64" fn only(x: *const &'static u8) -> *const &'static u8 {
        ret(x);
    }
}
