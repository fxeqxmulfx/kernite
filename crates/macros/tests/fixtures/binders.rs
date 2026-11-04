//! Kernels whose types begin with a binder, `for<'a> fn(&'a u8)`, which Rust writes before a function's
//! pointer or a trait object to say that it holds for every lifetime. The binder is part of the type, and
//! kernite reads it whole: a kernel that takes or returns one is refused at the type, in its machine's words,
//! which quote the type as Rust writes it, with a space after the binder. A pointer to one is refused in the
//! lexer's words, as a pointer to any type with a lifetime is. A pointer to a function's type with no binder,
//! `*const fn(&u8)`, is taken: no error is told of it.

#[kernite_macros::kernels]
pub mod cpu {
    pub unsafe extern "sysv64" fn by_value(x: for<'a> fn(&'a u8)) {
        ret();
    }

    pub unsafe extern "sysv64" fn in_a_type(x: Option<for<'a> fn(&'a u8)>) {
        ret();
    }

    pub unsafe extern "sysv64" fn in_a_trait_object(x: Option<&dyn for<'a> Fn(&'a u8)>) {
        ret();
    }

    pub unsafe extern "sysv64" fn returned(x: u64) -> for<'a> fn(&'a u8) {
        ret();
    }

    pub unsafe extern "sysv64" fn pointer_to(x: *const for<'a> fn(&'a u8)) {
        ret();
    }

    pub unsafe extern "sysv64" fn pointer_to_a_trait_object(x: *const dyn for<'a> Fn(&'a u8)) {
        ret();
    }

    pub unsafe extern "sysv64" fn pointer_to_a_plain_one(x: *const fn(&u8)) {
        ret();
    }
}

#[kernite_macros::kernels]
pub mod gpu {
    pub unsafe extern "ptx-kernel" fn by_value(x: for<'a> fn(&'a u8)) {
        ret();
    }

    pub unsafe extern "ptx-kernel" fn pointer_to(x: *const for<'a> fn(&'a u8)) {
        ret();
    }
}
