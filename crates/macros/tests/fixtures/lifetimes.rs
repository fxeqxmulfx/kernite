//! Kernels that take a pointer to a type with a lifetime. The macro is given the lifetime as
//! punctuation, where the lexer of a string refuses it; and Rust passes a pointer to a slice of
//! references, or to a `dyn` that lives as long as `'static`, in two registers, where a kernel's
//! pointer is one. Each is refused at its type, in the lexer's words.

#[kernite_macros::kernels]
pub mod cpu {
    pub unsafe extern "sysv64" fn slice(p: *const [&'static str], n: usize) {
        mov(qword[p], n);
    }

    pub unsafe extern "sysv64" fn object(p: *const (dyn core::fmt::Debug + 'static), n: usize) {
        mov(qword[p], n);
    }
}
