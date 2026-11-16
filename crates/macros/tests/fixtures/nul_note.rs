//! A NUL in a note is refused at the string, in a module of GPU kernels and in one of x86 kernels, as `ptx_str` and
//! `x86_str` refuse it: written by an escape, or as the character itself, in a plain string and in a raw one. The
//! fixture has the character as `<NUL>`, which the test writes as a NUL before Cargo builds it, for git takes a file
//! with a NUL in it for a binary.

#[kernite_macros::kernels]
pub mod gpu {
    pub unsafe extern "ptx-kernel" fn escape() {
        note("a\0b");
    }

    pub unsafe extern "ptx-kernel" fn hex() {
        note("a\x00b");
    }

    pub unsafe extern "ptx-kernel" fn unicode() {
        note("a\u{0}b");
    }

    pub unsafe extern "ptx-kernel" fn character() {
        note("a<NUL>b");
    }

    pub unsafe extern "ptx-kernel" fn raw() {
        note(r"a<NUL>b");
    }

    /// Another control character is no trouble: this one compiles.
    pub unsafe extern "ptx-kernel" fn other() {
        note("a\u{1}b");
    }
}

#[kernite_macros::kernels]
pub mod cpu {
    pub unsafe extern "sysv64" fn escape() {
        note("a\0b");
    }

    pub unsafe extern "sysv64" fn hex() {
        note("a\x00b");
    }

    pub unsafe extern "sysv64" fn unicode() {
        note("a\u{0}b");
    }

    pub unsafe extern "sysv64" fn character() {
        note("a<NUL>b");
    }

    pub unsafe extern "sysv64" fn raw() {
        note(r"a<NUL>b");
    }

    /// Another control character is no trouble: this one compiles.
    pub unsafe extern "sysv64" fn other() {
        note("a\u{1}b");
    }
}
