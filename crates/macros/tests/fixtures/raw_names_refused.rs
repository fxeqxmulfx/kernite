//! Each name Rust lets be written raw in no place, written raw in a module of kernels: as a variable, where the
//! macros hand the kernel's body to kernite alone, and in a type, which Rust reads. Rust refuses each at each place
//! it is written, in its own words, and says of a type what it says in plain Rust. kernite says nothing.

#[kernite_macros::kernels]
pub mod variables {
    pub unsafe extern "sysv64" fn lower_self(x: u64) -> u64 {
        let r#self: gpr;
        mov(r#self, x);
        ret(r#self);
    }

    pub unsafe extern "sysv64" fn upper_self(x: u64) -> u64 {
        let r#Self: gpr;
        mov(r#Self, x);
        ret(r#Self);
    }

    pub unsafe extern "sysv64" fn parent(x: u64) -> u64 {
        let r#super: gpr;
        mov(r#super, x);
        ret(r#super);
    }

    pub unsafe extern "sysv64" fn root(x: u64) -> u64 {
        let r#crate: gpr;
        mov(r#crate, x);
        ret(r#crate);
    }

    pub unsafe extern "sysv64" fn underscore(x: u64) -> u64 {
        let r#_: gpr;
        mov(r#_, x);
        ret(r#_);
    }
}

#[kernite_macros::kernels]
pub mod types {
    pub unsafe extern "sysv64" fn parent(p: *const r#super::Foo) -> *const r#super::Foo {
        ret(p);
    }

    pub unsafe extern "sysv64" fn upper_self(p: *const r#Self) -> *const r#Self {
        ret(p);
    }
}
