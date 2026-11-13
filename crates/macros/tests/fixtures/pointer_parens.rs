//! The parentheses of the type a pointer points to are Rust's to lint in a kernel's parameter, as in any function's:
//! in an x86 kernel's, a GPU kernel's and a plain function's. Where a pointer needs them, `*const (dyn Debug + Send)`,
//! Rust says nothing; where it does not, `*const (dyn Debug)`, `*mut ([u8])` and `*mut (u32)`, it says that they are
//! unnecessary, once for each. kernite refuses a pointer to a trait object or to a slice, which has two words, and says
//! nothing else. A fragment, `$t` in `*const (dyn Fn() -> $t + Send)` with `$t` as `dyn Debug + Sync`, is one type to
//! Rust. Two bounds with no `dyn`, `*const (Debug + Send)`, are Rust's error in edition 2024, and no warning.

#[kernite_macros::kernels]
pub mod needed_gpu {
    pub unsafe extern "ptx-kernel" fn a(x: *const (dyn core::fmt::Debug + Send)) {
        ret();
    }
}

#[kernite_macros::kernels]
pub mod needed_x86 {
    pub unsafe extern "sysv64" fn a(x: *const (dyn core::fmt::Debug + Send)) {
        ret();
    }
}

pub mod needed_plain {
    pub fn a(_x: *const (dyn core::fmt::Debug + Send)) {}
}

#[kernite_macros::kernels]
pub mod one_bound_gpu {
    pub unsafe extern "ptx-kernel" fn a(x: *const (dyn core::fmt::Debug)) {
        ret();
    }
}

#[kernite_macros::kernels]
pub mod one_bound_x86 {
    pub unsafe extern "sysv64" fn a(x: *const (dyn core::fmt::Debug)) {
        ret();
    }
}

pub mod one_bound_plain {
    pub fn a(_x: *const (dyn core::fmt::Debug)) {}
}

#[kernite_macros::kernels]
pub mod slice_gpu {
    pub unsafe extern "ptx-kernel" fn a(x: *mut ([u8])) {
        ret();
    }
}

#[kernite_macros::kernels]
pub mod slice_x86 {
    pub unsafe extern "sysv64" fn a(x: *mut ([u8])) {
        ret();
    }
}

pub mod slice_plain {
    pub fn a(_x: *mut ([u8])) {}
}

#[kernite_macros::kernels]
pub mod number_gpu {
    pub unsafe extern "ptx-kernel" fn a(x: *mut (u32)) {
        st.global.u32([x], 1);
    }
}

#[kernite_macros::kernels]
pub mod number_x86 {
    pub unsafe extern "sysv64" fn a(x: *mut (u32)) {
        ret();
    }
}

pub mod number_plain {
    pub fn a(_x: *mut (u32)) {}
}

macro_rules! returns_bound {
    ($gpu:ident, $x86:ident, $plain:ident, $t:ty) => {
        #[kernite_macros::kernels]
        pub mod $gpu {
            pub unsafe extern "ptx-kernel" fn a(x: *const (dyn Fn() -> $t + Send)) {
                ret();
            }
        }

        #[kernite_macros::kernels]
        pub mod $x86 {
            pub unsafe extern "sysv64" fn a(x: *const (dyn Fn() -> $t + Send)) {
                ret();
            }
        }

        pub mod $plain {
            pub fn a(_x: *const (dyn Fn() -> $t + Send)) {}
        }
    };
}

returns_bound!(returns_gpu, returns_x86, returns_plain, dyn core::fmt::Debug + Sync);

#[kernite_macros::kernels]
pub mod bare_gpu {
    pub unsafe extern "ptx-kernel" fn a(x: *const (core::fmt::Debug + Send)) {
        ret();
    }
}

#[kernite_macros::kernels]
pub mod bare_x86 {
    pub unsafe extern "sysv64" fn a(x: *const (core::fmt::Debug + Send)) {
        ret();
    }
}

pub mod bare_plain {
    pub fn a(_x: *const (core::fmt::Debug + Send)) {}
}
