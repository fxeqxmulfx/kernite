//! Kernels that take or give back a pointer to a type with a part that has no size: a tuple whose first part has none,
//! `(dyn Debug, u8)`, and a slice of parts that have none, `[dyn Debug]`. Rust refuses the type by itself and says that
//! a size cannot be known, once, at the type from its first token to its last, as it says it of a plain function's, in
//! a kernel of every kind: plain, of compile-time arguments, cooperative, and cooperative of compile-time arguments, on
//! x86 and on a GPU, and in one that returns the pointer. kernite says too of a slice's pointer that it has two words,
//! where it is.

pub fn plain_tuple(_x: *const (dyn core::fmt::Debug, u8)) {}

pub fn plain_slice(_x: *const [dyn core::fmt::Debug]) {}

#[kernite_macros::kernels]
pub mod cpu {
    pub unsafe extern "sysv64" fn tuple(x: *const (dyn core::fmt::Debug, u8)) {
        ret();
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tuple_tiles<const N: u64>(x: *const (dyn core::fmt::Debug, u8)) {
        mov(qword[x], N);
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn tuple_together(x: *const (dyn core::fmt::Debug, u8)) {
        ret();
    }

    #[cooperative]
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tuple_together_tiles<const N: u64>(x: *const (dyn core::fmt::Debug, u8)) {
        mov(qword[x], N);
    }

    pub unsafe extern "sysv64" fn tuple_result(x: u64) -> *const (dyn core::fmt::Debug, u8) {
        ret(x);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tuple_result_tiles<const N: u64>(x: u64) -> *const (dyn core::fmt::Debug, u8) {
        ret(x);
    }

    pub unsafe extern "sysv64" fn slice(x: *const [dyn core::fmt::Debug]) {
        ret();
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn slice_tiles<const N: u64>(x: *const [dyn core::fmt::Debug]) {
        mov(qword[x], N);
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn slice_together(x: *const [dyn core::fmt::Debug]) {
        ret();
    }

    #[cooperative]
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn slice_together_tiles<const N: u64>(x: *const [dyn core::fmt::Debug]) {
        mov(qword[x], N);
    }

    pub unsafe extern "sysv64" fn slice_result(x: u64) -> *const [dyn core::fmt::Debug] {
        ret(x);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn slice_result_tiles<const N: u64>(x: u64) -> *const [dyn core::fmt::Debug] {
        ret(x);
    }
}

#[kernite_macros::kernels]
pub mod gpu {
    pub unsafe extern "ptx-kernel" fn tuple(x: *const (dyn core::fmt::Debug, u8)) {
        ret();
    }

    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn tuple_tiles<const N: u32>(x: *const (dyn core::fmt::Debug, u8)) {
        st.global.u32([x], N);
    }

    #[cooperative]
    pub unsafe extern "ptx-kernel" fn tuple_together(x: *const (dyn core::fmt::Debug, u8)) {
        ret();
    }

    #[cooperative]
    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn tuple_together_tiles<const N: u32>(x: *const (dyn core::fmt::Debug, u8)) {
        st.global.u32([x], N);
    }

    pub unsafe extern "ptx-kernel" fn slice(x: *const [dyn core::fmt::Debug]) {
        ret();
    }

    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn slice_tiles<const N: u32>(x: *const [dyn core::fmt::Debug]) {
        st.global.u32([x], N);
    }

    #[cooperative]
    pub unsafe extern "ptx-kernel" fn slice_together(x: *const [dyn core::fmt::Debug]) {
        ret();
    }

    #[cooperative]
    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn slice_together_tiles<const N: u32>(x: *const [dyn core::fmt::Debug]) {
        st.global.u32([x], N);
    }
}
