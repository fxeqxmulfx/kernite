//! Kernels that take a pointer to a type with no size. Rust passes such a pointer in two registers, an
//! address and a length or a table, where a kernel's is one. kernite tells `[f32]` by its text and
//! says so; only Rust knows that `CStr`, `str` by its path, `Floats`, `Tail` and a tuple that ends in
//! `CStr` have no size, and says it at the type, in a kernel of any kind, x86 or GPU. A kernel kernite
//! refuses for another reason has it in its stub. Rust refuses `[CStr; 3]`, `(CStr, u8)` and
//! `(u8, [CStr; 3])` by itself, at the type, and says no more of them, in a kernel of any kind but
//! one: a kernel of compile-time arguments casts back a pointer it returns, and for `(u8, [CStr; 3])`
//! Rust says the array in the signature, and the tuple at the cast.

/// A name for a slice.
pub type Floats = [f32];

/// A struct of the crate's whose last field is a slice.
pub struct Tail {
    pub head: u32,
    pub rest: [f32],
}

#[kernite_macros::kernels]
pub mod cpu {
    pub unsafe extern "sysv64" fn c_str(p: *const core::ffi::CStr, n: usize) {
        mov(qword[p], n);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn c_str_tiles<const N: u64>(p: *const core::ffi::CStr, n: usize) {
        add(n, N);
        mov(qword[p], n);
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn c_str_together(p: *const core::ffi::CStr, n: usize) {
        mov(qword[p], n);
    }

    pub unsafe extern "sysv64" fn str_path(p: *const core::primitive::str, n: usize) {
        mov(qword[p], n);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn str_path_tiles<const N: u64>(p: *const core::primitive::str, n: usize) {
        add(n, N);
        mov(qword[p], n);
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn str_path_together(p: *const core::primitive::str, n: usize) {
        mov(qword[p], n);
    }

    pub unsafe extern "sysv64" fn floats(p: *const crate::Floats, n: usize) {
        mov(qword[p], n);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn floats_tiles<const N: u64>(p: *const crate::Floats, n: usize) {
        add(n, N);
        mov(qword[p], n);
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn floats_together(p: *const crate::Floats, n: usize) {
        mov(qword[p], n);
    }

    pub unsafe extern "sysv64" fn tail(p: *const crate::Tail, n: usize) {
        mov(qword[p], n);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tail_tiles<const N: u64>(p: *const crate::Tail, n: usize) {
        add(n, N);
        mov(qword[p], n);
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn tail_together(p: *const crate::Tail, n: usize) {
        mov(qword[p], n);
    }

    pub unsafe extern "sysv64" fn two(p: *const core::ffi::CStr, q: *mut crate::Floats) {
        mov(qword[q], p);
    }

    pub unsafe extern "sysv64" fn slice(p: *const [f32], n: usize) {
        mov(qword[p], n);
    }

    pub unsafe extern "sysv64" fn returns(p: *const u8) -> *const core::ffi::CStr {
        ret(p);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn returns_tiles<const N: u64>(p: *const u8) -> *const core::ffi::CStr {
        add(p, N);
        ret(p);
    }

    pub unsafe extern "sysv64" fn tuple(p: *const (u8, core::ffi::CStr), n: usize) {
        mov(qword[p], n);
    }

    #[cooperative]
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tuple_together_tiles<const N: u64>(p: *const (u8, core::ffi::CStr), n: usize) {
        add(n, N);
        mov(qword[p], n);
    }

    pub unsafe extern "sysv64" fn refused(p: *const core::ffi::CStr, n: usize) {
        mov(qword[p], y);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn refused_tiles<const N: u64>(p: *const core::ffi::CStr, n: usize) {
        mov(qword[p], y);
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn refused_together(p: *const core::ffi::CStr, n: usize) {
        mov(qword[p], y);
    }

    pub unsafe extern "sysv64" fn refused_returns(p: *const u8) -> *const core::ffi::CStr {
        mov(qword[p], y);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn refused_returns_tiles<const N: u64>(p: *const u8) -> *mut crate::Floats {
        mov(qword[p], y);
    }

    pub extern "sysv64" fn safe(p: *const core::ffi::CStr, n: usize) {
        mov(qword[p], n);
    }

    pub unsafe extern "C" fn unknown(p: *const core::ffi::CStr, n: usize) {
        mov(qword[p], n);
    }

    pub unsafe extern "sysv64" fn array_of_c_str(p: *const [core::ffi::CStr; 3], n: usize) {
        mov(qword[p], n);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn array_of_c_str_tiles<const N: u64>(p: *const [core::ffi::CStr; 3], n: usize) {
        add(n, N);
        mov(qword[p], n);
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn array_of_c_str_together(p: *const [core::ffi::CStr; 3], n: usize) {
        mov(qword[p], n);
    }

    pub unsafe extern "sysv64" fn array_of_c_str_returns(p: *const u8) -> *const [core::ffi::CStr; 3] {
        ret(p);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn array_of_c_str_returns_tiles<const N: u64>(
        p: *const u8,
    ) -> *mut [core::ffi::CStr; 3] {
        add(p, N);
        ret(p);
    }

    pub unsafe extern "sysv64" fn tuple_c_str_first(p: *const (core::ffi::CStr, u8), n: usize) {
        mov(qword[p], n);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tuple_c_str_first_tiles<const N: u64>(p: *const (core::ffi::CStr, u8), n: usize) {
        add(n, N);
        mov(qword[p], n);
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn tuple_c_str_first_together(p: *const (core::ffi::CStr, u8), n: usize) {
        mov(qword[p], n);
    }

    pub unsafe extern "sysv64" fn tuple_ends_in_array(p: *const (u8, [core::ffi::CStr; 3]), n: usize) {
        mov(qword[p], n);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tuple_ends_in_array_tiles<const N: u64>(
        p: *const (u8, [core::ffi::CStr; 3]),
        n: usize,
    ) {
        add(n, N);
        mov(qword[p], n);
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn tuple_ends_in_array_together(p: *const (u8, [core::ffi::CStr; 3]), n: usize) {
        mov(qword[p], n);
    }

    pub unsafe extern "sysv64" fn tuple_ends_in_array_returns(p: *const u8) -> *const (u8, [core::ffi::CStr; 3]) {
        ret(p);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tuple_ends_in_array_returns_tiles<const N: u64>(
        p: *const u8,
    ) -> *const (u8, [core::ffi::CStr; 3]) {
        add(p, N);
        ret(p);
    }
}

#[kernite_macros::kernels]
pub mod gpu {
    pub unsafe extern "ptx-kernel" fn slice(p: *mut [f32], n: u32) {
        st.global.u32([p], n);
    }

    pub unsafe extern "ptx-kernel" fn c_str(p: *mut core::ffi::CStr, n: u32) {
        st.global.u32([p], n);
    }

    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn tail_tiles<const N: u32>(p: *mut crate::Tail) {
        st.global.u32([p], N);
    }

    #[cooperative]
    pub unsafe extern "ptx-kernel" fn c_str_together(p: *mut core::ffi::CStr, n: u32) {
        st.global.u32([p], n);
    }

    #[cooperative]
    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn tail_together_tiles<const N: u32>(p: *mut crate::Tail) {
        st.global.u32([p], N);
    }

    pub unsafe extern "ptx-kernel" fn refused(p: *mut core::ffi::CStr, n: u32) {
        st.global.u32([p], y);
    }

    #[cooperative]
    pub unsafe extern "ptx-kernel" fn refused_together(p: *mut core::ffi::CStr, n: u32) {
        st.global.u32([p], y);
    }

    pub unsafe extern "ptx-kernel" fn array_of_c_str(p: *mut [core::ffi::CStr; 3], n: u32) {
        st.global.u32([p], n);
    }

    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn array_of_c_str_tiles<const N: u32>(p: *const [core::ffi::CStr; 3]) {
        st.global.u32([p], N);
    }

    #[cooperative]
    pub unsafe extern "ptx-kernel" fn array_of_c_str_together(p: *mut [core::ffi::CStr; 3], n: u32) {
        st.global.u32([p], n);
    }

    pub unsafe extern "ptx-kernel" fn tuple_c_str_first(p: *mut (core::ffi::CStr, u8), n: u32) {
        st.global.u32([p], n);
    }

    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn tuple_c_str_first_tiles<const N: u32>(p: *mut (core::ffi::CStr, u8)) {
        st.global.u32([p], N);
    }

    pub unsafe extern "ptx-kernel" fn tuple_ends_in_array(p: *mut (u8, [core::ffi::CStr; 3]), n: u32) {
        st.global.u32([p], n);
    }

    #[cooperative]
    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn tuple_ends_in_array_together_tiles<const N: u32>(
        p: *mut (u8, [core::ffi::CStr; 3]),
    ) {
        st.global.u32([p], N);
    }

    pub unsafe extern "ptx-kernel" fn returns_c_str(p: *mut u32) -> *mut core::ffi::CStr {
        st.global.u32([p], 1);
    }

    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn returns_c_str_tiles<const N: u32>(p: *mut u32) -> *mut core::ffi::CStr {
        st.global.u32([p], N);
    }
}
