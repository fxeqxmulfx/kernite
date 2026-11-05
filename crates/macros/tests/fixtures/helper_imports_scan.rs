//! Helpers that no kernel calls, beside an error or an item like one. Rust is silent about a name it cannot find only
//! where an error is an item at the module's top. `h` takes `Ret`, a type of the crate's that the module imports, and
//! `language` the language's types. In `after_an_attribute` the error is right after the module's own attribute, where
//! an error of the module as a whole is. In `after_a_kernel` it is right after a kernel `cfg` may leave out, whose
//! `cfg` is not the error's. In `rusts_error` it is Rust's own, in a type, and no item. `a_macro_item` has no error,
//! but an item written as the macros write one, `::std::thread_local! {..}`. Rust is shown the helpers' types in the
//! first two, and says nothing of the types it cannot find. In the last two it is not, and says the import is unused.

pub struct Ret(pub u64);

#[kernite_macros::kernels]
pub mod after_an_attribute {
    #![allow(dead_code)]
    use super::Ret;

    static COUNTS: global<[u32; 4]>;

    fn h(x: Ret) {
        nop();
    }

    fn language(a: gpr, b: ymm, c: local<[u32; 4], 16>) -> r32 {
        nop();
    }

    pub unsafe extern "sysv64" fn k(x: u64) -> u64 {
        ret(x);
    }

    pub unsafe extern "ptx-kernel" fn g(p: *mut u32) {
        let v: u32;
        ld.global.u32(v, [COUNTS]);
        st.global.u32([p], v);
    }
}

#[kernite_macros::kernels]
pub mod after_a_kernel {
    use super::Ret;

    fn h(x: Ret) {
        nop();
    }

    fn language(a: gpr, b: ymm, c: local<[u32; 4], 16>) -> r32 {
        nop();
    }

    #[cfg(feature = "a")]
    pub unsafe extern "sysv64" fn kept(x: *mut u64) {
        mov(qword[x], 1);
    }

    pub unsafe extern "C" fn unknown() {}
}

#[kernite_macros::kernels]
pub mod rusts_error {
    use super::Ret;

    type Boom = ::core::compile_error!("an error of Rust's own");

    fn h(x: Ret) {
        nop();
    }

    fn language(a: gpr, b: ymm, c: local<[u32; 4], 16>) -> r32 {
        nop();
    }

    pub unsafe extern "sysv64" fn k(x: *mut u64) {
        mov(qword[x], 1);
    }
}

#[kernite_macros::kernels]
pub mod a_macro_item {
    use super::Ret;

    ::std::thread_local! {
        static COUNT: ::core::cell::Cell<u32> = const { ::core::cell::Cell::new(0) };
    }

    fn h(x: Ret) {
        nop();
    }

    fn language(a: gpr, b: ymm, c: local<[u32; 4], 16>) -> r32 {
        nop();
    }

    pub unsafe extern "sysv64" fn k(x: *mut u64) {
        mov(qword[x], 1);
    }
}
