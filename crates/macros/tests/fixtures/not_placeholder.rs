//! Kernels that give back a pointer to a type with a `_` in it that is no placeholder: a pattern in a block,
//! `[u8; { let _ = 1; 3 }]` or `Arr<{ let _ = 1; 3 }>`; an argument of a macro, `[u8; len!(_)]` or `byte!(_)`,
//! which drops it; a pattern in an array's length that is no block, `[u8; if let Some(_) = Some(1) { 3 } else
//! { 4 }]`; and the name of a function pointer's parameter, `fn(_: u8)`, or `fn(_: ::core::primitive::u8)` for a
//! type that is a path from the root. Rust takes each in a signature, and builds the crate without a word, under
//! `#![deny(warnings)]`. The test calls each x86 kernel, plain and of compile-time arguments: each gives back the
//! pointer it is given, and the second adds its argument to it first. A stub, which is what kernite builds for a
//! kernel that does not compile, would abort.

#![deny(warnings)]

/// A macro that drops what it is given: `len!(_)` is 3.
macro_rules! len {
    ($x:tt) => {
        3
    };
}

/// A macro that gives a type, and drops what it is given: `byte!(_)` is `u8`.
macro_rules! byte {
    ($x:tt) => {
        u8
    };
}

/// A type with a number as its argument.
#[repr(C)]
pub struct Arr<const N: usize>(pub [u8; N]);

/// x86 kernels: each gives back its pointer, and those of arguments add `N` to it.
#[kernite_macros::kernels]
pub mod cpu {
    pub unsafe extern "sysv64" fn block(p: *const u8) -> *const [u8; { let _ = 1; 3 }] {
        ret(p);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn block_tiles<const N: u64>(p: *const u8) -> *const [u8; { let _ = 1; 3 }] {
        add(p, N);
        ret(p);
    }

    pub unsafe extern "sysv64" fn macro_args(p: *const u8) -> *const [u8; len!(_)] {
        ret(p);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn macro_args_tiles<const N: u64>(p: *const u8) -> *const [u8; len!(_)] {
        add(p, N);
        ret(p);
    }

    pub unsafe extern "sysv64" fn array_length(p: *const u8) -> *const [u8; if let Some(_) = Some(1) { 3 } else { 4 }] {
        ret(p);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn array_length_tiles<const N: u64>(
        p: *const u8,
    ) -> *const [u8; if let Some(_) = Some(1) { 3 } else { 4 }] {
        add(p, N);
        ret(p);
    }

    pub unsafe extern "sysv64" fn const_argument(p: *const u8) -> *const crate::Arr<{ let _ = 1; 3 }> {
        ret(p);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn const_argument_tiles<const N: u64>(
        p: *const u8,
    ) -> *const crate::Arr<{ let _ = 1; 3 }> {
        add(p, N);
        ret(p);
    }

    pub unsafe extern "sysv64" fn type_macro(p: *const u8) -> *const byte!(_) {
        ret(p);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn type_macro_tiles<const N: u64>(p: *const u8) -> *const byte!(_) {
        add(p, N);
        ret(p);
    }

    pub unsafe extern "sysv64" fn function(p: *const u8) -> *const fn(_: u8) {
        ret(p);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn function_tiles<const N: u64>(p: *const u8) -> *const fn(_: u8) {
        add(p, N);
        ret(p);
    }

    pub unsafe extern "sysv64" fn function_path(p: *const u8) -> *const fn(_: ::core::primitive::u8) {
        ret(p);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn function_path_tiles<const N: u64>(
        p: *const u8,
    ) -> *const fn(_: ::core::primitive::u8) {
        add(p, N);
        ret(p);
    }
}

#[test]
fn each_kernel_gives_back_its_pointer() {
    let bytes = [7u8; 3];
    let p = bytes.as_ptr();
    unsafe {
        assert_eq!(cpu::block(p).cast::<u8>(), p);
        assert_eq!(cpu::block_tiles::<1>(p).cast::<u8>(), p.wrapping_add(1));
        assert_eq!(cpu::block_tiles::<2>(p).cast::<u8>(), p.wrapping_add(2));
        assert_eq!(cpu::macro_args(p).cast::<u8>(), p);
        assert_eq!(cpu::macro_args_tiles::<1>(p).cast::<u8>(), p.wrapping_add(1));
        assert_eq!(cpu::macro_args_tiles::<2>(p).cast::<u8>(), p.wrapping_add(2));
        assert_eq!(cpu::array_length(p).cast::<u8>(), p);
        assert_eq!(cpu::array_length_tiles::<1>(p).cast::<u8>(), p.wrapping_add(1));
        assert_eq!(cpu::array_length_tiles::<2>(p).cast::<u8>(), p.wrapping_add(2));
        assert_eq!(cpu::const_argument(p).cast::<u8>(), p);
        assert_eq!(cpu::const_argument_tiles::<1>(p).cast::<u8>(), p.wrapping_add(1));
        assert_eq!(cpu::const_argument_tiles::<2>(p).cast::<u8>(), p.wrapping_add(2));
        assert_eq!(cpu::type_macro(p).cast::<u8>(), p);
        assert_eq!(cpu::type_macro_tiles::<1>(p).cast::<u8>(), p.wrapping_add(1));
        assert_eq!(cpu::type_macro_tiles::<2>(p).cast::<u8>(), p.wrapping_add(2));
        assert_eq!(cpu::function(p).cast::<u8>(), p);
        assert_eq!(cpu::function_tiles::<1>(p).cast::<u8>(), p.wrapping_add(1));
        assert_eq!(cpu::function_tiles::<2>(p).cast::<u8>(), p.wrapping_add(2));
        assert_eq!(cpu::function_path(p).cast::<u8>(), p);
        assert_eq!(cpu::function_path_tiles::<1>(p).cast::<u8>(), p.wrapping_add(1));
        assert_eq!(cpu::function_path_tiles::<2>(p).cast::<u8>(), p.wrapping_add(2));
    }
}
