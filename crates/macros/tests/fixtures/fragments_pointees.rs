//! A macro's fragment `$t:ty` that holds an operator or a bracket, `crate::Wrap<u32>` or `fn(u8) -> u8`, is one type
//! behind a pointer to Rust and to the kernel, wherever the macros write it back, with nothing to say of it: a kernel
//! kernite takes is built without a word and runs, an x86 kernel and an instance of one, and a GPU kernel writes its
//! PTX. So is a fragment of bounds, `dyn Debug + Send`, in the argument of a type, `*const Wrap<$t>`, and behind a
//! pointer in a type of the module's own, `type Fat = *const $t;`.

/// A type of the crate's with an argument that is a type, of any size.
pub struct Wrap<T: ?Sized>(core::marker::PhantomData<T>);

/// Kernels that take and give a pointer to `$t`, which has a size, and a test of them.
macro_rules! pointees {
    ($module:ident, $gpu:ident, $test:ident, $t:ty) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "sysv64" fn read(x: *const $t) -> *const $t {
                add(x, 8);
                ret(x);
            }
            pub unsafe extern "sysv64" fn write(x: *mut $t) -> *mut $t {
                add(x, 16);
                ret(x);
            }
            #[instances(1, 2)]
            pub unsafe extern "sysv64" fn counted<const N: u64>(x: *const $t) -> *const $t {
                add(x, N);
                ret(x);
            }
        }

        #[kernite_macros::kernels]
        pub mod $gpu {
            pub unsafe extern "ptx-kernel" fn store(x: *mut $t) {
                st.global.u32([x], 1);
            }
        }

        #[test]
        fn $test() {
            let at = 0x1000usize;
            // SAFETY: the kernels only move the pointer they are given, and read nothing through it.
            unsafe {
                assert_eq!($module::read(at as *const $t) as usize, at + 8);
                assert_eq!($module::write(at as *mut $t) as usize, at + 16);
                assert_eq!($module::counted::<1>(at as *const $t) as usize, at + 1);
                assert_eq!($module::counted::<2>(at as *const $t) as usize, at + 2);
            }
            assert!($gpu::PTX.contains("st.global.u32 [%x], 1;"), "{}", $gpu::PTX);
        }
    };
}

pointees!(generic, generic_gpu, a_pointer_to_a_generic_type_is_taken_and_given, crate::Wrap<u32>);
pointees!(function, function_gpu, a_pointer_to_a_function_pointer_is_taken_and_given, fn(u8) -> u8);

/// Kernels that take and give a pointer to `Wrap<$t>`, which has a size whatever `$t` has, and a test of them.
macro_rules! wrapped {
    ($module:ident, $gpu:ident, $test:ident, $t:ty) => {
        #[kernite_macros::kernels]
        pub mod $module {
            use super::Wrap;

            pub unsafe extern "sysv64" fn k(x: *const Wrap<$t>) -> *const Wrap<$t> {
                add(x, 8);
                ret(x);
            }
        }

        #[kernite_macros::kernels]
        pub mod $gpu {
            use super::Wrap;

            pub unsafe extern "ptx-kernel" fn k(x: *mut Wrap<$t>) {
                st.global.u32([x], 2);
            }
        }

        #[test]
        fn $test() {
            let at = 0x1000usize;
            // SAFETY: the kernel only moves the pointer it is given, and reads nothing through it.
            assert_eq!(unsafe { $module::k(at as *const Wrap<$t>) } as usize, at + 8);
            assert!($gpu::PTX.contains("st.global.u32 [%x], 2;"), "{}", $gpu::PTX);
        }
    };
}

wrapped!(
    wrapped_bounds,
    wrapped_bounds_gpu,
    a_pointer_to_a_type_of_bounds_in_an_argument_is_taken_and_given,
    dyn core::fmt::Debug + Send
);

/// A module with a type of its own that holds a pointer to `$t`, of two words if `$t` is a trait object.
macro_rules! fat {
    ($module:ident, $test:ident, $t:ty) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub type Fat = *const $t;
            pub type Mut = *mut $t;

            pub unsafe extern "sysv64" fn k(x: u64) -> u64 {
                ret(x);
            }
        }

        #[test]
        fn $test() {
            let two_words = 2 * core::mem::size_of::<usize>();
            assert_eq!(core::mem::size_of::<$module::Fat>(), two_words);
            assert_eq!(core::mem::size_of::<$module::Mut>(), two_words);
            // SAFETY: the kernel reads no memory.
            assert_eq!(unsafe { $module::k(7) }, 7);
        }
    };
}

fat!(fat_bounds, a_type_of_the_module_holds_a_pointer_to_a_type_of_bounds, dyn core::fmt::Debug + Send);
