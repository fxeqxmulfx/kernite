//! A macro's fragment is one operand, or one type, in the places besides an x86 kernel's expression and a const: a GPU
//! kernel's operand, under a `-`, an array's length, `#[instances(..)]`, and, for a type, a parameter and a result.
//! With `$n` as `1 + 2`, `$n * 2` is 6, and not the 5 that `1 + 2 * 2` is.

/// A type of the crate's with an argument that is a type.
pub struct Wrap<T>(core::marker::PhantomData<T>);

/// The PTX line of `ptx` that holds `instruction`, if there is one.
#[cfg(test)]
fn line<'a>(ptx: &'a str, instruction: &str) -> Option<&'a str> {
    ptx.lines().map(str::trim).find(|line| line.starts_with(instruction))
}

/// A module of GPU kernels that store an operand of `$n`, one that a directive gives threads by `$n`, and a static as
/// long as `$n * 2` words.
macro_rules! gpu_operands {
    ($module:ident, $test:ident, $n:expr) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "ptx-kernel" fn scaled(p: *mut u32) {
                st.global.u32([p], $n * 2);
            }
            pub unsafe extern "ptx-kernel" fn negated(p: *mut i32) {
                st.global.s32([p], -$n);
            }
            pub unsafe extern "ptx-kernel" fn shifted(p: *mut u32) {
                st.global.u32([p], 1 << $n);
            }
            #[maxntid($n * 32)]
            pub unsafe extern "ptx-kernel" fn directed(p: *mut u32) {
                st.global.u32([p], 1);
            }
            pub static COUNTS: global<[u32; $n * 2]>;
        }

        #[test]
        fn $test() {
            let ptx = $module::PTX;
            assert_eq!(line(ptx, "st.global.u32 [%p], 6;"), Some("st.global.u32 [%p], 6;"), "{ptx}");
            assert_eq!(line(ptx, "st.global.s32 [%p], -3;"), Some("st.global.s32 [%p], -3;"), "{ptx}");
            assert_eq!(line(ptx, "st.global.u32 [%p], 8;"), Some("st.global.u32 [%p], 8;"), "{ptx}");
            assert_eq!(ptx.matches("st.global").count(), 4, "{ptx}");
            assert!(ptx.contains(".maxntid 96"), "{ptx}");
            assert!(ptx.contains("COUNTSE[24];"), "{ptx}");
        }
    };
}

gpu_operands!(gpu_plain, a_gpu_kernel_stores_what_a_number_is, 3);
gpu_operands!(gpu_sum, a_gpu_kernel_stores_one_operand_of_a_sum, 1 + 2);

/// A module of x86 kernels that declare an array of variables as long as `$n * 2` and use all of it.
macro_rules! lengths {
    ($module:ident, $test:ident, $n:expr) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "sysv64" fn last(x: u64) -> u64 {
                let v: [gpr; $n * 2];
                for k in 0..$n * 2 {
                    mov(v[k], k);
                }
                add(x, v[5]);
                ret(x);
            }
        }

        #[test]
        fn $test() {
            // SAFETY: the kernel reads no memory.
            assert_eq!(unsafe { $module::last(10) }, 10 + 5);
            assert_eq!($n * 2, 6);
        }
    };
}

lengths!(length_plain, an_array_is_as_long_as_a_number_says, 3);
lengths!(length_sum, an_array_is_as_long_as_a_sum_says, 1 + 2);

/// A module of x86 kernels that take and give a pointer to arrays of arrays, `$n * 2` long, which Rust checks against
/// the arrays it is given.
macro_rules! tables {
    ($module:ident, $test:ident, $n:expr) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "sysv64" fn k(p: *const [[u8; $n * 2]; 3]) -> *const [[u8; $n * 2]; 3] {
                add(p, 18);
                ret(p);
            }
        }

        #[test]
        fn $test() {
            let rows = [[0u8; $n * 2]; 3];
            // SAFETY: the kernel only moves the pointer it is given, and reads nothing through it.
            assert_eq!(unsafe { $module::k(&rows) } as usize - (&rows as *const _ as usize), 18);
        }
    };
}

tables!(tables_plain, arrays_of_arrays_are_as_long_as_a_number_says, 3);
tables!(tables_sum, arrays_of_arrays_are_as_long_as_a_sum_says, 1 + 2);

/// A module of x86 kernels that write a static as long as `$n * 2` words, which holds `$n * 2` and `2 * $n` to start.
macro_rules! memory {
    ($module:ident, $test:ident, $n:expr) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub static COUNTS: global<[u32; $n * 2]>;
            pub static START: global<[u32; 2]> = [$n * 2, 2 * $n];

            pub unsafe extern "sysv64" fn put() {
                mov(dword[COUNTS + 20], 7);
            }
        }

        #[test]
        fn $test() {
            // SAFETY: the kernel writes the sixth word of `COUNTS`, which nothing else touches.
            unsafe { $module::put() };
            assert_eq!($module::COUNTS.len(), $n * 2);
            assert_eq!($module::COUNTS.read()[5], 7);
            assert_eq!($module::START.read(), [$n * 2, 2 * $n]);
        }
    };
}

memory!(memory_plain, a_static_is_as_long_as_a_number_says, 3);
memory!(memory_sum, a_static_is_as_long_as_a_sum_says, 1 + 2);

/// A module of x86 kernels that are compiled for `$n * 2` and `$n`.
macro_rules! counted {
    ($module:ident, $test:ident, $n:expr) => {
        #[kernite_macros::kernels]
        pub mod $module {
            #[instances($n * 2, $n)]
            pub unsafe extern "sysv64" fn k<const N: u64>(x: u64) -> u64 {
                add(x, N);
                ret(x);
            }
        }

        #[test]
        fn $test() {
            // SAFETY: the kernels read no memory.
            unsafe {
                assert_eq!($module::k::<6>(1), 1 + $n * 2);
                assert_eq!($module::k::<3>(1), 1 + $n);
            }
            let names: Vec<&str> = $module::LISTINGS.iter().map(|(name, _)| *name).collect();
            assert_eq!(names, ["k::<6>", "k::<3>"]);
        }
    };
}

counted!(counted_plain, instances_listed_by_a_number_are_that_number, 3);
counted!(counted_sum, instances_listed_by_a_sum_are_one_value_each, 1 + 2);

/// Modules of kernels that take and give a type that is `$t`, as it is, and as an argument of a type of the crate's.
macro_rules! typed {
    ($module:ident, $gpu:ident, $test:ident, $t:ty) => {
        #[kernite_macros::kernels]
        pub mod $module {
            use super::Wrap;

            pub unsafe extern "sysv64" fn k(x: $t) -> $t {
                add(x, 8);
                ret(x);
            }
            pub unsafe extern "sysv64" fn wrapped(x: *const Wrap<$t>) -> *const Wrap<$t> {
                add(x, 16);
                ret(x);
            }
        }

        #[kernite_macros::kernels]
        pub mod $gpu {
            use super::Wrap;

            pub unsafe extern "ptx-kernel" fn k(x: $t) {
                st.global.u32([x], 1);
            }
            pub unsafe extern "ptx-kernel" fn wrapped(x: *mut Wrap<$t>) {
                st.global.u32([x], 2);
            }
        }

        #[test]
        fn $test() {
            // SAFETY: the kernels only move the pointer they are given, and read nothing through it.
            unsafe {
                assert_eq!($module::k(0x1000usize as $t) as usize, 0x1000 + 8);
                assert_eq!($module::wrapped(0x1000usize as *const Wrap<$t>) as usize, 0x1000 + 16);
            }
            assert!($gpu::PTX.contains("st.global.u32 [%x], 1;"), "{}", $gpu::PTX);
            assert!($gpu::PTX.contains("st.global.u32 [%x], 2;"), "{}", $gpu::PTX);
        }
    };
}

typed!(typed_u32, typed_u32_gpu, a_pointer_to_a_number_is_taken_and_given, *mut u32);
typed!(typed_wrap, typed_wrap_gpu, a_pointer_to_a_type_with_an_argument_is_taken_and_given, *mut Wrap<u32>);
