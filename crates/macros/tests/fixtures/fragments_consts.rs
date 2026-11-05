//! A const of a module that a macro writes from a fragment is one value to Rust and to the kernel that reads it: with
//! `$n` as `1 + 2`, `pub const C: u64 = $n * 2;` is 6, and not the 5 that `1 + 2 * 2` is. Rust gets the module
//! written back, with the fragment in parentheses where it stands in the const, and as it came in an array. The consts
//! are in the module of kernels, or in a module of their own that the kernels take by its path.

/// A module of kernels with consts, all of `$n`, and a kernel that reads each. Rust works out the same expressions in
/// the test, and the kernel gives back what it reads.
macro_rules! sized {
    ($module:ident, $test:ident, $n:expr) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub const C: u64 = $n * 2;
            pub const FROM: u64 = 100 - $n;
            pub const SHIFT: u64 = 1 << $n;
            pub const WIDE: u64 = $n as u64 * 3;
            pub const PAIR: [u64; 2] = [$n * 2, 2 * $n];
            pub const ROW: [u64; $n * 2] = [7; $n * 2];
            pub const TILES: [[u64; 2]; 2] = [[$n * 2, $n * 3], [$n * 4, $n * 5]];
            pub const NAMED: u64 = C + 1;

            pub unsafe extern "sysv64" fn c(x: u64) -> u64 {
                add(x, C);
                ret(x);
            }
            pub unsafe extern "sysv64" fn from(x: u64) -> u64 {
                add(x, FROM);
                ret(x);
            }
            pub unsafe extern "sysv64" fn shift(x: u64) -> u64 {
                add(x, SHIFT);
                ret(x);
            }
            pub unsafe extern "sysv64" fn wide(x: u64) -> u64 {
                add(x, WIDE);
                ret(x);
            }
            pub unsafe extern "sysv64" fn pair(x: u64) -> u64 {
                add(x, PAIR[0] * 1000 + PAIR[1]);
                ret(x);
            }
            pub unsafe extern "sysv64" fn row(x: u64) -> u64 {
                add(x, ROW.len());
                ret(x);
            }
            pub unsafe extern "sysv64" fn tile(x: u64) -> u64 {
                add(x, TILES[1][0]);
                ret(x);
            }
            pub unsafe extern "sysv64" fn named(x: u64) -> u64 {
                add(x, NAMED);
                ret(x);
            }
            pub unsafe extern "sysv64" fn built(x: u64) -> u64 {
                add(x, $n * 2 + C);
                ret(x);
            }
        }

        #[test]
        fn $test() {
            // What Rust reads: the module's consts are the crate's own.
            assert_eq!($module::C, $n * 2);
            assert_eq!($module::FROM, 100 - $n);
            assert_eq!($module::SHIFT, 1 << $n);
            assert_eq!($module::WIDE, $n as u64 * 3);
            assert_eq!($module::PAIR, [$n * 2, 2 * $n]);
            assert_eq!($module::ROW, [7; $n * 2]);
            assert_eq!($module::TILES, [[$n * 2, $n * 3], [$n * 4, $n * 5]]);
            assert_eq!($module::NAMED, $n * 2 + 1);
            // What the kernels read.
            // SAFETY: the kernels read no memory.
            unsafe {
                assert_eq!($module::c(0), $n * 2);
                assert_eq!($module::from(0), 100 - $n);
                assert_eq!($module::shift(0), 1 << $n);
                assert_eq!($module::wide(0), $n as u64 * 3);
                assert_eq!($module::pair(0), ($n * 2) * 1000 + 2 * $n);
                assert_eq!($module::row(0), $module::ROW.len() as u64);
                assert_eq!($module::tile(0), $n * 4);
                assert_eq!($module::named(0), $n * 2 + 1);
                assert_eq!($module::built(0), $n * 2 + $n * 2);
            }
        }
    };
}

sized!(const_plain, a_const_built_from_a_number_is_the_number_it_is, 3);
sized!(const_sum, a_const_built_from_a_sum_is_one_value, 1 + 2);
sized!(const_or, a_const_built_from_an_or_is_one_value, 2 | 1);

/// A fragment that holds a fragment: `doubled_then_tripled` calls `tripled` with `$n * 2`, where `$n` is its own
/// fragment, and `tripled` multiplies that by 3 in turn.
macro_rules! tripled {
    ($module:ident, $test:ident, $m:expr) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub const D: u64 = $m * 3;

            pub unsafe extern "sysv64" fn d(x: u64) -> u64 {
                add(x, D);
                ret(x);
            }
        }

        #[test]
        fn $test() {
            assert_eq!($module::D, $m * 3);
            // SAFETY: the kernel reads no memory.
            assert_eq!(unsafe { $module::d(0) }, $m * 3);
        }
    };
}

macro_rules! doubled_then_tripled {
    ($module:ident, $test:ident, $n:expr) => {
        tripled!($module, $test, $n * 2);
    };
}

doubled_then_tripled!(const_passed_on, a_const_built_from_a_fragment_passed_on_is_one_value, 1 + 2);

/// A module of consts that kernels in another module take by its path, with a const built from `$n`.
macro_rules! shared {
    ($module:ident, $n:expr) => {
        #[kernite_macros::consts]
        pub mod $module {
            pub const C: u64 = $n * 2;
            pub const ROW: [u64; $n * 2] = [7; $n * 2];
            pub const FROM: u64 = 100 - $n;
        }
    };
}

shared!(sizes, 1 + 2);

#[kernite_macros::kernels(consts = crate::sizes)]
pub mod reads_sizes {
    pub unsafe extern "sysv64" fn c(x: u64) -> u64 {
        add(x, C);
        ret(x);
    }
    pub unsafe extern "sysv64" fn row(x: u64) -> u64 {
        add(x, ROW.len());
        ret(x);
    }
    pub unsafe extern "sysv64" fn from(x: u64) -> u64 {
        add(x, FROM);
        ret(x);
    }
}

#[test]
fn consts_a_module_of_its_own_hands_on_are_one_value_to_rust_and_to_the_kernel() {
    assert_eq!(sizes::C, 6, "Rust's C");
    assert_eq!(sizes::ROW.len(), 6, "Rust's ROW");
    assert_eq!(sizes::FROM, 97, "Rust's FROM");
    // SAFETY: the kernels read no memory.
    unsafe {
        assert_eq!(reads_sizes::c(0), 6, "the kernel's C");
        assert_eq!(reads_sizes::row(0), 6, "the kernel's ROW");
        assert_eq!(reads_sizes::from(0), 97, "the kernel's FROM");
    }
}
