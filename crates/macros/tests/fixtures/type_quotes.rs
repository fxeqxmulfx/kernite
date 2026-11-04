//! Kernels that take or give back a type kernite refuses, with a block, an array's length in a tuple or a macro's
//! arguments, packed and spaced. kernite says each as its source has it, as the messages of rustc say a block they do
//! not work out: `Marker<{N*2}>` from `{N*2}`, `Marker<{ N * 2 }>` from `{ N * 2 }`. A block over several lines, with
//! a comment, is one line. In a module that a macro writes, a block it builds from a fragment or from tokens has a
//! space where the macro's body and what it was given meet, and a block it is given whole is as the call has it.

/// A type of the crate's with a const argument.
pub struct Marker<const N: usize>;

/// What the blocks and the lengths name.
pub const N: usize = 4;

/// A macro that takes any tokens and gives a type.
macro_rules! sum {
    ($($t:tt)*) => {
        u8
    };
}

/// x86 kernels.
#[kernite_macros::kernels]
pub mod cpu {
    use super::{Marker, N};

    pub unsafe extern "sysv64" fn block_packed(x: Marker<{N*2}>) {
        ret();
    }

    pub unsafe extern "sysv64" fn block_spaced(x: Marker<{ N * 2 }>) {
        ret();
    }

    pub unsafe extern "sysv64" fn block_inside(x: Marker<{N * 2}>) {
        ret();
    }

    pub unsafe extern "sysv64" fn block_lines(
        x: Marker<{
            // twice
            N *
            2
        }>,
    ) {
        ret();
    }

    pub unsafe extern "sysv64" fn block_with_a_comment(x: Marker<{N/* a */*2}>) {
        ret();
    }

    pub unsafe extern "sysv64" fn block_with_generics(x: Marker<{size_of::<Vec<u8>>()*2}>) {
        ret();
    }

    pub unsafe extern "sysv64" fn array_packed(x: (u8, [u8; N*2])) {
        ret();
    }

    pub unsafe extern "sysv64" fn array_spaced(x: (u8, [u8; N * 2])) {
        ret();
    }

    pub unsafe extern "sysv64" fn macro_packed(x: sum!(a+b)) {
        ret();
    }

    pub unsafe extern "sysv64" fn macro_spaced(x: sum!(a + b)) {
        ret();
    }

    pub unsafe extern "sysv64" fn macro_braces(x: sum!{a+b}) {
        ret();
    }

    pub unsafe extern "sysv64" fn result_packed(x: u64) -> Marker<{N*2}> {
        ret();
    }

    pub unsafe extern "sysv64" fn result_spaced(x: u64) -> Marker<{ N * 2 }> {
        ret();
    }
}

/// GPU kernels.
#[kernite_macros::kernels]
pub mod gpu {
    use super::{Marker, N};

    pub unsafe extern "ptx-kernel" fn block_packed(x: Marker<{N*2}>) {
        ret();
    }

    pub unsafe extern "ptx-kernel" fn block_spaced(x: Marker<{ N * 2 }>) {
        ret();
    }

    pub unsafe extern "ptx-kernel" fn macro_packed(x: sum!(a+b)) {
        ret();
    }

    pub unsafe extern "ptx-kernel" fn macro_spaced(x: sum!(a + b)) {
        ret();
    }
}

/// Modules that a macro writes, with the blocks of their types built from fragments.
macro_rules! built {
    ($module:ident, $n:expr, $t:ty, $p:tt) => {
        #[kernite_macros::kernels]
        pub mod $module {
            use super::{Marker, N};

            pub unsafe extern "sysv64" fn spaced(x: Marker<{ N * $n }>) {
                ret();
            }

            pub unsafe extern "sysv64" fn packed(x: Marker<{N*$n}>) {
                ret();
            }

            pub unsafe extern "sysv64" fn whole(x: Marker<{ $n }>) {
                ret();
            }

            pub unsafe extern "sysv64" fn name(x: Marker<{ size_of::<$t>() }>) {
                ret();
            }

            pub unsafe extern "sysv64" fn operator(x: Marker<{ N $p 2 }>) {
                ret();
            }
        }
    };
}

built!(from_numbers, 2, u8, <<);
built!(from_sums, N + 1, [u8; N + 1], +);

/// A module that a macro writes, with the arguments of a type macro built from two fragments: each is a
/// `<`, which is joint to the `,` after it at the call.
macro_rules! two_parts {
    ($p:tt, $q:tt) => {
        #[kernite_macros::kernels]
        pub mod two_parts {
            pub unsafe extern "sysv64" fn pair(x: sum!(a $p $q b)) {
                ret();
            }

            pub unsafe extern "sysv64" fn single(x: sum!(a $p b)) {
                ret();
            }
        }
    };
}

two_parts!(<, <);

/// A block over three lines, each token starting on the column where the one before it ends: a line break is a
/// space, though no column differs.
#[kernite_macros::kernels]
pub mod staircase {
    use super::{Marker, N};

    pub unsafe extern "sysv64" fn block(
        x: Marker<{
            N
             *
              2
        }>,
    ) {
        ret();
    }
}

/// A module that a macro writes, with a block that the macro is given whole, as a fragment: it is as the call
/// has it.
macro_rules! whole {
    ($module:ident, $b:expr) => {
        #[kernite_macros::kernels]
        pub mod $module {
            use super::{Marker, N};

            pub unsafe extern "sysv64" fn block(x: Marker<$b>) {
                ret();
            }
        }
    };
}

whole!(whole_packed, {N*2});
whole!(whole_spaced, { N * 2 });

/// Modules that a macro writes, with a block that it builds from the tokens it is given: there is a space where
/// its body and the tokens meet, though the body has none.
macro_rules! tokens {
    ($module:ident, $($b:tt)*) => {
        #[kernite_macros::kernels]
        pub mod $module {
            use super::{Marker, N};

            pub unsafe extern "sysv64" fn spaced(x: Marker<{ $($b)* }>) {
                ret();
            }

            pub unsafe extern "sysv64" fn packed(x: Marker<{$($b)*}>) {
                ret();
            }
        }
    };
}

tokens!(tokens_packed, N*2);
tokens!(tokens_spaced, N * 2);
