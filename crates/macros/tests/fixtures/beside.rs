//! Kernels whose parameters name a type beside their module, by each path Rust has for it, in each
//! kind of x86 kernel: one that is called as it is written, one of compile-time arguments and one
//! that is cooperative. A parameter's type means in all of them what it means in any function of
//! the module. Rust builds the crate without a word, and the test runs each x86 kernel, which
//! writes through its pointer: what it wrote is what the test finds. The doc test of `W` is a crate
//! of its own, which calls two of them from outside.

/// A type beside the modules of kernels: a word, which a kernel writes. A doc test is another
/// crate, which calls a kernel of compile-time arguments and a cooperative one as one would:
///
/// ```
/// let mut w = beside::W::default();
/// unsafe {
///     beside::by_super::tile::<3>(&mut w);
///     assert_eq!(w.0, 3);
///     beside::by_super::together(kernite_cpu::Grid::new(1), &mut w).unwrap();
/// }
/// assert_eq!(w.0, 4);
/// ```
#[derive(Default)]
pub struct W(pub u64);

/// The type is named `super::W`: the module of kernels is the crate's root's.
#[kernite_macros::kernels]
pub mod by_super {
    pub unsafe extern "sysv64" fn plain(p: *mut super::W) {
        mov(qword[p], 1);
    }

    #[instances(2, 3)]
    pub unsafe extern "sysv64" fn tile<const N: u64>(p: *mut super::W) -> *mut super::W {
        mov(qword[p], N);
        ret(p);
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn together(p: *mut super::W) {
        mov(qword[p], 4);
    }
}

/// The type is named `crate::W`.
#[kernite_macros::kernels]
pub mod by_crate {
    pub unsafe extern "sysv64" fn plain(p: *mut crate::W) {
        mov(qword[p], 1);
    }

    #[instances(2, 3)]
    pub unsafe extern "sysv64" fn tile<const N: u64>(p: *mut crate::W) -> *mut crate::W {
        mov(qword[p], N);
        ret(p);
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn together(p: *mut crate::W) {
        mov(qword[p], 4);
    }
}

/// The type is named `W`, which the module takes from beside it.
#[kernite_macros::kernels]
pub mod by_use {
    use super::W;

    pub unsafe extern "sysv64" fn plain(p: *mut W) {
        mov(qword[p], 1);
    }

    #[instances(2, 3)]
    pub unsafe extern "sysv64" fn tile<const N: u64>(p: *mut W) -> *mut W {
        mov(qword[p], N);
        ret(p);
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn together(p: *mut W) {
        mov(qword[p], 4);
    }
}

/// The type is named `self::W`: the module holds a `W` of its own, not the crate's.
#[kernite_macros::kernels]
pub mod by_self {
    /// A word of the module's own.
    #[derive(Default)]
    pub struct W(pub u64);

    pub unsafe extern "sysv64" fn plain(p: *mut self::W) {
        mov(qword[p], 1);
    }

    #[instances(2, 3)]
    pub unsafe extern "sysv64" fn tile<const N: u64>(p: *mut self::W) -> *mut self::W {
        mov(qword[p], N);
        ret(p);
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn together(p: *mut self::W) {
        mov(qword[p], 4);
    }
}

/// The type is named `W`, which the module holds: its own, not the crate's.
#[kernite_macros::kernels]
pub mod by_name {
    /// A word of the module's own.
    #[derive(Default)]
    pub struct W(pub u64);

    pub unsafe extern "sysv64" fn plain(p: *mut W) {
        mov(qword[p], 1);
    }

    #[instances(2, 3)]
    pub unsafe extern "sysv64" fn tile<const N: u64>(p: *mut W) -> *mut W {
        mov(qword[p], N);
        ret(p);
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn together(p: *mut W) {
        mov(qword[p], 4);
    }
}

/// A module that holds a type of its own and the modules of kernels beside it, one level down.
pub mod outer {
    /// A word of this module's, beside the modules of kernels.
    #[derive(Default)]
    pub struct V(pub u64);

    /// The type is named `super::V`, one module up; the crate's root is two.
    #[kernite_macros::kernels]
    pub mod near {
        pub unsafe extern "sysv64" fn plain(p: *mut super::V) {
            mov(qword[p], 1);
        }

        #[instances(2, 3)]
        pub unsafe extern "sysv64" fn tile<const N: u64>(p: *mut super::V) -> *mut super::V {
            mov(qword[p], N);
            ret(p);
        }

        #[cooperative]
        pub unsafe extern "sysv64" fn together(p: *mut super::V) {
            mov(qword[p], 4);
        }
    }

    /// The type is named `super::super::W`, two modules up: the crate's root.
    #[kernite_macros::kernels]
    pub mod far {
        pub unsafe extern "sysv64" fn plain(p: *mut super::super::W) {
            mov(qword[p], 1);
        }

        #[instances(2, 3)]
        pub unsafe extern "sysv64" fn tile<const N: u64>(p: *mut super::super::W) -> *mut super::super::W {
            mov(qword[p], N);
            ret(p);
        }

        #[cooperative]
        pub unsafe extern "sysv64" fn together(p: *mut super::super::W) {
            mov(qword[p], 4);
        }
    }
}

/// GPU kernels, which the launcher of each takes by the same paths. They are built here, not run.
#[kernite_macros::kernels]
pub mod gpu {
    pub unsafe extern "ptx-kernel" fn plain(p: *mut super::W) {
        st.global.u64([p], 1);
    }

    #[instances(2, 3)]
    pub unsafe extern "ptx-kernel" fn tile<const N: u64>(p: *mut crate::W) {
        st.global.u64([p], N);
    }

    #[cooperative]
    pub unsafe extern "ptx-kernel" fn together(p: *mut super::W) {
        st.global.u64([p], 4);
    }
}

/// Each GPU kernel, launched as Rust launches it.
pub fn on_the_gpu(kernels: &gpu::Kernels, p: kernite_cuda::Ptr<W>) -> Result<u32, kernite_cuda::Error> {
    let grid = kernite_cuda::Grid::new(1, 1);
    unsafe {
        kernels.plain(grid, p)?;
        kernels.tile::<2>(grid, p)?;
        kernels.together(grid, p)?;
    }
    kernels.together_blocks(32, 0)
}

#[test]
fn each_x86_kernel_writes_through_its_pointer() {
    use outer::{far, near};

    // That the three kernels of the module `$m`, called on a word of type `$w`, each write theirs.
    macro_rules! writes {
        ($m:ident, $w:ty) => {{
            let mut w = <$w>::default();
            let grid = kernite_cpu::Grid::new(1);
            unsafe {
                $m::plain(&mut w);
                assert_eq!(w.0, 1, "{}::plain", stringify!($m));
                assert_eq!($m::tile::<2>(&mut w), &raw mut w, "{}::tile::<2>", stringify!($m));
                assert_eq!(w.0, 2, "{}::tile::<2>", stringify!($m));
                assert_eq!($m::tile::<3>(&mut w), &raw mut w, "{}::tile::<3>", stringify!($m));
                assert_eq!(w.0, 3, "{}::tile::<3>", stringify!($m));
                $m::together(grid, &mut w).unwrap();
                assert_eq!(w.0, 4, "{}::together", stringify!($m));
            }
        }};
    }

    writes!(by_super, W);
    writes!(by_crate, W);
    writes!(by_use, W);
    writes!(by_self, by_self::W);
    writes!(by_name, by_name::W);
    writes!(near, outer::V);
    writes!(far, W);
}
