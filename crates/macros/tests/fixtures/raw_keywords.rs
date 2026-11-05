//! Kernels that take pointers to types named by a keyword written raw. Rust reads `r#dyn` and `r#mut` as names, so
//! a pointer to `r#dyn::Foo`, a struct of a module named `dyn`, or to `r#mut<u64>`, a generic struct, has one word,
//! where a pointer to `dyn Foo` has two. Each is a name from the macro on: Rust builds the crate without a word, each
//! x86 kernel has the listing of the kernel beside it that names types of the same layout by words that are no
//! keywords, `dyn_::Foo` and `Mut<u64>` say, and runs as it does; and the GPU kernels, whose types are aliases of
//! numbers, which a buffer has, run on the GPU. They skip where there is none.

/// Modules named by a keyword, and by a word that is none.
pub mod paths {
    pub mod r#dyn {
        #[repr(C)]
        pub struct Foo(pub u64);

        pub type Word = u32;
    }

    pub mod dyn_ {
        #[repr(C)]
        pub struct Foo(pub u64);
    }
}

/// Generic types named by a keyword, and by a word that is none.
#[allow(non_camel_case_types)]
pub mod generics {
    #[repr(C)]
    pub struct r#mut<T>(pub T);

    #[repr(C)]
    pub struct r#for<T>(pub T);

    #[repr(C)]
    pub struct Mut<T>(pub T);

    #[repr(C)]
    pub struct For<T>(pub T);

    pub type r#const<T> = T;
}

#[kernite_macros::kernels]
pub mod keyword {
    use super::generics::{r#for, r#mut};
    use super::paths::r#dyn;

    /// Writes `n` through `p`, and gives `p` back.
    pub unsafe extern "sysv64" fn path(p: *mut r#dyn::Foo, n: u64) -> *mut r#dyn::Foo {
        mov(qword[p], n);
        ret(p);
    }

    /// The same, for a generic struct named `mut`.
    pub unsafe extern "sysv64" fn generic(p: *mut r#mut<u64>, n: u64) -> *mut r#mut<u64> {
        mov(qword[p], n);
        ret(p);
    }

    /// The same, for a generic struct named `for`, the word that opens a binder, `for<'a>`.
    pub unsafe extern "sysv64" fn binder(p: *mut r#for<u64>, n: u64) -> *mut r#for<u64> {
        mov(qword[p], n);
        ret(p);
    }

    /// Writes `N` through `p`, and gives `p` back.
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tiles<const N: u64>(p: *mut r#dyn::Foo) -> *mut r#dyn::Foo {
        mov(qword[p], N);
        ret(p);
    }

    /// Every thread writes 1 through `p`.
    #[cooperative]
    pub unsafe extern "sysv64" fn together(p: *mut r#mut<u32>) {
        mov(dword[p], 1);
    }
}

#[kernite_macros::kernels]
pub mod no_keyword {
    use super::generics::{For, Mut};
    use super::paths::dyn_;

    /// Writes `n` through `p`, and gives `p` back.
    pub unsafe extern "sysv64" fn path(p: *mut dyn_::Foo, n: u64) -> *mut dyn_::Foo {
        mov(qword[p], n);
        ret(p);
    }

    /// The same, for a generic struct named `Mut`.
    pub unsafe extern "sysv64" fn generic(p: *mut Mut<u64>, n: u64) -> *mut Mut<u64> {
        mov(qword[p], n);
        ret(p);
    }

    /// The same, for a generic struct named `For`.
    pub unsafe extern "sysv64" fn binder(p: *mut For<u64>, n: u64) -> *mut For<u64> {
        mov(qword[p], n);
        ret(p);
    }

    /// Writes `N` through `p`, and gives `p` back.
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tiles<const N: u64>(p: *mut dyn_::Foo) -> *mut dyn_::Foo {
        mov(qword[p], N);
        ret(p);
    }

    /// Every thread writes 1 through `p`.
    #[cooperative]
    pub unsafe extern "sysv64" fn together(p: *mut Mut<u32>) {
        mov(dword[p], 1);
    }
}

#[kernite_macros::kernels]
pub mod gpu {
    use super::generics::r#const;
    use super::paths::r#dyn;

    /// Writes `n` at `p`.
    pub unsafe extern "ptx-kernel" fn path(p: *mut r#dyn::Word, n: u32) {
        st.global.u32([p], n);
    }

    /// Writes `n` at `p`, a pointer to an alias named `const`.
    pub unsafe extern "ptx-kernel" fn generic(p: *mut r#const<u32>, n: u32) {
        st.global.u32([p], n);
    }

    /// Writes `base + N` at `p`.
    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn offset<const N: u32>(p: *mut r#dyn::Word, base: u32) {
        let at: u32;
        add.u32(at, base, N);
        st.global.u32([p], at);
    }
}

/// The same checks for each module of x86 kernels, which take a pointer to a word, whatever type they name.
#[cfg(test)]
macro_rules! x86_kernels_run {
    ($kernels:ident) => {
        let (mut word, mut half) = (0u64, 0u32);
        let at: *mut u64 = &raw mut word;
        // SAFETY: each kernel writes the word or half it is given, which is there, and gives the pointer back.
        unsafe {
            assert_eq!($kernels::path(at.cast(), 5).cast::<u64>(), at);
            assert_eq!(word, 5);
            assert_eq!($kernels::generic(at.cast(), 6).cast::<u64>(), at);
            assert_eq!(word, 6);
            assert_eq!($kernels::binder(at.cast(), 7).cast::<u64>(), at);
            assert_eq!(word, 7);
            assert_eq!($kernels::tiles::<1>(at.cast()).cast::<u64>(), at);
            assert_eq!(word, 1);
            assert_eq!($kernels::tiles::<2>(at.cast()).cast::<u64>(), at);
            assert_eq!(word, 2);
            $kernels::together(kernite_cpu::Grid::new(1), (&raw mut half).cast()).unwrap();
        }
        assert_eq!(half, 1);
    };
}

#[cfg(test)]
mod tests {
    use kernite_cuda::{Cuda, Grid};

    use super::{gpu, keyword, no_keyword};

    #[test]
    fn a_kernel_that_names_types_by_keywords_has_the_listing_of_one_that_does_not() {
        // `no_keyword` names types of the same layout by words that are no keywords. It has `path`, `generic`,
        // `binder`, two of `tiles` and `together`.
        assert_eq!(no_keyword::LISTINGS.len(), 6);
        assert_eq!(keyword::LISTINGS, no_keyword::LISTINGS);
    }

    #[test]
    fn a_kernel_that_names_no_type_by_a_keyword_runs() {
        x86_kernels_run!(no_keyword);
    }

    #[test]
    fn a_kernel_that_names_types_by_keywords_runs() {
        x86_kernels_run!(keyword);
    }

    #[test]
    fn a_gpu_kernel_that_names_types_by_keywords_runs() {
        let Ok(cuda) = Cuda::new().map_err(|e| eprintln!("no GPU: skipped. {e}")) else { return };
        let kernels = gpu::Kernels::load(&cuda).unwrap();
        let words = cuda.alloc::<u32>(4).unwrap();
        let grid = Grid::new(1, 1);
        // SAFETY: each kernel writes one word to a buffer that has four, and every launch has ended when they are read.
        unsafe {
            kernels.path(grid, words.ptr(), 5).unwrap();
            kernels.generic(grid, words.at(1), 6).unwrap();
            kernels.offset::<1>(grid, words.at(2), 10).unwrap();
            kernels.offset::<2>(grid, words.at(3), 20).unwrap();
        }
        cuda.sync().unwrap();
        assert_eq!(words.read().unwrap(), [5, 6, 11, 22]);
    }
}
