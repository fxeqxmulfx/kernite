//! A static that does not fit beside those before it that `cfg` keeps is refused at its type, in the
//! words it has without `cfg`, and one beside which `cfg` leaves the rest out is not. Each table of
//! constant memory is 48000 bytes, of the 64 KB a module has, and each of global memory 2 GB, of the
//! less than 4 GB it has. The crate has the feature `new`. A static that is refused takes no room:
//! `plain`'s `C` fits beside `A` alone.

/// No `cfg`: `B` does not fit beside `A`, and `C` does.
#[kernite_macros::kernels]
pub mod plain {
    pub static A: constant<[u32; 12000]> = [0; 12000];
    pub static B: constant<[u32; 12000]> = [0; 12000];
    pub static C: constant<[u32; 4]> = [0; 4];
}

/// Both tables are there with the feature, and `B` does not fit beside `A`; without it, neither is.
#[kernite_macros::kernels]
pub mod kept {
    #[cfg(feature = "new")]
    pub static A: constant<[u32; 12000]> = [0; 12000];
    #[cfg(feature = "new")]
    pub static B: constant<[u32; 12000]> = [0; 12000];
}

/// The same, for global memory.
#[kernite_macros::kernels]
pub mod kept_global {
    #[cfg(feature = "new")]
    pub static A: global<[u32; 536870912]>;
    #[cfg(feature = "new")]
    pub static B: global<[u32; 536870912]>;
}

/// `cfg` leaves out one of the two tables, whichever the feature says: `A` is there with it, and
/// `B`, under a `cfg_attr(..)`, without it.
#[kernite_macros::kernels]
pub mod either {
    #[cfg(feature = "new")]
    pub static A: constant<[u32; 12000]> = [0; 12000];
    #[cfg_attr(feature = "new", cfg(any()))]
    pub static B: constant<[u32; 12000]> = [0; 12000];
}
