//! A static that does not fit beside those before it that `cfg` keeps is refused at its type, in the
//! words it has without `cfg`, and one beside which `cfg` leaves the rest out is not. Each table of
//! constant memory is 48000 bytes, of the 64 KB a module has. The crate has the feature `new`. A static that is refused
//! takes no room: `plain`'s `C` fits beside `A` alone. A static of global memory that the module has no room for is
//! outside it, and its address takes 8 bytes of constant memory: `slot`'s.

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

/// `cfg` leaves out one of the two tables, whichever the feature says: `A` is there with it, and
/// `B`, under a `cfg_attr(..)`, without it.
#[kernite_macros::kernels]
pub mod either {
    #[cfg(feature = "new")]
    pub static A: constant<[u32; 12000]> = [0; 12000];
    #[cfg_attr(feature = "new", cfg(any()))]
    pub static B: constant<[u32; 12000]> = [0; 12000];
}

/// `BIG` is outside the module, in global memory the driver allocates, and the constant memory holds its address, 8
/// bytes at a multiple of 8: it does not fit beside a table of 65529 bytes, and is refused.
#[kernite_macros::kernels]
pub mod slot {
    pub static TABLE: constant<[u8; 65529]>;
    pub static BIG: global<[u8; 6442450944]>;
}
