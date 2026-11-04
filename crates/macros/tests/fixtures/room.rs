//! The room of a module's statics counts the statics `cfg` keeps. `alt` has two tables of constant
//! memory of 48000 bytes each, which together are more than the 64 KB a module has, and `cfg` keeps
//! one: `FAST` with the feature `new`, and `SLOW` without it. `room` has two statics of global memory of
//! 2 GB each, which make 4 GB together, where a module has less, and `cfg` leaves out the first. The
//! crate builds without a word under `deny(warnings)`, with the feature and without it, and its tests
//! assert which table the module has and, on a machine with a GPU, load the module and read the table
//! that is there.

#![deny(warnings)]

/// Two tables, of which `cfg` keeps one, and the kernel of each that reads the last word of it.
#[kernite_macros::kernels]
pub mod alt {
    /// With the feature: every word is 7.
    #[cfg(feature = "new")]
    pub static FAST: constant<[u32; 12000]> = [7; 12000];

    /// Without it: every word is 9.
    #[cfg(not(feature = "new"))]
    pub static SLOW: constant<[u32; 12000]> = [9; 12000];

    /// Writes the last word of `FAST` to `out`.
    #[cfg(feature = "new")]
    pub unsafe extern "ptx-kernel" fn fast(out: *mut u32) {
        let word: u32;
        ld.r#const.u32(word, [FAST + 47996]);
        st.global.u32([out], word);
    }

    /// Writes the last word of `SLOW` to `out`.
    #[cfg(not(feature = "new"))]
    pub unsafe extern "ptx-kernel" fn slow(out: *mut u32) {
        let word: u32;
        ld.r#const.u32(word, [SLOW + 47996]);
        st.global.u32([out], word);
    }
}

/// A static of global memory that `cfg` leaves out, and one that it keeps: each is 2 GB.
#[kernite_macros::kernels]
pub mod room {
    /// Left out.
    #[cfg(any())]
    pub static A: global<[u32; 536870912]>;

    /// Kept: it fits alone.
    pub static B: global<[u32; 536870912]>;
}

#[cfg(test)]
mod tests {
    use kernite_cuda::{Cuda, Grid};

    use super::alt::{Kernels, PTX};

    /// The module declares the table `cfg` keeps, and not the other.
    #[test]
    fn the_module_has_the_table_cfg_keeps() {
        let new = cfg!(feature = "new");
        assert_eq!(PTX.contains("4FASTE"), new, "{PTX}");
        assert_eq!(PTX.contains("4SLOWE"), !new, "{PTX}");
    }

    /// The module loads on a GPU, and the kernel reads the table that is there: 7 with the feature,
    /// where `FAST` is, and 9 without it.
    #[test]
    fn the_kernel_reads_the_table_cfg_keeps() {
        let Ok(cuda) = Cuda::new().inspect_err(|e| eprintln!("no GPU: skipped. {e}")) else { return };
        let kernels = Kernels::load(&cuda).unwrap();
        let out = cuda.alloc::<u32>(1).unwrap();
        // SAFETY: the kernel writes the one word of `out`.
        #[cfg(feature = "new")]
        unsafe { kernels.fast(Grid::new(1, 1), out.ptr()) }.unwrap();
        // SAFETY: as above.
        #[cfg(not(feature = "new"))]
        unsafe { kernels.slow(Grid::new(1, 1), out.ptr()) }.unwrap();
        assert_eq!(out.read().unwrap(), [if cfg!(feature = "new") { 7 } else { 9 }]);
    }
}
