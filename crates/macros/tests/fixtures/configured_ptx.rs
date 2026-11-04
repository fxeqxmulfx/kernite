//! A module of GPU kernels holds what `cfg` keeps: a static, a function that kernels call or a kernel
//! that `cfg` leaves out is neither in the module's `PTX` nor among the names `Kernels::load` asks the
//! driver for, and what comes after it is where it was. The crate has the feature `more`, and its
//! tests run with it and without it: they assert which of the module is in `PTX` and, on a machine
//! with a GPU, load it and launch each kernel that is there, which writes its own number to a
//! buffer and to a static of its own.

#![deny(warnings)]

#[kernite_macros::kernels]
pub mod parts {
    /// Always there, first.
    pub static SEEN: global<[u32; 2]>;

    /// There with the feature.
    #[cfg(feature = "more")]
    pub static MORE: global<[u32; 2]>;

    /// There with the feature, as `MORE` is: `LATE` comes after two statics of one `cfg`.
    #[cfg(feature = "more")]
    pub static AGAIN: global<[u32; 2]>;

    /// Always there, after the statics that may not be.
    pub static LATE: global<[u32; 2]>;

    /// Writes 1 to `out` and to `SEEN`.
    pub unsafe extern "ptx-kernel" fn first(out: *mut u32) {
        let seen: u64;
        mov.u64(seen, SEEN);
        st.global.u32([seen], 1);
        st.global.u32([out], 1);
    }

    /// There with the feature: writes 42, which `twice` makes of 21, to `out` and to `MORE`.
    #[cfg(feature = "more")]
    pub unsafe extern "ptx-kernel" fn middle(out: *mut u32) {
        let (v, more): (u32, u64);
        call(v, twice, 21);
        mov.u64(more, MORE);
        st.global.u32([more], v);
        st.global.u32([out], v);
    }

    /// There with the feature: the sum of a number and itself.
    #[cfg(feature = "more")]
    extern "ptx-func" fn twice(n: u32) -> u32 {
        let v: u32;
        add.u32(v, n, n);
        ret(v);
    }

    /// There with the feature, as `middle` is, an instance each: `last` comes after three names of
    /// one `cfg`. Writes its number, 7 or 8, to `out` and to `AGAIN`.
    #[cfg(feature = "more")]
    #[instances(7, 8)]
    pub unsafe extern "ptx-kernel" fn tile<const N: u32>(out: *mut u32) {
        let again: u64;
        mov.u64(again, AGAIN);
        st.global.u32([again], N);
        st.global.u32([out], N);
    }

    /// There without the feature, under a `cfg_attr(..)`: writes 3 to `out`.
    #[cfg_attr(feature = "more", cfg(any()))]
    pub unsafe extern "ptx-kernel" fn less(out: *mut u32) {
        st.global.u32([out], 3);
    }

    /// Always there, after the kernels that may not be: writes 4 to `out` and to `LATE`.
    pub unsafe extern "ptx-kernel" fn last(out: *mut u32) {
        let late: u64;
        mov.u64(late, LATE);
        st.global.u32([late], 4);
        st.global.u32([out], 4);
    }
}

#[cfg(test)]
mod tests {
    use kernite_cuda::{Cuda, Grid};

    use super::parts::{Kernels, PTX};

    /// Whether the module's PTX has the name of an item, mangled as the module's are, `5parts5firstE`.
    fn has(name: &str) -> bool {
        PTX.contains(&format!("5parts{}{name}E", name.len()))
    }

    /// The module's PTX has each item `cfg` keeps and none that it leaves out.
    #[test]
    fn ptx_holds_the_items_cfg_keeps() {
        let more = cfg!(feature = "more");
        let items = [
            ("SEEN", true),
            ("MORE", more),
            ("AGAIN", more),
            ("LATE", true),
            ("first", true),
            ("middle", more),
            ("twice", more),
            ("less", !more),
            ("last", true),
        ];
        for (name, there) in items {
            assert_eq!(has(name), there, "{name} in the PTX:\n{PTX}");
        }
        // An instance is named with its argument, `5parts4tileILj7EE`.
        for n in [7, 8] {
            assert_eq!(PTX.contains(&format!("5parts4tileILj{n}EE")), more, "tile::<{n}> in the PTX:\n{PTX}");
        }
    }

    /// The module loads on a GPU, which finds each name `load` asks for, and each kernel that is there
    /// runs, as the one it is: it writes its number to a buffer and to its static, which the host reads.
    #[test]
    fn each_kernel_that_is_there_loads_and_runs() {
        let Ok(cuda) = Cuda::new().inspect_err(|e| eprintln!("no GPU: skipped. {e}")) else { return };
        let kernels = Kernels::load(&cuda).unwrap();
        let out = cuda.alloc::<u32>(1).unwrap();
        let grid = Grid::new(1, 1);
        // SAFETY: each kernel writes the one word of `out`, and a static of the module.
        unsafe { kernels.first(grid, out.ptr()) }.unwrap();
        assert_eq!((out.read().unwrap(), kernels.SEEN().read().unwrap()), (vec![1], vec![1, 0]));
        #[cfg(feature = "more")]
        {
            // SAFETY: as above.
            unsafe { kernels.middle(grid, out.ptr()) }.unwrap();
            assert_eq!((out.read().unwrap(), kernels.MORE().read().unwrap()), (vec![42], vec![42, 0]));
            // SAFETY: as above.
            unsafe { kernels.tile::<7>(grid, out.ptr()) }.unwrap();
            assert_eq!((out.read().unwrap(), kernels.AGAIN().read().unwrap()), (vec![7], vec![7, 0]));
            // SAFETY: as above.
            unsafe { kernels.tile::<8>(grid, out.ptr()) }.unwrap();
            assert_eq!((out.read().unwrap(), kernels.AGAIN().read().unwrap()), (vec![8], vec![8, 0]));
        }
        #[cfg(not(feature = "more"))]
        {
            // SAFETY: as above.
            unsafe { kernels.less(grid, out.ptr()) }.unwrap();
            assert_eq!(out.read().unwrap(), [3]);
        }
        // SAFETY: as above.
        unsafe { kernels.last(grid, out.ptr()) }.unwrap();
        assert_eq!((out.read().unwrap(), kernels.LATE().read().unwrap()), (vec![4], vec![4, 0]));
    }
}
