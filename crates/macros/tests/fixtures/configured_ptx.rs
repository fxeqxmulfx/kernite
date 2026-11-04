//! A module of GPU kernels holds what `cfg` keeps: a static, a function that kernels call or a kernel
//! that `cfg` leaves out is neither in the module's `PTX` nor among the names `Kernels::load` asks the
//! driver for, and what comes after it is where it was. The module's `.version` and `.target` are the
//! oldest that have every instruction of the kernels and functions `cfg` keeps, though the kernel that
//! calls a function is left out. The crate has the features `more` and `new`, and its tests run with
//! each and without: they assert which of the module is in `PTX` and, on a machine with a GPU, load it
//! and launch each kernel that is there, which writes its own number to a buffer and to a static of its
//! own, or what the warp's threads sum to.

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

/// A kernel for any GPU, and one for `sm_80` and newer, PTX 7.0, under the feature `new`.
#[kernite_macros::kernels]
pub mod mixed {
    /// Writes 1.
    pub unsafe extern "ptx-kernel" fn plain(out: *mut u32) {
        st.global.u32([out], 1);
    }

    /// Writes `n` summed over the warp: with 32 threads each, 32 `n`.
    #[cfg(feature = "new")]
    pub unsafe extern "ptx-kernel" fn newer(out: *mut u32, n: u32) {
        let sum: u32;
        redux.sync.add.u32(sum, n, 0xffff_ffff);
        st.global.u32([out], sum);
    }
}

/// A function for `sm_75` and newer, PTX 7.0, under the feature `more`, and the one kernel that calls
/// it, which `cfg` leaves out.
#[kernite_macros::kernels]
pub mod lent {
    /// Writes 1.
    pub unsafe extern "ptx-kernel" fn plain(out: *mut u32) {
        st.global.u32([out], 1);
    }

    /// There with the feature: the hyperbolic tangent, an instruction of `sm_75` and newer.
    #[cfg(feature = "more")]
    extern "ptx-func" fn fast(f: f32) -> f32 {
        let v: f32;
        tanh.approx.f32(v, f);
        ret(v);
    }

    /// Never there, and the only kernel that calls `fast`.
    #[cfg(any())]
    pub unsafe extern "ptx-kernel" fn shine(out: *mut f32, f: f32) {
        let v: f32;
        call(v, fast, f);
        st.global.f32([out], v);
    }
}

#[cfg(test)]
mod tests {
    use kernite_cuda::{Cuda, Grid};

    use super::{lent, mixed};
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

    /// The module's `.version` and `.target` are the floor, PTX 3.0 and `sm_30`, without the kernel
    /// for `sm_80`, and `sm_80`'s, PTX 7.0, with it, which is in `PTX` with it alone.
    #[test]
    fn the_header_is_of_the_kernels_cfg_keeps() {
        let new = cfg!(feature = "new");
        let (version, target) = if new { ("7.0", "sm_80") } else { ("3.0", "sm_30") };
        let want = format!("\n.version {version}\n.target {target}\n.address_size 64\n");
        assert!(mixed::PTX.contains(&want), "{want} in the PTX:\n{}", mixed::PTX);
        assert!(mixed::PTX.contains("5mixed5plainE"), "{}", mixed::PTX);
        assert_eq!(mixed::PTX.contains("5mixed5newerE"), new, "{}", mixed::PTX);
    }

    /// A function `cfg` keeps sets the module's `.version` and `.target` as a kernel does, though the
    /// one kernel that calls it is left out: they are PTX 7.0 and `sm_75` with the feature `more`, which
    /// keeps the function alone, and the floor without it.
    #[test]
    fn a_function_cfg_keeps_sets_the_header_though_its_kernel_is_left_out() {
        let more = cfg!(feature = "more");
        let (version, target) = if more { ("7.0", "sm_75") } else { ("3.0", "sm_30") };
        let want = format!("\n.version {version}\n.target {target}\n.address_size 64\n");
        assert!(lent::PTX.contains(&want), "{want} in the PTX:\n{}", lent::PTX);
        assert_eq!(lent::PTX.contains("4lent4fastE"), more, "{}", lent::PTX);
        assert!(!lent::PTX.contains("5shine"), "{}", lent::PTX);
    }

    /// The module loads on the GPU, as the driver compiles it for the GPU it finds, and its kernels
    /// run. With the feature `new`, whose kernel is for `sm_80` and newer, the test says it is skipped
    /// on an older GPU, where the module cannot load.
    #[test]
    fn a_module_of_the_header_it_needs_loads_and_runs() {
        let Ok(cuda) = Cuda::new().inspect_err(|e| eprintln!("no GPU: skipped. {e}")) else { return };
        if cfg!(feature = "new") && cuda.sm() < 80 {
            eprintln!("the {} is older than Ampere, sm_80: skipped", cuda.name());
            return;
        }
        let kernels = mixed::Kernels::load(&cuda).unwrap();
        let out = cuda.alloc::<u32>(1).unwrap();
        // SAFETY: each kernel writes the one word of `out`.
        unsafe { kernels.plain(Grid::new(1, 32), out.ptr()) }.unwrap();
        assert_eq!(out.read().unwrap(), [1]);
        #[cfg(feature = "new")]
        {
            // SAFETY: as above.
            unsafe { kernels.newer(Grid::new(1, 32), out.ptr(), 2) }.unwrap();
            assert_eq!(out.read().unwrap(), [64]);
        }
    }
}
