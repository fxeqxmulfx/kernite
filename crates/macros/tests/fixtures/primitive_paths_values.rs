//! A constant of a primitive named by the primitive's path, `core::primitive::u32::MAX`, is the constant named by the
//! primitive's name, `u32::MAX`: Rust names it by either, and so does a module of kernels, in the four spellings Rust
//! has, `core::primitive::u32::MAX`, `::core::primitive::u32::MAX`, `std::primitive::u32::MAX` and
//! `::std::primitive::u32::MAX`, and with a name written raw, `r#core::primitive::u32::MAX` and
//! `::core::primitive::r#u32::MAX`; in a const, a static's data, a compile-time argument, the values an instance lists,
//! and a kernel's operand. Each module here is twice, once by the names and once by the paths. Rust builds each without
//! a word, a kernel by the paths is called as the same kernel by the names is and gives the same result, and its
//! listing, or its PTX, is the one of the kernel by the names.

#![deny(warnings)]

/// The module by the names, for x86.
#[kernite_macros::kernels]
pub mod plain_x86 {
    pub const TOP: u64 = u32::MAX as u64;
    pub const LOW: i32 = i16::MIN as i32;
    pub const SIZE: u32 = u64::BITS;
    pub const ONE: f32 = f32::from_bits(0x3f80_0000);
    pub static LIMITS: global<[f32; 2]> = [f32::INFINITY, f32::NEG_INFINITY];

    /// `x` moved `N` on.
    fn moved<const N: u8>(x: gpr) {
        add(x, N);
    }

    /// `x`, the constants, and `N`.
    #[instances(u8::MAX, 7)]
    pub unsafe extern "sysv64" fn sum<const N: u8>(x: u64) -> u64 {
        let g: gpr;
        mov(g, TOP);
        add(g, u16::MAX);
        add(g, SIZE);
        add(g, i8::MIN);
        moved::<u8::MAX>(g);
        add(g, N);
        add(g, x);
        ret(g);
    }
}

/// The module by the paths, for x86: the kernels of `plain_x86`.
#[kernite_macros::kernels]
pub mod paths_x86 {
    pub const TOP: u64 = r#core::primitive::u32::MAX as u64;
    pub const LOW: i32 = ::core::primitive::r#i16::MIN as i32;
    pub const SIZE: u32 = std::primitive::u64::BITS;
    pub const ONE: f32 = ::std::primitive::f32::from_bits(0x3f80_0000);
    pub static LIMITS: global<[f32; 2]> = [core::primitive::f32::INFINITY, ::std::primitive::f32::NEG_INFINITY];

    /// `x` moved `N` on.
    fn moved<const N: u8>(x: gpr) {
        add(x, N);
    }

    /// `x`, the constants, and `N`.
    #[instances(::std::primitive::u8::MAX, 7)]
    pub unsafe extern "sysv64" fn sum<const N: u8>(x: u64) -> u64 {
        let g: gpr;
        mov(g, TOP);
        add(g, std::primitive::r#u16::MAX);
        add(g, SIZE);
        add(g, ::core::primitive::i8::MIN);
        moved::<core::primitive::u8::MAX>(g);
        add(g, N);
        add(g, x);
        ret(g);
    }
}

/// The module by the names, for a GPU.
#[kernite_macros::kernels]
pub mod plain_gpu {
    pub const WIDTH: u32 = u16::MAX as u32;
    pub const SCALE: f32 = f32::from_bits(0x3fc0_0000);
    pub static LIMITS: constant<[f32; 2]> = [f32::NEG_INFINITY, f32::MAX];

    /// `x` moved `K` on.
    fn moved<const K: u32>(x: u32) {
        add.u32(x, x, K);
    }

    /// `N`, the constants, and `w`, at `out`; and `u64::MAX` at `wide`.
    #[instances(u32::MAX, 8)]
    pub unsafe extern "ptx-kernel" fn fill<const N: u32>(out: *mut u32, wide: *mut u64, w: u32) {
        let v: u32;
        let x: u64;
        mov.u32(v, N);
        add.u32(v, v, WIDTH);
        add.u32(v, v, u8::BITS);
        add.u32(v, v, w);
        moved::<usize::BITS>(v);
        st.global.u32([out], v);
        mov.u64(x, u64::MAX);
        st.global.u64([wide], x);
    }

    /// A float of bits and the least double, at `out` and `d`.
    pub unsafe extern "ptx-kernel" fn floats(out: *mut f32, d: *mut f64) {
        let x: f32;
        let y: f64;
        mov.f32(x, SCALE);
        mov.f64(y, f64::MIN);
        st.global.f32([out], x);
        st.global.f64([d], y);
    }
}

/// The module by the paths, for a GPU: the kernels of `plain_gpu`.
#[kernite_macros::kernels]
pub mod paths_gpu {
    pub const WIDTH: u32 = ::core::primitive::r#u16::MAX as u32;
    pub const SCALE: f32 = std::primitive::f32::from_bits(0x3fc0_0000);
    pub static LIMITS: constant<[f32; 2]> = [::std::primitive::f32::NEG_INFINITY, core::primitive::f32::MAX];

    /// `x` moved `K` on.
    fn moved<const K: u32>(x: u32) {
        add.u32(x, x, K);
    }

    /// `N`, the constants, and `w`, at `out`; and `u64::MAX` at `wide`.
    #[instances(::core::primitive::u32::MAX, 8)]
    pub unsafe extern "ptx-kernel" fn fill<const N: u32>(out: *mut u32, wide: *mut u64, w: u32) {
        let v: u32;
        let x: u64;
        mov.u32(v, N);
        add.u32(v, v, WIDTH);
        add.u32(v, v, r#std::primitive::u8::BITS);
        add.u32(v, v, w);
        moved::<core::primitive::usize::BITS>(v);
        st.global.u32([out], v);
        mov.u64(x, ::std::primitive::u64::MAX);
        st.global.u64([wide], x);
    }

    /// A float of bits and the least double, at `out` and `d`.
    pub unsafe extern "ptx-kernel" fn floats(out: *mut f32, d: *mut f64) {
        let x: f32;
        let y: f64;
        mov.f32(x, SCALE);
        mov.f64(y, ::core::primitive::f64::MIN);
        st.global.f32([out], x);
        st.global.f64([d], y);
    }
}

#[cfg(test)]
mod tests {
    use kernite_cuda::{Cuda, Grid};

    use super::{paths_gpu, paths_x86, plain_gpu, plain_x86};

    /// The listings are the same, kernel for kernel.
    #[test]
    fn a_kernel_with_constants_named_by_paths_has_the_listing_of_the_kernel_with_them_named_by_names() {
        assert_eq!(paths_x86::LISTINGS, plain_x86::LISTINGS);
        assert_eq!(plain_x86::LISTINGS.len(), 2);
    }

    /// The PTX is the same, but for the module's name, which both mangle into their kernels' names.
    #[test]
    fn a_gpu_kernel_with_constants_named_by_paths_has_the_ptx_of_the_kernel_with_them_named_by_names() {
        assert_eq!(paths_gpu::PTX.replace("paths_gpu", "plain_gpu"), plain_gpu::PTX);
        assert!(plain_gpu::PTX.contains(".entry"));
    }

    /// The x86 kernels give the sum of the constants, by the paths and by the names, and the module's consts and
    /// static have the value the names give.
    #[test]
    fn an_x86_kernel_with_constants_named_by_paths_gives_the_value_the_names_give() {
        let sum = |n: u64, x: u64| {
            [u32::MAX as u64, u16::MAX as u64, u64::BITS as u64, i8::MIN as i64 as u64, u8::MAX as u64, n, x]
                .into_iter()
                .fold(0u64, u64::wrapping_add)
        };
        // SAFETY: the kernels read and write nothing through a pointer.
        let (by_name, by_path) = unsafe { (plain_x86::sum::<7>(1), paths_x86::sum::<7>(1)) };
        assert_eq!((by_name, by_path), (sum(7, 1), sum(7, 1)));
        // SAFETY: as above.
        let (by_name, by_path) = unsafe { (plain_x86::sum::<255>(2), paths_x86::sum::<255>(2)) };
        assert_eq!((by_name, by_path), (sum(255, 2), sum(255, 2)));
        assert_eq!((paths_x86::TOP, paths_x86::LOW, paths_x86::SIZE), (u32::MAX as u64, i16::MIN as i32, 64));
        assert_eq!((plain_x86::TOP, plain_x86::LOW, plain_x86::SIZE), (u32::MAX as u64, i16::MIN as i32, 64));
        assert_eq!((paths_x86::ONE, plain_x86::ONE), (1.0, 1.0));
        assert_eq!(paths_x86::LIMITS.read(), plain_x86::LIMITS.read());
        assert_eq!(plain_x86::LIMITS.read(), [f32::INFINITY, f32::NEG_INFINITY]);
    }

    /// The GPU kernels give the same, launched by the paths and by the names, on the GPU.
    #[test]
    fn a_gpu_kernel_with_constants_named_by_paths_gives_the_value_the_names_give() {
        let Ok(cuda) = Cuda::new().map_err(|e| eprintln!("no GPU: skipped. {e}")) else { return };
        let (plain, paths) = (plain_gpu::Kernels::load(&cuda).unwrap(), paths_gpu::Kernels::load(&cuda).unwrap());
        let ints = [cuda.alloc::<u32>(2).unwrap(), cuda.alloc::<u32>(2).unwrap()];
        let wides = [cuda.alloc::<u64>(2).unwrap(), cuda.alloc::<u64>(2).unwrap()];
        let floats = [cuda.alloc::<f32>(1).unwrap(), cuda.alloc::<f32>(1).unwrap()];
        let doubles = [cuda.alloc::<f64>(1).unwrap(), cuda.alloc::<f64>(1).unwrap()];
        // SAFETY: each kernel writes one number to each buffer it is given, which holds one or more, and the launches
        // have ended when they are read.
        unsafe {
            plain.fill::<8>(Grid::new(1, 1), ints[0].ptr(), wides[0].ptr(), 5).unwrap();
            paths.fill::<8>(Grid::new(1, 1), ints[1].ptr(), wides[1].ptr(), 5).unwrap();
            plain.floats(Grid::new(1, 1), floats[0].ptr(), doubles[0].ptr()).unwrap();
            paths.floats(Grid::new(1, 1), floats[1].ptr(), doubles[1].ptr()).unwrap();
        }
        cuda.sync().unwrap();
        for i in 0..2 {
            assert_eq!(ints[i].read().unwrap()[0], 8 + u16::MAX as u32 + u8::BITS + 5 + usize::BITS);
            assert_eq!(wides[i].read().unwrap()[0], u64::MAX);
            assert_eq!(floats[i].read().unwrap(), [1.5]);
            assert_eq!(doubles[i].read().unwrap(), [f64::MIN]);
        }
    }
}
