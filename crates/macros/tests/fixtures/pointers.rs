//! Kernels that take a pointer to any type, an address in one register whatever it points to: an
//! array, a pointer, a tuple, and a type of the crate's with two arguments, whose comma is no
//! parameter's end. Rust builds each module as written, without a word.

/// A type of the crate's with two arguments, one of them a number.
#[repr(C)]
pub struct Pair<T, const N: usize>(pub [T; N]);

/// Each x86 kernel, called as Rust calls it.
pub fn on_the_cpu(
    shape: *const [usize; 3],
    out: *mut *const f32,
    pair: *const (u32, u32),
    cell: *mut Pair<u8, 2>,
) -> Result<(), kernite_cpu::Error> {
    unsafe {
        cpu::points(shape, out, pair, cell);
        cpu::tiles::<2>(out, cell);
        cpu::together(kernite_cpu::Grid::new(1), out, cell)
    }
}

/// Each GPU kernel, launched as Rust launches it.
pub fn on_the_gpu(
    kernels: &gpu::Kernels,
    shape: kernite_cuda::Ptr<[u32; 3]>,
    out: kernite_cuda::Ptr<*const f32>,
    pair: kernite_cuda::Ptr<(u32, u32)>,
    cell: kernite_cuda::Ptr<Pair<u8, 2>>,
) -> Result<(), kernite_cuda::Error> {
    let grid = kernite_cuda::Grid::new(1, 1);
    unsafe {
        kernels.points(grid, shape, out, pair, cell)?;
        kernels.tiles::<2>(grid, cell)
    }
}

/// x86 kernels. `points` is called as it is written; `tiles` is of instances, which pass the
/// parameters on as their types are written; `together` is cooperative, and its launcher passes
/// each address on as the number it is.
#[kernite_macros::kernels]
pub mod cpu {
    pub unsafe extern "sysv64" fn points(
        shape: *const [usize; 3],
        out: *mut *const f32,
        pair: *const (u32, u32),
        cell: *mut crate::Pair<u8, 2>,
    ) {
        mov(qword[out], shape);
        mov(qword[cell], pair);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tiles<const N: u64>(out: *mut *const f32, cell: *mut crate::Pair<u8, 2>) {
        add(out, N);
        mov(qword[cell], out);
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn together(out: *mut *const f32, cell: *mut crate::Pair<u8, 2>) {
        mov(qword[cell], out);
    }
}

/// GPU kernels. `points` reads through the array and the tuple and writes through the others;
/// `tiles` is of instances.
#[kernite_macros::kernels]
pub mod gpu {
    pub unsafe extern "ptx-kernel" fn points(
        shape: *const [u32; 3],
        out: *mut *const f32,
        pair: *const (u32, u32),
        cell: *mut crate::Pair<u8, 2>,
    ) {
        let n: u32;
        let m: u32;
        ld.global.u32(n, [shape + 4]);
        ld.global.u32(m, [pair]);
        add.u32(n, n, m);
        st.global.u32([out], n);
        st.global.u32([cell], n);
    }

    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn tiles<const N: u32>(cell: *mut crate::Pair<u8, 2>) {
        st.global.u32([cell], N);
    }
}
