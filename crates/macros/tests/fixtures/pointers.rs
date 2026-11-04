//! Kernels that take a pointer to any type, an address in one register whatever it points to: an
//! array, a pointer, a tuple, a function's pointer, and a type of the crate's with two arguments,
//! whose comma is no parameter's end, even when one is a function's pointer, whose arrow closes no
//! `<`; and a GPU kernel's, an array as long as its compile-time argument says, and `f16` and
//! `bf16`, which a launcher names as kernite-cuda does. Rust builds each module as written,
//! without a word.

/// A type of the crate's with two arguments, one of them a number.
#[repr(C)]
pub struct Pair<T, const N: usize>(pub [T; N]);

/// Each x86 kernel, called as Rust calls it.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn on_the_cpu(
    shape: *const [usize; 3],
    out: *mut *const f32,
    pair: *const (u32, u32),
    call: *const fn(u32) -> u32,
    table: *const Pair<fn() -> u8, 2>,
    cell: *mut Pair<u8, 2>,
) -> Result<(), kernite_cpu::Error> {
    unsafe {
        cpu::points(shape, out, pair, call, table, cell);
        cpu::tiles::<2>(out, table, cell);
        cpu::together(kernite_cpu::Grid::new(1), out, table, cell)
    }
}

/// Each GPU kernel, launched as Rust launches it.
pub fn on_the_gpu(
    kernels: &gpu::Kernels,
    shape: kernite_cuda::Ptr<[u32; 3]>,
    out: kernite_cuda::Ptr<*const f32>,
    pair: kernite_cuda::Ptr<(u32, u32)>,
    call: kernite_cuda::Ptr<fn(u32) -> u32>,
    table: kernite_cuda::Ptr<Pair<fn() -> u8, 2>>,
    cell: kernite_cuda::Ptr<Pair<u8, 2>>,
) -> Result<(), kernite_cuda::Error> {
    let grid = kernite_cuda::Grid::new(1, 1);
    unsafe {
        kernels.points(grid, shape, out, pair, call, table, cell)?;
        kernels.tiles::<2>(grid, cell)
    }
}

/// The GPU kernels whose pointers are to an array as long as a compile-time argument, and to `f16`
/// and `bf16`.
pub fn on_the_gpu_too(
    kernels: &gpu::Kernels,
    array: kernite_cuda::Ptr<[u8; 2]>,
    half: kernite_cuda::Ptr<kernite_cuda::f16>,
    brain: kernite_cuda::Ptr<kernite_cuda::bf16>,
) -> Result<(), kernite_cuda::Error> {
    let grid = kernite_cuda::Grid::new(1, 1);
    unsafe {
        kernels.arrays::<2>(grid, array)?;
        kernels.halves(grid, half, brain)
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
        call: *const fn(u32) -> u32,
        table: *const crate::Pair<fn() -> u8, 2>,
        cell: *mut crate::Pair<u8, 2>,
    ) {
        mov(qword[out], shape);
        mov(qword[cell], pair);
        mov(qword[out], call);
        mov(qword[cell], table);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tiles<const N: u64>(
        out: *mut *const f32,
        table: *const crate::Pair<fn() -> u8, 2>,
        cell: *mut crate::Pair<u8, 2>,
    ) {
        add(out, N);
        mov(qword[cell], out);
        mov(qword[cell], table);
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn together(
        out: *mut *const f32,
        table: *const crate::Pair<fn() -> u8, 2>,
        cell: *mut crate::Pair<u8, 2>,
    ) {
        mov(qword[cell], out);
        mov(qword[cell], table);
    }
}

/// GPU kernels. `points` reads through the array, the tuple and the function's pointer and writes
/// through the others; `tiles` is of instances, and so is `arrays`, whose pointer is to an array of
/// its length; `halves` points to the numbers Rust has not.
#[kernite_macros::kernels]
pub mod gpu {
    pub unsafe extern "ptx-kernel" fn points(
        shape: *const [u32; 3],
        out: *mut *const f32,
        pair: *const (u32, u32),
        call: *const fn(u32) -> u32,
        table: *mut crate::Pair<fn() -> u8, 2>,
        cell: *mut crate::Pair<u8, 2>,
    ) {
        let n: u32;
        let m: u32;
        ld.global.u32(n, [shape + 4]);
        ld.global.u32(m, [pair]);
        add.u32(n, n, m);
        ld.global.u32(m, [call]);
        add.u32(n, n, m);
        st.global.u32([out], n);
        st.global.u32([table], n);
        st.global.u32([cell], n);
    }

    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn tiles<const N: u32>(cell: *mut crate::Pair<u8, 2>) {
        st.global.u32([cell], N);
    }

    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn arrays<const N: usize>(array: *mut [u8; N]) {
        st.global.u32([array], 1);
    }

    pub unsafe extern "ptx-kernel" fn halves(half: *mut f16, brain: *const bf16) {
        let h: f16;
        let b: bf16;
        ld.global.b16(h, [half]);
        ld.global.b16(b, [brain]);
        st.global.b16([half], h);
    }
}
