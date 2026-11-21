//! Memory of a module's, which its x86 kernels name, and the host reads and writes.

use std::fmt;
use std::mem::MaybeUninit;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::atomic::{
    AtomicI8, AtomicI16, AtomicI32, AtomicI64, AtomicIsize, AtomicU8, AtomicU16, AtomicU32, AtomicU64, AtomicUsize,
};

/// Memory of a module's, a static of `N` elements of `T` at an address that is a multiple of
/// `A`: the module's x86 kernels name it, as in `[COUNTS + 8]`, and the host reads and writes
/// it, `kernels::COUNTS.read()`. It is in the process's memory, so it keeps what the kernels
/// write from one call to the next, and nothing done to it fails.
///
/// Each element is an atomic of its width, which the host loads and stores, and the kernels
/// reach as the plain memory it is. A kernel's safety contract says that nothing else touches
/// the module's memory it names while it runs, so what the host reads after a call is what the
/// kernel left.
///
/// `#[kernite_macros::kernels]` makes one of each static of a module of x86 kernels, as
/// `static COUNTS: global<[u32; 16]>;` declares it.
#[repr(C)]
pub struct Global<T: Element, const N: usize, const A: usize = 1>
where
    Align<A>: Alignment,
{
    /// Nothing, at a multiple of `A`, which puts the elements after it at one.
    align: [<Align<A> as Alignment>::Zst; 0],
    /// The elements, each in an atomic of its width, kept as bytes that may be anything: Rust checks an array of
    /// atomics one element at a time, 0.3 seconds for each MiB of a static, and these bytes at once.
    cells: MaybeUninit<[T::Cell; N]>,
}

impl<T: Element, const N: usize, const A: usize> Global<T, N, A>
where
    Align<A>: Alignment,
{
    /// Its elements, as atomics.
    #[allow(unsafe_code)]
    fn cells(&self) -> &[T::Cell; N] {
        // SAFETY: Each way to make one, `new`, `zeroed` and the linker's zeros of a static the macro declares, gives
        // every byte of it a value, and an atomic of an integer holds any.
        unsafe { self.cells.assume_init_ref() }
    }

    /// Its address, for a kernel to be given; it points at its elements while the process
    /// lasts.
    pub fn ptr(&self) -> *mut T {
        self.cells.as_ptr().cast::<T>().cast_mut()
    }

    /// The address of its element `i`, for a kernel to be given the part from there on; `None`
    /// if `i` is past its end.
    pub fn at(&self, i: usize) -> Option<*mut T> {
        (i <= N).then(|| self.ptr().wrapping_add(i))
    }

    /// How many elements it has.
    pub fn len(&self) -> usize {
        N
    }

    /// Whether it has none.
    pub fn is_empty(&self) -> bool {
        N == 0
    }

    /// Copies `data` to it.
    pub fn write(&self, data: &[T; N]) {
        for (cell, &x) in self.cells().iter().zip(data) {
            T::store(cell, x);
        }
    }

    /// What it holds.
    pub fn read(&self) -> Vec<T> {
        self.cells().iter().map(T::load).collect()
    }

    /// Copies what it holds to `out`.
    pub fn read_into(&self, out: &mut [T; N]) {
        for (x, cell) in out.iter_mut().zip(self.cells()) {
            *x = T::load(cell);
        }
    }

    /// Sets every element to zero.
    pub fn zero(&self) {
        for cell in self.cells() {
            T::store(cell, T::default());
        }
    }
}

impl<T: Element, const N: usize, const A: usize> fmt::Debug for Global<T, N, A>
where
    Align<A>: Alignment,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Global({N} elements at {:p})", self.ptr())
    }
}

/// One of Rust's numbers, which memory of a module's holds: `u8` to `u64`, `i8` to `i64`,
/// `usize`, `isize`, `f32` and `f64`, each kept in the atomic of its width.
pub trait Element: Copy + Default + sealed::Sealed + 'static {
    /// The atomic of its width it is kept in.
    #[doc(hidden)]
    type Cell: Sync;

    /// What `cell` holds.
    #[doc(hidden)]
    fn load(cell: &Self::Cell) -> Self;

    /// Makes `cell` hold `x`.
    #[doc(hidden)]
    fn store(cell: &Self::Cell, x: Self);
}

/// Each number, the atomic it is kept in, and how it is that atomic's value and back.
macro_rules! elements {
    ($($t:ident in $cell:ident, $to:path, $from:path;)*) => {$(
        impl sealed::Sealed for $t {}

        impl Element for $t {
            type Cell = $cell;

            fn load(cell: &$cell) -> $t {
                $from(cell.load(Relaxed))
            }

            fn store(cell: &$cell, x: $t) {
                cell.store($to(x), Relaxed);
            }
        }

        impl<const N: usize, const A: usize> Global<$t, N, A>
        where
            Align<A>: Alignment,
        {
            /// Memory holding `values` until something writes it, for a static to be.
            pub const fn new(values: [$t; N]) -> Self {
                let mut cells = [const { $cell::new(0) }; N];
                let (mut to, mut from): (&mut [$cell], &[$t]) = (&mut cells, &values);
                while let ([cell, more_cells @ ..], [value, more_values @ ..]) = (to, from) {
                    *cell = $cell::new($to(*value));
                    (to, from) = (more_cells, more_values);
                }
                Global { align: [], cells: MaybeUninit::new(cells) }
            }

            /// Memory holding zeros until something writes it, for a static to be.
            pub const fn zeroed() -> Self {
                Global { align: [], cells: MaybeUninit::zeroed() }
            }
        }
    )*};
}

/// A number as it is, the atomic's value for an integer.
const fn same<T>(x: T) -> T {
    x
}

elements! {
    u8 in AtomicU8, same, same;
    u16 in AtomicU16, same, same;
    u32 in AtomicU32, same, same;
    u64 in AtomicU64, same, same;
    usize in AtomicUsize, same, same;
    i8 in AtomicI8, same, same;
    i16 in AtomicI16, same, same;
    i32 in AtomicI32, same, same;
    i64 in AtomicI64, same, same;
    isize in AtomicIsize, same, same;
    f32 in AtomicU32, f32::to_bits, f32::from_bits;
    f64 in AtomicU64, f64::to_bits, f64::from_bits;
}

/// What a [`Global`]'s address is a multiple of, `A`: a power of two up to a page, 4096.
#[derive(Debug)]
pub struct Align<const A: usize>;

/// The alignments a [`Global`] is at: each power of two up to a page, 4096.
pub trait Alignment: sealed::Sealed {
    /// Nothing, at a multiple of the alignment.
    #[doc(hidden)]
    type Zst: Sync;
}

/// Each alignment, and the type of nothing at a multiple of it.
macro_rules! alignments {
    ($($a:literal $zst:ident,)*) => {$(
        #[doc(hidden)]
        #[repr(align($a))]
        #[derive(Debug)]
        pub struct $zst;

        impl sealed::Sealed for Align<$a> {}

        impl Alignment for Align<$a> {
            type Zst = $zst;
        }
    )*};
}

alignments! {
    1 Z1, 2 Z2, 4 Z4, 8 Z8, 16 Z16, 32 Z32, 64 Z64, 128 Z128, 256 Z256, 512 Z512, 1024 Z1024, 2048 Z2048,
    4096 Z4096,
}

/// What stands for a static that did not compile, beside the error, when its type or its name is
/// wrong: each of [`Global`]'s methods, which does nothing, as the code that uses it does not
/// build, and the type of its elements left to the code that uses it, which says it, or Rust asks
/// for it, beside kernite's error. A static whose value alone is wrong stands as a [`Global`] of
/// its type, holding zeros.
#[doc(hidden)]
#[derive(Debug)]
pub struct Stub;

#[doc(hidden)]
impl Stub {
    pub fn ptr<T>(&self) -> *mut T {
        std::ptr::null_mut()
    }

    pub fn at<T>(&self, _: usize) -> Option<*mut T> {
        None
    }

    pub fn len(&self) -> usize {
        0
    }

    pub fn is_empty(&self) -> bool {
        true
    }

    pub fn write<T>(&self, _: &[T]) {}

    pub fn read<T>(&self) -> Vec<T> {
        Vec::new()
    }

    pub fn read_into<T>(&self, _: &mut [T]) {}

    pub fn zero(&self) {}
}

/// What only this crate implements.
mod sealed {
    pub trait Sealed {}
}

#[cfg(test)]
mod tests {
    use std::mem::{align_of, size_of};

    use super::Global;

    type Result<T = (), E = Box<dyn std::error::Error + Send + Sync>> = std::result::Result<T, E>;

    static COUNTS: Global<u32, 16, 4> = Global::<u32, 16, 4>::zeroed();
    static WEIGHTS: Global<f32, 8, 32> = Global::<f32, 8, 32>::new([1.0, 0.5, 0.25, 2.0, -1.0, 0.0, 3.0, 1.5]);
    static SIGNED: Global<i8, 3, 1> = Global::<i8, 3, 1>::new([-1, 2, -128]);

    /// Its elements are where its address is, at the multiple it says, nothing before them and
    /// nothing after but what makes the size a multiple of it.
    #[test]
    fn memory_is_its_elements_at_its_alignment() -> Result {
        assert_eq!(COUNTS.ptr().addr(), (&raw const COUNTS).addr());
        assert_eq!((align_of::<Global<u32, 16, 4>>(), size_of::<Global<u32, 16, 4>>()), (4, 64));
        assert_eq!((align_of::<Global<f32, 8, 32>>(), size_of::<Global<f32, 8, 32>>()), (32, 32));
        assert_eq!((align_of::<Global<u8, 3, 4096>>(), size_of::<Global<u8, 3, 4096>>()), (4096, 4096));
        assert_eq!((align_of::<Global<u64, 1>>(), size_of::<Global<u64, 1>>()), (8, 8));
        assert!(WEIGHTS.ptr().addr().is_multiple_of(32));
        assert_eq!(WEIGHTS.at(2).ok_or("at gave None")?.addr() - WEIGHTS.ptr().addr(), 8);
        assert_eq!((COUNTS.len(), COUNTS.is_empty()), (16, false));
        Ok(())
    }

    /// Memory made zeroed holds zeros, as a number of each width reads them, a float as `0.0`.
    #[test]
    fn zeroed_memory_holds_zeros() {
        assert_eq!(COUNTS.read(), [0; 16]);
        assert_eq!(Global::<f32, 3, 8>::zeroed().read(), [0.0; 3]);
        assert_eq!(Global::<i16, 2>::zeroed().read(), [0; 2]);
    }

    /// It holds what it is made with, which the host reads, writes and zeroes, the bits of a
    /// float as they are.
    #[test]
    fn the_host_reads_and_writes_it() {
        assert_eq!(WEIGHTS.read(), [1.0, 0.5, 0.25, 2.0, -1.0, 0.0, 3.0, 1.5]);
        assert_eq!(SIGNED.read(), [-1, 2, -128]);
        let g: Global<f64, 3> = Global::<f64, 3>::new([-0.0, f64::MIN_POSITIVE, 1.0]);
        let bits = |xs: Vec<f64>| xs.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(g.read()), bits(vec![-0.0, f64::MIN_POSITIVE, 1.0]));
        let nan = f64::from_bits(0x7ff8_0000_dead_beef);
        g.write(&[nan, 2.0, 3.0]);
        assert_eq!(g.read()[0].to_bits(), nan.to_bits());
        let mut out = [0.0; 3];
        g.read_into(&mut out);
        assert_eq!(out[1..], [2.0, 3.0]);
        g.zero();
        assert_eq!(bits(g.read()), [0; 3]);
        let u: Global<usize, 2> = Global::<usize, 2>::zeroed();
        u.write(&[usize::MAX, 1]);
        assert_eq!(u.read(), [usize::MAX, 1]);
    }

    #[test]
    fn an_address_is_in_it_or_one_past_its_end() -> Result {
        assert_eq!(COUNTS.at(16).ok_or("at gave None")?.addr() - COUNTS.ptr().addr(), 64);
        assert_eq!(COUNTS.at(17), None);
        Ok(())
    }
}
