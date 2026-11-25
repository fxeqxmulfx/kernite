//! A launch of a kernel that is no `unsafe fn`: the memory it is lent, and what the host checks of the launch before
//! the kernel runs.

use std::ops::{Bound, Deref, RangeBounds};

use super::{Buffer, Cuda, Error, Grid, Module, Plain, Ptr, same_len, sealed};

/// Memory a kernel that is no `unsafe fn` reads, which its launcher borrows for the call: a [`Buffer`], a [`Part`] of
/// memory, or [`Unified`](super::Unified) memory of `'static`: managed or page-locked memory, or a slice the host lends
/// or registers for good. [`Cuda::copy`] takes it too, but for a slice that [`Cuda::lend`](super::Cuda::lend) lent,
/// which it refuses. The kernel and the copy are trusted with it until they are done, and the host's calls that reach
/// the memory wait for them first, as `read`, `write`, `host` and dropping the memory do; `atomics`, `host_now` and
/// `host_now_mut` wait for none, and the last two are `unsafe` for it. A slice lent or registered for less is not, as a
/// `mem::forget` of what [`Cuda::lend`](super::Cuda::lend) or [`Cuda::register`](super::Cuda::register) gave would give
/// it back to the host with no wait.
pub trait Memory<T: Plain>: sealed::Sealed {
    /// Its address and how many elements it has, as a kernel of `module` is lent them. It is an error if it is memory
    /// of another GPU than the module's.
    fn lend(&self, module: &Module) -> Result<(Ptr<T>, usize), Error>;

    /// Its address and how many elements it has, as a copy on the GPU of `cuda` reaches them. It is an error if it is
    /// memory of another GPU than `cuda`'s, or a slice that [`Cuda::lend`](super::Cuda::lend) lent.
    fn reach(&self, cuda: &Cuda) -> Result<(Ptr<T>, usize), Error>;
}

/// Memory a kernel that is no `unsafe fn` writes: its launcher borrows it mutably, so that the host holds no slice of
/// it while the call lasts, and no other launcher is lent it. So the compiler refuses a slice the host holds across a
/// launch that writes the memory, and one memory lent twice, to be read and to be written:
///
/// ```compile_fail,E0499
/// # let cuda = kernite_cuda::Cuda::new()?;
/// fn launch(_: &mut impl kernite_cuda::MemoryMut<f32>) {}
/// let mut ys = cuda.managed::<f32>(4)?;
/// let slice = ys.host_mut()?;
/// launch(&mut ys);
/// slice[0] = 1.0;
/// # Ok::<(), kernite_cuda::Error>(())
/// ```
///
/// ```compile_fail,E0502
/// # let cuda = kernite_cuda::Cuda::new()?;
/// fn launch(_: &impl kernite_cuda::Memory<f32>, _: &mut impl kernite_cuda::MemoryMut<f32>) {}
/// let mut ys = cuda.alloc::<f32>(4)?;
/// launch(&ys, &mut ys);
/// # Ok::<(), kernite_cuda::Error>(())
/// ```
pub trait MemoryMut<T: Plain>: Memory<T> {}

impl<T: Plain> sealed::Sealed for Buffer<T> {}

impl<T: Plain> Memory<T> for Buffer<T> {
    fn lend(&self, module: &Module) -> Result<(Ptr<T>, usize), Error> {
        module.reaches(&self.driver)?;
        Ok((self.ptr(), self.len))
    }

    fn reach(&self, cuda: &Cuda) -> Result<(Ptr<T>, usize), Error> {
        cuda.reaches(&self.driver)?;
        Ok((self.ptr(), self.len))
    }
}

impl<T: Plain> MemoryMut<T> for Buffer<T> {}

/// A part of memory, from an element for a count, borrowed as `R` says: `&Buffer<T>` or `&Unified<'static, T>` to read
/// it, and `&mut` of either to write it, as `&xs[i..j]` and `&mut xs[i..j]` borrow a slice. [`Buffer::part`],
/// [`Buffer::part_mut`], [`Unified::part`](super::Unified::part) and
/// [`Unified::part_mut`](super::Unified::part_mut) give it. It is [`Memory`], and [`MemoryMut`] where it borrows
/// mutably, for a kernel that is no `unsafe fn` and for a copy to be lent that part alone, and none of the rest of the
/// memory it is a part of. A launch or a copy of it keeps the checks of that memory, which GPU it is on and whether it
/// is a slice that [`Cuda::lend`](super::Cuda::lend) lent.
///
/// ```no_run
/// # let cuda = kernite_cuda::Cuda::new()?;
/// let from = cuda.pinned::<f32>(16)?;
/// let mut to = cuda.alloc::<f32>(16)?;
/// // Elements 2 to 4 of `from` over elements 5 to 7 of `to`, queued as any copy is.
/// cuda.copy(&mut to.part_mut(5..8).ok_or("past the end")?, &from.part(2..5).ok_or("past the end")?)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// A part to write borrows the memory mutably, so rustc keeps the host from touching the memory while the part lasts,
/// and from taking another part of it, one that overlaps it or not, as `&mut xs[..8]` and `&mut xs[8..]` are refused:
///
/// ```compile_fail,E0502
/// # let cuda = kernite_cuda::Cuda::new()?;
/// let mut xs = cuda.alloc::<f32>(16)?;
/// let mut part = xs.part_mut(3..10).ok_or("past the end")?;
/// let all = xs.read()?;
/// part.write(&[0.0; 7])?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ```compile_fail,E0499
/// # let cuda = kernite_cuda::Cuda::new()?;
/// let mut xs = cuda.alloc::<f32>(16)?;
/// let mut first = xs.part_mut(..8).ok_or("past the end")?;
/// let mut second = xs.part_mut(8..).ok_or("past the end")?;
/// first.write(&[0.0; 8])?;
/// second.write(&[0.0; 8])?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// A part to read is no memory to write, so a kernel that writes its memory, and a copy to it, are refused it:
///
/// ```compile_fail,E0277
/// # let cuda = kernite_cuda::Cuda::new()?;
/// fn launch(_: &mut impl kernite_cuda::MemoryMut<f32>) {}
/// let xs = cuda.alloc::<f32>(16)?;
/// launch(&mut xs.part(3..10).ok_or("past the end")?);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// A part of a buffer is read and written by the host as the buffer is: [`read`](Part::read),
/// [`read_into`](Part::read_into) and [`write`](Part::write) wait for the kernels launched before.
pub struct Part<R> {
    of: R,
    /// The address of the first element, and how many there are.
    addr: u64,
    len: usize,
}

impl<R> Part<R> {
    /// The `len` elements of `of` from the one at `at`.
    pub(super) fn new<T>(of: R, at: Ptr<T>, len: usize) -> Part<R> {
        Part { of, addr: at.addr(), len }
    }
}

/// Where `range` starts and how many elements it takes of `len`: `None` if it ends past the end, or before it starts.
pub(super) fn bounds(range: impl RangeBounds<usize>, len: usize) -> Option<(usize, usize)> {
    let from = match range.start_bound() {
        Bound::Included(&i) => i,
        Bound::Excluded(&i) => i.checked_add(1)?,
        Bound::Unbounded => 0,
    };
    let to = match range.end_bound() {
        Bound::Included(&i) => i.checked_add(1)?,
        Bound::Excluded(&i) => i,
        Bound::Unbounded => len,
    };
    (from <= to && to <= len).then(|| (from, to - from))
}

impl<R> sealed::Sealed for Part<R> {}

impl<T: Plain, M: Memory<T>, R: Deref<Target = M>> Memory<T> for Part<R> {
    fn lend(&self, module: &Module) -> Result<(Ptr<T>, usize), Error> {
        self.of.lend(module)?;
        Ok((Ptr::new(self.addr), self.len))
    }

    fn reach(&self, cuda: &Cuda) -> Result<(Ptr<T>, usize), Error> {
        self.of.reach(cuda)?;
        Ok((Ptr::new(self.addr), self.len))
    }
}

impl<T: Plain, M: MemoryMut<T>> MemoryMut<T> for Part<&mut M> {}

impl<T: Plain, R: Deref<Target = Buffer<T>>> Part<R> {
    /// What it holds, after the kernels launched before have run.
    pub fn read(&self) -> Result<Vec<T>, Error> {
        let mut out = vec![T::default(); self.len];
        self.read_into(&mut out)?;
        Ok(out)
    }

    /// Copies what it holds to `out`, after the kernels launched before have run. It is an error if `out` has not as
    /// many elements: `cuMemcpyDtoH: a part of a buffer of 4 elements read into 2`.
    pub fn read_into(&self, out: &mut [T]) -> Result<(), Error> {
        same_len("cuMemcpyDtoH", "a part of a buffer", self.len, "read into", out.len())?;
        self.of.driver.read_into(self.addr, out)
    }
}

impl<T: Plain> Part<&mut Buffer<T>> {
    /// Copies `data` to it, after the kernels launched before have run. It is an error if `data` has not as many
    /// elements: `cuMemcpyHtoD: a part of a buffer of 4 elements written with 2`.
    pub fn write(&mut self, data: &[T]) -> Result<(), Error> {
        same_len("cuMemcpyHtoD", "a part of a buffer", self.len, "written with", data.len())?;
        self.of.driver.write(self.addr, data)
    }
}

/// A number a condition on a launch is over.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quantity {
    /// How many threads each block has: the kernel's `ntid.x`.
    Threads,
    /// How many blocks the grid has: the kernel's `nctaid.x`.
    Blocks,
    /// The number that parameter `i` is.
    Param(usize),
    /// How many elements the memory has that pointer parameter `i` is lent.
    Len(usize),
}

/// What a condition on a launch keeps true of the kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Why {
    /// No number the kernel works out wraps.
    Fits,
    /// Each element the kernel reaches of the memory of pointer parameter `i` is inside it.
    Inside(usize),
    /// No two threads write one element of the memory of pointer parameter `i`, and none reads one another writes.
    Apart(usize),
}

/// A condition on a launch of a kernel that is no `unsafe fn`: the sum of its terms is not negative. A term is a factor
/// and the quantities multiplied by it, none for a number alone.
#[derive(Clone, Copy, Debug)]
pub struct Requirement {
    /// The terms.
    pub terms: &'static [(i128, &'static [Quantity])],
    /// What the condition keeps true.
    pub why: Why,
}

/// What a launch of `kernel` over `grid` is, for its conditions to be worked out: the names of its parameters, and the
/// number each is, or how many elements the memory lent for it has; a float is 0.
struct Launch<'a> {
    grid: &'a Grid,
    params: &'a [&'a str],
    values: &'a [i128],
}

impl Launch<'_> {
    /// What `q` is in this launch.
    fn value(&self, q: Quantity) -> i128 {
        let of = |i: usize| self.values.get(i).copied().unwrap_or_default();
        match q {
            Quantity::Threads => self.grid.threads[0].into(),
            Quantity::Blocks => self.grid.blocks[0].into(),
            Quantity::Param(i) | Quantity::Len(i) => of(i),
        }
    }

    /// `q` as the launcher's Rust writes it.
    fn name(&self, q: Quantity) -> String {
        let param = |i: usize| self.params.get(i).copied().unwrap_or("?");
        match q {
            Quantity::Threads => "grid.threads[0]".into(),
            Quantity::Blocks => "grid.blocks[0]".into(),
            Quantity::Param(i) => param(i).into(),
            Quantity::Len(i) => format!("{}.len()", param(i)),
        }
    }

    /// The term of `factor` times `quantities`, as Rust writes it, where `factor` is the size of the term's factor: no
    /// `i128` holds that of `i128::MIN`.
    fn term(&self, factor: u128, quantities: &[Quantity]) -> String {
        let mut parts: Vec<String> = quantities.iter().map(|&q| self.name(q)).collect();
        if factor != 1 || parts.is_empty() {
            parts.insert(0, factor.to_string());
        }
        parts.join(" * ")
    }

    /// The condition as Rust writes it, the terms with a factor above 0 on the left, and the others on the right.
    fn written(&self, r: &Requirement) -> String {
        let side = |above: bool| {
            let terms: Vec<String> = r
                .terms
                .iter()
                .filter(|(factor, _)| (*factor > 0) == above)
                .map(|(factor, quantities)| self.term(factor.unsigned_abs(), quantities))
                .collect();
            if terms.is_empty() { "0".to_string() } else { terms.join(" + ") }
        };
        format!("{} >= {}", side(true), side(false))
    }

    /// Each quantity the condition names, with its value, in the order the condition is written.
    fn values(&self, r: &Requirement) -> String {
        let mut named: Vec<Quantity> = vec![];
        let (left, right): (Vec<_>, Vec<_>) = r.terms.iter().partition(|(factor, _)| *factor > 0);
        for &q in left.iter().chain(&right).flat_map(|(_, quantities)| quantities.iter()) {
            if !named.contains(&q) {
                named.push(q);
            }
        }
        let each: Vec<String> = named.iter().map(|&q| format!("`{}` is {}", self.name(q), self.value(q))).collect();
        each.join(", ")
    }

    /// The sum of the terms of `r`; `None` for one past what a number of 128 bits holds.
    fn sum(&self, r: &Requirement) -> Option<i128> {
        r.terms.iter().try_fold(0i128, |sum, (factor, quantities)| {
            let term = quantities.iter().try_fold(*factor, |t, &q| t.checked_mul(self.value(q)))?;
            sum.checked_add(term)
        })
    }
}

/// Checks a launch of `kernel`, which is no `unsafe fn`, over `grid`, before the kernel runs: what kernite proved of it
/// holds only for a launch that does. `params` are the names of its parameters, and `values` the number each is, or how
/// many elements the memory lent for it has, as the launcher's Rust writes them. The grid must run its threads along x
/// alone, fewer than 2^32 of them, so that `ctaid.x * ntid.x + tid.x` numbers each apart in 32 bits; and each of
/// `requires`, which the kernel's proof leaves for the launch, must hold. An error says which does not, as the Rust of
/// the launcher writes it, with the value of each number it is over, and no kernel runs.
pub fn holds(
    grid: &Grid,
    kernel: &str,
    params: &[&str],
    values: &[i128],
    requires: &[Requirement],
) -> Result<(), Error> {
    let refused = |message: String| Error::new("cuLaunchKernel", message);
    let (blocks, threads) = (grid.blocks.map(i128::from), grid.threads.map(i128::from));
    for (size, along, what) in [
        (blocks[1], "y", "blocks"),
        (blocks[2], "z", "blocks"),
        (threads[1], "y", "threads in each block"),
        (threads[2], "z", "threads in each block"),
    ] {
        if size != 1 {
            return Err(refused(format!(
                "`{kernel}` is safe on a grid that runs along x alone, for kernite proves that its threads reach \
                 memory apart by their index along x: the grid has {size} {what} along {along}"
            )));
        }
    }
    let total = blocks[0] * threads[0];
    if total >> 32 != 0 {
        return Err(refused(format!(
            "`{kernel}` numbers its threads in 32 bits, so its grid has fewer than 4294967296 of them: the grid has \
             {} blocks of {} threads, {total} in all",
            blocks[0], threads[0]
        )));
    }
    let launch = Launch { grid, params, values };
    for r in requires {
        if launch.sum(r).is_some_and(|sum| sum >= 0) {
            continue;
        }
        let param = |i: usize| params.get(i).copied().unwrap_or("?");
        let so = match r.why {
            Why::Fits => "no number it works out wraps".to_string(),
            Why::Inside(i) => format!("each element it reaches of `{}` is inside it", param(i)),
            Why::Apart(i) => {
                format!("no two threads write one element of `{0}`, and none reads one that another writes", param(i))
            }
        };
        return Err(refused(format!(
            "`{kernel}` is safe on this launch only if `{}`, so that {so}, and that does not hold: {}",
            launch.written(r),
            launch.values(r)
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const X: Quantity = Quantity::Len(1);
    const N: Quantity = Quantity::Param(3);
    const NAMES: [&str; 4] = ["a", "x", "y", "n"];

    /// `x.len() >= n`, as the proof of `saxpy` leaves it.
    const INSIDE: Requirement = Requirement { terms: &[(1, &[X]), (-1, &[N])], why: Why::Inside(1) };

    fn saxpy(grid: Grid, x: i128, n: i128) -> Result<(), Error> {
        holds(&grid, "saxpy", &NAMES, &[0, x, x, n], &[INSIDE])
    }

    #[test]
    fn a_launch_that_holds_what_the_kernel_needs_runs() {
        assert_eq!(saxpy(Grid::new(4, 256), 1000, 1000), Ok(()));
        assert_eq!(saxpy(Grid::new(4, 256), 1001, 1000), Ok(()));
    }

    #[test]
    fn a_condition_that_does_not_hold_is_said_in_rust_with_its_values() {
        let e = saxpy(Grid::new(4, 256), 999, 1000).err();
        assert_eq!(
            e.map(|e| e.to_string()),
            Some(
                "cuLaunchKernel: `saxpy` is safe on this launch only if `x.len() >= n`, so that each element it \
                 reaches of `x` is inside it, and that does not hold: `x.len()` is 999, `n` is 1000"
                    .to_string()
            )
        );
    }

    #[test]
    fn a_grid_past_x_is_refused_before_the_conditions() {
        let along_x = "`saxpy` is safe on a grid that runs along x alone, for kernite proves that its threads reach \
                       memory apart by their index along x: the grid has";
        let refused = |change: fn(&mut Grid), said: &str| {
            let mut grid = Grid::new(4, 256);
            change(&mut grid);
            // The memory is too short as well, and the grid is what is said.
            let e = saxpy(grid, 0, 1000).err().map(|e| e.message);
            assert_eq!(e, Some(format!("{along_x} {said}")));
        };
        refused(|g| g.blocks[1] = 2, "2 blocks along y");
        refused(|g| g.blocks[2] = 3, "3 blocks along z");
        refused(|g| g.threads[1] = 4, "4 threads in each block along y");
        refused(|g| g.threads[2] = 5, "5 threads in each block along z");
    }

    #[test]
    fn a_grid_of_2_to_the_32_threads_is_refused() {
        assert_eq!(saxpy(Grid::new(16_777_215, 256), 0, 0), Ok(()));
        let e = saxpy(Grid::new(16_777_216, 256), 0, 0).err().map(|e| e.message);
        assert_eq!(
            e.as_deref(),
            Some(
                "`saxpy` numbers its threads in 32 bits, so its grid has fewer than 4294967296 of them: the grid has \
                 16777216 blocks of 256 threads, 4294967296 in all"
            )
        );
    }

    #[test]
    fn a_term_with_a_factor_and_a_constant_is_written_with_the_rest_to_the_right() {
        let r = Requirement {
            terms: &[(4, &[X]), (-1, &[Quantity::Blocks, Quantity::Threads]), (-4, &[]), (2, &[])],
            why: Why::Fits,
        };
        let grid = Grid::new(3, 8);
        let launch = Launch { grid: &grid, params: &NAMES, values: &[0, 10, 0, 0] };
        assert_eq!(launch.written(&r), "4 * x.len() + 2 >= grid.blocks[0] * grid.threads[0] + 4");
        assert_eq!(launch.sum(&r), Some(4 * 10 - 24 - 4 + 2));
        assert_eq!(launch.values(&r), "`x.len()` is 10, `grid.blocks[0]` is 3, `grid.threads[0]` is 8");
    }

    /// A range of the elements of 16 is where it starts and how many it takes, whichever way it is written, as a
    /// slice's is: `3..=2` is empty, at 3. One that ends past the end or before it starts is none, and so is a bound
    /// that is past what a number holds.
    #[test]
    #[expect(clippy::reversed_empty_ranges, reason = "a range back to front is what the test refuses")]
    fn a_range_is_where_it_starts_and_how_many_it_takes_and_none_ends_past_the_end_or_before_the_start() {
        assert_eq!(bounds(3..10, 16), Some((3, 7)));
        assert_eq!(bounds(3.., 16), Some((3, 13)));
        assert_eq!(bounds(..5, 16), Some((0, 5)));
        assert_eq!(bounds(.., 16), Some((0, 16)));
        assert_eq!(bounds(3..=9, 16), Some((3, 7)));
        assert_eq!(bounds(3..=2, 16), Some((3, 0)));
        assert_eq!(bounds(..=15, 16), Some((0, 16)));
        assert_eq!(
            (bounds(2..2, 16), bounds(16.., 16), bounds(16..16, 16)),
            (Some((2, 0)), Some((16, 0)), Some((16, 0)))
        );
        assert_eq!(bounds((Bound::Excluded(2), Bound::Excluded(5)), 16), Some((3, 2)));
        for (what, got) in [
            ("5..3", bounds(5..3, 16)),
            ("0..17", bounds(0..17, 16)),
            ("17..", bounds(17.., 16)),
            ("..=16", bounds(..=16, 16)),
            ("5..=3", bounds(5..=3, 16)),
            ("usize::MAX..", bounds(usize::MAX.., 16)),
            ("..=usize::MAX", bounds(..=usize::MAX, 16)),
        ] {
            assert_eq!(got, None, "{what}");
        }
        assert_eq!(bounds((Bound::Excluded(usize::MAX), Bound::Unbounded), 16), None);
        assert_eq!(bounds(0..0, 0), Some((0, 0)));
        assert_eq!(bounds(1..1, 0), None);
    }

    #[test]
    fn a_factor_of_the_least_number_128_bits_hold_is_written_as_the_size_it_has() {
        let r = Requirement { terms: &[(1, &[]), (i128::MIN, &[X])], why: Why::Fits };
        let grid = Grid::new(1, 1);
        let e = holds(&grid, "k", &NAMES, &[0, 4, 0, 0], &[r]).err().map(|e| e.message);
        assert_eq!(
            e.as_deref(),
            Some(
                "`k` is safe on this launch only if `1 >= 170141183460469231731687303715884105728 * x.len()`, so that \
                 no number it works out wraps, and that does not hold: `x.len()` is 4"
            )
        );
    }

    #[test]
    fn a_sum_past_128_bits_does_not_hold() {
        let big = Requirement { terms: &[(i128::MAX, &[X, X])], why: Why::Fits };
        let grid = Grid::new(1, 1);
        assert_eq!(Launch { grid: &grid, params: &NAMES, values: &[0, 4, 0, 0] }.sum(&big), None);
        assert!(holds(&grid, "k", &NAMES, &[0, 4, 0, 0], &[big]).is_err());
    }
}
