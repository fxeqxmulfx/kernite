//! A launch of a kernel that is no `unsafe fn`: the memory it is lent, and what the host checks of the launch before
//! the kernel runs.

use super::{Buffer, Cuda, Error, Grid, Module, Plain, Ptr, sealed};

/// Memory a kernel that is no `unsafe fn` reads, which its launcher borrows for the call: a [`Buffer`], a [`View`] of
/// a buffer, or [`Unified`](super::Unified) memory of `'static`: managed or page-locked memory, or a slice the host
/// lends for good. [`Cuda::copy`] takes it too, but for a slice that [`Cuda::lend`](super::Cuda::lend) lent, which it
/// refuses. The kernel and the copy are trusted with it until they are done, and the host's calls that reach the memory
/// wait for them first, as `read`, `write`, `host` and dropping the memory do; `atomics`, `host_now` and `host_now_mut`
/// wait for none, and the last two are `unsafe` for it. A slice lent for less is not, as a
/// `mem::forget` of what [`Cuda::lend`](super::Cuda::lend) gave would give it back to the host with no wait.
pub trait Memory<T: Plain>: sealed::Sealed {
    /// Its address and how many elements it has, as a kernel of `module` is lent them. It is an error if it is memory
    /// of another GPU than the module's, or managed memory that the module's GPU cannot reach while the host does.
    fn lend(&self, module: &Module) -> Result<(Ptr<T>, usize), Error>;

    /// Its address and how many elements it has, as a copy on the GPU of `cuda` reaches them. It is an error if it is
    /// memory of another GPU than `cuda`'s, a slice that [`Cuda::lend`](super::Cuda::lend) lent, or managed memory that
    /// `cuda`'s GPU cannot reach while the host does.
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

/// The elements of a [`Buffer`] from one on, borrowed: what [`Buffer::tail`] gives, for a kernel to be lent the part of
/// the buffer from there on.
pub struct View<'a, T: Plain> {
    of: &'a Buffer<T>,
    /// The address of the first element, and how many there are.
    ptr: Ptr<T>,
    len: usize,
}

impl<'a, T: Plain> View<'a, T> {
    /// The elements of `of` from `from` on; `None` if `from` is past its end.
    pub(super) fn new(of: &'a Buffer<T>, from: usize) -> Option<View<'a, T>> {
        Some(View { of, ptr: of.at(from)?, len: of.len - from })
    }
}

impl<T: Plain> sealed::Sealed for View<'_, T> {}

impl<T: Plain> Memory<T> for View<'_, T> {
    fn lend(&self, module: &Module) -> Result<(Ptr<T>, usize), Error> {
        module.reaches(&self.of.driver)?;
        Ok((self.ptr, self.len))
    }

    fn reach(&self, cuda: &Cuda) -> Result<(Ptr<T>, usize), Error> {
        cuda.reaches(&self.of.driver)?;
        Ok((self.ptr, self.len))
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

    /// The term of `factor` times `quantities`, as Rust writes it.
    fn term(&self, factor: i128, quantities: &[Quantity]) -> String {
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
                .map(|(factor, quantities)| self.term(factor.abs(), quantities))
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

    #[test]
    fn a_sum_past_128_bits_does_not_hold() {
        let big = Requirement { terms: &[(i128::MAX, &[X, X])], why: Why::Fits };
        let grid = Grid::new(1, 1);
        assert_eq!(Launch { grid: &grid, params: &NAMES, values: &[0, 4, 0, 0] }.sum(&big), None);
        assert!(holds(&grid, "k", &NAMES, &[0, 4, 0, 0], &[big]).is_err());
    }
}
