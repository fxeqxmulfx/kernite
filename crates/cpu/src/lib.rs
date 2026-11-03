//! What x86 kernels need of the process beyond their code, for what `#[kernite_macros::kernels]`
//! writes: the threads a cooperative kernel runs on, and the memory of a module's.
//!
//! - [`launch`] runs a kernel on each thread of a [`Grid`], given its [`Place`] there, which
//!   holds the barrier its `grid.sync()` waits at.
//! - [`Global`] is a static of the module's, which its kernels name and the host reads and
//!   writes: `kernels::COUNTS.read()`.
//!
//! ```ignore
//! #[kernite_macros::kernels]
//! mod cpu {
//!     /// `x[t] = t` for each thread `t` of the grid.
//!     #[cooperative]
//!     pub unsafe extern "sysv64" fn fill(x: *mut u64) {
//!         let t: gpr;
//!         mov(t, grid.thread);
//!         mov([x + t * 8], t);
//!     }
//! }
//!
//! let mut x = vec![0; kernite_cpu::most()?];
//! unsafe { cpu::fill(kernite_cpu::Grid::new(x.len()), x.as_mut_ptr())? };
//! ```
//!
//! A thread that waits at `grid.sync()` spins until the last comes, so each needs a core of
//! its own while it waits, as each block of a cooperative GPU kernel needs room on the GPU: a
//! grid has at most as many threads as the machine runs at once, [`most`].
//!
//! A static of a module of x86 kernels is a [`Global`] of its elements, which the kernels reach
//! at a displacement from `rip`, and which keeps what they write from one call to the next:
//!
//! ```ignore
//! #[kernite_macros::kernels]
//! mod cpu {
//!     /// How many times `bump` has run.
//!     pub static RUNS: global<u64>;
//!
//!     pub unsafe extern "sysv64" fn bump() {
//!         let n: gpr;
//!         mov(n, [RUNS]);
//!         inc(n);
//!         mov([RUNS], n);
//!     }
//! }
//!
//! unsafe { cpu::bump() };
//! assert_eq!(cpu::RUNS.read(), [1]);
//! ```
//!
//! The crate depends on no other. kernite-x86 writes the kernels, which read a place as [`Place`]
//! lays it out, and the macro writes the launchers, which call [`launch`] and, in each thread,
//! the kernel, and the statics. Nothing here is `unsafe`: calling the kernel is, and the launcher
//! does that.

mod global;

#[doc(hidden)]
pub use global::Stub;
pub use global::{Align, Alignment, Element, Global};

use std::fmt;
use std::num::NonZero;
use std::sync::atomic::AtomicU64;
use std::sync::{Condvar, Mutex, PoisonError};
use std::thread;

/// How many threads a kernel runs on, each all of it, at once.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Grid {
    /// The threads: one at least, and at most [`most`].
    pub threads: usize,
}

impl Grid {
    /// A grid of `threads` threads.
    pub fn new(threads: usize) -> Grid {
        Grid { threads }
    }
}

/// What a thread of a grid is given, in the register after the kernel's parameters: its index,
/// how many threads there are, and the address of the grid's barrier, a word each, in that
/// order, as kernite-x86 reads them. The barrier is two words: how many threads have come to it
/// since all last did, and how many times all have.
#[repr(C)]
#[derive(Debug)]
pub struct Place {
    thread: u64,
    threads: u64,
    barrier: *const Barrier,
}

impl Place {
    /// The thread's index in the grid, from 0, as a kernel reads it: `grid.thread`.
    pub fn thread(&self) -> u64 {
        self.thread
    }

    /// How many threads the grid has, as a kernel reads it: `grid.threads`.
    pub fn threads(&self) -> u64 {
        self.threads
    }
}

/// What `grid.sync()` waits at: how many threads have come to it since all last did, and how
/// many times all have, its generation, which a thread that has come waits to see change. A
/// thread counts itself in with `lock xadd`, and the last to come puts the count back to 0,
/// then moves the generation on. It has a cache line of its own, which only the grid's threads
/// touch.
#[repr(C, align(64))]
#[derive(Debug)]
struct Barrier {
    arrived: AtomicU64,
    generation: AtomicU64,
}

/// Why a grid did not run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    /// What went wrong.
    pub message: String,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for Error {}

/// How many threads a grid has at most: as many as the machine runs at once, as
/// [`std::thread::available_parallelism`] says, since a thread that waits at `grid.sync()`
/// spins, and needs a core while it does.
pub fn most() -> Result<usize, Error> {
    thread::available_parallelism()
        .map(NonZero::get)
        .map_err(|e| Error { message: format!("the machine does not say how many threads it runs at once: {e}") })
}

/// Runs `kernel` on each thread of `grid`, given its place there, and returns once all have
/// ended, everything they wrote seen by the caller. The caller's thread is the first; the
/// others start for the launch, with the stack Rust gives a thread, 2 MB unless
/// `RUST_MIN_STACK` says otherwise. None runs the kernel until all have started, so that one
/// that cannot start leaves none waiting for it at `grid.sync()`: the launch is an error then,
/// as it is for a grid of no threads, or of more than [`most`].
pub fn launch(grid: Grid, kernel: &(dyn Fn(&Place) + Sync)) -> Result<(), Error> {
    let most = most()?;
    let threads = grid.threads;
    if threads == 0 {
        return Err(Error { message: "a grid has one thread at least, and this has none".into() });
    }
    if threads > most {
        return Err(Error { message: format!("the grid has {threads} threads, and the machine runs {most} at once") });
    }
    let barrier = Barrier { arrived: AtomicU64::new(0), generation: AtomicU64::new(0) };
    let gate = Gate::default();
    let count = threads as u64;
    thread::scope(|s| {
        let mut failed = None;
        for t in 1..count {
            let (gate, barrier) = (&gate, &barrier);
            let started = thread::Builder::new().name(format!("grid thread {t}")).spawn_scoped(s, move || {
                if gate.wait() {
                    kernel(&Place { thread: t, threads: count, barrier });
                }
            });
            if let Err(e) = started {
                failed = Some(e);
                break;
            }
        }
        gate.open(failed.is_none());
        match failed {
            Some(e) => Err(Error { message: format!("a thread of the grid did not start: {e}") }),
            None => {
                kernel(&Place { thread: 0, threads: count, barrier: &barrier });
                Ok(())
            }
        }
    })
}

/// Where the threads a launch starts wait until the caller knows whether all have: then they
/// run the kernel, or end.
#[derive(Debug, Default)]
struct Gate {
    open: Mutex<Option<bool>>,
    changed: Condvar,
}

impl Gate {
    /// Whether to run the kernel, once the caller says.
    fn wait(&self) -> bool {
        let open = self.open.lock().unwrap_or_else(PoisonError::into_inner);
        let open = self.changed.wait_while(open, |o| o.is_none()).unwrap_or_else(PoisonError::into_inner);
        *open == Some(true)
    }

    /// Says whether to run the kernel, `run`, to every thread that waits or will.
    fn open(&self, run: bool) {
        *self.open.lock().unwrap_or_else(PoisonError::into_inner) = Some(run);
        self.changed.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use std::mem::{offset_of, size_of};
    use std::sync::Mutex;
    use std::thread;
    use std::time::Duration;

    use super::{Barrier, Grid, Place, launch, most};

    #[test]
    fn each_thread_runs_the_kernel_once_given_its_place() {
        for threads in 1..=most().unwrap().min(4) {
            let seen = Mutex::new(vec![]);
            launch(Grid::new(threads), &|place| seen.lock().unwrap().push((place.thread(), place.threads()))).unwrap();
            let mut seen = seen.into_inner().unwrap();
            seen.sort();
            let want: Vec<(u64, u64)> = (0..threads as u64).map(|t| (t, threads as u64)).collect();
            assert_eq!(seen, want);
        }
    }

    /// The caller's thread is the grid's first, and the launch returns once every thread has
    /// ended: the others take a while, and each has run when it does.
    #[test]
    fn the_callers_thread_is_the_first_and_the_launch_returns_when_all_have_ended() {
        let caller = thread::current().id();
        let threads = most().unwrap().min(4);
        let ran = Mutex::new(vec![]);
        launch(Grid::new(threads), &|place| {
            if place.thread() > 0 {
                thread::sleep(Duration::from_millis(20));
            }
            ran.lock().unwrap().push((place.thread(), thread::current().id() == caller));
        })
        .unwrap();
        let mut ran = ran.into_inner().unwrap();
        ran.sort();
        let want: Vec<(u64, bool)> = (0..threads as u64).map(|t| (t, t == 0)).collect();
        assert_eq!(ran, want);
    }

    #[test]
    fn a_grid_has_one_thread_to_as_many_as_the_machine_runs_at_once() {
        let most = most().unwrap();
        let never = |_: &Place| panic!("no thread runs");
        assert_eq!(
            launch(Grid::new(0), &never).unwrap_err().message,
            "a grid has one thread at least, and this has none"
        );
        assert_eq!(
            launch(Grid::new(most + 1), &never).unwrap_err().message,
            format!("the grid has {} threads, and the machine runs {most} at once", most + 1)
        );
    }

    /// The words of a place and of the barrier where kernite-x86 reads them.
    #[test]
    fn a_place_is_laid_out_as_a_kernel_reads_it() {
        assert_eq!((offset_of!(Place, thread), offset_of!(Place, threads), offset_of!(Place, barrier)), (0, 8, 16));
        assert_eq!(size_of::<Place>(), 24);
        assert_eq!((offset_of!(Barrier, arrived), offset_of!(Barrier, generation)), (0, 8));
    }
}
