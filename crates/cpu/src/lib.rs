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
//! The threads a launch needs besides the caller's are kept for the caller's launches after it:
//! the first starts them, and each next hands the kernel to threads that run already. A kept
//! thread with no kernel to run spins for [`SPIN`], then sleeps until a launch wakes it, and it
//! ends when the caller's thread does. A process that forks and does not `exec` has none of
//! them in the child, where a launch of more than one thread waits for ever.
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

use std::cell::RefCell;
use std::fmt;
use std::hint::spin_loop;
use std::io;
use std::num::NonZero;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::AtomicU64;
use std::sync::mpsc::{self, Receiver, RecvError, Sender, TryRecvError};
use std::sync::{Arc, LazyLock};
use std::thread;
use std::time::{Duration, Instant};

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
#[derive(Debug, Default)]
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
/// [`std::thread::available_parallelism`] says the first time it is asked, since a thread that
/// waits at `grid.sync()` spins, and needs a core while it does.
pub fn most() -> Result<usize, Error> {
    static MOST: LazyLock<Result<usize, Error>> = LazyLock::new(|| {
        thread::available_parallelism()
            .map(NonZero::get)
            .map_err(|e| Error { message: format!("the machine does not say how many threads it runs at once: {e}") })
    });
    MOST.clone()
}

/// How long a thread of a grid that waits spins before it sleeps: a kept thread for the next
/// launch, and the caller for the others to end. A spinning thread sees what it waits for at
/// once; a sleeping one is woken by the system, which costs the launch some microseconds, and
/// more once the machine has been idle.
pub const SPIN: Duration = Duration::from_micros(200);

/// Runs `kernel` on each thread of `grid`, given its place there, and returns once all have
/// ended, everything they wrote seen by the caller. The caller's thread is the first; the
/// others are threads kept from the caller's launches before, and those the grid needs that
/// none has started are started now, with the stack Rust gives a thread, 2 MB unless
/// `RUST_MIN_STACK` says otherwise. Each is the same thread of the grid in every launch, and
/// named by it: `grid thread 3` is the fourth. None runs the kernel until all exist, so that
/// one that cannot start leaves none waiting for it at `grid.sync()`: the launch is an error
/// then, as it is for a grid of no threads, or of more than [`most`]. Launches from threads at
/// once run at once, each on threads of its own.
///
/// A thread of the grid that panics ends the launch, when all have, with that panic in the
/// caller; the threads are kept as they are.
pub fn launch(grid: Grid, kernel: impl Fn(&Place) + Send + Sync + 'static) -> Result<(), Error> {
    let most = most()?;
    let threads = grid.threads;
    if threads == 0 {
        return Err(Error { message: "a grid has one thread at least, and this has none".into() });
    }
    if threads > most {
        return Err(Error { message: format!("the grid has {threads} threads, and the machine runs {most} at once") });
    }
    let mut team = KEPT.take();
    while team.len() < threads - 1 {
        let started = Worker::start(team.len() as u64 + 1);
        team.push(started.map_err(|e| Error { message: format!("a thread of the grid did not start: {e}") })?);
    }
    let launch: Arc<Launch> = Arc::new(Launch { threads: threads as u64, barrier: Barrier::default(), kernel });
    let others = &team[..threads - 1];
    for worker in others {
        worker.jobs.send(Arc::clone(&launch)).expect(GONE);
    }
    let mut ended = launch.run(0);
    for worker in others {
        ended = ended.and(wait(&worker.done).expect(GONE));
    }
    KEPT.set(team);
    match ended {
        Ok(()) => Ok(()),
        Err(payload) => panic::resume_unwind(payload),
    }
}

/// What a launch says if a thread of its grid has ended, which a panic in the kernel does not
/// make it do: the thread catches that.
const GONE: &str = "a thread of the grid ended before its launch did";

/// What every thread of a launch shares: the kernel, and what its places hold.
struct Launch<K: ?Sized = dyn Fn(&Place) + Send + Sync> {
    threads: u64,
    barrier: Barrier,
    /// The last field, for a launch of any kernel to be one of `dyn Fn`.
    kernel: K,
}

impl Launch {
    /// Runs the kernel as thread `t`, and says how it ended.
    fn run(&self, t: u64) -> thread::Result<()> {
        let place = Place { thread: t, threads: self.threads, barrier: &self.barrier };
        panic::catch_unwind(AssertUnwindSafe(|| (self.kernel)(&place)))
    }
}

/// A thread kept for launches, and the two ends it is spoken to by: the launches to run its
/// place in, and how each ended.
struct Worker {
    jobs: Sender<Arc<Launch>>,
    done: Receiver<thread::Result<()>>,
}

impl Worker {
    /// Starts the thread that runs thread `t` of each launch it is given and says how each
    /// ended, until the `Worker` is dropped. It is named by its place, `grid thread 3`, as a
    /// profiler or a debugger lists it.
    fn start(t: u64) -> io::Result<Worker> {
        let (jobs, inbox) = mpsc::channel::<Arc<Launch>>();
        let (outbox, done) = mpsc::channel();
        thread::Builder::new().name(format!("grid thread {t}")).spawn(move || {
            while let Ok(launch) = wait(&inbox) {
                let ended = launch.run(t);
                // The launch ends when every thread has said, so this one lets go of it first.
                drop(launch);
                let _ = outbox.send(ended);
            }
        })?;
        Ok(Worker { jobs, done })
    }
}

thread_local! {
    /// The threads this thread's launches have started, which its next launches run on, the one
    /// at index `i` as thread `i + 1` of each grid. They end with this thread.
    static KEPT: RefCell<Vec<Worker>> = const { RefCell::new(Vec::new()) };
}

/// The next message from `from`, waited for by spinning for [`SPIN`], then asleep.
fn wait<T>(from: &Receiver<T>) -> Result<T, RecvError> {
    let start = Instant::now();
    while start.elapsed() < SPIN {
        match from.try_recv() {
            Err(TryRecvError::Empty) => spin_loop(),
            Ok(message) => return Ok(message),
            Err(TryRecvError::Disconnected) => return Err(RecvError),
        }
    }
    from.recv()
}

#[cfg(test)]
mod tests {
    use std::mem::{offset_of, size_of};
    use std::panic;
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::Duration;

    use super::{Barrier, Grid, Place, launch, most};

    #[test]
    fn each_thread_runs_the_kernel_once_given_its_place() {
        for threads in 1..=most().unwrap().min(4) {
            let seen = Arc::new(Mutex::new(vec![]));
            let into = Arc::clone(&seen);
            launch(Grid::new(threads), move |place| into.lock().unwrap().push((place.thread(), place.threads())))
                .unwrap();
            let mut seen = seen.lock().unwrap().clone();
            seen.sort();
            let want: Vec<(u64, u64)> = (0..threads as u64).map(|t| (t, threads as u64)).collect();
            assert_eq!(seen, want);
        }
    }

    /// A thread of the grid is named by its place, as a profiler or a debugger lists threads,
    /// in the launches after the first as in it; the caller's keeps its own name.
    #[test]
    fn a_thread_of_the_grid_is_named_by_its_place() {
        let threads = most().unwrap().min(4);
        let caller = thread::current().name().map(String::from);
        for _ in 0..3 {
            let named = Arc::new(Mutex::new(vec![]));
            let into = Arc::clone(&named);
            launch(Grid::new(threads), move |place| {
                into.lock().unwrap().push((place.thread(), thread::current().name().map(String::from)));
            })
            .unwrap();
            let mut named = named.lock().unwrap().clone();
            named.sort();
            let want: Vec<(u64, Option<String>)> = (0..threads as u64)
                .map(|t| (t, if t == 0 { caller.clone() } else { Some(format!("grid thread {t}")) }))
                .collect();
            assert_eq!(named, want);
        }
    }

    /// The caller's thread is the grid's first, and the launch returns once every thread has
    /// ended: the others take a while, and each has run when it does.
    #[test]
    fn the_callers_thread_is_the_first_and_the_launch_returns_when_all_have_ended() {
        let caller = thread::current().id();
        let threads = most().unwrap().min(4);
        let ran = Arc::new(Mutex::new(vec![]));
        let into = Arc::clone(&ran);
        launch(Grid::new(threads), move |place| {
            if place.thread() > 0 {
                thread::sleep(Duration::from_millis(20));
            }
            into.lock().unwrap().push((place.thread(), thread::current().id() == caller));
        })
        .unwrap();
        let mut ran = ran.lock().unwrap().clone();
        ran.sort();
        let want: Vec<(u64, bool)> = (0..threads as u64).map(|t| (t, t == 0)).collect();
        assert_eq!(ran, want);
    }

    #[test]
    fn a_grid_has_one_thread_to_as_many_as_the_machine_runs_at_once() {
        let most = most().unwrap();
        let never = |_: &Place| panic!("no thread runs");
        assert_eq!(
            launch(Grid::new(0), never).unwrap_err().message,
            "a grid has one thread at least, and this has none"
        );
        assert_eq!(
            launch(Grid::new(most + 1), never).unwrap_err().message,
            format!("the grid has {} threads, and the machine runs {most} at once", most + 1)
        );
    }

    /// Two launches at once, from two threads, each have all their threads running: every
    /// thread of both waits for all of both, which none could if a launch waited for the other
    /// to end, or ran on the other's threads.
    #[test]
    fn launches_from_two_threads_at_once_run_at_once_each_on_threads_of_its_own() {
        let threads = most().unwrap().min(3);
        let meet = Arc::new(std::sync::Barrier::new(2 * threads));
        let ids = Arc::new(Mutex::new(vec![]));
        let callers: Vec<_> = (0..2)
            .map(|_| {
                let (meet, ids) = (Arc::clone(&meet), Arc::clone(&ids));
                thread::spawn(move || {
                    launch(Grid::new(threads), move |_| {
                        meet.wait();
                        ids.lock().unwrap().push(thread::current().id());
                    })
                    .unwrap();
                })
            })
            .collect();
        for caller in callers {
            caller.join().unwrap();
        }
        let ids = ids.lock().unwrap();
        assert_eq!(ids.len(), 2 * threads);
        assert!(ids.iter().enumerate().all(|(i, id)| !ids[i + 1..].contains(id)), "{ids:?}");
    }

    /// A thread of the grid that panics ends the launch, once all have, as a panic in the
    /// caller with the payload it was given; the threads run the next launch as before.
    #[test]
    fn a_panic_in_a_thread_of_the_grid_ends_the_launch_in_the_caller() {
        let threads = most().unwrap().min(3);
        let last = threads as u64 - 1;
        // `resume_unwind` panics without the message a hook would print.
        let launched = panic::catch_unwind(|| {
            launch(Grid::new(threads), move |place| {
                if place.thread() == last {
                    panic::resume_unwind(Box::new("the last thread panics"));
                }
            })
        });
        assert_eq!(*launched.unwrap_err().downcast::<&str>().unwrap(), "the last thread panics");
        let ran = Arc::new(Mutex::new(vec![]));
        let into = Arc::clone(&ran);
        launch(Grid::new(threads), move |place| into.lock().unwrap().push(place.thread())).unwrap();
        let mut ran = ran.lock().unwrap().clone();
        ran.sort();
        assert_eq!(ran, (0..threads as u64).collect::<Vec<_>>());
    }

    /// The words of a place and of the barrier where kernite-x86 reads them.
    #[test]
    fn a_place_is_laid_out_as_a_kernel_reads_it() {
        assert_eq!((offset_of!(Place, thread), offset_of!(Place, threads), offset_of!(Place, barrier)), (0, 8, 16));
        assert_eq!(size_of::<Place>(), 24);
        assert_eq!((offset_of!(Barrier, arrived), offset_of!(Barrier, generation)), (0, 8));
    }
}
