//! What x86 kernels need of the process beyond their code, for what `#[kernite_macros::kernels]`
//! writes: the threads a cooperative kernel runs on and the CPUs they run on, and the memory of a
//! module's.
//!
//! - [`launch`] runs a kernel on each thread of a [`Grid`], given its [`Place`] there, which
//!   holds the barrier its `grid.sync()` waits at.
//! - [`launch_blocks`] runs a kernel once for each of some blocks, on the threads of a [`Grid`]:
//!   each thread takes the next block when it ends one, and no block waits for another.
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
//! grid has at most as many threads as the process runs at once, [`most`]. That is the CPUs
//! [`cpus`] lists, or fewer where a cgroup limits the process to fewer.
//!
//! The threads a launch needs besides the caller's are kept for the caller's launches after it:
//! the first starts them, and each next hands the kernel to threads that run already. A kept
//! thread with no kernel to run spins for as long as the grid says, [`SPIN`] unless [`Grid::spin`]
//! says otherwise, then sleeps until a launch wakes it, and it ends when the caller's thread does.
//! A process that forks and does not `exec` has none of them in the child, where a launch of more
//! than one thread waits for ever.
//!
//! Where the threads run is the system's to say, unless the caller places them. A
//! [`Grid::new`] runs each thread on any CPU the caller's thread may run on, and the system moves
//! it from one to another as it likes. A [`Grid::on`] puts thread `t` on CPU `cpus[t]`, and on no
//! other, in every launch. [`cpus`] lists the CPUs to choose from: those the caller's thread may
//! run on, which `numactl --physcpubind` and `taskset` set for a process when they start it.
//!
//! The caller's thread is the grid's first. The first launch that places it puts it on `cpus[0]`,
//! and it stays there, as the grid's other threads stay where they are placed, until a launch from
//! it places it elsewhere, or places none, which puts it and every thread kept for it back on the
//! CPUs it could run on. A thread it starts meanwhile starts on that one CPU, as a thread starts on
//! the CPUs of its creator, and [`cpus`] still lists the CPUs it could run on before. If the
//! program moves the caller's thread itself in between, kernite does not know. A launch makes one
//! call to the system for each thread it moves, and none when none moves. A thread bound to one
//! CPU cannot run while another spins there, so a launch that moves threads first ends the spin of
//! each kept thread on a CPU a thread moves to, and of each the system places, however long it was
//! to last: it sleeps until a launch uses it. A launch that moves none ends no spin.
//!
//! A page of memory is on the NUMA node of the CPU that first writes it, so a grid's threads are
//! near their memory when each writes its part of a buffer first. The crate places no memory.
//!
//! A static of a module of x86 kernels is a [`Global`] of its elements, which the kernels reach
//! at a displacement from `rip`, or by its address where that is past `rip`'s reach, and which keeps what they write
//! from one call to the next:
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
//! the kernel, and the statics. The only `unsafe` here is in `affinity`, which reads and sets the
//! CPUs a thread may run on through the C library, as Rust has no call for that, and in the one function of `Global`
//! that reads its elements, which Rust does not check one at a time, for that takes it 0.3 seconds for each MiB of a
//! static. Each block says why it holds. Calling the kernel is `unsafe` too, and the launcher does that.

// The module the crate's lints let have `unsafe` in it; `Global` has one function that does.
#[allow(unsafe_code)]
mod affinity;
mod global;

use affinity::Cpus;

#[doc(hidden)]
pub use global::Stub;
pub use global::{Align, Alignment, Element, Global};

use std::any::Any;
use std::cell::{Cell, RefCell};
use std::fmt;
use std::hint::spin_loop;
use std::io;
use std::num::NonZero;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{
    AtomicU64,
    Ordering::{AcqRel, Acquire, Relaxed},
};
use std::sync::mpsc::{self, Receiver, RecvError, Sender, SyncSender, TryRecvError, TrySendError};
use std::sync::{Arc, LazyLock, Mutex, PoisonError};
use std::thread::{self, JoinHandle, Thread};
use std::time::{Duration, Instant};

/// How many threads a kernel runs on, each all of it, at once, where they run: on any CPU the
/// caller's thread may run on, as the system likes, or on the CPUs the caller gives; and how long
/// they wait spinning before they sleep, [`SPIN`] unless [`Grid::spin`] says otherwise.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Grid<'a> {
    /// The threads: one at least, and at most [`most`].
    threads: usize,
    /// The CPU of each thread, if the caller places them.
    cpus: Option<&'a [usize]>,
    /// How long a thread of the grid that waits spins before it sleeps.
    spin: Duration,
}

impl Grid<'static> {
    /// A grid of `threads` threads, which the system runs on any CPU the caller's thread may run
    /// on, and moves from one to another as it likes.
    pub fn new(threads: usize) -> Grid<'static> {
        Grid { threads, cpus: None, spin: SPIN }
    }
}

impl<'a> Grid<'a> {
    /// A grid with a thread for each of `cpus`, thread `t` on CPU `cpus[t]` and on no other, in
    /// every launch. [`cpus`] lists the CPUs to choose from. A CPU the caller's thread may not run
    /// on is an error, and so is one given to two threads, since a thread that waits at
    /// `grid.sync()` needs a CPU of its own. Grids launched from threads at once are not checked
    /// against each other: two that place a thread on one CPU share it.
    ///
    /// The caller's thread is the first. The first launch that places it puts it on `cpus[0]`, and
    /// it stays there, as the grid's other threads stay where they are placed, until a launch from
    /// it places it elsewhere, or places none, which puts it and every thread kept for it back on
    /// the CPUs it could run on. A thread it starts meanwhile starts on that one CPU, as a thread
    /// starts on the CPUs of its creator. [`launch`] says the rest.
    pub fn on(cpus: &'a [usize]) -> Grid<'a> {
        Grid { threads: cpus.len(), cpus: Some(cpus), spin: SPIN }
    }

    /// The grid with its threads waiting spinning for `spin` before they sleep, not for [`SPIN`]: a
    /// kept thread for the next launch, and the caller for each of the others to end. A thread that
    /// spins sees what it waits for at once; one that sleeps is woken by the system, which costs a launch
    /// some microseconds, and some hundreds once the cores have gone idle. A kept thread uses its
    /// CPU for `spin` after each launch that uses it, so a grid whose launches come less than `spin`
    /// apart finds its threads spinning, and one of `Duration::ZERO` has them sleep at once. A
    /// launch that moves threads ends the spins it must at once, however long they were to last, as
    /// [`launch`] says. Kept threads of two callers that a placed grid each puts on one CPU share it
    /// while both spin, as any two threads on one CPU do: a launch waits for its thread to get the
    /// CPU, and not for the other's spin to end.
    pub fn spin(self, spin: Duration) -> Grid<'a> {
        Grid { spin, ..self }
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

/// Why a grid did not run, or why memory of a module's was not read or written.
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

/// How many threads a grid has at most: as many as the process runs at once, as
/// [`std::thread::available_parallelism`] says the first time it is asked, since a thread that
/// waits at `grid.sync()` spins, and needs a core while it does. That is the CPUs [`cpus`]
/// lists, or fewer where a cgroup limits the process to fewer.
pub fn most() -> Result<usize, Error> {
    static MOST: LazyLock<Result<usize, Error>> = LazyLock::new(|| {
        thread::available_parallelism().map(NonZero::get).map_err(|e| Error {
            message: format!("the system does not say how many threads the process runs at once: {e}"),
        })
    });
    MOST.clone()
}

/// The CPUs the calling thread may run on, by number, in order. `numactl --physcpubind` and
/// `taskset` set them for a process when they start it, and a program may narrow them for a
/// thread. These are the CPUs for [`Grid::on`] to give. A thread that a placed launch has left on
/// one CPU lists the CPUs it could run on before.
pub fn cpus() -> Result<Vec<usize>, Error> {
    let kept = CALLER.with_borrow(|caller| caller.as_ref().map(|placed| placed.home.iter().collect()));
    match kept {
        Some(cpus) => Ok(cpus),
        None => Ok(mask()?.iter().collect()),
    }
}

/// The CPUs the calling thread may run on, as a set.
fn mask() -> Result<Cpus, Error> {
    Cpus::current()
        .map_err(|e| Error { message: format!("the system does not say which CPUs the thread may run on: {e}") })
}

/// Whether `on` gives each thread of a grid a CPU of `home`, the CPUs the caller's thread may run
/// on, and no two threads the same.
fn checked(on: &[usize], home: &Cpus) -> Result<(), Error> {
    // The CPUs passed, for the check to take time as the grid's threads do, not as their square.
    let mut passed = Cpus::empty();
    for (t, &cpu) in on.iter().enumerate() {
        if !home.contains(cpu) {
            let message = format!(
                "thread {t} of the grid is placed on CPU {cpu}, and the caller's thread may run on CPUs {home}"
            );
            return Err(Error { message });
        }
        // `home` has room for `cpu`, so `passed` has: it changes unless a thread passed `cpu`.
        if !passed.insert(cpu) {
            let other = on.iter().take_while(|&&other| other != cpu).count();
            let message = format!(
                "threads {other} and {t} of the grid are both placed on CPU {cpu}, and a thread that waits needs a CPU \
                 of its own"
            );
            return Err(Error { message });
        }
    }
    Ok(())
}

/// How long a thread of a grid that waits spins before it sleeps, unless [`Grid::spin`] says
/// otherwise: a kept thread for the next launch, and the caller for each of the others to end. A spinning
/// thread sees what it waits for at once; a sleeping one is woken by the system, which costs the
/// launch some microseconds, and more once the machine has been idle.
pub const SPIN: Duration = Duration::from_micros(200);

/// Runs `kernel` on each thread of `grid`, given its place there, and returns once all have
/// ended, everything they wrote seen by the caller. The caller's thread is the first; the
/// others are threads kept from the caller's launches before, and those the grid needs that
/// none has started are started now, with the stack Rust gives a thread, 2 MB unless
/// `RUST_MIN_STACK` says otherwise. Each is the same thread of the grid in every launch, and
/// named by it: `grid thread 3` is the fourth. None runs the kernel until all exist and are
/// where the grid places them, so that one that cannot start, or move, leaves none waiting for
/// it at `grid.sync()`: the launch is an error then, as it is for a grid of no threads, or of
/// more than [`most`], or a placed one that gives a thread a CPU the caller's thread may not run
/// on, or two threads the same, for which no kernel runs and no thread moves.
///
/// A grid made by [`Grid::new`] leaves its threads to the system, on the CPUs the caller's thread
/// could run on, whatever the launches before it did: it puts the caller's thread back there, and
/// every kept thread a placed launch placed. A grid made by [`Grid::on`] has each thread on its
/// CPU, and the caller's thread stays on `cpus[0]` when the launch ends, as the kept threads stay
/// where they are placed, until a launch from it places it elsewhere, or places none. A thread
/// the caller starts meanwhile starts on that one CPU, as a thread starts on the CPUs of its
/// creator. If the program moves the caller's thread itself in between, kernite does not know:
/// the next launch that places it on the CPU kernite left it on does not move it. A launch moves
/// a thread only when it is not where the launch puts it, with one call to the system for each,
/// and makes none when none moves; the first that places the caller's thread, and the first after
/// one that placed none, also asks which CPUs the thread may run on, once.
///
/// A kept thread waits for the next launch spinning for the grid's [`Grid::spin`], [`SPIN`] if it
/// says none, and the caller waits for each of the others to end spinning for as long.
///
/// A thread bound to one CPU cannot run while another spins there, and a spin that no launch ends
/// lasts as long as its grid said. So a launch that moves threads first ends the spin of each kept
/// thread on a CPU a thread moves to, and of each the system places, which may spin on any,
/// however long it was to last: it sleeps until a launch uses it, and the launch does not wait for
/// the spin. So does a kept thread that takes a launch of blocks late, after this one has begun: it
/// sleeps once it has let go of it, and does not spin for it. A launch that moves none ends no
/// spin. A thread a placed launch starts sleeps until the launch has put it on its CPU, and waits
/// for launches there spinning, so that none spins on the caller's CPU, where the caller has to run
/// to start the next.
///
/// Launches from threads at once run at once, each on threads of its own. A thread of the grid
/// that panics ends the launch, when all have, with that panic in the caller; the threads are
/// kept as they are.
pub fn launch(grid: Grid<'_>, kernel: impl Fn(&Place) + Send + Sync + 'static) -> Result<(), Error> {
    on_team(grid, |others, threads, id| run_on(others, threads, id, grid.spin, kernel))
}

/// Runs `kernel` once for each block from 0 to `blocks`, on the threads of `grid`, and returns
/// once the last block has ended. Each thread, the caller's the first, takes the next block when
/// it ends one, so a thread that is late holds nothing back: the system may have put it aside for
/// another process, and the others take its blocks. A launch of one block or of none wakes no
/// thread, and one of `n` wakes `n - 1` at most, and none that has four launches waiting that it
/// has not taken: the others take its blocks too, and no launch queues for it for ever.
///
/// A thread that comes to a launch with no block left runs none, and lets go of the kernel. It may
/// be the last to hold it, so the kernel is dropped on the thread that lets go of it last, which
/// may be after the launch has returned. A panic in that drop is no launch's: the panic hook
/// prints it, as it prints that of a thread that no one joins, and the thread goes on to the next
/// launch.
///
/// No block waits for another: the kernel is given the index of its block, and no [`Place`], whose
/// barrier a `grid.sync()` waits at, and a block that waits for one taken later waits for ever, if
/// every thread is in one. A closure that takes the index runs, and one that takes a place does not
/// compile, as these two show, which differ in that alone:
///
/// ```
/// kernite_cpu::launch_blocks(kernite_cpu::Grid::new(1), 4, |block: u64| _ = block)?;
/// # Ok::<(), kernite_cpu::Error>(())
/// ```
///
/// ```compile_fail,E0631
/// kernite_cpu::launch_blocks(kernite_cpu::Grid::new(1), 4, |block: &kernite_cpu::Place| _ = block)?;
/// # Ok::<(), kernite_cpu::Error>(())
/// ```
///
/// A block that panics ends the launch, once every block taken has ended, as a panic in the caller
/// with the payload it was given, the first if more than one panicked; no block is taken after it.
/// The threads run the next launch as before.
///
/// The grid is as [`launch`] says: the same errors, in the same words, and no block runs when
/// there is one. The kept threads, their CPUs and their spin are as [`launch`] leaves them, and a
/// launch of blocks and a launch of threads each can follow the other on the same kept threads. A
/// launch has `u64::MAX / 2` blocks at most, and says so if it has more.
pub fn launch_blocks(grid: Grid<'_>, blocks: u64, kernel: impl Fn(u64) + Send + Sync + 'static) -> Result<(), Error> {
    if blocks > MOST_BLOCKS {
        let message = format!("a launch has {MOST_BLOCKS} blocks at most, and this has {blocks}");
        return Err(Error { message });
    }
    on_team(grid, |others, threads, id| run_blocks(others, threads, id, grid.spin, blocks, kernel))
}

/// How many launches wait in a kept thread's queue at most. A queue that holds any number takes a block of memory for
/// each 31 launches sent, and the queues of all the threads do so on the same launch. So the slowest launch of blocks
/// in a hundred, on twelve threads, costs 7.5 to 7.9 microseconds with such a queue, and 1.5 to 2.6 with one that holds
/// this many, which takes none. But this queue makes each launch of threads dearer: one of twelve threads by 0.2 to 0.4
/// microseconds.
const QUEUE: usize = 4;

/// How many blocks a launch has at most: half of what a count holds, so that the count of the tries to
/// take a block has room for the last try of each thread, which finds none.
const MOST_BLOCKS: u64 = u64::MAX / 2;

/// Puts the grid's threads where it says, starts those it needs that none has, and runs `run` on
/// the kept threads, the grid's others, how many threads the grid has, and the number of the
/// launch, which counts the launches that the caller's thread has made, a launch of blocks as one
/// of threads: it says how they ended, or that a thread is gone. A launch gives its caller
/// nothing, or the panic of a thread, which it passes on. A team that has a thread gone is no
/// more: the next launch starts another.
fn on_team(
    grid: Grid<'_>,
    run: impl FnOnce(&[Worker], usize, u64) -> Result<thread::Result<()>, Error>,
) -> Result<(), Error> {
    let most = most()?;
    let threads = grid.threads;
    if threads == 0 {
        return Err(Error { message: "a grid has one thread at least, and this has none".into() });
    }
    if threads > most {
        let message = format!(
            "the grid has {threads} threads, and the process runs {most} at once: the CPUs the caller's thread may run \
             on, or fewer where a cgroup limits the process to fewer"
        );
        return Err(Error { message });
    }
    let id = LAUNCHES.get() + 1;
    LAUNCHES.set(id);
    let mut team = KEPT.take();
    let ready = CALLER.with_borrow_mut(|caller| match grid.cpus {
        Some(on) => put_on(&mut team, caller, on, id),
        None => put_anywhere(&mut team, caller, threads - 1),
    });
    let ran = ready.and_then(|()| {
        let others = team.get(..threads - 1).ok_or_else(|| gone(()));
        others.and_then(|others| run(others, threads, id)).inspect_err(|_| team.clear())
    });
    KEPT.set(team);
    match ran? {
        Ok(()) => Ok(()),
        Err(payload) => panic::resume_unwind(payload),
    }
}

/// Runs `kernel` as thread 0 on the caller's thread, and as the others on `others`, and says how
/// they ended once all have; or that a thread of the grid is gone.
fn run_on(
    others: &[Worker],
    threads: usize,
    id: u64,
    spin: Duration,
    kernel: impl Fn(&Place) + Send + Sync + 'static,
) -> Result<thread::Result<()>, Error> {
    let launch: Arc<Launch> =
        Arc::new(Launch { id, threads: threads as u64, spin, answers: true, barrier: Barrier::default(), kernel });
    for worker in others {
        worker.jobs.send(Some(Arc::clone(&launch))).map_err(gone)?;
    }
    let mut ended = launch.run(0);
    for worker in others {
        ended = ended.and(wait(&worker.done, spin, || false).map_err(gone)?);
    }
    Ok(ended)
}

/// Runs `kernel` for each of `blocks` on the caller's thread and on those of `others` that take
/// it, and says how they ended once the last block has: nothing, or the first panic of a block;
/// or that a thread of the grid is gone. It wakes a thread for each block but the caller's, at
/// most, and none whose queue is full. None says how it ended, as the caller waits for the
/// blocks, and not for them.
fn run_blocks(
    others: &[Worker],
    threads: usize,
    id: u64,
    spin: Duration,
    blocks: u64,
    kernel: impl Fn(u64) + Send + Sync + 'static,
) -> Result<thread::Result<()>, Error> {
    let shared = Arc::new(Blocks {
        blocks,
        next: AtomicU64::new(0),
        ended: AtomicU64::new(0),
        panicked: Mutex::new(None),
        caller: thread::current(),
        kernel,
    });
    let taker = Arc::clone(&shared);
    let launch: Arc<Launch> = Arc::new(Launch {
        id,
        threads: threads as u64,
        spin,
        answers: false,
        barrier: Barrier::default(),
        kernel: move |_: &Place| taker.take(),
    });
    let mut wake = usize::try_from(blocks.saturating_sub(1)).unwrap_or(usize::MAX);
    for worker in others {
        if wake == 0 {
            break;
        }
        match worker.jobs.try_send(Some(Arc::clone(&launch))) {
            Ok(()) => wake -= 1,
            Err(TrySendError::Full(_)) => {}
            Err(TrySendError::Disconnected(_)) => return Err(gone(())),
        }
    }
    shared.take();
    let began = Instant::now();
    while shared.ended.load(Acquire) < blocks && began.elapsed() < spin {
        spin_loop();
    }
    // A wake that is not the last block's, which `park` may give, finds the blocks not ended.
    while shared.ended.load(Acquire) < blocks {
        thread::park();
    }
    let panicked = shared.panicked.lock().unwrap_or_else(PoisonError::into_inner).take();
    Ok(panicked.map_or(Ok(()), Err))
}

/// What the threads of a launch of blocks share: the kernel, and which blocks are taken and have
/// ended.
struct Blocks<K> {
    blocks: u64,
    /// The next block to take, which each try counts, and a thread that finds none tries no more:
    /// it is `blocks` or more when no block is left.
    next: AtomicU64,
    /// How many blocks have ended, or never will be taken, because one panicked.
    ended: AtomicU64,
    /// The first panic of a block.
    panicked: Mutex<Option<Box<dyn Any + Send>>>,
    /// The caller's thread, which waits for the last block.
    caller: Thread,
    kernel: K,
}

impl<K: Fn(u64)> Blocks<K> {
    /// Takes the next block and runs it, until none is left. The thread that ends the last block
    /// wakes the caller. A block that panics keeps its panic if it is the first, and no block is
    /// taken after it: the blocks not taken count as ended, once.
    fn take(&self) {
        loop {
            let block = self.next.fetch_add(1, Relaxed);
            if block >= self.blocks {
                return;
            }
            let mut ended = 1;
            if let Err(payload) = panic::catch_unwind(AssertUnwindSafe(|| (self.kernel)(block))) {
                self.panicked.lock().unwrap_or_else(PoisonError::into_inner).get_or_insert(payload);
                // The tries so far took a block each, but those that found none. The blocks left, which no
                // thread will take, count as ended: by this thread alone, as it is the first to say none is left.
                ended += self.blocks.saturating_sub(self.next.fetch_max(self.blocks, Relaxed));
            }
            if self.ended.fetch_add(ended, AcqRel) + ended == self.blocks {
                self.caller.unpark();
            }
        }
    }
}

/// What a launch says if a thread of its grid has ended, which a panic in the kernel does not
/// make it do: the thread catches that.
fn gone<E>(_: E) -> Error {
    Error { message: "a thread of the grid ended before its launch did".into() }
}

/// What every thread of a launch shares: the kernel, and what its places hold.
struct Launch<K: ?Sized = dyn Fn(&Place) + Send + Sync> {
    /// Which launch of the caller's thread this is, counted from 1, a launch of blocks as one of threads: a kept
    /// thread that takes it spins after it only if no launch with a later number has told the thread to rest.
    id: u64,
    threads: u64,
    /// How long a kept thread waits spinning for the next launch after this one.
    spin: Duration,
    /// Whether each kept thread says how it ended, for the caller to wait for it: a launch waits for every thread, and
    /// a launch of blocks for its blocks.
    answers: bool,
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

/// Where a placed launch has put the caller's thread, which stays there.
struct Placed {
    /// The placement the last launch gave, which was checked: the caller's thread is on `on[0]`. A
    /// launch that gives the same checks nothing again.
    on: Vec<usize>,
    /// The CPUs it could run on before: those a placement is checked against, and [`cpus`] lists,
    /// and the ones a launch that places none gives back.
    home: Cpus,
}

/// Puts the caller's thread on `on[0]` and the kept threads on the rest of `on`, when they are
/// not there, and starts the threads the grid needs that none has, each on its CPU. The CPUs are
/// checked first, unless the last launch gave the same, and an error from the check moves nothing.
/// `id` is the number of the launch.
fn put_on(team: &mut Vec<Worker>, caller: &mut Option<Placed>, on: &[usize], id: u64) -> Result<(), Error> {
    // A grid has a thread at least, which `launch` sees to, and the caller's is the first.
    let [first, others @ ..] = on else { return Ok(()) };
    match caller {
        Some(placed) => {
            let same = placed.on == on;
            if !same {
                checked(on, &placed.home)?;
            }
            rest(team, placed.on.first().copied(), on, id);
            if placed.on.first() != Some(first) {
                move_caller(*first)?;
            }
            if !same {
                on.clone_into(&mut placed.on);
            }
        }
        None => {
            let home = mask()?;
            checked(on, &home)?;
            rest(team, None, on, id);
            move_caller(*first)?;
            *caller = Some(Placed { on: on.to_vec(), home });
        }
    }
    for (worker, &cpu) in team.iter_mut().zip(others) {
        worker.go_to(cpu)?;
    }
    start(team, others.len(), Some(on))
}

/// Before any thread moves, ends the spin of each kept thread that a thread would wait for on its new CPU: those on a
/// CPU a thread moves to, and those the system places, which may spin on any. A thread that may run on one CPU alone
/// cannot run while another spins there, and a spin that no launch ends lasts as long as its grid said. A launch that
/// moves no thread ends no spin. `here` is the CPU the caller's thread is on, if a launch placed it. It stores the
/// launch's number, `id`, for each of them, so that a thread that takes late a launch handed to it before this one
/// rests after that one too.
fn rest(team: &[Worker], here: Option<usize>, on: &[usize], id: u64) {
    let now = |t: usize| if t == 0 { here } else { team.get(t - 1).and_then(|worker| worker.at) };
    // The CPUs threads move to, in a set made once a thread moves, for the work to take time as the
    // threads do, not as their square.
    let mut to = None;
    for (t, &cpu) in on.iter().enumerate() {
        if now(t) != Some(cpu) {
            to.get_or_insert_with(Cpus::empty).insert(cpu);
        }
    }
    if let Some(to) = to {
        for worker in team.iter().filter(|worker| worker.at.is_none_or(|cpu| to.contains(cpu))) {
            worker.rest.store(id, Relaxed);
        }
    }
}

/// Lets the caller's thread and every thread kept for it run on the CPUs the caller's thread could
/// run on, those of them a placed launch put on one CPU, and starts the threads the grid needs
/// that none has.
fn put_anywhere(team: &mut Vec<Worker>, caller: &mut Option<Placed>, workers: usize) -> Result<(), Error> {
    if team.iter().any(|worker| worker.at.is_some()) {
        // The CPUs a launch noted, or, if none did, the ones the thread may run on now.
        let read;
        let home = match caller {
            Some(placed) => &placed.home,
            None => {
                read = mask()?;
                &read
            }
        };
        for worker in team.iter_mut() {
            worker.free(home)?;
        }
    }
    if let Some(placed) = caller {
        let moved = placed.home.bind(None);
        moved.map_err(|e| Error {
            message: format!("the caller's thread did not move to the CPUs it could run on: {e}"),
        })?;
        *caller = None;
    }
    start(team, workers, None)
}

/// Puts the calling thread on `cpu`.
fn move_caller(cpu: usize) -> Result<(), Error> {
    let moved = Cpus::one(cpu).bind(None);
    moved.map_err(|e| Error { message: format!("the caller's thread did not move to CPU {cpu}: {e}") })
}

/// Starts the kept threads that none has, for `workers` to be kept. A thread starts on the CPUs of
/// its creator: the one CPU the caller's thread is on, if the grid is placed, as `on` says. It
/// sleeps there, and only when the launch has put it on its CPU is it told to wait for launches,
/// spinning, before the next starts: none spins on the caller's CPU, where the caller has to run
/// to start the next.
fn start(team: &mut Vec<Worker>, workers: usize, on: Option<&[usize]>) -> Result<(), Error> {
    while team.len() < workers {
        let t = team.len() + 1;
        let started = Worker::start(t as u64, on.and_then(|on| on.first().copied()));
        let mut worker = started.map_err(|e| Error { message: format!("a thread of the grid did not start: {e}") })?;
        if let Some(&cpu) = on.and_then(|on| on.get(t)) {
            worker.go_to(cpu)?;
            worker.jobs.send(None).map_err(gone)?;
        }
        team.push(worker);
    }
    Ok(())
}

/// A thread kept for launches, and the two ends it is spoken to by: the launches to run its
/// place in, and how each ended.
struct Worker {
    /// A launch to run, or `None` to say that the thread is on its CPU, for it to wait spinning.
    jobs: SyncSender<Option<Arc<Launch>>>,
    done: Receiver<thread::Result<()>>,
    /// The thread, whose CPUs [`Worker::go_to`] and [`Worker::free`] set.
    thread: JoinHandle<()>,
    /// The CPU the thread is placed on; none if it runs on any the caller's thread may.
    at: Option<usize>,
    /// The number of the last launch that told the thread to stop spinning, which its spin reads. A launch that moves a
    /// thread stores its number here for each thread on a CPU a thread moves to, and for each the system places. A
    /// thread spins after a launch only if no launch with a later number has told it to stop, so after a launch that it
    /// takes late, handed to it before the launch that moved a thread, it rests as well. A thread asleep is not woken
    /// by it.
    rest: Arc<AtomicU64>,
}

impl Worker {
    /// Starts the thread that serves as thread `t` of each launch it is given, as [`serve`] says, until the `Worker` is
    /// dropped. It is named by its place, `grid thread 3`, as a profiler or a debugger lists it. It starts on the CPUs
    /// of the caller's thread, which are the one CPU `at`, or any it may run on if there is none.
    fn start(t: u64, at: Option<usize>) -> io::Result<Worker> {
        let (jobs, inbox) = mpsc::sync_channel::<Option<Arc<Launch>>>(QUEUE);
        let (outbox, done) = mpsc::channel();
        let rest = Arc::new(AtomicU64::new(0));
        let resting = Arc::clone(&rest);
        let name = format!("grid thread {t}");
        let thread =
            thread::Builder::new().name(name).spawn(move || serve(t, &inbox, &outbox, &resting, at.is_some()))?;
        Ok(Worker { jobs, done, thread, at, rest })
    }

    /// Puts the thread on `cpu`, if it is not there already.
    fn go_to(&mut self, cpu: usize) -> Result<(), Error> {
        if self.at != Some(cpu) {
            let moved = Cpus::one(cpu).bind(Some(&self.thread));
            moved.map_err(|e| Error { message: format!("a thread of the grid did not move to CPU {cpu}: {e}") })?;
            self.at = Some(cpu);
        }
        Ok(())
    }

    /// Lets the thread run on `home`, the CPUs the caller's thread could run on, if it is placed
    /// on one CPU.
    fn free(&mut self, home: &Cpus) -> Result<(), Error> {
        if self.at.is_some() {
            let moved = home.bind(Some(&self.thread));
            let to = "the CPUs the caller's thread may run on";
            moved.map_err(|e| Error { message: format!("a thread of the grid did not move to {to}: {e}") })?;
            self.at = None;
        }
        Ok(())
    }
}

thread_local! {
    /// The threads this thread's launches have started, which its next launches run on, the one
    /// at index `i` as thread `i + 1` of each grid. They end with this thread.
    static KEPT: RefCell<Vec<Worker>> = const { RefCell::new(Vec::new()) };
    /// Where a placed launch from this thread has put it, if no launch has let it go since.
    static CALLER: RefCell<Option<Placed>> = const { RefCell::new(None) };
    /// How many launches this thread has made, a launch of blocks as one of threads: the number of the last, which
    /// each launch has.
    static LAUNCHES: Cell<u64> = const { Cell::new(0) };
}

/// What a kept thread does as thread `t`, until its `Worker` is dropped: takes each launch it is sent, in order, runs
/// it, and says how it ended if the launch is of threads, which the caller waits for. A thread that is `placed` has
/// been started on the one CPU of the caller's, where it must not spin: it sleeps until it is sent `None`. Until it has
/// taken a launch it waits for one spinning for [`SPIN`], and after each for as long as that launch's grid says, unless
/// a launch with a later number than the one it took has told it to rest, as `rest` says.
///
/// The thread may be the last to hold a launch of blocks, and then drops its kernel. A panic in that drop ends nothing:
/// the hook prints it, as it prints that of a thread that no one joins, and the thread goes on to the next launch.
fn serve(
    t: u64,
    inbox: &Receiver<Option<Arc<Launch>>>,
    outbox: &Sender<thread::Result<()>>,
    rest: &AtomicU64,
    placed: bool,
) {
    let (mut spin, mut taken) = (SPIN, 0);
    let spinning = |spin, taken| wait(inbox, spin, || rest.load(Relaxed) > taken);
    let mut next = if placed { inbox.recv() } else { spinning(spin, taken) };
    while let Ok(sent) = next {
        if let Some(launch) = sent {
            // A launch uses the thread, so it spins again after, for as long as the launch says.
            (spin, taken) = (launch.spin, launch.id);
            let ended = launch.run(t);
            let answers = launch.answers;
            // The launch ends when every thread has said, so this one lets go of it first.
            let _ = panic::catch_unwind(AssertUnwindSafe(|| drop(launch)));
            if answers {
                let _ = outbox.send(ended);
            }
        }
        next = spinning(spin, taken);
    }
}

/// The next message from `from`, waited for by spinning for `spin`, or until `rest` says to stop,
/// then asleep.
fn wait<T>(from: &Receiver<T>, spin: Duration, rest: impl Fn() -> bool) -> Result<T, RecvError> {
    let start = Instant::now();
    while start.elapsed() < spin && !rest() {
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
    use std::fs;
    use std::hint::black_box;
    use std::mem::{align_of, offset_of, size_of};
    use std::panic;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
    use std::sync::{Arc, Mutex, PoisonError, mpsc};
    use std::thread;
    use std::time::{Duration, Instant};

    use super::{
        Barrier, Cpus, Grid, Launch, Place, QUEUE, SPIN, Worker, checked, cpus, launch, launch_blocks, most, put_on,
        rest, serve,
    };

    type Result<T = (), E = Box<dyn std::error::Error + Send + Sync>> = std::result::Result<T, E>;

    #[test]
    fn each_thread_runs_the_kernel_once_given_its_place() -> Result {
        for threads in 1..=most()?.min(4) {
            let seen = Arc::new(Mutex::new(vec![]));
            let into = Arc::clone(&seen);
            launch(Grid::new(threads), move |place| {
                into.lock().unwrap_or_else(PoisonError::into_inner).push((place.thread(), place.threads()))
            })?;
            let mut seen = seen.lock().unwrap_or_else(PoisonError::into_inner).clone();
            seen.sort();
            let want: Vec<(u64, u64)> = (0..threads as u64).map(|t| (t, threads as u64)).collect();
            assert_eq!(seen, want);
        }
        Ok(())
    }

    /// A thread of the grid is named by its place, as a profiler or a debugger lists threads,
    /// in the launches after the first as in it; the caller's keeps its own name.
    #[test]
    fn a_thread_of_the_grid_is_named_by_its_place() -> Result {
        let threads = most()?.min(4);
        let caller = thread::current().name().map(String::from);
        for _ in 0..3 {
            let named = Arc::new(Mutex::new(vec![]));
            let into = Arc::clone(&named);
            launch(Grid::new(threads), move |place| {
                into.lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .push((place.thread(), thread::current().name().map(String::from)));
            })?;
            let mut named = named.lock().unwrap_or_else(PoisonError::into_inner).clone();
            named.sort();
            let want: Vec<(u64, Option<String>)> = (0..threads as u64)
                .map(|t| (t, if t == 0 { caller.clone() } else { Some(format!("grid thread {t}")) }))
                .collect();
            assert_eq!(named, want);
        }
        Ok(())
    }

    /// The caller's thread is the grid's first, and the launch returns once every thread has
    /// ended: the others take a while, and each has run when it does.
    #[test]
    fn the_callers_thread_is_the_first_and_the_launch_returns_when_all_have_ended() -> Result {
        let caller = thread::current().id();
        let threads = most()?.min(4);
        let ran = Arc::new(Mutex::new(vec![]));
        let into = Arc::clone(&ran);
        launch(Grid::new(threads), move |place| {
            if place.thread() > 0 {
                thread::sleep(Duration::from_millis(20));
            }
            into.lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push((place.thread(), thread::current().id() == caller));
        })?;
        let mut ran = ran.lock().unwrap_or_else(PoisonError::into_inner).clone();
        ran.sort();
        let want: Vec<(u64, bool)> = (0..threads as u64).map(|t| (t, t == 0)).collect();
        assert_eq!(ran, want);
        Ok(())
    }

    #[test]
    fn a_grid_has_one_thread_to_as_many_as_the_process_runs_at_once() -> Result {
        let most = most()?;
        let never = |_: &Place| {};
        assert_eq!(
            launch(Grid::new(0), never).err().ok_or("launch gave Ok")?.message,
            "a grid has one thread at least, and this has none"
        );
        assert_eq!(
            launch(Grid::new(most + 1), never).err().ok_or("launch gave Ok")?.message,
            format!(
                "the grid has {} threads, and the process runs {most} at once: the CPUs the caller's thread may run \
                 on, or fewer where a cgroup limits the process to fewer",
                most + 1
            )
        );
        Ok(())
    }

    /// Two launches at once, from two threads, each have all their threads running: every
    /// thread of both waits for all of both, which none could if a launch waited for the other
    /// to end, or ran on the other's threads.
    #[test]
    fn launches_from_two_threads_at_once_run_at_once_each_on_threads_of_its_own() -> Result {
        let threads = most()?.min(3);
        let meet = Arc::new(std::sync::Barrier::new(2 * threads));
        let ids = Arc::new(Mutex::new(vec![]));
        let callers: Vec<_> = Vec::from_fn(2, |_| {
            let (meet, ids) = (Arc::clone(&meet), Arc::clone(&ids));
            thread::spawn(move || -> Result {
                launch(Grid::new(threads), move |_| {
                    meet.wait();
                    ids.lock().unwrap_or_else(PoisonError::into_inner).push(thread::current().id());
                })?;
                Ok(())
            })
        });
        for caller in callers {
            caller.join().map_err(|_| "a thread panicked")??;
        }
        let ids = ids.lock().unwrap_or_else(PoisonError::into_inner);
        assert_eq!(ids.len(), 2 * threads);
        assert!(ids.iter().enumerate().all(|(i, id)| !ids[i + 1..].contains(id)), "{ids:?}");
        Ok(())
    }

    /// A thread of the grid that panics ends the launch, once all have, as a panic in the
    /// caller with the payload it was given; the threads run the next launch as before.
    #[test]
    fn a_panic_in_a_thread_of_the_grid_ends_the_launch_in_the_caller() -> Result {
        let threads = most()?.min(3);
        let last = threads as u64 - 1;
        // `resume_unwind` panics without the message a hook would print.
        let launched = panic::catch_unwind(|| {
            launch(Grid::new(threads), move |place| {
                if place.thread() == last {
                    panic::resume_unwind(Box::new("the last thread panics"));
                }
            })
        });
        let payload = launched.err().ok_or("launched gave Ok")?;
        assert_eq!(payload.downcast_ref::<&str>(), Some(&"the last thread panics"));
        let ran = Arc::new(Mutex::new(vec![]));
        let into = Arc::clone(&ran);
        launch(Grid::new(threads), move |place| {
            into.lock().unwrap_or_else(PoisonError::into_inner).push(place.thread())
        })?;
        let mut ran = ran.lock().unwrap_or_else(PoisonError::into_inner).clone();
        ran.sort();
        assert_eq!(ran, (0..threads as u64).collect::<Vec<_>>());
        Ok(())
    }

    /// A launch of the placement the launch before it gave checks nothing again, and a launch of
    /// another checks it, and notes it when it passes. The first launch notes its placement, and a
    /// next that differs moves the threads, and notes that. Then the CPUs the caller's thread could
    /// run on are made ones that no placement fits, which a check would refuse: the placement noted
    /// is not checked again, so it passes, and any other is, so it fails and leaves the note as it
    /// was.
    #[test]
    fn a_launch_of_the_placement_the_launch_before_gave_checks_nothing_again() -> Result {
        let all = cpus()?;
        if all.len() < 2 {
            eprintln!("the thread may run on one CPU: skipped");
            return Ok(());
        }
        let (first, second) = ([all[0], all[1]], [all[1], all[0]]);
        let (mut team, mut caller) = (Vec::new(), None);
        put_on(&mut team, &mut caller, &first, 1)?;
        assert_eq!(caller.as_ref().ok_or("as_ref gave None")?.on, first);
        put_on(&mut team, &mut caller, &second, 2)?;
        assert_eq!(caller.as_ref().ok_or("as_ref gave None")?.on, second);
        let none_fits = all[1] + 1;
        caller.as_mut().ok_or("as_mut gave None")?.home = Cpus::one(none_fits);
        assert_eq!(put_on(&mut team, &mut caller, &second, 3), Ok(()));
        let message = format!(
            "thread 0 of the grid is placed on CPU {}, and the caller's thread may run on CPUs {none_fits}",
            first[0]
        );
        assert_eq!(put_on(&mut team, &mut caller, &first, 4).err().ok_or("put_on gave Ok")?.message, message);
        assert_eq!(caller.ok_or("caller gave None")?.on, second);
        Ok(())
    }

    /// Asserts that `work` on 1024 threads takes less than 8 times as long as on 256: 4 times if it
    /// grows as its threads do, 16 if it grows as their square. Each is the quickest of many tries,
    /// which other work on the machine slows the same.
    fn assert_grows_as_threads_do(what: &str, work: impl Fn(usize)) -> Result {
        let quickest = |threads: usize| -> Result<_> {
            let tries = (0..100).map(|_| {
                let began = Instant::now();
                work(threads);
                began.elapsed()
            });
            Ok(tries.min().ok_or("min gave None")?)
        };
        let (few, many) = (quickest(256)?, quickest(1024)?);
        assert!(many < 8 * few, "{what} took {few:?} on 256 threads and {many:?} on 1024");
        Ok(())
    }

    /// A placement of many CPUs is checked in time that grows as its threads do, not as their
    /// square.
    #[test]
    fn a_placement_is_checked_in_time_that_grows_as_its_threads_do() -> Result {
        let on: Vec<usize> = (0..1024).collect();
        let mut home = Cpus::empty();
        on.iter().for_each(|&cpu| _ = home.insert(cpu));
        assert_grows_as_threads_do("the check", |threads| {
            assert_eq!(black_box(checked(black_box(&on[..threads]), &home)), Ok(()));
        })?;
        Ok(())
    }

    /// The check stops at the first thread it refuses, for a CPU the caller's thread may not run
    /// on or for one taken, and for a CPU taken it names the thread that had it first and the one
    /// that has it again.
    #[test]
    fn the_check_names_the_two_threads_given_one_cpu() -> Result {
        let mut home = Cpus::empty();
        (0..8).for_each(|cpu| _ = home.insert(cpu));
        let said = |on: &[usize]| -> Result<_> { Ok(checked(on, &home).err().ok_or("checked gave Ok")?.message) };
        let twice = |other, t, cpu| {
            format!(
                "threads {other} and {t} of the grid are both placed on CPU {cpu}, and a thread that waits needs a CPU \
                 of its own"
            )
        };
        let outside = |t, cpu| {
            format!("thread {t} of the grid is placed on CPU {cpu}, and the caller's thread may run on CPUs 0-7")
        };
        assert_eq!(checked(&[7, 0, 3, 5], &home), Ok(()));
        assert_eq!(said(&[3, 5, 3])?, twice(0, 2, 3));
        assert_eq!(said(&[1, 2, 3, 2])?, twice(1, 3, 2));
        assert_eq!(said(&[4, 5, 6, 5, 4, 6])?, twice(1, 3, 5));
        assert_eq!(said(&[2, 2, 2])?, twice(0, 1, 2));
        assert_eq!(said(&[1, 8, 1])?, outside(1, 8));
        assert_eq!(said(&[1, 1, 8])?, twice(0, 1, 1));
        assert_eq!(said(&[8, 8])?, outside(0, 8));
        Ok(())
    }

    /// The directory in `/proc/self/task` of the thread of this process named `name`, if there is
    /// one.
    fn task_named(name: &str) -> Option<PathBuf> {
        let tasks = fs::read_dir("/proc/self/task").ok()?;
        tasks
            .flatten()
            .map(|task| task.path())
            .find(|task| fs::read_to_string(task.join("comm")).is_ok_and(|comm| comm.trim_end() == name))
    }

    /// Whether Linux says the thread sleeps, which a thread that spins does not: it runs, or waits
    /// for a CPU.
    fn asleep(task: &Path) -> bool {
        let stat = fs::read_to_string(task.join("stat")).unwrap_or_default();
        stat.rsplit_once(") ").is_some_and(|(_, state)| state.starts_with('S'))
    }

    /// For how long the thread has used the CPU, as Linux counts it; exact for a thread that sleeps.
    fn used(task: &Path) -> Result<Duration> {
        let schedstat = fs::read_to_string(task.join("schedstat"))?;
        Ok(Duration::from_nanos(schedstat.split_whitespace().next().ok_or("next gave None")?.parse()?))
    }

    /// A thread started where the caller's thread is on one CPU sleeps, and uses no more CPU than
    /// it takes to start, until it is told it is on its CPU, so that it never spins where the
    /// caller has to run; told, it waits for launches there spinning, and uses CPU for [`SPIN`].
    /// Other work on the machine keeps a thread from running, never from sleeping, so the test
    /// looks at the least CPU a thread used before it was told and the most after, in a few threads.
    #[test]
    fn a_thread_started_for_a_placed_grid_sleeps_until_it_is_told_and_then_spins() -> Result {
        let (mut least_before, mut most_after) = (Duration::MAX, Duration::ZERO);
        for t in 70..90 {
            let worker = Worker::start(t, Some(0))?;
            let name = format!("grid thread {t}");
            let deadline = Instant::now() + Duration::from_secs(5);
            let task = loop {
                match task_named(&name) {
                    Some(task) if asleep(&task) => break task,
                    _ => {
                        assert!(Instant::now() < deadline, "{name} does not sleep in 5 s");
                        thread::sleep(SPIN);
                    }
                }
            };
            // Longer than a spin: a thread that spun would have used its `SPIN` by now.
            thread::sleep(4 * SPIN);
            let before = used(&task)?;
            assert!(asleep(&task), "{name} does not sleep, though it has not been told");
            least_before = least_before.min(before);
            worker.jobs.send(None)?;
            thread::sleep(4 * SPIN);
            while !asleep(&task) {
                assert!(Instant::now() < deadline, "{name} does not sleep in 5 s after it spun");
                thread::sleep(SPIN);
            }
            most_after = most_after.max(used(&task)? - before);
        }
        assert!(least_before < SPIN / 2, "no thread slept as it started: the least CPU one used was {least_before:?}");
        assert!(most_after > SPIN / 2, "no thread spun when told: the most CPU one used was {most_after:?}");
        Ok(())
    }

    /// A kept thread placed on `at`, as `rest` reads one: its CPU and its flag. A `Worker` holds
    /// the handle of a thread, so this one has a thread, which ends at once; nothing speaks to it.
    fn placed(at: usize) -> Worker {
        let (jobs, _) = mpsc::sync_channel(QUEUE);
        let (_, done) = mpsc::channel();
        Worker { jobs, done, thread: thread::spawn(|| {}), at: Some(at), rest: Arc::new(AtomicU64::new(0)) }
    }

    /// A thread of the team that is gone, which a panic in the kernel does not make a thread, ends the launch
    /// as an error, before any thread runs the kernel; and the team is no more, so the launch after starts
    /// one, and runs.
    #[test]
    fn a_thread_that_is_gone_ends_the_launch_in_an_error_and_the_next_launch_starts_afresh() -> Result {
        if most()? < 2 {
            eprintln!("the process runs one thread at once: skipped");
            return Ok(());
        }
        let (jobs, _) = mpsc::sync_channel(QUEUE);
        let (_, done) = mpsc::channel();
        let gone = Worker { jobs, done, thread: thread::spawn(|| {}), at: None, rest: Arc::new(AtomicU64::new(0)) };
        super::KEPT.set(vec![gone]);
        let ran = Arc::new(AtomicU64::new(0));
        let into = Arc::clone(&ran);
        let refused = launch(Grid::new(2), move |_| {
            into.fetch_add(1, Relaxed);
        });
        let refused = refused.err().ok_or("launch gave Ok")?;
        assert_eq!(refused.message, "a thread of the grid ended before its launch did");
        assert_eq!(ran.load(Relaxed), 0);
        let into = Arc::clone(&ran);
        launch(Grid::new(2), move |_| {
            into.fetch_add(1, Relaxed);
        })?;
        assert_eq!(ran.load(Relaxed), 2);
        Ok(())
    }

    /// The end of its channel that a kept thread takes its launches from.
    type Inbox = mpsc::Receiver<Option<Arc<Launch>>>;

    /// What a test holds of a kept thread it made by hand: the end of its channel the launches come out of, and the
    /// end its answers would go into, which stays open, as the thread's would.
    struct Held {
        inbox: Inbox,
        _answers: mpsc::Sender<thread::Result<()>>,
    }

    /// The kept threads a test makes by hand, for the system not to run them: each takes no launch it is given, unless
    /// the test runs it, from the end of its channel, which the test holds, and says nothing of how a launch ended.
    fn fakes(threads: usize) -> (Vec<Worker>, Vec<Held>) {
        (0..threads)
            .map(|_| {
                let (jobs, inbox) = mpsc::sync_channel(QUEUE);
                let (answers, done) = mpsc::channel();
                let rest = Arc::new(AtomicU64::new(0));
                (Worker { jobs, done, thread: thread::spawn(|| {}), at: None, rest }, Held { inbox, _answers: answers })
            })
            .unzip()
    }

    /// Runs `test` on a thread of its own, where it makes the kept threads it needs by hand, and says `waited` if it
    /// does not end in a minute: a launch that waits for a thread that never comes fails the test, and does not hang
    /// it. The thread it leaves behind ends with the process.
    fn within<T: Send + 'static>(waited: &str, test: impl FnOnce() -> Result<T> + Send + 'static) -> Result<T> {
        let (said, heard) = mpsc::channel();
        thread::spawn(move || _ = said.send(test()));
        heard.recv_timeout(Duration::from_secs(60)).map_err(|_| waited)?
    }

    /// A kept thread that never takes its launch holds a launch of blocks back by nothing: the caller runs each block,
    /// once, and the launch returns, though the thread has both its channels open and the launch is in one.
    #[test]
    fn a_thread_that_never_comes_holds_a_launch_of_blocks_back_by_nothing() -> Result {
        if most()? < 2 {
            eprintln!("the process runs one thread at once: skipped");
            return Ok(());
        }
        let (mut seen, caller, handed) = within("a launch of blocks waited for a thread that never comes", || {
            let (team, held) = fakes(1);
            super::KEPT.set(team);
            let seen = Arc::new(Mutex::new(vec![]));
            let into = Arc::clone(&seen);
            launch_blocks(Grid::new(2), 100, move |block| {
                into.lock().unwrap_or_else(PoisonError::into_inner).push((block, thread::current().id()));
            })?;
            let seen = seen.lock().unwrap_or_else(PoisonError::into_inner).clone();
            let handed = held.iter().flat_map(|held| held.inbox.try_iter()).filter(Option::is_some).count();
            Ok((seen, thread::current().id(), handed))
        })?;
        seen.sort_by_key(|&(block, _)| block);
        assert_eq!(seen.iter().map(|&(block, _)| block).collect::<Vec<_>>(), (0..100).collect::<Vec<_>>());
        assert!(seen.iter().all(|&(_, runner)| runner == caller), "a block ran on a thread that never came");
        assert_eq!(handed, 1, "launches in the channel of the thread that never came");
        Ok(())
    }

    /// A launch of blocks hands its launch to a thread for each block but the caller's, and to no more: none for 0
    /// blocks or 1, one for 2, two for 3, and two for any count above, of the three threads the grid has besides the
    /// caller's.
    #[test]
    fn a_launch_of_blocks_wakes_a_thread_for_each_block_but_the_callers_at_most() -> Result {
        if most()? < 3 {
            eprintln!("the process runs fewer than 3 threads at once: skipped");
            return Ok(());
        }
        for (blocks, woken) in [(0, 0), (1, 0), (2, 1), (3, 2), (4, 2), (100, 2)] {
            let (team, held) = fakes(2);
            super::KEPT.set(team);
            launch_blocks(Grid::new(3), blocks, |_| {})?;
            let handed = held.iter().filter(|held| held.inbox.try_recv().is_ok_and(|launch| launch.is_some())).count();
            assert_eq!(handed, woken, "threads handed a launch of {blocks} blocks");
        }
        Ok(())
    }

    /// A kept thread with `QUEUE` launches waiting is handed no more by a launch of blocks, which hands its launch to
    /// another thread, and does not wait for it: a thread so far behind holds nothing back, and no launch queues
    /// for it for ever. The first of two kept threads, which takes none, gets the launches of `QUEUE` launches of two
    /// blocks, which wake one thread each; the launch after finds its queue full, and the second gets it.
    #[test]
    fn a_launch_of_blocks_hands_no_launch_to_a_thread_that_has_a_queue_of_them_waiting() -> Result {
        if most()? < 3 {
            eprintln!("the process runs fewer than 3 threads at once: skipped");
            return Ok(());
        }
        let waiting = within("a launch of blocks waited for a thread whose queue is full", || {
            let (team, held) = fakes(2);
            super::KEPT.set(team);
            for _ in 0..=QUEUE {
                launch_blocks(Grid::new(3), 2, |_| {})?;
            }
            Ok(held.iter().map(|held| held.inbox.try_iter().count()).collect::<Vec<_>>())
        })?;
        assert_eq!(waiting, [QUEUE, 1], "launches waiting for each of the two threads");
        Ok(())
    }

    /// A thread of the team that is gone ends a launch of blocks in the error a launch of threads has, in the same
    /// words; and the team is no more, so the launch after starts one, and each block runs. No block runs before the
    /// error here, since the team has the one thread, and a launch hands its launch to the threads in order: those
    /// before a thread that is gone may run blocks.
    #[test]
    fn a_thread_that_is_gone_ends_a_launch_of_blocks_in_an_error_and_the_next_launch_starts_afresh() -> Result {
        if most()? < 2 {
            eprintln!("the process runs one thread at once: skipped");
            return Ok(());
        }
        let (jobs, _) = mpsc::sync_channel(QUEUE);
        let (_, done) = mpsc::channel();
        let gone = Worker { jobs, done, thread: thread::spawn(|| {}), at: None, rest: Arc::new(AtomicU64::new(0)) };
        super::KEPT.set(vec![gone]);
        let ran = Arc::new(AtomicU64::new(0));
        let into = Arc::clone(&ran);
        let refused = launch_blocks(Grid::new(2), 8, move |_| _ = into.fetch_add(1, Relaxed));
        let refused = refused.err().ok_or("launch_blocks gave Ok")?;
        assert_eq!(refused.message, "a thread of the grid ended before its launch did");
        assert_eq!(ran.load(Relaxed), 0);
        let into = Arc::clone(&ran);
        launch_blocks(Grid::new(2), 8, move |_| _ = into.fetch_add(1, Relaxed))?;
        assert_eq!(ran.load(Relaxed), 8);
        Ok(())
    }

    /// A kept thread that comes to a launch of blocks after it has ended runs no block of it, and lets go of it. Two
    /// launches of 1000 blocks, back to back: the caller runs the first, since the thread, a thread of the test's, does
    /// not come until the second has begun. The first block of the second launch opens the way, and waits until the
    /// thread has run a block of that launch, for ten seconds at most. Each block of each launch ran once, while its
    /// launch ran; the thread ran none of the first launch, and some of the second.
    #[test]
    fn a_thread_that_comes_late_runs_no_block_of_a_launch_that_has_ended() -> Result {
        if most()? < 2 {
            eprintln!("the process runs one thread at once: skipped");
            return Ok(());
        }
        let (team, mut held) = fakes(1);
        super::KEPT.set(team);
        let Held { inbox, _answers } = held.pop().ok_or("pop gave None")?;
        let (open, gate) = mpsc::channel();
        let late = thread::spawn(move || {
            let _ = gate.recv();
            for launch in inbox.try_iter().flatten() {
                let _ = launch.run(1);
            }
        });
        let late_id = late.thread().id();
        // What each block of each launch saw: the launch, the block, who ran it, and whether the launch ran.
        let live = Arc::new([AtomicBool::new(false), AtomicBool::new(false)]);
        let seen = Arc::new(Mutex::new(vec![]));
        for first in [true, false] {
            let which = usize::from(!first);
            let (live, seen, open) = (Arc::clone(&live), Arc::clone(&seen), open.clone());
            live[which].store(true, Relaxed);
            let kernel = {
                let live = Arc::clone(&live);
                move |block: u64| {
                    let ran = live[which].load(Relaxed);
                    seen.lock().unwrap_or_else(PoisonError::into_inner).push((
                        which,
                        block,
                        thread::current().id(),
                        ran,
                    ));
                    if !first && block == 0 {
                        // The thread comes now, to the first launch, which ended, and to this.
                        let _ = open.send(());
                        let late_ran = || {
                            let seen = seen.lock().unwrap_or_else(PoisonError::into_inner);
                            seen.iter().any(|seen| seen.0 == 1 && seen.2 == late_id)
                        };
                        let give_up = Instant::now() + Duration::from_secs(10);
                        while !late_ran() && Instant::now() < give_up {
                            thread::yield_now();
                        }
                    }
                }
            };
            launch_blocks(Grid::new(2), 1000, kernel)?;
            live[which].store(false, Relaxed);
        }
        late.join().map_err(|_| "the late thread panicked")?;
        let seen = seen.lock().unwrap_or_else(PoisonError::into_inner);
        for which in [0, 1] {
            let mut blocks: Vec<u64> = seen.iter().filter(|seen| seen.0 == which).map(|seen| seen.1).collect();
            blocks.sort();
            assert_eq!(blocks, (0..1000).collect::<Vec<_>>(), "launch {which}: the blocks that ran");
        }
        assert!(seen.iter().all(|seen| seen.3), "a block ran when its launch had ended");
        assert!(
            seen.iter().all(|seen| seen.0 == 1 || seen.2 != late_id),
            "the late thread ran a block of the first launch"
        );
        assert!(
            seen.iter().any(|seen| seen.0 == 1 && seen.2 == late_id),
            "the late thread ran no block of the second launch"
        );
        Ok(())
    }

    /// The kernel of a launch of blocks is dropped by the thread that lets go of it last. A launch that hands no thread
    /// its launch drops the kernel in the caller, before it returns. One whose thread does not come, as the test holds
    /// it back, returns with the kernel in the thread's channel, still held; the thread drops it when it has found no
    /// block in the launch, and lets go.
    #[test]
    fn the_kernel_of_a_launch_of_blocks_is_dropped_by_the_thread_that_lets_go_of_it_last() -> Result {
        if most()? < 2 {
            eprintln!("the process runs one thread at once: skipped");
            return Ok(());
        }
        /// Says which thread dropped it.
        struct Guard(mpsc::Sender<thread::ThreadId>);
        impl Drop for Guard {
            fn drop(&mut self) {
                let _ = self.0.send(thread::current().id());
            }
        }
        let (team, mut held) = fakes(1);
        super::KEPT.set(team);
        let Held { inbox, _answers } = held.pop().ok_or("pop gave None")?;
        let (dropped, said) = mpsc::channel();
        // One block wakes no thread: the caller runs it, and lets go of the kernel when the launch ends.
        let guard = Guard(dropped.clone());
        launch_blocks(Grid::new(2), 1, move |_| _ = &guard)?;
        assert_eq!(
            said.try_recv(),
            Ok(thread::current().id()),
            "the kernel of a launch that handed no thread its launch"
        );
        // Two blocks hand the thread the launch, which it does not take: the launch returns, the kernel held.
        let guard = Guard(dropped);
        launch_blocks(Grid::new(2), 2, move |_| _ = &guard)?;
        assert_eq!(said.try_recv(), Err(mpsc::TryRecvError::Empty), "the kernel was dropped while a thread held it");
        let late = thread::spawn(move || {
            for launch in inbox.try_iter().flatten() {
                let _ = launch.run(1);
            }
        });
        let late_id = late.thread().id();
        late.join().map_err(|_| "the late thread panicked")?;
        assert_eq!(said.try_recv(), Ok(late_id), "the thread that dropped the kernel of a launch it came to late");
        Ok(())
    }

    /// A panic in the drop of a kernel on a kept thread ends nothing of the thread: it goes on to the next launch, and
    /// runs some of the blocks of the launches after it, each of which runs each of its blocks once, before it returns,
    /// as when no kernel panics. The thread comes late to a launch of two blocks that the caller has ended, as the test
    /// holds it back, and drops the kernel, as the last to hold it, which panics there.
    #[test]
    fn a_panic_in_the_drop_of_a_kernel_on_a_kept_thread_ends_nothing_of_the_thread() -> Result {
        if most()? < 2 {
            eprintln!("the process runs one thread at once: skipped");
            return Ok(());
        }
        /// Says it is dropped on the thread named `late`, and panics there, without the message a hook would print.
        struct Bomb(mpsc::Sender<()>);
        impl Drop for Bomb {
            fn drop(&mut self) {
                if thread::current().name() == Some("late") {
                    let _ = self.0.send(());
                    panic::resume_unwind(Box::new("the kernel's drop panics on the late thread"));
                }
            }
        }
        let (team, mut held) = fakes(1);
        super::KEPT.set(team);
        let Held { inbox, _answers } = held.pop().ok_or("pop gave None")?;
        let (dropped, said) = mpsc::channel();
        let bomb = Bomb(dropped);
        launch_blocks(Grid::new(2), 2, move |_| _ = &bomb)?;
        let late = thread::Builder::new().name("late".into()).spawn(move || {
            let (outbox, rest) = (mpsc::channel().0, AtomicU64::new(0));
            serve(1, &inbox, &outbox, &rest, false);
        })?;
        said.recv_timeout(Duration::from_secs(10))?;
        // The caller takes blocks as the kept thread does, and could run all of each launch alone: so the blocks that
        // run on the late thread are counted, for the thread not to take each launch and run none of it.
        let by_late = Arc::new(AtomicU64::new(0));
        for _ in 0..20 {
            let ran = Arc::new(AtomicU64::new(0));
            let (into, late_into) = (Arc::clone(&ran), Arc::clone(&by_late));
            launch_blocks(Grid::new(2), 50, move |_| {
                thread::sleep(Duration::from_millis(1));
                into.fetch_add(1, Relaxed);
                if thread::current().name() == Some("late") {
                    late_into.fetch_add(1, Relaxed);
                }
            })?;
            assert_eq!(ran.load(Relaxed), 50, "blocks that had run when the launch returned");
        }
        assert!(by_late.load(Relaxed) > 0, "the late thread ran no block of the launches after");
        assert!(!late.is_finished(), "the thread ended");
        // The thread ends when its team does.
        super::KEPT.take();
        late.join().map_err(|_| "the late thread panicked")?;
        Ok(())
    }

    /// A launch that moves two threads ends the spins it must in time that grows as the grid's
    /// threads do, not as their square. The team is 1023 threads on CPUs 1 to 1023, but for threads
    /// 1 and 2, which are on each other's. A launch of 256 threads puts them back, and so does one
    /// of 1024: each ends the spin of those two threads, and of no other.
    #[test]
    fn a_launch_that_moves_threads_ends_the_spins_in_time_that_grows_as_the_threads_do() -> Result {
        let on: Vec<usize> = (0..1024).collect();
        let mut before = on.clone();
        before.swap(1, 2);
        let team: Vec<Worker> = before[1..].iter().map(|&cpu| placed(cpu)).collect();
        assert_grows_as_threads_do("the end of the spins", |threads| {
            rest(&team[..threads - 1], Some(0), &on[..threads], 1);
        })?;
        let ended: Vec<usize> = (0..team.len()).filter(|&i| team[i].rest.load(Relaxed) == 1).collect();
        assert_eq!(ended, [0, 1]);
        Ok(())
    }

    /// The words of a place and of the barrier where kernite-x86 reads them, the barrier a cache
    /// line of its own.
    #[test]
    fn a_place_is_laid_out_as_a_kernel_reads_it() {
        assert_eq!((offset_of!(Place, thread), offset_of!(Place, threads), offset_of!(Place, barrier)), (0, 8, 16));
        assert_eq!(size_of::<Place>(), 24);
        assert_eq!((offset_of!(Barrier, arrived), offset_of!(Barrier, generation)), (0, 8));
        assert_eq!((size_of::<Barrier>(), align_of::<Barrier>()), (64, 64));
    }
}
