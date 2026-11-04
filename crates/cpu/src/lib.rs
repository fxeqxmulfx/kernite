//! What x86 kernels need of the process beyond their code, for what `#[kernite_macros::kernels]`
//! writes: the threads a cooperative kernel runs on and the CPUs they run on, and the memory of a
//! module's.
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
//! each kept thread on a CPU a thread moves to, and of each the system places: it sleeps until a
//! launch uses it. A launch that moves none ends no spin.
//!
//! A page of memory is on the NUMA node of the CPU that first writes it, so a grid's threads are
//! near their memory when each writes its part of a buffer first. The crate places no memory.
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
//! the kernel, and the statics. The only `unsafe` here is in `affinity`, which reads and sets the
//! CPUs a thread may run on through the C library, as Rust has no call for that, and each of its
//! blocks says why it holds. Calling the kernel is `unsafe` too, and the launcher does that.

// The one module with `unsafe` in it, as the crate's lints say.
#[allow(unsafe_code)]
mod affinity;
mod global;

use affinity::Cpus;

#[doc(hidden)]
pub use global::Stub;
pub use global::{Align, Alignment, Element, Global};

use std::cell::RefCell;
use std::fmt;
use std::hint::spin_loop;
use std::io;
use std::num::NonZero;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::mpsc::{self, Receiver, RecvError, Sender, TryRecvError};
use std::sync::{Arc, LazyLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// How many threads a kernel runs on, each all of it, at once, and where they run: on any CPU the
/// caller's thread may run on, as the system likes, or on the CPUs the caller gives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Grid<'a> {
    /// The threads: one at least, and at most [`most`].
    threads: usize,
    /// The CPU of each thread, if the caller places them.
    cpus: Option<&'a [usize]>,
}

impl Grid<'static> {
    /// A grid of `threads` threads, which the system runs on any CPU the caller's thread may run
    /// on, and moves from one to another as it likes.
    pub fn new(threads: usize) -> Grid<'static> {
        Grid { threads, cpus: None }
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
        Grid { threads: cpus.len(), cpus: Some(cpus) }
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

/// How many threads a grid has at most: as many as the process runs at once, as
/// [`std::thread::available_parallelism`] says the first time it is asked, since a thread that
/// waits at `grid.sync()` spins, and needs a core while it does. That is the CPUs [`cpus`]
/// lists, or fewer where a cgroup limits the process to fewer.
pub fn most() -> Result<usize, Error> {
    static MOST: LazyLock<Result<usize, Error>> = LazyLock::new(|| {
        thread::available_parallelism()
            .map(NonZero::get)
            .map_err(|e| Error { message: format!("the machine does not say how many threads it runs at once: {e}") })
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
/// A thread bound to one CPU cannot run while another spins there, and a spin that no launch ends
/// lasts [`SPIN`]. So a launch that moves threads first ends the spin of each kept thread on a CPU
/// a thread moves to, and of each the system places, which may spin on any: it sleeps until a
/// launch uses it, and the launch does not wait for the spin. A launch that moves none ends no
/// spin. A thread a placed launch starts sleeps until the launch has put it on its CPU, and waits
/// for launches there spinning, so that none spins on the caller's CPU, where the caller has to
/// run to start the next.
///
/// Launches from threads at once run at once, each on threads of its own. A thread of the grid
/// that panics ends the launch, when all have, with that panic in the caller; the threads are
/// kept as they are.
pub fn launch(grid: Grid<'_>, kernel: impl Fn(&Place) + Send + Sync + 'static) -> Result<(), Error> {
    let most = most()?;
    let threads = grid.threads;
    if threads == 0 {
        return Err(Error { message: "a grid has one thread at least, and this has none".into() });
    }
    if threads > most {
        return Err(Error { message: format!("the grid has {threads} threads, and the machine runs {most} at once") });
    }
    let mut team = KEPT.take();
    let ready = CALLER.with_borrow_mut(|caller| match grid.cpus {
        Some(on) => put_on(&mut team, caller, on),
        None => put_anywhere(&mut team, caller, threads - 1),
    });
    let ended = ready.map(|()| run_on(&team[..threads - 1], threads, kernel));
    KEPT.set(team);
    match ended? {
        Ok(()) => Ok(()),
        Err(payload) => panic::resume_unwind(payload),
    }
}

/// Runs `kernel` as thread 0 on the caller's thread, and as the others on `others`, and says how
/// they ended once all have.
fn run_on(others: &[Worker], threads: usize, kernel: impl Fn(&Place) + Send + Sync + 'static) -> thread::Result<()> {
    let launch: Arc<Launch> = Arc::new(Launch { threads: threads as u64, barrier: Barrier::default(), kernel });
    for worker in others {
        worker.jobs.send(Some(Arc::clone(&launch))).expect(GONE);
    }
    let mut ended = launch.run(0);
    for worker in others {
        ended = ended.and(wait(&worker.done, || false).expect(GONE));
    }
    ended
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
fn put_on(team: &mut Vec<Worker>, caller: &mut Option<Placed>, on: &[usize]) -> Result<(), Error> {
    match caller {
        Some(placed) => {
            let same = placed.on == on;
            if !same {
                checked(on, &placed.home)?;
            }
            rest(team, Some(placed.on[0]), on);
            if placed.on[0] != on[0] {
                move_caller(on[0])?;
            }
            if !same {
                on.clone_into(&mut placed.on);
            }
        }
        None => {
            let home = mask()?;
            checked(on, &home)?;
            rest(team, None, on);
            move_caller(on[0])?;
            *caller = Some(Placed { on: on.to_vec(), home });
        }
    }
    for (worker, &cpu) in team.iter_mut().zip(&on[1..]) {
        worker.go_to(cpu)?;
    }
    start(team, on.len() - 1, Some(on))
}

/// Before any thread moves, ends the spin of each kept thread that a thread would wait for on its
/// new CPU: those on a CPU a thread moves to, and those the system places, which may spin on any.
/// A thread that may run on one CPU alone cannot run while another spins there, and a spin that no
/// launch ends lasts [`SPIN`]. A launch that moves no thread ends no spin. `here` is the CPU the
/// caller's thread is on, if a launch placed it.
fn rest(team: &[Worker], here: Option<usize>, on: &[usize]) {
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
            worker.rest.store(true, Relaxed);
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
        let started = Worker::start(t as u64, on.map(|on| on[0]));
        let mut worker = started.map_err(|e| Error { message: format!("a thread of the grid did not start: {e}") })?;
        if let Some(on) = on {
            worker.go_to(on[t])?;
            worker.jobs.send(None).expect(GONE);
        }
        team.push(worker);
    }
    Ok(())
}

/// A thread kept for launches, and the two ends it is spoken to by: the launches to run its
/// place in, and how each ended.
struct Worker {
    /// A launch to run, or `None` to say that the thread is on its CPU, for it to wait spinning.
    jobs: Sender<Option<Arc<Launch>>>,
    done: Receiver<thread::Result<()>>,
    /// The thread, whose CPUs [`Worker::go_to`] and [`Worker::free`] set.
    thread: JoinHandle<()>,
    /// The CPU the thread is placed on; none if it runs on any the caller's thread may.
    at: Option<usize>,
    /// Whether the thread is to stop spinning, which its spin reads. A launch that moves a thread
    /// sets it for each thread on a CPU a thread moves to, and for each the system places; the
    /// thread clears it when it takes a launch. A thread asleep is not woken by it.
    rest: Arc<AtomicBool>,
}

impl Worker {
    /// Starts the thread that runs thread `t` of each launch it is given and says how each
    /// ended, until the `Worker` is dropped. It is named by its place, `grid thread 3`, as a
    /// profiler or a debugger lists it. It starts on the CPUs of the caller's thread, which are
    /// the one CPU `at`, or any it may run on if there is none. Where there is one, the thread
    /// sleeps until it is sent `None`: it is not to spin there.
    fn start(t: u64, at: Option<usize>) -> io::Result<Worker> {
        let (jobs, inbox) = mpsc::channel::<Option<Arc<Launch>>>();
        let (outbox, done) = mpsc::channel();
        let rest = Arc::new(AtomicBool::new(false));
        let resting = Arc::clone(&rest);
        let thread = thread::Builder::new().name(format!("grid thread {t}")).spawn(move || {
            let spin = || wait(&inbox, || resting.load(Relaxed));
            let mut next = if at.is_some() { inbox.recv() } else { spin() };
            while let Ok(sent) = next {
                if let Some(launch) = sent {
                    // A launch uses the thread, so it spins again after.
                    resting.store(false, Relaxed);
                    let ended = launch.run(t);
                    // The launch ends when every thread has said, so this one lets go of it first.
                    drop(launch);
                    let _ = outbox.send(ended);
                }
                next = spin();
            }
        })?;
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
}

/// The next message from `from`, waited for by spinning for [`SPIN`], or until `rest` says to
/// stop, then asleep.
fn wait<T>(from: &Receiver<T>, rest: impl Fn() -> bool) -> Result<T, RecvError> {
    let start = Instant::now();
    while start.elapsed() < SPIN && !rest() {
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
    use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
    use std::sync::{Arc, Mutex, mpsc};
    use std::thread;
    use std::time::{Duration, Instant};

    use super::{Barrier, Cpus, Grid, Place, SPIN, Worker, checked, cpus, launch, most, put_on, rest};

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

    /// A launch of the placement the launch before it gave checks nothing again, and a launch of
    /// another checks it, and notes it when it passes. The first launch notes its placement, and a
    /// next that differs moves the threads, and notes that. Then the CPUs the caller's thread could
    /// run on are made ones that no placement fits, which a check would refuse: the placement noted
    /// is not checked again, so it passes, and any other is, so it fails and leaves the note as it
    /// was.
    #[test]
    fn a_launch_of_the_placement_the_launch_before_gave_checks_nothing_again() {
        let all = cpus().unwrap();
        if all.len() < 2 {
            return eprintln!("the thread may run on one CPU: skipped");
        }
        let (first, second) = ([all[0], all[1]], [all[1], all[0]]);
        let (mut team, mut caller) = (Vec::new(), None);
        put_on(&mut team, &mut caller, &first).unwrap();
        assert_eq!(caller.as_ref().unwrap().on, first);
        put_on(&mut team, &mut caller, &second).unwrap();
        assert_eq!(caller.as_ref().unwrap().on, second);
        let none_fits = all[1] + 1;
        caller.as_mut().unwrap().home = Cpus::one(none_fits);
        assert_eq!(put_on(&mut team, &mut caller, &second), Ok(()));
        let message = format!(
            "thread 0 of the grid is placed on CPU {}, and the caller's thread may run on CPUs {none_fits}",
            first[0]
        );
        assert_eq!(put_on(&mut team, &mut caller, &first).unwrap_err().message, message);
        assert_eq!(caller.unwrap().on, second);
    }

    /// Asserts that `work` on 1024 threads takes less than 8 times as long as on 256: 4 times if it
    /// grows as its threads do, 16 if it grows as their square. Each is the quickest of many tries,
    /// which other work on the machine slows the same.
    fn assert_grows_as_threads_do(what: &str, work: impl Fn(usize)) {
        let quickest = |threads: usize| {
            let tries = (0..100).map(|_| {
                let began = Instant::now();
                work(threads);
                began.elapsed()
            });
            tries.min().unwrap()
        };
        let (few, many) = (quickest(256), quickest(1024));
        assert!(many < 8 * few, "{what} took {few:?} on 256 threads and {many:?} on 1024");
    }

    /// A placement of many CPUs is checked in time that grows as its threads do, not as their
    /// square.
    #[test]
    fn a_placement_is_checked_in_time_that_grows_as_its_threads_do() {
        let on: Vec<usize> = (0..1024).collect();
        let mut home = Cpus::empty();
        on.iter().for_each(|&cpu| _ = home.insert(cpu));
        assert_grows_as_threads_do("the check", |threads| {
            black_box(checked(black_box(&on[..threads]), &home)).unwrap();
        });
    }

    /// The check stops at the first thread it refuses, for a CPU the caller's thread may not run
    /// on or for one taken, and for a CPU taken it names the thread that had it first and the one
    /// that has it again.
    #[test]
    fn the_check_names_the_two_threads_given_one_cpu() {
        let mut home = Cpus::empty();
        (0..8).for_each(|cpu| _ = home.insert(cpu));
        let said = |on: &[usize]| checked(on, &home).unwrap_err().message;
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
        assert_eq!(said(&[3, 5, 3]), twice(0, 2, 3));
        assert_eq!(said(&[1, 2, 3, 2]), twice(1, 3, 2));
        assert_eq!(said(&[4, 5, 6, 5, 4, 6]), twice(1, 3, 5));
        assert_eq!(said(&[2, 2, 2]), twice(0, 1, 2));
        assert_eq!(said(&[1, 8, 1]), outside(1, 8));
        assert_eq!(said(&[1, 1, 8]), twice(0, 1, 1));
        assert_eq!(said(&[8, 8]), outside(0, 8));
    }

    /// The directory in `/proc/self/task` of the thread of this process named `name`, if there is
    /// one.
    fn task_named(name: &str) -> Option<PathBuf> {
        let tasks = fs::read_dir("/proc/self/task").ok()?;
        tasks
            .map(|task| task.unwrap().path())
            .find(|task| fs::read_to_string(task.join("comm")).is_ok_and(|comm| comm.trim_end() == name))
    }

    /// Whether Linux says the thread sleeps, which a thread that spins does not: it runs, or waits
    /// for a CPU.
    fn asleep(task: &Path) -> bool {
        let stat = fs::read_to_string(task.join("stat")).unwrap_or_default();
        stat.rsplit_once(") ").is_some_and(|(_, state)| state.starts_with('S'))
    }

    /// For how long the thread has used the CPU, as Linux counts it; exact for a thread that sleeps.
    fn used(task: &Path) -> Duration {
        let schedstat = fs::read_to_string(task.join("schedstat")).unwrap();
        Duration::from_nanos(schedstat.split_whitespace().next().unwrap().parse().unwrap())
    }

    /// A thread started where the caller's thread is on one CPU sleeps, and uses no more CPU than
    /// it takes to start, until it is told it is on its CPU, so that it never spins where the
    /// caller has to run; told, it waits for launches there spinning, and uses CPU for [`SPIN`].
    /// Other work on the machine keeps a thread from running, never from sleeping, so the test
    /// looks at the least CPU a thread used before it was told and the most after, in a few threads.
    #[test]
    fn a_thread_started_for_a_placed_grid_sleeps_until_it_is_told_and_then_spins() {
        let (mut least_before, mut most_after) = (Duration::MAX, Duration::ZERO);
        for t in 70..90 {
            let worker = Worker::start(t, Some(0)).unwrap();
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
            let before = used(&task);
            assert!(asleep(&task), "{name} does not sleep, though it has not been told");
            least_before = least_before.min(before);
            worker.jobs.send(None).unwrap();
            thread::sleep(4 * SPIN);
            while !asleep(&task) {
                assert!(Instant::now() < deadline, "{name} does not sleep in 5 s after it spun");
                thread::sleep(SPIN);
            }
            most_after = most_after.max(used(&task) - before);
        }
        assert!(least_before < SPIN / 2, "no thread slept as it started: the least CPU one used was {least_before:?}");
        assert!(most_after > SPIN / 2, "no thread spun when told: the most CPU one used was {most_after:?}");
    }

    /// A kept thread placed on `at`, as `rest` reads one: its CPU and its flag. A `Worker` holds
    /// the handle of a thread, so this one has a thread, which ends at once; nothing speaks to it.
    fn placed(at: usize) -> Worker {
        let (jobs, _) = mpsc::channel();
        let (_, done) = mpsc::channel();
        Worker { jobs, done, thread: thread::spawn(|| {}), at: Some(at), rest: Arc::new(AtomicBool::new(false)) }
    }

    /// A launch that moves two threads ends the spins it must in time that grows as the grid's
    /// threads do, not as their square. The team is 1023 threads on CPUs 1 to 1023, but for threads
    /// 1 and 2, which are on each other's. A launch of 256 threads puts them back, and so does one
    /// of 1024: each ends the spin of those two threads, and of no other.
    #[test]
    fn a_launch_that_moves_threads_ends_the_spins_in_time_that_grows_as_the_threads_do() {
        let on: Vec<usize> = (0..1024).collect();
        let mut before = on.clone();
        before.swap(1, 2);
        let team: Vec<Worker> = before[1..].iter().map(|&cpu| placed(cpu)).collect();
        assert_grows_as_threads_do("the end of the spins", |threads| {
            rest(&team[..threads - 1], Some(0), &on[..threads]);
        });
        let ended: Vec<usize> = (0..team.len()).filter(|&i| team[i].rest.load(Relaxed)).collect();
        assert_eq!(ended, [0, 1]);
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
