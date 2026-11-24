//! How long the threads of a grid wait spinning, and what a launch that moves threads does to the
//! spin of the kept threads. A thread waits spinning for as long as its grid says, [`SPIN`] if the
//! grid says none: a kept thread for the next launch, and the caller for each of the others to end.
//!
//! A thread that may run on one CPU alone cannot run while another spins there, and a kept thread
//! spins after its launch, so a launch that moves a thread onto the CPU of a kept thread would
//! wait for the spin to end. It ends the spin of each kept thread on a CPU a thread moves to, and
//! of each the system places, which may spin on any, before anything moves, and the spin of no
//! other: a launch that moves none ends none.
//!
//! No test here times a launch, for other work on the machine slows a launch by as much as the spin
//! it looks for. A test counts, as Linux does. Each thread of a launch counts itself as its kernel
//! starts: for how long it has used a CPU, how many times it has stopped to wait, and how many
//! times another thread has taken its CPU, all as of the last time it stopped running. The counts
//! of two launches that use a thread, one after the other, say what it did between them. A thread
//! told to stop spinning stops to wait after a few microseconds of CPU, with no thread taking its
//! CPU. One that is not told spins until its spin runs out, which uses as much CPU as the spin lasts, or until
//! another thread takes its CPU, and does not stop to wait before. Other work on the machine delays
//! threads, and so changes how long one spins before a launch tells it, but it never makes a thread
//! stop to wait early, nor spin on after it is told. A test of a thread that must be told needs one
//! round of many in which no thread was delayed; a test of one that must not is sure of it whatever
//! the machine does.

use std::fs::File;
use std::io;
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Barrier, Mutex, MutexGuard, OnceLock, PoisonError};
use std::thread;
use std::time::{Duration, Instant};

use kernite_cpu::{Grid, SPIN, cpus, launch, most};

type Result<T = (), E = Box<dyn std::error::Error + Send + Sync>> = std::result::Result<T, E>;

/// How many rounds a test makes at most to see a thread told to stop spinning. Many rounds show it
/// on a quiet machine, and one in a few hundred beside busy work on every CPU, where a launch
/// takes milliseconds; a test ends at the first that does.
const TRIES: usize = 5000;

/// How many rounds a test makes to see that no thread is told to stop spinning.
const ROUNDS: usize = 300;

/// How many threads a grid of these tests has.
const THREADS: usize = 3;

/// A spin that no test waits out: a launch goes on long before such a spin would end, so a test that is not done in
/// a fraction of it has waited for something else.
const LONG: Duration = Duration::from_secs(10);

/// How long a test lets a thread sleep between two launches, for a thread whose spin has run out to
/// have gone to sleep.
const GAP: Duration = Duration::from_millis(2);

/// How much CPU a kept thread may use between two launches beyond its spin: the system adds some
/// microseconds, to wake it, to run its kernel and to count it. Four spins of `SPIN` are room to spare.
const SLACK: Duration = Duration::from_micros(800);

/// The tests place threads on the same CPUs, and a thread that spins on one keeps another from
/// running there, so they run one at a time.
fn alone() -> MutexGuard<'static, ()> {
    static ONE: Mutex<()> = Mutex::new(());
    ONE.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The first four CPUs the caller's thread may run on, one more than a grid has threads, if the
/// process runs that many threads at once; none if not.
fn four() -> Result<Option<Vec<usize>>> {
    let all = cpus()?;
    let four = all.get(..THREADS + 1);
    if four.is_none() || most()? < THREADS {
        eprintln!(
            "the caller's thread may run on fewer than four CPUs, or the process runs fewer than three threads at \
             once: skipped"
        );
        return Ok(None);
    }
    Ok(four.map(<[usize]>::to_vec))
}

/// The text in the first `size` bytes of `buffer`, which a read filled.
fn text(buffer: &[u8], size: usize) -> Result<&str> {
    Ok(str::from_utf8(buffer.get(..size).ok_or("a read gave more bytes than the buffer has")?)?)
}

/// What Linux has counted of a thread, as of the last time it stopped running.
#[derive(Clone, Copy)]
struct Count {
    /// For how long the thread has used a CPU.
    cpu: Duration,
    /// How many times it has stopped to wait.
    stopped: u64,
    /// How many times another thread has taken its CPU.
    displaced: u64,
}

impl Count {
    /// The count of the thread that the open files `status` and `schedstat` in `/proc` are of. The
    /// thread reads `status` last, with its buffers made, and asks nothing of the system after:
    /// a call that stopped the thread to wait would be counted by the next count, as if a launch
    /// had told it to stop.
    fn of(status: &File, schedstat: &File) -> Result<Count> {
        let (mut times, mut switches) = ([0; 256], [0; 8192]);
        let (times_size, switches_size) = (schedstat.read_at(&mut times, 0)?, status.read_at(&mut switches, 0)?);
        let (times, switches) = (text(&times, times_size)?, text(&switches, switches_size)?);
        let field = |name: &str| -> Result<u64> {
            let line = switches.lines().find_map(|line| line.strip_prefix(name)?.strip_prefix(':'));
            Ok(line.ok_or("find_map gave None")?.trim().parse()?)
        };
        Ok(Count {
            cpu: Duration::from_nanos(times.split_whitespace().next().ok_or("next gave None")?.parse()?),
            stopped: field("voluntary_ctxt_switches")?,
            displaced: field("nonvoluntary_ctxt_switches")?,
        })
    }

    /// What the thread did between `before` and this count.
    fn since(&self, before: &Count) -> Since {
        Since {
            cpu: self.cpu.saturating_sub(before.cpu),
            stopped: self.stopped.saturating_sub(before.stopped),
            displaced: self.displaced.saturating_sub(before.displaced),
        }
    }
}

/// What a thread did between two counts.
#[derive(Clone, Copy)]
struct Since {
    /// How much CPU it used.
    cpu: Duration,
    /// How many times it stopped to wait.
    stopped: u64,
    /// How many times another thread took its CPU.
    displaced: u64,
}

impl Since {
    /// Whether the thread was told to stop spinning: it stopped to wait after less than half of `SPIN` of CPU, and no
    /// thread took its CPU. A thread whose spin runs out has used `SPIN` of CPU when it stops, or has had its CPU
    /// taken.
    fn told(&self) -> bool {
        self.stopped > 0 && self.displaced == 0 && self.cpu < SPIN / 2
    }

    /// Whether the thread spun for `spin`, as far as CPU says, and then stopped to wait, with no thread taking its CPU:
    /// it used half of `spin` of CPU at least. Other work on the machine gives a thread less CPU than its spin, never
    /// more.
    fn spun(&self, spin: Duration) -> bool {
        self.stopped > 0 && self.displaced == 0 && self.cpu >= spin / 2
    }
}

/// The files in `/proc` of a thread, `status` and `schedstat`, as it opened them.
type Files = io::Result<(File, File)>;

/// Launches that count the threads they run. A count is as of the last time the thread stopped
/// running, so the CPU a thread used between two counts is that of all the time it was awake: short
/// when a launch tells it to stop soon after it wakes, as the launches of a test do, one after the
/// other.
#[derive(Default)]
struct Counter {
    /// The files in `/proc` of each thread. A thread opens its own in the first launch it runs in,
    /// and reads them again in the ones after, which costs less than opening.
    files: Arc<[OnceLock<Files>; THREADS]>,
    /// The count of each thread at the last launch that used it.
    last: [Option<Count>; THREADS],
}

impl Counter {
    /// Launches `grid`, and says, by thread, which of its kept threads were told to stop spinning
    /// since the last launch that used them. The caller's thread, the first, is never one.
    fn launch(&mut self, grid: Grid<'_>) -> Result<[bool; THREADS]> {
        Ok(self.since(grid)?.map(|since| since.is_some_and(|since| since.told())))
    }

    /// Launches `grid`, and says, by thread, what each of its kept threads did since the last launch that used it: none
    /// for the caller's thread, the first, for a thread that has not run a launch before, and for one the grid does not
    /// use.
    fn since(&mut self, grid: Grid<'_>) -> Result<[Option<Since>; THREADS]> {
        let (files, counts) = (Arc::clone(&self.files), Arc::<[OnceLock<Result<Count>>; THREADS]>::default());
        let into = Arc::clone(&counts);
        launch(grid, move |place| {
            let t = place.thread() as usize;
            let (Some(files), Some(into)) = (files.get(t), into.get(t)) else { return };
            // `/proc/thread-self` is the thread that opens it.
            let opened = files.get_or_init(|| {
                Ok((File::open("/proc/thread-self/status")?, File::open("/proc/thread-self/schedstat")?))
            });
            let _ = into.set(match opened {
                Ok((status, schedstat)) => Count::of(status, schedstat),
                Err(e) => Err(e.to_string().into()),
            });
        })?;
        let mut since = [None; THREADS];
        for ((count, last), slot) in counts.iter().zip(&mut self.last).zip(&mut since).skip(1) {
            if let Some(count) = count.get() {
                let count = *count.as_ref().map_err(|e| e.to_string())?;
                *slot = last.replace(count).map(|before| count.since(&before));
            }
        }
        Ok(since)
    }
}

/// A launch that moves a thread onto the CPU of a kept thread that spins ends that thread's spin
/// before it moves anything, so it does not wait for the spin: in some round, the thread stops to
/// wait after a few microseconds of CPU, and no thread has taken its CPU. A launch that waited for
/// the spin would find the thread used `SPIN` of CPU, or its CPU taken, and one that ended the spin
/// after it moved would be too late. A round is two launches: three threads on the first three
/// CPUs, kept; and the second, which moves a thread onto the CPU of one, and not as its own place:
///
/// - the caller's thread goes where a kept thread the second launch does not use spins;
/// - the caller's thread goes where a kept thread the second launch uses spins, and that thread
///   goes where the caller's was;
/// - a kept thread goes where another that the launch does not use spins.
///
/// The first launch of the next round puts them back, and counts the thread the second does not
/// use.
#[test]
fn a_launch_that_moves_a_thread_onto_the_cpu_of_a_kept_thread_ends_its_spin_before_it_moves() -> Result {
    let _alone = alone();
    let Some(on) = four()? else { return Ok(()) };
    let first = Grid::on(&on[..THREADS]);
    let (caller_to_unused, caller_to_used, kept_to_unused) = ([on[2], on[1]], [on[1], on[0]], [on[0], on[2]]);
    // The second launch, and the kept thread in the way.
    let cases = [
        ("the caller's thread goes where a thread the launch does not use spins", Grid::on(&caller_to_unused), 2),
        ("the caller's thread goes where a thread the launch uses spins", Grid::on(&caller_to_used), 1),
        ("a kept thread goes where a thread the launch does not use spins", Grid::on(&kept_to_unused), 2),
    ];
    for (what, second, in_the_way) in cases {
        let mut counter = Counter::default();
        let mut ended = false;
        for _ in 0..TRIES {
            let (told_first, told_second) = (counter.launch(first)?, counter.launch(second)?);
            ended = told_first[in_the_way] || told_second[in_the_way];
            if ended {
                break;
            }
        }
        let message = "rounds found the kept thread in the way stop to wait early, with no thread taking its CPU";
        assert!(ended, "{what}: none of {TRIES} {message}");
    }
    Ok(())
}

/// A launch ends the spin of no kept thread that no thread moves onto the CPU of: not when it
/// moves none, and not when the CPU a thread moves to is free. A round is two launches: three
/// threads on the first three CPUs; and a second that uses the first kept thread and not the
/// second. In no round is either kept thread told to stop spinning, by either launch.
#[test]
fn a_launch_ends_the_spin_of_no_kept_thread_that_no_thread_moves_onto_the_cpu_of() -> Result {
    let _alone = alone();
    let Some(on) = four()? else { return Ok(()) };
    let first = Grid::on(&on[..THREADS]);
    let (none_moves, to_a_free_cpu) = ([on[0], on[1]], [on[3], on[1]]);
    let cases = [
        ("no thread moves", Grid::on(&none_moves)),
        ("the caller's thread goes to a CPU where none spins", Grid::on(&to_a_free_cpu)),
    ];
    for (what, second) in cases {
        let mut counter = Counter::default();
        let mut rounds_told = 0;
        for _ in 0..ROUNDS {
            let (told_first, told_second) = (counter.launch(first)?, counter.launch(second)?);
            rounds_told += usize::from(told_first.iter().chain(&told_second).any(|&told| told));
        }
        let message = "rounds found a kept thread told to stop spinning, where no thread moved onto its CPU";
        assert_eq!(rounds_told, 0, "{what}: {rounds_told} of {ROUNDS} {message}");
    }
    Ok(())
}

/// A launch that moves a thread ends the spin of the kept threads the system places, which may
/// spin on any CPU: in some round, each stops to wait after a few microseconds of CPU, and no
/// thread has taken its CPU. A round is a launch of three threads that the system places, and
/// a launch that moves the caller's thread.
#[test]
fn a_launch_that_moves_a_thread_ends_the_spin_of_the_kept_threads_the_system_places() -> Result {
    let _alone = alone();
    let Some(on) = four()? else { return Ok(()) };
    let (anywhere, here) = (Grid::new(THREADS), Grid::on(&on[..1]));
    let mut counter = Counter::default();
    let mut ended = [false; 2];
    for _ in 0..TRIES {
        let told = counter.launch(anywhere)?;
        counter.launch(here)?;
        ended = [ended[0] || told[1], ended[1] || told[2]];
        if ended == [true; 2] {
            break;
        }
    }
    let message = "stopped to wait early, with no thread taking its CPU, in some of";
    assert_eq!(ended, [true; 2], "kept threads the system places did not all stop: {ended:?} {message} {TRIES} rounds");
    Ok(())
}

/// A kept thread whose spin a launch ended spins again after the next launch that uses it. A round
/// is three launches: the second moves the caller's thread onto the CPU of the third thread, which
/// it does not use, and ends its spin; and the first launch twice, which puts the caller back, and
/// uses the third thread each time. In some round, the thread stops to wait early between the
/// launch before the round and the first of the two. In no round does it stop to wait early
/// between the two: a thread that is told to stop spinning stays told until a launch uses it, and a
/// launch that does not clear that leaves the thread to sleep at once, where it was to spin.
#[test]
fn a_kept_thread_whose_spin_a_launch_ended_spins_again_after_the_next_launch_that_uses_it() -> Result {
    let _alone = alone();
    let Some(on) = four()? else { return Ok(()) };
    let (first, caller_to_third) = (Grid::on(&on[..THREADS]), [on[2], on[1]]);
    let second = Grid::on(&caller_to_third);
    let mut counter = Counter::default();
    let (mut ended, mut asleep_again, mut rounds) = (false, 0, 0);
    while rounds < ROUNDS || (!ended && rounds < TRIES) {
        counter.launch(second)?;
        // The third thread, at index 2.
        ended |= counter.launch(first)?[2];
        asleep_again += usize::from(counter.launch(first)?[2]);
        rounds += 1;
    }
    let message = "rounds found the thread stop to wait early after a launch that used it";
    assert!(ended, "none of {rounds} rounds found the thread the second launch left stop to wait early");
    assert_eq!(asleep_again, 0, "{asleep_again} of {rounds} {message}");
    Ok(())
}

/// A kept thread waits for the next launch spinning for as long as its grid says, and then sleeps: not at all if the
/// spin is nothing, 50 ms if it is 50 ms, and `SPIN` if the grid says none, whether the system places its threads or
/// the caller does. A round is a sleep that outlasts the spin, and a launch, which counts what each kept thread did
/// since the launch before.
///
/// A thread that spins uses half its spin in CPU at least, and stops to wait, in some round; in no round does it use
/// more than its spin and some microseconds, nor is it told to stop early. A thread whose spin is nothing stops to wait
/// after a few microseconds of CPU, with no thread taking its CPU, in some round: other work on the machine may give a
/// thread its CPU late, which counts as taken, but never makes it stop early.
#[test]
fn a_kept_thread_spins_for_as_long_as_its_grid_says_and_then_sleeps() -> Result {
    let _alone = alone();
    let Some(on) = four()? else { return Ok(()) };
    let fifty = Duration::from_millis(50);
    for (name, grid) in [("Grid::new", Grid::new(THREADS)), ("Grid::on", Grid::on(&on[..THREADS]))] {
        // The grid, the spin it says, and how many rounds to make: a round of 50 ms lasts some 25 times as long.
        let cases = [
            ("a spin of nothing", grid.spin(Duration::ZERO), Duration::ZERO, ROUNDS),
            ("a spin of 50 ms", grid.spin(fifty), fifty, 10),
            ("no spin said", grid, SPIN, ROUNDS),
        ];
        for (what, grid, spin, rounds) in cases {
            let mut counter = Counter::default();
            // The first launch starts the threads, and counts them for the first time.
            counter.launch(grid)?;
            let (mut told, mut spun) = (0, 0);
            for _ in 0..rounds {
                thread::sleep(spin + GAP);
                for since in counter.since(grid)?.into_iter().flatten() {
                    let used = since.cpu;
                    assert!(
                        used <= spin + SLACK,
                        "{name}, {what}: a kept thread used {used:?} of CPU between launches"
                    );
                    told += usize::from(since.told());
                    spun += usize::from(since.spun(spin));
                }
            }
            if spin.is_zero() {
                assert!(
                    told > 0,
                    "{name}, {what}: no kept thread slept at once in any of {rounds} rounds, \
                     though its spin was nothing"
                );
            } else {
                assert_eq!(told, 0, "{name}, {what}: kept threads slept at once, though the grid said to spin");
                assert!(spun > 0, "{name}, {what}: no thread of {rounds} rounds spun for {spin:?} and then slept");
            }
        }
    }
    Ok(())
}

/// How long the second thread of a launch of two takes: more than the `SPIN` the caller spins for, and less than the
/// spin a grid says.
const LATE: Duration = Duration::from_millis(1);

/// A launch of two threads whose second ends `LATE` after the caller's, and what the caller, whose files in `/proc` are
/// `files`, did while it waited for it.
fn waited_for_the_second(grid: Grid<'_>, files: &(File, File)) -> Result<Since> {
    let before = Count::of(&files.0, &files.1)?;
    launch(grid, |place| {
        if place.thread() == 1 {
            thread::sleep(LATE);
        }
    })?;
    Ok(Count::of(&files.0, &files.1)?.since(&before))
}

/// The caller waits for the others to end spinning for as long as its grid says, as a kept thread waits for the next
/// launch. The second thread of the launch takes longer than `SPIN` and less than 5 ms. With no spin said, the caller
/// stops to wait in every round, for its spin ends before the second thread does. With a spin of 5 ms, in some of many
/// rounds it does not stop to wait at all: other work on the machine may keep the second thread from ending in 5 ms,
/// but never makes the caller stop early.
#[test]
fn the_caller_waits_for_the_others_spinning_for_as_long_as_its_grid_says() -> Result {
    let _alone = alone();
    if most()? < 2 {
        eprintln!("the process runs one thread at once: skipped");
        return Ok(());
    }
    // The caller's own counts are exact: they are read as it runs.
    let files = (File::open("/proc/thread-self/status")?, File::open("/proc/thread-self/schedstat")?);
    // This launch starts the second thread, and a launch that starts one is not the one to count.
    waited_for_the_second(Grid::new(2), &files)?;
    for round in 0..ROUNDS {
        let since = waited_for_the_second(Grid::new(2), &files)?;
        assert!(since.stopped > 0, "round {round}: the caller did not stop to wait for a thread that took {LATE:?}");
    }
    let grid = Grid::new(2).spin(Duration::from_millis(5));
    let mut waited = false;
    for _ in 0..TRIES {
        waited = waited_for_the_second(grid, &files)?.stopped == 0;
        if waited {
            break;
        }
    }
    assert!(
        waited,
        "the caller stopped to wait in each of {TRIES} rounds, for a thread that took {LATE:?}, with a spin of 5 ms"
    );
    Ok(())
}

/// A launch that moves a thread onto the CPU of a kept thread that was to spin for ten seconds ends that spin before it
/// moves anything, as it ends one of `SPIN`: in some round the kept thread in the way stops to wait after a few
/// microseconds of CPU, and no thread has taken its CPU. A round is two launches, as in the first case of
/// `a_launch_that_moves_a_thread_onto_the_cpu_of_a_kept_thread_ends_its_spin_before_it_moves`: three threads on the
/// first three CPUs, kept; and the second, which moves the caller's thread onto the CPU of the third, which it does
/// not use. No round lasts as long as the spin: a launch does not wait for it.
#[test]
fn a_launch_that_moves_a_thread_ends_a_spin_that_was_to_last_ten_seconds_at_once() -> Result {
    let _alone = alone();
    let Some(on) = four()? else { return Ok(()) };
    let first = Grid::on(&on[..THREADS]).spin(LONG);
    let caller_to_third = [on[2], on[1]];
    let second = Grid::on(&caller_to_third).spin(LONG);
    let mut counter = Counter::default();
    let (mut ended, mut slowest) = (false, Duration::ZERO);
    for _ in 0..TRIES {
        let began = Instant::now();
        let (told_first, told_second) = (counter.launch(first)?, counter.launch(second)?);
        slowest = slowest.max(began.elapsed());
        // The third thread, at index 2.
        ended = told_first[2] || told_second[2];
        if ended {
            break;
        }
    }
    let message = "rounds found the kept thread in the way stop to wait early, with no thread taking its CPU";
    assert!(ended, "none of {TRIES} {message}, though its spin was to last {LONG:?}");
    assert!(slowest < LONG / 10, "a round of two launches took {slowest:?}, where a spin of {LONG:?} was to end");
    Ok(())
}

/// Kept threads of two callers that a placed grid each puts on one CPU share it while they spin, as any two threads on
/// one CPU do: each launch waits for its kept thread to get the CPU, and not for the other thread's spin to end. Each
/// caller makes 50 launches, each of which runs its two threads once, with a spin of ten seconds, and none takes a
/// tenth as long.
#[test]
fn kept_threads_of_two_callers_placed_on_one_cpu_share_it_while_they_spin() -> Result {
    let _alone = alone();
    let Some(on) = four()? else { return Ok(()) };
    let start = Arc::new(Barrier::new(2));
    // Each caller's thread is on a CPU of its own, and the kept thread of each is on the third.
    let callers = [[on[0], on[2]], [on[1], on[2]]].map(|placed| {
        let start = Arc::clone(&start);
        thread::spawn(move || -> Result<Duration> {
            start.wait();
            let mut slowest = Duration::ZERO;
            for _ in 0..50 {
                let ran = Arc::new(AtomicU64::new(0));
                let into = Arc::clone(&ran);
                let began = Instant::now();
                launch(Grid::on(&placed).spin(LONG), move |_| {
                    into.fetch_add(1, Relaxed);
                })?;
                slowest = slowest.max(began.elapsed());
                assert_eq!(
                    ran.load(Relaxed),
                    2,
                    "a launch of two threads on {placed:?} ran its kernel that many times"
                );
            }
            Ok(slowest)
        })
    });
    for caller in callers {
        let slowest = caller.join().map_err(|panic| format!("a caller panicked: {panic:?}"))??;
        assert!(slowest < LONG / 10, "a launch took {slowest:?}, where the other caller's spin of {LONG:?} was to end");
    }
    Ok(())
}
