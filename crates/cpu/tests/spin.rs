//! What a launch that moves threads does to the spin of the kept threads. A thread that may run on
//! one CPU alone cannot run while another spins there, and a kept thread spins for [`SPIN`] after
//! its launch, so a launch that moves a thread onto the CPU of a kept thread would wait for the
//! spin to end. It ends the spin of each kept thread on a CPU a thread moves to, and of each the
//! system places, which may spin on any, before anything moves, and the spin of no other: a launch
//! that moves none ends none.
//!
//! No test here times a launch, for other work on the machine slows a launch by as much as the spin
//! it looks for. A test counts, as Linux does. Each thread of a launch counts itself as its kernel
//! starts: for how long it has used a CPU, how many times it has stopped to wait, and how many
//! times another thread has taken its CPU, all as of the last time it stopped running. The counts
//! of two launches that use a thread, one after the other, say what it did between them. A thread
//! told to stop spinning stops to wait after a few microseconds of CPU, with no thread taking its
//! CPU. One that is not told spins until its spin runs out, which takes `SPIN` of CPU, or until
//! another thread takes its CPU, and does not stop to wait before. Other work on the machine delays
//! threads, and so changes how long one spins before a launch tells it, but it never makes a thread
//! stop to wait early, nor spin on after it is told. A test of a thread that must be told needs one
//! round of many in which no thread was delayed; a test of one that must not is sure of it whatever
//! the machine does.

use std::fs::File;
use std::os::unix::fs::FileExt;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};
use std::time::Duration;

use kernite_cpu::{Grid, SPIN, cpus, launch, most};

/// How many rounds a test makes at most to see a thread told to stop spinning. Many rounds show it
/// on a quiet machine, and one in a few hundred beside busy work on every CPU, where a launch
/// takes milliseconds; a test ends at the first that does.
const TRIES: usize = 5000;

/// How many rounds a test makes to see that no thread is told to stop spinning.
const ROUNDS: usize = 300;

/// How many threads a grid of these tests has.
const THREADS: usize = 3;

/// The tests place threads on the same CPUs, and a thread that spins on one keeps another from
/// running there, so they run one at a time.
fn alone() -> MutexGuard<'static, ()> {
    static ONE: Mutex<()> = Mutex::new(());
    ONE.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The first four CPUs the caller's thread may run on, one more than a grid has threads, if the
/// machine runs that many threads at once; none if not.
fn four() -> Option<Vec<usize>> {
    let all = cpus().unwrap();
    if all.len() < THREADS + 1 || most().unwrap() < THREADS {
        eprintln!("the machine has fewer than four CPUs, or runs fewer than three threads at once: skipped");
        return None;
    }
    Some(all[..4].to_vec())
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
    fn of(status: &File, schedstat: &File) -> Count {
        let (mut times, mut switches) = ([0; 256], [0; 8192]);
        let (times_size, switches_size) =
            (schedstat.read_at(&mut times, 0).unwrap(), status.read_at(&mut switches, 0).unwrap());
        let (times, switches) =
            (str::from_utf8(&times[..times_size]).unwrap(), str::from_utf8(&switches[..switches_size]).unwrap());
        let field = |name: &str| -> u64 {
            switches
                .lines()
                .find_map(|line| line.strip_prefix(name)?.strip_prefix(':'))
                .unwrap()
                .trim()
                .parse()
                .unwrap()
        };
        Count {
            cpu: Duration::from_nanos(times.split_whitespace().next().unwrap().parse().unwrap()),
            stopped: field("voluntary_ctxt_switches"),
            displaced: field("nonvoluntary_ctxt_switches"),
        }
    }

    /// Whether the thread was told to stop spinning between `before` and this count: it stopped to
    /// wait after less than half of `SPIN` of CPU, and no thread took its CPU. A thread whose spin
    /// runs out has used `SPIN` of CPU when it stops, or has had its CPU taken.
    fn told_since(&self, before: &Count) -> bool {
        self.stopped > before.stopped
            && self.displaced == before.displaced
            && self.cpu.saturating_sub(before.cpu) < SPIN / 2
    }
}

/// Launches that count the threads they run. A count is as of the last time the thread stopped
/// running, so the CPU a thread used between two counts is that of all the time it was awake: short
/// when a launch tells it to stop soon after it wakes, as the launches of a test do, one after the
/// other.
#[derive(Default)]
struct Counter {
    /// The files in `/proc` of each thread. A thread opens its own in the first launch it runs in,
    /// and reads them again in the ones after, which costs less than opening.
    files: Arc<[OnceLock<(File, File)>; THREADS]>,
    /// The count of each thread at the last launch that used it.
    last: [Option<Count>; THREADS],
}

impl Counter {
    /// Launches `grid`, and says, by thread, which of its kept threads were told to stop spinning
    /// since the last launch that used them. The caller's thread, the first, is never one.
    fn launch(&mut self, grid: Grid<'_>) -> [bool; THREADS] {
        let (files, counts) = (Arc::clone(&self.files), Arc::<[OnceLock<Count>; THREADS]>::default());
        let into = Arc::clone(&counts);
        launch(grid, move |place| {
            let t = place.thread() as usize;
            // `/proc/thread-self` is the thread that opens it.
            let (status, schedstat) = files[t].get_or_init(|| {
                (File::open("/proc/thread-self/status").unwrap(), File::open("/proc/thread-self/schedstat").unwrap())
            });
            let _ = into[t].set(Count::of(status, schedstat));
        })
        .unwrap();
        let mut told = [false; THREADS];
        for (t, count) in counts.iter().enumerate().skip(1) {
            if let Some(&count) = count.get() {
                told[t] = self.last[t].replace(count).is_some_and(|before| count.told_since(&before));
            }
        }
        told
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
fn a_launch_that_moves_a_thread_onto_the_cpu_of_a_kept_thread_ends_its_spin_before_it_moves() {
    let _alone = alone();
    let Some(on) = four() else { return };
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
        let ended = (0..TRIES).any(|_| {
            let (told_first, told_second) = (counter.launch(first), counter.launch(second));
            told_first[in_the_way] || told_second[in_the_way]
        });
        let message = "rounds found the kept thread in the way stop to wait early, with no thread taking its CPU";
        assert!(ended, "{what}: none of {TRIES} {message}");
    }
}

/// A launch ends the spin of no kept thread that no thread moves onto the CPU of: not when it
/// moves none, and not when the CPU a thread moves to is free. A round is two launches: three
/// threads on the first three CPUs; and a second that uses the first kept thread and not the
/// second. In no round is either kept thread told to stop spinning, by either launch.
#[test]
fn a_launch_ends_the_spin_of_no_kept_thread_that_no_thread_moves_onto_the_cpu_of() {
    let _alone = alone();
    let Some(on) = four() else { return };
    let first = Grid::on(&on[..THREADS]);
    let (none_moves, to_a_free_cpu) = ([on[0], on[1]], [on[3], on[1]]);
    let cases = [
        ("no thread moves", Grid::on(&none_moves)),
        ("the caller's thread goes to a CPU where none spins", Grid::on(&to_a_free_cpu)),
    ];
    for (what, second) in cases {
        let mut counter = Counter::default();
        let rounds_told = (0..ROUNDS)
            .filter(|_| {
                let (told_first, told_second) = (counter.launch(first), counter.launch(second));
                told_first.iter().chain(&told_second).any(|&told| told)
            })
            .count();
        let message = "rounds found a kept thread told to stop spinning, where no thread moved onto its CPU";
        assert_eq!(rounds_told, 0, "{what}: {rounds_told} of {ROUNDS} {message}");
    }
}

/// A launch that moves a thread ends the spin of the kept threads the system places, which may
/// spin on any CPU: in some round, each stops to wait after a few microseconds of CPU, and no
/// thread has taken its CPU. A round is a launch of three threads that the system places, and
/// a launch that moves the caller's thread.
#[test]
fn a_launch_that_moves_a_thread_ends_the_spin_of_the_kept_threads_the_system_places() {
    let _alone = alone();
    let Some(on) = four() else { return };
    let (anywhere, here) = (Grid::new(THREADS), Grid::on(&on[..1]));
    let mut counter = Counter::default();
    let mut ended = [false; 2];
    for _ in 0..TRIES {
        let told = counter.launch(anywhere);
        counter.launch(here);
        ended = [ended[0] || told[1], ended[1] || told[2]];
        if ended == [true; 2] {
            break;
        }
    }
    let message = "stopped to wait early, with no thread taking its CPU, in some of";
    assert_eq!(ended, [true; 2], "kept threads the system places did not all stop: {ended:?} {message} {TRIES} rounds");
}

/// A kept thread whose spin a launch ended spins again after the next launch that uses it. A round
/// is three launches: the second moves the caller's thread onto the CPU of the third thread, which
/// it does not use, and ends its spin; and the first launch twice, which puts the caller back, and
/// uses the third thread each time. In some round, the thread stops to wait early between the
/// launch before the round and the first of the two. In no round does it stop to wait early
/// between the two: a thread that is told to stop spinning stays told until a launch uses it, and a
/// launch that does not clear that leaves the thread to sleep at once, where it was to spin.
#[test]
fn a_kept_thread_whose_spin_a_launch_ended_spins_again_after_the_next_launch_that_uses_it() {
    let _alone = alone();
    let Some(on) = four() else { return };
    let (first, caller_to_third) = (Grid::on(&on[..THREADS]), [on[2], on[1]]);
    let second = Grid::on(&caller_to_third);
    let mut counter = Counter::default();
    let (mut ended, mut asleep_again, mut rounds) = (false, 0, 0);
    while rounds < ROUNDS || (!ended && rounds < TRIES) {
        counter.launch(second);
        // The third thread, at index 2.
        ended |= counter.launch(first)[2];
        asleep_again += usize::from(counter.launch(first)[2]);
        rounds += 1;
    }
    let message = "rounds found the thread stop to wait early after a launch that used it";
    assert!(ended, "none of {rounds} rounds found the thread the second launch left stop to wait early");
    assert_eq!(asleep_again, 0, "{asleep_again} of {rounds} {message}");
}
