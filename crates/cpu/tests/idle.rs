//! What the threads a launch keeps do while no kernel runs for them. This file holds one test,
//! which has its process to itself: no other thread starts or ends while it looks at them.

use std::fs;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};

use kernite_cpu::{Grid, SPIN, launch, most};

/// How many launches the test makes.
const LAUNCHES: usize = 1000;

/// How long the test waits for threads to sleep and to end: far longer than other work on the
/// machine keeps a thread from running.
const PATIENCE: Duration = Duration::from_secs(5);

/// How many threads the process has, as Linux lists them in `/proc/self/task`; none if it does
/// not.
fn tasks() -> Option<usize> {
    Some(fs::read_dir("/proc/self/task").ok()?.count())
}

/// A thread's name, from its directory in `/proc/self/task`.
fn name(task: &Path) -> String {
    fs::read_to_string(task.join("comm")).unwrap().trim().to_string()
}

/// The threads a grid keeps, by their directories in `/proc/self/task`: those named
/// `grid thread N`.
fn kept() -> Vec<PathBuf> {
    let all = fs::read_dir("/proc/self/task").unwrap().map(|task| task.unwrap().path());
    all.filter(|task| name(task).starts_with("grid thread ")).collect()
}

/// Whether Linux says the thread sleeps. A thread that spins is running, or waits for a CPU to
/// run on.
fn asleep(task: &Path) -> bool {
    let stat = fs::read_to_string(task.join("stat")).unwrap();
    // The state is the letter after the name, which is in brackets.
    stat.rsplit_once(") ").unwrap().1.starts_with('S')
}

/// For how long the threads have used the CPU in all, as Linux counts it. It is exact for threads
/// that sleep, and lags for those that run.
fn cpu(tasks: &[PathBuf]) -> Duration {
    tasks
        .iter()
        .map(|task| {
            let schedstat = fs::read_to_string(task.join("schedstat")).unwrap();
            Duration::from_nanos(schedstat.split_whitespace().next().unwrap().parse().unwrap())
        })
        .sum()
}

/// Waits, as long as it is patient, until each of the threads sleeps.
fn wait_for_sleep(tasks: &[PathBuf]) {
    let deadline = Instant::now() + PATIENCE;
    for task in tasks {
        while !asleep(task) {
            assert!(
                Instant::now() < deadline,
                "{} spins {PATIENCE:?} after the last launch, and does not sleep",
                name(task)
            );
            thread::sleep(SPIN);
        }
    }
}

/// Whether one of the threads is found asleep less than `SPIN` after `began`.
fn asleep_early(tasks: &[PathBuf], began: Instant) -> bool {
    while began.elapsed() < SPIN {
        for task in tasks {
            // The clock is read after the state, so the thread was asleep before the time it shows.
            if asleep(task) && began.elapsed() < SPIN {
                return true;
            }
        }
    }
    false
}

/// A thread with no kernel to run spins for [`SPIN`], for the next to come, then sleeps and uses
/// no CPU. It ends with the thread whose launches it ran.
///
/// Other work on the machine makes a thread sleep late, never early: it keeps the thread from
/// running, and the `SPIN` the thread spins for is of the clock, which runs on. So the test does
/// not count sleeps, of which other work makes more. After each launch it looks for a thread
/// asleep less than `SPIN` after the launch began. A thread that spins is found so in few
/// launches, if any: a lock may put it to sleep for a moment. One that does not spin is found so
/// in nearly all, unless the machine is too busy for the test to look in time.
///
/// Nor does other work make a thread use more than `SPIN` of CPU while it spins: a thread that
/// waits for a CPU does not use one. So the test waits, as long as it is patient, for the threads
/// to sleep. Then it counts the CPU they use in one more launch and the spin after it, and the
/// CPU they use in 100 ms of sleep.
#[test]
fn kept_threads_spin_for_a_launch_that_comes_soon_then_sleep_and_end_with_their_caller() {
    // The test looks for `SPIN` after a launch began, so a `SPIN` of nothing would let it find nothing.
    assert_eq!(SPIN, Duration::from_micros(200), "README says a kept thread spins for 200 microseconds");
    let threads = most().unwrap().min(4);
    if threads < 2 {
        return eprintln!("the machine runs one thread at once, so no grid keeps a thread: skipped");
    }
    let Some(before) = tasks() else { return eprintln!("no /proc/self/task: skipped") };
    let caller = thread::spawn(move || {
        launch(Grid::new(threads), |_| {}).unwrap();
        let kept = kept();
        assert_eq!(kept.len(), threads - 1, "a grid of {threads} keeps threads named `grid thread N`");

        let mut early = 0;
        for _ in 0..LAUNCHES {
            let began = Instant::now();
            launch(Grid::new(threads), |_| {}).unwrap();
            early += usize::from(asleep_early(&kept, began));
        }
        assert_eq!(
            tasks().unwrap(),
            before + threads,
            "the threads beyond the {before} the process had are the caller's and the grid's others, {threads} in all"
        );
        assert!(
            early < LAUNCHES / 10,
            "{early} of {LAUNCHES} launches found a kept thread asleep less than {SPIN:?} after the launch began"
        );

        wait_for_sleep(&kept);
        let rested = cpu(&kept);
        launch(Grid::new(threads), |_| {}).unwrap();
        wait_for_sleep(&kept);
        let spun = cpu(&kept);
        // A thread spins for `SPIN` and the system adds a few microseconds: four times is room to spare.
        let allowed = 4 * SPIN * kept.len() as u32;
        assert!(
            spun - rested < allowed,
            "{:?} of CPU in a launch and the spin after it, where {} threads spin for {SPIN:?} each",
            spun - rested,
            kept.len()
        );
        thread::sleep(Duration::from_millis(100));
        let later = cpu(&kept);
        assert!(
            later - spun < Duration::from_millis(10),
            "{:?} of CPU in 100 ms, from threads that sleep",
            later - spun
        );
    });
    if let Err(panic) = caller.join() {
        std::panic::resume_unwind(panic);
    }
    let deadline = Instant::now() + PATIENCE;
    loop {
        let now = tasks().unwrap();
        if now == before {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "the process has {now} threads, where it had {before}, {PATIENCE:?} after the thread that launched ended"
        );
        thread::sleep(Duration::from_millis(10));
    }
}
