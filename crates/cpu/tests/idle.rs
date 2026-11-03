//! What the threads a launch keeps do while no kernel runs for them. This file holds one test,
//! which has its process to itself: no other thread sleeps, uses the CPU or ends while it
//! counts.

use std::fs;
use std::thread;
use std::time::Duration;

use kernite_cpu::{Grid, SPIN, launch, most};

/// How many threads the process has, as Linux lists them in `/proc/self/task`; none if it does
/// not.
fn tasks() -> Option<usize> {
    Some(fs::read_dir("/proc/self/task").ok()?.count())
}

/// How many times the process's threads have gone to sleep, and for how many nanoseconds they
/// have used the CPU, as Linux counts them for each thread in `/proc/self/task`; none if it
/// does not.
fn counts() -> Option<(u64, u64)> {
    let (mut sleeps, mut cpu) = (0, 0);
    for task in fs::read_dir("/proc/self/task").ok()? {
        let task = task.ok()?.path();
        let status = fs::read_to_string(task.join("status")).ok()?;
        let asleep = status.lines().find_map(|line| line.strip_prefix("voluntary_ctxt_switches:"))?;
        sleeps += asleep.trim().parse::<u64>().ok()?;
        let schedstat = fs::read_to_string(task.join("schedstat")).ok()?;
        cpu += schedstat.split_whitespace().next()?.parse::<u64>().ok()?;
    }
    Some((sleeps, cpu))
}

/// A thread with no kernel to run spins for the next, which comes soon in a loop of launches:
/// none of a thousand finds the threads asleep, but for those the system leaves waiting. Left
/// alone for longer than they spin, the threads sleep and use no CPU: spinning, each would use
/// all of the time it is left. They end with the thread whose launches they ran.
#[test]
fn kept_threads_spin_for_a_launch_that_comes_soon_then_sleep_and_end_with_their_caller() {
    let threads = most().unwrap().min(4);
    if threads < 2 {
        return eprintln!("the machine runs one thread at once, so no grid keeps a thread: skipped");
    }
    let Some(before) = tasks() else { return eprintln!("no /proc/self/task: skipped") };
    let caller = thread::spawn(move || {
        launch(Grid::new(threads), |_| {}).unwrap();
        let (asleep, _) = counts().unwrap();
        for _ in 0..1000 {
            launch(Grid::new(threads), |_| {}).unwrap();
        }
        let (now, _) = counts().unwrap();
        let chances = 1000 * (threads as u64 - 1);
        assert!(now - asleep < chances / 2, "{} sleeps in {chances} chances to sleep", now - asleep);

        thread::sleep(10 * SPIN);
        let (_, used) = counts().unwrap();
        thread::sleep(Duration::from_millis(100));
        let (_, later) = counts().unwrap();
        assert!(later - used < 10_000_000, "{} ns of CPU in 100 ms", later - used);
        assert_eq!(tasks().unwrap(), before + threads);
    });
    if let Err(panic) = caller.join() {
        std::panic::resume_unwind(panic);
    }
    for _ in 0..500 {
        if tasks().unwrap() == before {
            return;
        }
        thread::sleep(Duration::from_millis(10));
    }
    panic!(
        "the process has {} threads, as it had {before}, five seconds after the thread that launched ended",
        tasks().unwrap()
    );
}
