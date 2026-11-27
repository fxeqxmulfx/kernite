//! What the threads a launch keeps do while no kernel runs for them. This file holds one test, which has its process to
//! itself: no other thread starts or ends while it counts them.

use std::fs;
use std::sync::{Arc, Mutex, PoisonError};
use std::thread;

use kernite_cpu::{Grid, launch, most};

mod waits;
use waits::{Result, asleep, me, stops, until};

/// How many times the test looks at the kept threads while they sleep.
const LOOKS: usize = 1000;

/// How many threads the process has, as Linux lists them in `/proc/self/task`.
fn tasks() -> Result<usize> {
    Ok(fs::read_dir("/proc/self/task")?.count())
}

/// A kept thread sleeps once its spin is over, and then does not run until a launch comes: in 1000 looks Linux finds
/// each asleep, and counts no other stop to wait for any. A thread that woke now and then to look for a launch would
/// stop to wait again between the first look and the last, or be awake at the last. The threads end with the
/// thread whose launches they ran: once it has ended, the process has the threads it had before it.
#[test]
fn kept_threads_sleep_until_a_launch_comes_and_end_with_their_caller() -> Result {
    let threads = most()?.min(4);
    if threads < 2 {
        eprintln!("the process runs one thread at once, so no grid keeps a thread: skipped");
        return Ok(());
    }
    let before = tasks()?;
    let caller = thread::spawn(move || -> Result {
        let kept = Arc::new(Mutex::new(vec![]));
        for launches in 1..=2 {
            let into = Arc::clone(&kept);
            launch(Grid::new(threads), move |place| {
                if launches == 1 && place.thread() > 0 {
                    into.lock().unwrap_or_else(PoisonError::into_inner).push(me().map_err(|e| e.to_string()));
                }
            })?;
            // The first sleep of a thread sets up what it waits with, so the test looks at the second.
            let kept = kept.lock().unwrap_or_else(PoisonError::into_inner).clone();
            for task in kept {
                let task = task?;
                until(&format!("{} asleep after launch {launches}", task.display()), || asleep(&task))?;
            }
        }
        let kept =
            kept.lock().unwrap_or_else(PoisonError::into_inner).clone().into_iter().collect::<Result<Vec<_>, _>>()?;
        assert_eq!(
            tasks()?,
            before + threads,
            "the threads beyond the {before} the process had are the caller's and the grid's others, {threads} in all"
        );
        let first = kept.iter().map(|task| stops(task)).collect::<Result<Vec<_>>>()?;
        for look in 0..LOOKS {
            for task in &kept {
                assert!(asleep(task)?, "{} is awake at look {look}, with no launch to run", task.display());
            }
        }
        let last = kept.iter().map(|task| stops(task)).collect::<Result<Vec<_>>>()?;
        assert_eq!(last, first, "the stops to wait of each kept thread, at the first look and the last");
        Ok(())
    });
    caller.join().unwrap_or_else(|panic| std::panic::resume_unwind(panic))?;
    until("the kept threads ended with their caller", || Ok(tasks()? == before))
}
