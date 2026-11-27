//! The threads a launch starts are the threads each launch after it runs on. This file holds one
//! test, which has its process to itself: no other launch takes a thread the grid is waiting for.

use std::path::PathBuf;
use std::sync::{Arc, Mutex, PoisonError};
use std::thread::{self, ThreadId};

use kernite_cpu::{Grid, launch, most};

mod waits;
use waits::{Result, asleep, me, until};

/// The thread each of the grid's threads ran on, by its index, and its directory in `/proc`.
fn threads_of(grid: usize) -> Result<Vec<(ThreadId, PathBuf)>> {
    let seen = Arc::new(Mutex::new(vec![]));
    let into = Arc::clone(&seen);
    launch(Grid::new(grid), move |place| {
        let task = me().map_err(|e| e.to_string());
        into.lock().unwrap_or_else(PoisonError::into_inner).push((place.thread(), thread::current().id(), task));
    })?;
    let mut seen = std::mem::take(&mut *seen.lock().unwrap_or_else(PoisonError::into_inner));
    seen.sort_by_key(|&(t, ..)| t);
    seen.into_iter().map(|(_, id, task)| Ok((id, task?))).collect()
}

/// The first launch of a grid starts its threads, the caller's the first. Launch after launch
/// runs on the same ones, a thousand launches back to back and then one after each kept thread has
/// slept; and a smaller grid runs on some of them. Thread ids are never used again, so a thread
/// started meanwhile would show.
#[test]
fn the_first_launch_starts_the_threads_and_every_launch_after_it_runs_on_them() -> Result {
    let grid = most()?.min(6);
    let first = threads_of(grid)?;
    assert_eq!(first[0].0, thread::current().id());
    assert!(first.iter().enumerate().all(|(i, id)| !first[i + 1..].contains(id)), "{first:?}");
    for _ in 0..1000 {
        assert_eq!(threads_of(grid)?, first);
    }
    for (_, task) in &first[1..] {
        until(&format!("{} asleep", task.display()), || asleep(task))?;
    }
    assert_eq!(threads_of(grid)?, first);
    for smaller in 1..grid {
        assert!(threads_of(smaller)?.iter().all(|thread| first.contains(thread)), "{smaller} threads");
    }
    Ok(())
}
