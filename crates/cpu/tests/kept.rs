//! The threads a launch starts are the threads each launch after it runs on. This file holds one
//! test, which has its process to itself: no other launch takes a thread the grid is waiting for.

use std::sync::{Arc, Mutex, PoisonError};
use std::thread::{self, ThreadId};

use kernite_cpu::{Grid, SPIN, launch, most};

type Result<T = (), E = Box<dyn std::error::Error + Send + Sync>> = std::result::Result<T, E>;

/// The thread each of the grid's threads ran on, by its index.
fn threads_of(grid: usize) -> Result<Vec<ThreadId>> {
    let seen = Arc::new(Mutex::new(vec![]));
    let into = Arc::clone(&seen);
    launch(Grid::new(grid), move |place| {
        into.lock().unwrap_or_else(PoisonError::into_inner).push((place.thread(), thread::current().id()))
    })?;
    let mut seen = seen.lock().unwrap_or_else(PoisonError::into_inner).clone();
    seen.sort_by_key(|(t, _)| *t);
    Ok(seen.into_iter().map(|(_, id)| id).collect())
}

/// The first launch of a grid starts its threads, the caller's the first. Launch after launch
/// runs on the same ones, a thousand launches back to back and then one after the threads have
/// slept; and a smaller grid runs on some of them. Thread ids are never used again, so a thread
/// started meanwhile would show.
#[test]
fn the_first_launch_starts_the_threads_and_every_launch_after_it_runs_on_them() -> Result {
    let grid = most()?.min(6);
    let first = threads_of(grid)?;
    assert_eq!(first[0], thread::current().id());
    assert!(first.iter().enumerate().all(|(i, id)| !first[i + 1..].contains(id)), "{first:?}");
    for _ in 0..1000 {
        assert_eq!(threads_of(grid)?, first);
    }
    thread::sleep(10 * SPIN);
    assert_eq!(threads_of(grid)?, first);
    for smaller in 1..grid {
        assert!(threads_of(smaller)?.iter().all(|id| first.contains(id)), "{smaller} threads");
    }
    Ok(())
}
