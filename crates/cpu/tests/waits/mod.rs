//! What the tests of how a grid's threads wait share: what Linux says of a thread, in `/proc`, and a wait for it to say
//! something.
#![allow(dead_code, reason = "a test file takes what it needs of this module, and rustc calls the rest dead")]

use std::fs;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};

/// What a test returns: nothing, or why it failed.
pub type Result<T = (), E = Box<dyn std::error::Error + Send + Sync>> = std::result::Result<T, E>;

/// How long a test waits for what must come before it says that it never came: only a hang takes this long, so no
/// test holds how long a thing takes.
pub const HANG: Duration = Duration::from_secs(60);

/// The directory in `/proc` of the calling thread, `/proc/<pid>/task/<tid>`, which `/proc/thread-self` links to.
pub fn me() -> Result<PathBuf> {
    Ok(Path::new("/proc").join(fs::read_link("/proc/thread-self")?))
}

/// Whether Linux says that the thread whose directory in `/proc` is `task` sleeps: it waits for something, which a
/// thread that spins never does. One that spins runs, or waits for a CPU to run on.
pub fn asleep(task: &Path) -> Result<bool> {
    let stat = fs::read_to_string(task.join("stat"))?;
    // The state is the letter after the name, which is in brackets.
    Ok(stat.rsplit_once(") ").ok_or("rsplit_once gave None")?.1.starts_with('S'))
}

/// How many times the thread whose directory in `/proc` is `task` has stopped to wait, as Linux counts them: a thread
/// asleep stops no more until it has woken.
pub fn stops(task: &Path) -> Result<u64> {
    let status = fs::read_to_string(task.join("status"))?;
    let stops = status.lines().find_map(|line| line.strip_prefix("voluntary_ctxt_switches:"));
    Ok(stops.ok_or("find_map gave None")?.trim().parse()?)
}

/// Waits until `now` says it has come, and says that `what` did not if it has not in `HANG`, as in a hang.
pub fn until(what: &str, mut now: impl FnMut() -> Result<bool>) -> Result {
    let hang = Instant::now() + HANG;
    while !now()? {
        if Instant::now() > hang {
            return Err(format!("{what}: not in {HANG:?}, a hang").into());
        }
        thread::yield_now();
    }
    Ok(())
}
