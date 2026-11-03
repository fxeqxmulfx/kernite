//! What a launch asks of the system once the threads of its grid are kept: a launch on one thread
//! asks for nothing, and one on many starts no thread and maps no stack. Each test runs itself
//! again under `strace`, and skips where there is none.

use std::io;
use std::path::PathBuf;
use std::process::Command;
use std::{env, fs};

use kernite_cpu::{Grid, launch, most};

/// What names the process `strace` runs, and says how many threads its grid has.
const CHILD: &str = "KERNITE_CPU_CHILD";

/// Two files that are not there, which the process asks for after its first launch and after
/// its last, for `strace` to show where they were.
const FIRST: &str = "/kernite-cpu-marks-the-first-launch";
const LAST: &str = "/kernite-cpu-marks-the-last-launch";

/// How many threads the grid of the process `strace` runs has, if this is that process.
fn child() -> Option<usize> {
    Some(env::var(CHILD).ok()?.parse().unwrap())
}

/// What the process `strace` runs does: a launch, which starts the threads, then a thousand.
fn launches(threads: usize) {
    launch(Grid::new(threads), |_| {}).unwrap();
    let _ = fs::metadata(FIRST);
    for _ in 0..1000 {
        launch(Grid::new(threads), |_| {}).unwrap();
    }
    let _ = fs::metadata(LAST);
}

/// The calls `strace` shows between the two files, when the test `name` runs again under it with
/// a grid of `threads`; none if there is no `strace`.
fn between_the_launches(name: &str, threads: usize) -> Option<Vec<String>> {
    let log = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("{name}.strace"));
    let strace = env::var_os("STRACE").unwrap_or_else(|| "strace".into());
    let run = match Command::new(strace)
        .args(["-f", "-o"])
        .arg(&log)
        .arg(env::current_exe().unwrap())
        .args(["--exact", name, "--test-threads=1"])
        .env(CHILD, threads.to_string())
        .status()
    {
        Ok(run) => run,
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            eprintln!("no strace here: skipped; set STRACE to run it");
            return None;
        }
        Err(e) => panic!("strace did not run: {e}"),
    };
    assert!(run.success(), "{name}, run again under strace, failed");
    let calls = fs::read_to_string(log).unwrap();
    let calls: Vec<&str> = calls.lines().collect();
    let at = |file: &str| calls.iter().position(|call| call.contains(file)).expect(file);
    Some(calls[at(FIRST) + 1..at(LAST)].iter().map(|call| call.to_string()).collect())
}

/// How a failure shows `calls`: the first few.
fn first(calls: &[String]) -> String {
    calls[..calls.len().min(5)].join("\n")
}

/// A thousand launches on one thread, after the first, which asks how many threads the machine
/// runs: `strace` shows no call between the two files the process asks for.
#[test]
fn a_launch_on_one_thread_makes_no_system_call() {
    if let Some(threads) = child() {
        return launches(threads);
    }
    let Some(calls) = between_the_launches("a_launch_on_one_thread_makes_no_system_call", 1) else { return };
    assert!(calls.is_empty(), "{} calls between the first launch and the last, as:\n{}", calls.len(), first(&calls));
}

/// A thousand launches on many threads, after the first, which starts them: `strace` shows no
/// thread started and no stack mapped between the two files. Threads that find no kernel for a
/// while sleep, and a launch wakes them with calls of that kind, which are allowed.
#[test]
fn a_launch_on_many_threads_starts_no_thread_and_maps_no_stack() {
    if let Some(threads) = child() {
        return launches(threads);
    }
    let threads = most().unwrap().min(6);
    if threads < 2 {
        return eprintln!("the machine runs one thread at once, so no grid has many: skipped");
    }
    let name = "a_launch_on_many_threads_starts_no_thread_and_maps_no_stack";
    let Some(calls) = between_the_launches(name, threads) else { return };
    let new: Vec<String> = calls.into_iter().filter(|call| call.contains("clone") || call.contains("mmap")).collect();
    assert!(new.is_empty(), "{} calls start a thread or map a stack, as:\n{}", new.len(), first(&new));
}
