//! What a launch asks of the system once the threads of its grid are kept: a launch on one thread
//! asks for nothing, and one on many starts no thread and maps no stack. Placed or not, a launch
//! asks about no CPU once the threads are where it places them: a thousand launches of one
//! placement make no call about CPUs, and a launch that changes the placement makes one call for
//! each thread that moves, the caller's among them, and no other. A launch that starts threads puts
//! each on its CPU before it starts the next, and none of these places memory or sets a policy for
//! it. Each test runs itself again under `strace`, and skips where there is none.

use std::io;
use std::path::PathBuf;
use std::process::Command;
use std::{env, fs};

use kernite_cpu::{Grid, cpus, launch, most};

type Result<T = (), E = Box<dyn std::error::Error + Send + Sync>> = std::result::Result<T, E>;

/// What names the process `strace` runs, and says how many threads its grid has.
const CHILD: &str = "KERNITE_CPU_CHILD";

/// What tells the process `strace` runs to place its grid, on the first CPUs it may run on.
const PLACED: &str = "KERNITE_CPU_PLACED";

/// The files that are not there, which the process asks for between its steps, for `strace` to show
/// where one ends and the next begins: `/kernite-cpu-mark-0` and so on.
const MARK: &str = "/kernite-cpu-mark-";

/// How many threads the grid of the process `strace` runs has, if this is that process.
fn child() -> Result<Option<usize>> {
    Ok(env::var(CHILD).ok().map(|threads| threads.parse::<usize>()).transpose()?)
}

/// Ends a step of the process `strace` runs, and begins the next.
fn mark(step: usize) {
    let _ = fs::metadata(format!("{MARK}{step}"));
}

/// The first `threads` of the CPUs the process may run on, for a grid of that many threads to be placed on.
fn first_cpus(threads: usize) -> Result<Vec<usize>> {
    let cpus = cpus()?;
    Ok(cpus.get(..threads).ok_or("the grid has more threads than the process has CPUs")?.to_vec())
}

/// What the process `strace` runs does: a launch, which starts the threads, then a thousand.
fn launches(threads: usize) -> Result {
    let on = first_cpus(threads)?;
    let placed = env::var_os(PLACED).is_some();
    let grid = || if placed { Grid::on(&on) } else { Grid::new(threads) };
    launch(grid(), |_| {})?;
    mark(0);
    for _ in 0..1000 {
        launch(grid(), |_| {})?;
    }
    mark(1);
    Ok(())
}

/// Where `strace` writes the calls of the process the test `name` runs again under it.
fn log_of(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("{name}.strace"))
}

/// The calls `strace` shows in each step of the process the test `name` runs again under it, a
/// grid of `threads` placed or not, between one mark and the next; none if there is no `strace`. What the process
/// prints is kept, for a failure to show it, not left among the output of `cargo test`.
fn steps_of(name: &str, threads: usize, placed: bool) -> Result<Option<Vec<Vec<String>>>> {
    let log = log_of(name);
    let strace = env::var_os("STRACE").unwrap_or_else(|| "strace".into());
    let run = match Command::new(strace)
        .args(["-f", "-o"])
        .arg(&log)
        .arg(env::current_exe()?)
        .args(["--exact", name, "--test-threads=1"])
        .env(CHILD, threads.to_string())
        .envs(placed.then_some((PLACED, "1")))
        .output()
    {
        Ok(run) => run,
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            eprintln!("no strace here: skipped; set STRACE to run it");
            return Ok(None);
        }
        Err(e) => return Err(format!("strace did not run: {e}").into()),
    };
    let text = format!("{}{}", String::from_utf8_lossy(&run.stdout), String::from_utf8_lossy(&run.stderr));
    assert!(run.status.success(), "{name}, run again under strace, failed:\n{text}");
    let calls = fs::read_to_string(log)?;
    // Each call is shown with the number of its thread. The first thread's are libtest's, which runs the test on a
    // thread of its own and waits for it: none is kernite's, and `strace` shows each where it sees it, which may come
    // after a call the test's thread made later, under load. They are left out.
    let libtest = calls.split_once(' ').map(|(thread, _)| format!("{thread} ")).ok_or("strace showed no call")?;
    let calls: Vec<&str> = calls.lines().filter(|call| !call.starts_with(&libtest)).collect();
    // The quote ends the name of the file, for `-1` not to be found in `-10`.
    let at = |step: usize| calls.iter().position(|call| call.contains(&format!("{MARK}{step}\"")));
    let marks: Vec<usize> = (0..).map_while(at).collect();
    assert!(marks.len() >= 2, "{name}: the marks are not in the calls");
    Ok(Some(
        marks
            .iter()
            .zip(marks.iter().skip(1))
            .map(|(&from, &to)| {
                calls.get(from + 1..to).unwrap_or_default().iter().map(|call| call.to_string()).collect()
            })
            .collect(),
    ))
}

/// How a failure shows `calls`: the first few.
fn first<S: AsRef<str>>(calls: &[S]) -> String {
    calls.iter().take(5).map(|call| call.as_ref()).collect::<Vec<_>>().join("\n")
}

/// Whether a call `strace` shows starts a thread or maps a stack.
fn starts_a_thread_or_maps_a_stack(call: &str) -> bool {
    call.contains("clone") || call.contains("mmap")
}

/// Whether a call `strace` shows is about the CPUs a thread may run on.
fn is_about_cpus(call: &str) -> bool {
    call.contains("affinity")
}

/// The calls that place memory or set a policy for it, as `strace` names them.
const MEMORY: [&str; 5] = ["mbind(", "set_mempolicy(", "set_mempolicy_home_node(", "migrate_pages(", "move_pages("];

/// Says if a call of the whole run of the process the test `name` ran again under `strace`, from its
/// start, is one that places memory or sets a policy for it.
fn assert_places_no_memory(name: &str) -> Result {
    let calls = fs::read_to_string(log_of(name))?;
    let memory: Vec<&str> = calls.lines().filter(|call| MEMORY.iter().any(|m| call.contains(m))).collect();
    let message = "calls place memory or set a policy for it, as";
    assert!(memory.is_empty(), "{name}: {} {message}:\n{}", memory.len(), first(&memory));
    Ok(())
}

/// A thousand launches on one thread, after the first, which asks how many threads the process
/// runs and, if the grid is placed, puts the caller on its CPU: `strace` shows no call between the
/// two files the process asks for.
#[test]
fn a_launch_on_one_thread_makes_no_system_call() -> Result {
    if let Some(threads) = child()? {
        return launches(threads);
    }
    for placed in [false, true] {
        let Some(steps) = steps_of("a_launch_on_one_thread_makes_no_system_call", 1, placed)? else { return Ok(()) };
        let calls = &steps[0];
        assert!(
            calls.is_empty(),
            "placed {placed}: {} calls between the first launch and the last, as:\n{}",
            calls.len(),
            first(calls)
        );
    }
    Ok(())
}

/// A thousand launches on many threads, after the first, which starts them and, if the grid is
/// placed, puts each on its CPU: `strace` shows no thread started, no stack mapped and no CPU asked
/// about between the two files, in a grid placed or not. Threads that find no kernel for a while
/// sleep, and a launch wakes them with calls of another kind, which are allowed.
#[test]
fn a_launch_on_many_threads_starts_no_thread_maps_no_stack_and_asks_about_no_cpu() -> Result {
    if let Some(threads) = child()? {
        return launches(threads);
    }
    let threads = most()?.min(6);
    if threads < 2 {
        eprintln!("the process runs one thread at once, so no grid has many: skipped");
        return Ok(());
    }
    let name = "a_launch_on_many_threads_starts_no_thread_maps_no_stack_and_asks_about_no_cpu";
    for placed in [false, true] {
        let Some(steps) = steps_of(name, threads, placed)? else { return Ok(()) };
        let new: Vec<String> = steps[0]
            .iter()
            .filter(|call| starts_a_thread_or_maps_a_stack(call) || is_about_cpus(call))
            .cloned()
            .collect();
        let message = "calls start a thread, map a stack or ask about a CPU, as";
        assert!(new.is_empty(), "placed {placed}: {} {message}:\n{}", new.len(), first(&new));
    }
    Ok(())
}

/// What a step of the process `strace` runs for the test of changes launches, and the calls it is
/// to make about CPUs: `reads` of the CPUs the caller's thread may run on, and `sets` of the CPUs a
/// thread may run on.
struct Step {
    /// The grid's CPUs, none if it is not placed.
    placed: Option<Vec<usize>>,
    /// How many threads the grid has.
    threads: usize,
    reads: usize,
    sets: usize,
}

/// The steps for a grid of `threads` threads, two at least, on `cpus`, those the caller's thread
/// may run on, which are as many. They begin with the grid not placed, which has started the
/// threads, so that the first step is the first to place it. It is an error where `cpus` are
/// fewer than `threads`, or `threads` are fewer than two.
fn script(cpus: &[usize], threads: usize) -> Result<Vec<Step>> {
    let on = |placed: &[usize], reads, sets| Step { placed: Some(placed.to_vec()), threads: placed.len(), reads, sets };
    let anywhere = |threads, reads, sets| Step { placed: None, threads, reads, sets };
    let (first, spare) = cpus.split_at_checked(threads).ok_or("the grid has more threads than the process has CPUs")?;
    let ((two, rest), (_, fewer)) =
        first.split_first_chunk::<2>().zip(first.split_last()).ok_or("a grid has two threads at least")?;
    // The first two threads swap CPUs, then the last moves to a CPU none has, if there is one.
    let swapped: Vec<usize> = two.iter().rev().chain(rest).copied().collect();
    let mut last = swapped.clone();
    let mut script = vec![
        // From not placed: the CPUs the caller's thread may run on are read, then each thread moves.
        on(first, 1, threads),
        on(first, 0, 0),
        on(&swapped, 0, 2),
    ];
    if let (Some(&spare), Some(slot)) = (spare.first(), last.last_mut()) {
        *slot = spare;
        script.push(on(&last, 0, 1));
    }
    script.extend([
        on(&last, 0, 0),
        // Not placed: the caller's thread, and each kept thread, go back to the CPUs it could run on.
        anywhere(threads, 0, threads),
        anywhere(threads, 0, 0),
        // Placed on two threads only, then not placed: the others were never moved, so only the
        // two go back.
        on(two, 1, 2),
        anywhere(threads, 0, 2),
        // Placed again: the caller's thread is free, so its CPUs are read, and all move.
        on(first, 1, threads),
        // A grid of fewer threads finds them where it places them, and leaves the last where it was.
        on(fewer, 0, 0),
        on(first, 0, 0),
        // A grid of the caller's thread alone that is not placed lets every kept thread go, too.
        anywhere(1, 0, threads),
        anywhere(threads, 0, 0),
    ]);
    Ok(script)
}

/// What the process `strace` runs for the test of changes does: a launch not placed, which starts
/// the threads and asks how many the process runs, then the steps, a mark before each.
fn changes(threads: usize) -> Result {
    launch(Grid::new(threads), |_| {})?;
    let script = script(&cpus()?, threads)?;
    for (step, Step { placed, threads, .. }) in script.iter().enumerate() {
        mark(step);
        let grid = match placed {
            Some(cpus) => Grid::on(cpus),
            None => Grid::new(*threads),
        };
        launch(grid, |_| {})?;
    }
    mark(script.len());
    Ok(())
}

/// A launch that changes the placement makes one call for each thread that moves, the caller's
/// among them, and no other, and reads the CPUs the caller's thread may run on only when it places
/// that thread and a launch before did not. A launch of the placement the threads are on makes
/// none, and a launch that places none sends back the threads it moves with a call for each. No
/// call of the whole run, the first launch's among them, is `mbind`, `set_mempolicy`,
/// `set_mempolicy_home_node`, `migrate_pages` or `move_pages`: kernite places no memory and sets no
/// policy, as GUIDE.md says.
#[test]
fn a_launch_that_changes_the_placement_makes_a_call_for_each_thread_that_moves_and_no_other() -> Result {
    if let Some(threads) = child()? {
        return changes(threads);
    }
    let all = cpus()?;
    let threads = most()?.min(all.len()).min(6);
    if threads < 2 {
        eprintln!("the process runs one thread at once, so no grid has many: skipped");
        return Ok(());
    }
    let name = "a_launch_that_changes_the_placement_makes_a_call_for_each_thread_that_moves_and_no_other";
    let Some(steps) = steps_of(name, threads, false)? else { return Ok(()) };
    let script = script(&all, threads)?;
    assert_eq!(steps.len(), script.len());
    for (i, (calls, step)) in steps.iter().zip(&script).enumerate() {
        let count = |call: &str| calls.iter().filter(|line| line.contains(call)).count();
        let (reads, sets) = (count("sched_getaffinity("), count("sched_setaffinity("));
        let about = calls.iter().filter(|call| is_about_cpus(call)).cloned().collect::<Vec<_>>();
        assert_eq!(
            (reads, sets),
            (step.reads, step.sets),
            "step {i}, {:?} of {} threads: reads and sets of the CPUs a thread may run on, as:\n{}",
            step.placed,
            step.threads,
            first(&about)
        );
        let new: Vec<String> = calls.iter().filter(|call| starts_a_thread_or_maps_a_stack(call)).cloned().collect();
        assert!(new.is_empty(), "step {i}: {} calls start a thread or map a stack, as:\n{}", new.len(), first(&new));
    }
    assert_places_no_memory(name)?;
    Ok(())
}

/// What the process `strace` runs for the test of starts does: one launch that places a grid, which
/// starts its threads, between two marks.
fn starts(threads: usize) -> Result {
    let on = first_cpus(threads)?;
    mark(0);
    launch(Grid::on(&on), |_| {})?;
    mark(1);
    Ok(())
}

/// A launch that places a grid and starts its threads puts the caller's thread on its CPU, then
/// puts each thread on its CPU right after it starts it, before it starts the next: a thread that
/// starts on the caller's CPU, where the caller has to run to start the next, never spins there.
/// `strace` shows the calls that start a thread, and the call that puts a thread on a CPU, in that
/// order, each with the number of the thread. The launch places no memory either.
#[test]
fn a_launch_that_starts_threads_puts_each_on_its_cpu_before_it_starts_the_next() -> Result {
    if let Some(threads) = child()? {
        return starts(threads);
    }
    let all = cpus()?;
    let threads = most()?.min(all.len()).min(6);
    if threads < 2 {
        eprintln!("the process runs one thread at once, so no grid has many: skipped");
        return Ok(());
    }
    let name = "a_launch_that_starts_threads_puts_each_on_its_cpu_before_it_starts_the_next";
    let Some(steps) = steps_of(name, threads, false)? else { return Ok(()) };
    // Each call that starts a thread or puts one on a CPU, with the number of the thread: the one
    // started, which a call ends with, or the one placed, which is its first argument.
    let events: Vec<(&str, u64)> = steps[0]
        .iter()
        .filter_map(|call| match call.split_once("sched_setaffinity(") {
            Some((_, args)) => Some(("place", args.split(',').next()?.parse().ok()?)),
            None if call.contains("clone") => Some(("start", call.rsplit_once("= ")?.1.parse().ok()?)),
            None => None,
        })
        .collect();
    let what: Vec<&str> = events.iter().map(|&(what, _)| what).collect();
    let mut want = vec!["place"];
    want.extend(["start", "place"].repeat(threads - 1));
    assert_eq!(what, want, "calls, in order, with the numbers of their threads: {events:?}");
    for pair in events[1..].chunks(2) {
        assert_eq!(pair[0].1, pair[1].1, "a thread is put on its CPU right after it starts: {events:?}");
    }
    assert_places_no_memory(name)?;
    Ok(())
}
