//! How long the threads of a grid wait spinning, and what a launch that moves threads does to the spin of the kept
//! threads. A thread waits spinning for as long as its grid says, `kernite_cpu::SPIN` if the grid says none, and then
//! sleeps: a kept thread for the next launch, and the caller for each of the others to end.
//!
//! A thread that may run on one CPU alone cannot run while another spins there, and a kept thread spins after its
//! launch, so a launch that moves a thread onto the CPU of a kept thread would wait for the spin to end. It ends the
//! spin of each kept thread on a CPU a thread moves to, and of each the system places, which may spin on any, and the
//! spin of no other: a launch that moves none ends none.
//!
//! No test here holds how long a thing takes, which other work on the machine stretches: a probe run alone measures
//! it. A test holds what Linux says of a thread. One whose spin is nothing, or has run out, or a launch has ended,
//! sleeps, and a test waits for that as long as it takes. One whose spin is without end runs for as long as it waits:
//! a test waits until Linux has counted it running for twice `SPIN` since a launch, longer than a spin of `SPIN` lasts,
//! and sees it awake then. However busy the machine is, that wait ends for a thread that spins, and never for one that
//! sleeps after `SPIN`, which the test says once it has waited `HANG`.

use std::fs;
use std::path::{Path, PathBuf};
use std::slice;
use std::sync::atomic::{
    AtomicBool, AtomicUsize,
    Ordering::{AcqRel, Acquire, Release},
};
use std::sync::{Arc, Barrier, Mutex, MutexGuard, PoisonError, mpsc};
use std::thread;
use std::time::Duration;

use kernite_cpu::{Grid, SPIN, cpus, launch, launch_blocks, most};

mod waits;
use waits::{HANG, Result, asleep, me, until};

/// How many threads a grid of these tests has.
const THREADS: usize = 3;

/// A spin that never runs out: only the next launch ends it, or a launch that moves a thread onto its CPU, or the end
/// of its caller.
const WITHOUT_END: Duration = Duration::MAX;

/// How many launches a test makes of threads that must spin after each.
const ROUNDS: usize = 10;

/// The tests spin threads on the same CPUs, where each test's would share them with the others', so they run one at a
/// time: no result depends on it, but each test takes only what its own launches take.
fn alone() -> MutexGuard<'static, ()> {
    static ONE: Mutex<()> = Mutex::new(());
    ONE.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The first four CPUs the caller's thread may run on, one more than a grid has threads, if the process runs that many
/// threads at once; none if not.
fn four() -> Result<Option<[usize; THREADS + 1]>> {
    let four = cpus()?.first_chunk().copied();
    if four.is_none() || most()? < THREADS {
        eprintln!(
            "the caller's thread may run on fewer than four CPUs, or the process runs fewer than three threads at \
             once: skipped"
        );
        return Ok(None);
    }
    Ok(four)
}

/// The name of the thread whose directory in `/proc` is `task`: `grid thread 2`.
fn name(task: &Path) -> String {
    fs::read_to_string(task.join("comm")).map_or_else(|e| e.to_string(), |name| name.trim_end().to_string())
}

/// How long the thread whose directory in `/proc` is `task` has run on a CPU, as Linux has counted it: it counts the
/// time of a thread that runs at each tick of its clock, a millisecond apart here.
fn ran(task: &Path) -> Result<Duration> {
    let schedstat = fs::read_to_string(task.join("schedstat"))?;
    let nanos = schedstat.split_whitespace().next().ok_or("next gave None")?.parse()?;
    Ok(Duration::from_nanos(nanos))
}

/// Waits until Linux has counted each thread of `tasks` running for twice `SPIN` since the call, and says whether one
/// sleeps then. The first count after the call may hold time from before it too, so the wait counts from that one.
fn spun(what: &str, tasks: &[PathBuf]) -> Result {
    let counted = tasks.iter().map(|task| ran(task)).collect::<Result<Vec<_>>>()?;
    for (task, counted) in tasks.iter().zip(counted) {
        let mut from = None;
        until(&format!("{what}: {} running for {:?}", name(task), 2 * SPIN), || {
            let now = ran(task)?;
            match from {
                Some(from) => return Ok(now >= from + 2 * SPIN),
                None if now != counted => from = Some(now),
                None => {}
            }
            Ok(false)
        })?;
        if asleep(task)? {
            return Err(format!("{what}: {} sleeps, where it is to spin", name(task)).into());
        }
    }
    Ok(())
}

/// Makes `ROUNDS` launches by `launches`, and after each waits until each thread of `tasks` has spun for twice `SPIN`,
/// as `spun` says.
fn spin_after_each(what: &str, tasks: &[PathBuf], mut launches: impl FnMut() -> Result) -> Result {
    for round in 0..ROUNDS {
        launches()?;
        spun(&format!("{what}, after launch {round}"), tasks)?;
    }
    Ok(())
}

/// Launches `grid`, each of whose kept threads says where it is in `/proc`: their directories, by their place in the
/// grid, from the second.
fn kept(grid: Grid<'_>) -> Result<Vec<PathBuf>> {
    let tasks = Arc::new(Mutex::new(vec![]));
    let into = Arc::clone(&tasks);
    launch(grid, move |place| {
        if place.thread() > 0 {
            let task = me().map_err(|e| e.to_string());
            into.lock().unwrap_or_else(PoisonError::into_inner).push((place.thread(), task));
        }
    })?;
    let mut tasks = std::mem::take(&mut *tasks.lock().unwrap_or_else(PoisonError::into_inner));
    tasks.sort_by_key(|&(t, _)| t);
    tasks.into_iter().map(|(_, task)| Ok(task?)).collect()
}

/// A launch of a block for each thread of `grid`, each of which waits until all have begun: so each thread runs one,
/// each kept thread too, which then waits for the next launch as the grid says.
fn one_block_each(grid: Grid<'_>) -> Result {
    let begun = Arc::new(AtomicUsize::new(0));
    launch_blocks(grid, THREADS as u64, move |_| {
        begun.fetch_add(1, AcqRel);
        let all = until("all the blocks begun", || Ok(begun.load(Acquire) >= THREADS));
        assert!(all.is_ok(), "{all:?}");
    })?;
    Ok(())
}

/// A grid that says no spin has its threads spin for `kernite_cpu::SPIN`, 200 microseconds, as GUIDE.md says: the grid
/// `Grid::new` or `Grid::on` makes is the one that says that spin.
#[test]
fn a_grid_that_says_no_spin_has_its_threads_spin_for_200_microseconds() {
    assert_eq!(SPIN, Duration::from_micros(200));
    assert_eq!(Grid::new(2), Grid::new(2).spin(SPIN));
    assert_eq!(Grid::on(&[0, 1]), Grid::on(&[0, 1]).spin(SPIN));
}

/// A kept thread spins after a launch for as long as its grid says: one whose grid spins without end has spun for
/// twice `SPIN` after each launch, of threads and of blocks, of a grid the system places and of one placed.
#[test]
fn a_kept_thread_spins_after_a_launch_for_as_long_as_its_grid_says() -> Result {
    let _alone = alone();
    let Some([a, b, c, _]) = four()? else { return Ok(()) };
    let three = [a, b, c];
    for (how, grid) in [("Grid::new", Grid::new(THREADS)), ("Grid::on", Grid::on(&three))] {
        let grid = grid.spin(WITHOUT_END);
        let tasks = kept(grid)?;
        spin_after_each(&format!("{how}, launches of threads"), &tasks, || Ok(launch(grid, |_| {})?))?;
        spin_after_each(&format!("{how}, launches of blocks"), &tasks, || one_block_each(grid))?;
    }
    Ok(())
}

/// A kept thread sleeps once its spin is over: at once where its grid says a spin of nothing, and after
/// `kernite_cpu::SPIN` where it says none. After a launch of threads and one of blocks, of a grid the system places
/// and of one placed, the test waits until each kept thread sleeps.
#[test]
fn a_kept_thread_sleeps_once_its_spin_is_over() -> Result {
    let _alone = alone();
    let Some([a, b, c, _]) = four()? else { return Ok(()) };
    let three = [a, b, c];
    for (how, grid) in [("Grid::new", Grid::new(THREADS)), ("Grid::on", Grid::on(&three))] {
        for (spin, grid) in [("a spin of nothing", grid.spin(Duration::ZERO)), ("no spin said", grid)] {
            let tasks = kept(grid)?;
            for blocks in [false, true] {
                if blocks {
                    one_block_each(grid)?;
                } else {
                    launch(grid, |_| {})?;
                }
                for task in &tasks {
                    until(&format!("{how}, {spin}, blocks {blocks}: {} asleep", name(task)), || asleep(task))?;
                }
            }
        }
    }
    Ok(())
}

/// The caller sleeps once its spin is over, as it waits for the others to end: at once where its grid says a spin of
/// nothing, and after `kernite_cpu::SPIN` where it says none. In a launch of two threads, and in one of two blocks,
/// each part waits for the other to begin, so that each thread has one; and the part of the thread that is not the
/// caller's ends once the caller sleeps, which has no block left then. A caller that spun until the others ended
/// would never sleep, and the launch would not end.
#[test]
fn the_caller_sleeps_once_its_spin_is_over_as_it_waits_for_the_others() -> Result {
    let _alone = alone();
    if most()? < 2 {
        eprintln!("the process runs one thread at once: skipped");
        return Ok(());
    }
    let (caller, id) = (Arc::new(me()?), thread::current().id());
    for (spin, grid) in [("a spin of nothing", Grid::new(2).spin(Duration::ZERO)), ("no spin said", Grid::new(2))] {
        for blocks in [false, true] {
            let (seen, begun, at) = (Arc::new(Mutex::new(None)), Arc::new(AtomicUsize::new(0)), Arc::clone(&caller));
            let into = Arc::clone(&seen);
            let part = move || {
                begun.fetch_add(1, AcqRel);
                let both = until("the other part begun", || Ok(begun.load(Acquire) >= 2));
                if thread::current().id() != id {
                    let slept = both.and_then(|()| until("the caller asleep", || asleep(&at)));
                    *into.lock().unwrap_or_else(PoisonError::into_inner) = Some(slept.map_err(|e| e.to_string()));
                }
            };
            if blocks {
                launch_blocks(grid, 2, move |_| part())?;
            } else {
                launch(grid, move |_| part())?;
            }
            let seen = seen.lock().unwrap_or_else(PoisonError::into_inner).take();
            assert_eq!(seen, Some(Ok(())), "{spin}, blocks {blocks}: the other thread waited for the caller to sleep");
        }
    }
    Ok(())
}

/// The caller spins for as long as its grid says as it waits for the others to end. In a launch of two threads, and in
/// one of two blocks, of a grid that spins without end, each part waits for the other to begin; the part of the thread
/// that is not the caller's waits until the caller's is over, and then until the caller has spun for twice `SPIN`, as
/// `spun` says.
#[test]
fn the_caller_spins_for_as_long_as_its_grid_says_as_it_waits_for_the_others() -> Result {
    let _alone = alone();
    if most()? < 2 {
        eprintln!("the process runs one thread at once: skipped");
        return Ok(());
    }
    let (caller, id) = (Arc::new(me()?), thread::current().id());
    let grid = Grid::new(2).spin(WITHOUT_END);
    for blocks in [false, true] {
        let (seen, begun, over) =
            (Arc::new(Mutex::new(None)), Arc::new(AtomicUsize::new(0)), Arc::new(AtomicBool::new(false)));
        let (into, at) = (Arc::clone(&seen), Arc::clone(&caller));
        let part = move || {
            begun.fetch_add(1, AcqRel);
            let both = until("the other part begun", || Ok(begun.load(Acquire) >= 2));
            if thread::current().id() == id {
                over.store(true, Release);
            } else {
                let over = both.and_then(|()| until("the caller's part over", || Ok(over.load(Acquire))));
                let spun = over.and_then(|()| spun("the caller", slice::from_ref(&*at)));
                *into.lock().unwrap_or_else(PoisonError::into_inner) = Some(spun.map_err(|e| e.to_string()));
            }
        };
        if blocks {
            launch_blocks(grid, 2, move |_| part())?;
        } else {
            launch(grid, move |_| part())?;
        }
        let seen = seen.lock().unwrap_or_else(PoisonError::into_inner).take();
        assert_eq!(seen, Some(Ok(())), "blocks {blocks}: what the other thread saw of the caller");
    }
    Ok(())
}

/// A launch that moves a thread onto the CPU of a kept thread, and does not use it, ends its spin, and the thread
/// sleeps; the kept thread that the launch uses spins after it, and after each launch of that grid that follows, as
/// the grid says, without end. The first launch puts three threads on the first three CPUs, and the second moves a
/// thread onto the CPU of the third:
///
/// - the caller's thread goes where the third spins, and the second launch uses the second thread;
/// - the second thread goes where the third spins, and the caller's thread stays.
#[test]
fn a_launch_that_moves_a_thread_onto_the_cpu_of_a_kept_thread_ends_its_spin() -> Result {
    let _alone = alone();
    let Some([a, b, c, _]) = four()? else { return Ok(()) };
    let three = [a, b, c];
    let first = Grid::on(&three).spin(WITHOUT_END);
    let cases = [
        ("the caller's thread goes where the third spins", [c, b]),
        ("the second thread goes where the third spins", [a, c]),
    ];
    for (what, second) in cases {
        let second = Grid::on(&second).spin(WITHOUT_END);
        let tasks = kept(first)?;
        let [used, in_the_way] = tasks.as_slice() else { return Err("kept gave other than two threads".into()) };
        // The first launch of the second grid is the one that moves the thread.
        spin_after_each(what, slice::from_ref(used), || Ok(launch(second, |_| {})?))?;
        until(&format!("{what}: the third thread asleep"), || asleep(in_the_way))?;
    }
    Ok(())
}

/// A launch ends the spin of no kept thread that no thread moves onto the CPU of: not when it moves none, and not when
/// the CPU a thread moves to is free. After a launch of three threads on the first three CPUs, which spin without end,
/// launches of a grid that uses the first kept thread and not the second find both spinning: the first as the second
/// grid says, and the second as the first grid said, without end.
#[test]
fn a_launch_ends_the_spin_of_no_kept_thread_that_no_thread_moves_onto_the_cpu_of() -> Result {
    let _alone = alone();
    let Some([a, b, c, d]) = four()? else { return Ok(()) };
    let three = [a, b, c];
    let first = Grid::on(&three).spin(WITHOUT_END);
    for (what, second) in [("no thread moves", [a, b]), ("the caller's thread goes to a CPU where none spins", [d, b])]
    {
        let second = Grid::on(&second).spin(WITHOUT_END);
        let tasks = kept(first)?;
        spin_after_each(what, &tasks, || Ok(launch(second, |_| {})?))?;
    }
    Ok(())
}

/// A launch that moves a thread ends the spin of the kept threads the system places, which may spin on any CPU: after a
/// launch of three threads that the system places, which spin without end, a launch that places the caller's thread
/// alone, and so moves it, has each kept thread sleep.
#[test]
fn a_launch_that_moves_a_thread_ends_the_spin_of_the_kept_threads_the_system_places() -> Result {
    let _alone = alone();
    let Some([a, ..]) = four()? else { return Ok(()) };
    let tasks = kept(Grid::new(THREADS).spin(WITHOUT_END))?;
    launch(Grid::on(&[a]), |_| {})?;
    for task in &tasks {
        until(&format!("{} asleep", name(task)), || asleep(task))?;
    }
    Ok(())
}

/// A kept thread whose spin a launch ended spins again after the next launch that uses it, for as long as that grid
/// says. The launch that moves the caller's thread onto the CPU of the third thread, and does not use it, ends its
/// spin, and the thread sleeps; the first launch again, which puts the caller's thread back and uses the third, has it
/// spin without end after, and after each launch of that grid that follows.
#[test]
fn a_kept_thread_whose_spin_a_launch_ended_spins_again_after_the_next_launch_that_uses_it() -> Result {
    let _alone = alone();
    let Some([a, b, c, _]) = four()? else { return Ok(()) };
    let (three, caller_to_third) = ([a, b, c], [c, b]);
    let first = Grid::on(&three).spin(WITHOUT_END);
    let tasks = kept(first)?;
    let [_, third] = tasks.as_slice() else { return Err("kept gave other than two threads".into()) };
    launch(Grid::on(&caller_to_third).spin(WITHOUT_END), |_| {})?;
    until("the third thread asleep, its spin ended", || asleep(third))?;
    spin_after_each("the third thread", slice::from_ref(third), || Ok(launch(first, |_| {})?))?;
    Ok(())
}

/// A launch that ends the spin of a kept thread, and uses it, leaves it spinning after, for as long as that launch's
/// grid says: the spin it ends is the one that was to last after the launch before. The launch that swaps the caller's
/// thread and the first kept thread ends the spin of the kept thread and uses it, and the thread spins without end
/// after that launch, and after each launch of that grid that follows, which moves none.
#[test]
fn a_launch_that_ends_the_spin_of_a_kept_thread_and_uses_it_leaves_it_spinning_after() -> Result {
    let _alone = alone();
    let Some([a, b, c, _]) = four()? else { return Ok(()) };
    let (three, swapped) = ([a, b, c], [b, a]);
    let (first, second) = (Grid::on(&three).spin(WITHOUT_END), Grid::on(&swapped).spin(WITHOUT_END));
    let tasks = kept(first)?;
    let [moved, _] = tasks.as_slice() else { return Err("kept gave other than two threads".into()) };
    // The first launch of the second grid is the one that swaps the threads.
    spin_after_each("the first kept thread", slice::from_ref(moved), || Ok(launch(second, |_| {})?))?;
    Ok(())
}

/// Kept threads of two callers that a placed grid each puts on one CPU share it while they spin, as any two threads on
/// one CPU do: each launch waits for its kept thread to get the CPU, and not for the other's spin, which is without
/// end, to end. Each caller makes 50 launches, each of which runs its two threads once, and all end.
#[test]
fn kept_threads_of_two_callers_placed_on_one_cpu_share_it_while_they_spin() -> Result {
    let _alone = alone();
    let Some([a, b, c, _]) = four()? else { return Ok(()) };
    let (start, (ended, all)) = (Arc::new(Barrier::new(2)), mpsc::channel());
    // Each caller's thread is on a CPU of its own, and the kept thread of each is on the third.
    for placed in [[a, c], [b, c]] {
        let (start, ended) = (Arc::clone(&start), ended.clone());
        thread::spawn(move || {
            start.wait();
            let launches = (0..50).try_for_each(|_| -> Result {
                let ran = Arc::new(AtomicUsize::new(0));
                let into = Arc::clone(&ran);
                launch(Grid::on(&placed).spin(WITHOUT_END), move |_| _ = into.fetch_add(1, AcqRel))?;
                match ran.load(Acquire) {
                    2 => Ok(()),
                    n => Err(format!("a launch of two threads on {placed:?} ran its kernel {n} times").into()),
                }
            });
            let _ = ended.send(launches.map_err(|e| e.to_string()));
        });
    }
    for _ in 0..2 {
        all.recv_timeout(HANG).map_err(|_| format!("a caller's launches did not end in {HANG:?}: a hang"))??;
    }
    Ok(())
}
