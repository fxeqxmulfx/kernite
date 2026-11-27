//! A launch of blocks: the kernel runs once for each block, on the threads of a grid, each thread taking the next block
//! when it ends one, and the launch returns once the last block has ended. What a thread that comes late does, which
//! needs a thread kernite does not start, is in the tests of the crate's own.

use std::sync::atomic::{
    AtomicBool, AtomicU8, AtomicU64,
    Ordering::{AcqRel, Acquire, Relaxed, Release},
};
use std::sync::{Arc, Barrier, Mutex, PoisonError, mpsc};
use std::{fs, panic, thread};

use kernite_cpu::{Grid, cpus, launch, launch_blocks, most};

mod waits;
use waits::{HANG, Result, asleep, me, until};

/// The kernel runs once for each block, and for no other: 0, 1, 7, a thousand and a million blocks, on 1, 2, 6 and 12
/// threads, which the system places and which the caller does. A grid of more threads than the process runs at once is
/// left out, and says so.
#[test]
fn each_block_runs_once() -> Result {
    let (most, on) = (most()?, cpus()?);
    for threads in [1, 2, 6, 12] {
        let Some(placed) = on.get(..threads).filter(|_| threads <= most) else {
            eprintln!("{threads} threads: skipped, the process runs {most} at once");
            continue;
        };
        for (how, grid) in [("Grid::new", Grid::new(threads)), ("Grid::on", Grid::on(placed))] {
            for blocks in [0, 1, 7, 1000, 1_000_000] {
                let seen: Arc<Vec<AtomicU8>> = Arc::new((0..blocks).map(|_| AtomicU8::new(0)).collect());
                let (into, stray) = (Arc::clone(&seen), Arc::new(AtomicU64::new(0)));
                let strays = Arc::clone(&stray);
                launch_blocks(grid, blocks, move |block| {
                    if let Some(seen) = usize::try_from(block).ok().and_then(|block| into.get(block)) {
                        seen.fetch_add(1, Relaxed);
                    } else {
                        strays.fetch_add(1, Relaxed);
                    }
                })?;
                let once = seen.iter().filter(|seen| seen.load(Relaxed) == 1).count();
                let what = format!("{how} of {threads} threads, {blocks} blocks");
                assert_eq!(once as u64, blocks, "{what}: blocks that ran once");
                assert_eq!(stray.load(Relaxed), 0, "{what}: blocks that ran, with an index of no block");
            }
        }
    }
    Ok(())
}

/// A grid that `launch` refuses, `launch_blocks` refuses in the same words, whatever the blocks, none too; and no block
/// runs. They are a grid of no threads, of more than the process runs at once, and two placements: of a CPU the
/// caller's thread may not run on, and of one CPU given to two threads. A launch of more than `u64::MAX / 2` blocks is
/// refused, where the grid is fine, and one of that many ends, by the panic of a block, as no launch of that many does
/// otherwise.
#[test]
fn a_grid_that_launch_refuses_launch_blocks_refuses_in_the_same_words() -> Result {
    let (most, home) = (most()?, cpus()?);
    let first = *home.first().ok_or("first gave None")?;
    let outside = [home.iter().max().map_or(0, |last| last + 1)];
    let twice = [first, first];
    for (what, grid) in [
        ("no threads", Grid::new(0)),
        ("too many threads", Grid::new(most + 1)),
        ("a CPU the caller's thread may not run on", Grid::on(&outside)),
        ("a CPU given to two threads", Grid::on(&twice)),
    ] {
        let said = launch(grid, |_| {}).err().ok_or("launch gave Ok")?;
        for blocks in [0, 5] {
            let ran = Arc::new(AtomicU64::new(0));
            let into = Arc::clone(&ran);
            let refused = launch_blocks(grid, blocks, move |_| _ = into.fetch_add(1, Relaxed));
            assert_eq!(refused.err().ok_or("launch_blocks gave Ok")?, said, "{what}, {blocks} blocks");
            assert_eq!(ran.load(Relaxed), 0, "{what}, {blocks} blocks: blocks that ran");
        }
    }
    let most_blocks = u64::MAX / 2;
    let too_many = launch_blocks(Grid::new(1), most_blocks + 1, |_| {}).err().ok_or("launch_blocks gave Ok")?;
    assert_eq!(too_many.message, "a launch has 9223372036854775807 blocks at most, and this has 9223372036854775808");
    // The most blocks a launch has ends when a block panics: the blocks left count as ended, with no count to overflow.
    // A launch that ran them would not end, so it runs on a thread of its own, which the test waits for until a hang.
    let (said, heard) = mpsc::channel();
    thread::spawn(move || {
        let launched = panic::catch_unwind(|| {
            launch_blocks(Grid::new(most.min(3)), most_blocks, |block| {
                if block == 3 {
                    panic::resume_unwind(Box::new("block 3 panics"));
                }
            })
        });
        _ = said.send(launched.err().map(|payload| payload.downcast_ref::<&str>().copied()));
    });
    let payload =
        heard.recv_timeout(HANG).map_err(|_| format!("a launch of the most blocks: not in {HANG:?}, a hang"))?;
    assert_eq!(payload, Some(Some("block 3 panics")), "how a launch of the most blocks, one of which panics, ended");
    Ok(())
}

/// A block that panics ends the launch, once every block taken has ended, as a panic in the caller with the payload it
/// was given, the first if more than one panicked; and no block is taken after it. On two threads, of a thousand
/// blocks, the first two each wait for the other to begin, so that each thread has one. The caller's panics. The
/// other's waits until the caller sleeps, which has no block left that it may take, and then panics too. The launch
/// ends then, with the caller's panic, and those two are the only blocks that began. The launches after run each block
/// and each thread once.
#[test]
fn a_block_that_panics_ends_the_launch_in_the_caller_once_every_block_taken_has_ended() -> Result {
    if most()? < 2 {
        eprintln!("the process runs one thread at once: skipped");
        return Ok(());
    }
    let (caller, id) = (Arc::new(me()?), thread::current().id());
    let (begun, other) = (Arc::new(AtomicU64::new(0)), Arc::new(Mutex::new(None)));
    let (began, into) = (Arc::clone(&begun), Arc::clone(&other));
    let launched = panic::catch_unwind(move || {
        launch_blocks(Grid::new(2), 1000, move |_| {
            if began.fetch_add(1, AcqRel) >= 2 {
                return;
            }
            let both = until("the other block begun", || Ok(began.load(Acquire) >= 2));
            // `resume_unwind` panics without the message a hook would print.
            if thread::current().id() == id {
                panic::resume_unwind(Box::new("the caller's block panics"));
            }
            let slept = both.and_then(|()| until("the caller asleep", || asleep(&caller)));
            *into.lock().unwrap_or_else(PoisonError::into_inner) = Some(slept.map_err(|e| e.to_string()));
            panic::resume_unwind(Box::new("the other block panics"));
        })
    });
    let payload = launched.err().ok_or("launched gave Ok")?;
    assert_eq!(payload.downcast_ref::<&str>(), Some(&"the caller's block panics"));
    let other = other.lock().unwrap_or_else(PoisonError::into_inner).take();
    assert_eq!(other, Some(Ok(())), "the other block waited for the caller to sleep, and ended before the launch did");
    assert_eq!(begun.load(Acquire), 2, "blocks that began, of a launch whose first two panicked");
    // The threads run the next launches as before.
    let blocks = Arc::new(Mutex::new(vec![]));
    let into = Arc::clone(&blocks);
    launch_blocks(Grid::new(2), 100, move |block| {
        into.lock().unwrap_or_else(PoisonError::into_inner).push(block);
    })?;
    let mut blocks = blocks.lock().unwrap_or_else(PoisonError::into_inner).clone();
    blocks.sort();
    assert_eq!(blocks, (0..100).collect::<Vec<u64>>());
    let ran = Arc::new(Mutex::new(vec![]));
    let into = Arc::clone(&ran);
    launch(Grid::new(2), move |place| into.lock().unwrap_or_else(PoisonError::into_inner).push(place.thread()))?;
    let mut ran = ran.lock().unwrap_or_else(PoisonError::into_inner).clone();
    ran.sort();
    assert_eq!(ran, [0, 1]);
    Ok(())
}

/// Two threads that each launch 3 blocks on 3 threads, at once, have every block run at once, each on a thread of its
/// own: each block waits at one barrier of 6, which none could if a launch waited for the other to end, or ran on the
/// other's threads.
#[test]
fn launches_of_blocks_from_two_threads_at_once_run_at_once_each_on_threads_of_its_own() -> Result {
    let most = most()?;
    if most < 3 {
        eprintln!("the process runs {most} threads at once: skipped");
        return Ok(());
    }
    let meet = Arc::new(Barrier::new(6));
    let callers: Vec<_> = Vec::from_fn(2, |_| {
        let meet = Arc::clone(&meet);
        thread::spawn(move || -> Result {
            launch_blocks(Grid::new(3), 3, move |_| _ = meet.wait())?;
            Ok(())
        })
    });
    for caller in callers {
        caller.join().map_err(|_| "a thread panicked")??;
    }
    Ok(())
}

/// A launch of blocks and a launch of threads each can follow the other, on the same kept threads: the second runs on
/// every thread, each once, when the first has left some of them late or not woken, as a launch of 2 blocks wakes one
/// thread of 3. Its threads wait for each other, as `grid.sync()` makes them, and each sees all of them come: one that
/// was still in the first launch would not.
#[test]
fn a_launch_of_threads_follows_a_launch_of_blocks_on_the_same_threads_and_the_reverse() -> Result {
    let threads = most()?.min(3);
    let grid = Grid::new(threads);
    let threads = threads as u64;
    let ran = Arc::new(Mutex::new(vec![]));
    let by_threads = |name: &'static str| -> Result {
        let (into, arrived) = (Arc::clone(&ran), Arc::new(AtomicU64::new(0)));
        launch(grid, move |place| {
            arrived.fetch_add(1, AcqRel);
            let all_came = until("all the threads come", || Ok(arrived.load(Acquire) >= threads)).is_ok();
            into.lock().unwrap_or_else(PoisonError::into_inner).push((place.thread(), all_came));
        })?;
        let mut ran = ran.lock().unwrap_or_else(PoisonError::into_inner);
        ran.sort();
        assert_eq!(
            *ran,
            (0..threads).map(|thread| (thread, true)).collect::<Vec<_>>(),
            "{name}: the threads, and whether each saw all come"
        );
        ran.clear();
        Ok(())
    };
    let by_blocks = |blocks: u64| -> Result {
        let seen = Arc::new(Mutex::new(vec![]));
        let into = Arc::clone(&seen);
        launch_blocks(grid, blocks, move |block| into.lock().unwrap_or_else(PoisonError::into_inner).push(block))?;
        let mut seen = seen.lock().unwrap_or_else(PoisonError::into_inner);
        seen.sort();
        assert_eq!(*seen, (0..blocks).collect::<Vec<_>>(), "{blocks} blocks");
        Ok(())
    };
    // Blocks, then threads; each count of blocks, then threads.
    for blocks in [0, 1, 2, 3, 100] {
        by_blocks(blocks)?;
        by_threads("after blocks")?;
    }
    // Threads, then blocks.
    for blocks in [0, 1, 2, 3, 100] {
        by_threads("before blocks")?;
        by_blocks(blocks)?;
    }
    Ok(())
}

/// The CPUs a thread may run on, as Linux lists them (`0-5,8`), for the calling thread.
fn allowed() -> Result<String> {
    let status = fs::read_to_string("/proc/thread-self/status")?;
    let line = status.lines().find_map(|line| line.strip_prefix("Cpus_allowed_list:"));
    Ok(line.ok_or("find_map gave None")?.trim().to_string())
}

/// A launch of blocks of a placed grid puts its threads as a launch does: each on its CPU alone, and the caller's
/// thread on the first, where it stays. Three blocks each wait at a barrier for the others, so that each of the three
/// threads takes one.
#[test]
fn a_launch_of_blocks_of_a_placed_grid_puts_its_threads_on_their_cpus() -> Result {
    if !std::path::Path::new("/proc/thread-self/status").exists() {
        eprintln!("no /proc/thread-self: skipped");
        return Ok(());
    }
    let on = cpus()?;
    let Some(placed) = on.get(..3).filter(|_| most().is_ok_and(|most| most >= 3)) else {
        eprintln!("the process runs fewer than 3 threads at once: skipped");
        return Ok(());
    };
    let (where_blocks_ran, caller, meet) =
        (Arc::new(Mutex::new(vec![])), thread::current().id(), Arc::new(Barrier::new(3)));
    let into = Arc::clone(&where_blocks_ran);
    launch_blocks(Grid::on(placed), 3, move |_| {
        meet.wait();
        let at = allowed().map(|cpus| (thread::current().id() == caller, cpus));
        into.lock().unwrap_or_else(PoisonError::into_inner).push(at);
    })?;
    let mut seen = vec![];
    for at in where_blocks_ran.lock().unwrap_or_else(PoisonError::into_inner).drain(..) {
        let (is_the_caller, cpus) = at?;
        seen.push(cpus.clone());
        let fits = |cpu: &usize| cpus == cpu.to_string();
        assert!(placed.iter().any(fits), "a block ran on {cpus}, where the grid's CPUs are {placed:?}");
        assert!(!is_the_caller || cpus == placed[0].to_string(), "a block ran on the caller's thread, on {cpus}");
    }
    seen.sort();
    seen.dedup();
    assert_eq!(seen.len(), 3, "the blocks ran on the CPUs {seen:?}, where each of 3 threads took one");
    assert_eq!(allowed()?, placed[0].to_string(), "the caller's thread is on the first CPU of the grid");
    Ok(())
}

/// A block that waits for a block taken later waits for ever, if every thread is in one: a grid of one thread takes
/// block 1 only once block 0 has ended. Block 0 looks for block 1 a thousand times, and never sees it begin, and block
/// 1 sees block 0 ended: block 0, had it waited for block 1, would have waited for ever.
#[test]
fn a_block_that_waits_for_a_block_taken_later_waits_for_ever_if_every_thread_is_in_one() -> Result {
    let (one_began, zero_ended) = (Arc::new(AtomicBool::new(false)), Arc::new(AtomicBool::new(false)));
    let saw: Arc<Mutex<Vec<(&str, bool)>>> = Arc::default();
    let into = Arc::clone(&saw);
    launch_blocks(Grid::new(1), 2, move |block| {
        let saw = |what, seen| into.lock().unwrap_or_else(PoisonError::into_inner).push((what, seen));
        if block == 1 {
            one_began.store(true, Release);
            saw("block 1 saw block 0 ended", zero_ended.load(Acquire));
            return;
        }
        let seen = (0..1000).fold(false, |seen, _| {
            thread::yield_now();
            seen || one_began.load(Acquire)
        });
        saw("block 0 saw block 1 begin", seen);
        zero_ended.store(true, Release);
    })?;
    let saw = std::mem::take(&mut *saw.lock().unwrap_or_else(PoisonError::into_inner));
    assert_eq!(saw, [("block 0 saw block 1 begin", false), ("block 1 saw block 0 ended", true)]);
    Ok(())
}

/// A launch of blocks returns once its last block has ended, and not before, whichever thread runs it. Two blocks each
/// wait for the other to begin, so that each runs on a thread of its own. The one that the caller does not run ends
/// once the caller sleeps, which has no block left and waits for it, and says so: a launch that returned before its
/// last block ended would find it not ended, as its caller would not sleep in it.
#[test]
fn the_launch_returns_once_the_last_block_has_ended() -> Result {
    if most()? < 2 {
        eprintln!("the process runs one thread at once: skipped");
        return Ok(());
    }
    let (caller, id) = (Arc::new(me()?), thread::current().id());
    let (begun, ended) = (Arc::new(AtomicU64::new(0)), Arc::new(Mutex::new(None)));
    let into = Arc::clone(&ended);
    launch_blocks(Grid::new(2), 2, move |_| {
        begun.fetch_add(1, AcqRel);
        let both = until("the other block begun", || Ok(begun.load(Acquire) >= 2));
        if thread::current().id() != id {
            let slept = both.and_then(|()| until("the caller asleep", || asleep(&caller)));
            *into.lock().unwrap_or_else(PoisonError::into_inner) = Some(slept.map_err(|e| e.to_string()));
        }
    })?;
    let ended = ended.lock().unwrap_or_else(PoisonError::into_inner).take();
    assert_eq!(ended, Some(Ok(())), "the block the caller did not run, when the launch returned");
    Ok(())
}
