//! A launch of blocks: the kernel runs once for each block, on the threads of a grid, each thread taking the next block
//! when it ends one, and the launch returns once the last block has ended. What a thread that comes late does, which
//! needs a thread kernite does not start, is in the tests of the crate's own.

use std::hint::spin_loop;
use std::sync::atomic::{
    AtomicU8, AtomicU64,
    Ordering::{AcqRel, Acquire, Relaxed},
};
use std::sync::{Arc, Barrier, Mutex, PoisonError};
use std::time::{Duration, Instant};
use std::{fs, panic, thread};

use kernite_cpu::{Grid, cpus, launch, launch_blocks, most};

type Result<T = (), E = Box<dyn std::error::Error + Send + Sync>> = std::result::Result<T, E>;

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
    let launched = panic::catch_unwind(|| {
        launch_blocks(Grid::new(most.min(3)), most_blocks, |block| {
            if block == 3 {
                panic::resume_unwind(Box::new("block 3 panics"));
            }
        })
    });
    let payload = launched.err().ok_or("launched gave Ok")?;
    assert_eq!(payload.downcast_ref::<&str>(), Some(&"block 3 panics"));
    Ok(())
}

/// A block that panics ends the launch, once every block taken has ended, as a panic in the caller with the payload it
/// was given. No block is taken after it: blocks that last 20 ms on 6 threads, of a thousand, begin a few after block
/// 10, which panics, and no more, though the launch had 3 s of them left. The first panic is the one the caller gets:
/// block 10 waits for block 11 to have begun, and panics; block 11 waits for it, and for 50 ms more, and panics too.
/// The launches after run each block and each thread once.
#[test]
fn a_block_that_panics_ends_the_launch_in_the_caller_once_every_block_taken_has_ended() -> Result {
    let threads = most()?.min(6);
    if threads < 2 {
        eprintln!("the process runs one thread at once: skipped");
        return Ok(());
    }
    let (begun, done) = (Arc::new(AtomicU64::new(0)), Arc::new(AtomicU64::new(0)));
    let (ten_panicked, eleven_began) = (Arc::new(AtomicU64::new(0)), Arc::new(AtomicU64::new(0)));
    let (began, ended) = (Arc::clone(&begun), Arc::clone(&done));
    let launched = panic::catch_unwind(move || {
        launch_blocks(Grid::new(threads), 1000, move |block| {
            began.fetch_add(1, Relaxed);
            // `resume_unwind` panics without the message a hook would print.
            if block == 10 {
                while eleven_began.load(Relaxed) == 0 {
                    thread::yield_now();
                }
                ten_panicked.store(1, Relaxed);
                panic::resume_unwind(Box::new("block 10 panics"));
            }
            if block == 11 {
                eleven_began.store(1, Relaxed);
                while ten_panicked.load(Relaxed) == 0 {
                    thread::yield_now();
                }
                thread::sleep(Duration::from_millis(50));
                panic::resume_unwind(Box::new("block 11 panics"));
            }
            thread::sleep(Duration::from_millis(20));
            ended.fetch_add(1, Relaxed);
        })
    });
    let payload = launched.err().ok_or("launched gave Ok")?;
    assert_eq!(payload.downcast_ref::<&str>(), Some(&"block 10 panics"));
    // Each block that began ended, but the two that panicked, which did not count themselves.
    let (begun_now, done_now) = (begun.load(Relaxed), done.load(Relaxed));
    assert_eq!(begun_now, done_now + 2, "blocks that began, and blocks that ended");
    assert!(begun_now < 100, "{begun_now} of 1000 blocks began, after block 10 panicked");
    thread::sleep(Duration::from_millis(100));
    assert_eq!(begun.load(Relaxed), begun_now, "a block began after the launch had ended");
    // The threads run the next launches as before.
    let blocks = Arc::new(Mutex::new(vec![]));
    let into = Arc::clone(&blocks);
    launch_blocks(Grid::new(threads), 100, move |block| {
        into.lock().unwrap_or_else(PoisonError::into_inner).push(block);
    })?;
    let mut blocks = blocks.lock().unwrap_or_else(PoisonError::into_inner).clone();
    blocks.sort();
    assert_eq!(blocks, (0..100).collect::<Vec<u64>>());
    let ran = Arc::new(Mutex::new(vec![]));
    let into = Arc::clone(&ran);
    launch(Grid::new(threads), move |place| into.lock().unwrap_or_else(PoisonError::into_inner).push(place.thread()))?;
    let mut ran = ran.lock().unwrap_or_else(PoisonError::into_inner).clone();
    ran.sort();
    assert_eq!(ran, (0..threads as u64).collect::<Vec<_>>());
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
/// was still in the first launch, or had a reply of it left, would not. They take 20 ms more, so that a launch which
/// returned when a thread of the first said it had ended would return before them.
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
            let give_up = Instant::now() + Duration::from_secs(10);
            while arrived.load(Acquire) < threads && Instant::now() < give_up {
                spin_loop();
            }
            let all_came = arrived.load(Acquire) == threads;
            thread::sleep(Duration::from_millis(20));
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
/// thread on the first, where it stays. Blocks that last 5 ms, sixty of them, are taken by each of the three threads.
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
    let (where_blocks_ran, caller) = (Arc::new(Mutex::new(vec![])), thread::current().id());
    let into = Arc::clone(&where_blocks_ran);
    launch_blocks(Grid::on(placed), 60, move |_| {
        thread::sleep(Duration::from_millis(5));
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
    assert_eq!(seen.len(), 3, "the blocks ran on the CPUs {seen:?}, where each of 3 threads took some");
    assert_eq!(allowed()?, placed[0].to_string(), "the caller's thread is on the first CPU of the grid");
    Ok(())
}

/// A block that waits for a block taken later waits for ever, if every thread is in one: a grid of one thread takes
/// block 1 after block 0 has ended, so that block 0, which waits for it to begin, sees it begin never. The test lets it
/// wait for 200 ms, and no longer.
#[test]
fn a_block_that_waits_for_a_block_taken_later_waits_for_ever_if_every_thread_is_in_one() -> Result {
    let (began, waited, saw) = (Arc::new(AtomicU64::new(0)), Arc::new(AtomicU64::new(0)), Arc::new(AtomicU64::new(0)));
    let (one_began, zero_waited, zero_saw) = (Arc::clone(&began), Arc::clone(&waited), Arc::clone(&saw));
    launch_blocks(Grid::new(1), 2, move |block| {
        if block == 1 {
            one_began.store(1, Relaxed);
            return;
        }
        let start = Instant::now();
        while one_began.load(Relaxed) == 0 && start.elapsed() < Duration::from_millis(200) {
            thread::yield_now();
        }
        zero_waited.store(u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX), Relaxed);
        zero_saw.store(one_began.load(Relaxed), Relaxed);
    })?;
    assert!(waited.load(Relaxed) >= 200, "block 0 waited {} ms, and not the 200 it was to", waited.load(Relaxed));
    assert_eq!(saw.load(Relaxed), 0, "block 0 saw block 1 begin, though the only thread was in block 0");
    assert_eq!(began.load(Relaxed), 1, "block 1 did not run");
    Ok(())
}

/// A launch of blocks returns once its last block has ended, and not before, whichever thread runs it. Two blocks each
/// wait for the other to begin, so that each runs on a thread of its own, and the one that the caller does not run
/// lasts 100 ms: the caller has no block left, and waits asleep for it. The launch takes that long, and no longer by
/// much.
#[test]
fn the_launch_returns_once_the_last_block_has_ended() -> Result {
    if most()? < 2 {
        eprintln!("the process runs one thread at once: skipped");
        return Ok(());
    }
    let (begun, caller) = (Arc::new(AtomicU64::new(0)), thread::current().id());
    let began = Instant::now();
    launch_blocks(Grid::new(2), 2, move |_| {
        begun.fetch_add(1, AcqRel);
        let give_up = Instant::now() + Duration::from_secs(10);
        while begun.load(Acquire) < 2 && Instant::now() < give_up {
            thread::yield_now();
        }
        if thread::current().id() != caller {
            thread::sleep(Duration::from_millis(100));
        }
    })?;
    let took = began.elapsed();
    assert!(took >= Duration::from_millis(100), "the launch took {took:?}, less than its longest block");
    assert!(took < Duration::from_millis(500), "the launch took {took:?}, where its longest block took 100 ms");
    Ok(())
}
