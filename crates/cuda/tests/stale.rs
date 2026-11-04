//! A buffer made at the address of one freed may start with the advice and the last prefetch the
//! freed one had, as README says: the driver here does not always clear them when threads use it
//! at once, and clears them when one thread does. A test of its own, as a program of its own, for
//! the pages it leaves with them would be read by the tests beside it. A driver that clears them
//! for threads fails this test, and then the sentences in README and in the docs of
//! `Unified::placement` go. So does a driver that makes no buffer at the address of the one freed:
//! it shows nothing of one thread, or of two, and the test counts the buffers that were.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

use kernite_cuda::{Advice, Cuda, Location};

mod common;
use common::{NO_ADVICE, NO_PREFETCH, PAGE, first_page, or_skip};

/// What the rounds of some threads saw. A round advises a buffer, prefetches it, frees it, and
/// makes another.
struct Seen {
    /// The rounds whose new buffer was made at the address of the one freed: the only rounds that
    /// can show a buffer start with what the one freed had.
    reused: usize,
    /// Of those, the rounds whose new buffer started with the advice or the last prefetch of the
    /// one freed.
    started_with_them: usize,
}

/// What `threads` threads saw, in at most `rounds` rounds each: they stop once a buffer has
/// started with the advice or the last prefetch of the one freed.
fn seen(threads: usize, rounds: usize) -> Seen {
    let (reused, started_with_them) = (AtomicUsize::new(0), AtomicUsize::new(0));
    thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                let cuda = Cuda::new().unwrap();
                for _ in 0..rounds {
                    if started_with_them.load(Ordering::Relaxed) > 0 {
                        return;
                    }
                    // Four pages of its own: the second and third read mostly, the third and fourth
                    // prefetched to the GPU, and the first and second preferred there.
                    let m = cuda.managed::<u32>(5 * PAGE).unwrap();
                    let at = first_page(&m);
                    m.advise(at + PAGE..at + 3 * PAGE, Advice::ReadMostly).unwrap();
                    m.advise(at..at + 2 * PAGE, Advice::PreferredLocation(Location::Gpu)).unwrap();
                    m.prefetch(at + 2 * PAGE..at + 4 * PAGE, Location::Gpu).unwrap();
                    let was = m.ptr().addr();
                    drop(m);

                    let m = cuda.managed::<u32>(5 * PAGE).unwrap();
                    if m.ptr().addr() != was {
                        continue;
                    }
                    reused.fetch_add(1, Ordering::Relaxed);
                    let at = first_page(&m);
                    let started = (0..4).any(|page| {
                        let p = m.placement(at + page * PAGE..at + (page + 1) * PAGE).unwrap();
                        p.read_mostly || p.preferred.is_some() || p.last_prefetch.is_some()
                    });
                    started_with_them.fetch_add(usize::from(started), Ordering::Relaxed);
                }
            });
        }
    });
    Seen { reused: reused.into_inner(), started_with_them: started_with_them.into_inner() }
}

#[test]
fn a_freed_buffers_advice_and_last_prefetch_outlive_it_for_two_threads_and_not_for_one() {
    let Ok(cuda) = Cuda::new().inspect_err(|e| eprintln!("no GPU: skipped. {e}")) else { return };
    let probe = cuda.managed::<u32>(PAGE).unwrap();
    let Some(()) = or_skip(probe.prefetch(0..PAGE, Location::Gpu), NO_PREFETCH) else { return };
    let Some(()) = or_skip(probe.advise(0..PAGE, Advice::PreferredLocation(Location::Gpu)), NO_ADVICE) else {
        return;
    };
    drop(probe);

    let one = seen(1, 1000);
    assert_eq!(
        one.started_with_them, 0,
        "{} of the {} buffers made at the address of the one freed started with its advice or its last \
         prefetch, though one thread used the driver",
        one.started_with_them, one.reused
    );
    // One thread shows nothing of the freed buffer's advice unless many of its rounds made a buffer
    // where the one freed was: all 1000 do here.
    assert!(
        one.reused >= 500,
        "only {} of 1000 rounds of one thread made a buffer at the address of the one freed, too few \
         to show that none starts with its advice or its last prefetch",
        one.reused
    );
    let two = seen(2, 5000);
    assert!(
        two.started_with_them > 0,
        "none of the {} buffers made at the address of the one freed started with its advice or its last \
         prefetch, in 5000 rounds of two threads: the driver clears them now, or makes no buffer at an \
         address twice",
        two.reused
    );
}
