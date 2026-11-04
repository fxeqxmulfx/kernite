//! What the driver says of the pages of managed memory that were moved and advised: where each was
//! last prefetched, and how each is advised. The tests run one at a time, in a program of their
//! own: the driver here may keep the advice and the last prefetch of a buffer it frees, and make
//! the buffer at that address start with them, when threads use it at once. Beside one another, or
//! beside the tests of `unified.rs`, they would read what another left.

use std::ffi::c_int;
use std::ops::Range;
use std::sync::{Mutex, MutexGuard, PoisonError};

use kernite_cuda::{Advice, Cuda, Location, Unified};

mod common;
use common::{NO_ADVICE, NO_PREFETCH, PAGE, first_page, or_skip};

/// Held by each test from its start to its end, after its memory is freed.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

/// The GPU, once no other test of this file is running, or `None` and why not.
fn gpu() -> Option<(MutexGuard<'static, ()>, Cuda)> {
    let one = ONE_AT_A_TIME.lock().unwrap_or_else(PoisonError::into_inner);
    let cuda = Cuda::new().map_err(|e| eprintln!("no GPU: skipped. {e}")).ok()?;
    Some((one, cuda))
}

/// A managed buffer with `n` whole pages in it, and the words of pages `from..to` of them, from
/// `from` and `to`. A managed buffer starts at a multiple of 512 bytes, not at a page.
fn paged(cuda: &Cuda, n: usize) -> (Unified<'static, u32>, impl Fn(usize, usize) -> Range<usize>) {
    let m = cuda.managed::<u32>((n + 1) * PAGE).unwrap();
    let at = first_page(&m);
    (m, move |from, to| at + from * PAGE..at + to * PAGE)
}

#[test]
fn managed_memory_is_prefetched_to_each_place_and_the_driver_says_where() {
    let Some((_one, cuda)) = gpu() else { return };
    let (m, pages) = paged(&cuda, 4);
    let all = pages(0, 4);
    assert_eq!(m.placement(all.clone()).unwrap().last_prefetch, None, "nothing was prefetched");
    for place in [Location::Gpu, Location::Host, Location::Numa(0), Location::NumaOfThread] {
        let Some(()) = or_skip(m.prefetch(all.clone(), place), NO_PREFETCH) else { return };
        assert_eq!(m.placement(all.clone()).unwrap().last_prefetch, Some(place), "{place:?}");
    }
}

#[test]
fn a_part_of_managed_memory_is_prefetched_and_the_rest_of_it_is_not() {
    let Some((_one, cuda)) = gpu() else { return };
    let (m, pages) = paged(&cuda, 4);
    let (first, second, all) = (pages(0, 2), pages(2, 4), pages(0, 4));
    let Some(()) = or_skip(m.prefetch(first.clone(), Location::Host), NO_PREFETCH) else { return };
    assert_eq!(m.placement(first).unwrap().last_prefetch, Some(Location::Host));
    assert_eq!(m.placement(second.clone()).unwrap().last_prefetch, None, "the rest was not");
    m.prefetch(second.clone(), Location::Gpu).unwrap();
    assert_eq!(m.placement(second).unwrap().last_prefetch, Some(Location::Gpu));
    assert_eq!(m.placement(all).unwrap().last_prefetch, None, "the whole was not sent to one place");
}

#[test]
fn a_range_reaches_the_whole_of_each_page_it_touches() {
    let Some((_one, cuda)) = gpu() else { return };
    let (m, pages) = paged(&cuda, 3);
    let (page0, page1, page2) = (pages(0, 1), pages(1, 2), pages(2, 3));
    // The second word of the first page.
    let word = page0.start + 1..page0.start + 2;
    let Some(()) = or_skip(m.prefetch(word, Location::Host), NO_PREFETCH) else { return };
    assert_eq!(m.placement(page0.clone()).unwrap().last_prefetch, Some(Location::Host), "the page went with it");
    assert_eq!(m.placement(page1.clone()).unwrap().last_prefetch, None, "the next page did not");
    // The last word of the second page and the first of the third.
    m.advise(page1.end - 1..page1.end + 1, Advice::ReadMostly).unwrap();
    let read_mostly = |range: &Range<usize>| m.placement(range.clone()).unwrap().read_mostly;
    assert!(read_mostly(&page1) && read_mostly(&page2), "both pages went with a word each");
    assert!(!read_mostly(&page0), "the page before did not");
}

#[test]
fn a_numa_node_the_host_does_not_have_is_refused_by_the_driver() {
    let Some((_one, cuda)) = gpu() else { return };
    let (m, pages) = paged(&cuda, 1);
    let page = pages(0, 1);
    let Some(()) = or_skip(m.prefetch(page.clone(), Location::Host), NO_PREFETCH) else { return };
    // The greatest node an `int` holds reaches the driver as it is; one past it is kept at that.
    for node in [Location::Numa(c_int::MAX as u32), Location::Numa(c_int::MAX as u32 + 1), Location::Numa(u32::MAX)] {
        let (prefetch, advise) =
            (m.prefetch(page.clone(), node), m.advise(page.clone(), Advice::PreferredLocation(node)));
        for (call, refused) in [("cuMemPrefetchAsync", prefetch), ("cuMemAdvise", advise)] {
            let e = refused.unwrap_err();
            assert_eq!(
                (e.call, e.message.as_str()),
                (call, "CUDA_ERROR_INVALID_VALUE (1): invalid argument"),
                "{node:?}"
            );
        }
    }
    assert_eq!(m.placement(page).unwrap().last_prefetch, Some(Location::Host), "the refused ones moved nothing");
}

#[test]
fn each_advice_is_in_force_for_the_range_it_is_given_for_and_the_driver_says_so() {
    let Some((_one, cuda)) = gpu() else { return };
    let (m, pages) = paged(&cuda, 4);
    let (first, second, all) = (pages(0, 2), pages(2, 4), pages(0, 4));
    let read_mostly = |range: &Range<usize>| m.placement(range.clone()).unwrap().read_mostly;
    m.advise(first.clone(), Advice::ReadMostly).unwrap();
    assert!(read_mostly(&first), "the first half is");
    assert!(!read_mostly(&second) && !read_mostly(&all), "the second half is not, so the whole is not");
    m.advise(first.clone(), Advice::NotReadMostly).unwrap();
    assert!(!read_mostly(&first));

    let preferred = || m.placement(all.clone()).unwrap().preferred;
    // The driver here takes a NUMA node as the preferred place and changes nothing, though cuda.h
    // says it sets the node: so the place is still the one it was. The day a driver sets the node,
    // this fails, and README and the docs of `PreferredLocation` and `preferred` change with it.
    let numa_changes_nothing = |was: Option<Location>| {
        for place in [Location::Numa(0), Location::NumaOfThread] {
            m.advise(all.clone(), Advice::PreferredLocation(place)).unwrap();
            assert_eq!(preferred(), was, "{place:?} after {was:?}");
        }
    };
    let Some(()) = or_skip(m.advise(all.clone(), Advice::PreferredLocation(Location::Gpu)), NO_ADVICE) else {
        return;
    };
    assert_eq!(preferred(), Some(Location::Gpu));
    numa_changes_nothing(Some(Location::Gpu));
    m.advise(all.clone(), Advice::PreferredLocation(Location::Host)).unwrap();
    assert_eq!(preferred(), Some(Location::Host));
    numa_changes_nothing(Some(Location::Host));
    m.advise(all.clone(), Advice::NoPreferredLocation).unwrap();
    assert_eq!(preferred(), None, "none");
    numa_changes_nothing(None);

    // Where the pages do not agree, the whole range has none.
    m.advise(first.clone(), Advice::PreferredLocation(Location::Host)).unwrap();
    assert_eq!(m.placement(first.clone()).unwrap().preferred, Some(Location::Host));
    assert_eq!(preferred(), None, "the second half prefers none");
    m.advise(first.clone(), Advice::NoPreferredLocation).unwrap();
    m.advise(first.clone(), Advice::AccessedBy(Location::Gpu)).unwrap();
    assert_eq!(m.placement(first.clone()).unwrap().accessed_by, [Location::Gpu]);
    assert_eq!(m.placement(all.clone()).unwrap().accessed_by, [], "the second half is accessed by none");
    m.advise(first.clone(), Advice::NotAccessedBy(Location::Gpu)).unwrap();

    let accessed_by = || m.placement(all.clone()).unwrap().accessed_by;
    assert_eq!(accessed_by(), []);
    m.advise(all.clone(), Advice::AccessedBy(Location::Gpu)).unwrap();
    assert_eq!(accessed_by(), [Location::Gpu]);
    m.advise(all.clone(), Advice::AccessedBy(Location::Host)).unwrap();
    let both = accessed_by();
    assert!(both.len() == 2 && both.contains(&Location::Gpu) && both.contains(&Location::Host), "{both:?}");
    m.advise(all.clone(), Advice::NotAccessedBy(Location::Gpu)).unwrap();
    assert_eq!(accessed_by(), [Location::Host]);
    m.advise(all.clone(), Advice::NotAccessedBy(Location::Host)).unwrap();
    assert_eq!(accessed_by(), []);
}
