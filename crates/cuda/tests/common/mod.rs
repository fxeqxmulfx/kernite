//! What the tests of unified memory have in common: the words of GUIDE.md's refusals, a page, and two skips, one for a
//! GPU that cannot make a kind of memory, and one for a machine that makes no managed memory.
#![allow(dead_code, reason = "a test file takes what it needs of this module, and rustc calls the rest dead")]

use kernite_cuda::{Error, Unified};

/// GUIDE.md's words for a GPU that cannot be told that memory is preferred at the GPU, or accessed by it.
pub const NO_ADVICE: &str = "cuMemAdvise: the GPU cannot reach managed memory while the host does: \
                             its CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS is 0";

/// GUIDE.md's words for a GPU that cannot allocate managed memory.
const NO_MANAGED: &str = "cuMemAllocManaged: the GPU cannot allocate managed memory: \
                          its CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY is 0";

/// GUIDE.md's words for a machine with a GPU that cannot reach managed memory while the host does, before the number of
/// that GPU and after it.
const NO_CONCURRENT: (&str, &str) = (
    "cuMemAllocManaged: GPU ",
    " cannot reach managed memory while the host does: its CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS is 0",
);

/// A page, in words. The driver moves and advises whole pages.
pub const PAGE: usize = 1024;

/// The index of the first word of `m` that starts a page. A buffer need not start at one, so the
/// pages at its ends may be shared with other memory; the pages between are its own.
pub fn first_page(m: &Unified<u32>) -> usize {
    let addr = m.ptr().addr();
    ((addr.next_multiple_of(4 * PAGE as u64) - addr) / 4) as usize
}

/// `made`, or `None` if the GPU cannot make that kind of memory: the error is then `refusal`, in
/// GUIDE.md's words, and the test is skipped, as for a machine with no GPU.
pub fn or_skip<T>(made: Result<T, Error>, refusal: &str) -> Option<T> {
    made.map_err(|e| {
        assert_eq!(e.to_string(), refusal);
        eprintln!("the GPU cannot: skipped. {e}");
    })
    .ok()
}

/// `made`, the result of `managed` or `managed_uninit`, or `None` if the machine makes no managed memory: the error is
/// then one of GUIDE.md's two, that the GPU cannot allocate managed memory, or that a GPU, which the error numbers,
/// cannot reach it while the host does, and the test is skipped, as for a machine with no GPU.
pub fn managed_or_skip<T>(made: Result<T, Error>) -> Option<T> {
    made.map_err(|e| {
        let said = e.to_string();
        let (before, after) = NO_CONCURRENT;
        let number = said.strip_prefix(before).and_then(|rest| rest.strip_suffix(after));
        assert!(said == NO_MANAGED || number.is_some_and(|n| n.parse::<usize>().is_ok()), "{said}");
        eprintln!("the machine makes no managed memory: skipped. {e}");
    })
    .ok()
}
