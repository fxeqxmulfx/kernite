//! What the tests of unified memory have in common: the words of the README's refusals, a page, and
//! the skip for a GPU that cannot.

use kernite_cuda::{Error, Unified};

/// The README's words for a GPU that cannot move managed memory.
pub const NO_PREFETCH: &str = "cuMemPrefetchAsync: the GPU cannot reach managed memory while the host does: \
                               its CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS is 0";

/// The README's words for a GPU that cannot be told that managed memory is preferred at the GPU.
pub const NO_ADVICE: &str = "cuMemAdvise: the GPU cannot reach managed memory while the host does: \
                             its CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS is 0";

/// A page, in words. The driver moves and advises whole pages.
pub const PAGE: usize = 1024;

/// The index of the first word of `m` that starts a page. A buffer need not start at one, so the
/// pages at its ends may be shared with other memory; the pages between are its own.
pub fn first_page(m: &Unified<u32>) -> usize {
    let addr = m.ptr().addr();
    ((addr.next_multiple_of(4 * PAGE as u64) - addr) / 4) as usize
}

/// `made`, or `None` if the GPU cannot make that kind of memory: the error is then `refusal`, in
/// the README's words, and the test is skipped, as for a machine with no GPU.
pub fn or_skip<T>(made: Result<T, Error>, refusal: &str) -> Option<T> {
    made.map_err(|e| {
        assert_eq!(e.to_string(), refusal);
        eprintln!("the GPU cannot: skipped. {e}");
    })
    .ok()
}
