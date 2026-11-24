//! What the tests of memory have in common: the words of GUIDE.md's refusals, a page, and the skips: for a GPU that
//! cannot make a kind of memory, for a machine that makes no managed memory, and for a GPU that cannot page-lock a
//! slice.
#![allow(dead_code, reason = "a test file takes what it needs of this module, and rustc calls the rest dead")]

use kernite_cuda::{Cuda, Error, Unified};

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

/// GUIDE.md's words for a GPU that cannot be lent a slice.
pub const NO_PAGEABLE: &str = "cuDeviceGetAttribute: the GPU cannot reach pageable memory, a slice of the host's: \
                               its CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS is 0";

/// GUIDE.md's words for a GPU that cannot map memory at addresses it reserves.
pub const NO_VMM: &str = "cuMemAddressReserve: the GPU cannot map memory at addresses it reserves: its \
                          CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED is 0";

/// GUIDE.md's words for a GPU that cannot page-lock memory the host has, and for one that cannot reach it at the host's
/// addresses.
const NO_REGISTER: [&str; 2] = [
    "cuMemHostRegister: the GPU cannot page-lock memory the host has: \
     its CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED is 0",
    "cuMemHostRegister: the GPU cannot reach page-locked memory the host has at the host's addresses: \
     its CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM is 0",
];

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

/// `cuda.register(slice)`; or `None` where the GPU cannot, after it says so in one of GUIDE.md's words, and the test is
/// skipped, as for a machine with no GPU.
pub fn register_or_skip<'a>(cuda: &Cuda, slice: &'a mut [u32]) -> Result<Option<Unified<'a, u32>>, Error> {
    match cuda.register(slice) {
        Err(e) if NO_REGISTER.contains(&e.to_string().as_str()) => {
            eprintln!("the GPU cannot: skipped. {e}");
            Ok(None)
        }
        made => made.map(Some),
    }
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

/// A kind of memory the host reaches that a copy takes, for the tests that go through each: managed, page-locked, and
/// registered for good, `len` elements of each. A kind the GPU does not take is left out, after a line that says so. A
/// slice that `lend` lent is not one: a copy refuses it, and `lent` makes one.
pub fn kinds(cuda: &Cuda, len: usize) -> Result<Vec<(&'static str, Unified<'static, u32>)>, Error> {
    let mut kinds = vec![("page-locked memory", cuda.pinned::<u32>(len)?)];
    if let Some(managed) = managed_or_skip(cuda.managed::<u32>(len)) {
        kinds.push(("managed memory", managed));
    }
    match cuda.register(Vec::leak(vec![0u32; len])) {
        Ok(registered) => kinds.push(("a slice registered for good", registered)),
        Err(e) => eprintln!("a GPU that page-locks no slice: none is registered. {e}"),
    }
    Ok(kinds)
}

/// `len` elements of a slice lent for good, if the GPU reaches pageable memory.
pub fn lent(cuda: &Cuda, len: usize) -> Option<Unified<'static, u32>> {
    let lent = cuda.lend(Vec::leak(vec![0u32; len]));
    lent.inspect_err(|e| eprintln!("a GPU that reaches no pageable memory: no slice is lent. {e}")).ok()
}
