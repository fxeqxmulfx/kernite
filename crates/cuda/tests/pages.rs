//! A managed buffer does not start at a page, and a range reaches the memory beside it on its first
//! page, which is another buffer's, as GUIDE.md says. A test of its own, as a program of its own,
//! for the driver hands out managed memory in the order it is asked for, and the tests that run
//! beside one another ask for theirs at once. A driver that places buffers another way fails this
//! test, and then the sentence in GUIDE.md goes.

use kernite_cuda::{Advice, Cuda};

/// A page, in bytes, as the driver moves and advises it.
const PAGE: u64 = 4096;

#[test]
fn an_advice_on_a_managed_buffer_reaches_the_buffer_before_it_on_the_page_they_share() {
    let Ok(cuda) = Cuda::new().inspect_err(|e| eprintln!("no GPU: skipped. {e}")) else { return };
    let before = cuda.managed::<u32>(128).unwrap();
    let after = cuda.managed::<u32>(4096).unwrap();
    let (a, b) = (before.ptr().addr(), after.ptr().addr());
    assert_eq!(b % 512, 0, "the second buffer starts at {b:#x}, which is not a multiple of 512");
    assert_ne!(b % PAGE, 0, "the second buffer starts at {b:#x}, which is a page");
    assert_eq!(a / PAGE, b / PAGE, "the buffers are on pages {:#x} and {:#x}", a / PAGE, b / PAGE);
    assert!(!before.placement(0..128).unwrap().read_mostly);
    // The first word of the second buffer, which the first buffer is on the page of.
    after.advise(0..1, Advice::ReadMostly).unwrap();
    assert!(before.placement(0..128).unwrap().read_mostly, "the buffer before went with it");
}
