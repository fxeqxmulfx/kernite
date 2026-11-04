//! The stack of a thread of a grid is the one Rust gives a thread, which `RUST_MIN_STACK` sets,
//! and a grid with a thread that cannot get one does not run. Rust reads the variable once, when
//! the first thread starts, so each test runs itself again in a process of its own, with the
//! value it needs.

use std::hint::black_box;
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};

use kernite_cpu::{Grid, launch, most};

/// What names the process that runs the body of a test.
const CHILD: &str = "KERNITE_CPU_CHILD";

/// Whether this is the process `again` started.
fn is_child() -> bool {
    std::env::var_os(CHILD).is_some()
}

/// Runs the test `name` again in a process of its own, where Rust gives a thread `stack` bytes
/// of stack, and fails if it fails.
fn again(name: &str, stack: usize) {
    let out = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", name, "--nocapture", "--test-threads=1"])
        .env(CHILD, "1")
        .env("RUST_MIN_STACK", stack.to_string())
        .output()
        .unwrap();
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    assert!(out.status.success() && text.contains("1 passed"), "{text}");
}

/// A launch of two threads, one of which cannot start, is an error, and no thread has run the
/// kernel; a launch of one thread, which starts none, still runs.
#[test]
fn a_thread_that_cannot_start_is_an_error_and_no_thread_runs_the_kernel() {
    if most().unwrap() < 2 {
        return eprintln!("the process runs one thread at once, so no grid starts a thread: skipped");
    }
    if !is_child() {
        // More than the address space has: no stack of it can be mapped.
        return again("a_thread_that_cannot_start_is_an_error_and_no_thread_runs_the_kernel", 1 << 60);
    }
    let ran = Arc::new(AtomicUsize::new(0));
    let into = Arc::clone(&ran);
    let error = launch(Grid::new(2), move |_| {
        into.fetch_add(1, SeqCst);
    })
    .unwrap_err();
    assert!(error.message.starts_with("a thread of the grid did not start: "), "{error}");
    assert_eq!(ran.load(SeqCst), 0, "{error}");
    let into = Arc::clone(&ran);
    launch(Grid::new(1), move |_| {
        into.fetch_add(1, SeqCst);
    })
    .unwrap();
    assert_eq!(ran.load(SeqCst), 1);
}

/// Every thread of a grid has the stack Rust gives, here 16 MB where it gives 2: a kernel that
/// takes 8 MB of it runs on each.
#[test]
fn a_thread_of_the_grid_has_the_stack_rust_gives() {
    if !is_child() {
        return again("a_thread_of_the_grid_has_the_stack_rust_gives", 16 << 20);
    }
    let threads = most().unwrap().min(3);
    let ran = Arc::new(AtomicUsize::new(0));
    let into = Arc::clone(&ran);
    launch(Grid::new(threads), move |_| {
        let mut frame = [0u8; 8 << 20];
        black_box(&mut frame);
        into.fetch_add(1, SeqCst);
    })
    .unwrap();
    assert_eq!(ran.load(SeqCst), threads);
}
