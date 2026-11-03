//! The driver is opened when a GPU is asked for, and not before, so that a program built with
//! GPU kernels runs where there is none. A test of its own, as a program of its own: one that
//! shares a process with the driver's tests could find it opened by one of them.

use kernite_cuda::Cuda;

/// Whether this process has the driver in its memory: a file whose name starts `libcuda.so`,
/// as `libcuda.so.1` and the file it links to do.
fn open() -> bool {
    let maps = std::fs::read_to_string("/proc/self/maps").unwrap();
    maps.lines().any(|l| l.rsplit('/').next().is_some_and(|file| file.starts_with("libcuda.so")))
}

#[test]
fn the_driver_is_opened_when_a_gpu_is_asked_for() {
    assert!(!open(), "the driver is open before a GPU is asked for");
    match Cuda::new() {
        Ok(_) => assert!(open(), "a GPU, and the driver not open"),
        // The driver is open if `dlopen` found it, whatever failed after.
        Err(e) => {
            assert_eq!(open(), e.call != "dlopen", "{e}");
            eprintln!("no GPU: {e}");
        }
    }
}
