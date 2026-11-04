//! What `Cuda::count` and `Cuda::on` say as the driver shows the program GPUs, or none, or is not
//! there. A test of its own, as a program of its own, and each case runs it again in a process
//! with the environment that makes the case: the driver reads `CUDA_VISIBLE_DEVICES` once, as it
//! starts, and the loader reads `LD_LIBRARY_PATH` once, as the program starts.

use std::path::Path;
use std::process::Command;
use std::{env, fs};

use kernite_cuda::Cuda;

/// What names the process that runs the test again, to say what it makes of its GPUs.
const CHILD: &str = "KERNITE_CUDA_CHILD";

/// The test's name, for the process that runs it again.
const NAME: &str = "count_and_on_say_what_the_driver_shows_the_program";

/// What a process says of its GPUs: the count, and what taking the GPU after the last is, or the
/// first, if there is none.
fn says() -> String {
    let count = Cuda::count();
    let next = count.as_ref().copied().unwrap_or_default();
    let count = count.map_or_else(|e| e.to_string(), |count| count.to_string());
    format!("count: {count}\non({next}): {}", Cuda::on(next).err().unwrap())
}

/// Four cases, each in a process of its own.
///
/// - A `CUDA_VISIBLE_DEVICES` that chooses none: the driver shows none, the count is 0, and
///   `Cuda::on` is the error of `cuInit`, whole.
/// - The last GPU alone: the driver shows it as GPU 0, the only one, and the next index is an
///   error that says the driver shows 1.
/// - A list that `cuInit` refuses, one that names a GPU twice: an error for both, and no count of
///   0.
/// - A `libcuda.so.1` that is no library, where the loader looks first: there is no driver, which
///   is an error for both.
#[test]
fn count_and_on_say_what_the_driver_shows_the_program() {
    if env::var_os(CHILD).is_some() {
        return println!("{}", says());
    }
    let count = match Cuda::count() {
        Ok(count @ 1..) => count,
        Ok(0) => return eprintln!("no GPU: skipped, the driver shows none"),
        Err(e) => return eprintln!("no GPU: skipped. {e}"),
    };
    let none = "cuInit: CUDA_ERROR_NO_DEVICE (100): no CUDA-capable device is detected";
    let past = "cuDeviceGet: CUDA_ERROR_INVALID_DEVICE (101): invalid device ordinal: the driver shows 1 GPU, \
                and none is numbered 1";
    let refused = "cuInit: CUDA_ERROR_INVALID_DEVICE (101): invalid device ordinal";
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("no-driver");
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join("libcuda.so.1"), "").unwrap();
    let absent = format!("dlopen: no CUDA driver: {}/libcuda.so.1: file too short", dir.display());
    let last = (count - 1).to_string();
    let cases = [
        ("CUDA_VISIBLE_DEVICES", "", format!("count: 0\non(0): {none}")),
        ("CUDA_VISIBLE_DEVICES", last.as_str(), format!("count: 1\non(1): {past}")),
        ("CUDA_VISIBLE_DEVICES", "0,0", format!("count: {refused}\non(0): {refused}")),
        ("LD_LIBRARY_PATH", dir.to_str().unwrap(), format!("count: {absent}\non(0): {absent}")),
    ];
    for (variable, value, want) in cases {
        let out = Command::new(env::current_exe().unwrap())
            .args(["--exact", NAME, "--nocapture", "--test-threads=1"])
            .env(CHILD, "1")
            .env(variable, value)
            .output()
            .unwrap();
        let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
        assert!(out.status.success() && text.contains(&want), "{variable}={value:?} should say\n{want}\n{text}");
    }
}
