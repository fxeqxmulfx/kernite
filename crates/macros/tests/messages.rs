//! What the macro says of the kernels it is given: each error where it is, at its
//! `line:column`, in the order Cargo prints them. Each fixture is built as a user's crate is,
//! and what Cargo says of it is compared line for line, the errors in rustc's short form,
//! `src/lib.rs:4:1: error: ..`. A fixture is a crate's `src/lib.rs`, so the line an error names
//! is that line of the fixture.

use std::fs;
use std::path::Path;
use std::process::Command;
use std::sync::{Mutex, PoisonError};

/// Held while Cargo builds a fixture: the fixtures share a target directory, and a Cargo that
/// waits for another's lock says so.
static CARGO: Mutex<()> = Mutex::new(());

/// Whether `cargo build` built fixture `name`, and what it said, a line each, in order.
///
/// The fixture is the `src/lib.rs` of a crate that depends on kgen-macros, kgen-cuda and
/// kgen-cpu, as a user's does. The crate is a workspace of its own in `CARGO_TARGET_TMPDIR`,
/// built offline into a target directory beside it, so that it never waits for the build that
/// runs these tests. What the environment would change of a build is taken out of it.
fn build(name: &str) -> (bool, Vec<String>) {
    let here = Path::new(env!("CARGO_MANIFEST_DIR"));
    let crates = here.parent().expect("kgen-macros is in crates/");
    let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join("kgen-macros");
    let dir = root.join(name);
    fs::create_dir_all(dir.join("src")).unwrap();
    let path = |c: &str| crates.join(c).display().to_string();
    let (macros, cuda, cpu) = (path("macros"), path("cuda"), path("cpu"));
    let manifest = format!(
        r#"[package]
name = {name:?}
version = "0.0.0"
edition = "2024"
publish = false

[dependencies]
kgen-macros = {{ path = {macros:?} }}
kgen-cuda = {{ path = {cuda:?} }}
kgen-cpu = {{ path = {cpu:?} }}

[workspace]
"#
    );
    fs::write(dir.join("Cargo.toml"), manifest).unwrap();
    fs::copy(here.join("tests/fixtures").join(format!("{name}.rs")), dir.join("src/lib.rs")).unwrap();
    let _one = CARGO.lock().unwrap_or_else(PoisonError::into_inner);
    let mut cargo = Command::new(env!("CARGO"));
    cargo.args(["build", "--offline", "--quiet", "--message-format=short", "--color=never", "--target-dir"]);
    cargo.arg(root.join("target")).current_dir(&dir);
    for var in [
        "RUSTFLAGS",
        "CARGO_ENCODED_RUSTFLAGS",
        "CARGO_BUILD_RUSTFLAGS",
        "RUSTC_WRAPPER",
        "RUSTC_WORKSPACE_WRAPPER",
        "CARGO_TARGET_DIR",
        "CARGO_BUILD_TARGET_DIR",
    ] {
        cargo.env_remove(var);
    }
    let out = cargo.output().expect("Cargo runs");
    let said = String::from_utf8(out.stderr).expect("Cargo writes UTF-8");
    (out.status.success(), said.lines().map(String::from).collect())
}

/// That Cargo builds fixture `name` without a word.
fn builds_quietly(name: &str) {
    let (built, said) = build(name);
    assert!(built && said.is_empty(), "fixtures/{name}.rs: built {built}, and Cargo said:\n{}", said.join("\n"));
}

/// That Cargo does not build fixture `name`, and says `lines`, in that order, and nothing else.
fn refuses(name: &str, lines: &[&str]) {
    let (built, said) = build(name);
    assert!(!built, "fixtures/{name}.rs built");
    assert_eq!(said, lines, "fixtures/{name}.rs: Cargo said:\n{}", said.join("\n"));
}

#[test]
fn the_macro_says_what_it_refuses_where_it_is() {
    // The module given `STEP` that has its own comes last: Rust expands it a step later, when the
    // macro of the module of consts calls `__kernels!` back.
    refuses(
        "refused",
        &[
            "src/lib.rs:4:1: error: `#[kernels]` goes on a module written in braces, `mod name { .. }`, where the \
             kernels are",
            "src/lib.rs:6:24: error: `#[kernels]` takes the module of consts its kernels read, as `consts = \
             crate::model::sizes`, a module `#[kgen_macros::consts]` is on; or nothing",
            "src/lib.rs:9:23: error: `#[consts]` takes no arguments",
            "src/lib.rs:13:1: error: `#[consts]` goes on a module written in braces, `mod name { .. }`, of the consts \
             kernels read",
            "src/lib.rs:17:5: error: a const of a module of consts is `pub`, for the modules of kernels that take it \
             to see",
            "src/lib.rs:22:5: error: a module of consts holds the consts kernels read, as `pub const E: usize = 16;`, \
             and nothing else",
            "src/lib.rs:25:1: error: `__kernels!` is how `#[kernels(consts = ..)]` is given its consts",
            "src/lib.rs:27:1: error: `__kernels!` is how `#[kernels(consts = ..)]` is given its consts",
            "src/lib.rs:31:23: error: kgen compiles `extern \"sysv64\"` kernels, for x86-64, and `extern \
             \"ptx-kernel\"` ones, for NVIDIA GPUs, with the `extern \"ptx-func\"` functions they call; `extern \
             \"C\"` is none of these",
            "src/lib.rs:36:28: error: a kernel trusts the pointers it is given, so it is an `unsafe fn`",
            "src/lib.rs:51:7: error: `#[cooperative]` gives `Kernels` a method `k_blocks`, which says how many blocks \
             a grid of `k` has at most, and a kernel or a static of the module is named `k_blocks` as well",
            "src/lib.rs:59:39: error: `shfl.idx.b32` in `old` is for GPUs older than sm_70 from PTX 6.4 on, and \
             `tanh.approx.f32` in `new` needs sm_75 and PTX 7.0: no module has both",
            "src/lib.rs:46:15: error: `STEP` names one of the consts the module takes from `crate::sizes` as well",
            "error: could not compile `refused` (lib) due to 13 previous errors",
        ],
    );
}

#[test]
fn an_instance_not_listed_stops_the_build() {
    // `cargo check` evaluates no generic function's constants, so it says nothing of these.
    refuses(
        "instances",
        &[
            "src/lib.rs:4:1: error[E0080]: evaluation panicked: `k` is compiled for `k::<1>`, `k::<2>` alone, as its \
             `#[instances(..)]` lists: evaluation of `cpu::k::<3>::{constant#0}` failed here",
            "src/lib.rs:4:1: note: erroneous constant encountered",
            "src/lib.rs:23:14: note: the above error was encountered while instantiating `fn k::<3>`",
            "src/lib.rs:13:1: error[E0080]: evaluation panicked: `k` is compiled for `k::<1>`, `k::<2>` alone, as its \
             `#[instances(..)]` lists: evaluation of `gpu::Kernels::k::<3>::{constant#0}` failed here",
            "src/lib.rs:13:1: note: erroneous constant encountered",
            "src/lib.rs:28:14: note: the above error was encountered while instantiating `fn Kernels::k::<3>`",
            "error: could not compile `instances` (lib) due to 2 previous errors",
        ],
    );
}

#[test]
fn errors_inside_a_module_point_into_it() {
    // Rust's parser speaks first; then the macro, of the module as a whole, then of each item.
    refuses(
        "inside",
        &[
            "src/lib.rs:11:13: error: expected pattern, found `;`: expected pattern",
            "src/lib.rs:19:16: error: a static is memory of the module's kernels: the GPU's, for PTX kernels, or the \
             process's, for x86 ones; and this module has both, so it is two modules, one of each kind of kernel and \
             the statics they name",
            "src/lib.rs:7:22: error: `vaddps` wants a ymm; this is a gpr",
            "src/lib.rs:11:13: error: expected a name, found `;`",
            "src/lib.rs:14:7: error: a helper is compiled where it is called, for what `::<..>` gives it there: \
             `#[instances(..)]` lists what a function that names an ABI is compiled for",
            "error: could not compile `inside` (lib) due to 5 previous errors",
        ],
    );
}

#[test]
fn a_static_that_does_not_compile_is_said_once() {
    // Once where it is, not for each kernel that names it; each of those says it cannot.
    refuses(
        "statics",
        &[
            "src/lib.rs:5:24: error: `global<[u32; 0]>` holds nothing",
            "src/lib.rs:9:18: error: `COUNTS` does not compile, so no kernel names it",
            "src/lib.rs:14:18: error: `COUNTS` does not compile, so no kernel names it",
            "src/lib.rs:20:24: error: `global<[u32; 0]>` holds nothing",
            "src/lib.rs:24:21: error: `COUNTS` does not compile, so no kernel names it",
            "src/lib.rs:29:21: error: `COUNTS` does not compile, so no kernel names it",
            "error: could not compile `statics` (lib) due to 6 previous errors",
        ],
    );
}

#[test]
fn a_kernel_that_does_not_compile_leaves_a_stub() {
    // Rust calls each stub, and a kernel calls two, and none of these calls is an error.
    refuses(
        "stubs",
        &[
            "src/lib.rs:8:13: error: `y` is not defined",
            "src/lib.rs:13:13: error: in `tile::<1>`: `y` is not defined",
            "src/lib.rs:13:13: error: in `tile::<2>`: `y` is not defined",
            "src/lib.rs:18:18: error: `y` is not defined",
            "src/lib.rs:32:28: error: `y` is not defined",
            "src/lib.rs:37:28: error: `y` is not defined",
            "error: could not compile `stubs` (lib) due to 6 previous errors",
        ],
    );
}

#[test]
fn consts_are_taken_by_any_path() {
    builds_quietly("paths");
}

#[test]
fn what_a_module_of_consts_hands_on_and_refuses() {
    // Its items in order, each past its doc: its own attribute passes; the const that is not `pub`
    // is refused for that, before kgen works it out; the `pub` one at the call kgen cannot work
    // out; then the function. None is handed on, so the module of kernels keeps a `STEP` of its
    // own and says nothing.
    refuses(
        "consts",
        &[
            "src/lib.rs:10:5: error: a const of a module of consts is `pub`, for the modules of kernels that take it \
             to see",
            "src/lib.rs:13:27: error: a call is a statement: an instruction or a helper, not a value",
            "src/lib.rs:16:5: error: a module of consts holds the consts kernels read, as `pub const E: usize = 16;`, \
             and nothing else",
            "error: could not compile `consts` (lib) due to 3 previous errors",
        ],
    );
}

#[test]
fn kernels_that_compile_say_nothing() {
    builds_quietly("quiet");
}
