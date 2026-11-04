//! What the macro says of the kernels it is given: each error where it is, at its
//! `line:column`, in the order Cargo prints them, and each kernel's safety contract, in its
//! docs. Each fixture is built as a user's crate is, and what Cargo says of it is compared line
//! for line, the errors in rustc's short form, `src/lib.rs:4:1: error: ..`. A fixture is a
//! crate's `src/lib.rs`, so the line an error names is that line of the fixture. One that holds
//! tests is run as well, to show that its kernels do what they are written to.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::{Mutex, PoisonError};

/// Held while Cargo builds a fixture: the fixtures share a target directory, and a Cargo that
/// waits for another's lock says so.
static CARGO: Mutex<()> = Mutex::new(());

/// Where in `CARGO_TARGET_TMPDIR` each fixture is a crate, `kernite-macros/name/project`, so that
/// its workspace is not named as the crate is; and where they are built, `kernite-macros/target`.
const ROOT: &str = concat!(env!("CARGO_TARGET_TMPDIR"), "/kernite-macros");

/// Whether Cargo did `command`, `["build"]` say, for fixture `name`, written afresh, and what it
/// said, a line each, in order.
fn cargo(command: &[&str], name: &str) -> (bool, Vec<String>) {
    cargo_in(command, &written(name))
}

/// Fixture `name` written as the `src/lib.rs` of a crate that depends on kernite-macros,
/// kernite-cuda and kernite-cpu, as a user's does: the crate's directory. The crate is a workspace
/// of its own in `CARGO_TARGET_TMPDIR`. Its manifest ends with what `{name}.toml` beside the
/// fixture says of the workspace, if there is one.
fn written(name: &str) -> PathBuf {
    let here = Path::new(env!("CARGO_MANIFEST_DIR"));
    let crates = here.parent().expect("kernite-macros is in crates/");
    let fixtures = here.join("tests/fixtures");
    let dir = Path::new(ROOT).join(name).join("project");
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
kernite-macros = {{ path = {macros:?} }}
kernite-cuda = {{ path = {cuda:?} }}
kernite-cpu = {{ path = {cpu:?} }}

[workspace]
"#
    );
    let workspace = fs::read_to_string(fixtures.join(format!("{name}.toml"))).unwrap_or_default();
    fs::write(dir.join("Cargo.toml"), manifest + &workspace).unwrap();
    fs::copy(fixtures.join(format!("{name}.rs")), dir.join("src/lib.rs")).unwrap();
    dir
}

/// Fixture `name` as `written` writes it, but with each `from` of `edits` in its manifest written as its `to`.
/// A dependency is left out by making its line a comment: `("kernite-cuda =", "# kernite-cuda =")`.
fn edited(name: &str, edits: &[(&str, &str)]) -> PathBuf {
    let dir = written(name);
    let manifest = dir.join("Cargo.toml");
    let text = fs::read_to_string(&manifest).unwrap();
    fs::write(&manifest, edits.iter().fold(text, |text, (from, to)| text.replace(from, to))).unwrap();
    dir
}

/// Whether Cargo did `command` in the crate at `dir`, and what it said, a line each, in order.
fn cargo_in(command: &[&str], dir: &Path) -> (bool, Vec<String>) {
    let out = cargo_out(command, dir);
    let said = String::from_utf8(out.stderr).expect("Cargo writes UTF-8");
    (out.status.success(), said.lines().map(String::from).collect())
}

/// What Cargo did when it ran `command` in the crate at `dir`. The crate is built offline into
/// `kernite-macros/target`, so that it never waits for the build that runs these tests. What the
/// environment would change of a build is taken out of it.
fn cargo_out(command: &[&str], dir: &Path) -> Output {
    let _one = CARGO.lock().unwrap_or_else(PoisonError::into_inner);
    let mut run = Command::new(env!("CARGO"));
    run.args(command);
    run.args(["--offline", "--quiet", "--message-format=short", "--color=never", "--target-dir"]);
    run.arg(Path::new(ROOT).join("target")).current_dir(dir);
    for var in [
        "RUSTFLAGS",
        "CARGO_ENCODED_RUSTFLAGS",
        "CARGO_BUILD_RUSTFLAGS",
        "RUSTDOCFLAGS",
        "CARGO_ENCODED_RUSTDOCFLAGS",
        "CARGO_BUILD_RUSTDOCFLAGS",
        "RUSTC_WRAPPER",
        "RUSTC_WORKSPACE_WRAPPER",
        "CARGO_TARGET_DIR",
        "CARGO_BUILD_TARGET_DIR",
    ] {
        run.env_remove(var);
    }
    run.output().expect("Cargo runs")
}

/// That Cargo builds fixture `name` without a word.
fn builds_quietly(name: &str) {
    builds_quietly_in(&["build"], name, &written(name));
}

/// That Cargo does `command`, `["build", "--features", "cuda"]` say, in `dir`, where fixture `name` is
/// written, without a word.
fn builds_quietly_in(command: &[&str], name: &str, dir: &Path) {
    let (built, said) = cargo_in(command, dir);
    assert!(
        built && said.is_empty(),
        "fixtures/{name}.rs: {command:?} built {built}, and Cargo said:\n{}",
        said.join("\n")
    );
}

/// That Cargo does `command`, `["test", "--features", "big"]` say, for fixture `name`, without a word, and
/// the tests in it pass, its own and its doc tests, which are crates that call it: the kernels run, as a
/// crate that uses them runs them. Returns what the tests printed. A test that finds the GPU missing says
/// so in a line with "skipped" and passes: that line is no word of Cargo's, and it is said again here, as a
/// test of this file says of a tool it lacks.
fn runs_quietly(command: &[&str], name: &str) -> String {
    let out = cargo_out(command, &written(name));
    let (said, printed) = (String::from_utf8_lossy(&out.stderr), String::from_utf8_lossy(&out.stdout));
    let (skipped, said): (Vec<&str>, Vec<&str>) = said.lines().partition(|line| line.contains("skipped"));
    for line in skipped {
        eprintln!("{line}");
    }
    let (passed, said) = (out.status.success() && said.is_empty(), said.join("\n"));
    assert!(passed, "fixtures/{name}.rs: ran {}, and Cargo said:\n{said}\n{printed}", out.status.success());
    printed.into_owned()
}

/// That Cargo documents fixture `name` without a word; then where rustdoc wrote its pages.
fn documents_quietly(name: &str) -> PathBuf {
    let (done, said) = cargo(&["doc", "--no-deps"], name);
    assert!(done && said.is_empty(), "fixtures/{name}.rs: documented {done}, and Cargo said:\n{}", said.join("\n"));
    Path::new(ROOT).join("target/doc").join(name)
}

/// That Cargo documents fixture `name` without a word; then, of each function `f` of its module
/// `module`, the lines rustdoc wrote of it, from `start` to the `</div>` after it.
fn documented(name: &str, module: &str, start: &str) -> impl Fn(&str) -> Vec<String> {
    let dir = documents_quietly(name).join(module);
    move |f: &str| {
        let page = fs::read_to_string(dir.join(format!("fn.{f}.html"))).unwrap();
        let (_, docs) = page.split_once(start).unwrap_or_else(|| panic!("fn.{f}.html has no `{start}`"));
        docs.split_once("</div>").expect("the docs end").0.lines().map(String::from).collect()
    }
}

/// That Cargo does not build fixture `name`, and says `lines`, in that order, and nothing else.
fn refuses(name: &str, lines: &[&str]) {
    refuses_by(&["build"], name, lines);
}

/// That Cargo does not do `command`, `["build", "--features", "cpu"]` say, for fixture `name`, and says
/// `lines`, in that order, and nothing else.
fn refuses_by(command: &[&str], name: &str, lines: &[&str]) {
    let (done, said) = cargo(command, name);
    assert!(!done, "fixtures/{name}.rs: {command:?} did it");
    assert_eq!(said, lines, "fixtures/{name}.rs: Cargo said:\n{}", said.join("\n"));
}

/// What Rust says of `name`, a type that is not there, at `place` of `src/lib.rs`.
fn not_found(place: &str, name: &str) -> String {
    format!("src/lib.rs:{place}: error[E0425]: cannot find type `{name}` in this scope: not found in this scope")
}

/// What Rust says of `name`, a value that is not there, at `place` of `src/lib.rs`.
fn no_value(place: &str, name: &str) -> String {
    format!("src/lib.rs:{place}: error[E0425]: cannot find value `{name}` in this scope: not found in this scope")
}

/// What kernite says of `name`, which `cfg` leaves out, named by `reader`, which `cfg` keeps, at `place` of
/// `src/lib.rs`.
fn left_out(place: &str, name: &str, reader: &str) -> String {
    format!("src/lib.rs:{place}: error: `{name}` is left out by `cfg` where `{reader}`, which names it, is kept")
}

/// What Rust says of a pointer to a type with no size, or of a value of one, `ty` the part of it that
/// has none, at `place` of `src/lib.rs`: that its size cannot be known.
fn size_unknown(place: &str, ty: &str) -> String {
    format!(
        "src/lib.rs:{place}: error[E0277]: the size for values of type `{ty}` cannot be known at compilation \
         time: doesn't have a size known at compile-time"
    )
}

/// What Rust says of a number's name that the module gives another type, `found`, at `place` of
/// `src/lib.rs`: that the type is not the number, `expected`.
fn not_the_number(place: &str, expected: &str, found: &str) -> String {
    format!("src/lib.rs:{place}: error[E0308]: mismatched types: expected `{expected}`, found `{found}`")
}

/// Whether this machine has Clippy, which the tests that run it skip without, and say so.
fn clippy_here() -> bool {
    let has = Command::new(env!("CARGO")).args(["clippy", "--version"]).output().is_ok_and(|out| out.status.success());
    if !has {
        eprintln!("no Clippy here: skipped");
    }
    has
}

/// A machine other than x86-64 to check a crate for: a 32-bit ARM, which has no `extern "sysv64"`. Cargo checks a
/// crate for it without a linker, so long as the target's standard library is here.
const ARM: &str = "arm-unknown-linux-gnueabihf";

/// Whether this machine has the standard library of [`ARM`], which the tests that check a crate for it skip
/// without, and say so.
fn arm_here() -> bool {
    let libs = Command::new("rustc").args(["--print", "target-libdir", "--target", ARM]).output();
    let has = libs.is_ok_and(|out| Path::new(String::from_utf8_lossy(&out.stdout).trim()).is_dir());
    if !has {
        eprintln!("no standard library for {ARM} here: skipped");
    }
    has
}

/// That Cargo checks fixture `name` for [`ARM`] without a word, if this machine can check a crate for it.
fn checks_quietly_for_arm(name: &str) {
    if arm_here() {
        builds_quietly_in(&["check", "--target", ARM], name, &written(name));
    }
}

#[test]
fn the_macro_says_what_it_refuses_where_it_is() {
    // The modules given `STEP` come last: Rust expands each a step later, when the macro of the
    // module of consts calls `__kernels!` back.
    refuses(
        "refused",
        &[
            "src/lib.rs:4:1: error: `#[kernels]` goes on a module written in braces, `mod name { .. }`, where the \
             kernels are",
            "src/lib.rs:6:27: error: `#[kernels]` takes the module of consts its kernels read, as `consts = \
             crate::model::sizes`, a module `#[kernite_macros::consts]` is on; or nothing",
            "src/lib.rs:9:26: error: `#[consts]` takes no arguments",
            "src/lib.rs:13:1: error: `#[consts]` goes on a module written in braces, `mod name { .. }`, of the consts \
             kernels read",
            "src/lib.rs:17:5: error: a const of a module of consts is `pub`, for the modules of kernels that take it \
             to see",
            "src/lib.rs:22:5: error: a module of consts holds the consts kernels read, as `pub const E: usize = 16;`, \
             and nothing else",
            "src/lib.rs:25:1: error: `__kernels!` is how `#[kernels(consts = ..)]` is given its consts",
            "src/lib.rs:27:1: error: `__kernels!` is how `#[kernels(consts = ..)]` is given its consts",
            "src/lib.rs:31:23: error: kernite compiles `extern \"sysv64\"` kernels, for x86-64, and `extern \
             \"ptx-kernel\"` ones, for NVIDIA GPUs, with the `extern \"ptx-func\"` functions they call; `extern \
             \"C\"` is none of these",
            "src/lib.rs:36:28: error: a kernel trusts the pointers it is given, so it is an `unsafe fn`",
            "src/lib.rs:51:7: error: `#[cooperative]` gives `Kernels` a method `k_blocks`, which says how many blocks \
             a grid of `k` has at most, and a kernel or a static of the module is named `k_blocks` as well",
            "src/lib.rs:59:39: error: `shfl.idx.b32` in `old` is for GPUs older than sm_70 from PTX 6.4 on, and \
             `tanh.approx.f32` in `new` needs sm_75 and PTX 7.0: no module has both",
            "src/lib.rs:73:8: error: `h` names another function of the module as well",
            "src/lib.rs:46:15: error: `STEP` names one of the consts the module takes from `crate::sizes` as well",
            "error: could not compile `refused` (lib) due to 14 previous errors",
        ],
    );
}

#[test]
fn brackets_nested_past_64_deep_are_refused_where_they_open() {
    // The 64th parenthesis is the first past the most, as `kernite::x86_str` counts the same
    // tokens.
    refuses(
        "nested",
        &[
            "src/lib.rs:7:80: error: this nests 64 deep at most",
            "error: could not compile `nested` (lib) due to 1 previous error",
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
fn the_workspace_is_named_as_its_directory_is_unless_its_manifest_names_it() {
    // Each fixture asserts, as it builds, what its kernel is named in PTX: as the directory of
    // its workspace, `project`, is; or as `renamed.toml` names the workspace.
    builds_quietly("named");
    builds_quietly("renamed");
}

#[test]
fn a_workspace_named_nothing_is_left_out_of_its_kernels_names() {
    // The fixture asserts, as it builds, that its kernel is named in PTX in its crate's namespace
    // and its module's alone: `unnamed.toml` names the workspace `''`.
    builds_quietly("unnamed");
}

#[test]
fn a_crate_is_built_again_when_its_workspace_is_renamed() {
    // The crate is a member of a workspace one directory up, as a user's is, each example among
    // them. It is not a workspace of its own, where its manifest and the workspace's would be one
    // file: the macro includes the workspace's manifest, not the crate's own, and this test tells
    // the two apart. The test writes the workspace's manifest, naming it `before`, and the fixture
    // asserts, as it builds, that its kernel is named so. The test then names it `after`, and
    // Cargo builds the crate again: the kernel is `after::rebuilt::gpu::k` now, and the assert
    // fails. Cargo names the file from the workspace's root, `project/src/lib.rs`.
    let dir = written("rebuilt");
    let manifest = dir.join("Cargo.toml");
    let member = fs::read_to_string(&manifest).unwrap().replace("[workspace]\n", "");
    fs::write(&manifest, member).unwrap();
    let named = |name: &str| {
        let workspace = format!(
            "[workspace]\nresolver = \"3\"\nmembers = [\"project\"]\n\n[workspace.metadata.kernite]\nname = {name:?}\n"
        );
        fs::write(Path::new(ROOT).join("rebuilt/Cargo.toml"), workspace).unwrap();
    };
    named("before");
    let (built, said) = cargo_in(&["build"], &dir);
    assert!(built && said.is_empty(), "fixtures/rebuilt.rs: built {built}, and Cargo said:\n{}", said.join("\n"));
    named("after");
    let (built, said) = cargo_in(&["build"], &dir);
    assert!(!built, "fixtures/rebuilt.rs built, its workspace named `after`");
    let error = "project/src/lib.rs:18:41: error[E0080]: evaluation panicked: the kernel is not named \
                 before::rebuilt::gpu::k: evaluation of `_` failed here";
    assert_eq!(said, [error, "error: could not compile `rebuilt` (lib) due to 1 previous error"]);
}

#[test]
fn a_workspace_named_by_what_is_not_a_string_is_refused_where_the_module_is() {
    // The error is at the attribute, and names the manifest, which is the crate's own: the crate is
    // a workspace of its own. The kernel compiles, so the error is the only one.
    let manifest = Path::new(ROOT).join("nonstring/project/Cargo.toml");
    let error = format!(
        "src/lib.rs:5:1: error: `name` in `[workspace.metadata.kernite]` of {} is not a string: a workspace is named by \
         one, as `name = \"kernite\"`",
        manifest.display()
    );
    refuses("nonstring", &[&error, "error: could not compile `nonstring` (lib) due to 1 previous error"]);
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
fn what_a_module_says_when_its_parts_fail() {
    // The module given `STEP` comes last: Rust expands it a step later, when the macro of the
    // module of consts calls `__kernels!` back.
    refuses(
        "parts",
        &[
            "src/lib.rs:8:7: error: `bad` takes no compile-time arguments, so it is compiled once, with no \
             `#[instances(..)]`",
            "src/lib.rs:12:7: error: `#[cooperative]` goes on a kernel whose blocks or threads wait for each other, \
             `extern \"ptx-kernel\"` or `extern \"sysv64\"`; `f` is a function the kernels call, which runs as the \
             kernel that calls it is launched",
            "src/lib.rs:17:32: error: a kernel trusts the pointers it is given, so it is an `unsafe fn`",
            "src/lib.rs:9:39: error: `shfl.idx.b32` in `old` is for GPUs older than sm_70 from PTX 6.4 on, and \
             `tanh.approx.f32` in `new` needs sm_75 and PTX 7.0: no module has both",
            "src/lib.rs:47:12: error: a static is memory of the module's kernels: the GPU's, for PTX kernels, or the \
             process's, for x86 ones; and this module has both, so it is two modules, one of each kind of kernel and \
             the statics they name",
            "src/lib.rs:55:23: error: kernite compiles `extern \"sysv64\"` kernels, for x86-64, and `extern \
             \"ptx-kernel\"` ones, for NVIDIA GPUs, with the `extern \"ptx-func\"` functions they call; `extern \
             \"C\"` is none of these",
            "src/lib.rs:36:16: error: `STEP` names one of the consts the module takes from `crate::sizes` as well",
            "error: could not compile `parts` (lib) due to 7 previous errors",
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

/// A crate that does not depend on `kernite-cpu` is told where it needs the crate, at each static of
/// a module of x86 kernels and at each cooperative kernel, by kernite and not by Rust, which would
/// say that `kernite_cpu` is not found, at the attribute.
#[test]
fn a_crate_without_kernite_cpu_is_told_at_what_needs_it() {
    let (built, said) = cargo_in(&["build"], &edited("no_cpu", &[("kernite-cpu =", "# kernite-cpu =")]));
    assert!(!built, "fixtures/no_cpu.rs built");
    assert_eq!(
        said,
        [
            "src/lib.rs:8:16: error: a static of x86 kernels needs a dependency named `kernite-cpu`, \
             which this crate does not have",
            "src/lib.rs:21:35: error: a cooperative x86 kernel needs a dependency named `kernite-cpu`, \
             which this crate does not have",
            "src/lib.rs:28:35: error: a cooperative x86 kernel needs a dependency named `kernite-cpu`, \
             which this crate does not have",
            "error: could not compile `no_cpu` (lib) due to 3 previous errors",
        ]
    );
}

/// A crate that does not depend on `kernite-cuda` is told where it needs the crate, once at each
/// module of GPU kernels, by kernite and not by Rust, which would say that `kernite_cuda` is not
/// found, at the attribute; and nothing is said of the code that uses the module, nor of a module
/// that gives `str` another type, whose empty `PTX` is a `&str` all the same. The crate's own name
/// holds `kernite-cuda`, which does not make it a dependency.
#[test]
fn a_crate_without_kernite_cuda_is_told_at_each_module_of_gpu_kernels() {
    let edits = [("kernite-cuda =", "# kernite-cuda ="), ("\"no_cuda\"", "\"my-kernite-cuda-app\"")];
    let (built, said) = cargo_in(&["build"], &edited("no_cuda", &edits));
    assert!(!built, "fixtures/no_cuda.rs built");
    assert_eq!(
        said,
        [
            "src/lib.rs:7:1: error: a module of GPU kernels needs a dependency named `kernite-cuda`, \
             which this crate does not have",
            "src/lib.rs:31:1: error: a module of GPU kernels needs a dependency named `kernite-cuda`, \
             which this crate does not have",
            "src/lib.rs:68:1: error: a module of GPU kernels needs a dependency named `kernite-cuda`, \
             which this crate does not have",
            "error: could not compile `my-kernite-cuda-app` (lib) due to 3 previous errors",
        ]
    );
}

/// A crate whose dependencies on `kernite-cuda` and `kernite-cpu` are renamed in its manifest,
/// `cuda = { package = "kernite-cuda", .. }`, is told as one that has none is, for the Rust kernite
/// writes names `kernite_cuda` and `kernite_cpu`, which it has not; and Rust says nothing.
#[test]
fn a_crate_that_renames_kernite_cuda_and_kernite_cpu_is_told_it_has_none() {
    let edits = [
        ("kernite-cuda = {", "cuda = { package = \"kernite-cuda\","),
        ("kernite-cpu = {", "cpu = { package = \"kernite-cpu\","),
    ];
    let (built, said) = cargo_in(&["build"], &edited("renamed_deps", &edits));
    assert!(!built, "fixtures/renamed_deps.rs built");
    assert_eq!(
        said,
        [
            "src/lib.rs:6:1: error: a module of GPU kernels needs a dependency named `kernite-cuda`, \
             which this crate does not have",
            "src/lib.rs:15:16: error: a static of x86 kernels needs a dependency named `kernite-cpu`, \
             which this crate does not have",
            "src/lib.rs:18:35: error: a cooperative x86 kernel needs a dependency named `kernite-cpu`, \
             which this crate does not have",
            "error: could not compile `renamed_deps` (lib) due to 3 previous errors",
        ]
    );
}

/// A kernel's safety contract says that nothing else touches the memory of the module's it
/// names while it runs, with what the functions it calls name, whether it calls them by name
/// or at an address: in a `# Safety` of its own, or after the one its docs have; and for a
/// grid's kernel, while its threads run.
#[test]
fn a_kernels_safety_contract_says_nothing_else_touches_the_memory_it_names() {
    // The paragraphs of the kernel's `# Safety`, as rustdoc writes them.
    let safety = documented("contract", "cpu", ">Safety</h2>\n");
    let runs = "<p>The kernel trusts the pointers it is given: each must be to memory as long as the kernel reads and \
                writes through it. Nothing else touches the module’s memory, <code>RUNS</code>, while it runs.</p>";
    assert_eq!(safety("run"), [runs]);
    // A call at an address names no function in the code, and the contract names the memory of
    // each it may reach: `run` is the last of the two, and `idle` names none.
    assert_eq!(safety("either"), [runs]);
    assert_eq!(
        safety("scale"),
        [
            "<p><code>x</code> points at an integer.</p>",
            "<p>Nothing else touches the module’s memory, <code>RUNS</code> and <code>FACTORS</code>, while it \
             runs.</p>",
        ]
    );
    assert_eq!(
        safety("runs"),
        ["<p>The kernel trusts the pointers it is given: each must be to memory as long as the kernel’s threads \
             read and write through it, which nothing else touches while they run. Nothing else touches the \
             module’s memory, <code>RUNS</code>, while they run.</p>"]
    );
}

/// The function's docs name the CPU features its kernel needs past x86-64's own, which nothing
/// checks for when it runs, as the listing's header does: three, two, one, or, for a kernel
/// that needs none, no paragraph. A generic kernel's docs name those of every instance, each
/// once, in the order of their names, and a grid's kernel's launcher's those of the kernel.
#[test]
fn the_functions_docs_name_the_cpu_features_a_kernel_needs() {
    // The paragraphs of the function's docs, as rustdoc writes them.
    let docs = documented("needs", "kernels", "<div class=\"docblock\">");
    let needs =
        |list: &str| format!("<p>Needs the CPU {list}, past x86-64’s own, which nothing checks for when it runs.</p>");
    assert_eq!(
        docs("bits"),
        [
            "<p>How many bits of <code>x</code> are set, lead and trail, summed.</p>".to_string(),
            needs("features <code>bmi1</code>, <code>lzcnt</code> and <code>popcnt</code>"),
        ]
    );
    assert_eq!(
        docs("axpy"),
        [
            "<p><code>y += a * x</code>, eight floats at once.</p>".to_string(),
            needs("features <code>avx</code> and <code>fma</code>"),
        ]
    );
    assert_eq!(
        docs("ones"),
        ["<p>How many bits of <code>x</code> are set.</p>".to_string(), needs("feature <code>popcnt</code>")]
    );
    assert_eq!(docs("next"), ["<p><code>x</code> and one.</p>"]);
    assert_eq!(
        docs("pick"),
        [
            "<p>How many zeros lead <code>x</code>, or how many of its bits are set.</p>".to_string(),
            needs("features <code>lzcnt</code> and <code>popcnt</code>"),
        ]
    );
    assert_eq!(
        docs("count"),
        [
            "<p>How many bits of <code>x</code> are set, or how many zeros lead it, when <code>N</code> is 2.</p>"
                .to_string(),
            needs("features <code>lzcnt</code> and <code>popcnt</code>"),
        ]
    );
    assert_eq!(
        docs("together")[..2],
        [
            "<p>How many bits are set of the word at <code>x</code>, once every thread of the grid is there.</p>"
                .to_string(),
            needs("feature <code>popcnt</code>"),
        ]
    );
}

/// The module's `LISTINGS` holds each x86 kernel's listing by its name, and an instance's by
/// its name and arguments, as `plus::<2>`. The fixture asserts each, name and listing, as it
/// builds.
#[test]
fn listings_holds_each_kernels_listing_by_its_name() {
    builds_quietly("listings");
}

/// `LISTINGS` holds the listings of the kernels `cfg` keeps, one for each instance of a kernel that
/// has them, and none of a kernel that `cfg` leaves out, by a `cfg` or by a `cfg_attr(..)`. The
/// fixture asserts, as it builds, which listings the module has with its feature and without it.
#[test]
fn listings_holds_what_cfg_keeps() {
    let dir = written("configured_listings");
    builds_quietly_in(&["build"], "configured_listings", &dir);
    builds_quietly_in(&["build", "--features", "more"], "configured_listings", &dir);
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

/// A module of GPU kernels holds what `cfg` keeps: a static, a function that kernels call or a kernel
/// that `cfg` leaves out is neither in its `PTX` nor among the names `Kernels::load` asks the driver
/// for, and what comes after it is where it was. Its `.version` and `.target` are the oldest that have
/// every instruction of the kernels and functions `cfg` keeps, though the kernel that calls a function
/// is left out. The fixture's tests run with each of its features and without; they assert which of the
/// module is in `PTX` and, on a machine with a GPU, load it and launch each kernel that is there.
#[test]
fn a_gpu_module_holds_what_cfg_keeps() {
    for features in ["", "more", "new", "more,new"] {
        let command = ["test", "--features", features];
        let printed = runs_quietly(&command, "configured_ptx");
        assert!(
            printed.contains("test result: ok. 5 passed"),
            "fixtures/configured_ptx.rs: {command:?} printed:\n{printed}"
        );
    }
}

/// A kernel, a helper or a static that `cfg` leaves out says nothing of what is wrong with it, as
/// Rust says nothing of a function it leaves out: the crate builds without a word, under
/// `deny(warnings)`, and with the feature that keeps them each error is said where it is, in its
/// words, as for any kernel. The fixture leaves them out by `cfg`, by a `cfg` written raw, by a
/// `cfg_attr(..)` and by one in another, and by two `cfg`s, the second of which never holds. It
/// ends with consts under the feature that a kernel, a function and the length of a static read,
/// each under it too; with one that a helper reads, which has no `cfg`, for the kernel that calls
/// it, which has; and with a const a module of consts gives that only a kernel under it reads. Rust
/// never sees a kernel read them, so kernite names each under the `cfg` of the item that reads it,
/// a helper's under its caller's, and says nothing of it where that item is not, nor, with the
/// feature, any error of its own.
#[test]
fn what_cfg_leaves_out_says_nothing() {
    builds_quietly("configured");
    refuses_by(
        &["build", "--features", "cpu"],
        "configured",
        &[
            // The x86 kernels: `narrow`, `undefined`, `unknown`, `abi`, `safe`, each instance of `tile`,
            // `plain`, `together`, `taken`, the helper `help`, the static `EMPTY`, `conditional` and `nested`; and
            // nothing of `never`, which two `cfg`s leave out whatever the feature says.
            "src/lib.rs:17:45: error: SysV leaves the high bits of a `u32` as they were; take a usize or a u64",
            "src/lib.rs:28:23: error: `y` is not defined",
            "src/lib.rs:34:9: error: x86 has no instruction `frobnicate` that kernite knows",
            "src/lib.rs:39:23: error: kernite compiles `extern \"sysv64\"` kernels, for x86-64, and `extern \
             \"ptx-kernel\"` ones, for NVIDIA GPUs, with the `extern \"ptx-func\"` functions they call; `extern \
             \"C\"` is none of these",
            "src/lib.rs:45:28: error: a kernel trusts the pointers it is given, so it is an `unsafe fn`",
            "src/lib.rs:53:13: error: in `tile::<1>`: `y` is not defined",
            "src/lib.rs:53:13: error: in `tile::<2>`: `y` is not defined",
            "src/lib.rs:58:7: error: `plain` takes no compile-time arguments, so it is compiled once, with no \
             `#[instances(..)]`",
            "src/lib.rs:67:23: error: `y` is not defined",
            "src/lib.rs:75:47: error: `grid` names a parameter of the launcher as well",
            "src/lib.rs:82:9: error: a kernel has no `while`: loops are labels and jumps, and `for` is unrolled",
            "src/lib.rs:87:23: error: `global<[u32; 0]>` holds nothing",
            "src/lib.rs:92:23: error: `y` is not defined",
            "src/lib.rs:98:23: error: `y` is not defined",
            // The GPU kernels in the same order, the function `func` among them; then the compile-time arguments
            // of `taken` and `counted` named as a parameter of a launcher, which the module says after the rest.
            "src/lib.rs:119:28: error: `y` is not defined",
            "src/lib.rs:125:20: error: PTX has no instruction `frobnicate` that kernite knows",
            "src/lib.rs:130:32: error: a kernel trusts the pointers it is given, so it is an `unsafe fn`",
            "src/lib.rs:138:28: error: in `tile::<1>`: `y` is not defined",
            "src/lib.rs:138:28: error: in `tile::<2>`: `y` is not defined",
            "src/lib.rs:143:7: error: `plain` takes no compile-time arguments, so it is compiled once, with no \
             `#[instances(..)]`",
            "src/lib.rs:152:28: error: `y` is not defined",
            "src/lib.rs:176:28: error: `y` is not defined",
            "src/lib.rs:181:23: error: `global<[u32; 0]>` holds nothing",
            "src/lib.rs:186:28: error: `y` is not defined",
            "src/lib.rs:159:51: error: `grid` names a parameter of the launcher as well",
            "src/lib.rs:169:53: error: `threads` names a parameter of `counted_blocks` as well",
            // A cooperative kernel whose method `k_blocks` has the name of a kernel of the module.
            "src/lib.rs:195:7: error: `#[cooperative]` gives `Kernels` a method `k_blocks`, which says how many \
             blocks a grid of `k` has at most, and a kernel or a static of the module is named `k_blocks` as well",
            "error: could not compile `configured` (lib) due to 27 previous errors",
        ],
    );
}

/// What rests on all of a module's items together is said as it is for a module that has no `cfg`,
/// and so is an item kernite cannot read, the head of a function, which rests on no other: kernite
/// reads every item before it knows which `cfg` holds, as Rust parses every item before it does.
/// Each case in the fixture has a function that `cfg` leaves out.
#[test]
fn what_rests_on_all_the_items_is_said_whatever_cfg_leaves_out() {
    // The modules in order: `head`, `twice`, `blocks`, and `ptx`, whose error is at its first
    // kernel, which is left out.
    refuses(
        "configured_module",
        &[
            "src/lib.rs:9:43: error: a function's generics are values, as in `<const N: usize>`",
            "src/lib.rs:23:35: error: `k` names another function of the module as well",
            "src/lib.rs:31:7: error: `#[cooperative]` gives `Kernels` a method `k_blocks`, which says how many \
             blocks a grid of `k` has at most, and a kernel or a static of the module is named `k_blocks` as well",
            "src/lib.rs:42:39: error: `shfl.idx.b32` in `old` is for GPUs older than sm_70 from PTX 6.4 on, and \
             `tanh.approx.f32` in `new` needs sm_75 and PTX 7.0: no module has both",
            "error: could not compile `configured_module` (lib) due to 4 previous errors",
        ],
    );
}

/// A const `cfg` keeps that only what `cfg` leaves out reads is unused to Rust, which says so, as it
/// does of a const that nothing reads; with the kernel there it is used, and Rust says nothing.
#[test]
fn rust_finds_a_const_unused_that_only_what_cfg_leaves_out_reads() {
    let (built, said) = cargo(&["build"], "configured_unused");
    assert!(built, "fixtures/configured_unused.rs did not build");
    assert_eq!(said, ["src/lib.rs:6:11: warning: constant `N` is never used"]);
    builds_quietly_in(&["build", "--features", "cpu"], "configured_unused", &written("configured_unused"));
}

/// A kernel that `cfg` keeps names only what `cfg` keeps: where it names an item `cfg` leaves out, the name
/// is an error, said at the name, as Rust says it of a function it keeps that names one it leaves out. Rust
/// says it, "cannot find value", of what is Rust's own: a const, a static of the process's and an x86
/// kernel that takes no compile-time arguments, named in a kernel's body, in its `#[instances(..)]`, in its
/// directives and in the type or the value of a static; a const whose definition names one is Rust's alone,
/// and Rust says it once, there. kernite says it of what is not Rust's: a helper, an instance of an x86
/// kernel, an `extern "ptx-func"` function and a static of the GPU's. Each place a name is written is said
/// once, in a kernel of instances too. What a helper names is its caller's, as it is for a helper a helper
/// calls, and is said where `cfg` keeps the helper as well: a kept kernel that calls a helper `cfg` leaves
/// out is told of the helper alone, whatever the helper names, and one that calls a helper at two places,
/// one in a helper `cfg` leaves out, is told of what the helper names, once. A helper that calls itself, or
/// two that call each other, as long as a value known now says, are followed once: what they name is said
/// once for each place, however often the kernel runs them. A kept kernel that calls a pair `cfg` leaves
/// out is told of the helper it calls alone, and so is one that reaches a pair only through a helper `cfg`
/// leaves out, though the pair names a const `cfg` leaves out with that helper. The fixture's modules are,
/// in order, the x86 kernels Rust has names for, those it has none for, those that call helpers, the GPU's
/// kernels, two of which nothing is said: one whose kernels `cfg` leaves out with what they name, or keeps
/// only where it keeps that, and one that the features keep as they keep what it names; and last the
/// helpers that loop: one that calls itself, a pair, a pair `cfg` leaves out, and a pair that only a helper
/// `cfg` leaves out calls.
#[test]
fn a_kernel_that_cfg_keeps_is_told_at_each_name_that_cfg_leaves_out() {
    refuses(
        "configured_names",
        &[
            // kernite's, an item at a time, in the order of the source, and the places of an item likewise.
            &left_out("106:9", "seven", "calls_helper"),
            &left_out("95:9", "inner", "calls_through"),
            &left_out("116:17", "tile", "calls_instance"),
            &left_out("122:9", "seven", "calls_helper_of_instances"),
            &left_out("123:9", "seven", "calls_helper_of_instances"),
            &left_out("171:9", "left_out", "calls_left_out"),
            &left_out("175:9", "calls_twice", "calls_both_ways"),
            &left_out("205:17", "nine", "via"),
            &left_out("224:9", "left_out", "calls_left_out"),
            &left_out("229:21", "COUNT", "reads_static"),
            &left_out("235:17", "nine", "calls_function"),
            &left_out("242:17", "nine", "calls_function_of_instances"),
            &left_out("383:9", "tick", "calls_left_out_pair"),
            &left_out("411:9", "entry", "calls_entry"),
            // Rust's, a module at a time, in the order it reads what the macro writes: the items first, a
            // const's definition and the type and value of a static, then the names its kernels say, in
            // the order of the source.
            &no_value("30:24", "SIZE"),
            &no_value("32:37", "LEN"),
            &no_value("34:37", "SIZE"),
            &no_value("44:23", "SIZE"),
            &no_value("53:22", "COUNT"),
            &no_value("59:17", "eight"),
            &no_value("63:17", "LIST"),
            &no_value("71:22", "COUNT"),
            &no_value("72:16", "SIZE"),
            &no_value("73:16", "SIZE"),
            &no_value("162:23", "FOUR"),
            &no_value("194:37", "LEN"),
            &no_value("206:23", "SIZE"),
            &no_value("220:28", "SIZE"),
            &no_value("246:15", "SIZE"),
            &no_value("338:23", "STEP"),
            "error: could not compile `configured_names` (lib) due to 30 previous errors",
        ],
    );
}

/// What is told is judged by the conditions of both, the kernel's and the item's. With `big`, an item that
/// `big` and `wide` keep, and a helper under `cfg_attr(big, cfg(wide))`, are gone where `big` holds alone,
/// and the kernels that name them are told there, a kernel that `big` keeps and one that is always kept;
/// not where neither holds, nor where both do.
#[test]
fn a_name_is_told_where_the_kernel_is_kept_and_the_item_is_not() {
    refuses_by(
        &["build", "--features", "big"],
        "configured_names",
        &[
            &left_out("320:9", "six", "calls_six"),
            &left_out("324:9", "five", "calls_five"),
            &no_value("315:23", "WIDE"),
            "error: could not compile `configured_names` (lib) due to 3 previous errors",
        ],
    );
}

/// What `cfg` keeps of what the kernels name builds without a word, under `deny(warnings)`, and the x86
/// kernels run as they did: each writes through its pointer what it names, as the test the fixture holds
/// finds.
#[test]
fn what_cfg_keeps_of_what_a_kernel_names_builds_and_runs() {
    let printed = runs_quietly(&["test", "--features", "big,wide"], "configured_names");
    assert!(printed.contains("1 passed"), "fixtures/configured_names.rs ran its test, and printed:\n{printed}");
}

/// A kernel of compile-time arguments and a cooperative one that a feature leaves out with what they name, a
/// static and an x86 kernel, say nothing without it. With it, they build without a word, and run.
#[test]
fn kernels_of_instances_and_cooperative_ones_that_a_feature_leaves_out_say_nothing() {
    builds_quietly("configured_hidden");
    let printed = runs_quietly(&["test", "--features", "big"], "configured_hidden");
    assert!(printed.contains("1 passed"), "fixtures/configured_hidden.rs ran its test, and printed:\n{printed}");
}

/// A crate that keeps its x86 kernels only on x86-64, by `cfg(target_arch = "x86_64")`, builds for another
/// machine without a word, under `deny(warnings)`. Only x86-64 has `extern "sysv64"`, so each function the
/// macros write for a kernel goes with it, one for each instance as well. The fixture has a kernel whose code
/// holds a constant, one of compile-time arguments, and one of those whose instances hold a constant. For
/// x86-64 it keeps them, and they run.
#[test]
fn x86_kernels_that_cfg_leaves_out_say_nothing_for_another_machine() {
    checks_quietly_for_arm("other_machine");
    let printed = runs_quietly(&["test"], "other_machine");
    assert!(printed.contains("1 passed"), "fixtures/other_machine.rs ran its test, and printed:\n{printed}");
}

/// So do cooperative x86 kernels, a plain one, one whose code holds a constant and one of compile-time
/// arguments, in a crate that takes `kernite-cpu` on every machine. For x86-64 they run on a grid.
#[test]
fn cooperative_x86_kernels_that_cfg_leaves_out_say_nothing_for_another_machine() {
    checks_quietly_for_arm("other_machine_cooperative");
    let printed = runs_quietly(&["test"], "other_machine_cooperative");
    assert!(
        printed.contains("1 passed"),
        "fixtures/other_machine_cooperative.rs ran its test, and printed:\n{printed}"
    );
}

/// So does an x86 kernel of compile-time arguments that kernite refuses: its stub has a function for each
/// instance, for the kernels that call one, and it goes with the kernel too. For x86-64 the kernel is kernite's
/// error for each instance, and nothing else.
#[test]
fn a_refused_x86_kernel_of_instances_that_cfg_leaves_out_says_nothing_for_another_machine() {
    checks_quietly_for_arm("other_machine_refused");
    refuses(
        "other_machine_refused",
        &[
            "src/lib.rs:10:13: error: in `tiles::<1>`: `v` is read before anything writes it",
            "src/lib.rs:10:13: error: in `tiles::<2>`: `v` is read before anything writes it",
            "error: could not compile `other_machine_refused` (lib) due to 2 previous errors",
        ],
    );
}

/// A module of GPU kernels needs `kernite-cuda` where `cfg` keeps one of its GPU items, and not
/// where it leaves out all: a feature that brings the crate, `cuda = ["dep:kernite-cuda"]`, can keep
/// them, and the crate builds without the feature, without a word, under `deny(warnings)`, and with
/// it. A kernel, a function that kernels call and a static each make a module of their own, under
/// `cfg` and under `cfg_attr(.., cfg(..))`.
#[test]
fn a_feature_that_brings_kernite_cuda_can_keep_the_gpu_items() {
    let optional = [("kernite-cuda = {", "kernite-cuda = { optional = true,")];
    let dir = edited("configured_cuda", &optional);
    builds_quietly_in(&["build"], "configured_cuda", &dir);
    builds_quietly_in(&["build", "--features", "cuda"], "configured_cuda", &dir);
}

/// A crate without `kernite-cuda` is told so once, where a module of GPU kernels is, and only where
/// `cfg` keeps one of its GPU items, which the features say. Each module of the fixture has its
/// GPU items under `a` and `b` in a form `cfg` takes them by, and the table has the modules that
/// are told so, for each set of features: a kernel under `a`; a kernel under `a`, written raw, and a
/// function under `b`; a kernel under two `cfg`s; a kernel under `cfg_attr(a, cfg(b))`, left out
/// only where `a` holds and `b` does not; one under a `cfg_attr(..)` in another, left out only where
/// both hold; a static under `b`; a kernel `cfg` does not touch and one under `a`; a GPU kernel
/// under `a` and an x86 kernel under `b`, which needs no crate.
#[test]
fn a_module_needs_kernite_cuda_where_cfg_keeps_one_of_its_gpu_items() {
    let dir = edited("configured_gpu", &[("kernite-cuda =", "# kernite-cuda =")]);
    let [just_a, a_or_b, a_and_b, a_then_b, nested, static_b, always, mixed] = [6, 15, 31, 41, 51, 60, 68, 81];
    let cases: [(&[&str], &[u32]); 4] = [
        (&["build"], &[a_then_b, nested, always]),
        (&["build", "--features", "a"], &[just_a, a_or_b, nested, always, mixed]),
        (&["build", "--features", "b"], &[a_or_b, a_then_b, nested, static_b, always]),
        (&["build", "--features", "a,b"], &[just_a, a_or_b, a_and_b, a_then_b, static_b, always, mixed]),
    ];
    for (command, modules) in cases {
        let (built, said) = cargo_in(command, &dir);
        assert!(!built, "fixtures/configured_gpu.rs: {command:?} built");
        let mut lines: Vec<String> = modules
            .iter()
            .map(|line| {
                format!(
                    "src/lib.rs:{line}:1: error: a module of GPU kernels needs a dependency named `kernite-cuda`, \
                     which this crate does not have"
                )
            })
            .collect();
        lines.push(format!("error: could not compile `configured_gpu` (lib) due to {} previous errors", modules.len()));
        assert_eq!(said, lines, "fixtures/configured_gpu.rs: {command:?}: Cargo said:\n{}", said.join("\n"));
    }
}

/// A module of kernels takes its consts by any path Rust has, and Rust sees them in it, those of
/// the module the path names: the fixture asserts, as it builds, which `STEP` each sees.
#[test]
fn consts_are_taken_by_any_path() {
    builds_quietly("paths");
}

/// The kernels read the module's consts as values known now, and so does Rust: those of a
/// module of consts too, numbers, floats, `bool`s, arrays of them and expressions of these,
/// which Rust sees in the module of kernels that takes them. The fixture asserts, as it builds,
/// that a kernel reads what Rust does.
#[test]
fn the_kernels_read_the_consts_as_values_known_now_and_so_does_rust() {
    builds_quietly("values");
}

#[test]
fn what_a_module_of_consts_hands_on_and_refuses() {
    // Its items in order, each past its doc: its own attribute passes; the const that is not `pub`
    // is refused for that, before kernite works it out; the `pub` one at the call kernite cannot
    // work out; then the function. None is handed on, so the module of kernels keeps a `STEP` of
    // its own and says nothing.
    refuses(
        "consts",
        &[
            "src/lib.rs:10:5: error: a const of a module of consts is `pub`, for the modules of kernels that take it \
             to see",
            "src/lib.rs:14:27: error: a call is a statement: an instruction or a helper, not a value",
            "src/lib.rs:17:5: error: a module of consts holds the consts kernels read, as `pub const E: usize = 16;`, \
             and nothing else",
            "error: could not compile `consts` (lib) due to 3 previous errors",
        ],
    );
}

#[test]
fn a_module_that_takes_consts_has_no_const_function_or_static_of_their_names() {
    // The function is refused at its name. The struct and the type are in Rust's other namespace,
    // which the `use` of the consts does not reach, and no kernel names them.
    refuses(
        "names",
        &[
            "src/lib.rs:20:35: error: `STEP` names one of the consts the module takes from `crate::sizes` as well",
            "error: could not compile `names` (lib) due to 1 previous error",
        ],
    );
}

#[test]
fn kernels_that_compile_say_nothing() {
    builds_quietly("quiet");
}

/// A GPU kernel's tuning directives, `#[maxntid(..)]`, `#[reqntid(..)]`, `#[minnctapersm(..)]` and
/// `#[maxnreg(..)]`, are the kernel's alone: Rust builds the crate without a word, though a const
/// is read by a directive alone, and the tests the fixture holds run each kernel on the GPU over the
/// blocks its directives allow, and over ones they refuse, which the launch says; and a kernel at the
/// edge of the registers a launch gives a block launches over its most threads, and not a step past
/// it. They skip where there is no GPU.
#[test]
fn a_gpu_kernel_runs_over_the_blocks_its_directives_allow() {
    let printed = runs_quietly(&["test"], "tuned");
    assert!(printed.contains("test result: ok. 3 passed"), "fixtures/tuned.rs ran its tests, and printed:\n{printed}");
}

#[test]
fn the_macro_says_what_is_wrong_with_a_directive_where_it_is() {
    // A value is said at the value, and a kernel of instances once, for the first that is wrong. A
    // directive on a function that is no GPU kernel is said at the directive once, though two
    // kernels call the helper, and for the helper nothing calls.
    refuses(
        "directives",
        &[
            "src/lib.rs:8:15: error: 2048 threads are too many: a block has 1024 at most along x",
            "src/lib.rs:11:7: error: these sizes make 1056 threads: a block has 1024 at most in all",
            "src/lib.rs:15:7: error: a kernel has `#[maxntid(..)]` or `#[reqntid(..)]`, not both: its blocks have at \
             most these threads, or exactly these",
            "src/lib.rs:18:7: error: `#[minnctapersm(..)]` needs `#[maxntid(..)]` or `#[reqntid(..)]`: `ptxas` \
             ignores it where it does not know the threads of a block",
            "src/lib.rs:21:15: error: 16 registers are too few: `ptxas` gives a thread 24 at least, and raises a \
             smaller number to it",
            "src/lib.rs:25:15: error: in `tile::<2048>`: 2048 threads are too many: a block has 1024 at most along x",
            "src/lib.rs:31:7: error: `#[maxnreg(..)]` goes on a GPU kernel, `extern \"ptx-kernel\"`; `called` is a \
             function the kernels call, which runs as the kernel that calls it is launched",
            "src/lib.rs:34:7: error: `#[maxnreg(..)]` goes on a GPU kernel, `extern \"ptx-kernel\"`; `helper` is a \
             helper, inlined where it is called",
            "src/lib.rs:39:7: error: `#[maxnreg(..)]` goes on a GPU kernel, `extern \"ptx-kernel\"`; `never_called` \
             is a helper, inlined where it is called",
            "src/lib.rs:53:7: error: `#[maxntid(..)]` goes on a GPU kernel, `extern \"ptx-kernel\"`; `tile` is a \
             kernel for x86-64, which the process runs",
            "src/lib.rs:60:7: error: PTX has no `.maxnctapersm` from 2.1 on; `#[minnctapersm(..)]` gives the blocks \
             an SM should hold at once, at least",
            "src/lib.rs:66:7: error: `#[maxnreg(..)]` gives the most registers a thread may use, as in \
             `#[maxnreg(32)]`",
            "src/lib.rs:73:7: error: a kernel has one `#[maxnreg(..)]`",
            "src/lib.rs:79:15: error: `#[maxntid(..)]` gives whole numbers; this is an f32",
            "error: could not compile `directives` (lib) due to 14 previous errors",
        ],
    );
}

/// A parameter named as a local the macro's Rust binds for itself, `index`, `grid`, `threads` or
/// `shared`, is the kernel's own, and reaches it: Rust builds the module without a word, where the
/// launcher's own `index` hid it, and Rust said "unused variable: `index`". So is a compile-time
/// argument, where no launcher or method binds its name.
#[test]
fn a_kernels_parameters_keep_their_names_beside_those_the_macros_rust_binds() {
    builds_quietly("parameters");
}

/// A parameter named `_` is a number or a pointer the kernel is given and does not read, as it is in any
/// Rust function: in an x86 kernel, plain, of compile-time arguments, cooperative, and cooperative of
/// arguments, and in a GPU kernel of those kinds. Rust builds the crate without a word, and the tests the
/// fixture holds run each kernel, the x86 kernels and the GPU kernels on the GPU: each reads the parameters
/// after the ones named `_` where the ABI puts them. The macros name each such parameter for Rust
/// themselves, `__kernite_0`, and a parameter or a compile-time argument of that name is the kernel's own.
#[test]
fn a_parameter_named_underscore_is_passed_and_not_read_in_a_kernel_of_whatever_kind_it_is() {
    let printed = runs_quietly(&["test"], "underscore");
    assert!(
        printed.contains("test result: ok. 2 passed"),
        "fixtures/underscore.rs ran its tests, and printed:\n{printed}"
    );
}

/// A kernel kernite refuses, with a number and a pointer named `_`, is kernite's error alone, where it
/// is: Rust says nothing at the module's attribute, for the stub the macros leave for the kernel takes
/// them as the kernel does. So it is for an x86 kernel, plain, of compile-time arguments, cooperative and
/// cooperative of arguments, one whose `extern` kernite does not know and one that is not `unsafe`;
/// and for a GPU kernel, plain, of compile-time arguments, cooperative and cooperative of arguments. In a
/// module that gives a number's name another type, Rust says as well, once, at the type of the number
/// named `_`: it is not that number.
#[test]
fn a_kernel_kernite_refuses_with_a_parameter_named_underscore_is_kernites_error_alone() {
    let y = "error: `y` is not defined";
    let abi = "error: kernite compiles `extern \"sysv64\"` kernels, for x86-64, and `extern \"ptx-kernel\"` ones, \
               for NVIDIA GPUs, with the `extern \"ptx-func\"` functions they call; `extern \"C\"` is none of these";
    let safe = "error: a kernel trusts the pointers it is given, so it is an `unsafe fn`";
    let kernite = [
        // The module of x86 kernels: a plain one, one of compile-time arguments, which kernite says
        // once for each instance, a cooperative one and a cooperative one of arguments; one whose
        // `extern` it does not know, and one that is not `unsafe`.
        ("17:13", y),
        ("22:13", "error: in `tiles::<1>`: `y` is not defined"),
        ("22:13", "error: in `tiles::<2>`: `y` is not defined"),
        ("27:13", y),
        ("33:13", "error: in `together_tiles::<1>`: `y` is not defined"),
        ("33:13", "error: in `together_tiles::<2>`: `y` is not defined"),
        ("36:23", abi),
        ("40:28", safe),
        // The module of GPU kernels: the same four kinds.
        ("48:24", y),
        ("53:24", "error: in `tiles::<1>`: `y` is not defined"),
        ("53:24", "error: in `tiles::<2>`: `y` is not defined"),
        ("58:24", y),
        ("64:24", "error: in `together_tiles::<1>`: `y` is not defined"),
        ("64:24", "error: in `together_tiles::<2>`: `y` is not defined"),
        // The same kernels in the modules that take the names of `u64` and `u32` from the crate's `alias`.
        ("73:13", y),
        ("78:13", "error: in `tiles::<1>`: `y` is not defined"),
        ("78:13", "error: in `tiles::<2>`: `y` is not defined"),
        ("83:13", y),
        ("89:13", "error: in `together_tiles::<1>`: `y` is not defined"),
        ("89:13", "error: in `together_tiles::<2>`: `y` is not defined"),
        ("92:23", abi),
        ("96:28", safe),
        ("106:24", y),
        ("111:24", "error: in `tiles::<1>`: `y` is not defined"),
        ("111:24", "error: in `tiles::<2>`: `y` is not defined"),
        ("116:24", y),
        ("122:24", "error: in `together_tiles::<1>`: `y` is not defined"),
        ("122:24", "error: in `together_tiles::<2>`: `y` is not defined"),
    ];
    // Where Rust says it, and of which number, at the type of the parameter named `_`: in each
    // kernel of the modules that rename a number, once, whatever the kernel is.
    let numbers = [
        ("72:44", "u64", "i64"),
        ("77:58", "u64", "i64"),
        ("82:47", "u64", "i64"),
        ("88:67", "u64", "i64"),
        ("92:43", "u64", "i64"),
        ("96:36", "u64", "i64"),
        ("105:48", "u32", "i32"),
        ("110:62", "u32", "i32"),
        ("115:51", "u32", "i32"),
        ("121:71", "u32", "i32"),
    ];
    let mut said: Vec<String> = kernite.map(|(at, what)| format!("src/lib.rs:{at}: {what}")).into();
    said.extend(numbers.map(|(at, number, found)| not_the_number(at, number, found)));
    said.push(format!("error: could not compile `underscore_refused` (lib) due to {} previous errors", said.len()));
    refuses("underscore_refused", &said.iter().map(String::as_str).collect::<Vec<_>>());
}

/// A parameter's type means in every x86 kernel what it means in any function of the module, whether
/// the kernel is called as it is written, of compile-time arguments, or cooperative: `super::W`,
/// `crate::W`, `self::W`, a `W` the module takes from beside it or holds, and `super::super::W`
/// where the module of kernels is nested. A GPU kernel's launcher takes them as before. Rust builds
/// the crate without a word, and the test the fixture holds runs each x86 kernel, which writes
/// through its pointer. So does its doc test, from another crate.
#[test]
fn a_parameters_type_means_what_it_means_where_the_kernel_is_written() {
    let printed = runs_quietly(&["test"], "beside");
    let passed = printed.matches("1 passed").count();
    assert_eq!(passed, 2, "fixtures/beside.rs ran its test and its doc test, and printed:\n{printed}");
}

/// A type in a kernel's parameters or return may name the kernel's compile-time arguments, as in any
/// generic function, in each instance of an x86 kernel: a pointer to an array of the length an
/// argument gives, alone or in braces, a result, a type of the crate's that takes it, with one
/// argument, two and a `bool`, in a kernel that is cooperative, and in one that calls an instance,
/// of any arguments. A GPU kernel's launcher takes it as before. Rust builds the crate without a
/// word, and the tests the fixture holds run each x86 kernel, which writes through its pointer. So
/// does its doc test, from another crate.
#[test]
fn a_type_in_a_kernel_may_name_its_compile_time_arguments() {
    let printed = runs_quietly(&["test"], "argument_pointers");
    for passed in ["test result: ok. 9 passed", "test result: ok. 1 passed"] {
        assert!(
            printed.contains(passed),
            "fixtures/argument_pointers.rs ran its tests and its doc test, and printed:\n{printed}"
        );
    }
}

/// Rust takes a compile-time argument in a type alone, `[u8; N]` or `[u8; { N }]`, and no expression
/// of it, `[u8; N + 1]`, in any generic function. The macro adds no refusal to Rust's: the error is
/// at the type as the kernel writes it, and said once, as is every error in a type of a kernel of
/// instances.
#[test]
fn an_expression_of_a_compile_time_argument_in_a_type_is_rusts_error() {
    refuses(
        "argument_expression",
        &[
            "src/lib.rs:8:68: error: generic parameters may not be used in const operations: cannot perform const \
             operation using `N`",
            "error: could not compile `argument_expression` (lib) due to 1 previous error",
        ],
    );
}

/// A type that is not there, in the parameters or the result of an x86 kernel of instances, is Rust's
/// error, said once, at the type, as it is of any function: not again at the attribute, where the
/// instances' own copies of the type were, nor once for each instance. So is a result that is a pointer
/// to a type with no size, `*mut CStr`.
#[test]
fn an_error_in_a_type_of_a_kernel_of_instances_is_said_once() {
    refuses(
        "type_errors_instances",
        &[
            &not_found("7:69", "Missing"),
            &not_found("12:78", "Lost"),
            &size_unknown("18:86", "[i8]"),
            "error: could not compile `type_errors_instances` (lib) due to 3 previous errors",
        ],
    );
}

/// So is an error in a type of a cooperative x86 kernel, of instances and not: not again at the
/// attribute, where its launcher's instances had it.
#[test]
fn an_error_in_a_type_of_a_cooperative_kernel_is_said_once() {
    refuses(
        "type_errors_cooperative",
        &[
            &not_found("8:68", "Missing"),
            &not_found("13:49", "Absent"),
            "error: could not compile `type_errors_cooperative` (lib) due to 2 previous errors",
        ],
    );
}

/// The kernels that said it once already still do: a plain x86 kernel, in a parameter and in a
/// result, and a GPU kernel of instances.
#[test]
fn an_error_in_a_type_of_any_other_kernel_is_said_once() {
    refuses(
        "type_errors_others",
        &[
            &not_found("6:53", "Gone"),
            &not_found("10:62", "Away"),
            &not_found("19:71", "Elsewhere"),
            "error: could not compile `type_errors_others` (lib) due to 3 previous errors",
        ],
    );
}

/// A lint in a type of a kernel of instances, or of a cooperative kernel, is said once, at the type,
/// at the level the kernel sets: an error where it says `deny` or `forbid`, a warning where it says
/// `warn`. Where it says `allow`, Rust says nothing, as the kernels of
/// `a_type_in_a_kernel_may_name_its_compile_time_arguments` show, and it says nothing where the `allow`
/// is in a `cfg_attr`. Where it says `expect`, Rust says nothing if the lint is in the type, and, if
/// not, that the expectation is unfulfilled, once, at the `expect`.
#[test]
fn a_lint_in_a_type_of_a_kernel_of_instances_is_said_once_at_the_level_the_kernel_sets() {
    let braces = "unnecessary braces around const expression";
    refuses(
        "type_lints",
        &[
            &format!("src/lib.rs:10:71: error: {braces}"),
            &format!("src/lib.rs:17:74: error: {braces}"),
            &format!("src/lib.rs:23:71: warning: {braces}"),
            "src/lib.rs:33:14: warning: this lint expectation is unfulfilled",
            "error: could not compile `type_lints` (lib) due to 2 previous errors; 2 warnings emitted",
        ],
    );
}

/// A kernel takes a pointer to any type, as it takes one to a number: to an array, a pointer, a
/// tuple, a function's pointer, `fn(u32) -> u32`, or a type of the crate's with two arguments,
/// `Pair<u8, 2>`, whose comma ends no parameter, even when an argument is a function's pointer,
/// `Pair<fn() -> u8, 2>`, whose arrow closes no `<`; on a GPU, to an array as long as a
/// compile-time argument says, and to `f16` and `bf16`. Each kernel's launcher or call takes the
/// pointee as written, but for those two, which a launcher names as kernite-cuda does, and Rust
/// builds the module without a word.
#[test]
fn a_kernel_takes_a_pointer_to_any_type() {
    builds_quietly("pointers");
}

/// A lifetime has no place in a kernel, and the macro hands one on as punctuation, where the lexer
/// of a string refuses it. A pointer to a type that names one, `*const [&'static str]` or
/// `*const (dyn core::fmt::Debug + 'static)`, which Rust passes in two registers, is refused at its
/// type in the lexer's words, not taken for one register, with the next parameter read from the
/// wrong one.
#[test]
fn a_pointer_to_a_type_with_a_lifetime_is_refused_at_its_type() {
    refuses(
        "lifetimes",
        &[
            "src/lib.rs:8:44: error: lifetimes and labels have no place in a kernel",
            "src/lib.rs:12:45: error: lifetimes and labels have no place in a kernel",
            "error: could not compile `lifetimes` (lib) due to 2 previous errors",
        ],
    );
}

/// A pointer to a type with no size, which Rust passes in two registers where a kernel's is one, is
/// one error, at the type, whatever the type is called. kernite tells `[f32]` by its text, and says so
/// in its words, on x86 and on a GPU. Rust tells the others, and says that the size cannot be known:
/// `CStr`, `str` by its path, a name for `[f32]`, a struct of the crate's whose last field is one, and
/// a tuple that ends in `CStr`. It does so in a kernel of every kind: an x86 kernel, plain, of
/// compile-time arguments, cooperative, and cooperative of compile-time arguments, and a GPU kernel of
/// each of these kinds; for two such pointers in a kernel, each, and for a pointer a kernel returns,
/// plain or of compile-time arguments. A kernel kernite refuses for another reason leaves a stub that
/// asks Rust as well, so that it is two errors, kernite's and Rust's, each where it is: an x86 kernel,
/// plain, of compile-time arguments or cooperative, and one that returns a pointer; one that is not
/// `unsafe`; one whose `extern` kernite does not know; and a GPU kernel, plain or cooperative. An
/// array or a tuple that holds a type with no size Rust refuses by itself, at the type, and says it
/// again of what kernite asks of it: Cargo prints the two as one.
#[test]
fn a_pointer_to_a_type_with_no_size_is_refused_at_its_type() {
    // What kernite says, in the order of the kernels: the x86 module's, then the GPU's.
    let told = "error: a pointer to `[f32]` has two words, an address and a length or a table, \
                and a kernel's pointer has one";
    let y = "error: `y` is not defined";
    let safe = "error: a kernel trusts the pointers it is given, so it is an `unsafe fn`";
    let abi = "error: kernite compiles `extern \"sysv64\"` kernels, for x86-64, and `extern \"ptx-kernel\"` ones, \
               for NVIDIA GPUs, with the `extern \"ptx-func\"` functions they call; `extern \"C\"` is none of these";
    let kernite = [
        ("83:44", told),
        ("109:23", y),
        ("114:23", "error: in `refused_tiles::<1>`: `y` is not defined"),
        ("114:23", "error: in `refused_tiles::<2>`: `y` is not defined"),
        ("119:23", y),
        ("123:23", y),
        ("128:23", "error: in `refused_returns_tiles::<1>`: `y` is not defined"),
        ("128:23", "error: in `refused_returns_tiles::<2>`: `y` is not defined"),
        ("131:28", safe),
        ("135:23", abi),
        ("150:48", told),
        ("175:28", y),
        ("180:28", y),
    ];
    // Where Rust says it, and of which type: at the pointer's type, in the order Rust says it.
    let sizes = [
        // An array of `CStr`, and a tuple with `CStr` before its last field: Rust refuses the types.
        ("139:60", "[i8]"),
        ("143:63", "[i8]"),
        // `CStr`: a plain kernel, one of compile-time arguments, a cooperative one.
        ("19:51", "[i8]"),
        ("24:71", "[i8]"),
        ("30:60", "[i8]"),
        // `str` by its path.
        ("34:54", "str"),
        ("39:74", "str"),
        ("45:63", "str"),
        // `Floats`, a name for `[f32]`.
        ("49:52", "[f32]"),
        ("54:72", "[f32]"),
        ("60:61", "[f32]"),
        // `Tail`, a struct whose last field is `[f32]`.
        ("64:50", "[f32]"),
        ("69:70", "[f32]"),
        ("75:59", "[f32]"),
        // Two pointers of one kernel, and a pointer a kernel returns, plain and of arguments.
        ("79:49", "[i8]"),
        ("79:74", "[f32]"),
        ("87:67", "[i8]"),
        ("92:87", "[i8]"),
        // A tuple that ends in `CStr`: a plain kernel, and a cooperative one of arguments.
        ("97:51", "[i8]"),
        ("103:80", "[i8]"),
        // An x86 kernel kernite refuses: a plain one, one of arguments, a cooperative one, a plain one
        // and one of arguments that return a pointer, one that is not `unsafe`, and one whose
        // `extern` kernite does not know.
        ("108:53", "[i8]"),
        ("113:73", "[i8]"),
        ("118:62", "[i8]"),
        ("122:75", "[i8]"),
        ("127:93", "[f32]"),
        ("131:43", "[i8]"),
        ("135:48", "[i8]"),
        // A GPU kernel: a plain one, one of compile-time arguments, a cooperative one, and a
        // cooperative one of arguments.
        ("154:53", "[i8]"),
        ("159:72", "[f32]"),
        ("164:62", "[i8]"),
        ("170:81", "[f32]"),
        // A GPU kernel kernite refuses: a plain one, and a cooperative one.
        ("174:55", "[i8]"),
        ("179:64", "[i8]"),
    ];
    let mut said: Vec<String> = kernite.map(|(at, what)| format!("src/lib.rs:{at}: {what}")).into();
    said.extend(sizes.map(|(at, ty)| size_unknown(at, ty)));
    // The array and the tuple Rust refuses are two errors more than the lines say.
    let errors = kernite.len() + sizes.len() + 2;
    said.push(format!("error: could not compile `no_size` (lib) due to {errors} previous errors"));
    refuses("no_size", &said.iter().map(String::as_str).collect::<Vec<_>>());
}

/// A pointer to a type with no size is Rust's error at the type, whatever the type holds and whatever
/// it is called, for the macros ask Rust of each pointer, and not of a clause on its type. Two cases
/// show it: a type with a `&` in it, `(&u8, CStr)` or `GTail<[&u8]>`, which a `where` clause cannot
/// hold; and `u8` in a module that takes it from the crate, where it is a name for `[u8]` and kernite
/// reads a number. Each is one error, at the type, in a kernel, plain, of compile-time arguments or
/// cooperative, on x86 and on a GPU.
#[test]
fn a_pointer_to_a_type_with_no_size_is_refused_whatever_it_holds_or_is_called() {
    let sizes = [
        // `(&u8, CStr)`, and `GTail<[&u8]>`: an x86 kernel, plain, of compile-time arguments,
        // cooperative.
        ("20:51", "[i8]"),
        ("25:71", "[i8]"),
        ("31:60", "[i8]"),
        ("35:50", "[&u8]"),
        ("40:70", "[&u8]"),
        ("46:59", "[&u8]"),
        // `u8`, which the module takes from the crate.
        ("55:51", "[u8]"),
        ("60:65", "[u8]"),
        ("66:54", "[u8]"),
        // The same, in a GPU kernel.
        ("73:53", "[i8]"),
        ("78:73", "[i8]"),
        ("83:62", "[i8]"),
        ("87:52", "[&u8]"),
        ("92:72", "[&u8]"),
        ("97:61", "[&u8]"),
        // `u8`, in a GPU kernel too: the module's own `u8` is a name for a slice, and the Rust the
        // macros write names the primitives by their paths.
        ("106:53", "[u8]"),
        ("111:67", "[u8]"),
        ("116:56", "[u8]"),
    ];
    let mut said: Vec<String> = sizes.map(|(at, ty)| size_unknown(at, ty)).into();
    said.push(format!("error: could not compile `no_size_names` (lib) due to {} previous errors", sizes.len()));
    refuses("no_size_names", &said.iter().map(String::as_str).collect::<Vec<_>>());
}

/// Asking Rust whether each pointer has a size costs a crate no word, under `#![deny(warnings)]`: of
/// a pointer to a struct with a lifetime that is written without one, `*const Bar`, which a parameter
/// takes, to `Option<&u8>`, to a tuple with a reference; of a pointer a kernel returns; of a kernel
/// that never returns; of a parameter and a kernel named as what the macros write; of a kernel
/// deprecated, by `#[deprecated]` or by `cfg_attr`, or configured out, by `#[cfg]` or by `cfg_attr`.
/// So it does in an x86 kernel, plain, of compile-time arguments, cooperative, and cooperative of
/// compile-time arguments; and in a GPU kernel, plain, of compile-time arguments and cooperative.
#[test]
fn a_kernel_with_a_pointer_builds_without_a_word_of_whatever_kind_it_is() {
    builds_quietly("pointer_checks");
}

/// A generic is a value, so one that is a type is refused where it is: `<T>`, at the `T`, and `impl` in a
/// parameter's type or the result's, wherever in it, at the first `impl`: in a parameter it makes the function
/// generic over a type, and in the result it hides one. So it is by value, in a pointer's type and in a type
/// that holds one, in a parameter and in the result, for each kind of kernel, and for `impl ?Sized`, a pointer
/// to which Rust passes in two words. The macro writes nothing else for the module, so Rust says nothing of its
/// own, nor of a call to a kernel, and a kernel of instances is said once.
#[test]
fn a_generic_that_is_a_type_is_refused_at_its_token() {
    let generics =
        |place: &str| format!("src/lib.rs:{place}: error: a function's generics are values, as in `<const N: usize>`");
    refuses(
        "type_generics",
        &[
            &generics("8:37"),
            &generics("15:47"),
            &generics("23:63"),
            &generics("31:47"),
            &generics("40:63"),
            &generics("47:49"),
            &generics("57:63"),
            &generics("66:48"),
            &generics("73:57"),
            &generics("80:54"),
            &generics("87:48"),
            &generics("94:52"),
            &generics("101:53"),
            &generics("109:69"),
            &generics("116:50"),
            &generics("123:57"),
            &generics("130:54"),
            "error: could not compile `type_generics` (lib) due to 17 previous errors",
        ],
    );
}

/// An `impl` makes a function generic in its parameters and its result alone. In the type of a variable, a
/// const or a static it is said as it was: the x86 target takes no such variable, Rust takes no `impl` in a
/// const's type, at the const and again at a kernel that reads it, and the PTX target takes no such memory.
#[test]
fn impl_in_the_type_of_a_variable_a_const_or_a_static_is_said_as_it_was() {
    refuses(
        "type_impls",
        &[
            "src/lib.rs:6:16: error: x86 variables are `gpr`, `ymm`, `xmm` and stack memory, \
             `local<..>`; not `impl Copy`",
            "src/lib.rs:22:30: error: this is an integer, not impl Copy",
            "src/lib.rs:31:15: error: memory holds `b8` to `b128`, `u8` to `u64`, `s8` to `s64`, `f16`, `f16x2`, \
             `bf16`, `bf16x2`, `tf32`, `f32` and `f64`; not `impl Copy`",
            "src/lib.rs:13:18: error[E0562]: `impl Trait` is not allowed in const types",
            "src/lib.rs:22:18: error[E0562]: `impl Trait` is not allowed in const types",
            "error: could not compile `type_impls` (lib) due to 5 previous errors",
        ],
    );
}

/// A number's name that the module gives another type is Rust's error at the type, once: kernite reads
/// the number the name says, and the macros ask Rust whether the type is that number. The names are
/// given by a `use` of a `pub type` of the crate's root, and of a module of aliases, to the type of
/// another number; in each parameter and result of an x86 kernel, plain, of compile-time arguments
/// or cooperative, and in each parameter of a GPU kernel of those kinds.
#[test]
fn a_number_named_as_another_type_is_refused_at_the_type_whatever_the_kernel_is() {
    let numbers = [
        // `u64` and `f64`, which the crate's root gives and a module takes by a `use`: `id` takes
        // a `u64`, and `twice` takes and gives back an `f64`.
        ("35:41", "u64", "u32"),
        ("39:44", "f64", "f32"),
        ("39:52", "f64", "f32"),
        // An x86 kernel, plain: a parameter of each number it takes.
        ("50:12", "f32", "f64"),
        ("51:12", "f64", "f32"),
        ("52:12", "usize", "isize"),
        ("53:12", "isize", "usize"),
        ("54:12", "u64", "i64"),
        ("55:12", "i64", "u64"),
        // The same, a result of each number it gives back, and `bool`.
        ("60:58", "f32", "f64"),
        ("61:58", "f64", "f32"),
        ("62:60", "usize", "isize"),
        ("63:60", "isize", "usize"),
        ("64:58", "u64", "i64"),
        ("65:58", "i64", "u64"),
        ("66:58", "u32", "i32"),
        ("67:58", "i32", "u32"),
        ("68:58", "u16", "i16"),
        ("69:58", "i16", "u16"),
        ("70:57", "u8", "i8"),
        ("71:57", "i8", "u8"),
        ("72:59", "bool", "u8"),
        // An x86 kernel of compile-time arguments: a parameter of each number.
        ("81:12", "f32", "f64"),
        ("82:12", "f64", "f32"),
        ("83:12", "usize", "isize"),
        ("84:12", "isize", "usize"),
        ("85:12", "u64", "i64"),
        ("86:12", "i64", "u64"),
        // A result of three of them.
        ("92:74", "f32", "f64"),
        ("94:73", "u8", "i8"),
        ("96:75", "bool", "u8"),
        // A cooperative x86 kernel, plain and of compile-time arguments.
        ("105:12", "f32", "f64"),
        ("106:12", "f64", "f32"),
        ("107:12", "usize", "isize"),
        ("108:12", "isize", "usize"),
        ("109:12", "u64", "i64"),
        ("110:12", "i64", "u64"),
        ("117:67", "f32", "f64"),
        ("117:75", "u64", "i64"),
        // A GPU kernel, plain: a parameter of each number it takes.
        ("127:12", "u8", "i8"),
        ("128:12", "u16", "i16"),
        ("129:12", "u32", "i32"),
        ("130:12", "u64", "i64"),
        ("131:12", "i8", "u8"),
        ("132:12", "i16", "u16"),
        ("133:12", "i32", "u32"),
        ("134:12", "i64", "u64"),
        ("135:12", "usize", "isize"),
        ("136:12", "isize", "usize"),
        ("137:12", "f32", "f64"),
        ("138:12", "f64", "f32"),
        // A GPU kernel of compile-time arguments, a cooperative one, and a cooperative one of arguments.
        ("143:69", "u8", "i8"),
        ("143:76", "f64", "f32"),
        ("147:58", "u32", "i32"),
        ("147:66", "f32", "f64"),
        ("152:78", "i64", "u64"),
        // A name for a type wider than the number: Rust would pass eight bytes where the kernel reads four.
        ("166:78", "u32", "u64"),
    ];
    let mut said: Vec<String> = numbers.map(|(at, number, found)| not_the_number(at, number, found)).into();
    said.push(format!("error: could not compile `number_names` (lib) due to {} previous errors", numbers.len()));
    refuses("number_names", &said.iter().map(String::as_str).collect::<Vec<_>>());
}

/// The same in kernels kernite refuses: kernite says what is wrong where it is, in the kernel, and
/// Rust says what it finds at the type, in the stub the macros leave for each, once.
#[test]
fn a_number_named_as_another_type_is_refused_at_the_type_in_a_kernel_kernite_refuses() {
    let kernite = [
        // An x86 kernel kernite refuses for a body that names a variable it has not: a plain one, one
        // of compile-time arguments, which it says once for each instance, a cooperative one, and a
        // cooperative one of arguments.
        "src/lib.rs:23:13: error: `y` is not defined",
        "src/lib.rs:27:13: error: `y` is not defined",
        "src/lib.rs:32:13: error: in `tiles::<1>`: `y` is not defined",
        "src/lib.rs:32:13: error: in `tiles::<2>`: `y` is not defined",
        "src/lib.rs:37:13: error: `y` is not defined",
        "src/lib.rs:43:13: error: in `together_tiles::<1>`: `y` is not defined",
        "src/lib.rs:43:13: error: in `together_tiles::<2>`: `y` is not defined",
        // One whose `extern` it does not know, and one that is not `unsafe`.
        "src/lib.rs:46:23: error: kernite compiles `extern \"sysv64\"` kernels, for x86-64, and `extern \
         \"ptx-kernel\"` ones, for NVIDIA GPUs, with the `extern \"ptx-func\"` functions they call; `extern \"C\"` \
         is none of these",
        "src/lib.rs:50:28: error: a kernel trusts the pointers it is given, so it is an `unsafe fn`",
        // A GPU kernel: a plain one, one of compile-time arguments, and a cooperative one.
        "src/lib.rs:60:24: error: `y` is not defined",
        "src/lib.rs:65:24: error: in `tiles::<1>`: `y` is not defined",
        "src/lib.rs:65:24: error: in `tiles::<2>`: `y` is not defined",
        "src/lib.rs:70:24: error: `y` is not defined",
    ];
    let numbers = [
        // An x86 kernel kernite refuses, plain: the parameters and the result, and a `bool`.
        ("22:44", "f32", "f64"),
        ("22:52", "usize", "isize"),
        ("22:62", "u64", "i64"),
        ("22:70", "f64", "f32"),
        ("26:49", "u64", "i64"),
        ("26:57", "bool", "u8"),
        // One of compile-time arguments, said once and not for each instance.
        ("31:60", "f32", "f64"),
        ("31:68", "u64", "i64"),
        ("31:76", "usize", "isize"),
        // A cooperative one, and a cooperative one of arguments.
        ("36:47", "f32", "f64"),
        ("36:55", "u64", "i64"),
        ("42:69", "f64", "f32"),
        // One whose `extern` kernite does not know, and one that is not `unsafe`.
        ("46:43", "u64", "i64"),
        ("46:51", "f64", "f32"),
        ("50:36", "f32", "f64"),
        ("50:44", "f32", "f64"),
        // A GPU kernel kernite refuses: a plain one, one of compile-time arguments and a cooperative one.
        ("59:48", "u32", "i32"),
        ("59:56", "f64", "f32"),
        ("59:64", "u8", "i8"),
        ("64:62", "u64", "i64"),
        ("69:51", "f32", "f64"),
    ];
    let mut said: Vec<String> = kernite.map(String::from).into();
    said.extend(numbers.map(|(at, number, found)| not_the_number(at, number, found)));
    said.push(format!("error: could not compile `number_names_refused` (lib) due to {} previous errors", said.len()));
    refuses("number_names_refused", &said.iter().map(String::as_str).collect::<Vec<_>>());
}

/// A number's name that the module gives a type with no size, `u8` a slice, is Rust's error at the
/// type: that it is not the number, and that its size cannot be known. It is Rust's at the type alone.
/// The code the macros write for the kernel itself, an instance, a dispatcher or a launcher, takes the
/// number and not the name, so it makes no value of the type, which Rust would say has no size at the
/// module's attribute, where nothing can be mended. So it is in an x86 kernel with the number as its
/// result or its parameter, and in a GPU kernel with it as a parameter: plain, of compile-time
/// arguments, cooperative, cooperative of compile-time arguments, and one kernite refuses. A type with
/// a size that is not `Copy`, `String`, is only that it is not the number, in a cooperative x86 kernel
/// and in a GPU one.
#[test]
fn a_number_named_as_a_type_with_no_size_is_refused_at_the_type_and_not_at_the_attribute() {
    let mut said: Vec<String> =
        ["src/lib.rs:33:13: error: `y` is not defined", "src/lib.rs:60:24: error: `y` is not defined"]
            .map(String::from)
            .into();
    // Rust says of a type with no size that it is not the number, then that its size cannot be known.
    let slice = |at: &str, number: &str| {
        let found = format!("[{number}]");
        [not_the_number(at, number, &found), size_unknown(at, &found)]
    };
    // An x86 kernel with the number as its result: plain, of compile-time arguments and refused.
    said.extend(slice("23:52", "u8"));
    said.extend(slice("28:66", "u8"));
    said.extend(slice("32:54", "u8"));
    // An x86 kernel with the number as its parameter.
    said.extend(slice("36:48", "usize"));
    // A GPU kernel with the number as its parameter: plain, of compile-time arguments, cooperative and
    // refused.
    said.extend(slice("45:48", "u8"));
    said.extend(slice("50:62", "u8"));
    said.extend(slice("55:51", "u8"));
    said.extend(slice("59:50", "u8"));
    // An x86 kernel with the number as its parameter: of compile-time arguments, cooperative, cooperative
    // of compile-time arguments, and a cooperative one whose number is a `String`.
    said.extend(slice("76:58", "usize"));
    said.extend(slice("81:47", "usize"));
    said.extend(slice("87:67", "usize"));
    said.push(not_the_number("92:44", "f64", "String"));
    // A GPU kernel with the number as its parameter: cooperative of compile-time arguments, and one whose
    // number is a `String`.
    said.extend(slice("104:71", "u8"));
    said.push(not_the_number("108:48", "u16", "String"));
    // Each line is said once, so the count is the lines' own.
    said.push(format!("error: could not compile `number_names_unsized` (lib) due to {} previous errors", said.len()));
    refuses("number_names_unsized", &said.iter().map(String::as_str).collect::<Vec<_>>());
}

/// An x86 kernel takes and gives back no `f16`, which only a GPU has: kernite refuses it at the type, once
/// for each instance. Rust says no more. It does not say that its `f16` is unstable, as it says wherever
/// the type is named, though the macros leave a stub for the kernel. Nor does it say, at a call of the
/// kernel, that the `f16` the kernel gives back is not the number the caller reads. So it is in a parameter
/// and in a result, of a kernel plain, of compile-time arguments, cooperative, and cooperative of
/// compile-time arguments, and at a call of each. So it is of a `bf16`, which Rust has not.
#[test]
fn an_x86_kernel_that_takes_or_gives_back_a_half_is_kernites_error_alone() {
    let takes = |place: &str, kernel: &str, ty: &str| {
        format!("src/lib.rs:{place}: error: {kernel}a kernel takes pointers, 64-bit integers and floats, not `{ty}`")
    };
    let gives = |place: &str, kernel: &str, ty: &str| {
        format!(
            "src/lib.rs:{place}: error: {kernel}a kernel returns a float, an integer or a pointer, or never, `!`; \
             not `{ty}`"
        )
    };
    let instance = |name: &str, n: u64| format!("in `{name}::<{n}>`: ");
    let mut said = vec![];
    for (ty, [plain, tiles, together, together_tiles, result, tiles_result]) in [
        ("f16", ["8:44", "13:58", "18:47", "24:67", "28:53", "33:73"]),
        ("bf16", ["40:44", "45:58", "50:47", "56:67", "60:53", "65:73"]),
    ] {
        said.push(takes(plain, "", ty));
        said.extend([1, 2].map(|n| takes(tiles, &instance("tiles", n), ty)));
        said.push(takes(together, "", ty));
        said.extend([1, 2].map(|n| takes(together_tiles, &instance("together_tiles", n), ty)));
        said.push(gives(result, "", ty));
        said.extend([1, 2].map(|n| gives(tiles_result, &instance("tiles_result", n), ty)));
    }
    said.push(format!("error: could not compile `halves_refused` (lib) due to {} previous errors", said.len()));
    refuses("halves_refused", &said.iter().map(String::as_str).collect::<Vec<_>>());
}

/// A pointer to an `f16` is a pointer, which an x86 kernel takes whatever it points to: kernite says
/// nothing of it, and Rust's word that its `f16` is unstable is the only one, said once, at the type, in
/// a kernel of every kind, in a parameter and in a result.
#[test]
fn a_pointer_to_an_f16_in_an_x86_kernel_is_rusts_error_alone() {
    let unstable = |place: &str| format!("src/lib.rs:{place}: error[E0658]: the type `f16` is unstable");
    refuses(
        "halves_pointers",
        &[
            &unstable("7:49"),
            &unstable("12:63"),
            &unstable("17:52"),
            &unstable("23:72"),
            &unstable("27:58"),
            &unstable("32:78"),
            "error: could not compile `halves_pointers` (lib) due to 6 previous errors",
        ],
    );
}

/// A GPU kernel takes an `f16` and a `bf16`, by value and by pointer: Rust builds the crate without a word,
/// the launcher takes kernite-cuda's two, and the test the fixture holds runs the kernel and reads back
/// what it wrote. It skips where there is no GPU.
#[test]
fn a_gpu_kernel_takes_the_halves_as_kernite_cuda_has_them() {
    let printed = runs_quietly(&["test"], "halves_gpu");
    assert!(
        printed.contains("test result: ok. 1 passed"),
        "fixtures/halves_gpu.rs ran its tests, and printed:\n{printed}"
    );
}

/// Asking Rust whether each number is that number costs a crate no word, under `#![deny(warnings)]`:
/// a parameter of each number, a result of each, of a kernel that never returns, and of parameters
/// named as the locals the macros write; of a kernel deprecated, by `#[deprecated]` or by `cfg_attr`,
/// or configured out, by `#[cfg]` or by `cfg_attr`. So it does in an x86 kernel, plain, of
/// compile-time arguments, cooperative, and cooperative of compile-time arguments; and in a GPU
/// kernel, plain, of compile-time arguments and cooperative.
#[test]
fn a_kernel_with_numbers_builds_without_a_word_of_whatever_kind_it_is() {
    builds_quietly("number_checks");
}

/// The closure that the launcher of a cooperative x86 kernel runs is the launcher's first, however many
/// numbers the kernel takes: Rust numbers the closures of a body in the order they come, and the names
/// of the symbols hold the numbers, so a question among the launcher's statements would change the
/// names in the code of a release build. The questions are in an item of their own. Needs `nm`, which
/// says the symbols of the library, and skips without it.
#[test]
fn the_launcher_of_a_cooperative_kernel_that_takes_numbers_runs_its_first_closure() {
    builds_quietly("closure_names");
    let library = Path::new(ROOT).join("target/debug/libclosure_names.rlib");
    let Ok(out) = Command::new("nm").arg("--demangle").arg(library).output() else {
        eprintln!("no nm here: skipped");
        return;
    };
    let symbols = String::from_utf8_lossy(&out.stdout);
    let ours: Vec<&str> = symbols.lines().filter(|s| s.contains("closure_names::")).collect();
    assert!(
        ours.iter().any(|s| s.ends_with(" T closure_names::cpu::together::{closure#0}")),
        "fixtures/closure_names.rs: `together` runs no first closure, and its library has:\n{}",
        ours.join("\n")
    );
}

/// A module that takes the name of `u8`, `u32`, `str` or any number from elsewhere builds as one that
/// does not: the Rust the macros write names the primitives by their paths, `::core::primitive::u8`.
#[test]
fn a_module_that_gives_a_primitive_another_type_builds_as_one_that_does_not() {
    builds_quietly("primitive_names");
}

/// A compile-time argument is no local, and no span of stable Rust keeps it apart from one of its
/// name: `grid`, the launcher's, and `threads` and `shared`, of a cooperative kernel's `_blocks`.
/// Each is refused at its name, and nothing at the module's attribute, where Rust would say
/// "expected `Grid`, found `u32`", as the launcher or method never runs.
#[test]
fn a_compile_time_argument_named_as_the_launchers_parameters_is_refused_at_its_name() {
    refuses(
        "arguments",
        &[
            "src/lib.rs:27:47: error: `grid` names a parameter of the launcher as well",
            "src/lib.rs:38:47: error: `threads` names a parameter of `k_blocks` as well",
            "src/lib.rs:49:47: error: `shared` names a parameter of `k_blocks` as well",
            "src/lib.rs:60:43: error: `grid` names a parameter of the launcher as well",
            "error: could not compile `arguments` (lib) due to 4 previous errors",
        ],
    );
}

/// A crate may forbid each lint the macro used to `allow`, and builds its kernels without a word.
#[test]
fn a_crate_that_forbids_lints_builds_its_kernels_without_a_word() {
    builds_quietly("forbidden");
}

/// The same crate is told of a kernel that does not compile where it does not, and of nothing
/// else: its stub and its launcher, which never run, use what they take, and ask Rust the size of
/// each pointer they take or give back.
#[test]
fn a_crate_that_forbids_lints_is_told_of_a_kernel_that_does_not_compile_and_of_nothing_else() {
    // The modules given consts come last, and the second first: Rust expands them a step later.
    refuses(
        "forbidden_stubs",
        &[
            "src/lib.rs:23:13: error: `y` is not defined",
            "src/lib.rs:28:13: error: in `tile::<1>`: `y` is not defined",
            "src/lib.rs:28:13: error: in `tile::<2>`: `y` is not defined",
            "src/lib.rs:33:18: error: `y` is not defined",
            "src/lib.rs:40:28: error: `y` is not defined",
            "src/lib.rs:45:28: error: `y` is not defined",
            "src/lib.rs:76:18: error: `y` is not defined",
            "src/lib.rs:81:18: error: in `tile::<1>`: `y` is not defined",
            "src/lib.rs:81:18: error: in `tile::<2>`: `y` is not defined",
            "src/lib.rs:64:28: error: `y` is not defined",
            "src/lib.rs:69:28: error: `y` is not defined",
            "src/lib.rs:52:13: error: `y` is not defined",
            "src/lib.rs:57:18: error: `y` is not defined",
            "error: could not compile `forbidden_stubs` (lib) due to 13 previous errors",
        ],
    );
}

/// The launcher of a cooperative kernel keeps the kernel's `expect` as written, so that one nothing
/// meets is refused once, where the kernel has it: `k_blocks` keeps no lint.
#[test]
fn the_launcher_keeps_a_kernels_expect_as_written() {
    refuses(
        "unmet",
        &[
            "src/lib.rs:10:14: error: this lint expectation is unfulfilled",
            "error: could not compile `unmet` (lib) due to 1 previous error",
        ],
    );
}

/// A crate that forbids a lint a kernel expects is refused once, at the `expect`, of a cooperative GPU
/// kernel, an x86 kernel of instances and a cooperative one: the launcher, or the function that picks
/// an instance, keeps the `expect` as written, and `k_blocks` and the instances keep no lint.
#[test]
fn a_crate_that_forbids_a_lint_a_kernel_expects_is_refused_once_at_the_expect() {
    let refused =
        "error[E0453]: expect(non_snake_case) incompatible with previous forbid: overruled by previous forbid";
    refuses(
        "forbidden_expect",
        &[
            &format!("src/lib.rs:10:14: {refused}"),
            &format!("src/lib.rs:18:14: {refused}"),
            &format!("src/lib.rs:24:14: {refused}"),
            "error: could not compile `forbidden_expect` (lib) due to 3 previous errors",
        ],
    );
}

/// Clippy says "too many arguments" of a function of more than seven parameters wherever its tokens
/// come from, and a launcher takes `self` and the grid as well as the kernel's parameters: so each
/// launcher allows it, and Clippy says nothing of one of eight or nine, of each kind of kernel. It
/// says nothing of the kernels that forbid lints either, whose instances include `bool`s, nor of those
/// that take or give back pointers, of every kind, which the macros ask Rust the size of, nor of those
/// that take parameters named `_`, which the macros name for Rust themselves.
#[test]
fn clippy_says_nothing_of_the_rust_the_macro_writes() {
    if clippy_here() {
        for name in
            ["many", "forbidden", "pointers", "pointer_checks", "number_checks", "primitive_names", "underscore"]
        {
            let (done, said) = cargo(&["clippy"], name);
            assert!(done && said.is_empty(), "fixtures/{name}.rs: Clippy {done}, and said:\n{}", said.join("\n"));
        }
    }
}

/// That lint is the one a crate of kernels may not forbid: each launcher allows it, of however many
/// parameters, so Clippy refuses each module that has one, at its attribute.
#[test]
fn a_crate_of_kernels_may_not_forbid_clippys_too_many_arguments() {
    if clippy_here() {
        let (done, said) = cargo(&["clippy"], "many_forbidden");
        let allowed =
            "allow(clippy::too_many_arguments) incompatible with previous forbid: overruled by previous forbid";
        let lines = [
            format!("src/lib.rs:6:1: error[E0453]: {allowed}"),
            format!("src/lib.rs:15:1: error[E0453]: {allowed}"),
            "error: could not compile `many_forbidden` (lib) due to 2 previous errors".to_string(),
        ];
        assert!(!done && said == lines, "fixtures/many_forbidden.rs: Clippy said:\n{}", said.join("\n"));
    }
}

/// A name written raw, `r#type`, reaches Rust as written: a parameter's and a compile-time
/// argument's, in each kind of kernel that passes them on, a kernel's, a function's that a kernel
/// calls, a static's and a const's.
#[test]
fn names_written_raw_reach_rust_as_written() {
    builds_quietly("raw");
}

/// The method that says how many blocks a cooperative GPU kernel's grid has at most, `k_blocks`,
/// keeps what the kernel says of lints by giving Rust nothing of the kernel's to lint: its name and
/// the compile-time arguments it takes are of the macro's span. Of the kernel's attributes it keeps
/// those of whether it is there, and nothing else. A name written raw, `r#doc`, is the name. Rust
/// builds each case without a word: a lint allowed, warned of or expected, of a compile-time
/// argument named in lower case or of a parameter, alone, after docs, in a `cfg_attr(..)` or in
/// two; a kernel left out, beside a method of the crate's own named `k_blocks`; one deprecated,
/// whose `k_blocks` a function calls and expects a warning of; a kernel that says what its code
/// needs, or its symbol; and one found in the docs by the name `k_blocks`. Where the kernel has
/// docs, plain or in a `cfg_attr(..)`, `k_blocks` has its own alone, and where the kernel is hidden
/// from the docs, so is `k_blocks`: whatever else the kernel's `doc(..)` says, in a `cfg_attr(..)`
/// or in two, or written by a macro. Where the kernel denies or forbids a lint, or warns of one its
/// module denies, Rust refuses the build, and says each lint of `n` once, at the launcher.
#[test]
fn a_cooperative_kernels_blocks_method_keeps_what_it_says_of_lints() {
    builds_quietly("blocks");
    let docs = documents_quietly("blocks");
    let page = |module: &str| fs::read_to_string(docs.join(module).join("struct.Kernels.html")).unwrap();
    // The docblock after `k_blocks`' anchor, as rustdoc writes it.
    let own = |module: &str| -> Vec<String> {
        let page = page(module);
        let (_, method) =
            page.split_once("id=\"method.k_blocks\"").unwrap_or_else(|| panic!("{module} has no `k_blocks`"));
        let (_, block) = method.split_once("<div class=\"docblock\">").expect("`k_blocks` has docs");
        block.split_once("</div>").expect("the docs end").0.lines().map(String::from).collect()
    };
    for module in ["listed", "conditional_docs_and_lint", "raw_names"] {
        let want = [
            format!(
                "<p>How many blocks a grid of <a href=\"struct.Kernels.html#method.k\" title=\"method \
                 blocks::{module}::Kernels::k\"><code>k</code></a> has at most, of <code>threads</code> threads each and"
            ),
            "<code>shared</code> bytes of the launch’s shared memory: as many as the GPU runs at once, for it".into(),
            "runs them all together.</p>".into(),
        ];
        assert_eq!(own(module), want, "{module}");
    }
    for module in ["hidden", "raw_hidden", "hidden_aliased", "conditional", "nested", "written", "written_in_cfg_attr"]
    {
        assert!(!page(module).contains("k_blocks"), "the docs of `{module}`'s `Kernels` show `k_blocks`");
    }
    refuses(
        "denied",
        &[
            "src/lib.rs:13:47: error: const parameter `n` should have an upper case name",
            "src/lib.rs:24:47: error: const parameter `n` should have an upper case name",
            "src/lib.rs:39:51: warning: const parameter `n` should have an upper case name",
            "error: could not compile `denied` (lib) due to 2 previous errors; 1 warning emitted",
        ],
    );
}

/// An error in an attribute that `k_blocks` keeps is said once, where the kernel has it, and not
/// again where the macro is.
#[test]
fn an_error_in_an_attribute_k_blocks_keeps_is_said_once() {
    refuses(
        "malformed",
        &[
            "src/lib.rs:32:7: error[E0805]: malformed `cfg_attr` attribute input",
            "src/lib.rs:12:5: error: malformed `expect` attribute input",
            "src/lib.rs:22:23: error: malformed `expect` attribute input",
            "error: could not compile `malformed` (lib) due to 3 previous errors",
        ],
    );
    refuses(
        "twice",
        &[
            "src/lib.rs:9:5: error: multiple `deprecated` attributes: help: remove this attribute",
            "error: could not compile `twice` (lib) due to 2 previous errors",
        ],
    );
}
