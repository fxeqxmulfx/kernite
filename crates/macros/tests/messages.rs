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

/// What Rust says of a struct's argument that is not `Copy`, `ty`, where the struct asks for one, at
/// `place` of `src/lib.rs`: that the trait bound is not satisfied.
fn not_copy(place: &str, ty: &str) -> String {
    format!(
        "src/lib.rs:{place}: error[E0277]: the trait bound `{ty}: Copy` is not satisfied: the trait `Copy` is \
         not implemented for `{ty}`"
    )
}

/// What Rust says of a `_` in a signature, at `place` of `src/lib.rs`: that it is not allowed in the types of
/// `what`, `functions`, `methods` or `return types`.
fn not_allowed(place: &str, what: &str) -> String {
    format!(
        "src/lib.rs:{place}: error[E0121]: the placeholder `_` is not allowed within types on item signatures for \
         {what}: not allowed in type signatures"
    )
}

/// What Rust says of a number's name that the module gives another type, `found`, at `place` of
/// `src/lib.rs`: that the type is not the number, `expected`.
fn not_the_number(place: &str, expected: &str, found: &str) -> String {
    format!("src/lib.rs:{place}: error[E0308]: mismatched types: expected `{expected}`, found `{found}`")
}

/// What Rust says of `ty`, `f16` or `f128`, at `place` of `src/lib.rs`: that the type is unstable.
fn unstable(place: &str, ty: &str) -> String {
    format!("src/lib.rs:{place}: error[E0658]: the type `{ty}` is unstable")
}

/// What Rust says of the type `!` behind a pointer, at `place` of `src/lib.rs`: that it is experimental.
fn experimental(place: &str) -> String {
    format!("src/lib.rs:{place}: error[E0658]: the `!` type is experimental")
}

/// What Rust says of `path`, a module, where a type is written, at `place` of `src/lib.rs`: that it is no type.
fn not_a_type(place: &str, path: &str) -> String {
    format!("src/lib.rs:{place}: error[E0573]: expected type, found module `{path}`: not a type")
}

/// What Rust says of a `&` in a result that has no lifetime to take, at `place` of `src/lib.rs`.
fn no_lifetime(place: &str) -> String {
    format!("src/lib.rs:{place}: error[E0106]: missing lifetime specifier: expected named lifetime parameter")
}

/// What Rust says of a lifetime, `name`, that is not declared, at `place` of `src/lib.rs`.
fn undeclared(place: &str, name: &str) -> String {
    format!("src/lib.rs:{place}: error[E0261]: use of undeclared lifetime name `{name}`: undeclared lifetime")
}

/// What Rust says of a trait object of `name`, a trait that is not dyn compatible, at `place` of `src/lib.rs`.
fn not_dyn_compatible(place: &str, name: &str) -> String {
    format!(
        "src/lib.rs:{place}: error[E0038]: the trait `{name}` is not dyn compatible: `{name}` is not dyn compatible"
    )
}

/// What kernite says of a pointer to `ty`, a type with no size, at `place` of `src/lib.rs`: that it has two
/// words, where a kernel's pointer has one; `kernel` begins the message, as it does for [`x86_takes`].
fn two_words(place: &str, kernel: &str, ty: &str) -> String {
    format!(
        "src/lib.rs:{place}: error: {kernel}a pointer to `{ty}` has two words, an address and a length or a table, \
         and a kernel's pointer has one"
    )
}

/// What kernite says of `ty`, a type an x86 kernel does not take, at `place` of `src/lib.rs`; `kernel`
/// begins the message: [`instance`], or nothing.
fn x86_takes(place: &str, kernel: &str, ty: &str) -> String {
    format!("src/lib.rs:{place}: error: {kernel}a kernel takes pointers, 64-bit integers and floats, not `{ty}`")
}

/// What kernite says of `ty`, a type an x86 kernel does not give back, at `place` of `src/lib.rs`; `kernel`
/// begins the message, as it does for [`x86_takes`].
fn x86_gives(place: &str, kernel: &str, ty: &str) -> String {
    format!(
        "src/lib.rs:{place}: error: {kernel}a kernel returns a float, an integer or a pointer, or never, `!`; \
         not `{ty}`"
    )
}

/// What kernite says of a pointer a cooperative x86 kernel gives back, at `place` of `src/lib.rs`: that its
/// threads return nothing; `kernel` begins the message, as it does for [`x86_takes`].
fn coop_returns(place: &str, kernel: &str) -> String {
    format!(
        "src/lib.rs:{place}: error: {kernel}a cooperative kernel's threads return nothing: they write what they \
         compute to memory"
    )
}

/// What kernite says of `ty`, a type a GPU kernel gives back, at `place` of `src/lib.rs`; `kernel` begins the
/// message, as it does for [`x86_takes`].
fn gpu_returns(place: &str, kernel: &str, ty: &str) -> String {
    format!(
        "src/lib.rs:{place}: error: {kernel}a GPU kernel returns nothing, not `{ty}`: it writes what it works out \
         to memory"
    )
}

/// What kernite says of `ty`, a type a GPU kernel does not take, at `place` of `src/lib.rs`; `kernel` begins
/// the message, as it does for [`x86_takes`].
fn gpu_takes(place: &str, kernel: &str, ty: &str) -> String {
    format!(
        "src/lib.rs:{place}: error: {kernel}a GPU kernel takes pointers, integers and the floats `f16`, `bf16`, \
         `f32` and `f64`, not `{ty}`"
    )
}

/// What a message of kernite's begins with where it is of the instance `n` of the kernel `name`.
fn instance(name: &str, n: u64) -> String {
    format!("in `{name}::<{n}>`: ")
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

/// kernite reads no item of a module whose brackets nest too deep. Rust says nothing of the code that names a kernel of
/// it by path or by `use`, but cannot find it by a glob, and says the import is unused.
#[test]
fn brackets_nested_past_64_deep_are_refused_where_they_open() {
    // The 64th parenthesis is the first past the most, as `kernite::x86_str` counts the same
    // tokens.
    refuses(
        "nested",
        &[
            "src/lib.rs:8:80: error: this nests 64 deep at most",
            "src/lib.rs:25:9: warning: unused import: `crate::brackets::*`",
            "src/lib.rs:28:18: error[E0425]: cannot find function `f` in this scope: not found in this scope",
            "error: could not compile `nested` (lib) due to 2 previous errors; 1 warning emitted",
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

/// The error is at the attribute, and names the manifest, which is the crate's own: the crate is a workspace of its
/// own. The kernel compiles, so the error is the only one of its module. A module that `#[kernels]` refuses at its
/// attribute, beside it, is told its refusal alone: nothing of it is compiled, so it reads no workspace.
#[test]
fn a_workspace_named_by_what_is_not_a_string_is_refused_where_the_module_is() {
    let manifest = Path::new(ROOT).join("nonstring/project/Cargo.toml");
    let error = format!(
        "src/lib.rs:5:1: error: `name` in `[workspace.metadata.kernite]` of {} is not a string: a workspace is named by \
         one, as `name = \"kernite\"`",
        manifest.display()
    );
    refuses(
        "nonstring",
        &[
            &error,
            &format!("src/lib.rs:14:27: error: {REFUSED}"),
            "error: could not compile `nonstring` (lib) due to 2 previous errors",
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

/// A static whose value alone is wrong stands as its type has it, so that the code that uses it is
/// read as it is for one that compiles: a use that leaves the type of the elements to the static,
/// `COUNTS.read()[0] as u64`, needs no type, and a use that the type does not fit, `COUNTS.read()`
/// where a `Vec<u64>` is wanted, is Rust's error, beside kernite's, as it is for one that compiles.
/// A static whose type kernite does not take, here one that holds nothing, stands as one whose
/// elements are of any type: a use that says the type is no error of Rust's, and a use that says none
/// is, beside kernite's. So does one whose type is refused and whose value is wrong too, `HOLLOW`:
/// kernite says the value only, since it reads that first, and Rust asks for the type of a use that says
/// none, of the process's static and of the GPU's.
#[test]
fn a_static_whose_value_alone_is_wrong_stands_as_its_type_has_it() {
    refuses(
        "static_stubs",
        &[
            "src/lib.rs:8:43: error: `global<[u32; 4]>` holds 4 elements; this has 5",
            "src/lib.rs:9:48: error: element 0 is an integer, and an `f32` a float: `1.0`",
            "src/lib.rs:10:35: error: 300 is out of range for u8",
            "src/lib.rs:11:45: error: `nope` is not defined",
            "src/lib.rs:12:23: error: `global<[u32; 0]>` holds nothing",
            "src/lib.rs:21:43: error: `global<[u32; 4]>` holds 4 elements; this has 5",
            "src/lib.rs:22:45: error: element 0 is an integer, and an `f16` a float: `1.0`",
            "src/lib.rs:23:45: error: `nope` is not defined",
            "src/lib.rs:24:23: error: `global<[u32; 0]>` holds nothing",
            // `HOLLOW`, whose type is refused too: kernite says the value, which it reads first.
            "src/lib.rs:151:44: error: `nope` is not defined",
            "src/lib.rs:160:44: error: `nope` is not defined",
            // Rust's, in the order of the uses: the process's `COUNTS` where `u64`s are wanted; its `EMPTY`, read
            // and cast to an address with no type said; the GPU's `COUNTS` where `u64`s are wanted; its `EMPTY`,
            // with no type said; and the `HOLLOW` of the process's, read with no type said, and the GPU's.
            "src/lib.rs:65:5: error[E0308]: mismatched types: expected `Vec<u64>`, found `Vec<u32>`",
            "src/lib.rs:76:16: error[E0282]: type annotations needed: cannot infer type of the type parameter `T` \
             declared on the method `read`",
            "src/lib.rs:80:25: error[E0641]: cannot cast from a pointer of an unknown kind",
            "src/lib.rs:133:5: error[E0308]: mismatched types: expected `Result<Vec<u64>, Error>`, found \
             `Result<Vec<u32>, Error>`",
            "src/lib.rs:143:13: error[E0283]: type annotations needed: cannot infer type of the type parameter `T` \
             declared on the method `EMPTY`",
            "src/lib.rs:169:24: error[E0282]: type annotations needed: cannot infer type of the type parameter `T` \
             declared on the method `read`",
            "src/lib.rs:174:13: error[E0283]: type annotations needed: cannot infer type of the type parameter `T` \
             declared on the method `HOLLOW`",
            "error: could not compile `static_stubs` (lib) due to 18 previous errors",
        ],
    );
}

/// An integer where an integer of another type is wanted is told by its type, `a u16` where a `u8` is wanted, and
/// `an i32` for a signed one: `an integer` would say no fault, for a `u8` is one too. So it is for a const, `A`,
/// whether the type wanted is signed, `B`, or not; for an element of an array const, `C`, told at the array; and for
/// a compile-time argument of `#[instances(..)]`, on x86 and on a GPU. A static's value says its type already, in
/// words of its own, on both. Where the type wanted is no integer's, `D`, what is wrong is that the value is an
/// integer, and kernite says `an integer`. Rust's own error stands beside kernite's, at the const.
#[test]
fn an_integer_of_another_type_is_told_by_its_type_beside_rusts_error() {
    refuses(
        "int_kinds",
        &[
            "src/lib.rs:9:23: error: this is a u16, not u8",
            "src/lib.rs:10:24: error: this is a u32, not i64",
            "src/lib.rs:11:28: error: this is a u16, not u8",
            "src/lib.rs:12:25: error: this is an integer, not bool",
            "src/lib.rs:37:17: error: this is a u16, not u8",
            "src/lib.rs:46:20: error: this is an i32, not u8",
            "src/lib.rs:56:33: error: this is a `u16`, not a `u32`",
            "src/lib.rs:65:33: error: this is a `u16`, not a `u32`",
            &not_the_number("9:23", "u8", "u16"),
            &not_the_number("10:24", "i64", "u32"),
            &not_the_number("11:34", "u8", "u16"),
            &not_the_number("12:25", "bool", "u16"),
            "error: could not compile `int_kinds` (lib) due to 12 previous errors",
        ],
    );
}

/// A crate that does not depend on `kernite-cpu` is told where it needs the crate, at each static of
/// a module of x86 kernels and at each cooperative kernel, by kernite and not by Rust, which would
/// say that `kernite_cpu` is not found, at the attribute. A cooperative kernel's result is said as
/// it is in a crate that has `kernite-cpu`: kernite's error at the type, and Rust's, after kernite's,
/// that a pointer's type has no size.
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
            "src/lib.rs:34:35: error: a cooperative x86 kernel needs a dependency named `kernite-cpu`, \
             which this crate does not have",
            coop_returns("34:58", "").as_str(),
            size_unknown("34:65", "[i8]").as_str(),
            "error: could not compile `no_cpu` (lib) due to 6 previous errors",
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

/// A crate whose dependencies on `kernite-macros`, `kernite-cuda` and `kernite-cpu` are renamed in its
/// manifest, `cuda = { package = "kernite-cuda", .. }`, is told as one that has none is, for the Rust
/// kernite writes names `kernite_macros`, for the room of a GPU module's statics and for the consts a
/// module of kernels is given, `kernite_cuda` and `kernite_cpu`, which it has not; and Rust says nothing
/// else, nor at the module of consts. A module whose statics `cfg` leaves out is told nothing, and one
/// with a static there only with a feature, `more`, is told with it. So is a module given consts, at its
/// attribute: Rust takes out one that `cfg` leaves out, written before the attribute, after it or in the
/// module's braces, before the macro runs, and the macro runs on each one that `cfg` keeps. A module of GPU kernels
/// with a static that `#[kernels]` refuses at its attribute, last, is told its refusal and the `kernite-cuda` it
/// needs, as the first is, and not the `kernite-macros` it needs for its static: no room of it is counted.
#[test]
fn a_crate_that_renames_kernite_macros_kernite_cuda_and_kernite_cpu_is_told_it_has_none() {
    let edits = [
        ("kernite-macros = {", "km = { package = \"kernite-macros\","),
        ("kernite-cuda = {", "cuda = { package = \"kernite-cuda\","),
        ("kernite-cpu = {", "cpu = { package = \"kernite-cpu\","),
    ];
    let dir = edited("renamed_deps", &edits);
    let (built, without) = cargo_in(&["build"], &dir);
    assert!(!built, "fixtures/renamed_deps.rs built");
    assert_eq!(
        without,
        [
            "src/lib.rs:8:1: error: a static of GPU kernels needs a dependency named `kernite-macros`, \
             which this crate does not have",
            "src/lib.rs:8:1: error: a module of GPU kernels needs a dependency named `kernite-cuda`, \
             which this crate does not have",
            "src/lib.rs:19:16: error: a static of x86 kernels needs a dependency named `kernite-cpu`, \
             which this crate does not have",
            "src/lib.rs:22:35: error: a cooperative x86 kernel needs a dependency named `kernite-cpu`, \
             which this crate does not have",
            "src/lib.rs:63:1: error: a module of kernels given consts needs a dependency named `kernite-macros`, \
             which this crate does not have",
            "src/lib.rs:93:1: error: a module of kernels given consts needs a dependency named `kernite-macros`, \
             which this crate does not have",
            &format!("src/lib.rs:111:15: error: {REFUSED}")[..],
            "src/lib.rs:111:1: error: a module of GPU kernels needs a dependency named `kernite-cuda`, which this \
             crate does not have",
            "error: could not compile `renamed_deps` (lib) due to 8 previous errors",
        ]
    );
    let (built, with) = cargo_in(&["build", "--features", "more"], &dir);
    assert!(!built, "fixtures/renamed_deps.rs built with the feature");
    assert_eq!(with[..4], without[..4]);
    assert_eq!(
        with[4..],
        [
            "src/lib.rs:47:1: error: a static of GPU kernels needs a dependency named `kernite-macros`, \
             which this crate does not have",
            "src/lib.rs:47:1: error: a module of GPU kernels needs a dependency named `kernite-cuda`, \
             which this crate does not have",
            "src/lib.rs:63:1: error: a module of kernels given consts needs a dependency named `kernite-macros`, \
             which this crate does not have",
            "src/lib.rs:83:1: error: a module of kernels given consts needs a dependency named `kernite-macros`, \
             which this crate does not have",
            "src/lib.rs:89:1: error: a module of kernels given consts needs a dependency named `kernite-macros`, \
             which this crate does not have",
            "src/lib.rs:104:1: error: a module of kernels given consts needs a dependency named `kernite-macros`, \
             which this crate does not have",
            &format!("src/lib.rs:111:15: error: {REFUSED}")[..],
            "src/lib.rs:111:1: error: a module of GPU kernels needs a dependency named `kernite-cuda`, which this \
             crate does not have",
            "error: could not compile `renamed_deps` (lib) due to 12 previous errors",
        ]
    );
}

/// A crate that renames `kernite-macros` builds a module of consts that no module of kernels takes without
/// a word: it is a module of kernels given consts that needs a dependency of that name.
#[test]
fn a_crate_that_renames_kernite_macros_builds_consts_no_module_takes() {
    let dir = edited("renamed_consts", &[("kernite-macros = {", "km = { package = \"kernite-macros\",")]);
    builds_quietly_in(&["build"], "renamed_consts", &dir);
}

/// What `#[kernels]` says of an argument that is not `consts = path`.
const REFUSED: &str = "`#[kernels]` takes the module of consts its kernels read, as `consts = crate::model::sizes`, a \
                       module `#[kernite_macros::consts]` is on; or nothing";

/// A module of kernels that `#[kernels]` refuses at its attribute, `consts = 5`, is told its refusal once, there. It is
/// still there, each kernel as of one that does not compile and each static as its type has it, and a module of x86
/// kernels has an empty `LISTINGS`, so Rust says nothing of the code that names its items: from the crate's root, from
/// another module by path and by `use`, or by a glob. A module kernite cannot read has the refusal alone, and the
/// items kernite reads of it. What else such a module is told is the tests' below.
#[test]
fn a_module_of_kernels_refused_at_its_attribute_is_still_there_for_the_code_that_names_it() {
    refuses(
        "refused_attribute",
        &[
            &format!("src/lib.rs:6:27: error: {REFUSED}"),
            &format!("src/lib.rs:49:27: error: {REFUSED}"),
            &format!("src/lib.rs:78:27: error: {REFUSED}"),
            "error: could not compile `refused_attribute` (lib) due to 3 previous errors",
        ],
    );
}

/// A static of a module that `#[kernels]` refuses at its attribute stands as its type has it, as a static whose value
/// alone is wrong does, though nothing of the module is compiled and nothing is said of what the static holds:
/// the code that uses it is read as for one that compiles, though it says no type for the elements, and a use the type
/// does not fit is Rust's error, beside the refusal. A static whose type reads a const the module would be given,
/// `SIZED`, stands as one whose elements are of any type: a use that says the type is no error of Rust's, and a use that
/// says none is, beside the refusal. The process's statics and the GPU's alike.
#[test]
fn a_static_of_a_module_refused_at_its_attribute_stands_as_its_type_has_it() {
    refuses(
        "refused_statics",
        &[
            &format!("src/lib.rs:8:27: error: {REFUSED}"),
            &format!("src/lib.rs:18:27: error: {REFUSED}"),
            // Rust's, in the order of the uses: the process's `COUNTS` where `u64`s are wanted; its `SIZED`, read
            // and cast to an address with no type said; the GPU's `COUNTS` where `u64`s are wanted; its `SIZED`,
            // with no type said.
            "src/lib.rs:46:5: error[E0308]: mismatched types: expected `Vec<u64>`, found `Vec<u32>`",
            "src/lib.rs:57:16: error[E0282]: type annotations needed: cannot infer type of the type parameter `T` \
             declared on the method `read`",
            "src/lib.rs:61:25: error[E0641]: cannot cast from a pointer of an unknown kind",
            "src/lib.rs:78:5: error[E0308]: mismatched types: expected `Result<Vec<u64>, Error>`, found \
             `Result<Vec<u32>, Error>`",
            "src/lib.rs:88:13: error[E0283]: type annotations needed: cannot infer type of the type parameter `T` \
             declared on the method `SIZED`",
            "error: could not compile `refused_statics` (lib) due to 7 previous errors",
        ],
    );
}

/// A module of kernels that `#[kernels]` refuses at its attribute is told its refusal, and what the macro finds in
/// writing it for Rust, as a module that is not refused is: the dependency its crate needs, which the tests below
/// hold, a compile-time argument named as a parameter of a launcher, `grid`, or of a `_blocks` method, `threads` and
/// `shared`, and a GPU kernel or a static named as a cooperative kernel's `_blocks` method. It is told nothing that
/// kernite would find in its items, for kernite checks none: a kernel that is not `unsafe`, a `#[instances(..)]` that
/// does not hold, a name that is not defined, an ABI kernite does not compile for, a static that holds nothing, and
/// kernels of both kinds with a static. With the feature `taken` the same modules are not refused, and each mistake
/// is said.
#[test]
fn a_module_refused_at_its_attribute_is_told_what_the_macro_finds_in_writing_it_and_nothing_in_its_items() {
    refuses(
        "refused_mistakes",
        &[
            &format!("src/lib.rs:7:60: error: {REFUSED}"),
            "src/lib.rs:43:50: error: `grid` names a parameter of the launcher as well",
            &format!("src/lib.rs:48:60: error: {REFUSED}"),
            "src/lib.rs:68:54: error: `grid` names a parameter of the launcher as well",
            "src/lib.rs:76:54: error: `threads` names a parameter of `together_blocks` as well",
            "src/lib.rs:76:74: error: `shared` names a parameter of `together_blocks` as well",
            "src/lib.rs:81:7: error: `#[cooperative]` gives `Kernels` a method `cut_blocks`, which says how many \
             blocks a grid of `cut` has at most, and a kernel or a static of the module is named `cut_blocks` as well",
            &format!("src/lib.rs:92:60: error: {REFUSED}"),
            "error: could not compile `refused_mistakes` (lib) due to 8 previous errors",
        ],
    );
    refuses_by(
        &["build", "--features", "taken"],
        "refused_mistakes",
        &[
            "src/lib.rs:11:28: error: a kernel trusts the pointers it is given, so it is an `unsafe fn`",
            "src/lib.rs:16:7: error: `once` takes no compile-time arguments, so it is compiled once, with no \
             `#[instances(..)]`",
            "src/lib.rs:23:16: error: `nope` is not defined",
            "src/lib.rs:28:7: error: a helper is compiled where it is called, for what `::<..>` gives it there: \
             `#[instances(..)]` lists what a function that names an ABI is compiled for",
            "src/lib.rs:34:23: error: kernite compiles `extern \"sysv64\"` kernels, for x86-64, and `extern \
             \"ptx-kernel\"` ones, for NVIDIA GPUs, with the `extern \"ptx-func\"` functions they call; `extern \
             \"C\"` is none of these",
            "src/lib.rs:37:23: error: `global<[u32; 0]>` holds nothing",
            "src/lib.rs:43:50: error: `grid` names a parameter of the launcher as well",
            "src/lib.rs:51:32: error: a kernel trusts the pointers it is given, so it is an `unsafe fn`",
            "src/lib.rs:55:7: error: `once` takes no compile-time arguments, so it is compiled once, with no \
             `#[instances(..)]`",
            "src/lib.rs:61:28: error: `nope` is not defined",
            "src/lib.rs:64:23: error: `global<[u32; 0]>` holds nothing",
            "src/lib.rs:68:54: error: `grid` names a parameter of the launcher as well",
            "src/lib.rs:76:54: error: `threads` names a parameter of `together_blocks` as well",
            "src/lib.rs:76:74: error: `shared` names a parameter of `together_blocks` as well",
            "src/lib.rs:81:7: error: `#[cooperative]` gives `Kernels` a method `cut_blocks`, which says how many \
             blocks a grid of `cut` has at most, and a kernel or a static of the module is named `cut_blocks` as well",
            "src/lib.rs:95:16: error: a static is memory of the module's kernels: the GPU's, for PTX kernels, or the \
             process's, for x86 ones; and this module has both, so it is two modules, one of each kind of kernel and \
             the statics they name",
            "error: could not compile `refused_mistakes` (lib) due to 16 previous errors",
        ],
    );
}

/// A module of kernels that kernite cannot read is still there, as far as kernite reads it, for the code that names its
/// items: a module of x86 kernels and one of GPU kernels, each with two items of a name, `same` and `scale`, say so
/// once, at the second, and are left the first of the two and every item beside them: a kernel as of one that does not
/// compile, a const, a struct, and a static as its type has it. So Rust says nothing of the code that names them, from
/// the crate's root, from another module by `use` or by a glob.
#[test]
fn a_module_with_two_items_of_a_name_is_still_there_for_the_code_that_names_it() {
    refuses(
        "unreadable_glob",
        &[
            "src/lib.rs:10:35: error: `same` names another function of the module as well",
            "src/lib.rs:37:39: error: `scale` names another function of the module as well",
            "error: could not compile `unreadable_glob` (lib) due to 2 previous errors",
        ],
    );
}

/// A module of kernels with the head of a function kernite does not take, `bad<T>`, says so once, and is still there as
/// far as kernite reads it: `good` comes before `bad`, so Rust says nothing of the code that names `good` by the
/// module's path, by `use` or by a glob, as it says nothing of one that names a module that compiles.
#[test]
fn a_module_with_the_head_of_a_kernel_it_cannot_read_is_still_there_for_the_kernel_it_reads() {
    refuses(
        "unreadable_head",
        &[
            "src/lib.rs:10:39: error: a function's generics are values, as in `<const N: usize>`",
            "error: could not compile `unreadable_head` (lib) due to 1 previous error",
        ],
    );
}

/// A function kernite cannot read ends at its body, though its head holds a brace group in `<..>`, as a const argument
/// does: of its result's type, of a bound or of a `where` clause. So nothing of the three is left over for Rust to read
/// as an item, which it would refuse. kernite's error is said alone, at the first. `good`, after them, is still there
/// for the glob that names it.
#[test]
fn a_function_kernite_cannot_read_ends_at_its_body_whatever_a_const_argument_of_its_head_holds() {
    refuses(
        "unreadable_angle",
        &[
            "src/lib.rs:11:22: error: a function's generics are values, as in `<const N: usize>`",
            "error: could not compile `unreadable_angle` (lib) due to 1 previous error",
        ],
    );
}

/// kernite reads every item of a module but the one it cannot read, `bad`: `good` and `scale` before it, `late` after
/// it. Rust says nothing of a path or a `use` that names `bad`, for the error is in the module, but cannot find it by a
/// glob, as it cannot find a name no module has; and the import is unused, as for any glob, where `bad` is all it
/// names. A GPU kernel is named as a method of `Kernels`: Rust finds `scale` and `late`, and not `bad`.
#[test]
fn rust_cannot_find_by_a_glob_what_kernite_does_not_know_of_a_module_it_cannot_read() {
    refuses(
        "unreadable_unknown",
        &[
            "src/lib.rs:11:39: error: a function's generics are values, as in `<const N: usize>`",
            "src/lib.rs:29:43: error: a function's generics are values, as in `<const N: usize>`",
            "src/lib.rs:76:9: warning: unused import: `crate::cpu::*`",
            "src/lib.rs:48:17: error[E0599]: no method named `bad` found for reference `&Kernels` in the current \
             scope: method not found in `&Kernels`",
            "src/lib.rs:70:28: error[E0425]: cannot find function `bad` in this scope: not found in this scope",
            "src/lib.rs:79:18: error[E0425]: cannot find function `bad` in this scope: not found in this scope",
            "error: could not compile `unreadable_unknown` (lib) due to 5 previous errors; 1 warning emitted",
        ],
    );
}

/// A module of x86 kernels that kernite cannot read has a static that kernite knows where it has read a function that
/// names an ABI, before the item it cannot read, `first`, or after it, `last`, and else knows none, for such a function
/// says whose a static is, the process's or the GPU's. Rust says nothing of the code that names the first two by a
/// path, by `use` or by a glob, nor of the third by a path or by `use`, and cannot find it by a glob. Nor does it ask
/// for `kernite-cuda`, as it would if kernite took the third for the GPU's.
#[test]
fn a_static_of_a_module_kernite_cannot_read_is_known_if_a_function_that_names_an_abi_is_read() {
    let lines = [
        "src/lib.rs:15:39: error: a function's generics are values, as in `<const N: usize>`",
        "src/lib.rs:24:39: error: a function's generics are values, as in `<const N: usize>`",
        "src/lib.rs:37:39: error: a function's generics are values, as in `<const N: usize>`",
        "src/lib.rs:73:9: error[E0425]: cannot find value `COUNTS` in this scope: not found in this scope",
        "src/lib.rs:70:9: warning: unused import: `crate::no_other::*`",
        "error: could not compile `unreadable_statics` (lib) due to 4 previous errors; 1 warning emitted",
    ];
    refuses("unreadable_statics", &lines);
    let (built, said) = cargo_in(&["build"], &edited("unreadable_statics", &[("kernite-cuda =", "# kernite-cuda =")]));
    assert!(!built, "fixtures/unreadable_statics.rs built without kernite-cuda");
    assert_eq!(said, lines, "without kernite-cuda, Cargo said:\n{}", said.join("\n"));
}

/// A `const` or a `static` that kernite cannot read ends at its `;`, though its value holds a brace group with more
/// after it: a block added to, a struct literal read for a field and a branch are each told once, and Rust says nothing
/// of what would be left of them as an item of its own, nor of the code that names the kernel beside them.
#[test]
fn a_const_or_a_static_kernite_cannot_read_ends_at_its_semicolon() {
    refuses(
        "unreadable_cut",
        &[
            "src/lib.rs:8:24: error: a block is a statement, not a value",
            "src/lib.rs:17:43: error: expected `;`, found `{ a: 1 }`",
            "src/lib.rs:26:27: error: expected `;`, found `true`",
            "error: could not compile `unreadable_cut` (lib) due to 3 previous errors",
        ],
    );
}

/// A module that takes consts and that kernite cannot read says so alone, and is still there as far as kernite reads
/// it, the consts it takes among its items, and a static whose type reads one of them. A kernel of its own named as a
/// const it takes, `STEP`, is not said, as it is in a module that compiles, for the error that can be mended first is
/// the head.
#[test]
fn a_module_given_consts_that_kernite_cannot_read_is_still_there_and_says_its_error_alone() {
    refuses(
        "unreadable_given",
        &[
            "src/lib.rs:25:39: error: a function's generics are values, as in `<const N: usize>`",
            "error: could not compile `unreadable_given` (lib) due to 1 previous error",
        ],
    );
}

/// An argument of `#[consts]` is told where it is, once, and changes nothing else. The module is the one it is without
/// the argument, so Rust says nothing of the code that reads its consts, nor of a module of kernels that takes them.
#[test]
fn a_module_of_consts_given_an_argument_is_still_there_for_the_code_that_reads_it() {
    refuses(
        "consts_argument",
        &[
            "src/lib.rs:6:26: error: `#[consts]` takes no arguments",
            "error: could not compile `consts_argument` (lib) due to 1 previous error",
        ],
    );
}

/// The module of consts is the one it is without the argument, so its other errors are said, beside the argument's:
/// they are other mistakes. The module of kernels that takes its consts reads the one that is right, and says its own
/// error.
#[test]
fn a_module_of_consts_given_an_argument_says_its_other_errors_as_well() {
    refuses(
        "consts_argument_beside",
        &[
            "src/lib.rs:6:26: error: `#[consts]` takes no arguments",
            "src/lib.rs:8:5: error: a const of a module of consts is `pub`, for the modules of kernels that take it to \
             see",
            "src/lib.rs:10:5: error: a module of consts holds the consts kernels read, as `pub const E: usize = 16;`, \
             and nothing else",
            "src/lib.rs:17:16: error: `nope` is not defined",
            "error: could not compile `consts_argument_beside` (lib) due to 4 previous errors",
        ],
    );
}

/// A module of GPU kernels that `#[kernels]` refuses at its attribute needs `kernite-cuda` where `cfg` keeps one of
/// its GPU items, as a module kernite does not refuse does: the crate is told once, at the module, beside the
/// attribute, and not where the feature leaves the kernel out.
#[test]
fn a_module_refused_at_its_attribute_needs_kernite_cuda_where_cfg_keeps_a_gpu_item() {
    let dir = edited("refused_features", &[("kernite-cuda =", "# kernite-cuda =")]);
    let (built, kept) = cargo_in(&["build", "--features", "a"], &dir);
    assert!(!built, "fixtures/refused_features.rs built with the feature");
    assert_eq!(
        kept,
        [
            &format!("src/lib.rs:6:27: error: {REFUSED}")[..],
            "src/lib.rs:6:1: error: a module of GPU kernels needs a dependency named `kernite-cuda`, which this crate \
             does not have",
            "error: could not compile `refused_features` (lib) due to 2 previous errors",
        ]
    );
    let (built, left_out) = cargo_in(&["build"], &dir);
    assert!(!built, "fixtures/refused_features.rs built without the feature");
    assert_eq!(
        left_out,
        [
            &format!("src/lib.rs:6:27: error: {REFUSED}")[..],
            "error: could not compile `refused_features` (lib) due to 1 previous error",
        ]
    );
}

/// A module of x86 kernels that `#[kernels]` refuses at its attribute needs `kernite-cpu` for its static and its
/// cooperative kernel, as a module kernite does not refuse does: each is told, at the static and at the kernel,
/// beside the attribute. Rust says nothing of the code that names them.
#[test]
fn a_module_refused_at_its_attribute_needs_kernite_cpu_for_a_static_and_a_cooperative_kernel() {
    let dir = edited("refused_cpu", &[("kernite-cpu =", "# kernite-cpu =")]);
    let (built, said) = cargo_in(&["build"], &dir);
    assert!(!built, "fixtures/refused_cpu.rs built");
    assert_eq!(
        said,
        [
            &format!("src/lib.rs:5:27: error: {REFUSED}")[..],
            "src/lib.rs:7:16: error: a static of x86 kernels needs a dependency named `kernite-cpu`, which this crate \
             does not have",
            "src/lib.rs:10:35: error: a cooperative x86 kernel needs a dependency named `kernite-cpu`, which this \
             crate does not have",
            "error: could not compile `refused_cpu` (lib) due to 3 previous errors",
        ]
    );
}

/// A module of kernels given consts, in a crate that renames `kernite-macros`, is told once, at its attribute. It
/// is still there, as a module that `#[kernels]` refuses is, so Rust says nothing of the code that names its items.
/// The static of the GPU one does not tell it again that it needs `kernite-macros`: no room of it is counted.
#[test]
fn a_module_of_kernels_given_consts_without_kernite_macros_is_still_there_for_the_code_that_names_it() {
    let dir = edited("renamed_given", &[("kernite-macros = {", "km = { package = \"kernite-macros\",")]);
    let (built, said) = cargo_in(&["build"], &dir);
    assert!(!built, "fixtures/renamed_given.rs built");
    assert_eq!(
        said,
        [
            "src/lib.rs:11:1: error: a module of kernels given consts needs a dependency named `kernite-macros`, \
             which this crate does not have",
            "src/lib.rs:19:1: error: a module of kernels given consts needs a dependency named `kernite-macros`, \
             which this crate does not have",
            "error: could not compile `renamed_given` (lib) due to 2 previous errors",
        ]
    );
}

/// A module of kernels, x86 or GPU, whose consts' path Rust cannot find, `crate::nope`, is told once, at the path, for
/// each attribute that has it: Rust says the crate has no `nope`, and nothing of the code that names the module's
/// items, wherever it is: from the crate's root, from another module by path, or through a glob of the module. The glob
/// brings in nothing, and Rust warns of that alone. kernite cannot tell that the path fails, for Rust finds it after
/// the macro runs, so the module is there, with nothing in it, for that code to find.
#[test]
fn a_module_of_kernels_whose_consts_path_rust_cannot_find_is_told_once() {
    refuses(
        "consts_path_missing",
        &[
            "src/lib.rs:5:43: error[E0432]: unresolved import `crate::nope`: could not find `nope` in the crate root",
            "src/lib.rs:15:43: error[E0432]: unresolved import `crate::nope`: could not find `nope` in the crate root",
            "src/lib.rs:45:9: warning: unused import: `crate::cpu::*`",
            "src/lib.rs:53:9: warning: unused import: `crate::gpu::*`",
            "error: could not compile `consts_path_missing` (lib) due to 2 previous errors; 2 warnings emitted",
        ],
    );
}

/// A module of kernels whose consts' path is a module with no `#[kernite_macros::consts]` on it is told once, at the
/// path, as one that is not there is: the module has no `__kernite_consts`, which that attribute writes. Rust says
/// nothing of the code that names the module's items. The glob brings in nothing, and Rust warns of that alone.
#[test]
fn a_module_of_kernels_whose_consts_path_is_a_module_with_no_consts_attribute_is_told_once() {
    refuses(
        "consts_path_plain",
        &[
            "src/lib.rs:9:43: error[E0432]: unresolved import `__plain_3`: no `__kernite_consts` in `plain`",
            "src/lib.rs:31:9: warning: unused import: `crate::cpu::*`",
            "error: could not compile `consts_path_plain` (lib) due to 1 previous error; 1 warning emitted",
        ],
    );
}

/// A consts path Rust cannot find is told once, at the name it cannot find, whatever form the path has: from the root
/// of the crate, `crate::nope`; of a crate, `::forms::nope`, and one that is not there; of a name, `self` or `super`,
/// which Rust says from the module the macro writes in the module of kernels, `nope` as `super::super::nope`; one that
/// ends at a const, which is no module; and one of which the first name is not there, or the last, or one between. A
/// name after the first is said in the module before it, by the name the macro gives that module, `__hidden_3::nope`.
/// The code that calls the kernels of each, from the crate's root and from the module they are in, is told nothing.
/// Rust says first the paths that fail at their first name, and then those that fail at a later one.
#[test]
fn a_consts_path_is_told_once_at_the_name_rust_cannot_find_in_each_form() {
    refuses(
        "consts_path_forms",
        &[
            "src/lib.rs:16:43: error[E0432]: unresolved import `crate::nope`: could not find `nope` in the crate root",
            "src/lib.rs:30:38: error[E0432]: unresolved import `nope`: could not find `nope` in the list of imported \
             crates",
            "src/lib.rs:37:36: error[E0432]: unresolved import `super::super::nope`: could not find `nope` in the \
             crate root",
            "src/lib.rs:44:42: error[E0432]: unresolved import `super::super::nope`: could not find `nope` in the \
             crate root",
            "src/lib.rs:51:43: error[E0432]: unresolved import `crate::SIZE`: could not find `SIZE` in the crate root",
            "src/lib.rs:58:43: error[E0432]: unresolved import `crate::nope`: could not find `nope` in the crate root",
            "src/lib.rs:81:47: error[E0432]: unresolved import `super::super::super::nope`: could not find `nope` in \
             the crate root",
            "src/lib.rs:88:46: error[E0432]: unresolved import `super::super::sizes`: could not find `sizes` in \
             `super`",
            "src/lib.rs:23:45: error[E0432]: unresolved import `__forms_2::nope`: could not find `nope` in the crate \
             root",
            "src/lib.rs:65:51: error[E0432]: unresolved import `__hidden_3::nope`: could not find `nope` in \
             `__hidden_3`",
            "src/lib.rs:72:51: error[E0432]: unresolved import `__hidden_3::nope`: could not find `nope` in \
             `__hidden_3`",
            "error: could not compile `consts_path_forms` (lib) due to 11 previous errors",
        ],
    );
}

/// A `use` of an item of such a module by name is Rust's error as well, beside the path's: the module has no such item
/// until the path is mended, and kernite cannot give it one, for what the module has is what the consts make of it. The
/// module itself, imported or renamed, is there, and Rust says nothing of the code that calls its kernels by it.
#[test]
fn a_use_of_an_item_by_name_of_a_module_whose_consts_path_rust_cannot_find_is_told_as_well() {
    refuses(
        "consts_path_named",
        &[
            "src/lib.rs:5:43: error[E0432]: unresolved import `crate::nope`: could not find `nope` in the crate root",
            "src/lib.rs:18:9: error[E0432]: unresolved import `crate::cpu::step`",
            "src/lib.rs:26:28: error[E0432]: unresolved imports `crate::cpu::step`, `crate::cpu::twice`: no `step` in \
             `cpu`, no `twice` in `cpu`",
            "src/lib.rs:34:13: error[E0432]: unresolved import `crate::cpu::step`",
            "error: could not compile `consts_path_named` (lib) due to 4 previous errors",
        ],
    );
}

/// A module of kernels given consts keeps its own attributes first in it, the docs `//!` and the attributes `#![..]`,
/// as a module that is given none does: its docs are on its page, and a lint that it forbids is not broken by what the
/// macro writes in it to find the module of consts. No page shows that, not even a page of private items.
#[test]
fn a_module_of_kernels_given_consts_keeps_its_own_attributes_first() {
    builds_quietly("consts_inner");
    let docs = documents_quietly("consts_inner");
    for (module, said) in [("cpu", "Kernels that step."), ("gpu", "Kernels that step on the GPU.")] {
        let page = fs::read_to_string(docs.join(module).join("index.html")).unwrap();
        assert!(page.contains(said), "the page of `{module}` lacks its docs, `{said}`");
    }
    builds_quietly_in(&["doc", "--no-deps", "--document-private-items"], "consts_inner", &written("consts_inner"));
    for module in ["cpu", "gpu"] {
        let page = fs::read_to_string(docs.join(module).join("index.html")).unwrap();
        assert!(!page.contains("__consts"), "the page of `{module}` shows the module the macro wrote in it");
    }
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

/// A GPU kernel's launcher says what a launch keeps to, as `Module::launch` does: the kernel's own
/// `# Safety`, or, where it has none, the macros': each pointer is to a buffer or to unified
/// memory of the GPU; from the launch until the kernel is done, the host holds no slice of memory
/// the kernel writes, and no slice it may write of memory the kernel reads; and where
/// `Cuda::concurrent_managed_access` is false, no slice of managed memory at all.
#[test]
fn a_gpu_kernels_launcher_says_what_a_launch_keeps_to() {
    let page = fs::read_to_string(documents_quietly("contract").join("gpu/struct.Kernels.html")).unwrap();
    // The paragraphs after the heading of `name`'s `# Safety`, as rustdoc writes them.
    let safety = |name: &str| -> Vec<String> {
        let (_, method) = page.split_once(&format!("id=\"method.{name}\"")).expect("the method is there");
        let (_, docs) = method.split_once(">Safety</h5>\n").unwrap_or_else(|| panic!("`{name}` has no `# Safety`"));
        docs.split_once("</div>").expect("the docs end").0.lines().map(String::from).collect()
    };
    let contract = "<p>The kernel trusts the pointers it is given: each must be to a buffer or to unified memory of \
                    the GPU the kernels are loaded on, as long as the kernel reads and writes through it. From the \
                    launch until the kernel is done, the host must hold no slice of memory the kernel writes, and no \
                    slice it may write of memory the kernel reads; and where \
                    <code>Cuda::concurrent_managed_access</code> is false, the GPU not reaching managed memory while \
                    the host does, the host must hold no slice of managed memory at all.</p>";
    assert_eq!(safety("one"), [contract]);
    assert_eq!(safety("two"), ["<p><code>x</code> points at an integer.</p>"]);
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

/// A module of x86 kernels has `LISTINGS` whether its kernels compile or not, so Rust says nothing of the code that
/// names it, a glob's included: kernite's errors are the only ones. It holds the listing of each kernel that compiles
/// and `cfg` keeps, and of no other, and nothing where no kernel compiles. A module of GPU kernels alone has none,
/// so a glob of it beside one of a module with a `LISTINGS` of its own is not ambiguous. The fixture's consts assert
/// what each holds, which Rust evaluates beside kernite's errors.
#[test]
fn a_module_of_x86_kernels_has_listings_whether_its_kernels_compile_or_not() {
    refuses(
        "listings_uncompiled",
        &[
            "src/lib.rs:15:16: error: `nope` is not defined",
            "src/lib.rs:29:16: error: `nope` is not defined",
            "error: could not compile `listings_uncompiled` (lib) due to 2 previous errors",
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

/// The room of a module's statics is that of the statics `cfg` keeps. The fixture has two tables of
/// constant memory that do not fit together, and two statics of global memory, one of which `cfg`
/// leaves out in one module; it builds under `deny(warnings)` without a word, with the feature `new` and
/// without, and its tests assert which table the module has and, on a machine with a GPU, load it and
/// read the table that is there.
#[test]
fn the_room_of_a_gpu_module_counts_the_statics_cfg_keeps() {
    for features in ["", "new"] {
        let command = ["test", "--features", features];
        let printed = runs_quietly(&command, "room");
        assert!(printed.contains("test result: ok. 2 passed"), "fixtures/room.rs: {command:?} printed:\n{printed}");
    }
}

/// A static that does not fit beside those before it that `cfg` keeps is refused at its type, in the words
/// it has with no `cfg`; one that `cfg` leaves out takes no room, nor does one that is refused, so a
/// small table after it fits. Without the feature `new` the fixture's `plain` is refused alone, at its
/// second table; with it, so are `kept`, at its second table, and `kept_global`, of global memory. `either`
/// is not, whichever `cfg` leaves out, `cfg_attr(..)` too.
#[test]
fn a_static_with_no_room_beside_those_cfg_keeps_is_refused_at_its_type() {
    let constant = "this makes the module's constant memory 96000 bytes, and a module has 64 KB";
    let global = "this makes the module's global memory as much as 4294967302 bytes, each static aligned, and \
                  `ptxas` writes its size in 32 bits: a module has less than 4 GB";
    refuses(
        "room_refused",
        &[
            &format!("src/lib.rs:11:19: error: {constant}"),
            "error: could not compile `room_refused` (lib) due to 1 previous error",
        ],
    );
    refuses_by(
        &["build", "--features", "new"],
        "room_refused",
        &[
            &format!("src/lib.rs:11:19: error: {constant}"),
            &format!("src/lib.rs:21:19: error: {constant}"),
            &format!("src/lib.rs:30:19: error: {global}"),
            "error: could not compile `room_refused` (lib) due to 3 previous errors",
        ],
    );
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
    // The modules in order: `head`, `twice` and `blocks`.
    refuses(
        "configured_module",
        &[
            "src/lib.rs:9:43: error: a function's generics are values, as in `<const N: usize>`",
            "src/lib.rs:23:35: error: `k` names another function of the module as well",
            "src/lib.rs:31:7: error: `#[cooperative]` gives `Kernels` a method `k_blocks`, which says how many \
             blocks a grid of `k` has at most, and a kernel or a static of the module is named `k_blocks` as well",
            "error: could not compile `configured_module` (lib) due to 3 previous errors",
        ],
    );
}

/// Kernels and functions that no one PTX module can hold are refused where `cfg` keeps them, and
/// where it keeps all of a set of them, at the module's first GPU item, kept or not: one error, in
/// the words of the module of those alone, whichever else `cfg` keeps. The fixture's items are
/// `first`, the first, which needs nothing, `old`, which is for GPUs older than `sm_70` from PTX 6.4
/// on, `mid` and `top`, which need `sm_75` and `sm_80`, and the function `fast`, which needs
/// `sm_75`, each under a feature; the one kernel that calls `fast` is left out by every `cfg`, and
/// the error of `old` and `fast` kept names those two and not that kernel. The error is at `first`
/// whether its feature keeps it or not.
#[test]
fn kernels_that_no_one_module_holds_are_refused_where_cfg_keeps_them() {
    let dir = written("configured_clash");
    let old = "`shfl.idx.b32` in `old` is for GPUs older than sm_70 from PTX 6.4 on, and";
    let with_mid = format!("{old} `tanh.approx.f32` in `mid` needs sm_75 and PTX 7.0: no module has both");
    let with_top =
        format!("{old} `redux.sync.add.u32` in `top` needs sm_80, which PTX has from 7.0 on: no module has both");
    let with_fast = format!("{old} `tanh.approx.f32` in `fast` needs sm_75 and PTX 7.0: no module has both");
    let cases = [
        ("", None),
        ("old", None),
        ("mid", None),
        ("top", None),
        ("mid,top", None),
        ("fast", None),
        ("mid,fast", None),
        ("old,mid", Some(&with_mid)),
        ("old,top", Some(&with_top)),
        ("old,fast", Some(&with_fast)),
        ("old,mid,top", Some(&with_mid)),
        ("first,old,mid", Some(&with_mid)),
        ("first,old,fast", Some(&with_fast)),
    ];
    for (features, said) in cases {
        let command = ["build", "--features", features];
        let Some(why) = said else {
            builds_quietly_in(&command, "configured_clash", &dir);
            continue;
        };
        let (built, lines) = cargo_in(&command, &dir);
        let want = [
            format!("src/lib.rs:13:39: error: {why}"),
            "error: could not compile `configured_clash` (lib) due to 1 previous error".to_string(),
        ];
        assert!(
            !built && lines == want,
            "fixtures/configured_clash.rs: {command:?}: Cargo said:\n{}",
            lines.join("\n")
        );
    }
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
/// leaves out, though the pair names a const `cfg` leaves out with that helper. Rust says nothing of a name it
/// cannot find in a module where kernite says an error, as each error is an item of the module. So Rust says
/// none of the names in the fixture's modules that have kernite's errors. The fixture's modules are, in order,
/// the x86 kernels Rust has names for, those it has none for, those that call helpers, the GPU's kernels, two
/// of which nothing is said: one whose kernels `cfg` leaves out with what they name, or keeps only where it
/// keeps that, and one that the features keep as they keep what it names; and the helpers that loop: one that
/// calls itself, a pair, a pair `cfg` leaves out, and a pair that only a helper `cfg` leaves out calls. Last
/// are three where kernite says no error. In the first two, Rust says the names in a helper and in one that
/// loops, on x86, and in the type of a static, a function, a helper, a kernel and a directive, on a GPU. Of the
/// third, nothing is said without `big`, with which the next test builds.
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
            // the order of the source. Only the modules where kernite says no error have them.
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
            &no_value("426:23", "FOUR"),
            &no_value("430:23", "STEP"),
            &no_value("455:37", "LEN"),
            &no_value("460:23", "SIZE"),
            &no_value("471:28", "SIZE"),
            &no_value("466:20", "SIZE"),
            &no_value("484:15", "SIZE"),
            "error: could not compile `configured_names` (lib) due to 31 previous errors",
        ],
    );
}

/// What is told is judged by the conditions of both, the kernel's and the item's. With `big`, an item that
/// `big` and `wide` keep, and a helper under `cfg_attr(big, cfg(wide))`, are gone where `big` holds alone,
/// and the kernels that name them are told there, a kernel that `big` keeps and one that is always kept;
/// not where neither holds, nor where both do. Rust says the const's name, `WIDE`, in the last module, where
/// kernite says no error.
#[test]
fn a_name_is_told_where_the_kernel_is_kept_and_the_item_is_not() {
    refuses_by(
        &["build", "--features", "big"],
        "configured_names",
        &[
            &left_out("320:9", "six", "calls_six"),
            &left_out("324:9", "five", "calls_five"),
            &no_value("499:23", "WIDE"),
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
/// the module the path names: the fixture asserts, as it builds, which `STEP` each sees. A name may
/// be on the path twice, or be a raw name, and a module the path goes through may have a `sizes` of its own.
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
/// argument gives, alone or in braces, a result, in braces as well, a type of the crate's that takes
/// it, with one argument, two and a `bool`, in a kernel that is cooperative, and in one that calls an
/// instance, of any arguments. A GPU kernel's launcher takes it as before. Rust builds the crate
/// without a word, and the tests the fixture holds run each x86 kernel, which writes through its
/// pointer. So does its doc test, from another crate.
#[test]
fn a_type_in_a_kernel_may_name_its_compile_time_arguments() {
    let printed = runs_quietly(&["test"], "argument_pointers");
    for passed in ["test result: ok. 10 passed", "test result: ok. 1 passed"] {
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

/// A type that begins with a binder, `for<'a> fn(&'a u8)`, is read whole, as any other type is: it was once
/// taken for a type named `for` with an argument, and the reading stopped at the `fn`. A kernel that takes one
/// by value, or as the argument of a type, or returns one, is refused at the type, in its machine's words, on
/// x86 and on a GPU; the type is quoted as Rust writes it, with a space after the binder. A pointer to one is
/// refused in the lexer's words, as a pointer to any type with a lifetime is, and a pointer to a function's
/// type that has no binder, `*const fn(&u8)`, is taken.
#[test]
fn a_type_that_begins_with_a_binder_is_read_whole() {
    refuses(
        "binders",
        &[
            "src/lib.rs:10:47: error: a kernel takes pointers, 64-bit integers and floats, not `for<'a> fn(&'a u8)`",
            "src/lib.rs:14:48: error: a kernel takes pointers, 64-bit integers and floats, not `Option<for<'a> \
             fn(&'a u8)>`",
            "src/lib.rs:18:56: error: a kernel takes pointers, 64-bit integers and floats, not `Option<&dyn \
             for<'a> Fn(&'a u8)>`",
            "src/lib.rs:22:55: error: a kernel returns a float, an integer or a pointer, or never, `!`; not \
             `for<'a> fn(&'a u8)`",
            "src/lib.rs:26:49: error: lifetimes and labels have no place in a kernel",
            "src/lib.rs:30:64: error: lifetimes and labels have no place in a kernel",
            // The pointer to a function's type with no binder, at line 34, has no error: it is taken.
            "src/lib.rs:41:51: error: a GPU kernel takes pointers, integers and the floats `f16`, `bf16`, `f32` and \
             `f64`, not `for<'a> fn(&'a u8)`",
            "src/lib.rs:45:53: error: lifetimes and labels have no place in a kernel",
            "error: could not compile `binders` (lib) due to 8 previous errors",
        ],
    );
}

/// A pointer to a type with no size, which Rust passes in two registers where a kernel's is one, is
/// one error, at the type, whatever the type is called. kernite tells `[f32]` and `str` by their text,
/// `str` by its path too, `core::primitive::str`, and says so in its words, on x86 and on a GPU. Rust
/// tells the others, and says that the size cannot be known: `CStr`, a name for `[f32]`, a struct of
/// the crate's whose last field is one, and a tuple that ends in `CStr`. It does so in a kernel of
/// every kind: an x86 kernel, plain, of compile-time arguments, cooperative, and cooperative of
/// compile-time arguments, and a GPU kernel of each of these kinds; for two such pointers in a kernel,
/// each, and for a pointer a kernel returns, plain or of compile-time arguments. A kernel kernite
/// refuses for another reason leaves a stub that asks Rust as well, so that it is two errors,
/// kernite's and Rust's, each where it is: an x86 kernel, plain, of compile-time arguments or
/// cooperative, and one that returns a pointer; one that is not `unsafe`; one whose `extern` kernite
/// does not know; and a GPU kernel, plain or cooperative, and one that returns a pointer, plain or of
/// compile-time arguments. An array or a tuple that holds a type with no size, `[CStr; 3]`,
/// `(CStr, u8)` or `(u8, [CStr; 3])`, Rust refuses by itself, at the type, before it asks of the
/// pointer: it says so once, and nothing asks again, in a kernel of every kind, a pointer that a
/// kernel returns too, so that the errors Cargo counts are the lines it prints. One case has two, at
/// two places: a kernel of compile-time arguments casts back a pointer it returns, so for
/// `(u8, [CStr; 3])` Rust says the array in the signature, and the tuple at the cast.
#[test]
fn a_pointer_to_a_type_with_no_size_is_refused_at_its_type() {
    // What kernite says, in the order of the kernels: the x86 module's, then the GPU's.
    let told = "error: a pointer to `[f32]` has two words, an address and a length or a table, \
                and a kernel's pointer has one";
    let told_str = "error: a pointer to `str` has two words, an address and a length or a table, \
                    and a kernel's pointer has one";
    let told_str_first = "error: in `str_path_tiles::<1>`: a pointer to `str` has two words, an address and a length \
                          or a table, and a kernel's pointer has one";
    let told_str_second = "error: in `str_path_tiles::<2>`: a pointer to `str` has two words, an address and a length \
                           or a table, and a kernel's pointer has one";
    let y = "error: `y` is not defined";
    let safe = "error: a kernel trusts the pointers it is given, so it is an `unsafe fn`";
    let abi = "error: kernite compiles `extern \"sysv64\"` kernels, for x86-64, and `extern \"ptx-kernel\"` ones, \
               for NVIDIA GPUs, with the `extern \"ptx-func\"` functions they call; `extern \"C\"` is none of these";
    let gives = "error: a GPU kernel returns nothing, not `*mut core::ffi::CStr`: it writes what it works out \
                 to memory";
    let gives_first = "error: in `returns_c_str_tiles::<1>`: a GPU kernel returns nothing, not \
                       `*mut core::ffi::CStr`: it writes what it works out to memory";
    let gives_second = "error: in `returns_c_str_tiles::<2>`: a GPU kernel returns nothing, not \
                        `*mut core::ffi::CStr`: it writes what it works out to memory";
    let kernite = [
        // `str` by its path: a plain kernel, one of compile-time arguments, a cooperative one.
        ("36:47", told_str),
        ("41:67", told_str_first),
        ("41:67", told_str_second),
        ("47:56", told_str),
        ("85:44", told),
        ("111:23", y),
        ("116:23", "error: in `refused_tiles::<1>`: `y` is not defined"),
        ("116:23", "error: in `refused_tiles::<2>`: `y` is not defined"),
        ("121:23", y),
        ("125:23", y),
        ("130:23", "error: in `refused_returns_tiles::<1>`: `y` is not defined"),
        ("130:23", "error: in `refused_returns_tiles::<2>`: `y` is not defined"),
        ("133:28", safe),
        ("137:23", abi),
        ("216:48", told),
        ("241:28", y),
        ("246:28", y),
        ("284:69", gives),
        ("289:89", gives_first),
        ("289:89", gives_second),
    ];
    // Where Rust says it, and of which type: at the pointer's type, in the order Rust says it.
    let sizes = [
        // An array of `CStr`, a tuple with `CStr` before its last field, and a tuple that ends in an
        // array of `CStr`: Rust refuses the types, before the pointers. In an x86 kernel: plain, of
        // compile-time arguments, cooperative, and one that returns a pointer, plain and of arguments.
        ("141:60", "[i8]"),
        ("146:80", "[i8]"),
        ("152:69", "[i8]"),
        ("156:82", "[i8]"),
        ("163:15", "[i8]"),
        ("168:63", "[i8]"),
        ("173:83", "[i8]"),
        ("179:72", "[i8]"),
        ("183:70", "[i8]"),
        ("189:24", "[i8]"),
        ("197:79", "[i8]"),
        ("201:92", "[i8]"),
        // The array of a tuple that a kernel of compile-time arguments returns a pointer to, in its
        // signature; the tuple is said below, where the pointer is cast back.
        ("208:22", "[i8]"),
        // And in a GPU kernel: plain, of compile-time arguments, cooperative, and cooperative of
        // compile-time arguments.
        ("249:62", "[i8]"),
        ("254:84", "[i8]"),
        ("259:71", "[i8]"),
        ("263:65", "[i8]"),
        ("268:85", "[i8]"),
        ("272:72", "[i8]"),
        ("279:22", "[i8]"),
        // `CStr`: a plain kernel, one of compile-time arguments, a cooperative one.
        ("21:51", "[i8]"),
        ("26:71", "[i8]"),
        ("32:60", "[i8]"),
        // `Floats`, a name for `[f32]`.
        ("51:52", "[f32]"),
        ("56:72", "[f32]"),
        ("62:61", "[f32]"),
        // `Tail`, a struct whose last field is `[f32]`.
        ("66:50", "[f32]"),
        ("71:70", "[f32]"),
        ("77:59", "[f32]"),
        // Two pointers of one kernel, the second before the first, as Rust says them, and a pointer a
        // kernel returns, plain and of arguments.
        ("81:74", "[f32]"),
        ("81:49", "[i8]"),
        ("89:67", "[i8]"),
        ("94:87", "[i8]"),
        // A tuple that ends in `CStr`: a plain kernel, and a cooperative one of arguments.
        ("99:51", "[i8]"),
        ("105:80", "[i8]"),
        // An x86 kernel kernite refuses: a plain one, one of arguments, a cooperative one, a plain one
        // and one of arguments that return a pointer, one that is not `unsafe`, and one whose
        // `extern` kernite does not know.
        ("110:53", "[i8]"),
        ("115:73", "[i8]"),
        ("120:62", "[i8]"),
        ("124:75", "[i8]"),
        ("129:93", "[f32]"),
        ("133:43", "[i8]"),
        ("137:48", "[i8]"),
        // The tuple of that kernel, where the pointer is cast back.
        ("208:17", "[i8]"),
        // A GPU kernel: a plain one, one of compile-time arguments, a cooperative one, and a
        // cooperative one of arguments.
        ("220:53", "[i8]"),
        ("225:72", "[f32]"),
        ("230:62", "[i8]"),
        ("236:81", "[f32]"),
        // A GPU kernel kernite refuses: a plain one, and a cooperative one.
        ("240:55", "[i8]"),
        ("245:64", "[i8]"),
        // And one that returns a pointer, plain and of compile-time arguments.
        ("284:74", "[i8]"),
        ("289:94", "[i8]"),
    ];
    let mut said: Vec<String> = kernite.map(|(at, what)| format!("src/lib.rs:{at}: {what}")).into();
    said.extend(sizes.map(|(at, ty)| size_unknown(at, ty)));
    let errors = kernite.len() + sizes.len();
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

/// A pointer to a type Rust refuses by itself, `NeedsCopy<String>` where the struct asks for an argument
/// that is `Copy`, is Rust's error, said once, at the type, as any type's is: in a kernel of every kind,
/// and not again at the module's attribute, nor once for each instance. So it is in an x86 kernel, plain,
/// of compile-time arguments, cooperative, and cooperative of compile-time arguments; in one that
/// returns the pointer, plain or of compile-time arguments; in one with two such pointers, each said at
/// its own; in an x86 kernel kernite refuses, plain or of compile-time arguments, where kernite's error
/// comes first; and in a GPU kernel, plain, of compile-time arguments, cooperative, and cooperative of
/// compile-time arguments, and one kernite refuses.
#[test]
fn a_pointer_to_a_type_rust_refuses_by_itself_is_said_once_at_its_type() {
    let y = "error: `y` is not defined";
    // What kernite says, in the order of the kernels: the x86 module's, then the GPU's.
    let kernite = [
        ("51:23", y),
        ("56:23", "error: in `refused_tiles::<1>`: `y` is not defined"),
        ("56:23", "error: in `refused_tiles::<2>`: `y` is not defined"),
        ("83:28", y),
    ];
    // Where Rust says it, and of which type: at the pointer's type, in the order Rust says it.
    let bounds = [
        // An x86 kernel: plain, of compile-time arguments, cooperative, and cooperative of
        // compile-time arguments.
        ("14:51", "String"),
        ("19:63", "String"),
        ("25:54", "String"),
        ("31:72", "String"),
        // One that returns the pointer, plain and of compile-time arguments.
        ("36:67", "String"),
        ("41:85", "String"),
        // One with two such pointers.
        ("46:49", "String"),
        ("46:83", "Vec<u8>"),
        // One kernite refuses: plain and of compile-time arguments.
        ("50:53", "String"),
        ("55:73", "String"),
        // A GPU kernel: plain, of compile-time arguments, cooperative, and cooperative of
        // compile-time arguments.
        ("62:53", "String"),
        ("67:69", "String"),
        ("72:56", "String"),
        ("78:76", "String"),
        // One kernite refuses.
        ("82:55", "String"),
    ];
    let mut said: Vec<String> = kernite.map(|(at, what)| format!("src/lib.rs:{at}: {what}")).into();
    said.extend(bounds.map(|(at, ty)| not_copy(at, ty)));
    let errors = kernite.len() + bounds.len();
    said.push(format!("error: could not compile `pointer_bound` (lib) due to {errors} previous errors"));
    refuses("pointer_bound", &said.iter().map(String::as_str).collect::<Vec<_>>());
}

/// A pointer to a type with a `_` in it, `*const [_; 3]` or `*mut (u8, _)`, is Rust's error alone: Rust does not
/// allow a `_` in a signature, and says so once, at the `_`. It does not say as well that it cannot tell the type
/// the `_` stands for, at the module's attribute or at the result, nor is the pointer asked whether it has a size.
/// So it is in an x86 kernel, plain, of compile-time arguments, cooperative, and one kernite refuses, where
/// kernite's error comes first; for a parameter, and for a result, `-> *const _` and `-> *mut (u8, _)`; and in a
/// GPU kernel, plain, of compile-time arguments, cooperative, and one kernite refuses. A cooperative x86 kernel's
/// result and a GPU kernel's are kernite's error, and Rust says its own beside it, at the `_`, as it does of a
/// plain kernel's. So is a `_` in a result that is no pointer, `-> [_; 3]`, beside kernite's error, and a `_` that
/// is an array's whole length, `*const [u8; _]`, in an x86 kernel's parameter and result.
#[test]
fn a_placeholder_in_a_pointers_type_is_rusts_error_alone_at_the_placeholder() {
    let y = "error: `y` is not defined";
    // What kernite says, in the order of the kernels: the x86 module's, then the GPU's.
    let kernite = [
        ("48:69", "error: a cooperative kernel's threads return nothing: they write what they compute to memory"),
        ("53:23", y),
        ("57:23", y),
        ("61:23", y),
        ("64:66", "error: a kernel returns a float, an integer or a pointer, or never, `!`; not `[_; 3]`"),
        ("85:63", "error: a GPU kernel returns nothing, not `*mut [_; 3]`: it writes what it works out to memory"),
        ("90:28", y),
    ];
    // Where Rust says it, and of what: at the `_`, in the order Rust says it.
    let holes = [
        // An x86 kernel: plain, with a parameter and with each result.
        ("12:52", "functions"),
        ("16:73", "return types"),
        ("20:81", "return types"),
        // Of compile-time arguments.
        ("25:66", "functions"),
        ("31:87", "return types"),
        ("37:95", "return types"),
        // Cooperative: a parameter, and a result, where kernite's error is beside Rust's.
        ("43:55", "functions"),
        ("48:76", "return types"),
        // One kernite refuses.
        ("52:54", "functions"),
        ("56:75", "return types"),
        ("60:83", "return types"),
        // A result that is no pointer.
        ("64:67", "return types"),
        // An array's whole length, `[u8; _]`: a parameter and a result.
        ("97:56", "functions"),
        ("101:78", "return types"),
        // A GPU kernel: plain, of compile-time arguments, cooperative, with a result, and one kernite
        // refuses.
        ("71:58", "methods"),
        ("76:74", "methods"),
        ("81:61", "methods"),
        ("85:69", "return types"),
        ("89:60", "methods"),
    ];
    let mut said: Vec<String> = kernite.map(|(at, what)| format!("src/lib.rs:{at}: {what}")).into();
    said.extend(holes.map(|(at, what)| not_allowed(at, what)));
    let errors = kernite.len() + holes.len();
    said.push(format!("error: could not compile `placeholder` (lib) due to {errors} previous errors"));
    refuses("placeholder", &said.iter().map(String::as_str).collect::<Vec<_>>());
}

/// A `_` is Rust's placeholder in a type, and is no placeholder where it is a pattern or a name, or the macro's
/// own: in a block, `[u8; { let _ = 1; 3 }]` or `Arr<{ let _ = 1; 3 }>`; in an array's length that is no block,
/// `[u8; if let Some(_) = Some(1) { 3 } else { 4 }]`, which is an expression; among a macro's arguments,
/// `[u8; len!(_)]`, `byte!(_)` or `byte!{ _ }`, which the macro drops; and as the name of a function pointer's
/// parameter, `fn(_: u8)`, or `fn(_: ::core::primitive::u8)` where the parameter's type is a path from the root.
/// Rust takes each of these types in a signature, so a kernel that gives back a pointer to one is a kernel: the
/// crate builds without a word, under `#![deny(warnings)]`, and the test the fixture holds calls each x86 kernel,
/// plain and of compile-time arguments, which gives its pointer back, and the second adds its argument to it
/// first. A kernel taken for one with a placeholder is a stub, which aborts.
#[test]
fn a_result_with_an_underscore_that_is_no_placeholder_is_a_kernel_that_runs() {
    let printed = runs_quietly(&["test"], "not_placeholder");
    assert!(
        printed.contains("test result: ok. 1 passed"),
        "fixtures/not_placeholder.rs ran its test, and printed:\n{printed}"
    );
}

/// A result whose type holds a block among its arguments, `Arr<{ 3 }>`, is read whole as a parameter's type is: the
/// braces are a const argument, no body. Rust builds the crate and lints each pair of braces as it lints any, at the
/// braces, once, in a kernel that is plain, of compile-time arguments, whose block stands beside another argument or
/// in a type nested in another, and in a parameter and inside an array, which kernite read whole before. kernite says
/// nothing, and the tests the fixture holds call each x86 kernel: each that gives back a pointer gives back the one
/// it is given, moved by its argument if it has one, and the one that takes a pointer writes a byte through it.
#[test]
fn a_result_whose_type_holds_a_block_is_a_kernel_and_rust_lints_the_braces_alone() {
    let out = cargo_out(&["test", "--lib"], &written("block_result"));
    let (said, printed) = (String::from_utf8_lossy(&out.stderr), String::from_utf8_lossy(&out.stdout));
    assert!(out.status.success(), "fixtures/block_result.rs: its tests failed:\n{said}\n{printed}");
    assert!(printed.contains("test result: ok. 2 passed"), "fixtures/block_result.rs ran no tests:\n{printed}");
    let braces = "unnecessary braces around const expression";
    let at = ["20:76", "25:90", "30:82", "34:89", "38:64", "42:77"];
    assert_eq!(
        said.lines().collect::<Vec<_>>(),
        at.map(|at| format!("src/lib.rs:{at}: warning: {braces}")),
        "fixtures/block_result.rs"
    );
}

/// A kernel with a result that none of its kind may have, a cooperative kernel or a GPU kernel, which give back
/// nothing, plain or of compile-time arguments, or an x86 kernel whose result is no number or pointer, is refused at
/// the result when its type holds a block, as it is when it holds none, in the words of its kind.
#[test]
fn a_result_its_kind_may_not_have_is_refused_at_the_result_when_its_type_holds_a_block() {
    let nothing = "a cooperative kernel's threads return nothing: they write what they compute to memory";
    let gpu = "a GPU kernel returns nothing, not `*const crate::Arr<{ 3 }>`: it writes what it works out to memory";
    refuses(
        "block_result_refused",
        &[
            &format!("src/lib.rs:17:64: error: {nothing}"),
            &format!("src/lib.rs:23:84: error: in `cooperative_tiles::<1>`: {nothing}"),
            &format!("src/lib.rs:23:84: error: in `cooperative_tiles::<2>`: {nothing}"),
            "src/lib.rs:28:59: error: a kernel returns a float, an integer or a pointer, or never, `!`; not \
             `crate::Arr<{ 3 }>`",
            &format!("src/lib.rs:36:62: error: {gpu}"),
            &format!("src/lib.rs:41:76: error: in `tiles::<1>`: {gpu}"),
            &format!("src/lib.rs:41:76: error: in `tiles::<2>`: {gpu}"),
            "error: could not compile `block_result_refused` (lib) due to 7 previous errors",
        ],
    );
}

/// A kernel with a result that none of its kind may have, a cooperative kernel or a GPU kernel, which give back
/// nothing, plain or of compile-time arguments, or an x86 kernel whose result is no number or pointer, is refused at
/// the result when its type is a macro in braces, `byte!{ _ }`, as it is when it holds a block or no macro, in the
/// words of its kind.
#[test]
fn a_result_its_kind_may_not_have_is_refused_at_the_result_when_its_type_is_a_macro_in_braces() {
    let nothing = "a cooperative kernel's threads return nothing: they write what they compute to memory";
    let gpu = "a GPU kernel returns nothing, not `*const byte! { _ }`: it writes what it works out to memory";
    refuses(
        "type_macro_refused",
        &[
            &format!("src/lib.rs:17:64: error: {nothing}"),
            &format!("src/lib.rs:23:84: error: in `cooperative_tiles::<1>`: {nothing}"),
            &format!("src/lib.rs:23:84: error: in `cooperative_tiles::<2>`: {nothing}"),
            "src/lib.rs:28:59: error: a kernel returns a float, an integer or a pointer, or never, `!`; not \
             `byte! { _ }`",
            &format!("src/lib.rs:36:62: error: {gpu}"),
            &format!("src/lib.rs:41:76: error: in `tiles::<1>`: {gpu}"),
            &format!("src/lib.rs:41:76: error: in `tiles::<2>`: {gpu}"),
            "error: could not compile `type_macro_refused` (lib) due to 7 previous errors",
        ],
    );
}

/// The `!` of never follows the arrow, a keyword or a lifetime, and is no macro's: the braces after it are the body.
/// So a kernel whose result is a reference to never, `&'static !`, is refused at the result, a reference, and not for
/// the want of a body.
#[test]
fn a_result_that_is_a_reference_to_never_is_refused_at_the_result_and_its_braces_are_the_body() {
    refuses(
        "never_after_lifetime",
        &[
            "src/lib.rs:7:50: error: a kernel returns a float, an integer or a pointer, or never, `!`; not `&'static !`",
            "error: could not compile `never_after_lifetime` (lib) due to 1 previous error",
        ],
    );
}

/// A pointer to a type with no size, a tuple that ends in `CStr` or a struct whose last field is a slice, is Rust's
/// error at the type, said once, if the type holds a `_` that is no placeholder, as it is if the type holds none: a
/// pattern in a block, in an array's length, the name of a function pointer's parameter, and an argument of a macro,
/// as the test of the results says. The macros ask Rust of the pointer, and do not take the `_` for a placeholder,
/// which Rust refuses alone. So it is in an x86 kernel, plain, of compile-time arguments and cooperative, and in a GPU
/// kernel.
#[test]
fn a_pointer_with_no_size_and_an_underscore_that_is_no_placeholder_is_rusts_error_at_its_type() {
    let sizes = [
        // A pattern in a block, and an argument of a macro, in an array's length: an x86 kernel, plain, of
        // compile-time arguments and cooperative.
        ("32:51", "[i8]"),
        ("38:19", "[i8]"),
        ("46:60", "[i8]"),
        ("50:56", "[i8]"),
        ("55:76", "[i8]"),
        ("61:65", "[i8]"),
        // A pattern in an array's length, and a function pointer's parameter: a plain x86 kernel.
        ("66:19", "[i8]"),
        ("72:54", "[i8]"),
        // A pattern in a block that is a const argument, and a macro that gives the type: a plain x86 kernel.
        ("76:60", "[u8]"),
        ("80:56", "[i8]"),
        // And a GPU kernel, with each of the six.
        ("87:53", "[i8]"),
        ("91:58", "[i8]"),
        ("96:17", "[i8]"),
        ("102:56", "[i8]"),
        ("106:62", "[u8]"),
        ("110:58", "[i8]"),
    ];
    let mut said: Vec<String> = sizes.map(|(at, ty)| size_unknown(at, ty)).into();
    let errors = sizes.len();
    said.push(format!("error: could not compile `not_placeholder_no_size` (lib) due to {errors} previous errors"));
    refuses("not_placeholder_no_size", &said.iter().map(String::as_str).collect::<Vec<_>>());
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

/// A number written as a primitive's path, `core::primitive::u64`, is the number wherever a type is written, as Rust
/// has it: a module of kernels by the paths, in the four spellings Rust has, builds without a word, each kernel is
/// called as the same kernel by its names is, with the same result, and its listing, or its PTX, is the one of the
/// kernel by its names. The tests the fixture holds run the kernels, the GPU ones on the GPU, which they skip where
/// there is none.
#[test]
fn a_kernel_typed_by_primitive_paths_builds_and_runs_as_the_kernel_typed_by_names() {
    let printed = runs_quietly(&["test"], "primitive_paths");
    assert!(
        printed.contains("test result: ok. 4 passed"),
        "fixtures/primitive_paths.rs ran its tests, and printed:\n{printed}"
    );
}

/// A constant named by a primitive's path, `core::primitive::u32::MAX`, is the constant its name gives, `u32::MAX`, as
/// Rust has it: in a const, a static's data, a compile-time argument, the values an instance lists, and a kernel's
/// operand, in the four spellings Rust has. A module of kernels by the paths builds without a word, each kernel is
/// called as the same kernel by the names is, with the same result, and its listing, or its PTX, is the one of the
/// kernel by the names. The tests the fixture holds run the kernels, the GPU ones on the GPU, which they skip where
/// there is none.
#[test]
fn a_constant_named_by_primitive_paths_builds_and_gives_the_value_the_names_give() {
    let printed = runs_quietly(&["test"], "primitive_paths_values");
    assert!(
        printed.contains("test result: ok. 4 passed"),
        "fixtures/primitive_paths_values.rs ran its tests, and printed:\n{printed}"
    );
}

/// A path that the place does not take is kernite's error alone, at the type, in the words it has for the same type by
/// its name; a primitive is named by its name, and any other path is quoted as written. Rust says of a compile-time
/// argument's type what it says of the name, once, at the type: it is no type for a const generic. It says nothing of
/// a kernel kernite refuses, as it says nothing of the name. So it is of a compile-time argument, a parameter, a
/// result and a variable of an x86 kernel and a GPU kernel, in the four kinds of type: a primitive the place does not
/// take, a type of the crate's, a `type` of the crate's for `u64`, and a path that names no primitive.
#[test]
fn a_path_that_names_no_type_the_place_takes_is_kernites_error_alone_at_the_type() {
    let compile_time = |at: &str, ty: &str| {
        format!(
            "src/lib.rs:{at}: error: a function that names an ABI takes integers and `bool`s at compile time, as \
             Rust's const generics are; `{ty}` is neither"
        )
    };
    let x86_variable = |at: &str, ty: &str| {
        format!(
            "src/lib.rs:{at}: error: x86 variables are `gpr`, `ymm`, `xmm` and stack memory, `local<..>`; not `{ty}`"
        )
    };
    let gpu_variable = |at: &str, ty: &str| {
        format!(
            "src/lib.rs:{at}: error: PTX variables are `pred`, `b8` to `b128`, `u8` to `u64`, `s8` to `s64`, `f16`, \
             `f16x2`, `bf16`, `bf16x2`, `tf32`, `f32` and `f64`, not `{ty}`"
        )
    };
    let forbidden = |at: &str, ty: &str| {
        format!("src/lib.rs:{at}: error: `{ty}` is forbidden as the type of a const generic parameter")
    };
    let mut said = vec![
        // x86, by the paths.
        compile_time("31:49", "f32"),
        compile_time("35:50", "crate::Word"),
        x86_takes("38:48", "", "str"),
        x86_takes("39:49", "", "char"),
        x86_takes("40:49", "", "u128"),
        x86_takes("41:49", "", "crate::Word"),
        x86_takes("42:49", "", "crate::Wide"),
        x86_takes("43:48", "", "core::primitive::u64::MAX"),
        x86_takes("44:47", "", "my::core::primitive::u64"),
        x86_takes("45:48", "", "core::primitive::f16"),
        x86_gives("46:75", "", "i128"),
        x86_variable("48:17", "str"),
        // x86, by the names.
        compile_time("58:49", "f32"),
        compile_time("62:50", "Word"),
        x86_takes("65:48", "", "str"),
        x86_takes("66:49", "", "char"),
        x86_takes("67:49", "", "u128"),
        x86_takes("68:49", "", "Word"),
        x86_takes("69:49", "", "Wide"),
        x86_gives("70:58", "", "i128"),
        x86_variable("72:17", "str"),
        // A GPU, by the paths.
        compile_time("80:53", "f32"),
        gpu_takes("83:52", "", "str"),
        gpu_takes("84:53", "", "char"),
        gpu_takes("85:53", "", "u128"),
        gpu_takes("86:53", "", "crate::Word"),
        gpu_takes("87:53", "", "crate::Wide"),
        gpu_takes("88:52", "", "core::primitive::f16"),
        gpu_returns("89:79", "", "i128"),
        gpu_variable("91:16", "char"),
        // A GPU, by the names.
        compile_time("101:53", "f32"),
        gpu_takes("104:52", "", "str"),
        gpu_takes("105:53", "", "char"),
        gpu_takes("106:53", "", "u128"),
        gpu_takes("107:53", "", "Word"),
        gpu_takes("108:53", "", "Wide"),
        gpu_returns("109:62", "", "i128"),
        gpu_variable("111:16", "char"),
        // Rust's, of a compile-time argument's type, by the paths and by the names.
        forbidden("31:52", "f32"),
        forbidden("35:53", "Word"),
        forbidden("58:52", "f32"),
        forbidden("62:53", "Word"),
        forbidden("80:56", "f32"),
        forbidden("101:56", "f32"),
    ];
    said.push(format!(
        "error: could not compile `primitive_paths_refused` (lib) due to {} previous errors",
        said.len()
    ));
    refuses("primitive_paths_refused", &said.iter().map(String::as_str).collect::<Vec<_>>());
}

/// A reference with a lifetime to a primitive by its path, `&'static ::core::primitive::u8`, is `&'static u8` to
/// kernite: a lifetime's name begins no path, so the path after it is a primitive's, from the root or not. A kernel
/// takes no reference, so kernite refuses it in its own words, by the primitive's name, as it does the same type
/// written by the name, and Rust says no more. So it is of a lifetime that a binder declares,
/// `for<'a> fn(&'a ::std::primitive::u8) -> &'a ::std::primitive::u16`, on x86 and on a GPU.
#[test]
fn a_primitive_written_as_a_path_after_a_lifetime_is_the_primitive_in_kernites_error() {
    let mut said = vec![
        // x86, by the paths and by the names.
        x86_takes("9:45", "", "&'static u8"),
        x86_takes("10:44", "", "&'static u8"),
        x86_takes("11:44", "", "&'static u8"),
        x86_takes("12:44", "", "for<'a> fn(&'a u8) -> &'a u16"),
        x86_takes("13:50", "", "for<'a> fn(&'a u8) -> &'a u16"),
        // A GPU, by the paths and by the names.
        gpu_takes("19:49", "", "&'static str"),
        gpu_takes("20:48", "", "&'static str"),
        gpu_takes("21:48", "", "&'static str"),
    ];
    said.push(format!(
        "error: could not compile `primitive_paths_lifetimes` (lib) due to {} previous errors",
        said.len()
    ));
    refuses("primitive_paths_lifetimes", &said.iter().map(String::as_str).collect::<Vec<_>>());
}

/// A module that gives `core` or `std` another meaning, `use crate::fake as core;`, has Rust say that a number written
/// as its path is not the number, as it says it of the number's name that a module gives another type: at the type,
/// once, for each parameter and result of an x86 kernel and each parameter of a GPU kernel. A pointer's type is not
/// asked of, as it is not for the name, and a path from the root, `::core::primitive::u64`, is the crate `core`
/// whatever the module gives, so Rust says nothing of it.
#[test]
fn a_number_written_as_a_path_through_a_core_the_module_gives_another_meaning_is_rusts_error_at_the_type() {
    let mut said: Vec<String> = [
        // x86, by the paths: a parameter and a result, a float, and a float whose result is from the root.
        ("31:43", "u64", "u32"),
        ("31:68", "u64", "u32"),
        ("34:45", "f32", "f64"),
        // x86, by the names: the same.
        ("50:43", "u64", "u32"),
        ("50:51", "u64", "u32"),
        ("53:45", "f32", "f64"),
        ("53:53", "f32", "f64"),
        // A GPU by the paths and by the names: a parameter.
        ("66:47", "u32", "u64"),
        ("77:47", "u32", "u64"),
    ]
    .map(|(at, number, found)| not_the_number(at, number, found))
    .into();
    said.push(format!(
        "error: could not compile `primitive_paths_shadowed` (lib) due to {} previous errors",
        said.len()
    ));
    refuses("primitive_paths_shadowed", &said.iter().map(String::as_str).collect::<Vec<_>>());
}

/// The types that a helper's or an `extern "ptx-func"` function's signature names are shown to Rust whole where a
/// kernel does not compile, when they are written as primitives' paths: `::core::primitive::u64` in a compile-time
/// argument, a parameter and a result, behind a pointer, in an array and in a tuple; and a path to a type of the
/// crate's, `crate::Word`. Rust says nothing of them, as it says nothing of the same types by their names. kernite
/// says its own error once for each module, at the kernel that does not compile. Some lists of compile-time
/// arguments end with a comma, which is no type to show Rust.
#[test]
fn rust_says_nothing_of_a_helper_or_a_function_typed_by_primitive_paths() {
    refuses(
        "primitive_paths_helpers",
        &[
            "src/lib.rs:16:13: error: `y` is not defined",
            "src/lib.rs:39:13: error: `y` is not defined",
            "src/lib.rs:59:28: error: `y` is not defined",
            "src/lib.rs:88:28: error: `y` is not defined",
            "error: could not compile `primitive_paths_helpers` (lib) due to 4 previous errors",
        ],
    );
}

/// An x86 kernel takes and gives back no `f16`, which only a GPU has: kernite refuses it at the type, once
/// for each instance. Rust says no more. It does not say that its `f16` is unstable, as it says wherever
/// the type is named, though the macros leave a stub for the kernel. Nor does it say, at a call of the
/// kernel, that the `f16` the kernel gives back is not the number the caller reads. So it is in a parameter
/// and in a result, of a kernel plain, of compile-time arguments, cooperative, and cooperative of
/// compile-time arguments, and at a call of each. So it is of a `bf16`, which Rust has not.
#[test]
fn an_x86_kernel_that_takes_or_gives_back_a_half_is_kernites_error_alone() {
    let mut said = vec![];
    for (ty, [plain, tiles, together, together_tiles, result, tiles_result]) in [
        ("f16", ["8:44", "13:58", "18:47", "24:67", "28:53", "33:73"]),
        ("bf16", ["40:44", "45:58", "50:47", "56:67", "60:53", "65:73"]),
    ] {
        said.push(x86_takes(plain, "", ty));
        said.extend([1, 2].map(|n| x86_takes(tiles, &instance("tiles", n), ty)));
        said.push(x86_takes(together, "", ty));
        said.extend([1, 2].map(|n| x86_takes(together_tiles, &instance("together_tiles", n), ty)));
        said.push(x86_gives(result, "", ty));
        said.extend([1, 2].map(|n| x86_gives(tiles_result, &instance("tiles_result", n), ty)));
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

/// A GPU kernel takes an `f16` and a `bf16`, by value and by pointer, written plain or raw, `r#f16`: Rust builds
/// the crate without a word, the launcher takes kernite-cuda's two, and the tests the fixture holds run each
/// kernel and read back what it wrote. They skip where there is no GPU.
#[test]
fn a_gpu_kernel_takes_the_halves_as_kernite_cuda_has_them() {
    let printed = runs_quietly(&["test"], "halves_gpu");
    assert!(
        printed.contains("test result: ok. 2 passed"),
        "fixtures/halves_gpu.rs ran its tests, and printed:\n{printed}"
    );
}

/// A type an x86 kernel takes or gives back that kernite refuses is kernite's error alone, at the type, once
/// for each instance: `f128`, an `f16` in an array, a tuple or behind a reference, `!` as a parameter, and a
/// type with no size, `[u64]`, `str` or `dyn Fn()`. Rust does not say that the type is unstable or has no
/// size, as it says wherever the type is named, though the macros leave a stub for the kernel. Nor does it
/// say, at a call of the kernel, that a value is not the type, or that the result is not the type the caller
/// reads. So it is for a parameter, of a kernel plain, of compile-time arguments, cooperative, and
/// cooperative of compile-time arguments; for a result, of a kernel plain and of compile-time arguments; and
/// for several parameters, of which kernite says the first it refuses. A type of the crate's is refused so
/// too, alone or beside an `f16`, though the module imports it. So is a type with an `f16` and a `!` deep in
/// it, `Option<fn(Vec<f16>) -> !>`, and one a macro writes, `gone!()`, whose `!` is not the type. So is a type
/// that holds tokens that are no types, which the macros leave as written: a `!` that is an operator, in an
/// array's length, `[u8; !0usize & 3]` and `[u8; (1 != 2) as usize]`, or in a block, `Marker<{ !0usize & 3 }>`;
/// and an `f16` or an `f128` that is a name, among a macro's arguments, `named!(f16)`, as a function pointer's
/// parameter, `fn(f16: u8)`, or `fn(f16: ::core::primitive::u8)` where its type is a path from the root, as the
/// first segment of a path, `f128::X` for a module named so, and `f16::X` and `f128::X` where no module so named
/// is in scope, as a later segment of a path, `core::primitive::f16`, as the name of a macro, `f16!()`, or as
/// the whole length of an array, a const's name, `[u8; f16]`, which kernite says is not defined; of a kernel
/// plain and of compile-time arguments. kernite says the parameter's type `::core::primitive::u8` as it says
/// `u8`, so both `fn` types are `fn(f16: u8)` in its words. So is a type of the crate's named `f16` or `f128`,
/// which the module imports, taken as `f16`, `f128`, `[f16; 4]` or `self::f16`: Rust finds the type, so it does
/// not call the import unused.
#[test]
fn an_x86_kernel_that_takes_or_gives_back_a_type_kernite_refuses_is_kernites_error_alone() {
    let mut said = vec![];
    for (ty, [plain, tiles, together, together_tiles]) in [
        ("f128", ["9:44", "14:58", "19:47", "25:67"]),
        ("[f16; 4]", ["32:44", "37:58", "42:47", "48:67"]),
        ("(f16, u64)", ["55:44", "60:58", "65:47", "71:67"]),
        ("&f16", ["78:44", "83:58", "88:47", "94:67"]),
        ("!", ["101:44", "106:58", "111:47", "117:67"]),
        ("[u64]", ["124:44", "129:58", "134:47", "140:67"]),
        ("str", ["147:44", "152:58", "157:47", "163:67"]),
        ("dyn Fn()", ["170:44", "175:58", "180:47", "186:67"]),
    ] {
        said.push(x86_takes(plain, "", ty));
        said.extend([1, 2].map(|n| x86_takes(tiles, &instance("tiles", n), ty)));
        said.push(x86_takes(together, "", ty));
        said.extend([1, 2].map(|n| x86_takes(together_tiles, &instance("together_tiles", n), ty)));
    }
    said.extend([
        x86_takes("195:52", "", "f128"),
        x86_takes("208:44", "", "Params"),
        x86_takes("219:44", "", "(Params, f16)"),
    ]);
    for (ty, [plain, tiles]) in [("f128", ["226:52", "231:66"]), ("[u64]", ["238:52", "243:66"])] {
        said.push(x86_gives(plain, "", ty));
        said.extend([1, 2].map(|n| x86_gives(tiles, &instance("tiles", n), ty)));
    }
    said.extend([x86_takes("251:44", "", "Option<fn(Vec<f16>) -> !>"), x86_takes("265:44", "", "gone!()")]);
    // kernite says an array's length as it works it out, a block as its source has it, and a parameter's colon
    // as Rust writes it.
    for (ty, [plain, tiles]) in [
        ("[u8; 3]", ["282:44", "287:58"]),
        ("[u8; 1]", ["295:44", "300:58"]),
        ("Marker<{ !0usize & 3 }>", ["310:44", "315:58"]),
        ("named!(f16)", ["323:44", "328:58"]),
        ("fn(f16: u8)", ["342:44", "347:58"]),
        ("f128::X", ["357:44", "362:58"]),
        ("f16::X", ["370:44", "375:58"]),
        ("f128::X", ["384:44", "389:58"]),
        ("f16!()", ["403:44", "408:58"]),
    ] {
        said.push(x86_takes(plain, "", ty));
        said.extend([1, 2].map(|n| x86_takes(tiles, &instance("tiles", n), ty)));
    }
    // A const the module does not hold is no value to kernite, which says so at the name.
    said.push("src/lib.rs:423:49: error: `f16` is not defined".to_string());
    // A type of the crate's named as an unstable type is, which the module imports, and what is written
    // beside it as a path or a parameter's name.
    for (ty, place) in [
        ("f16", "443:44"),
        ("f128", "453:44"),
        ("[f16; 4]", "463:44"),
        ("self::f16", "474:44"),
        ("core::primitive::f16", "482:44"),
        ("fn(f16: u8)", "491:44"),
    ] {
        said.push(x86_takes(place, "", ty));
    }
    said.push(format!("error: could not compile `types_refused` (lib) due to {} previous errors", said.len()));
    refuses("types_refused", &said.iter().map(String::as_str).collect::<Vec<_>>());
}

/// A type a GPU kernel takes that kernite refuses is kernite's error alone, at the type, once for each
/// instance: `f128`, an `f16` in an array, `!`, and a type with no size, `[u64]` or `str`. Rust does not say
/// that the type is unstable or has no size, as it says wherever the type is named, though the macros leave a
/// launcher for the kernel that never runs. Nor does it say, at a call of the launcher, that a value is not
/// the type. So it is for a kernel plain, of compile-time arguments, cooperative, and cooperative of
/// compile-time arguments; and for several parameters, of which kernite says the first it refuses. A type of
/// the crate's is refused so too, alone or beside an `f16`, though the module imports it. So is a type that
/// holds tokens that are no types, which the macros leave as written: `[u8; !0usize & 3]`,
/// `[u8; (1 != 2) as usize]`, `Marker<{ !0usize & 3 }>`, `named!(f16)`, `fn(f128: u8)`,
/// `fn(f128: ::core::primitive::u8)`, `f16::X` for a module named so, `f16::X` and `f128::X` where no module so
/// named is in scope, `core::primitive::f128`, `f16!()` for a macro named so, and `[u8; f128]`, whose `f128` is
/// a const's name, which kernite says is not defined. kernite says the parameter's type `::core::primitive::u8`
/// as it says `u8`, so both `fn` types are `fn(f128: u8)` in its words. So is a type of the crate's named `f16`
/// or `f128`, which the module imports, taken as `f128`, `[f16; 4]` or `self::f16`: Rust finds the type, so it
/// does not call the import unused.
#[test]
fn a_gpu_kernel_that_takes_a_type_kernite_refuses_is_kernites_error_alone() {
    let mut said = vec![];
    for (ty, [plain, tiles, together, together_tiles]) in [
        ("f128", ["9:48", "14:62", "19:51", "25:71"]),
        ("[f16; 4]", ["32:48", "37:62", "42:51", "48:71"]),
        ("!", ["55:48", "60:62", "65:51", "71:71"]),
        ("[u64]", ["78:48", "83:62", "88:51", "94:71"]),
        ("str", ["101:48", "106:62", "111:51", "117:71"]),
    ] {
        said.push(gpu_takes(plain, "", ty));
        said.extend([1, 2].map(|n| gpu_takes(tiles, &instance("tiles", n), ty)));
        said.push(gpu_takes(together, "", ty));
        said.extend([1, 2].map(|n| gpu_takes(together_tiles, &instance("together_tiles", n), ty)));
    }
    said.extend([
        gpu_takes("126:56", "", "f128"),
        gpu_takes("139:48", "", "Params"),
        gpu_takes("150:48", "", "(Params, f16)"),
        gpu_takes("167:48", "", "[u8; 3]"),
        gpu_takes("175:48", "", "[u8; 1]"),
        gpu_takes("185:48", "", "Marker<{ !0usize & 3 }>"),
        gpu_takes("193:48", "", "named!(f16)"),
        gpu_takes("207:48", "", "fn(f128: u8)"),
        gpu_takes("217:48", "", "f16::X"),
        gpu_takes("226:48", "", "f16::X"),
        gpu_takes("234:48", "", "f128::X"),
        gpu_takes("248:48", "", "f16!()"),
        "src/lib.rs:263:53: error: `f128` is not defined".to_string(),
        gpu_takes("283:48", "", "f128"),
        gpu_takes("293:48", "", "[f16; 4]"),
        gpu_takes("304:48", "", "self::f16"),
        gpu_takes("312:48", "", "core::primitive::f128"),
        gpu_takes("321:48", "", "fn(f128: u8)"),
    ]);
    said.push(format!("error: could not compile `types_refused_gpu` (lib) due to {} previous errors", said.len()));
    refuses("types_refused_gpu", &said.iter().map(String::as_str).collect::<Vec<_>>());
}

/// A type kernite refuses, written raw, is kernite's error alone, at the type, once for each instance, as it is
/// written plain: Rust reads `r#f16` as `f16`, so it says no more of the one than of the other, and kernite quotes
/// the type as Rust reads it, `f16`. So are `[r#f16; 4]`, `r#f128`, a result `r#f16`, and a pointer to a type with
/// no size, `*const r#str` and `*mut [r#u64]`, in an x86 kernel and in a GPU kernel, plain, of compile-time
/// arguments, cooperative, or both. So is a result that a cooperative kernel or a GPU kernel may not have, `r#f16`,
/// `r#f128` or `r#u64`: the launcher hands the type to Rust, which reads it as it reads the one written plain.
#[test]
fn a_type_kernite_refuses_written_raw_is_kernites_error_alone() {
    let two = |place: &str, kernel: &str, ty: &str| {
        format!(
            "src/lib.rs:{place}: error: {kernel}a pointer to `{ty}` has two words, an address and a length or a \
             table, and a kernel's pointer has one"
        )
    };
    let mut said = vec![];
    for (ty, [plain, tiles, together, together_tiles]) in [
        ("f16", ["9:44", "14:58", "19:47", "25:67"]),
        ("[f16; 4]", ["41:44", "46:58", "51:47", "57:67"]),
        ("f128", ["64:44", "69:58", "74:47", "80:67"]),
    ] {
        said.push(x86_takes(plain, "", ty));
        said.extend([1, 2].map(|n| x86_takes(tiles, &instance("tiles", n), ty)));
        said.push(x86_takes(together, "", ty));
        said.extend([1, 2].map(|n| x86_takes(together_tiles, &instance("together_tiles", n), ty)));
        if ty == "f16" {
            said.push(x86_gives("29:53", "", ty));
            said.extend([1, 2].map(|n| x86_gives("34:73", &instance("tiles_result", n), ty)));
        }
    }
    said.push(two("87:43", "", "str"));
    said.extend([1, 2].map(|n| two("92:58", &instance("words", n), "[u64]")));
    said.push(gpu_takes("99:48", "", "f128"));
    said.extend([1, 2].map(|n| gpu_takes("104:62", &instance("tiles", n), "f128")));
    said.push(gpu_takes("109:51", "", "f128"));
    said.extend([1, 2].map(|n| gpu_takes("115:71", &instance("together_tiles", n), "f128")));
    said.extend([gpu_takes("122:48", "", "[f16; 4]"), two("126:47", "", "str")]);
    said.push(x86_gives("134:55", "", "f16"));
    said.extend([1, 2].map(|n| x86_gives("140:75", &instance("together_tiles", n), "f128")));
    said.push(gpu_returns("144:67", "", "f16"));
    said.extend([1, 2].map(|n| gpu_returns("149:79", &instance("gpu_tiles", n), "f128")));
    said.extend([gpu_returns("154:76", "", "u64"), coop_returns("159:62", "")]);
    said.push(format!("error: could not compile `types_refused_raw` (lib) due to {} previous errors", said.len()));
    refuses("types_refused_raw", &said.iter().map(String::as_str).collect::<Vec<_>>());
}

/// A type kernite refuses whose name is a keyword written raw, `r#dyn::Foo`, `r#mut` or `r#for<u8>`, is kernite's
/// error alone, at the type, once for each instance, with the keyword as written, for `r#mut` is a name and `mut` is
/// not. So is a pointer to `[r#mut]`, which has two words; and a pointer to `r#str` has two as well, quoted `str`,
/// for `str` is no keyword. Rust says nothing of the type, nor at a call of the kernel, which the crate's own
/// function makes.
#[test]
fn a_type_named_by_a_keyword_written_raw_that_kernite_refuses_is_kernites_error_alone() {
    let two = |place: &str, ty: &str| {
        format!(
            "src/lib.rs:{place}: error: a pointer to `{ty}` has two words, an address and a length or a table, \
             and a kernel's pointer has one"
        )
    };
    let variable = "src/lib.rs:51:17: error: x86 variables are `gpr`, `ymm`, `xmm` and stack memory, `local<..>`; \
                    not `r#mut`";
    let mut said = vec![x86_takes("28:43", "", "r#dyn::Foo"), x86_takes("32:43", "", "r#mut")];
    said.extend([1, 2].map(|n| x86_takes("37:58", &instance("tiles", n), "r#for<u8>")));
    said.extend([x86_takes("42:47", "", "r#mut"), x86_gives("46:53", "", "r#mut"), variable.to_string()]);
    said.extend([two("59:43", "[r#mut]"), two("63:43", "str"), gpu_takes("72:47", "", "r#dyn::Foo")]);
    said.extend([1, 2].map(|n| gpu_takes("77:62", &instance("tiles", n), "r#mut")));
    said.push(two("81:47", "[r#mut]"));
    said.push(format!("error: could not compile `keywords_refused` (lib) due to {} previous errors", said.len()));
    refuses("keywords_refused", &said.iter().map(String::as_str).collect::<Vec<_>>());
}

/// Rust says of a type kernite refuses what it says of any function's signature, once, at the fault, beside
/// kernite's error at the type: a lifetime it cannot find, as for a result `&u64` with none to take, in a kernel
/// plain and of compile-time arguments, or for an `'a` that is not declared, on x86 and on a GPU, and so for
/// `'f16` and `'f128`, a lifetime's name that is no type; and a `_` in a result. A result `&u64` that a
/// parameter's lifetime serves is not one of these. So Rust says that an `f16` or an `f128` is unstable, where
/// kernite does not see it as a type, on x86 and on a GPU: one a macro writes, `half!()`, and one in an array's
/// length, `[u8; size_of::<f16>()]`. And Rust says that it expected a type and found a module, where a type is
/// written: `core::f16`, `[core::f128; 2]`, and an `f16` or an `f128` where the kernel's module imports a
/// module so named, alone or in an array. Rust says nothing at a call of any kernel.
#[test]
fn rust_says_of_a_type_kernite_refuses_what_it_says_of_any_signature() {
    let call = |place: &str| {
        format!("src/lib.rs:{place}: error: a call is a statement: an instruction or a helper, not a value")
    };
    // kernite says its errors first, in the order of the kernels, and Rust says its own after them.
    let mut said = vec![x86_gives("9:52", "", "&u64")];
    said.extend([1, 2].map(|n| x86_gives("14:66", &instance("tiles", n), "&u64")));
    said.extend([
        x86_takes("19:46", "", "&u64"),
        x86_gives("27:52", "", "_"),
        x86_takes("35:44", "", "&'a u64"),
        gpu_takes("43:48", "", "&'a u64"),
        x86_takes("51:44", "", "&'f16 u64"),
        gpu_takes("59:48", "", "&'f128 u64"),
        x86_takes("79:44", "", "half!()"),
        call("87:49"),
        gpu_takes("95:48", "", "wide!()"),
        call("103:53"),
        x86_takes("117:44", "", "core::f16"),
        x86_takes("128:44", "", "f16"),
        x86_takes("139:44", "", "[f16; 4]"),
        gpu_takes("147:48", "", "[core::f128; 2]"),
        gpu_takes("158:48", "", "f128"),
        gpu_takes("169:48", "", "[f16; 4]"),
    ]);
    said.extend([
        no_lifetime("9:52"),
        no_lifetime("14:66"),
        undeclared("35:45", "'a"),
        undeclared("43:49", "'a"),
        undeclared("51:45", "'f16"),
        undeclared("59:49", "'f128"),
        unstable("66:9", "f16"),
        unstable("87:59", "f16"),
        unstable("72:9", "f128"),
        unstable("103:63", "f128"),
        not_a_type("117:44", "core::f16"),
        not_a_type("128:44", "self::f16"),
        not_a_type("139:45", "self::f16"),
        not_a_type("147:49", "core::f128"),
        not_a_type("158:48", "self::f128"),
        not_a_type("169:49", "self::f16"),
        not_allowed("27:52", "return types"),
    ]);
    said.push(format!("error: could not compile `types_signature` (lib) due to {} previous errors", said.len()));
    refuses("types_signature", &said.iter().map(String::as_str).collect::<Vec<_>>());
}

/// The result of a kernel that can have none, a cooperative x86 kernel's or a GPU kernel's, is kernite's error at
/// its type, and Rust sees that type as it sees the result of a plain x86 kernel, beside kernite's error: it says
/// the same words, once, at the same place. So Rust says that a pointer's type has no size, `*const CStr`, or
/// lacks what the type it holds needs, `*const NeedsCopy<String>`; that a `&` has no lifetime to take, or that
/// its lifetime is not declared, `&'a u64`; that a `_` is not allowed; that `core::f16` is a module, and that an
/// `f128` is unstable, and that the `!` behind a pointer, `*const !`, is experimental. It says nothing of `!`
/// itself, nor of a pointer to an array of a compile-time argument, `*const [u8; N]`, nor of a type that only an
/// import names, `Ret`, which counts the import as used, so that Rust does not warn that it is not. A `&` in the
/// result of a cooperative x86 kernel or a GPU kernel takes a lifetime from what its launcher takes, the grid or
/// `self`, so Rust has none to ask for. A pointer of two words to a trait that is not dyn compatible,
/// `*mut dyn Copy`, is the one type Rust says of a plain kernel's result and not of a launcher's: it is kernite's
/// error alone there. A result of `()` is none, so those kernels are kernels, and their launchers give back the
/// `Result` of the launch. The functions that call each kernel as Rust calls it, with `?`, `unwrap`, a `match` and
/// a type for the result, get no word from Rust.
#[test]
fn rust_sees_the_result_a_kernel_cannot_have_as_it_sees_the_result_of_a_plain_kernel() {
    // What kernite says, in the order of the kernels: plain x86, cooperative x86, cooperative x86 of compile-time
    // arguments, then the GPU kernels, plain, of compile-time arguments, and cooperative, and last the kernels with
    // a pointer to a trait object and with never.
    let mut said = vec![
        x86_gives("24:62", "", "&u64"),
        x86_gives("28:61", "", "&'a u64"),
        x86_gives("36:59", "", "core::f16"),
        x86_gives("44:59", "", "Ret"),
        coop_returns("54:57", ""),
        coop_returns("59:58", ""),
        x86_gives("64:62", "", "&u64"),
        x86_gives("69:61", "", "&'a u64"),
        coop_returns("74:58", ""),
        x86_gives("79:59", "", "core::f16"),
        coop_returns("84:57", ""),
        x86_gives("89:59", "", "Ret"),
    ];
    said.extend([1, 2].map(|n| coop_returns("100:71", &instance("size", n))));
    said.extend([1, 2].map(|n| coop_returns("107:74", &instance("array", n))));
    said.extend([1, 2].map(|n| x86_gives("113:73", &instance("import", n), "Ret")));
    said.extend([
        gpu_returns("123:68", "", "*mut core::ffi::CStr"),
        gpu_returns("127:69", "", "*mut NeedsCopy<String>"),
        gpu_returns("131:73", "", "&u64"),
        gpu_returns("135:72", "", "&'a u64"),
        gpu_returns("139:69", "", "*mut _"),
        gpu_returns("143:70", "", "core::f16"),
        gpu_returns("147:68", "", "*mut f128"),
        gpu_returns("151:70", "", "Ret"),
    ]);
    said.extend([1, 2].map(|n| gpu_returns("161:74", &instance("size", n), "*mut core::ffi::CStr")));
    said.extend([1, 2].map(|n| gpu_returns("166:77", &instance("array", n), "*mut [u8; N]")));
    said.extend([1, 2].map(|n| gpu_returns("171:76", &instance("import", n), "Ret")));
    said.extend([gpu_returns("181:68", "", "*mut core::ffi::CStr"), gpu_returns("186:70", "", "Ret")]);
    said.extend([
        two_words("212:59", "", "dyn Copy"),
        two_words("220:59", "", "dyn Copy"),
        gpu_returns("227:70", "", "*mut dyn Copy"),
        coop_returns("244:59", ""),
        coop_returns("249:60", ""),
        gpu_returns("256:71", "", "!"),
        gpu_returns("260:71", "", "*mut !"),
    ]);
    // What Rust says, in the order it says it. Each word it says of `plain` it says of `cpu` and of `gpu`, at the
    // same place in the result, but that a `&` has no lifetime to take, in `cpu` and `gpu`.
    said.extend([
        no_lifetime("24:62"),
        undeclared("28:62", "'a"),
        unstable("40:64", "f128"),
        undeclared("69:62", "'a"),
        unstable("84:64", "f128"),
        undeclared("135:73", "'a"),
        unstable("147:73", "f128"),
        not_a_type("36:59", "core::f16"),
        not_a_type("79:59", "core::f16"),
        not_a_type("143:70", "core::f16"),
        experimental("236:67"),
        experimental("249:67"),
        experimental("260:76"),
        not_copy("20:65", "String"),
        not_allowed("32:65", "return types"),
        not_allowed("74:65", "return types"),
        not_dyn_compatible("212:64", "Copy"),
        not_allowed("139:74", "return types"),
        size_unknown("16:64", "[i8]"),
        size_unknown("54:64", "[i8]"),
        not_copy("59:65", "String"),
        size_unknown("100:78", "[i8]"),
        size_unknown("123:73", "[i8]"),
        not_copy("127:74", "String"),
        size_unknown("161:79", "[i8]"),
        size_unknown("181:73", "[i8]"),
    ]);
    said.push(format!("error: could not compile `result_kinds` (lib) due to {} previous errors", said.len()));
    refuses("result_kinds", &said.iter().map(String::as_str).collect::<Vec<_>>());
}

/// A helper is compiled where a kernel calls it, and an `extern "ptx-func"` function is a `.func` of the GPU's
/// module, so Rust sees neither. Where kernite says an error of an item, Rust is shown the types their
/// signatures name, as it sees a kernel's: a type of the crate's that only they name, `Ret`, is used, so Rust does
/// not call its `use` unused beside kernite's error, which is said once. So it is of a parameter, a result and a
/// compile-time argument, the second of two for a helper; of a type nested in a parameter, `[Ret; 2]`, `*const Ret`
/// or `Option<Ret>`; of a helper on x86 and on a GPU, called by a kernel of instances, which says its error for
/// each, or by a helper; and of a function of instances, which says its error for each. kernite says it at the
/// argument of the call of a helper, at the result of a helper that is called, and at the type of a function.
#[test]
fn rust_sees_the_types_a_helper_or_a_function_names_as_it_sees_a_kernels() {
    let gpr = "`x`: a gpr is seen as `gpr`, `r32`, `r16` or `r8`, not `Ret`";
    let func = "a function takes and returns PTX's types, as its variables are: `u32`, `f32`, `b64`; not `Ret`";
    let nothing = "a helper returns nothing; it writes the variables it is given";
    let generic = "a function that names an ABI takes integers and `bool`s at compile time, as Rust's const generics \
                   are; `Ret` is neither";
    let gpu = "`x`: PTX variables are `pred`, `b8` to `b128`, `u8` to `u64`, `s8` to `s64`, `f16`, `f16x2`, `bf16`, \
               `bf16x2`, `tf32`, `f32` and `f64`, not `Ret`";
    let said = |place: &str, kernel: &str, what: &str| format!("src/lib.rs:{place}: error: {kernel}{what}");
    // In the order of the modules: a helper's parameter, its result, its compile-time argument and the types nested
    // in its parameters, of which a call refuses the first; a helper of a kernel of instances, and a helper called
    // by a helper; then an `extern "ptx-func"` function's parameter, result, compile-time argument and
    // instances; last the GPU helper's parameter and result.
    let mut lines = vec![
        said("18:11", "", gpr),
        said("27:21", "", nothing),
        said("46:16", "", "this is an integer, not Ret"),
        said("68:15", "", "`x`: wants variables declared together; this is a variable"),
    ];
    lines.extend([1, 2].map(|n| said("85:11", &instance("k", n), gpr)));
    lines.extend([
        said("99:15", "", gpr),
        said("112:42", "", func),
        said("125:50", "", func),
        said("139:45", "", generic),
    ]);
    lines.extend([1, 2].map(|n| said("153:56", &instance("g", n), func)));
    lines.extend([said("171:11", "", gpu), said("180:21", "", nothing)]);
    lines.push(format!("error: could not compile `helper_imports` (lib) due to {} previous errors", lines.len()));
    refuses("helper_imports", &lines.iter().map(String::as_str).collect::<Vec<_>>());
}

/// Rust says of the types a helper's or an `extern "ptx-func"` function's signature names only three things, in a
/// module where kernite says an error: that a lifetime is not declared, `&'a Ret`, in a parameter and in a result;
/// that a module is where a type is written, `core::f16` in a parameter and in a result, and `core`, a crate; and
/// what a macro writes, `half!()`, an `f16`, which is unstable. It says nothing of the types the language has, `gpr`,
/// `ymm`, `[gpr; N]` and `local<[u32; 4], 16>`; nor of a type it refuses by itself, or kernite does: `Ret` behind a
/// `&`, a `*const`, in an array, an `Option`, a `Box<dyn Tr>` and a `fn`, a `[u64]`, a `str`, a `dyn Tr`, an `f16`, an
/// `f128`, a `!`, a `Self` and a `_`; nor of a compile-time argument of a type kernite refuses. Rust says the three of
/// a helper no kernel calls too, of whose types kernite says nothing. kernite says its own once, at the type of each
/// function it refuses.
#[test]
fn rust_says_of_the_types_a_helper_or_a_function_names_what_it_says_of_any_signature() {
    let func = |place: &str, ty: &str| {
        format!(
            "src/lib.rs:{place}: error: a function takes and returns PTX's types, as its variables are: `u32`, \
             `f32`, `b64`; not `{ty}`"
        )
    };
    let generic = "src/lib.rs:115:56: error: a function that names an ABI takes integers and `bool`s at compile time, \
                   as Rust's const generics are; `Ret` is neither";
    // kernite's errors first, in the order of the items, and Rust says its own after them.
    let mut said = vec![
        "src/lib.rs:22:13: error: `y` is not defined".to_string(),
        "src/lib.rs:75:28: error: `y` is not defined".to_string(),
        func("98:51", "Ret"),
        func("102:54", "&'a Ret"),
        func("106:60", "core::f16"),
        func("110:53", "half!()"),
        generic.to_string(),
    ];
    said.extend([
        undeclared("41:21", "'a"),
        unstable("11:9", "f16"),
        undeclared("57:36", "'a"),
        undeclared("86:21", "'a"),
        undeclared("102:55", "'a"),
        not_a_type("45:18", "core::f16"),
        "src/lib.rs:49:17: error[E0573]: expected type, found crate `core`: not a type".to_string(),
        not_a_type("61:33", "core::f16"),
        not_a_type("90:18", "core::f16"),
        not_a_type("106:60", "core::f16"),
    ]);
    // Rust counts the `f16` of `half!()` for each signature that writes it, three, and prints it once.
    said.push(format!("error: could not compile `helper_types` (lib) due to {} previous errors", said.len() + 2));
    refuses("helper_types", &said.iter().map(String::as_str).collect::<Vec<_>>());
}

/// A module that builds shows Rust none of the types of its helpers and functions, so an import of the crate's that
/// nothing names is unused to Rust, which says so as it does of any import; and so is one that only a helper names
/// that no kernel calls, for a helper is compiled where it is called. One that a kernel names, as the type a pointer
/// it takes points to, is used, on x86 and on a GPU.
#[test]
fn a_module_that_builds_shows_rust_no_types_of_a_helper_or_a_function() {
    let (built, said) = cargo(&["build"], "helper_imports_unused");
    assert!(built, "fixtures/helper_imports_unused.rs did not build");
    let unused = |place: &str| format!("src/lib.rs:{place}: warning: unused import: `super::Ret`");
    assert_eq!(said, [unused("9:9"), unused("37:9"), unused("50:9")]);
}

/// A function under `cfg` shows Rust its types where `cfg` keeps it and kernite says an error, there. The modules
/// are, in order: a helper that a kernel calls, both under `a`; a helper under `a` beside a kernel that does not
/// compile, kept; a helper kept beside a kernel under `a` that does not compile; an `extern "ptx-func"` function
/// under `a` beside a kernel that does not compile, kept; a helper under `a` beside a kernel under `b` that does not
/// compile; a helper under `a` beside a kernel that does not compile under `a` and another under `b`; a helper under
/// `a` beside a kernel under `a` and another under `b`, of which no module of PTX has both; and a helper under `a` in
/// a module that is two modules, which kernite says under no `cfg`. With no feature each import is unused to Rust,
/// for a function `cfg` leaves out names nothing, though the eighth module has its error. With `b` the kernels that
/// do not compile are kept, but no function that names `Ret` is kept in a module with an error, so every import is
/// unused. With `a` the fifth module has no error, and its import is unused, while the sixth has the error of its
/// kernel under `a`, which shows its helper though the kernel under `b` is left out; the seventh has none, for one
/// of its kernels is left out, and its import is unused; and the eighth shows its helper, for `cfg` keeps it. With
/// both every import is used, and kernite's errors are the only words, the seventh's among them: it is said where
/// both its kernels are kept.
#[test]
fn rust_sees_the_types_of_a_function_where_cfg_keeps_it_and_the_module_has_an_error() {
    let unused = |place: &str| format!("src/lib.rs:{place}: warning: unused import: `super::Ret`");
    let undefined = |place: &str| format!("src/lib.rs:{place}: error: `y` is not defined");
    let argument = "src/lib.rs:20:11: error: `x`: a gpr is seen as `gpr`, `r32`, `r16` or `r8`, not `Ret`".to_string();
    let result = "src/lib.rs:61:39: error: a function takes and returns PTX's types, as its variables are: `u32`, \
                  `f32`, `b64`; not `Ret`"
        .to_string();
    // What Cargo says with `features`, none if empty: kernite's errors, then Rust's warnings, then the count.
    let builds = |features: &str, mut said: Vec<String>, unused_at: &[&str], count: &str| {
        said.extend(unused_at.iter().map(|at| unused(at)));
        said.push(format!("error: could not compile `helper_imports_configured` (lib) due to {count}"));
        let command: &[&str] = if features.is_empty() { &["build"] } else { &["build", "--features", features] };
        refuses_by(command, "helper_imports_configured", &said.iter().map(String::as_str).collect::<Vec<_>>());
    };
    let two = "src/lib.rs:142:12: error: a static is memory of the module's kernels: the GPU's, for PTX kernels, or \
               the process's, for x86 ones; and this module has both, so it is two modules, one of each kind of \
               kernel and the statics they name"
        .to_string();
    let none = "src/lib.rs:124:39: error: `shfl.idx.b32` in `old` is for GPUs older than sm_70 from PTX 6.4 on, and \
                `tanh.approx.f32` in `new` needs sm_75 and PTX 7.0: no module has both"
        .to_string();
    let all = ["11:9", "28:9", "43:9", "58:9", "73:9", "94:9", "116:9", "140:9"];
    builds(
        "",
        vec![undefined("36:13"), undefined("66:28"), two.clone()],
        &all,
        "3 previous errors; 8 warnings emitted",
    );
    let said = vec![
        argument.clone(),
        undefined("36:13"),
        undefined("51:13"),
        result.clone(),
        undefined("66:28"),
        undefined("103:13"),
        two.clone(),
    ];
    builds("a", said, &["73:9", "116:9"], "7 previous errors; 2 warnings emitted");
    let said = vec![undefined("36:13"), undefined("66:28"), undefined("86:28"), undefined("108:13"), two.clone()];
    builds("b", said, &all, "5 previous errors; 8 warnings emitted");
    let said = vec![
        argument,
        undefined("36:13"),
        undefined("51:13"),
        result,
        undefined("66:28"),
        undefined("86:28"),
        undefined("103:13"),
        undefined("108:13"),
        none,
        two,
    ];
    builds("a,b", said, &[], "10 previous errors");
}

/// A crate that forbids lints is told of helpers and of `extern "ptx-func"` functions in a module where kernite says
/// an error, and of nothing else, as Rust and Clippy are: what the macros write for Rust to see their types needs
/// no `allow`, and a `forbid(unused_imports)` is not tripped by the import that only they name.
#[test]
fn a_crate_that_forbids_lints_is_told_of_the_helpers_that_do_not_compile_and_of_nothing_else() {
    let lines = [
        "src/lib.rs:35:13: error: `y` is not defined",
        "src/lib.rs:43:48: error: a function takes and returns PTX's types, as its variables are: `u32`, `f32`, \
         `b64`; not `Ret`",
        "src/lib.rs:56:28: error: `y` is not defined",
        "error: could not compile `helper_imports_forbidden` (lib) due to 3 previous errors",
    ];
    refuses("helper_imports_forbidden", &lines);
    if clippy_here() {
        refuses_by(&["clippy"], "helper_imports_forbidden", &lines);
    }
}

/// Rust is shown the types a helper's signature names where kernite says an error of the module as a whole, as it is
/// where kernite says one of an item: where the module is two modules; where no module of PTX has its GPU kernels;
/// where an item of its own has the name of a const it takes; where the module is refused at its attribute, which
/// `consts = 5` is; and where it has an item kernite cannot read, `bad<T>`. No kernel calls the helpers, so kernite
/// says nothing of their types. The import that only they name is used, so Rust does not call it unused beside the
/// module's error, and Rust says nothing of the language's types, `gpr`, `ymm`, `local<[u32; 4], 16>` and `r32`,
/// which it cannot find. A module that takes its consts from another is expanded after the rest, so its error is the
/// last.
#[test]
fn rust_sees_the_types_a_helper_names_where_kernite_says_an_error_of_the_module_as_a_whole() {
    let attribute = format!("src/lib.rs:77:27: error: {REFUSED}");
    refuses(
        "helper_imports_whole",
        &[
            "src/lib.rs:18:12: error: a static is memory of the module's kernels: the GPU's, for PTX kernels, or the \
             process's, for x86 ones; and this module has both, so it is two modules, one of each kind of kernel and \
             the statics they name",
            "src/lib.rs:51:39: error: `shfl.idx.b32` in `old` is for GPUs older than sm_70 from PTX 6.4 on, and \
             `tanh.approx.f32` in `new` needs sm_75 and PTX 7.0: no module has both",
            &attribute,
            "src/lib.rs:102:12: error: a function's generics are values, as in `<const N: usize>`",
            "src/lib.rs:66:15: error: `STEP` names one of the consts the module takes from `crate::sizes` as well",
            "error: could not compile `helper_imports_whole` (lib) due to 5 previous errors",
        ],
    );
}

/// Rust is shown the types a helper's signature names where the module is refused for a workspace named by what is
/// not a string: the error is the module's, at its attribute, and the import that only the helper names is used.
#[test]
fn rust_sees_the_types_a_helper_names_where_the_workspace_is_named_by_what_is_not_a_string() {
    let manifest = Path::new(ROOT).join("helper_imports_workspace/project/Cargo.toml");
    let error = format!(
        "src/lib.rs:9:1: error: `name` in `[workspace.metadata.kernite]` of {} is not a string: a workspace is named \
         by one, as `name = \"kernite\"`",
        manifest.display()
    );
    let count = "error: could not compile `helper_imports_workspace` (lib) due to 1 previous error";
    refuses("helper_imports_workspace", &[&error, count]);
}

/// Rust is shown the types a helper's signature names where a macro finds an error in writing the module, as it is
/// where kernite says one of an item: a compile-time argument named `grid` beside the launcher, of a cooperative
/// x86 kernel and of a GPU kernel; `threads` and `shared` beside the parameters of `k_blocks`, the method of a
/// cooperative GPU kernel; a `k_blocks` that a kernel and a static of the module are named as well; statics with no
/// room beside each other; and a name that `cfg` leaves out where the kernel that names it is kept. Each module has
/// a helper no kernel calls, whose parameter is `Ret`, a type of the crate's that nothing else names, and another
/// whose types are the language's, `gpr`, `ymm`, `local<[u32; 4], 16>` and `r32`. kernite's errors are all the
/// words: Rust does not call the import unused, and says nothing of the types it cannot find.
#[test]
fn rust_sees_the_types_a_helper_names_where_a_macro_finds_an_error_in_writing_the_module() {
    let grid = "names a parameter of the launcher as well";
    let blocks = "names a parameter of `k_blocks` as well";
    let twice = "`#[cooperative]` gives `Kernels` a method `k_blocks`, which says how many blocks a grid of `k` has at \
                 most, and a kernel or a static of the module is named `k_blocks` as well";
    let room = "this makes the module's constant memory 96000 bytes, and a module has 64 KB";
    let error = |place: &str, what: &str| format!("src/lib.rs:{place}: error: {what}");
    refuses(
        "helper_imports_macros",
        &[
            &error("26:43", &format!("`grid` {grid}")),
            &error("45:47", &format!("`grid` {grid}")),
            &error("65:47", &format!("`threads` {blocks}")),
            &error("65:67", &format!("`shared` {blocks}")),
            &error("82:7", twice),
            &error("103:7", twice),
            &error("112:19", room),
            &left_out("145:9", "g", "k"),
            "error: could not compile `helper_imports_macros` (lib) due to 8 previous errors",
        ],
    );
}

/// Rust is shown the types a helper's signature names where the crate lacks a dependency the Rust the macros write
/// needs, as it is where kernite says an error of the module as a whole: a static of GPU kernels needs
/// `kernite-macros`, to count its room, and `kernite-cuda`, for its module; a static of x86 kernels and a cooperative
/// x86 kernel need `kernite-cpu`; and any other module of GPU kernels needs `kernite-cuda`. The crate names its
/// dependencies otherwise, `km`, `cuda` and `cpu`, so it has none of them. Each module has a helper no kernel calls,
/// whose parameter is `Ret`, a type of the crate's that nothing else names, and another whose types are the
/// language's, `gpr`, `ymm`, `local<[u32; 4], 16>` and `r32`. kernite's errors are all the words.
#[test]
fn rust_sees_the_types_a_helper_names_where_the_crate_lacks_a_dependency_the_macros_write() {
    let edits = [
        ("kernite-macros = {", "km = { package = \"kernite-macros\","),
        ("kernite-cuda = {", "cuda = { package = \"kernite-cuda\","),
        ("kernite-cpu = {", "cpu = { package = \"kernite-cpu\","),
    ];
    let (built, said) = cargo_in(&["build"], &edited("helper_imports_deps", &edits));
    assert!(!built, "fixtures/helper_imports_deps.rs built");
    assert_eq!(
        said,
        [
            "src/lib.rs:10:1: error: a static of GPU kernels needs a dependency named `kernite-macros`, which this \
             crate does not have",
            "src/lib.rs:10:1: error: a module of GPU kernels needs a dependency named `kernite-cuda`, which this \
             crate does not have",
            "src/lib.rs:33:16: error: a static of x86 kernels needs a dependency named `kernite-cpu`, which this \
             crate does not have",
            "src/lib.rs:61:35: error: a cooperative x86 kernel needs a dependency named `kernite-cpu`, which this \
             crate does not have",
            "src/lib.rs:67:1: error: a module of GPU kernels needs a dependency named `kernite-cuda`, which this \
             crate does not have",
            "error: could not compile `helper_imports_deps` (lib) due to 5 previous errors",
        ]
    );
}

/// Rust is silent about a name it cannot find only where an error is an item of the module, and the macros find each
/// error they write as an item, wherever it comes. Each module has a helper no kernel calls: `h` takes `Ret`, a type
/// of the crate's that nothing else names, and `language` the language's types. In the first, the error is the
/// module's as a whole, that it is two modules, right after the module's own attribute. In the second, it is a
/// kernel's, of an ABI kernite does not know, right after a kernel under `a`, whose `cfg` is not the error's. In the
/// third, it is Rust's own, in a type, and no item. The fourth builds: an item of its own,
/// `::std::thread_local! {..}`, is written as the macros write an error, and is none. Rust is shown the helpers'
/// types in the first two, so it does not call the import unused, and says nothing of the types it cannot find. In
/// the last two it is not, and says the import is unused.
#[test]
fn rust_sees_the_types_a_helper_names_wherever_the_error_a_macro_writes_is_an_item_of_the_module() {
    let two = "src/lib.rs:16:12: error: a static is memory of the module's kernels: the GPU's, for PTX kernels, or the \
               process's, for x86 ones; and this module has both, so it is two modules, one of each kind of kernel \
               and the statics they name";
    let abi = "src/lib.rs:54:23: error: kernite compiles `extern \"sysv64\"` kernels, for x86-64, and `extern \
               \"ptx-kernel\"` ones, for NVIDIA GPUs, with the `extern \"ptx-func\"` functions they call; `extern \
               \"C\"` is none of these";
    let lines = [
        two,
        abi,
        "src/lib.rs:61:17: error: an error of Rust's own",
        "src/lib.rs:59:9: warning: unused import: `super::Ret`",
        "src/lib.rs:78:9: warning: unused import: `super::Ret`",
        "error: could not compile `helper_imports_scan` (lib) due to 3 previous errors; 2 warnings emitted",
    ];
    refuses("helper_imports_scan", &lines);
    refuses_by(&["build", "--features", "a"], "helper_imports_scan", &lines);
}

/// A GPU module whose statics have no room beside each other where `cfg` keeps both is told so at the type of the
/// second, by `__Room`, which counts the room of the statics Rust keeps. The helpers `cfg` keeps are shown to Rust
/// there too: `h` takes `Ret`, a type of the crate's that nothing else names, `lifetimes` a lifetime the module has
/// not declared, which Rust says, and `language` the language's types, of which it says nothing. Where the room is
/// right the module builds, and shows Rust none of them. Where a kernel that does not compile is kept, `broken`, the
/// module has an error of its own, and Rust says of the types once, beside both. The features: `a` and `b` keep a
/// static each, `c` keeps `h` and `lifetimes`, and `d` the kernel. `language` is always kept.
#[test]
fn rust_sees_the_types_a_helper_names_where_the_statics_have_no_room_and_cfg_keeps_the_helper() {
    let unused = "src/lib.rs:12:9: warning: unused import: `super::Ret`";
    let room = "src/lib.rs:18:19: error: this makes the module's constant memory 96000 bytes, and a module has 64 KB";
    let broken = "src/lib.rs:40:13: error: a bool is no operand";
    let lifetime = undeclared("26:22", "'a");
    let says = |features: &str, lines: &[&str], count: &str| {
        let mut lines = lines.to_vec();
        let count = format!("error: could not compile `helper_imports_room` (lib) due to {count}");
        lines.push(&count);
        refuses_by(&["build", "--features", features], "helper_imports_room", &lines);
    };
    // The room is right: the module builds, and shows Rust no helper.
    let (built, said) = cargo(&["build", "--features", "a,c"], "helper_imports_room");
    assert!(built, "fixtures/helper_imports_room.rs did not build with a,c");
    assert_eq!(said, [unused]);
    // The room is the only error, and `h` is left out: the import is unused.
    says("a,b", &[room, unused], "1 previous error; 1 warning emitted");
    // The room is the only error, and the helpers are kept: `__Room` shows Rust their types beside it.
    says("a,b,c", &[room, &lifetime], "2 previous errors");
    // The kernel is kept as well, and `h` is not: the import is unused.
    says("a,b,d", &[broken, room, unused], "2 previous errors; 1 warning emitted");
    // The kernel and the helpers are kept: Rust says of the types once.
    says("a,b,c,d", &[broken, room, &lifetime], "3 previous errors");
    // The kernel and the helpers are kept, and no static: the types are shown beside the kernel's error alone.
    says("c,d", &[broken, &lifetime], "2 previous errors");
}

/// A compile-time argument named as the launcher's parameter, `grid`, of a cooperative kernel that `cfg` keeps under
/// `a`, x86 or the GPU's, and one named as its `_blocks` method's, `threads`, are errors the macros find in writing the
/// module, and are not said of a kernel that `cfg` leaves out. The helpers are under `b`: `h` takes `Ret`, a type of
/// the crate's that nothing else names, and `language` the language's types. Rust is shown their types where the errors
/// are said and the helpers are kept, so it does not call the import unused, and says nothing of the types it cannot
/// find; otherwise the import is unused.
#[test]
fn rust_sees_the_types_a_helper_names_where_cfg_keeps_the_error_the_macros_find_and_the_helper() {
    let unused = [
        "src/lib.rs:11:9: warning: unused import: `super::Ret`",
        "src/lib.rs:38:9: warning: unused import: `super::Ret`",
    ];
    let errors = [
        "src/lib.rs:27:43: error: `grid` names a parameter of the launcher as well",
        "src/lib.rs:54:47: error: `grid` names a parameter of the launcher as well",
        "src/lib.rs:54:64: error: `threads` names a parameter of `g_blocks` as well",
    ];
    let says = |features: &str, lines: &[&str], count: &str| {
        let mut lines = lines.to_vec();
        let count = format!("error: could not compile `helper_imports_clash` (lib) due to {count}");
        lines.push(&count);
        refuses_by(&["build", "--features", features], "helper_imports_clash", &lines);
    };
    // The kernels are kept and the helpers are not: the import is unused.
    says("a", &[errors.as_slice(), unused.as_slice()].concat(), "3 previous errors; 2 warnings emitted");
    // The helpers are kept and the kernels are not: nothing is wrong, and the import is unused.
    let (built, said) = cargo(&["build", "--features", "b"], "helper_imports_clash");
    assert!(built, "fixtures/helper_imports_clash.rs did not build with b");
    assert_eq!(said, unused);
    // Both are kept: Rust is shown the types, and says nothing of them.
    says("a,b", &errors, "3 previous errors");
}

/// A pointer to a type kernite refuses by value is said as it was. A pointer of one word is taken, whatever
/// it points to: Rust's word that an `f128` is unstable is the only one, said once, at the type, in a
/// parameter and in a result, on x86 and on a GPU. A pointer of two words, to a `[u64]`, is kernite's error
/// alone, at the pointer's type. Both are so in a kernel of every kind.
#[test]
fn a_pointer_to_a_type_kernite_refuses_by_value_is_said_as_it_was() {
    let two = |place: &str, kernel: &str| two_words(place, kernel, "[u64]");
    // kernite says its errors first, in the order of the kernels, and Rust says its own after them.
    let mut said = vec![two("33:44", "")];
    said.extend([1, 2].map(|n| two("38:58", &instance("tiles", n))));
    said.extend([two("43:47", ""), two("47:45", "")]);
    said.extend([1, 2].map(|n| two("52:65", &instance("tiles_result", n))));
    said.push(two("76:48", ""));
    said.extend([1, 2].map(|n| two("81:62", &instance("tiles", n))));
    said.push(two("86:51", ""));
    said.extend(
        ["7:49", "12:63", "17:52", "21:50", "21:64", "26:70", "26:84", "59:53", "64:67", "69:56"]
            .map(|place| unstable(place, "f128")),
    );
    said.push(format!("error: could not compile `types_pointers` (lib) due to {} previous errors", said.len()));
    refuses("types_pointers", &said.iter().map(String::as_str).collect::<Vec<_>>());
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
/// "expected `Grid`, found `u32`", as the launcher or method never runs. So is a name written raw,
/// `r#grid`, which Rust reads as `grid`.
#[test]
fn a_compile_time_argument_named_as_the_launchers_parameters_is_refused_at_its_name() {
    refuses(
        "arguments",
        &[
            "src/lib.rs:27:47: error: `grid` names a parameter of the launcher as well",
            "src/lib.rs:38:47: error: `threads` names a parameter of `k_blocks` as well",
            "src/lib.rs:49:47: error: `shared` names a parameter of `k_blocks` as well",
            "src/lib.rs:60:43: error: `grid` names a parameter of the launcher as well",
            "src/lib.rs:71:43: error: `grid` names a parameter of the launcher as well",
            "error: could not compile `arguments` (lib) due to 5 previous errors",
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

/// A type's name written raw, `r#u64`, is the type Rust reads: Rust builds the crate without a word, an x86 kernel
/// has the listing of the same kernel written plain, and runs as it does, whether its types are numbers,
/// floats, pointers, a compile-time argument's, a variable's, or a static's, and the kernel is plain, of
/// compile-time arguments or cooperative; and a GPU kernel runs with the launcher Rust reads, and its static is
/// read back as the type Rust reads.
#[test]
fn a_type_written_raw_is_the_type_rust_reads() {
    let printed = runs_quietly(&["test"], "raw_types");
    assert!(
        printed.contains("test result: ok. 4 passed"),
        "fixtures/raw_types.rs ran its tests, and printed:\n{printed}"
    );
}

/// A type named by a keyword written raw is the type Rust reads: a pointer to `r#dyn::Foo`, a struct of a module
/// named `dyn`, or to `r#mut<u64>`, a generic struct, has one word, as Rust has it, where a pointer to `dyn Foo` has
/// two. Rust builds the crate without a word, an x86 kernel has the listing of the kernel that names types of the same
/// layout by words that are no keywords, `dyn_::Foo` say, and runs as it does, whether the kernel is plain, of
/// compile-time arguments or cooperative, and a GPU kernel runs with the launcher Rust reads.
#[test]
fn a_type_named_by_a_keyword_written_raw_is_the_type_rust_reads() {
    let printed = runs_quietly(&["test"], "raw_keywords");
    assert!(
        printed.contains("test result: ok. 4 passed"),
        "fixtures/raw_keywords.rs ran its tests, and printed:\n{printed}"
    );
}

/// An attribute kernite reads, written raw, is the attribute written plain: a kernel's `#[r#instances(..)]` and
/// `#[r#cooperative]`, a GPU kernel's `#[r#maxntid(..)]`, `#[r#reqntid(..)]`, `#[r#minnctapersm(..)]` and
/// `#[r#maxnreg(..)]`, and the module's `#[kernels(r#consts = ..)]`. Rust builds the crate without a word. An x86
/// module written so has the listings of the one written plain, and runs as it does. A GPU module written so is the
/// PTX of the one written plain, has its launchers, `together_blocks` among them, and the driver holds a launch to
/// its directives. A module given its consts by `r#consts` reads them as one given them by `consts`.
#[test]
fn an_attribute_written_raw_is_the_attribute_written_plain() {
    let printed = runs_quietly(&["test"], "raw_attributes");
    assert!(
        printed.contains("test result: ok. 7 passed"),
        "fixtures/raw_attributes.rs ran its tests, and printed:\n{printed}"
    );
}

/// An attribute kernite reads, written raw, is wrong as the plain one is: kernite's error alone, in the same words, at
/// the same column of a line of its own. So it is for an `#[r#instances]` with no list; two `#[r#instances(..)]`,
/// whichever way each is written, for the two are one attribute; an `#[r#cooperative(1)]`, which takes nothing; two
/// `#[r#cooperative]`; a directive with no list, `#[r#maxnreg]`; two of one directive; a directive PTX has no more,
/// `#[r#maxnctapersm(2)]`; a directive on a kernel for x86-64; and the module's own, `#[kernels(r#consts = 5)]`.
#[test]
fn an_attribute_written_raw_is_wrong_as_the_plain_one_is() {
    // The line of the attribute refused in each module of the fixture, the plain one first, then the raw one, and
    // for two of one the mixed one; the column of its name; and what is said.
    let cases: [(&[u32], u32, &str); 9] = [
        (&[8, 14], 7, "`#[instances(..)]` lists what a function is compiled for, as in `#[instances(4, 8)]`"),
        (&[21, 28, 35], 7, "a function has one `#[instances(..)]`, which lists each instance it is compiled for"),
        (&[41, 47], 7, "`#[cooperative]` takes nothing: it says that a kernel's blocks or threads wait for each other"),
        (&[54, 61, 68], 7, "a kernel has one `#[cooperative]`"),
        (&[74, 80], 7, "`#[maxnreg(..)]` gives the most registers a thread may use, as in `#[maxnreg(32)]`"),
        (&[87, 94, 101], 7, "a kernel has one `#[maxntid(..)]`"),
        (
            &[107, 113],
            7,
            "PTX has no `.maxnctapersm` from 2.1 on; `#[minnctapersm(..)]` gives the blocks an SM should hold at once, \
             at least",
        ),
        (
            &[119, 125],
            7,
            "`#[maxntid(..)]` goes on a GPU kernel, `extern \"ptx-kernel\"`; `k` is a kernel for x86-64, which the \
             process runs",
        ),
        (&[129, 134], 27, REFUSED),
    ];
    let mut said = vec![];
    for (lines, column, message) in cases {
        said.extend(lines.iter().map(|line| format!("src/lib.rs:{line}:{column}: error: {message}")));
    }
    said.push(format!("error: could not compile `raw_attributes_refused` (lib) due to {} previous errors", said.len()));
    refuses("raw_attributes_refused", &said.iter().map(String::as_str).collect::<Vec<_>>());
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

/// kernite says a block, the length of an array in a tuple and a macro's arguments with the spaces their source has,
/// as the messages of rustc write a block they do not work out: `Marker<{N*2}>` from `{N*2}`, `Marker<{ N * 2 }>` from
/// `{ N * 2 }` and `sum!(a+b)` from `sum!(a+b)`. A space, a line break or a comment is one space, so a block over
/// several lines, with a comment, is one line, even where each line starts on the column where the one before it ends.
/// It is so for a parameter and a result of an x86 kernel, and for a parameter of a GPU kernel. A block that a macro
/// builds from a fragment has a space where the macro's body and the fragment meet, `{ N * 2 }` from `{ N * $n }` and
/// `{N* 2 }` from `{N*$n}`, and the fragment `N + 1` has no brackets round it. A block that a macro is given whole, as
/// a fragment, is as the call has it, `{N*2}` from `{N*2}`. A punct that the stream says is joint to the next has
/// nothing after it, whatever the spans say: two fragments that are each a `<`, joint to the `,` after each at the
/// call, are `<<`.
#[test]
fn a_block_an_arrays_length_and_a_macros_arguments_are_said_as_their_source_has_them() {
    let mut said = vec![
        x86_takes("25:51", "", "Marker<{N*2}>"),
        x86_takes("29:51", "", "Marker<{ N * 2 }>"),
        x86_takes("33:51", "", "Marker<{N * 2}>"),
        x86_takes("38:12", "", "Marker<{ N * 2 }>"),
        x86_takes("47:59", "", "Marker<{N *2}>"),
        x86_takes("51:58", "", "Marker<{size_of::<Vec<u8>>()*2}>"),
        x86_takes("55:51", "", "(u8, [u8; N*2])"),
        x86_takes("59:51", "", "(u8, [u8; N * 2])"),
        x86_takes("63:51", "", "sum!(a+b)"),
        x86_takes("67:51", "", "sum!(a + b)"),
        x86_takes("71:51", "", "sum! {a+b}"),
        x86_gives("75:60", "", "Marker<{N*2}>"),
        x86_gives("79:60", "", "Marker<{ N * 2 }>"),
        gpu_takes("89:55", "", "Marker<{N*2}>"),
        gpu_takes("93:55", "", "Marker<{ N * 2 }>"),
        gpu_takes("97:55", "", "sum!(a+b)"),
        gpu_takes("101:55", "", "sum!(a + b)"),
    ];
    // The modules that `built!` writes, one for each call of it, at the same places in its body.
    for blocks in [
        ["{ N * 2 }", "{N* 2 }", "{ 2 }", "{ size_of::< u8 >() }", "{ N << 2 }"],
        ["{ N * N + 1 }", "{N* N + 1 }", "{ N + 1 }", "{ size_of::< [u8; N + 1] >() }", "{ N + 2 }"],
    ] {
        for (place, block) in ["113:53", "117:53", "121:52", "125:51", "129:55"].into_iter().zip(blocks) {
            said.push(x86_takes(place, "", &format!("Marker<{block}>")));
        }
    }
    said.extend([x86_takes("145:51", "", "sum!(a << b)"), x86_takes("149:53", "", "sum!(a < b)")]);
    // A block whose lines start where the one before ends, so that no column tells a line break from a space.
    said.push(x86_takes("165:12", "", "Marker<{ N * 2 }>"));
    // A block that a macro is given whole is as the call has it. One that it builds from the tokens it is given
    // has a space where its body and the tokens meet, though the body has none.
    said.extend([
        x86_takes("183:52", "", "Marker<{N*2}>"),
        x86_takes("183:52", "", "Marker<{ N * 2 }>"),
        x86_takes("201:53", "", "Marker<{ N*2 }>"),
        x86_takes("205:53", "", "Marker<{ N*2 }>"),
        x86_takes("201:53", "", "Marker<{ N * 2 }>"),
        x86_takes("205:53", "", "Marker<{ N * 2 }>"),
    ]);
    said.push(format!("error: could not compile `type_quotes` (lib) due to {} previous errors", said.len()));
    refuses("type_quotes", &said.iter().map(String::as_str).collect::<Vec<_>>());
}

/// A macro's fragment `$n:expr` is one operand where an x86 kernel reads it, as Rust reads it: `$n * 2` with `$n` as
/// `1 + 2` is 6, and not the 5 that `1 + 2 * 2` is. The fixture's kernels put a fragment of each operator of an
/// integer and of a bool known now before and after each operator, and before `as`, `.` and `..`, and give what Rust
/// gives for the same fragment worked out beside them. It is so for a fragment a macro passes on to another, whole or
/// as tokens, and under a `-`, for a number with a sign too.
#[test]
fn a_fragment_is_one_operand_in_an_x86_kernels_expression() {
    let printed = runs_quietly(&["test"], "fragments_operands");
    assert!(
        printed.contains("test result: ok. 42 passed"),
        "fixtures/fragments_operands.rs ran, and printed:\n{printed}"
    );
}

/// A const of a module that a macro writes from a fragment is one value to Rust, which gets the module written back,
/// and to the kernel that reads it: `pub const C: u64 = $n * 2;` is 6 for both, as is an array's length, `[u64; $n * 2]`,
/// and an element of an array, of an array of arrays too. So it is for a fragment that holds a fragment, and for a
/// const in a module of its own that the kernels take by its path.
#[test]
fn a_const_built_from_a_fragment_is_one_value_to_rust_and_to_the_kernel() {
    let printed = runs_quietly(&["test"], "fragments_consts");
    assert!(printed.contains("test result: ok. 5 passed"), "fixtures/fragments_consts.rs ran, and printed:\n{printed}");
}

/// A fragment is one operand, or one type, in the places besides an x86 kernel's expression and a const: a GPU
/// kernel's operand, where the PTX writes 6 for `$n * 2` and -3 for `-$n`, an array's length, `#[instances(..)]`, and,
/// for a `$t:ty`, a parameter's type, a result and an argument of a type, of an x86 kernel and of a GPU kernel.
#[test]
fn a_fragment_is_one_operand_or_one_type_in_each_place_kernite_reads_it() {
    let printed = runs_quietly(&["test"], "fragments_places");
    assert!(
        printed.contains("test result: ok. 12 passed"),
        "fixtures/fragments_places.rs ran, and printed:\n{printed}"
    );
}

/// A fragment that is no group, `$i:ident` and `$t:tt`, and a literal of one token, a group of one, are read as they
/// were, and so is a fragment that no operator is next to or that holds none of its own.
#[test]
fn a_fragment_that_is_no_group_is_read_as_it_was() {
    let printed = runs_quietly(&["test"], "fragments_bare");
    assert!(printed.contains("test result: ok. 6 passed"), "fixtures/fragments_bare.rs ran, and printed:\n{printed}");
}

/// An error at a fragment is said where the fragment is, in the call of the macro that has it. kernite's: at the name
/// that is not defined, the `x` of `x + 1`; at the operator that has no meaning between its operands, the `+` of
/// `1.5 + 2`; at the string; and at the `+` of `200 + 100`, which is out of range for a `u8`. Rust's, of the fragment as
/// a whole, where it starts: that `200 + 100` overflows a `u8`, and that `1u16 + 2` is no `u8`. A fragment is one
/// operand, so `1 - 1` under `10 / $n` is the division by zero it is for Rust, an array as long as `$n * 2`, which is
/// 6 for `1 + 2`, has no index 6, and a fragment where a `;` should be is told as one thing, `1 + 2`. A type that is a
/// fragment is said whole, `Wrap<Wrap<u8>>`, and, next to a result, as it was.
#[test]
fn an_error_at_a_fragment_is_said_where_the_fragment_is() {
    let said_at = |place: &str, message: &str| format!("src/lib.rs:{place}: error: {message}");
    let mut said = vec![
        said_at("20:19", "`x` is not defined"),
        said_at("21:21", "`+` has no meaning between an f32 and an integer"),
        said_at("22:18", "a string is no value here; `note(\"..\")` takes one"),
        said_at("29:27", "10 / 0"),
        said_at("44:22", "`v` has 6 variables; there is no [6]"),
        said_at("67:22", "300 is out of range for u8"),
        said_at("57:34", "this is a u16, not u8"),
        x86_takes("87:17", "", "&f16"),
        x86_takes("95:48", "", "Wrap<Wrap<[u8; 6]>>"),
        x86_takes("95:48", "", "Wrap<Wrap<u8>>"),
        // Rust's own word on the stray fragment, in the macro's body; kernite's is at the call.
        said_at(
            "123:27",
            "expected one of `.`, `;`, `?`, `}`, or an operator, found `expr` metavariable: expected one of `.`, \
             `;`, `?`, `}`, or an operator",
        ),
        said_at("130:22", "expected `;`, found `1 + 2`"),
        unstable("86:25", "f128"),
        unstable("87:18", "f16"),
        "src/lib.rs:67:18: error[E0080]: attempt to compute `200_u8 + 100_u8`, which would overflow: evaluation of \
         `too_wide::C` failed here"
            .to_string(),
        not_the_number("68:20", "u8", "u16"),
        size_unknown("116:31", "[i8]"),
    ];
    said.push(format!("error: could not compile `fragments_refused` (lib) due to {} previous errors", said.len()));
    refuses("fragments_refused", &said.iter().map(String::as_str).collect::<Vec<_>>());
}
