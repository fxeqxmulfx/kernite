//! What the macro says of the kernels it is given: each error where it is, at its
//! `line:column`, in the order Cargo prints them, and each kernel's safety contract, in its
//! docs. Each fixture is built as a user's crate is, and what Cargo says of it is compared line
//! for line, the errors in rustc's short form, `src/lib.rs:4:1: error: ..`. A fixture is a
//! crate's `src/lib.rs`, so the line an error names is that line of the fixture.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
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

/// Whether Cargo did `command` in the crate at `dir`, and what it said, a line each, in order.
/// The crate is built offline into `kernite-macros/target`, so that it never waits for the build
/// that runs these tests. What the environment would change of a build is taken out of it.
fn cargo_in(command: &[&str], dir: &Path) -> (bool, Vec<String>) {
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
    let out = run.output().expect("Cargo runs");
    let said = String::from_utf8(out.stderr).expect("Cargo writes UTF-8");
    (out.status.success(), said.lines().map(String::from).collect())
}

/// That Cargo builds fixture `name` without a word.
fn builds_quietly(name: &str) {
    let (built, said) = cargo(&["build"], name);
    assert!(built && said.is_empty(), "fixtures/{name}.rs: built {built}, and Cargo said:\n{}", said.join("\n"));
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
    let (built, said) = cargo(&["build"], name);
    assert!(!built, "fixtures/{name}.rs built");
    assert_eq!(said, lines, "fixtures/{name}.rs: Cargo said:\n{}", said.join("\n"));
}

/// Whether this machine has Clippy, which the tests that run it skip without, and say so.
fn clippy_here() -> bool {
    let has = Command::new(env!("CARGO")).args(["clippy", "--version"]).output().is_ok_and(|out| out.status.success());
    if !has {
        eprintln!("no Clippy here: skipped");
    }
    has
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
    let dir = written("no_cpu");
    let manifest = dir.join("Cargo.toml");
    let text = fs::read_to_string(&manifest).unwrap();
    let without: Vec<&str> = text.lines().filter(|line| !line.starts_with("kernite-cpu")).collect();
    fs::write(&manifest, without.join("\n") + "\n").unwrap();
    let (built, said) = cargo_in(&["build"], &dir);
    assert!(!built, "fixtures/no_cpu.rs built");
    assert_eq!(
        said,
        [
            "src/lib.rs:8:16: error: a static of x86 kernels needs `kernite-cpu`, which this crate does not depend on",
            "src/lib.rs:21:35: error: a cooperative x86 kernel needs `kernite-cpu`, which this crate does not depend \
             on",
            "src/lib.rs:28:35: error: a cooperative x86 kernel needs `kernite-cpu`, which this crate does not depend \
             on",
            "error: could not compile `no_cpu` (lib) due to 3 previous errors",
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

/// A parameter named as a local the macro's Rust binds for itself, `index`, `grid`, `threads` or
/// `shared`, is the kernel's own, and reaches it: Rust builds the module without a word, where the
/// launcher's own `index` hid it, and Rust said "unused variable: `index`". So is a compile-time
/// argument, where no launcher or method binds its name.
#[test]
fn a_kernels_parameters_keep_their_names_beside_those_the_macros_rust_binds() {
    builds_quietly("parameters");
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
/// else: its stub and its launcher, which never run, use what they take.
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
            "src/lib.rs:64:28: error: `y` is not defined",
            "src/lib.rs:69:28: error: `y` is not defined",
            "src/lib.rs:52:13: error: `y` is not defined",
            "src/lib.rs:57:18: error: `y` is not defined",
            "error: could not compile `forbidden_stubs` (lib) due to 10 previous errors",
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

/// A crate that forbids a lint a cooperative kernel expects is refused once, at the `expect`: the
/// launcher keeps it as written, and `k_blocks` keeps no lint.
#[test]
fn a_crate_that_forbids_a_lint_a_kernel_expects_is_refused_once_at_the_expect() {
    refuses(
        "forbidden_expect",
        &[
            "src/lib.rs:10:14: error[E0453]: expect(non_snake_case) incompatible with previous forbid: overruled by \
             previous forbid",
            "error: could not compile `forbidden_expect` (lib) due to 1 previous error",
        ],
    );
}

/// Clippy says "too many arguments" of a function of more than seven parameters wherever its tokens
/// come from, and a launcher takes `self` and the grid as well as the kernel's parameters: so each
/// launcher allows it, and Clippy says nothing of one of eight or nine, of each kind of kernel. It
/// says nothing of the kernels that forbid lints either, whose instances include `bool`s.
#[test]
fn clippy_says_nothing_of_the_rust_the_macro_writes() {
    if clippy_here() {
        for name in ["many", "forbidden"] {
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
