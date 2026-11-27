//! What the tests of the macros share: each fixture is written as the crate of a user's, in a directory of its own.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};

/// What a test returns: nothing, or why it failed.
pub type Result<T = (), E = Box<dyn std::error::Error + Send + Sync>> = std::result::Result<T, E>;

/// Where in `CARGO_TARGET_TMPDIR` each fixture is a crate, `kernite-macros/name/project`, so that
/// its workspace is not named as the crate is; and where they are built, `kernite-macros/target-0` and on.
pub const ROOT: &str = concat!(env!("CARGO_TARGET_TMPDIR"), "/kernite-macros");

/// Held while a test writes fixture `name`, or Cargo builds it: two tests may write and build the same fixture, and the
/// one must not write it while Cargo builds it for the other.
pub fn fixture(name: &str) -> MutexGuard<'static, ()> {
    static EACH: Mutex<BTreeMap<String, &'static Mutex<()>>> = Mutex::new(BTreeMap::new());
    let one = *EACH
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .entry(name.into())
        .or_insert_with(|| Box::leak(Box::default()));
    one.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Fixture `name` written as the `src/lib.rs` of a crate that depends on kernite-macros, kernite-cuda and kernite-cpu,
/// as a user's does: the crate's directory. The crate is a workspace of its own in `CARGO_TARGET_TMPDIR`. Few fixtures
/// use all three dependencies, so its manifest allows `unused_dependencies`, Cargo's lint of a dependency that a crate
/// does not use, and ends with what `{name}.toml` beside the fixture says of the workspace, if there is one. The files
/// of the folder `{name}/` beside it, if there is one, are the crate's other files: `src/macros.rs`, for `mod macros;`.
pub fn written(name: &str) -> Result<PathBuf> {
    crate_of(name, |fixtures, dir| {
        fs::copy(fixtures.join(format!("{name}.rs")), dir.join("src/lib.rs"))?;
        for file in fs::read_dir(fixtures.join(name)).into_iter().flatten() {
            let file = file?;
            fs::copy(file.path(), dir.join("src").join(file.file_name()))?;
        }
        Ok(())
    })
}

/// The directory of crate `name`, as `written` writes it, with its manifest, and the files that `files`, given the
/// directory of the fixtures and the crate's, writes in it.
pub fn crate_of(name: &str, files: impl FnOnce(&Path, &Path) -> Result) -> Result<PathBuf> {
    let _one = fixture(name);
    let here = Path::new(env!("CARGO_MANIFEST_DIR"));
    let crates = here.parent().ok_or("kernite-macros is in no directory")?;
    let fixtures = here.join("tests/fixtures");
    let dir = Path::new(ROOT).join(name).join("project");
    fs::create_dir_all(dir.join("src"))?;
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

[lints.cargo]
unused_dependencies = "allow"

[workspace]
"#
    );
    let workspace = fs::read_to_string(fixtures.join(format!("{name}.toml"))).unwrap_or_default();
    fs::write(dir.join("Cargo.toml"), manifest + &workspace)?;
    files(&fixtures, &dir)?;
    Ok(dir)
}

/// What the environment would change of a build, which a test takes out of the environment of Cargo and of a program
/// that runs it.
pub const BUILD_ENV: [&str; 10] = [
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
];
