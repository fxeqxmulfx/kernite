//! The output is deterministic, as GUIDE.md guarantees: clippy forbids hash maps and hash sets,
//! here; and no crate depends on anything outside the workspace, in architecture.rs.

use std::fs;
use std::path::Path;

/// The workspace's root.
const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");

/// The text of the file at `path` in the workspace.
fn read(path: &str) -> String {
    fs::read_to_string(Path::new(ROOT).join(path)).unwrap()
}

/// The lines of `text` after the line `head`, up to one that starts a table or ends an array:
/// those of a table of a manifest, or of an array of clippy.toml.
fn under<'a>(text: &'a str, head: &str) -> impl Iterator<Item = &'a str> {
    text.lines().skip_while(move |line| *line != head).skip(1).take_while(|line| !line.starts_with(['[', ']']))
}

#[test]
fn clippy_forbids_hash_maps_and_hash_sets() {
    // The order they iterate in differs from run to run, and would reach the output.
    let listed = read("clippy.toml");
    for ty in ["std::collections::HashMap", "std::collections::HashSet"] {
        let entry = format!("{{ path = \"{ty}\"");
        assert!(
            under(&listed, "disallowed-types = [").any(|line| line.trim_start().starts_with(&entry)),
            "clippy.toml lets {ty} be used"
        );
    }
    // Each crate denies what clippy.toml lists, or takes the workspace's lints, which deny it.
    let deny = "disallowed_types = \"deny\"";
    assert!(
        under(&read("Cargo.toml"), "[workspace.lints.clippy]").any(|line| line == deny),
        "the workspace's lints let a crate use what clippy.toml lists"
    );
    let root = Path::new(ROOT);
    for dir in ["crates", "examples"] {
        for entry in fs::read_dir(root.join(dir)).unwrap() {
            let path = format!("{dir}/{}", entry.unwrap().file_name().display());
            let manifest = read(&format!("{path}/Cargo.toml"));
            assert!(
                under(&manifest, "[lints.clippy]").any(|line| line == deny)
                    || under(&manifest, "[lints]").any(|line| line == "workspace = true"),
                "{path}/Cargo.toml lets its crate use what clippy.toml lists"
            );
            // Clippy reads the first `.clippy.toml` or `clippy.toml` it finds, from the crate's
            // directory up: the workspace's.
            let home = root.join(&path);
            let first =
                home.ancestors().flat_map(|d| [d.join(".clippy.toml"), d.join("clippy.toml")]).find(|f| f.exists());
            assert_eq!(
                first,
                Some(root.join("clippy.toml")),
                "clippy reads another file than the workspace's clippy.toml for {path}"
            );
        }
    }
}
