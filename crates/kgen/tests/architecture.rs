//! The dependency rule: each crate depends only on the layers inside it, and nothing depends
//! on a crate outside the workspace. Every crate must have a line here, so a new one cannot
//! join without its place being decided.

use std::fs;
use std::path::Path;

/// Each crate and the crates it may depend on.
const RULE: &[(&str, &[&str])] = &[
    ("kgen-core", &[]),
    ("kgen-compile", &["kgen-core"]),
    ("kgen-x86", &["kgen-core"]),
    ("kgen", &["kgen-core", "kgen-compile", "kgen-x86"]),
    ("kgen-macros", &["kgen"]),
];

/// The package name and every dependency of the manifest at `path`, of any kind.
fn manifest(path: &Path) -> (String, Vec<String>) {
    let text = fs::read_to_string(path).unwrap();
    let (mut name, mut deps, mut section) = (None, vec![], String::new());
    for line in text.lines().map(str::trim) {
        if line.starts_with('[') {
            section = line.trim_matches(['[', ']']).to_string();
            continue;
        }
        let Some((key, value)) = line.split_once('=') else { continue };
        let key = key.trim();
        if section == "package" && key == "name" {
            name = Some(value.trim().trim_matches('"').to_string());
        }
        if section.ends_with("dependencies") && !key.is_empty() && !key.starts_with('#') {
            deps.push(key.to_string());
        }
    }
    (name.unwrap(), deps)
}

#[test]
fn every_crate_depends_only_inward() {
    let crates = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let mut seen = vec![];
    for entry in fs::read_dir(&crates).unwrap() {
        let path = entry.unwrap().path().join("Cargo.toml");
        let (name, deps) = manifest(&path);
        let allowed =
            RULE.iter().find(|(n, _)| *n == name).unwrap_or_else(|| panic!("{name} has no place in the rule")).1;
        for dep in &deps {
            assert!(allowed.contains(&dep.as_str()), "{name} depends on {dep}; it may depend on {allowed:?}");
        }
        seen.push(name);
    }
    for (name, _) in RULE {
        assert!(seen.iter().any(|s| s == name), "{name} is in the rule but not in crates/");
    }
}
