//! The name of the workspace the crate is built in: the first namespace of its GPU kernels.

use std::path::PathBuf;

/// The workspace the crate is built in, by name, the first namespace its GPU kernels are in:
/// as its manifest has it, `name` in `[workspace.metadata.kgen]`, or the name of the directory
/// the manifest is in, with that manifest. A package of no workspace is one of its own, named
/// as the package is.
pub(crate) fn workspace() -> (String, Option<PathBuf>) {
    let package = std::env::var("CARGO_PKG_NAME").unwrap_or_default();
    let Some(dir) = std::env::var_os("CARGO_MANIFEST_DIR").map(PathBuf::from) else {
        return (package, None);
    };
    for dir in dir.ancestors() {
        let manifest = dir.join("Cargo.toml");
        let Ok(text) = std::fs::read_to_string(&manifest) else { continue };
        if !text.lines().filter_map(table).any(|t| t == "workspace" || t.starts_with("workspace.")) {
            continue;
        }
        let named = metadata_name(&text).or_else(|| dir.file_name()?.to_str().map(String::from));
        return (named.unwrap_or(package), Some(manifest));
    }
    (package, None)
}

/// The table a line of a manifest opens, `[name]`, if it opens one.
fn table(line: &str) -> Option<&str> {
    line.trim().strip_prefix('[')?.split(']').next().map(str::trim)
}

/// `name = ".."` in `[workspace.metadata.kgen]` of `manifest`, if it is there.
fn metadata_name(manifest: &str) -> Option<String> {
    let mut inside = false;
    for line in manifest.lines() {
        if let Some(t) = table(line) {
            inside = t == "workspace.metadata.kgen";
            continue;
        }
        let Some((key, value)) = line.split_once('=').filter(|_| inside) else { continue };
        if key.trim() == "name" {
            let value = value.trim();
            let quote = value.chars().next().filter(|&c| c == '"' || c == '\'')?;
            return value[1..].split(quote).next().map(String::from);
        }
    }
    None
}
