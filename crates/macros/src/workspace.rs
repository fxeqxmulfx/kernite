//! The name of the workspace the crate is built in: the first namespace of its GPU kernels.

use std::path::{Path, PathBuf};

/// The workspace the crate is built in, by name, the first namespace its GPU kernels are in:
/// as its manifest has it, `name` in `[workspace.metadata.kgen]`, or the name of the directory
/// the manifest is in, with that manifest. A package of no workspace is one of its own, named
/// as the package is.
pub(crate) fn workspace() -> (String, Option<PathBuf>) {
    let package = std::env::var("CARGO_PKG_NAME").unwrap_or_default();
    let dir = std::env::var_os("CARGO_MANIFEST_DIR").map(PathBuf::from);
    workspace_from(package, dir.as_deref(), &|manifest| std::fs::read_to_string(manifest).ok())
}

/// [`workspace`], for the package `package` whose manifest is in `dir`: the nearest manifest of
/// a workspace from `dir` up names it. Each manifest is read by `read`, and one `read` does not
/// give is passed by: the walk reads no file itself.
fn workspace_from(
    package: String,
    dir: Option<&Path>,
    read: &dyn Fn(&Path) -> Option<String>,
) -> (String, Option<PathBuf>) {
    for dir in dir.into_iter().flat_map(Path::ancestors) {
        let manifest = dir.join("Cargo.toml");
        let Some(text) = read(&manifest) else { continue };
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

#[cfg(test)]
mod tests {
    use super::workspace_from;
    use std::path::{Path, PathBuf};

    /// A reader that gives each of `files`, a path and its text, and no other.
    fn files(files: &[(&str, &str)]) -> impl Fn(&Path) -> Option<String> {
        |path| files.iter().find(|(p, _)| Path::new(p) == path).map(|(_, text)| text.to_string())
    }

    #[test]
    fn the_workspace_is_named_by_its_metadata() {
        // A manifest with no `[workspace]`, only `[workspace.metadata.kgen]`, is a workspace's,
        // and the name may be in single quotes.
        let read = files(&[("/home/w/Cargo.toml", "[workspace.metadata.kgen]\nname = 'acme'\n")]);
        let got = workspace_from("app".into(), Some(Path::new("/home/w")), &read);
        assert_eq!(got, ("acme".into(), Some(PathBuf::from("/home/w/Cargo.toml"))));
        // `""` names the workspace nothing, rather than as its directory is: its kernels' names
        // leave it out. The manifest is still given: the macro includes it, so that the PTX is
        // built again when the name changes.
        let read = files(&[("/home/w/Cargo.toml", "[workspace.metadata.kgen]\nname = \"\"\n")]);
        let got = workspace_from("app".into(), Some(Path::new("/home/w")), &read);
        assert_eq!(got, (String::new(), Some(PathBuf::from("/home/w/Cargo.toml"))));
    }

    #[test]
    fn the_workspace_is_named_by_its_directory() {
        // From a member's directory up, its own manifest, of no workspace, and one that is not
        // there are passed by. The nearest manifest of a workspace names it by its directory,
        // though that manifest names a package too: its `[workspace.metadata.kgen]` has no
        // `name`, and no other table's `name` is read. One farther up does not name it.
        const ACME: &str = r#"
            [workspace]
            members = ["crates/*"]

            [workspace.metadata.kgen]

            [workspace.metadata.docs]
            name = "docs"

            [package]
            name = "root"

            [package.metadata.kgen]
            name = "root"
        "#;
        let read = files(&[
            ("/home/acme/crates/app/Cargo.toml", "[package]\nname = \"app\"\n"),
            ("/home/acme/Cargo.toml", ACME),
            ("/home/Cargo.toml", "[workspace]\n"),
        ]);
        let got = workspace_from("app".into(), Some(Path::new("/home/acme/crates/app")), &read);
        assert_eq!(got, ("acme".into(), Some(PathBuf::from("/home/acme/Cargo.toml"))));
        // A workspace in a directory with no name, `/`, is named as the package is.
        let read = files(&[("/Cargo.toml", "[workspace]\n")]);
        let got = workspace_from("app".into(), Some(Path::new("/app")), &read);
        assert_eq!(got, ("app".into(), Some(PathBuf::from("/Cargo.toml"))));
    }

    #[test]
    fn a_package_of_no_workspace_is_its_own() {
        // A package of no workspace is one of its own, named as the package is, not as its
        // directory is, with no manifest.
        let read = files(&[("/home/proj/Cargo.toml", "[package]\nname = \"app\"\n")]);
        assert_eq!(workspace_from("app".into(), Some(Path::new("/home/proj")), &read), ("app".into(), None));
        // So is a package of no directory, though every path has a workspace's manifest.
        assert_eq!(workspace_from("app".into(), None, &|_| Some("[workspace]\n".into())), ("app".into(), None));
    }
}
