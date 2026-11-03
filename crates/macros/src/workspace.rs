//! The name of the workspace the crate is built in: the first namespace of its GPU kernels.

use std::ffi::OsString;
use std::path::{Component, Path, PathBuf};

/// The workspace the crate is built in, by name, the first namespace its GPU kernels are in: the
/// one Cargo takes for the package. Its name is as its manifest has it, `name` in
/// `[workspace.metadata.kgen]`, or the name of the directory the manifest is in, with that
/// manifest. A package of no workspace is one of its own, named as the package is. The error says
/// that `name` is not a string, where it can be mended.
pub(crate) fn workspace() -> Result<(String, Option<PathBuf>), String> {
    let package = std::env::var("CARGO_PKG_NAME").unwrap_or_default();
    let dir = std::env::var_os("CARGO_MANIFEST_DIR").map(PathBuf::from);
    let home = cargo_home(std::env::var_os("CARGO_HOME"), std::env::home_dir());
    workspace_from(package, dir.as_deref(), home.as_deref(), &|manifest| std::fs::read_to_string(manifest).ok())
}

/// Cargo's home, where its walk for a workspace stops: `var`, the value of `CARGO_HOME`, unless it
/// is not set or empty; else `.cargo` in `user`, the user's home directory. A relative `var` is
/// made absolute from the current directory, which is rustc's: Cargo makes it absolute from the one
/// it was run in, which may be another.
fn cargo_home(var: Option<OsString>, user: Option<PathBuf>) -> Option<PathBuf> {
    match var.filter(|home| !home.is_empty()) {
        Some(home) => std::path::absolute(home).ok(),
        None => user.map(|user| user.join(".cargo")),
    }
}

/// The path of the workspace's name in a manifest.
const NAME: [&str; 4] = ["workspace", "metadata", "kgen", "name"];

/// [`workspace`], for the package `package` whose manifest is in `dir`, and Cargo's home `home`:
/// the workspace is [`root`]'s. Each manifest is read by `read`; one it does not give, or that
/// `parse` finds is not TOML, is passed by: the walk reads no file itself.
fn workspace_from(
    package: String,
    dir: Option<&Path>,
    home: Option<&Path>,
    read: &dyn Fn(&Path) -> Option<String>,
) -> Result<(String, Option<PathBuf>), String> {
    let Some((manifest, entries)) = dir.and_then(|dir| root(dir, home, read)) else { return Ok((package, None)) };
    let name = match entries.iter().find(|(path, _)| path.iter().take(NAME.len()).eq(NAME)) {
        Some((path, Some(name))) if *path == NAME => name.clone(),
        Some(_) => {
            return Err(format!(
                "`name` in `[workspace.metadata.kgen]` of {} is not a string: a workspace is named by one, as \
                 `name = \"kgen\"`",
                manifest.display()
            ));
        }
        None => {
            manifest.parent().and_then(Path::file_name).and_then(|name| name.to_str()).map_or(package, String::from)
        }
    };
    Ok((name, Some(manifest)))
}

/// The manifest of the workspace that Cargo takes for the package whose manifest is in `dir`, and
/// its entries, as Cargo's `find_root` finds it. Each manifest is read by `read`. It is the first
/// of these:
/// - The package's own manifest, if it has a `workspace` table.
/// - The manifest in the directory that the `workspace` key of the package's own `[package]` names,
///   if it has a `workspace` table. If it has not, the package is of no workspace.
/// - A manifest with a `workspace` table, unless it [`excludes`] the package, in a directory above,
///   the nearest first. The walk goes up to Cargo's home `home`, whose own manifest is the last,
///   and never into a directory that ends in `target/package`, where `cargo package` builds the
///   crate it checks.
/// - The manifest in the directory that the `workspace` key of the `[package]` of one of those
///   manifests names, the nearest key first, if it has a `workspace` table. A key that names none
///   is passed by.
fn root(dir: &Path, home: Option<&Path>, read: &dyn Fn(&Path) -> Option<String>) -> Option<(PathBuf, Vec<Entry>)> {
    let read = |manifest: &Path| read(manifest).and_then(|text| parse(&text));
    let package = dir.join("Cargo.toml");
    let mut pointed = None;
    for above in dir.ancestors() {
        // The package's own directory is not one Cargo stops at: it looks above it.
        let own = above == dir;
        if !own && above.ends_with("target/package") {
            break;
        }
        let manifest = above.join("Cargo.toml");
        if let Some(entries) = read(&manifest) {
            if has_workspace(&entries) {
                if own || !excludes(&entries, above, &package) {
                    return Some((manifest, entries));
                }
            } else if let Some(key) = strings(&entries, &["package", "workspace"]).next() {
                let manifest = normalize(&above.join(key).join("Cargo.toml"));
                let named = read(&manifest).filter(|entries| has_workspace(entries)).map(|entries| (manifest, entries));
                // The package's own key is followed at once. An ancestor's is followed if no
                // workspace's manifest is above.
                if own {
                    return named;
                }
                pointed = pointed.or(named);
            }
        }
        if !own && home == Some(above) {
            break;
        }
    }
    pointed
}

/// Whether the manifest whose `entries` these are has a `workspace` table.
fn has_workspace(entries: &[Entry]) -> bool {
    entries.iter().any(|(path, _)| path[0] == "workspace")
}

/// Whether the workspace whose manifest `entries` are in `dir` leaves out the package whose
/// manifest is `package`, as Cargo's `is_excluded` says: the package is under a path of
/// `exclude` and under none of `members`, each from `dir`, whole, not as a glob.
fn excludes(entries: &[Entry], dir: &Path, package: &Path) -> bool {
    let under = |key| strings(entries, &["workspace", key, ""]).any(|path| package.starts_with(dir.join(path)));
    under("exclude") && !under("members")
}

/// The strings at `path` among `entries`: the one a key has, or those of an array, which have an
/// empty part after the array's path.
fn strings<'a>(entries: &'a [Entry], path: &'a [&str]) -> impl Iterator<Item = &'a str> {
    entries.iter().filter(move |(at, _)| at == path).filter_map(|(_, string)| string.as_deref())
}

/// `path` with each `.` left out and each `..` taking the part before it, as Cargo reads the
/// directory of a `workspace` key: by the path alone, no file looked at.
fn normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for part in path.components() {
        if part == Component::ParentDir {
            out.pop();
        } else {
            out.push(part);
        }
    }
    out
}

/// A table or a key of a manifest, by its path from the top, and its string, if its value is one.
/// An element of an array is an entry of its own if it is a string, at the array's path with an
/// empty part after it.
type Entry = (Vec<String>, Option<String>);

/// Each table and key of `text`, a TOML document, in order, or `None` where the reader finds it is
/// not one. A value that is not a string is passed by whole, but for the keys of an inline table
/// and the strings of an array, which are entries of their own.
fn parse(text: &str) -> Option<Vec<Entry>> {
    let mut toml = Toml(text.strip_prefix('\u{feff}').unwrap_or(text));
    let (mut entries, mut table, mut arrays) = (vec![], vec![], vec![]);
    loop {
        toml.skip();
        if toml.0.is_empty() {
            return Some(entries);
        }
        if toml.eat('[') {
            let array = toml.eat('[');
            // A table under an array of tables is in its last element: the empty part after it.
            table = vec![];
            for part in toml.key()? {
                if arrays.contains(&table) {
                    table.push(String::new());
                }
                table.push(part);
            }
            if !toml.eat(']') || array && !toml.eat(']') {
                return None;
            }
            entries.push((table.clone(), None));
            if array {
                arrays.push(table.clone());
                // The keys that follow are an element of an array's, not of a table of that name.
                table.push(String::new());
            }
        } else {
            toml.pair(&table, &mut entries)?;
        }
    }
}

/// TOML's whitespace, spaces and tabs, and its newlines.
const BLANK: [char; 4] = [' ', '\t', '\r', '\n'];

/// The rest of a TOML document to read.
struct Toml<'a>(&'a str);

impl Toml<'_> {
    /// Passes by what comes before the next token: whitespace, newlines and comments.
    fn skip(&mut self) {
        self.0 = self.0.trim_start_matches(BLANK);
        while let Some(comment) = self.0.strip_prefix('#') {
            self.0 = comment.split_once('\n').map_or("", |(_, rest)| rest).trim_start_matches(BLANK);
        }
    }

    /// Passes by `c`, if it is the next token.
    fn eat(&mut self, c: char) -> bool {
        self.skip();
        let found = self.0.strip_prefix(c);
        self.0 = found.unwrap_or(self.0);
        found.is_some()
    }

    /// A key, bare, quoted or dotted, as the names of its parts.
    fn key(&mut self) -> Option<Vec<String>> {
        let mut parts = vec![];
        loop {
            self.skip();
            if self.0.starts_with(['"', '\'']) {
                parts.push(self.string()?);
            } else {
                let end = self.0.find(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, '_' | '-')));
                let (bare, rest) = self.0.split_at(end.unwrap_or(self.0.len()));
                if bare.is_empty() {
                    return None;
                }
                parts.push(bare.to_string());
                self.0 = rest;
            }
            if !self.eat('.') {
                return Some(parts);
            }
        }
    }

    /// A string, of any of the four kinds, as the text it says: its escapes read.
    fn string(&mut self) -> Option<String> {
        let quote = self.0.chars().next()?;
        // A long string is fenced by three quotes, and holds up to two together.
        let fence = if self.0.starts_with(&quote.to_string().repeat(3)) { 3 } else { 1 };
        let mut rest = &self.0[fence..];
        if fence == 3 {
            rest = rest.strip_prefix("\r\n").or_else(|| rest.strip_prefix('\n')).unwrap_or(rest);
        }
        let mut text = String::new();
        loop {
            let c = rest.chars().next()?;
            if c == quote {
                let run = rest.len() - rest.trim_start_matches(quote).len();
                let ends = run >= fence;
                // The last quotes of a run end a long string, and the ones before them are in it.
                text.extend(std::iter::repeat_n(quote, if ends { run - fence } else { run }));
                rest = &rest[run..];
                if ends {
                    self.0 = rest;
                    return Some(text);
                }
                continue;
            }
            rest = &rest[c.len_utf8()..];
            if c != '\\' || quote == '\'' {
                text.push(c);
                continue;
            }
            let escape = rest.chars().next()?;
            rest = &rest[escape.len_utf8()..];
            text.push(match escape {
                'b' => '\u{8}',
                't' => '\t',
                'n' => '\n',
                'f' => '\u{c}',
                'r' => '\r',
                'e' => '\u{1b}',
                '"' | '\\' => escape,
                'x' | 'u' | 'U' => {
                    let digits = match escape {
                        'x' => 2,
                        'u' => 4,
                        _ => 8,
                    };
                    let code = u32::from_str_radix(rest.get(..digits)?, 16).ok()?;
                    rest = &rest[digits..];
                    char::from_u32(code)?
                }
                // A backslash at the end of a line, in a long string, passes by the whitespace
                // after it.
                _ if fence == 3 && BLANK.contains(&escape) => {
                    rest = rest.trim_start_matches(BLANK);
                    continue;
                }
                _ => return None,
            });
        }
    }

    /// A key and its value, `key = value`, in the table at `table`, with the entries it makes.
    fn pair(&mut self, table: &[String], entries: &mut Vec<Entry>) -> Option<()> {
        let key = self.key()?;
        if !self.eat('=') {
            return None;
        }
        self.value([table, &key].concat(), entries)
    }

    /// The value at `path`, with the entries it makes: its own, a string or not; and, if it is an
    /// inline table, those of its keys. Those of a value in an array are dropped, but for the
    /// strings of the array: no table of the document is there.
    fn value(&mut self, path: Vec<String>, entries: &mut Vec<Entry>) -> Option<()> {
        self.skip();
        if self.0.starts_with(['"', '\'']) {
            entries.push((path, Some(self.string()?)));
        } else if self.eat('[') {
            entries.push((path.clone(), None));
            let element = [path, vec![String::new()]].concat();
            while !self.eat(']') {
                if !self.eat(',') {
                    let mut item = vec![];
                    self.value(element.clone(), &mut item)?;
                    entries.extend(item.into_iter().filter(|(at, string)| *at == element && string.is_some()));
                }
            }
        } else if self.eat('{') {
            entries.push((path.clone(), None));
            while !self.eat('}') {
                if !self.eat(',') {
                    self.pair(&path, entries)?;
                }
            }
        } else {
            // A number, a boolean or a date: up to the end of the line, a comment, or what
            // closes it.
            let end = self.0.find(['\n', '#', ',', ']', '}']).unwrap_or(self.0.len());
            if end == 0 {
                return None;
            }
            self.0 = &self.0[end..];
            entries.push((path, None));
        }
        Some(())
    }
}

#[cfg(test)]
mod tests {
    use super::{cargo_home, workspace_from};
    use std::path::{Path, PathBuf};

    /// What `workspace_from` gives.
    type Walked = Result<(String, Option<PathBuf>), String>;

    /// A reader that gives each of `files`, a path and its text, and no other.
    fn files(files: &[(&str, &str)]) -> impl Fn(&Path) -> Option<String> {
        |path| files.iter().find(|(p, _)| Path::new(p) == path).map(|(_, text)| text.to_string())
    }

    #[test]
    fn the_workspace_is_named_by_its_metadata() {
        // A manifest with no `[workspace]`, only `[workspace.metadata.kgen]`, is a workspace's,
        // and the name may be in single quotes.
        let read = files(&[("/home/w/Cargo.toml", "[workspace.metadata.kgen]\nname = 'acme'\n")]);
        let got = workspace_from("app".into(), Some(Path::new("/home/w")), None, &read);
        assert_eq!(got, Ok(("acme".into(), Some(PathBuf::from("/home/w/Cargo.toml")))));
        // `""` names the workspace nothing, rather than as its directory is: its kernels' names
        // leave it out. The manifest is still given: the macro includes it, so that the PTX is
        // built again when the name changes.
        let read = files(&[("/home/w/Cargo.toml", "[workspace.metadata.kgen]\nname = \"\"\n")]);
        let got = workspace_from("app".into(), Some(Path::new("/home/w")), None, &read);
        assert_eq!(got, Ok((String::new(), Some(PathBuf::from("/home/w/Cargo.toml")))));
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
        let got = workspace_from("app".into(), Some(Path::new("/home/acme/crates/app")), None, &read);
        assert_eq!(got, Ok(("acme".into(), Some(PathBuf::from("/home/acme/Cargo.toml")))));
        // A workspace in a directory with no name, `/`, is named as the package is.
        let read = files(&[("/Cargo.toml", "[workspace]\n")]);
        let got = workspace_from("app".into(), Some(Path::new("/app")), None, &read);
        assert_eq!(got, Ok(("app".into(), Some(PathBuf::from("/Cargo.toml")))));
    }

    #[test]
    fn a_package_of_no_workspace_is_its_own() {
        // A package of no workspace is one of its own, named as the package is, not as its
        // directory is, with no manifest.
        let read = files(&[("/home/proj/Cargo.toml", "[package]\nname = \"app\"\n")]);
        assert_eq!(workspace_from("app".into(), Some(Path::new("/home/proj")), None, &read), Ok(("app".into(), None)));
        // So is a package of no directory, though every path has a workspace's manifest.
        assert_eq!(
            workspace_from("app".into(), None, None, &|_| Some("[workspace]\n".into())),
            Ok(("app".into(), None))
        );
    }

    /// What `workspace_from` makes of one manifest, the only one there is, `/home/w/Cargo.toml`,
    /// for the package `app` in `/home/w`.
    fn only(manifest: &str) -> Walked {
        workspace_from("app".into(), Some(Path::new("/home/w")), None, &files(&[("/home/w/Cargo.toml", manifest)]))
    }

    /// What `only` makes of a manifest of a workspace named `name`.
    fn named(name: &str) -> Walked {
        Ok((name.into(), Some(PathBuf::from("/home/w/Cargo.toml"))))
    }

    #[test]
    fn a_name_is_a_string_of_any_of_the_four_kinds_with_its_escapes_read() {
        // Each is `name = ..` in `[workspace.metadata.kgen]`, with the name it says. A long string
        // trims the newline after its quotes, and a backslash at the end of a line with the
        // whitespace that follows, TOML's: a U+00A0 or a U+3000 is the name's. It holds one quote
        // or two, and, at its end, two before its own.
        for (value, name) in [
            (r#""acme""#, "acme"),
            ("'acme'", "acme"),
            (r#""""acme""""#, "acme"),
            ("'''acme'''", "acme"),
            ("\"\"\"\nacme\"\"\"", "acme"),
            ("'''\r\nacme'''", "acme"),
            (r#""a\"b""#, "a\"b"),
            (r#""a\U00000061\x61""#, "aaa"),
            (r#""\b\t\n\f\r\e\"\\""#, "\u{8}\t\n\u{c}\r\u{1b}\"\\"),
            (r"'a\b'", r"a\b"),
            ("'''a\\\nb'''", "a\\\nb"),
            ("\"\"\"a \\ \t\n\n   b\"\"\"", "a b"),
            ("\"\"\"\\\n\u{a0}\u{3000}acme\"\"\"", "\u{a0}\u{3000}acme"),
            (r#""""a""b""""#, "a\"\"b"),
            ("\"\"\"a\"\"\"\"\"", "a\"\""),
            ("'''a'''''", "a''"),
            ("\"é\\u00e9\"", "éé"),
        ] {
            let manifest = format!("[workspace.metadata.kgen]\nname = {value}\n");
            assert_eq!(only(&manifest), named(name), "{value}");
        }
    }

    #[test]
    fn a_name_is_found_by_any_form_of_key_and_table() {
        // Each of these says `workspace.metadata.kgen.name`: with dotted keys, inline tables,
        // quoted keys, spaces about the dots, a comment, and CRLF for newlines.
        for manifest in [
            "[workspace.metadata]\nkgen.name = \"acme\"\n",
            "[workspace.metadata]\nkgen = { name = \"acme\" }\n",
            "[workspace]\nmetadata.kgen.name = \"acme\"\n",
            "workspace.metadata.kgen.name = \"acme\"\n",
            "workspace = { metadata = { kgen = { name = \"acme\" } } }\n",
            "[workspace.metadata.\"kgen\"]\nname = \"acme\"\n",
            "[workspace.metadata.'kgen']\nname = \"acme\"\n",
            "[\"workspace\".metadata.kgen]\nname = \"acme\"\n",
            "[workspace.metadata.\"k\\u0067en\"]\nname = \"acme\"\n",
            "[workspace . metadata . kgen]\nname = \"acme\"\n",
            "[ workspace.metadata.kgen ] # kgen\n\"name\" = \"acme\" # the name\n",
            "[workspace.metadata]\nkgen . name = \"acme\"\n",
            "[workspace.metadata]\nkgen = {\n  name = \"acme\", # a trailing comma\n}\n",
            "[workspace.metadata.kgen]\r\nname = \"acme\"\r\n",
            "\u{feff}[workspace.metadata.kgen]\nname = \"acme\"\n",
        ] {
            assert_eq!(only(manifest), named("acme"), "{manifest:?}");
        }
    }

    #[test]
    fn a_name_that_is_not_a_string_is_refused_naming_the_manifest() {
        // Cargo reads each of these as a `workspace.metadata.kgen.name` that is not a string: a
        // number, a boolean, an array, an inline table, a date, or a table: of its own, by a dotted
        // key, or by a table inside it.
        let refused = Err("`name` in `[workspace.metadata.kgen]` of /home/w/Cargo.toml is not a string: a workspace \
                            is named by one, as `name = \"kgen\"`"
            .into());
        for manifest in [
            "[workspace.metadata.kgen]\nname = 5\n",
            "[workspace.metadata.kgen]\nname = true\n",
            "[workspace.metadata.kgen]\nname = [\"acme\"]\n",
            "[workspace.metadata.kgen]\nname = { a = \"acme\" }\n",
            "[workspace.metadata.kgen]\nname = 1979-05-27 07:32:00\n",
            "[workspace.metadata.kgen.name]\na = \"acme\"\n",
            "workspace.metadata.kgen.name = 5\n",
            "[workspace.metadata]\nkgen = { name = 5 }\n",
            "[workspace.metadata.kgen]\nname.first = \"acme\"\n",
            "[workspace.metadata.kgen.name.first]\n",
        ] {
            assert_eq!(only(manifest), refused, "{manifest:?}");
        }
        // The nearest manifest of a workspace is the one named: a farther one is not read.
        let read = files(&[
            ("/home/w/Cargo.toml", "[workspace.metadata.kgen]\nname = 5\n"),
            ("/home/Cargo.toml", "[workspace]\n"),
        ]);
        assert_eq!(workspace_from("app".into(), Some(Path::new("/home/w")), None, &read), refused);
    }

    #[test]
    fn a_name_that_is_not_where_cargo_reads_it_is_passed_by() {
        // The workspace is named as its directory is. `name` is another table's, or `kgen` or
        // `metadata` is not a table: an array of tables, or an array with a table in it. A table
        // under an array of tables is in the array's last element.
        for manifest in [
            "[workspace]\n[package.metadata.kgen]\nname = \"acme\"\n",
            "[workspace.metadata]\nkgen.nam = \"acme\"\n",
            "[workspace.metadata.kgen.sub]\nname = \"acme\"\n",
            "[workspace.metadata]\nsub.kgen.name = \"acme\"\n",
            "[[workspace.metadata.kgen]]\nname = \"acme\"\n",
            "[workspace.metadata]\nx = [{ kgen = { name = \"acme\" } }]\n",
            "[workspace.metadata]\nkgen = [{ name = \"acme\" }]\n",
            "[[workspace.metadata]]\n[workspace.metadata.kgen]\nname = \"acme\"\n",
            "[[workspace.metadata]]\n[[workspace.metadata.kgen]]\n[workspace.metadata.kgen.x]\n",
        ] {
            assert_eq!(only(manifest), named("w"), "{manifest:?}");
        }
        // A table of an array of tables is that array's last, and `kgen` is the table beside it.
        let manifest = "[[workspace.metadata.kgen.x]]\n[workspace.metadata.kgen]\nname = \"acme\"\n";
        assert_eq!(only(manifest), named("acme"));
    }

    #[test]
    fn a_manifest_has_a_workspace_by_any_form_of_key_and_table() {
        // Each is a workspace's, named as its directory is: with the workspace as a dotted key or
        // an inline table at the top, an array of tables beside a package, a byte-order mark, a
        // `[workspace]` after an array that has brackets in its strings and comments.
        for manifest in [
            "workspace.members = [\"app\"]\n\n[package]\nname = \"w\"\n",
            "workspace = { members = [\"app\"] }\n\n[package]\nname = \"w\"\n",
            "[package]\nname = \"w\"\n\n[[workspace.metadata.x]]\n",
            "[package]\nname = \"w\"\n\n[workspace.metadata.kgen]\n",
            "\u{feff}[workspace]\nmembers = [\"app\"]\n",
            "[package]\nkeywords = [\"[workspace]\", ['x'], # ]\n  [[\"]\"]],\n]\n\n[workspace]\n",
            "\"workspace\" = {}\n",
            "[ workspace ]\n",
        ] {
            assert_eq!(only(manifest), named("w"), "{manifest:?}");
        }
    }

    #[test]
    fn a_manifest_has_no_workspace_where_workspace_is_in_a_string_a_comment_or_another_table() {
        // Each is passed by: the package is of no workspace.
        for manifest in [
            "[package]\ndescription = \"\"\"\n[workspace]\n\"\"\"\n",
            "[package]\ndescription = '''\n[workspace]\n'''\n",
            "[package]\n# [workspace]\n",
            "[package]\nkeywords = [\"[workspace]\"]\n",
            "[package]\nworkspace = \"../w\"\n",
            "[dependencies]\nworkspace.version = \"1\"\n",
            "[package]\nx = [{ workspace = 1 }]\n",
            "[package.metadata]\nx = { workspace = { members = [] } }\n",
            "[lib]\nworkspace = \".\"\n",
        ] {
            assert_eq!(only(manifest), Ok(("app".into(), None)), "{manifest:?}");
        }
    }

    #[test]
    fn a_manifest_that_is_not_toml_is_passed_by() {
        // As one that is not there is: the walk goes on, here to a workspace one directory up.
        let up = ("/home/Cargo.toml", "[workspace]\n");
        for broken in ["name = \"never ends\n[workspace]\n", "[workspace\n", "workspace = \n", "[workspace] = 1 }"] {
            let manifests = [("/home/w/Cargo.toml", broken), up];
            let got = workspace_from("app".into(), Some(Path::new("/home/w")), None, &files(&manifests));
            assert_eq!(got, Ok(("home".into(), Some(PathBuf::from("/home/Cargo.toml")))), "{broken:?}");
        }
    }

    /// What `workspace_from` gives for the package `app` in `dir`, Cargo's home `home`, and the
    /// manifests `manifests`.
    fn walk(dir: &str, home: Option<&str>, manifests: &[(&str, &str)]) -> Walked {
        workspace_from("app".into(), Some(Path::new(dir)), home.map(Path::new), &files(manifests))
    }

    /// What `walk` gives for the workspace `name`, whose manifest is `manifest`.
    fn is(name: &str, manifest: &str) -> Walked {
        Ok((name.into(), Some(PathBuf::from(manifest))))
    }

    /// What `walk` gives for a package of no workspace.
    fn alone() -> Walked {
        Ok(("app".into(), None))
    }

    /// The manifest of a package.
    const PACKAGE: &str = "[package]\nname = \"app\"\n";
    /// The manifest of a workspace.
    const WORKSPACE: &str = "[workspace]\n";

    #[test]
    fn the_walk_stops_at_cargos_home_after_its_manifest() {
        // A crate unpacked from a registry, under the home, is a workspace of its own, though a
        // workspace's manifest is above the home.
        let dep = "/home/u/.cargo/registry/src/index/dep-1.0.0";
        let manifests =
            [("/home/u/.cargo/registry/src/index/dep-1.0.0/Cargo.toml", PACKAGE), ("/home/u/Cargo.toml", WORKSPACE)];
        assert_eq!(walk(dep, Some("/home/u/.cargo"), &manifests), alone());
        // With no home, or one that is not above the package, the walk goes on to that workspace.
        for home in [None, Some("/home/v/.cargo")] {
            assert_eq!(walk(dep, home, &manifests), is("u", "/home/u/Cargo.toml"));
        }
        // The home's own manifest is the last one read: a workspace's there is the package's.
        let manifests = [("/home/u/.cargo/Cargo.toml", WORKSPACE), ("/home/u/Cargo.toml", WORKSPACE)];
        assert_eq!(walk(dep, Some("/home/u/.cargo"), &manifests), is(".cargo", "/home/u/.cargo/Cargo.toml"));
        // The package's own directory is not one the walk stops at: it looks above, as Cargo does.
        let manifests = [("/home/u/Cargo.toml", WORKSPACE)];
        assert_eq!(walk("/home/u/app", Some("/home/u/app"), &manifests), is("u", "/home/u/Cargo.toml"));
    }

    #[test]
    fn the_walk_never_goes_into_target_package() {
        // `cargo package` builds the crate it checks at `target/package/<name>`, a workspace of its
        // own, though the workspace's manifest is above, and one is in `target/package` itself.
        let manifests = [("/home/w/Cargo.toml", WORKSPACE), ("/home/w/target/package/Cargo.toml", WORKSPACE)];
        for dir in ["/home/w/target/package/app-0.1.0", "/home/w/target/package/x/y/app"] {
            assert_eq!(walk(dir, None, &manifests), alone(), "{dir}");
        }
        // A workspace's manifest nearer than `target/package` is the package's.
        let manifests = [("/home/w/Cargo.toml", WORKSPACE), ("/home/w/target/package/x/Cargo.toml", WORKSPACE)];
        assert_eq!(
            walk("/home/w/target/package/x/app", None, &manifests),
            is("x", "/home/w/target/package/x/Cargo.toml")
        );
        // Only a directory that ends in those two parts: others are walked through.
        for dir in ["/home/w/target/other/app", "/home/w/package/app", "/home/w/my-target/package/app"] {
            assert_eq!(walk(dir, None, &manifests), is("w", "/home/w/Cargo.toml"), "{dir}");
        }
    }

    #[test]
    fn a_package_in_target_package_keeps_its_workspace_and_its_own() {
        // A member of the workspace may be the directory `target/package`: the walk does not
        // stop at the package's own directory, but looks above it. A workspace's manifest
        // there is its own.
        let manifests = [("/home/w/Cargo.toml", "[workspace]\nmembers = [\"target/package\"]\n")];
        assert_eq!(walk("/home/w/target/package", None, &manifests), is("w", "/home/w/Cargo.toml"));
        let manifests = [("/home/w/Cargo.toml", WORKSPACE), ("/home/w/target/package/Cargo.toml", WORKSPACE)];
        assert_eq!(
            walk("/home/w/target/package", None, &manifests),
            is("package", "/home/w/target/package/Cargo.toml")
        );
    }

    #[test]
    fn a_package_follows_the_workspace_its_manifest_names() {
        // `workspace = ".."` in `[package]` is the directory of the workspace's manifest from the
        // package's, read by its parts, as Cargo does: each `.` and `..` taken, no file looked at;
        // or an absolute one. A workspace's manifest above is passed by.
        for manifest in [
            "[package]\nname = \"app\"\nworkspace = \"../ws\"\n",
            "package.workspace = '../ws'\n",
            "package = { workspace = \"../ws\" }\n",
            "[package]\nworkspace = \"./../x/../ws/.\"\n",
            "[package]\nworkspace = \"/home/ws\"\n",
        ] {
            let manifests = [
                ("/home/app/Cargo.toml", manifest),
                ("/home/ws/Cargo.toml", WORKSPACE),
                ("/home/Cargo.toml", WORKSPACE),
            ];
            assert_eq!(walk("/home/app", None, &manifests), is("ws", "/home/ws/Cargo.toml"), "{manifest:?}");
        }
        // A key that is not a string names nothing, nor does one in another table.
        for manifest in
            ["[package]\nworkspace = 5\n", "[package]\nworkspace = [\"../ws\"]\n", "[lib]\nworkspace = \"../ws\"\n"]
        {
            let manifests = [("/home/app/Cargo.toml", manifest), ("/home/ws/Cargo.toml", WORKSPACE)];
            assert_eq!(walk("/home/app", None, &manifests), alone(), "{manifest:?}");
        }
    }

    #[test]
    fn an_ancestor_names_the_workspace_where_none_is_above_it() {
        // The key is read from the ancestor's directory, not the package's.
        let named = ("/home/t/Cargo.toml", "[package]\nname = \"t\"\nworkspace = \"../ws\"\n");
        let ws = ("/home/ws/Cargo.toml", WORKSPACE);
        assert_eq!(walk("/home/t/app", None, &[named, ws]), is("ws", "/home/ws/Cargo.toml"));
        // A workspace's manifest above is the package's instead, for Cargo looks for one before it
        // follows a key; unless it excludes the package, or is past Cargo's home.
        let above = ("/home/Cargo.toml", WORKSPACE);
        assert_eq!(walk("/home/t/app", None, &[named, ws, above]), is("home", "/home/Cargo.toml"));
        let excluding = ("/home/Cargo.toml", "[workspace]\nexclude = [\"t\"]\n");
        assert_eq!(walk("/home/t/app", None, &[named, ws, excluding]), is("ws", "/home/ws/Cargo.toml"));
        assert_eq!(walk("/home/t/app", Some("/home/t"), &[named, ws, above]), is("ws", "/home/ws/Cargo.toml"));
        // Of two keys, the nearer is followed.
        let nearer = ("/home/u/t/Cargo.toml", "[package]\nname = \"n\"\nworkspace = \"../../v\"\n");
        let farther = ("/home/u/Cargo.toml", "[package]\nname = \"f\"\nworkspace = \"../ws\"\n");
        let v = ("/home/v/Cargo.toml", WORKSPACE);
        assert_eq!(walk("/home/u/t/app", None, &[nearer, farther, ws, v]), is("v", "/home/v/Cargo.toml"));
    }

    #[test]
    fn a_workspace_that_excludes_the_package_is_passed_by() {
        // Cargo's `is_excluded`: the package is under a path of `exclude`, and under none of
        // `members`, each from the workspace's directory, whole: a glob is no path. The walk goes
        // on to the workspace above.
        const EXCLUDING: &str = "[workspace]\nexclude = [\"skip\"]\n";
        for (workspace, dir, excluded) in [
            (EXCLUDING, "/home/w/skip", true),
            (EXCLUDING, "/home/w/skip/inner", true),
            ("[workspace]\nmembers = [\"other\"]\nexclude = [\"skip\"]\n", "/home/w/other", false),
            ("[workspace]\nmembers = [\"skip\"]\nexclude = [\"skip\"]\n", "/home/w/skip", false),
            ("[workspace]\nmembers = [\"skip/inner\"]\nexclude = [\"skip\"]\n", "/home/w/skip/inner", false),
            ("[workspace]\nmembers = [\"c/*\"]\nexclude = [\"c/skip\"]\n", "/home/w/c/skip", true),
            ("[workspace]\nexclude = [\n  \"other\", # not this one\n  \"skip\",\n]\n", "/home/w/skip", true),
            ("workspace = { members = [\"a\"], exclude = [\"skip\"] }\n", "/home/w/skip", true),
            ("workspace.exclude = [\"skip\"]\n", "/home/w/skip", true),
            ("[workspace]\nexclude = [\"./skip\"]\n", "/home/w/skip", true),
            ("[workspace]\nexclude = ['/home/w/skip']\n", "/home/w/skip", true),
        ] {
            let manifests = [("/home/w/Cargo.toml", workspace), ("/home/Cargo.toml", WORKSPACE)];
            let want = if excluded { is("home", "/home/Cargo.toml") } else { is("w", "/home/w/Cargo.toml") };
            assert_eq!(walk(dir, None, &manifests), want, "{workspace:?} for {dir}");
        }
        // The package's own manifest is the workspace's, though its `exclude` names the package.
        let manifests = [("/home/w/Cargo.toml", "[workspace]\nexclude = [\".\"]\n"), ("/home/Cargo.toml", WORKSPACE)];
        assert_eq!(walk("/home/w", None, &manifests), is("w", "/home/w/Cargo.toml"));
    }

    #[test]
    fn cargos_home_is_cargo_home_else_dot_cargo_of_the_user() {
        let user = || Some(PathBuf::from("/home/u"));
        let dot_cargo = Some(PathBuf::from("/home/u/.cargo"));
        // `CARGO_HOME`, if it is set and not empty, whoever the user is.
        assert_eq!(cargo_home(Some("/opt/cargo".into()), user()), Some(PathBuf::from("/opt/cargo")));
        assert_eq!(cargo_home(Some("/opt/cargo".into()), None), Some(PathBuf::from("/opt/cargo")));
        // Else `.cargo` in the user's home, if there is one.
        assert_eq!(cargo_home(Some("".into()), user()), dot_cargo);
        assert_eq!(cargo_home(None, user()), dot_cargo);
        assert_eq!(cargo_home(None, None), None);
        // A relative one is from the current directory, which is rustc's: Cargo's may be another.
        let here = std::env::current_dir().unwrap();
        assert_eq!(cargo_home(Some(".cargo".into()), user()), Some(here.join(".cargo")));
    }
}
