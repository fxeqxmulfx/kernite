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

/// The path of the workspace's name in a manifest.
const NAME: [&str; 4] = ["workspace", "metadata", "kgen", "name"];

/// [`workspace`], for the package `package` whose manifest is in `dir`: the nearest manifest of
/// a workspace from `dir` up names it. Each manifest is read by `read`; one it does not give, or
/// that `parse` finds is not TOML, is passed by: the walk reads no file itself.
fn workspace_from(
    package: String,
    dir: Option<&Path>,
    read: &dyn Fn(&Path) -> Option<String>,
) -> (String, Option<PathBuf>) {
    for dir in dir.into_iter().flat_map(Path::ancestors) {
        let manifest = dir.join("Cargo.toml");
        let Some(entries) = read(&manifest).and_then(|text| parse(&text)) else { continue };
        if !entries.iter().any(|(path, _)| path[0] == "workspace") {
            continue;
        }
        let named = entries.iter().find(|(path, _)| *path == NAME).and_then(|(_, name)| name.clone());
        let named = named.or_else(|| dir.file_name()?.to_str().map(String::from));
        return (named.unwrap_or(package), Some(manifest));
    }
    (package, None)
}

/// A table or a key of a manifest, by its path from the top, and its string, if its value is one.
type Entry = (Vec<String>, Option<String>);

/// Each table and key of `text`, a TOML document, in order, or `None` where the reader finds it
/// is not one. A value that is not a string is passed by whole, but for the keys of an inline
/// table, which are entries of their own.
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
    /// inline table, those of its keys. Those of a value in an array are dropped: no table of the
    /// document is there.
    fn value(&mut self, path: Vec<String>, entries: &mut Vec<Entry>) -> Option<()> {
        self.skip();
        if self.0.starts_with(['"', '\'']) {
            entries.push((path, Some(self.string()?)));
        } else if self.eat('[') {
            while !self.eat(']') {
                if !self.eat(',') {
                    self.value(vec![], &mut vec![])?;
                }
            }
            entries.push((path, None));
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

    /// What `workspace_from` makes of one manifest, the only one there is, `/home/w/Cargo.toml`,
    /// for the package `app` in `/home/w`.
    fn only(manifest: &str) -> (String, Option<PathBuf>) {
        workspace_from("app".into(), Some(Path::new("/home/w")), &files(&[("/home/w/Cargo.toml", manifest)]))
    }

    /// What `only` makes of a manifest of a workspace named `name`.
    fn named(name: &str) -> (String, Option<PathBuf>) {
        (name.into(), Some(PathBuf::from("/home/w/Cargo.toml")))
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
            assert_eq!(only(manifest), ("app".into(), None), "{manifest:?}");
        }
    }

    #[test]
    fn a_manifest_that_is_not_toml_is_passed_by() {
        // As one that is not there is: the walk goes on, here to a workspace one directory up.
        let up = ("/home/Cargo.toml", "[workspace]\n");
        for broken in ["name = \"never ends\n[workspace]\n", "[workspace\n", "workspace = \n", "[workspace] = 1 }"] {
            let manifests = [("/home/w/Cargo.toml", broken), up];
            let got = workspace_from("app".into(), Some(Path::new("/home/w")), &files(&manifests));
            assert_eq!(got, ("home".into(), Some(PathBuf::from("/home/Cargo.toml"))), "{broken:?}");
        }
    }
}
