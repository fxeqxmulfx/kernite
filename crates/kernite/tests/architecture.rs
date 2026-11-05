//! The dependency rule: each crate depends only on the layers inside it, and nothing depends
//! on a crate outside the workspace. Every crate must have a line here, so a new one cannot
//! join without its place being decided. Inside a crate, no modules use each other in a cycle.
//! The composition hands on only the language and the use cases, and the macro names only
//! those, `module`, `ptx_scope` and `ptx_room`. Every crate has a README.md with an example that
//! `examples/readmes` runs, and a line in the README.md of the repository.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

/// Each crate and the crates it may depend on.
const RULE: &[(&str, &[&str])] = &[
    ("kernite-core", &[]),
    ("kernite-lang", &[]),
    ("kernite-compile", &["kernite-core"]),
    ("kernite-build", &["kernite-lang"]),
    ("kernite-x86", &["kernite-core", "kernite-lang", "kernite-compile", "kernite-build"]),
    ("kernite-ptx", &["kernite-core", "kernite-lang", "kernite-compile", "kernite-build"]),
    // kernite-core and kernite-compile are dev-dependencies: the tests of PTX read the items of an
    // entry, and those of x86 compile what the builder wrote.
    ("kernite", &["kernite-core", "kernite-compile", "kernite-lang", "kernite-build", "kernite-x86", "kernite-ptx"]),
    ("kernite-macros", &["kernite"]),
    ("kernite-cuda", &[]),
    ("kernite-cpu", &[]),
];

/// The name of the package of the manifest at `path`.
fn package(path: &Path) -> String {
    let text = fs::read_to_string(path).unwrap();
    let (mut name, mut section) = (None, "");
    for line in text.lines().map(str::trim) {
        if line.starts_with('[') {
            section = line.trim_matches(['[', ']']);
            continue;
        }
        let Some((key, value)) = line.split_once('=') else { continue };
        if section == "package" && key.trim() == "name" {
            name = Some(value.trim().trim_matches('"').to_string());
        }
    }
    name.unwrap()
}

/// The folder of the workspace, two above this crate's folder, crates/kernite.
fn workspace() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(2).unwrap().to_path_buf()
}

/// The name of the package of each crate in `parent`, `crates` or `examples`, by the name of its folder.
fn crates(parent: &str) -> BTreeMap<String, String> {
    let dirs = fs::read_dir(workspace().join(parent)).unwrap();
    dirs.map(|dir| {
        let dir = dir.unwrap().path();
        (dir.file_name().unwrap().to_str().unwrap().to_string(), package(&dir.join("Cargo.toml")))
    })
    .collect()
}

/// Each package of `lock`, a Cargo.lock's text, and the packages it depends on, of every kind.
/// However a manifest writes a dependency, cargo writes it in the lock in one form:
/// `[[package]]`, `name = "…"`, `dependencies = [`, one `"name",` a line, `]`.
fn locked(lock: &str) -> Vec<(&str, Vec<&str>)> {
    let mut packages = vec![];
    let mut lines = lock.lines().map(str::trim);
    while let Some(line) = lines.next() {
        match line.split_once(" = ") {
            Some(("name", name)) => packages.push((name.trim_matches('"'), vec![])),
            Some(("dependencies", _)) => {
                let deps = lines.by_ref().take_while(|l| *l != "]");
                packages.last_mut().unwrap().1 = deps.map(|d| d.trim_end_matches(',').trim_matches('"')).collect();
            }
            _ => {}
        }
    }
    packages
}

#[test]
fn every_crate_depends_only_inward() {
    // Cargo writes the lock before a test runs, each crate's dependencies as it resolved them.
    let lock = fs::read_to_string(workspace().join("Cargo.lock")).unwrap();
    let packages = locked(&lock);
    let mut seen = vec![];
    for name in crates("crates").into_values() {
        let allowed =
            RULE.iter().find(|(n, _)| *n == name).unwrap_or_else(|| panic!("{name} has no place in the rule")).1;
        let (_, deps) = packages
            .iter()
            .find(|(n, _)| *n == name)
            .unwrap_or_else(|| panic!("{name} is in crates/ but not in Cargo.lock"));
        for dep in deps {
            assert!(allowed.contains(dep), "{name} depends on {dep}; it may depend on {allowed:?}");
        }
        seen.push(name);
    }
    for (name, _) in RULE {
        assert!(seen.iter().any(|s| s == name), "{name} is in the rule but not in crates/");
    }
}

#[test]
fn no_crate_depends_on_anything_outside_the_workspace() {
    // The lock holds every package that the workspace's crates depend on, and every package those
    // depend on, whether from a registry, git or a path. Each is one of the workspace's crates,
    // those of crates/ and examples/.
    let members: Vec<_> = ["crates", "examples"].into_iter().flat_map(|parent| crates(parent).into_values()).collect();
    let lock = fs::read_to_string(workspace().join("Cargo.lock")).unwrap();
    for (name, _) in locked(&lock) {
        assert!(members.iter().any(|m| m == name), "a crate depends on {name}, from outside the workspace");
    }
}

/// Each crate has a README.md in its folder, and in it an example to copy, in a block of Rust.
#[test]
fn every_crate_has_a_readme_with_an_example() {
    for (dir, name) in crates("crates") {
        let path = workspace().join(format!("crates/{dir}/README.md"));
        let readme = fs::read_to_string(path).unwrap_or_else(|_| panic!("{name} has no README.md in crates/{dir}"));
        assert!(
            readme.lines().any(|l| l.starts_with("```rust")),
            "{name}'s crates/{dir}/README.md has no example; it needs a ```rust block"
        );
    }
}

/// `examples/readmes` runs the example of each crate's README.md as a doctest: an item of its lib.rs
/// has the README.md for its docs.
#[test]
fn examples_readmes_runs_the_example_of_every_crates_readme() {
    let lib = fs::read_to_string(workspace().join("examples/readmes/src/lib.rs")).unwrap();
    for (dir, name) in crates("crates") {
        let docs = format!("#[doc = include_str!(\"../../../crates/{dir}/README.md\")]");
        assert!(
            lib.lines().any(|l| l.trim() == docs),
            "examples/readmes runs no example of {name}; its src/lib.rs needs an item with {docs}"
        );
    }
}

/// The README.md of the repository lists the crates, a line each, linked to the crate's folder.
#[test]
fn the_readme_lists_every_crate_linked_to_its_folder() {
    let readme = fs::read_to_string(workspace().join("README.md")).unwrap();
    for (dir, name) in crates("crates") {
        let line = format!("- [`{name}`](crates/{dir})");
        assert!(
            readme.lines().any(|l| l.starts_with(&line)),
            "README.md's list of crates has no line for {name}; it needs one that starts with {line}"
        );
    }
}

#[test]
fn each_crates_dependencies_are_read_from_the_lock() {
    // A lock as cargo writes it. kernite-macros depends on kernite-x86 here, as it would by
    // `[dependencies.kernite-x86]` in its manifest, or by any other way of writing it.
    let lock = r#"# This file is automatically @generated by Cargo.
# It is not intended for manual editing.
version = 4

[[package]]
name = "kernite-core"
version = "0.1.0"

[[package]]
name = "kernite-macros"
version = "0.1.0"
dependencies = [
 "kernite",
 "kernite-x86",
]

[[package]]
name = "kernite-x86"
version = "0.1.0"
dependencies = [
 "kernite-core",
]
"#;
    let want = [
        ("kernite-core", vec![]),
        ("kernite-macros", vec!["kernite", "kernite-x86"]),
        ("kernite-x86", vec!["kernite-core"]),
    ];
    assert_eq!(locked(lock), want);
}

/// Whether `c` can be part of a name.
fn word(c: char) -> bool {
    c == '_' || c.is_alphanumeric()
}

/// The code of `src`, a Rust file, as text: each comment, string and character a space, and each
/// item `#[cfg(test)]` is on left out, to the `;` or `}` that ends it.
fn code(src: &str) -> String {
    let mut out = String::new();
    let mut rest = src;
    while let Some(c) = rest.chars().next() {
        let hashes = rest.strip_prefix('r').map_or(0, |r| r.len() - r.trim_start_matches('#').len());
        let len = if rest.starts_with("//") {
            rest.find('\n').unwrap_or(rest.len())
        } else if rest.starts_with("/*") {
            // Comments nest: each `/*` waits for a `*/` of its own.
            let (mut depth, mut len) = (0, 0);
            loop {
                let here = &rest[len..];
                if here.starts_with("/*") {
                    (depth, len) = (depth + 1, len + 2);
                } else if here.starts_with("*/") {
                    (depth, len) = (depth - 1, len + 2);
                } else {
                    len += here.chars().next().unwrap().len_utf8();
                }
                if depth == 0 {
                    break len;
                }
            }
        } else if c == 'r' && rest[1 + hashes..].starts_with('"') {
            let end = format!("\"{}", "#".repeat(hashes));
            2 + hashes + rest[2 + hashes..].find(&end).unwrap() + end.len()
        } else if c == '"' {
            let mut chars = rest.char_indices().skip(1);
            loop {
                match chars.next().unwrap() {
                    (_, '\\') => _ = chars.next(),
                    (i, '"') => break i + 1,
                    _ => {}
                }
            }
        } else if c == '\'' {
            // A character, unless it is a lifetime or a label, as `'a` is.
            let mut then = rest[1..].chars();
            match (then.next(), then.next()) {
                (Some('\\'), _) => 4 + rest[3..].find('\'').unwrap(),
                (Some(x), Some('\'')) => 2 + x.len_utf8(),
                _ => 0,
            }
        } else {
            0
        };
        out.push(if len == 0 { c } else { ' ' });
        rest = &rest[len.max(c.len_utf8())..];
    }
    let mut kept = String::new();
    let mut rest = out.as_str();
    while let Some(at) = rest.find("#[cfg(test)]") {
        kept += &rest[..at];
        let mut depth = 0;
        let end = rest[at..].find(|c: char| {
            depth += match c {
                '(' | '[' | '{' => 1,
                ')' | ']' | '}' => -1,
                _ => 0,
            };
            depth <= 0 && matches!(c, ';' | '}')
        });
        rest = &rest[at + end.unwrap() + 1..];
    }
    kept + rest
}

/// The modules each of a crate's modules uses: those it names after `crate::`, or after
/// `crate::{` and each `,` in those braces, and those it reaches by a name the crate's root takes
/// from them with `use`. `root` is the root's code, and `modules` each module's, by its name.
fn uses<'a>(root: &'a str, modules: &'a BTreeMap<String, String>) -> BTreeMap<&'a str, BTreeSet<&'a str>> {
    let module = |name: &str| modules.get_key_value(name).map(|(m, _)| m.as_str());
    // The module each name the root takes with `use` comes from.
    let mut taken = BTreeMap::new();
    for (at, _) in root.match_indices("use ").filter(|&(at, _)| !root[..at].ends_with(word)) {
        let path = root[at + 4..].split(';').next().unwrap().trim_start().trim_start_matches("crate::");
        let Some(from) = module(path.split(|c: char| !word(c)).next().unwrap()) else { continue };
        for leaf in path.split([',', '{', '}']).filter(|l| !l.trim_end().ends_with("::")) {
            taken.extend(leaf.rsplit(|c: char| !word(c)).find(|w| !w.is_empty()).map(|name| (name, from)));
        }
    }
    let mut uses = BTreeMap::new();
    for (name, text) in modules {
        let used: &mut BTreeSet<&str> = uses.entry(name.as_str()).or_default();
        for (at, _) in text.match_indices("crate::").filter(|&(at, _)| !text[..at].ends_with(word)) {
            // A name, or paths in braces, each of which starts with one.
            let rest = &text[at + "crate::".len()..];
            let (mut depth, mut starts) = (0, true);
            for (i, c) in rest.char_indices() {
                if starts && word(c) {
                    let named = rest[i..].split(|c: char| !word(c)).next().unwrap();
                    used.extend(module(named).or_else(|| taken.get(named).copied()));
                }
                depth += match c {
                    '{' => 1,
                    '}' => -1,
                    _ => 0,
                };
                starts = depth == 1 && matches!(c, '{' | ',') || starts && c.is_whitespace();
                if depth == 0 && !starts {
                    break;
                }
            }
        }
        used.remove(name.as_str());
    }
    uses
}

/// A cycle in `uses`, each module and the modules it uses: the modules in it in order, the first
/// again at the end.
fn cycle<'a>(uses: &BTreeMap<&'a str, BTreeSet<&'a str>>) -> Option<Vec<&'a str>> {
    // A module that uses none of those left is in no cycle, so it goes.
    let mut left = uses.clone();
    while let Some(m) = left.iter().find(|(_, used)| used.iter().all(|u| !left.contains_key(u))).map(|(m, _)| *m) {
        left.remove(m);
    }
    // Each module left uses one left: a walk from the first comes back to a module it has passed.
    let mut path = vec![*left.keys().next()?];
    loop {
        let next = *left[path.last().unwrap()].iter().find(|u| left.contains_key(*u)).unwrap();
        if let Some(at) = path.iter().position(|&m| m == next) {
            return Some([&path[at..], &[next]].concat());
        }
        path.push(next);
    }
}

/// The files in `dir`, and in each directory in it.
fn files(dir: &Path) -> Vec<PathBuf> {
    let paths = fs::read_dir(dir).unwrap().map(|e| e.unwrap().path());
    paths.flat_map(|p| if p.is_dir() { files(&p) } else { vec![p] }).collect()
}

/// The path and the code of each Rust file in `src`, a crate's directory of them.
fn sources(src: &Path) -> Vec<(PathBuf, String)> {
    let rust = files(src).into_iter().filter(|f| f.extension().is_some_and(|e| e == "rs"));
    rust.map(|f| {
        let text = code(&fs::read_to_string(&f).unwrap());
        (f, text)
    })
    .collect()
}

#[test]
fn each_modules_uses_are_read_from_its_code() {
    // A crate whose modules a, b and c use each other in a cycle, which d uses. a uses b by a name
    // the root takes from b. It names c only in comments, in strings, and in a test whose
    // characters must not end it early.
    let root = code("pub use b::{one, inner::Two as Three};\nuse crate::c::Five;\nfn four() {}\n");
    let a = r###"use crate::{Three, four};
/// [`crate::c`]
fn f<'a>(s: &'a str) -> &'a str { /* crate::c /* crate::c */ crate::c */ s }
const S: [&str; 2] = ["crate::c \" crate::c", r##"crate::c " crate::c"##];
#[cfg(test)]
mod tests { const C: [char; 3] = ['}', '"', '\'']; use crate::c; }
"###;
    // b uses c by `$crate::`, and itself, which does not count. c uses a; the paths in a's braces
    // name a's items, not the root's. d uses c by a name the root takes with `use crate::`.
    let b = "macro_rules! m { () => { $crate::c::f() } }\nfn g() { crate::b::g() }\n";
    let c = "use crate::{a::{b, one}};\n";
    let d = "fn h() -> crate::Five { todo!() }\n";
    let modules = BTreeMap::from([("a", a), ("b", b), ("c", c), ("d", d)].map(|(m, text)| (m.to_string(), code(text))));
    // Each module, and the one it uses.
    let graph = |pairs: &[(&'static str, &'static str)]| -> BTreeMap<_, _> {
        pairs.iter().map(|&(m, used)| (m, BTreeSet::from([used]))).collect()
    };
    let uses = uses(&root, &modules);
    assert_eq!(uses, graph(&[("a", "b"), ("b", "c"), ("c", "a"), ("d", "c")]));
    assert_eq!(cycle(&uses), Some(vec!["a", "b", "c", "a"]));
    // A module that uses a cycle is not part of it. Without a cycle there is none.
    assert_eq!(cycle(&graph(&[("a", "b"), ("b", "c"), ("c", "b")])), Some(vec!["b", "c", "b"]));
    assert_eq!(cycle(&graph(&[("a", "b")])), None);
}

/// No crate's modules use each other in a cycle. A module's code is in its file, or in the files
/// of its directory, and not in those named tests.rs: a test may use what it tests.
#[test]
fn no_crates_modules_use_each_other_in_a_cycle() {
    let mut cycles = vec![];
    for (dir, name) in crates("crates") {
        let src = workspace().join(format!("crates/{dir}/src"));
        let mut modules = BTreeMap::<String, String>::new();
        for (file, text) in sources(&src) {
            if !file.ends_with("tests.rs") {
                let module = file.strip_prefix(&src).unwrap().iter().next().unwrap().to_str().unwrap();
                *modules.entry(module.trim_end_matches(".rs").to_string()).or_default() += &text;
            }
        }
        let root = modules.remove("lib").unwrap();
        if let Some(cycle) = cycle(&uses(&root, &modules)) {
            let path = cycle[1..].join(", which uses ");
            cycles.push(format!("{name}'s modules use each other in a cycle: {} uses {path}", cycle[0]));
        }
    }
    assert!(cycles.is_empty(), "{}", cycles.join("\n"));
}

/// The first name of each path that `code` writes after `prefix`: of the one path, or of each path
/// in the braces that follow, as in `use kernite::{lang::{self, Tree}, Given}`, which gives `lang`
/// and `Given` after `kernite::`, and `self` and `Tree` after `lang::`.
fn starts<'a>(code: &'a str, prefix: &str) -> Vec<&'a str> {
    let mut names = vec![];
    for (at, _) in code.match_indices(prefix).filter(|&(at, _)| !code[..at].ends_with(word)) {
        let rest = &code[at + prefix.len()..];
        let first = |s: &'a str| s.split(|c: char| !word(c)).next().unwrap();
        let Some(group) = rest.strip_prefix('{') else {
            names.push(first(rest));
            continue;
        };
        let (mut depth, mut new) = (0, true);
        for (i, c) in group.char_indices() {
            match c {
                '{' => depth += 1,
                '}' if depth == 0 => break,
                '}' => depth -= 1,
                ',' if depth == 0 => new = true,
                c if new && word(c) => {
                    names.push(first(&group[i..]));
                    new = false;
                }
                _ => {}
            }
        }
    }
    names
}

#[test]
fn what_kernite_hands_on_and_what_the_macro_names_are_read_from_their_code() {
    // A `pub use` of one line or of many, but not in a comment, and not `pub(crate)`.
    let lib = code(
        "//! pub use no::Comment;\npub use a::{\n    B,\n    C,\n};\npub use d as e;\npub(crate) use f::G;\nuse h::I;\n",
    );
    assert_eq!(starts(&lib, "pub use "), ["a", "d"]);
    assert_eq!(starts(&lib, "pub use a::"), ["B", "C"]);
    // The path or each path in braces, not the names inside a path; not in a comment, in a string,
    // or after another name.
    let src = code(
        "use kernite::{lang::{self, Tree}, Given};\nuse kernite::lang::{constant as c};\n\
         fn f() { kernite::module(m); xkernite::no(); /* kernite::no */ \"kernite::no\"; kernite::lang::kernel(); }\n",
    );
    assert_eq!(starts(&src, "kernite::"), ["lang", "Given", "lang", "module", "lang"]);
    assert_eq!(starts(&src, "lang::"), ["self", "Tree", "constant", "kernel"]);
}

/// kernite hands on the language and the use cases, which is what its callers need. The types of a
/// machine, of the allocator and of an instruction are none of those: whoever needs one depends on
/// the crate that has it.
#[test]
fn kernite_re_exports_only_the_language_and_the_use_cases() {
    for (file, code) in sources(&workspace().join("crates/kernite/src")) {
        for from in starts(&code, "pub use ") {
            assert!(
                ["kernite_lang", "kernite_build"].contains(&from),
                "{} hands on what `{from}` has; it may hand on kernite_lang's and kernite_build's only",
                file.display()
            );
        }
    }
}

/// The macro writes Rust for what kernite has decided, and decides nothing. It names the language,
/// the use cases kernite hands on, `module`, which calls them, `ptx_scope`, and `ptx_room`, which
/// counts the room of the statics Rust keeps, once `cfg` has left out the rest. It calls none of
/// the language's functions that evaluate: `kernite::module` does that. `lang::instance` names an
/// instance and evaluates nothing, so it stays.
#[test]
fn the_macro_names_only_the_language_the_use_cases_ptx_scope_and_ptx_room() {
    let lib = sources(&workspace().join("crates/kernite/src"));
    let built: Vec<_> = lib.iter().flat_map(|(_, code)| starts(code, "pub use kernite_build::")).collect();
    let allowed = [&["lang", "module", "ptx_scope", "ptx_room"][..], &built].concat();
    for (file, code) in sources(&workspace().join("crates/macros/src")) {
        for name in starts(&code, "kernite::") {
            assert!(
                allowed.contains(&name),
                "{} names `kernite::{name}`; it may name the language, the use cases, `module`, `ptx_scope` and `ptx_room`",
                file.display()
            );
        }
        for name in starts(&code, "lang::") {
            assert!(
                !["constant", "kernel", "instances", "memory"].contains(&name),
                "{} names `lang::{name}`, which evaluates; `kernite::module` does that",
                file.display()
            );
        }
    }
}
