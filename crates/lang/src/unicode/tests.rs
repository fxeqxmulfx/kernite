//! The tables and the rules of `unicode`, held to rustc, which is the one that says what a name is: the `rustc` of the
//! machine builds `tests/fixtures/names.rs`, a proc macro that makes a name of each line it is given, as Rust does, in
//! NFC, or none. The lines are each character from U+0080 up, alone and after `_`, and strings of the characters that
//! NFC works on. The test is skipped, and says so, where there is no `rustc`: `$RUSTC` is the path of one.

use super::tables::{COMBINING, COMPOSITION, DECOMPOSITION};
use super::{is_xid_continue, is_xid_start, nfc};
use std::path::Path;
use std::process::Command;
use std::{env, fs};

type Result<T = (), E = Box<dyn std::error::Error + Send + Sync>> = std::result::Result<T, E>;

/// What `rustc` makes of each of `lines`: the name, in NFC, or `-` where it makes none. The proc macro is built in
/// `dir` the first time.
fn names(rustc: &str, dir: &Path, lines: &[String]) -> Result<Vec<String>> {
    let lib = dir.join(format!("{}names{}", env::consts::DLL_PREFIX, env::consts::DLL_SUFFIX));
    if !lib.exists() {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/names.rs");
        let built = Command::new(rustc)
            .args(["--edition", "2024", "--crate-type", "proc-macro", "-o"])
            .arg(&lib)
            .arg(fixture)
            .output()?;
        assert!(built.status.success(), "{}", String::from_utf8_lossy(&built.stderr));
    }
    let (input, output, user) = (dir.join("in.txt"), dir.join("out.txt"), dir.join("user.rs"));
    fs::write(&input, lines.join("\n") + "\n")?;
    fs::write(&user, "names::names!();")?;
    let run = Command::new(rustc)
        .args(["--edition", "2024", "--crate-type", "lib", "--emit=metadata", "--extern"])
        .arg(format!("names={}", lib.display()))
        .arg("-o")
        .arg(dir.join("user.rmeta"))
        .arg(user)
        .env("NAMES_IN", input)
        .env("NAMES_OUT", &output)
        .output()?;
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    let names: Vec<String> = fs::read_to_string(output)?.lines().map(String::from).collect();
    assert_eq!(names.len(), lines.len());
    Ok(names)
}

/// The strings that NFC takes apart, puts in order and puts together, each a name for the `_` it starts with, which
/// makes a character with no other.
fn sequences() -> Vec<String> {
    // Each pair that Unicode makes one character, and each character that it takes apart, put together again.
    let mut out: Vec<String> = COMPOSITION.iter().map(|&((a, b), _)| format!("{a}{b}")).collect();
    out.extend(DECOMPOSITION.iter().map(|&(_, parts)| parts.to_string()));
    // After `a`, each mark with each that `a` makes a character with, in each order: the one may stand between the
    // other and `a`, and the two may be of one class, which keeps them as they are, or of two, which puts them in
    // order.
    let with_a: Vec<char> = COMPOSITION.iter().filter(|&&((a, _), _)| a == 'a').map(|&((_, b), _)| b).collect();
    for &(mark, _) in &COMBINING {
        for &with in &with_a {
            out.push(format!("a{mark}{with}"));
            out.push(format!("a{with}{mark}"));
        }
    }
    // One mark of each class, two at a time, in each order.
    let (mut classes, mut firsts) = (vec![], vec![]);
    for &(mark, class) in &COMBINING {
        if !classes.contains(&class) {
            classes.push(class);
            firsts.push(mark);
        }
    }
    out.extend(firsts.iter().flat_map(|a| firsts.iter().map(move |b| format!("{a}{b}"))));
    // The Hangul jamo, two at a time: the leading consonants, the vowels and the trailing consonants with each other
    // and with the jamo beside them that are none of the three. Before each jamo, the first and the last syllable with
    // a trailing consonant and with none. And a leading consonant, a vowel and a trailing consonant, in threes.
    let jamo: Vec<char> = ('\u{1100}'..='\u{11ff}').collect();
    out.extend(jamo.iter().flat_map(|a| jamo.iter().map(move |b| format!("{a}{b}"))));
    for syllable in ['\u{ac00}', '\u{ac01}', '\u{d788}', '\u{d7a3}'] {
        out.extend(jamo.iter().map(|jamo| format!("{syllable}{jamo}")));
    }
    for leading in '\u{1100}'..='\u{1112}' {
        for vowel in '\u{1161}'..='\u{1175}' {
            out.extend(('\u{11a8}'..='\u{11c2}').map(|trailing| format!("{leading}{vowel}{trailing}")));
        }
    }
    // And more, of every character that has a place in NFC, in a number that a generator gives, the same each time.
    let mut pool: Vec<char> = COMBINING.iter().map(|&(c, _)| c).collect();
    pool.extend(COMPOSITION.iter().flat_map(|&((a, b), c)| [a, b, c]));
    pool.extend(DECOMPOSITION.iter().map(|&(c, _)| c));
    pool.sort();
    pool.dedup();
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next = |below: usize| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state % below as u64) as usize
    };
    for _ in 0..100_000 {
        let length = 2 + next(5);
        out.push((0..length).map(|_| pool[next(pool.len())]).collect());
    }
    out.iter().map(|s| format!("_{s}")).collect()
}

/// A name is what rustc says it is: of each character from U+0080 up, whether it goes on a name, and whether it starts
/// one, and of each string of those that NFC works on, which one name it is.
#[test]
fn a_name_is_what_rustc_reads_as_one() -> Result {
    let rustc = env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    if !Command::new(&rustc).arg("--version").output().is_ok_and(|out| out.status.success()) {
        eprintln!("no rustc here: skipped; set RUSTC to run it");
        return Ok(());
    }
    let dir = env::temp_dir().join(format!("kernite-lang-names-{}", std::process::id()));
    fs::create_dir_all(&dir)?;
    let chars: Vec<char> = (0x80..=0x10ffff).filter_map(char::from_u32).collect();
    // After `_`, which a name goes on with, as it starts with one.
    let after: Vec<String> = chars.iter().map(|c| format!("_{c}")).collect();
    let mut goes_on = vec![];
    for ((&c, name), line) in chars.iter().zip(names(&rustc, &dir, &after)?).zip(&after) {
        assert_eq!(name != "-", is_xid_continue(c), "{c:?} goes on a name");
        if name != "-" {
            assert_eq!(name, nfc(line), "{c:?} after `_`");
            goes_on.push(c);
        }
    }
    // Alone, each that goes on one, which a name starts with if rustc takes it, and strings of NFC's characters.
    let alone: Vec<String> = goes_on.iter().map(char::to_string).collect();
    let strings = sequences();
    let named = names(&rustc, &dir, &[alone.clone(), strings.clone()].concat())?;
    for ((&c, line), name) in goes_on.iter().zip(&alone).zip(&named) {
        assert_eq!(*name != "-", is_xid_start(c), "{c:?} starts a name");
        if *name != "-" {
            assert_eq!(*name, nfc(line), "{c:?} alone");
        }
    }
    for (line, name) in strings.iter().zip(&named[alone.len()..]) {
        assert_eq!(*name, nfc(line), "{line:?}");
    }
    fs::remove_dir_all(dir)?;
    Ok(())
}
