//! The text of a module's GPU kernels, and their names, with the Rust module they are in
//! written into them as Rust compiles it, and with what its `cfg` keeps. kernite-macros writes
//! the text in parts, before Rust knows the module, which a macro is not told, nor which `cfg`
//! holds: it leaves a hole wherever the module's path goes, and puts each part under the `cfg` of
//! the items it is of. [`__named!`](crate::__named) joins the parts that are kept and fills each
//! hole with `module_path!()` as C++ mangles a namespace's names, `<length><name>` each:
//! `my_app::gpu` is `6my_app3gpu`. The tools that read PTX show `_ZN7kernite6my_app3gpu5saxpyE` as
//! `kernite::my_app::gpu::saxpy`.
//!
//! It is all `const`, so the text is a constant all the same: nothing is written as the
//! program runs.

/// A place to write bytes in order: what is left of the array that is written. With no room left, nothing is written.
struct Put<'a>(Option<&'a mut [u8]>);

impl Put<'_> {
    /// Writes `byte`.
    const fn byte(&mut self, byte: u8) {
        if let Some([slot, rest @ ..]) = self.0.take() {
            *slot = byte;
            self.0 = Some(rest);
        }
    }

    /// Writes `bytes`.
    const fn bytes(&mut self, mut bytes: &[u8]) {
        while let [byte, rest @ ..] = bytes {
            self.byte(*byte);
            bytes = rest;
        }
    }

    /// Writes `n` in `base`, its digits after 9 lowercase letters.
    const fn number(&mut self, n: usize, base: usize) {
        if n >= base {
            self.number(n / base, base);
        }
        let digit = (n % base) as u8;
        self.byte(if digit < 10 { b'0' + digit } else { b'a' + digit - 10 });
    }
}

/// How many bytes `path`'s names take as C++ mangles them.
pub const fn scope_len(path: &str) -> usize {
    let (mut n, mut rest) = (0, path.as_bytes());
    while !rest.is_empty() {
        let (name, after) = split_name(rest);
        let len = written_len(name);
        n += digits(len, 10) + len;
        rest = after;
    }
    n
}

/// `path`'s names as C++ mangles them, `<length><name>` each, for `N` the bytes
/// [`scope_len`] says they take. A raw name, `r#type`, is the name, `type`; and a character
/// that cannot be in a name of PTX's, which takes nothing beyond ASCII, is written as kernite-ptx
/// writes it, as Rust's legacy mangling does: `σ` as `$u3c3$`.
pub const fn scope<const N: usize>(path: &str) -> [u8; N] {
    let mut out = [0; N];
    let mut put = Put(Some(&mut out));
    let mut rest = path.as_bytes();
    while !rest.is_empty() {
        let (mut name, after) = split_name(rest);
        put.number(written_len(name), 10);
        while let Some((c, tail)) = char_at(name) {
            if c >= 0x80 {
                put.bytes(b"$u");
                put.number(c as usize, 16);
                put.byte(b'$');
            } else {
                let c = c as u8;
                put.byte(if c.is_ascii_alphanumeric() || c == b'_' { c } else { b'_' });
            }
            name = tail;
        }
        rest = after;
    }
    out
}

/// The first name of `b`, a path, once its `r#` is gone, and what follows the `::` that ends it: nothing, for the last.
const fn split_name(b: &[u8]) -> (&[u8], &[u8]) {
    let b = match b {
        [b'r', b'#', raw @ ..] => raw,
        _ => b,
    };
    let mut rest = b;
    while let [_, tail @ ..] = rest
        && !matches!(rest, [b':', b':', ..])
    {
        rest = tail;
    }
    match b.split_at_checked(b.len() - rest.len()) {
        Some((name, [_, _, after @ ..])) => (name, after),
        Some((name, _)) => (name, &[]),
        None => (b, &[]),
    }
}

/// How many bytes the name `name` takes as [`scope`] writes it.
const fn written_len(mut name: &[u8]) -> usize {
    let mut n = 0;
    while let Some((c, tail)) = char_at(name) {
        n += if c >= 0x80 { 3 + digits(c as usize, 16) } else { 1 };
        name = tail;
    }
    n
}

/// The character at the start of `b`, UTF-8, and what follows it; none for no bytes.
const fn char_at(b: &[u8]) -> Option<(u32, &[u8])> {
    let [lead, rest @ ..] = b else { return None };
    let lead = *lead as u32;
    let (mut c, following) = match lead {
        0..=0x7f => return Some((lead, rest)),
        0xc0..=0xdf => (lead & 0x1f, 1),
        0xe0..=0xef => (lead & 0x0f, 2),
        _ => (lead & 0x07, 3),
    };
    let (mut rest, mut k) = (rest, 0);
    while k < following
        && let [byte, tail @ ..] = rest
    {
        c = c << 6 | (*byte & 0x3f) as u32;
        (rest, k) = (tail, k + 1);
    }
    Some((c, rest))
}

/// How many digits `n` is written in, in `base`.
const fn digits(mut n: usize, base: usize) -> usize {
    let mut d = 1;
    while n >= base {
        (n, d) = (n / base, d + 1);
    }
    d
}

/// How many bytes `parts` take joined, each part's pieces with `hole` between each two.
pub const fn joined_len(mut parts: &[&[&str]], hole: &[u8]) -> usize {
    let mut n = 0;
    while let [pieces, rest @ ..] = parts {
        let mut pieces = *pieces;
        while let [piece, more @ ..] = pieces {
            n += piece.len();
            if !more.is_empty() {
                n += hole.len();
            }
            pieces = more;
        }
        parts = rest;
    }
    n
}

/// `parts` joined, each part's pieces with `hole` between each two, for `N` the bytes
/// [`joined_len`] says they take.
pub const fn join<const N: usize>(mut parts: &[&[&str]], hole: &[u8]) -> [u8; N] {
    let mut out = [0; N];
    let mut put = Put(Some(&mut out));
    while let [pieces, rest @ ..] = parts {
        let mut pieces = *pieces;
        while let [piece, more @ ..] = pieces {
            put.bytes(piece.as_bytes());
            if !more.is_empty() {
                put.bytes(hole);
            }
            pieces = more;
        }
        parts = rest;
    }
    out
}

/// `bytes` as the text they are, or no text if they are none; pieces of text joined by names of letters, digits and `_`
/// are text.
pub const fn text(bytes: &'static [u8]) -> &'static str {
    match core::str::from_utf8(bytes) {
        Ok(s) => s,
        Err(_) => "",
    }
}

/// The parts joined, a constant `&'static str`: the pieces of each with the names of `path`, a Rust
/// module's path, between each two, as C++ mangles a namespace's names. A part with an attribute,
/// `#[cfg(..)]`, is there where Rust keeps it, and only there. It is what kernite-macros writes for
/// a module's PTX and its kernels' names, `path` its `module_path!()`. It is expanded in the module
/// of kernels, which may give `u8` and `str` another type, so it names the primitives by path.
#[doc(hidden)]
#[macro_export]
macro_rules! __named {
    ($path:expr, [$($(#[$cfg:meta])* [$($piece:expr),* $(,)?]),* $(,)?]) => {{
        const PATH: &::core::primitive::str = $path;
        const SCOPE: [::core::primitive::u8; $crate::named::scope_len(PATH)] = $crate::named::scope(PATH);
        const PARTS: &[&[&::core::primitive::str]] = &[$($(#[$cfg])* &[$($piece),*]),*];
        const TEXT: [::core::primitive::u8; $crate::named::joined_len(PARTS, &SCOPE)] =
            $crate::named::join(PARTS, &SCOPE);
        $crate::named::text(&TEXT)
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_name_of_the_path_is_its_length_and_itself() {
        const NAME: &str = crate::__named!("my_app::gpu", [["_ZN7kernite", "5saxpyE"]]);
        assert_eq!(NAME, "_ZN7kernite6my_app3gpu5saxpyE");
        const LONG: &str = crate::__named!("a::abcdefghijklmnopqrstuvwxyz_0123456789::b", [["_ZN", "1fE"]]);
        assert_eq!(LONG, "_ZN1a37abcdefghijklmnopqrstuvwxyz_01234567891b1fE");
        assert_eq!(crate::__named!(module_path!(), [["_ZN", "1kE"]]), "_ZN12kernite_cuda5named5tests1kE");
    }

    #[test]
    fn the_path_goes_in_each_hole_and_only_there() {
        const TEXT: &str = crate::__named!("x::y", [["a ", " b ", ""]]);
        assert_eq!(TEXT, "a 1x1y b 1x1y");
        assert_eq!(crate::__named!("x", [["just text"]]), "just text");
        assert_eq!(crate::__named!("x", []), "");
        assert_eq!(crate::__named!("x", [[]]), "");
    }

    #[test]
    fn the_parts_are_joined_where_cfg_keeps_them() {
        const TEXT: &str = crate::__named!(
            "x",
            [
                ["a "],
                #[cfg(any())]
                ["gone "],
                #[cfg(all())]
                [" b ", "!"],
                #[cfg(not(all()))]
                [" never ", "?"],
                ["c"],
            ]
        );
        assert_eq!(TEXT, "a  b 1x!c");
        const NONE: &str = crate::__named!(
            "x",
            [
                #[cfg(any())]
                ["gone"],
                #[cfg(any())]
                ["too"]
            ]
        );
        assert_eq!(NONE, "");
    }

    #[test]
    fn a_raw_name_is_the_name_and_what_ptx_cannot_name_is_escaped_as_rust_escapes_it() {
        assert_eq!(scope_len("r#type::r#match"), 11);
        assert_eq!(scope::<11>("r#type::r#match"), *b"4type5match");
        // `м` and `о` are U+043C and U+043E; `σ` is U+03C3, and `𝔸` U+1D538, four bytes.
        assert_eq!(scope_len("app::мо"), 18);
        assert_eq!(scope::<18>("app::мо"), *b"3app12$u43c$$u43e$");
        assert_eq!(scope::<17>("σ::a𝔸"), *b"6$u3c3$9a$u1d538$");
    }

    #[test]
    fn a_length_of_many_digits_is_written_in_all_of_them() {
        let path = ["x"; 120].concat();
        assert_eq!(digits(path.len(), 10), 3);
        assert_eq!(scope_len(&path), 123);
        let out = scope::<123>(&path);
        assert_eq!(&out[..4], b"120x");
        for n in [0, 9, 10, 99, 100, 101, 1000] {
            assert_eq!(digits(n, 10), n.to_string().len(), "{n}");
            assert_eq!(digits(n, 16), format!("{n:x}").len(), "{n}");
        }
    }
}
