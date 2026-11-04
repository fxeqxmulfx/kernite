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

/// How many bytes `path`'s names take as C++ mangles them.
pub const fn scope_len(path: &str) -> usize {
    let b = path.as_bytes();
    let (mut n, mut at) = (0, 0);
    while at < b.len() {
        let (start, end) = name(b, at);
        let len = written_len(b, start, end);
        n += digits(len, 10) + len;
        at = end + 2;
    }
    n
}

/// `path`'s names as C++ mangles them, `<length><name>` each, for `N` the bytes
/// [`scope_len`] says they take. A raw name, `r#type`, is the name, `type`; and a character
/// that cannot be in a name of PTX's, which takes nothing beyond ASCII, is written as kernite-ptx
/// writes it, as Rust's legacy mangling does: `σ` as `$u3c3$`.
pub const fn scope<const N: usize>(path: &str) -> [u8; N] {
    let b = path.as_bytes();
    let mut out = [0; N];
    let (mut i, mut at) = (0, 0);
    while at < b.len() {
        let (start, end) = name(b, at);
        i = number(&mut out, i, written_len(b, start, end), 10);
        let mut j = start;
        while j < end {
            let (c, len) = char_at(b, j);
            if c >= 0x80 {
                (out[i], out[i + 1]) = (b'$', b'u');
                i = number(&mut out, i + 2, c as usize, 16);
                out[i] = b'$';
            } else if (c as u8).is_ascii_alphanumeric() || c as u8 == b'_' {
                out[i] = c as u8;
            } else {
                out[i] = b'_';
            }
            (i, j) = (i + 1, j + len);
        }
        at = end + 2;
    }
    out
}

/// Where the name of `b` that starts at `at` starts once its `r#` is gone, and where it ends:
/// at `::`, or at the end.
const fn name(b: &[u8], at: usize) -> (usize, usize) {
    let start = if at + 1 < b.len() && b[at] == b'r' && b[at + 1] == b'#' { at + 2 } else { at };
    let mut end = start;
    while end < b.len() && !(b[end] == b':' && end + 1 < b.len() && b[end + 1] == b':') {
        end += 1;
    }
    (start, end)
}

/// How many bytes the name `b[start..end]` takes as [`scope`] writes it.
const fn written_len(b: &[u8], start: usize, end: usize) -> usize {
    let (mut n, mut j) = (0, start);
    while j < end {
        let (c, len) = char_at(b, j);
        n += if c >= 0x80 { 3 + digits(c as usize, 16) } else { 1 };
        j += len;
    }
    n
}

/// The character of `b`, UTF-8, that starts at `j`, and how many bytes it takes.
const fn char_at(b: &[u8], j: usize) -> (u32, usize) {
    let lead = b[j] as u32;
    let (mut c, len) = match lead {
        0..=0x7f => return (lead, 1),
        0xc0..=0xdf => (lead & 0x1f, 2),
        0xe0..=0xef => (lead & 0x0f, 3),
        _ => (lead & 0x07, 4),
    };
    let mut k = 1;
    while k < len {
        c = c << 6 | (b[j + k] & 0x3f) as u32;
        k += 1;
    }
    (c, len)
}

/// How many digits `n` is written in, in `base`.
const fn digits(mut n: usize, base: usize) -> usize {
    let mut d = 1;
    while n >= base {
        (n, d) = (n / base, d + 1);
    }
    d
}

/// Writes `n` in `base`, its digits after 9 lowercase letters, at `i` of `out`; and says where
/// it ends.
const fn number<const N: usize>(out: &mut [u8; N], i: usize, mut n: usize, base: usize) -> usize {
    let d = digits(n, base);
    let mut k = d;
    while k > 0 {
        let digit = (n % base) as u8;
        out[i + k - 1] = if digit < 10 { b'0' + digit } else { b'a' + digit - 10 };
        (n, k) = (n / base, k - 1);
    }
    i + d
}

/// How many bytes `parts` take joined, each part's pieces with `hole` between each two.
pub const fn joined_len(parts: &[&[&str]], hole: &[u8]) -> usize {
    let (mut n, mut p) = (0, 0);
    while p < parts.len() {
        let pieces = parts[p];
        let mut i = 0;
        while i < pieces.len() {
            n += pieces[i].len();
            if i + 1 < pieces.len() {
                n += hole.len();
            }
            i += 1;
        }
        p += 1;
    }
    n
}

/// `parts` joined, each part's pieces with `hole` between each two, for `N` the bytes
/// [`joined_len`] says they take.
pub const fn join<const N: usize>(parts: &[&[&str]], hole: &[u8]) -> [u8; N] {
    let mut out = [0; N];
    let (mut at, mut p) = (0, 0);
    while p < parts.len() {
        let pieces = parts[p];
        let mut i = 0;
        while i < pieces.len() {
            let piece = pieces[i].as_bytes();
            out.split_at_mut(at).1.split_at_mut(piece.len()).0.copy_from_slice(piece);
            at += piece.len();
            if i + 1 < pieces.len() {
                out.split_at_mut(at).1.split_at_mut(hole.len()).0.copy_from_slice(hole);
                at += hole.len();
            }
            i += 1;
        }
        p += 1;
    }
    out
}

/// `bytes` as the text they are: pieces of text joined by names of letters, digits and `_`.
pub const fn text(bytes: &'static [u8]) -> &'static str {
    match core::str::from_utf8(bytes) {
        Ok(s) => s,
        Err(_) => panic!("text joined by names is text"),
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
