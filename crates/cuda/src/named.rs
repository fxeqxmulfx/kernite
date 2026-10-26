//! The text of a module's GPU kernels, and their names, with the Rust module they are in
//! written into them as Rust compiles it. kgen-macros writes the text before Rust knows the
//! module, which a macro is not told, so it leaves a hole wherever the module's path goes, and
//! [`__named!`](crate::__named) fills each with `module_path!()` as C++ mangles a namespace's
//! names, `<length><name>` each: `my_app::gpu` is `6my_app3gpu`. The tools that read PTX
//! show `_ZN4kgen6my_app3gpu5saxpyE` as `kgen::my_app::gpu::saxpy`.
//!
//! It is all `const`, so the text is a constant all the same: nothing is written as the
//! program runs.

/// How many bytes `path`'s names take as C++ mangles them.
pub const fn scope_len(path: &str) -> usize {
    let b = path.as_bytes();
    let (mut n, mut at) = (0, 0);
    while at < b.len() {
        let (start, end) = name(b, at);
        n += digits(end - start) + end - start;
        at = end + 2;
    }
    n
}

/// `path`'s names as C++ mangles them, `<length><name>` each, for `N` the bytes
/// [`scope_len`] says they take. A raw name, `r#type`, is the name, `type`; and a byte that
/// cannot be in a name of PTX's is `_`, as each of a name in Cyrillic is.
pub const fn scope<const N: usize>(path: &str) -> [u8; N] {
    let b = path.as_bytes();
    let mut out = [0; N];
    let (mut i, mut at) = (0, 0);
    while at < b.len() {
        let (start, end) = name(b, at);
        let (len, d) = (end - start, digits(end - start));
        let mut k = 0;
        while k < d {
            out[i + d - 1 - k] = b'0' + (len / 10usize.pow(k as u32) % 10) as u8;
            k += 1;
        }
        i += d;
        let mut j = start;
        while j < end {
            out[i] = if b[j].is_ascii_alphanumeric() || b[j] == b'_' { b[j] } else { b'_' };
            (i, j) = (i + 1, j + 1);
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

/// How many digits `n` is written in.
const fn digits(mut n: usize) -> usize {
    let mut d = 1;
    while n >= 10 {
        (n, d) = (n / 10, d + 1);
    }
    d
}

/// How many bytes `pieces` take joined, with `hole` between each two.
pub const fn joined_len(pieces: &[&str], hole: &[u8]) -> usize {
    let mut n = 0;
    let mut i = 0;
    while i < pieces.len() {
        n += pieces[i].len();
        if i + 1 < pieces.len() {
            n += hole.len();
        }
        i += 1;
    }
    n
}

/// `pieces` joined, with `hole` between each two, for `N` the bytes [`joined_len`] says they
/// take.
pub const fn join<const N: usize>(pieces: &[&str], hole: &[u8]) -> [u8; N] {
    let mut out = [0; N];
    let (mut at, mut i) = (0, 0);
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
    out
}

/// `bytes` as the text they are: pieces of text joined by names of letters, digits and `_`.
pub const fn text(bytes: &'static [u8]) -> &'static str {
    match core::str::from_utf8(bytes) {
        Ok(s) => s,
        Err(_) => panic!("text joined by names is text"),
    }
}

/// `pieces` joined, a constant `&'static str`, with the names of `path`, a Rust module's path,
/// between each two, as C++ mangles a namespace's names: what kgen-macros writes for a
/// module's PTX and its kernels' names, `path` its `module_path!()`.
#[doc(hidden)]
#[macro_export]
macro_rules! __named {
    ($path:expr, [$($piece:expr),* $(,)?]) => {{
        const PATH: &str = $path;
        const SCOPE: [u8; $crate::named::scope_len(PATH)] = $crate::named::scope(PATH);
        const PIECES: &[&str] = &[$($piece),*];
        const TEXT: [u8; $crate::named::joined_len(PIECES, &SCOPE)] = $crate::named::join(PIECES, &SCOPE);
        $crate::named::text(&TEXT)
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_name_of_the_path_is_its_length_and_itself() {
        const NAME: &str = crate::__named!("my_app::gpu", ["_ZN4kgen", "5saxpyE"]);
        assert_eq!(NAME, "_ZN4kgen6my_app3gpu5saxpyE");
        const LONG: &str = crate::__named!("a::abcdefghijklmnopqrstuvwxyz_0123456789::b", ["_ZN", "1fE"]);
        assert_eq!(LONG, "_ZN1a37abcdefghijklmnopqrstuvwxyz_01234567891b1fE");
        assert_eq!(crate::__named!(module_path!(), ["_ZN", "1kE"]), "_ZN9kgen_cuda5named5tests1kE");
    }

    #[test]
    fn the_path_goes_in_each_hole_and_only_there() {
        const TEXT: &str = crate::__named!("x::y", ["a ", " b ", ""]);
        assert_eq!(TEXT, "a 1x1y b 1x1y");
        assert_eq!(crate::__named!("x", ["just text"]), "just text");
        assert_eq!(crate::__named!("x", []), "");
    }

    #[test]
    fn a_raw_name_is_the_name_and_what_ptx_cannot_name_is_an_underscore() {
        assert_eq!(scope_len("r#type::r#match"), 11);
        assert_eq!(scope::<11>("r#type::r#match"), *b"4type5match");
        // `м` and `о` are two bytes each.
        assert_eq!(scope_len("app::мо"), 9);
        assert_eq!(scope::<9>("app::мо"), *b"3app4____");
    }

    #[test]
    fn a_length_of_many_digits_is_written_in_all_of_them() {
        let path = ["x"; 120].concat();
        assert_eq!(digits(path.len()), 3);
        assert_eq!(scope_len(&path), 123);
        let out = scope::<123>(&path);
        assert_eq!(&out[..4], b"120x");
        for n in [0, 9, 10, 99, 100, 101, 1000] {
            assert_eq!(digits(n), n.to_string().len(), "{n}");
        }
    }
}
