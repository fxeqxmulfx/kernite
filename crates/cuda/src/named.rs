//! The text of a module's GPU kernels, and their names, with the Rust module they are in
//! written into them as Rust compiles it, and with what its `cfg` keeps. kernite-macros writes
//! the text in parts, before Rust knows the module, which a macro is not told, nor which `cfg`
//! holds: it leaves a hole wherever the module's path goes, and puts each part under the `cfg` of
//! the items it is of. [`__named!`](crate::__named) joins the parts that are kept and fills each
//! hole with `module_path!()` as C++ mangles a namespace's names, `<length><name>` each:
//! `my_app::gpu` is `6my_app3gpu`. The tools that read PTX show `_ZN7kernite6my_app3gpu5saxpyE` as
//! `kernite::my_app::gpu::saxpy`.
//!
//! It is all `const`, so the text is a constant all the same: nothing is written as the program
//! runs. Rust stops a constant that takes more than 2 000 000 steps, a step being a call or a turn
//! of a loop. A piece is copied whole, in about 80 steps whatever its size, so a text of megabytes
//! takes no more steps than one of a few bytes, and a text of many pieces takes many. So a constant
//! joins few. kernite-macros cuts a part of many pieces in parts of a few thousand pieces, each but
//! the last ending with an empty piece, for the hole that follows its last piece, and each part is
//! a constant of its own. A few dozen parts are joined by one constant, as many of those by
//! another, [`__joined!`](crate::__joined), and so on in a tree, so that no constant takes many
//! steps, however many parts the module has.

/// `N` bytes of text, UTF-8: what [`scope`] and [`join`] make, and [`text`] gives as a `&str`. Nothing else makes one.
pub struct Text<const N: usize>([u8; N]);

/// Where bytes are written in order, and how many there are: `out` is what is left of the array they are written to, or
/// none, to count them and write none. A walk over a path or over pieces writes to one or counts with one, so that the
/// length it counts is the bytes it writes. A write that has no room is not made, and leaves no room for those after
/// it: what is written is whole strings.
struct Put<'a> {
    out: Option<&'a mut [u8]>,
    len: usize,
}

impl<'a> Put<'a> {
    /// A place that counts the bytes it is given.
    const fn counting() -> Put<'a> {
        Put { out: None, len: 0 }
    }

    /// A place that writes the bytes it is given to `array`, and counts them.
    const fn writing(array: &'a mut [u8]) -> Put<'a> {
        Put { out: Some(array), len: 0 }
    }

    /// Writes `bytes`, all of them or none.
    const fn bytes(&mut self, bytes: &[u8]) {
        self.len += bytes.len();
        if let Some(room) = self.out.take()
            && let Some((here, rest)) = room.split_at_mut_checked(bytes.len())
        {
            here.copy_from_slice(bytes);
            self.out = Some(rest);
        }
    }

    /// Writes `byte`.
    const fn byte(&mut self, byte: u8) {
        self.bytes(&[byte]);
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
    let mut put = Put::counting();
    put_scope(&mut put, path);
    put.len
}

/// `path`'s names as C++ mangles them, `<length><name>` each, for `N` the bytes
/// [`scope_len`] says they take. A raw name, `r#type`, is the name, `type`. A character of ASCII is written as it is: a
/// path of Rust's has no other than letters, digits and `_` in a name. A character beyond ASCII, which a name of PTX's
/// cannot have, is written as kernite-ptx writes it, as Rust's legacy mangling does: `σ` as `$u3c3$`.
pub const fn scope<const N: usize>(path: &str) -> Text<N> {
    let mut out = [0; N];
    put_scope(&mut Put::writing(&mut out), path);
    Text(out)
}

/// Writes `path`'s names as `scope` does.
const fn put_scope(put: &mut Put, path: &str) {
    let mut rest = path.as_bytes();
    while !rest.is_empty() {
        let (name, after) = split_name(rest);
        put.number(written_len(name), 10);
        put_name(put, name);
        rest = after;
    }
}

/// Writes the name `name`, each of its characters as it can be in a name of PTX's.
const fn put_name(put: &mut Put, mut name: &[u8]) {
    while let Some((c, tail)) = char_at(name) {
        if c >= 0x80 {
            put.bytes(b"$u");
            put.number(c as usize, 16);
            put.byte(b'$');
        } else {
            put.byte(c as u8);
        }
        name = tail;
    }
}

/// How many bytes the name `name` takes as `put_name` writes it.
const fn written_len(name: &[u8]) -> usize {
    let mut put = Put::counting();
    put_name(&mut put, name);
    put.len
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

/// How many bytes `pieces` take joined, with `hole` between each two.
pub const fn joined_len(pieces: &[&str], hole: &str) -> usize {
    let mut put = Put::counting();
    put_pieces(&mut put, pieces, hole);
    put.len
}

/// `pieces` joined, with `hole` between each two, for `N` the bytes [`joined_len`] says they take.
pub const fn join<const N: usize>(pieces: &[&str], hole: &str) -> Text<N> {
    let mut out = [0; N];
    put_pieces(&mut Put::writing(&mut out), pieces, hole);
    Text(out)
}

/// Writes `pieces`, with `hole` between each two, as `join` does.
const fn put_pieces(put: &mut Put, mut pieces: &[&str], hole: &str) {
    while let [piece, more @ ..] = pieces {
        put.bytes(piece.as_bytes());
        if !more.is_empty() {
            put.bytes(hole.as_bytes());
        }
        pieces = more;
    }
}

/// The text `bytes` holds.
pub const fn text<const N: usize>(bytes: &'static Text<N>) -> &'static str {
    // SAFETY: only `scope` and `join` make a `Text`, and each writes whole strings to an array of zeros, the pieces it
    // is given or ASCII, and nothing else: a write that has no room is not made. So it is UTF-8, whatever `N` is.
    // Checking it here would take a step for each byte, and Rust stops a constant that takes 2 000 000.
    unsafe { core::str::from_utf8_unchecked(&bytes.0) }
}

/// The texts joined, a constant `&'static str`. Each text is a `&'static str`. A text with an attribute,
/// `#[cfg(..)]`, is there where Rust keeps it, and only there. A constant takes about 80 steps for each text, so
/// kernite-macros gives it a few dozen at most, and joins more in a tree: a `__joined!` of the `__joined!`s of a few
/// dozen texts, and so on. It is expanded in the module of kernels, which may give `u8` and `str` another type, so
/// it names the primitives by path.
#[doc(hidden)]
#[macro_export]
macro_rules! __joined {
    ($($text:expr),* $(,)?) => {{
        const TEXTS: &[&::core::primitive::str] = &[$($text),*];
        const TEXT: $crate::named::Text<{ $crate::named::joined_len(TEXTS, "") }> = $crate::named::join(TEXTS, "");
        $crate::named::text(&TEXT)
    }};
}

/// The parts joined, a constant `&'static str`: the pieces of each with the names of `path`, a Rust
/// module's path, between each two, as C++ mangles a namespace's names. A part with an attribute,
/// `#[cfg(..)]`, is there where Rust keeps it, and only there. It is what kernite-macros writes for
/// a module's PTX and its kernels' names, `path` its `module_path!()`. Each part is a constant of
/// its own, as Rust stops a constant that takes too many steps, and [`__joined!`](crate::__joined)
/// joins them: kernite-macros gives this a few dozen parts at most, and joins the texts of more in
/// a tree. It is expanded in the module of kernels, which may give `u8` and `str` another type, so
/// it names the primitives by path.
#[doc(hidden)]
#[macro_export]
macro_rules! __named {
    ($path:expr, [$($(#[$cfg:meta])* [$($piece:expr),* $(,)?]),* $(,)?]) => {{
        $crate::__joined!($($(#[$cfg])* {
            const PATH: &::core::primitive::str = $path;
            const SCOPE: $crate::named::Text<{ $crate::named::scope_len(PATH) }> = $crate::named::scope(PATH);
            const PIECES: &[&::core::primitive::str] = &[$($piece),*];
            const PART: $crate::named::Text<{ $crate::named::joined_len(PIECES, $crate::named::text(&SCOPE)) }> =
                $crate::named::join(PIECES, $crate::named::text(&SCOPE));
            $crate::named::text(&PART)
        }),*)
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What `text` gives, checked as UTF-8 here, where it is not trusted.
    fn checked<const N: usize>(made: &Text<N>) -> &str {
        core::str::from_utf8(&made.0).unwrap_or("not UTF-8")
    }

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
    fn texts_are_joined_in_a_tree_of_any_depth_where_cfg_keeps_them() {
        const TEXT: &str = crate::__joined!(
            crate::__joined!(
                "a",
                #[cfg(any())]
                "gone",
                "b",
            ),
            #[cfg(all())]
            crate::__named!(
                "x::y",
                [
                    ["c ", " d"],
                    #[cfg(any())]
                    ["gone"]
                ]
            ),
            crate::__joined!(crate::__joined!(crate::__joined!("e"), crate::__joined!()), crate::__joined!("f", "g")),
            #[cfg(any())]
            crate::__joined!("never"),
        );
        assert_eq!(TEXT, "abc 1x1y defg");
        assert_eq!(crate::__joined!(), "");
        assert_eq!(crate::__joined!(crate::__joined!(), crate::__joined!()), "");
    }

    #[test]
    fn a_hole_at_the_end_of_a_part_is_there_and_the_text_is_text_of_any_characters() {
        // A part that is cut ends with an empty piece: the hole after its last piece is the join of two pieces.
        const CUT: &str = crate::__named!("x::y", [["a", "b", ""], ["c"]]);
        assert_eq!(CUT, "a1x1yb1x1yc");
        // Pieces and holes of two, three and four bytes in UTF-8.
        const WIDE: &str = crate::__named!("σ::a𝔸", [["é", "𝔸", "€"]]);
        assert_eq!(WIDE, "é6$u3c3$9a$u1d538$𝔸6$u3c3$9a$u1d538$€");
    }

    #[test]
    fn a_raw_name_is_the_name_and_what_ptx_cannot_name_is_escaped_as_rust_escapes_it() {
        assert_eq!(scope_len("r#type::r#match"), 11);
        const RAW: Text<11> = scope("r#type::r#match");
        assert_eq!(text(&RAW), "4type5match");
        // `м` and `о` are U+043C and U+043E; `σ` is U+03C3, and `𝔸` U+1D538, four bytes.
        assert_eq!(scope_len("app::мо"), 18);
        const CYRILLIC: Text<18> = scope("app::мо");
        assert_eq!(text(&CYRILLIC), "3app12$u43c$$u43e$");
        const GREEK: Text<17> = scope("σ::a𝔸");
        assert_eq!(text(&GREEK), "6$u3c3$9a$u1d538$");
    }

    #[test]
    fn each_character_of_a_name_is_itself_or_its_code_point_between_dollars() {
        // Every character there is, as a name of its own: its length, then the character, or `$u`, its code point in
        // lowercase hex and `$`, as Rust's legacy mangling writes it. The room is more than any of them takes, and what
        // is left is zeros.
        for c in (0..=char::MAX as u32).filter_map(char::from_u32) {
            let name = if c.is_ascii() { c.to_string() } else { format!("$u{:x}$", c as u32) };
            let want = format!("{}{name}", name.len());
            let path = c.to_string();
            let made = scope::<16>(&path);
            assert_eq!(scope_len(&path), want.len(), "{c:?}");
            assert_eq!(&made.0[..want.len()], want.as_bytes(), "{c:?}");
            assert!(made.0[want.len()..].iter().all(|byte| *byte == 0), "{c:?}");
        }
    }

    #[test]
    fn a_number_is_written_in_all_its_digits_and_counted_as_many() {
        let numbers = [0, 9, 10, 99, 100, 101, 1000, 123_456_789, usize::MAX];
        for n in numbers {
            for (base, written) in [(10, n.to_string()), (16, format!("{n:x}"))] {
                let mut counting = Put::counting();
                counting.number(n, base);
                assert_eq!(counting.len, written.len(), "{n} in base {base}");
                let mut out = [0; 20];
                let mut put = Put::writing(&mut out);
                put.number(n, base);
                assert_eq!(put.len, written.len(), "{n} in base {base}");
                assert_eq!(&out[..written.len()], written.as_bytes(), "{n} in base {base}");
            }
        }
        // A length of three digits is written in all three, in front of the name.
        let path = ["x"; 120].concat();
        assert_eq!(scope_len(&path), 123);
        assert_eq!(&scope::<123>(&path).0[..4], b"120x");
    }

    #[test]
    fn what_is_counted_is_what_is_written() {
        let pieces = ["a", "", "é", "𝔸b", "€"];
        // The pieces are 11 bytes, and the four holes between them 2 each.
        assert_eq!(joined_len(&pieces, "σ"), 11 + 4 * 2);
        const TEXT: Text<{ joined_len(&["a", "", "é", "𝔸b", "€"], "σ") }> = join(&["a", "", "é", "𝔸b", "€"], "σ");
        assert_eq!(text(&TEXT), "aσσéσ𝔸bσ€");
        const NO_PIECES: Text<{ joined_len(&[], "σ") }> = join(&[], "σ");
        assert_eq!((NO_PIECES.0.len(), text(&NO_PIECES)), (0, ""));
        for path in ["a", "a::b", "r#fn::r#type::x", "σ::𝔸::м", "::", ""] {
            assert_eq!(scope_len(path), written_len_of_path(path), "{path}");
        }
    }

    /// The bytes `scope` writes for `path`, counted by writing them to a place that is room enough, and seeing what is
    /// written: not by `scope_len`.
    fn written_len_of_path(path: &str) -> usize {
        let mut out = [0u8; 64];
        let mut put = Put::writing(&mut out);
        put_scope(&mut put, path);
        out.iter().rposition(|byte| *byte != 0).map_or(0, |last| last + 1)
    }

    /// A text of another length than the pieces take is cut where a piece does not fit whole, and padded with zeros:
    /// it is UTF-8 all the same, which is what `text` trusts.
    #[test]
    fn a_text_of_the_wrong_length_is_whole_strings_and_zeros_and_so_is_text() {
        assert_eq!(checked(&join::<3>(&["é", "é"], "")), "é\0");
        assert_eq!(checked(&join::<6>(&["é", "é"], "")), "éé\0\0");
        assert_eq!(checked(&join::<1>(&["a", "é"], "")), "a");
        // A piece that does not fit leaves no room for the one after it, which would.
        assert_eq!(checked(&join::<3>(&["ab", "é", "c"], "")), "ab\0");
        assert_eq!(checked(&scope::<4>("ab::c")), "2ab1");
        assert_eq!(checked(&scope::<7>("σ::a")), "6$u3c3$");
        assert_eq!(checked(&scope::<10>("σ::a")), "6$u3c3$1a\0");
    }
}
