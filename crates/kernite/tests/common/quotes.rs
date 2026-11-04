//! Types that have a block or a macro's arguments in them, as a source spells them and as a message quotes
//! them, for the tests of the targets that refuse a parameter of such a type. The length of an array is not
//! here: kernite works it out in some types and not in others, so a test of it has the names of the length in
//! its module.

/// Expressions as rustfmt writes them: the rows of the review of #110, which wrote them in a block, as an
/// array's length and as a macro's arguments.
const EXPRESSIONS: &[&str] = &[
    "!0usize & 3",
    "N * 2",
    "<Cfg as Tile>::N",
    "<Cfg<4> as Tile>::N",
    "<Vec<u8> as Tr>::N",
    "<u8 as Tr<3>>::N",
    "<Foo<u8>>::N",
    "let v: Option<usize> = Some(3); match v { Some(n) => n, None => 0 }",
    "match A { 1 => 2, _ => 3 }",
    "if A { 1 } else { 2 }",
    "size_of::<Vec<u8>>()",
    "f::<u8>() < 3",
    "A < B && C > D",
    "(A < B) as usize",
    "A as i32 - 1",
    "x as *const u8 as usize",
    "[1, 2][0]",
    "[0u8; 4].len()",
    "A - (-1)",
    "A & !B",
    "A && &B",
    "A < -B",
    "1..=3",
    "S { a: 1 }.a",
    "Foo::<u8>::N + 1",
    "f::<{ 3 }>()",
    "size_of::<[u8; 4]>()",
    "size_of::<u8>() * 2",
    "{ 3 }",
    "a + b",
    "a: b",
    "a & b",
    "A::B",
    "-A",
    "*&A",
    "A.0",
    "A?",
    "A + B * C",
    "A < B || f::<u8>() > 2",
    "A >> 1",
    "N % 2 == 0",
    "(N + 1) / 2",
    "N.min(3)",
    "u8::MAX as usize",
    "core::mem::size_of::<u64>()",
    "<u64>::BITS as usize",
    "A as usize * 2",
    "!A as usize",
    "-1i32 as u8",
    "[0; 3]",
    "const { 3 }",
    "unsafe { 3 }",
    "loop { break 3 }",
    "(3)",
    "(3, 4).0",
    "true as usize",
    "A != B",
    "A | B | C",
    "<Foo<u8>>::N < 3",
    "<<u8 as Tr>::A as Tr>::B",
    "A && <u8>::MAX > 3",
    "<u8>::MAX << 1",
    "match A { n @ 1..=3 => n, _ => 0 }",
    "match A { 1 | 2 => 3, _ => 4 }",
    "0 as *const Vec<u8> as usize",
    "|x: Vec<u8>| x",
    "|x| x",
    "let f = |x: Vec<u8>| x; 3",
    "let v: Option<usize> = Some(3); 3",
    "(0 as *const Vec<u8>).is_null() as usize",
];

/// `expr` with its optional spaces taken out, those that do not stand between two word characters.
fn packed(expr: &str) -> String {
    let chars: Vec<char> = expr.chars().collect();
    let word = |c: char| c.is_alphanumeric() || c == '_';
    (0..chars.len())
        .filter(|&i| chars[i] != ' ' || (0 < i && i + 1 < chars.len() && word(chars[i - 1]) && word(chars[i + 1])))
        .map(|i| chars[i])
        .collect()
}

/// Each type as its source spells it, with how a message quotes it: a block and a macro's arguments are as
/// the source has them, whether packed or spaced.
pub fn quoted() -> Vec<(String, String)> {
    let mut out = vec![];
    for expr in EXPRESSIONS {
        for e in [packed(expr), expr.to_string()] {
            for ty in
                [format!("Marker<{{{e}}}>"), format!("Marker<{{ {e} }}>"), format!("m!({e})"), format!("m![ {e} ]")]
            {
                out.push((ty.clone(), ty));
            }
            out.push((format!("m!{{{e}}}"), format!("m! {{{e}}}")));
            out.push((format!("m! {{ {e} }}"), format!("m! {{ {e} }}")));
        }
    }
    // A line break and a comment are a space, and a run of them is one.
    for (ty, quote) in [
        ("Marker<{\n    N *\n    2\n}>", "Marker<{ N * 2 }>"),
        ("Marker<{ // twice\n    N * // the product\n    2 // of two\n}>", "Marker<{ N * 2 }>"),
        ("Marker<{N /* a */ * /* b */ 2}>", "Marker<{N * 2}>"),
        ("m!(\n    a + // plus\n    b\n)", "m!( a + b )"),
    ] {
        out.push((ty.to_string(), quote.to_string()));
    }
    out
}
