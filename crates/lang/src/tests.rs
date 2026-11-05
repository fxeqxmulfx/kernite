//! The lexer, how a type is read and written and which types are pointers, the rounding of 16-bit
//! floats, how a value known now is told when it is another type than the one wanted, whether a function
//! is `unsafe`, where a place is, and which items of a module kernite reads when it cannot read all. The
//! rest of the language is tested where it meets the targets it is compiled for, in `kernite`'s
//! `tests/lang.rs`.

use crate::token::{Span, Tree};
use crate::value::{IntTy, Value, article, cast, typed};
use crate::{Mention, parse};

#[test]
fn the_lexer_reads_what_rust_writes() {
    let src = "a.b(0..n, 1.5e-3, 0x1F_u8, r#type, \"s\\\"q\", 1..=2) /* c /* nested */ */ // end";
    let trees = crate::lex(src).unwrap();
    assert_eq!(crate::text(&trees), r#"a.b(0..n, 1.5e-3, 0x1F_u8, r#type, "s\"q", 1..=2)"#);
}

/// The lexer refuses a name written raw that Rust refuses, at its `r#`: `self`, `Self`, `super`, `crate` and `_`, which
/// Rust lets be written raw in no place, and an `r#` that no name follows. Every other name written raw is one name,
/// whatever it starts as or is: a keyword, a weak one, a reserved one, or one that has `self` or `crate` in it.
#[test]
fn the_lexer_refuses_a_name_written_raw_that_rust_refuses() {
    let error = |src: &str| crate::lex(src).unwrap_err().at(src);
    for name in ["self", "Self", "super", "crate", "_"] {
        let src = format!("a r#{name} b");
        assert_eq!(error(&src), format!("1:3: `{name}` cannot be written raw"), "{src}");
    }
    // No name after the `r#`: a space, the end, a bracket, a dot, a digit or a quote.
    for src in ["a r# b", "a r#", "a r#(b)", "a r#.b", "a r#9", "a r#1x", "a r#\"x\""] {
        assert_eq!(error(src), "1:3: `r#` comes before a name, as in `r#u64`", "{src}");
    }
    let names = "r#u64 r#dyn r#union r#gen r#try r#_x r#__ r#selfie r#crate_ r#é self Self super crate _";
    let idents: Vec<String> = crate::lex(names)
        .unwrap()
        .into_iter()
        .map(|tree| match tree {
            Tree::Ident { name, .. } => name,
            other => panic!("{other:?}"),
        })
        .collect();
    assert_eq!(idents, names.split(' ').collect::<Vec<_>>());
}

/// The tokens of `src`, a source with lifetimes in it, as `for<'a> fn(&'a u8)` has. The lexer of a
/// string refuses a lifetime, but the macro is given one as `proc_macro` hands it over: a `'` right before
/// the name that follows it. So each `'` of `src` is lexed as the start of a word, `lt_a`, which is then
/// made into the two tokens.
fn lexed_with_lifetimes(src: &str) -> Vec<Tree> {
    fn made(trees: Vec<Tree>) -> Vec<Tree> {
        trees
            .into_iter()
            .flat_map(|tree| match tree {
                Tree::Ident { name, space, span } if name.starts_with("lt_") => vec![
                    Tree::Punct { ch: '\'', space, span },
                    Tree::Ident { name: name["lt_".len()..].to_string(), space: false, span },
                ],
                Tree::Group { delim, trees, space, space_close, span } => {
                    vec![Tree::Group { delim, trees: made(trees), space, space_close, span }]
                }
                tree => vec![tree],
            })
            .collect()
    }
    made(crate::lex(&src.replace('\'', "lt_")).unwrap())
}

/// A type as the parser holds it, written out: a name, a name with arguments, an array or a slice, whose length is
/// left out.
fn shown(ty: &crate::Type) -> String {
    match ty {
        crate::Type::Name(name, _) => name.clone(),
        crate::Type::Generic(name, args, _) => {
            format!("{name}<{}>", args.iter().map(shown).collect::<Vec<_>>().join(", "))
        }
        crate::Type::Array(elem, ..) => format!("[{}; ..]", shown(elem)),
        crate::Type::Slice(elem, _) => format!("[{}]", shown(elem)),
    }
}

/// A type is written as Rust writes it, however its source spaced it: a space between words, after
/// `,` and `;`, after `const` and `mut`, after the `>` of a binder, `for<'a> fn(&'a u8)`, on each
/// side of a `+` or a `=` that no operator holds and of an arrow, `->`, and none beside the rest, nor
/// inside an operator, `+=`, `==`, `=>`, `<=`, `..=` or `->`.
#[test]
fn a_type_is_written_as_rust_writes_it() {
    let written = |src: &str| crate::text(&crate::lex(src).unwrap());
    for ty in [
        "*const [usize; 3]",
        "*mut *const f32",
        "*const *mut *const u8",
        "(u32, u32)",
        "&mut [f32]",
        "&[u32]",
        "*const ()",
        "*const (u8, *mut u8)",
        "*const [[u8; 4]; 2]",
        "*const fn(u32)",
        "*const fn(u32) -> u32",
        "*const fn() -> !",
        "*const fn(fn() -> u8) -> &u8",
        "*const dyn Fn(u32)",
        "*const Foo<u8, u16>",
        "*const Pair<fn() -> u8, 2>",
        "*const a::b::C",
        "*const (dyn A + Send)",
        "Box<dyn A + B + C>",
        "Box<dyn Iterator<Item = u8>>",
        "Box<dyn Iterator<Item = u8> + Send>",
        "*const (dyn Iterator<Item = u8> + Send)",
        "*const Foo<T = u8, U = u16>",
        "Box<dyn Fn(u8) -> u8 + Send>",
        "*const (dyn Fn(u8) -> u8 + Send)",
    ] {
        assert_eq!(written(ty), ty);
    }
    // The text of an expression has such operators, which get no space inside or beside them.
    for expr in ["a+=b", "a==b", "a=>b", "a<=b", "0..=n"] {
        assert_eq!(written(expr), expr);
    }
    assert_eq!(written("*  mut*const[ usize ;3 ]"), "*mut *const [usize; 3]");
    assert_eq!(written("(u32 ,u32 , )"), "(u32, u32,)");
    assert_eq!(written("Box<dyn A+B>"), "Box<dyn A + B>");
    assert_eq!(written("dyn  Iterator<Item=u8>"), "dyn Iterator<Item = u8>");
    // Punctuation beside a `=` or a `+` does not hold it: `=&`, `>+` and `+?` are no operators.
    assert_eq!(written("Box<dyn Iterator<Item=&u8>>"), "Box<dyn Iterator<Item = &u8>>");
    assert_eq!(written("Box<dyn Iterator<Item = u8>+Send>"), "Box<dyn Iterator<Item = u8> + Send>");
    assert_eq!(written("*const dyn A+?Sized"), "*const dyn A + ?Sized");
    // The `>` of an arrow, with punctuation right after it, `->&`, is the arrow's and no other's.
    assert_eq!(written("*const fn(u32)->u32"), "*const fn(u32) -> u32");
    assert_eq!(written("Pair<fn()->&u8,2>"), "Pair<fn() -> &u8, 2>");
    assert_eq!(written("Box<dyn Fn(u8)->u8+Send>"), "Box<dyn Fn(u8) -> u8 + Send>");
    assert_eq!(written("fn(fn()->!)  ->  !"), "fn(fn() -> !) -> !");
    // A binder is followed by a space; the `>` that closes the arguments of a type is no binder's.
    let bound = |src: &str| crate::text(&lexed_with_lifetimes(src));
    for ty in [
        "for<'a> fn(&'a u8)",
        "for<'a> fn(&'a u8) -> &'a u8",
        "for<'a, 'b> fn(&'a u8, &'b u8)",
        "for<> fn(u8)",
        "dyn for<'a> Fn(&'a u8)",
        "Option<&dyn for<'a> Fn(&'a u8)>",
        "*const for<'a> fn(&'a u8)",
        "Pair<for<'a> fn(&'a u8), 2>",
        "fn(for<'a> fn(&'a u8)) -> for<'a> fn(&'a u8)",
        "Box<dyn Send + for<'a> Fn(&'a u8)>",
        "Box<dyn for<'a> Tr<'a>>",
        "dyn for<'a> Tr<'a> + Send",
    ] {
        assert_eq!(bound(ty), ty);
    }
    assert_eq!(bound("for<'a>fn(&'a u8)"), "for<'a> fn(&'a u8)");
    assert_eq!(bound("for < 'a >  fn(&'a u8)->&'a u8"), "for<'a> fn(&'a u8) -> &'a u8");
    assert_eq!(bound("Option<&dyn for<'a>Fn(&'a u8)>"), "Option<&dyn for<'a> Fn(&'a u8)>");
    assert_eq!(bound("Box<dyn Send+for<'a>Fn(&'a u8)>"), "Box<dyn Send + for<'a> Fn(&'a u8)>");
}

/// A type has a space after a `:` that is not half of a `::`, `fn(x: u8)`, `Item: Copy` and `'b: 'a`; after
/// the name of a lifetime, `&'a [u8]`, `&'static !` and `&'a ::b::C`, a path from the root; after `dyn` and
/// `impl`, `dyn (A) + B` and `impl 'a + A`; and before `as`, `<Vec<u8> as Tr>::Out`. The braces of a macro have
/// a space before them, `byte! {..}`; the parentheses of a macro have none, `byte!(_)`.
#[test]
fn a_colon_a_lifetime_and_a_macros_braces_are_written_as_rust_writes_them() {
    let written = |src: &str| crate::text(&crate::lex(src).unwrap());
    for (src, ty) in [
        ("fn(f16:u8)", "fn(f16: u8)"),
        ("fn(f16 : u8)", "fn(f16: u8)"),
        ("fn(a:u8,b:u16)", "fn(a: u8, b: u16)"),
        ("fn(_:u8)", "fn(_: u8)"),
        ("fn(x:fn(y:u8)->u8)->u8", "fn(x: fn(y: u8) -> u8) -> u8"),
        ("fn(f128: ::core::primitive::u8)", "fn(f128: ::core::primitive::u8)"),
        ("Box<dyn Iterator<Item:Copy>>", "Box<dyn Iterator<Item: Copy>>"),
        ("impl Iterator<Item : Copy>", "impl Iterator<Item: Copy>"),
        ("a::b::C<u8>", "a::b::C<u8>"),
        ("<A as B>::C", "<A as B>::C"),
        ("<Vec<u8>as Tr>::Out", "<Vec<u8> as Tr>::Out"),
        ("::core::primitive::u8", "::core::primitive::u8"),
        ("dyn(A)+B", "dyn (A) + B"),
        ("impl(A)", "impl (A)"),
        ("Box<dyn ::a::B>", "Box<dyn ::a::B>"),
        ("byte!{_}", "byte! {_}"),
        ("byte!{ _ }", "byte! { _ }"),
        ("byte!{}", "byte! {}"),
        ("m!{a,b}", "m! {a,b}"),
        ("byte!(_)", "byte!(_)"),
        ("byte![_]", "byte![_]"),
        ("f16!()", "f16!()"),
        ("crate::byte!{_}", "crate::byte! {_}"),
        ("*const byte!{_}", "*const byte! {_}"),
        ("Wrap<byte!{_}>", "Wrap<byte! {_}>"),
        ("[byte!{_};1]", "[byte! {_}; 1]"),
        ("fn(u8)->byte!{_}", "fn(u8) -> byte! {_}"),
    ] {
        assert_eq!(written(src), ty, "{src}");
    }
    let bound = |src: &str| crate::text(&lexed_with_lifetimes(src));
    for (src, ty) in [
        ("&'a [u8]", "&'a [u8]"),
        ("&'static [u8]", "&'static [u8]"),
        ("&'a(u8,u8)", "&'a (u8, u8)"),
        ("&'a&'b u8", "&'a &'b u8"),
        ("&'a*const u8", "&'a *const u8"),
        ("&'a<T as Tr>::Out", "&'a <T as Tr>::Out"),
        ("&'static ::core::ffi::CStr", "&'static ::core::ffi::CStr"),
        ("&'a ::my::Word", "&'a ::my::Word"),
        ("&'a::my::Word", "&'a ::my::Word"),
        ("Option<&'a ::my::Word>", "Option<&'a ::my::Word>"),
        ("Foo<'a, ::my::Word>", "Foo<'a, ::my::Word>"),
        ("Foo<'a,::my::Word>", "Foo<'a, ::my::Word>"),
        ("for<'a> fn(&'a ::core::ffi::CStr)", "for<'a> fn(&'a ::core::ffi::CStr)"),
        ("&'a mut u8", "&'a mut u8"),
        ("&'a !", "&'a !"),
        ("&'static!", "&'static !"),
        ("&'a mut !", "&'a mut !"),
        ("Option<&'a [u8]>", "Option<&'a [u8]>"),
        ("fn(&'a [u8],&'b str)->&'a str", "fn(&'a [u8], &'b str) -> &'a str"),
        ("Foo<'a,'b>", "Foo<'a, 'b>"),
        ("Foo<'a>", "Foo<'a>"),
        ("dyn A+'a", "dyn A + 'a"),
        ("Box<dyn 'a+A>", "Box<dyn 'a + A>"),
        ("impl 'a+A", "impl 'a + A"),
        ("<A<'a>as B>::C", "<A<'a> as B>::C"),
        ("Box<dyn Fn()+Send+'static>", "Box<dyn Fn() + Send + 'static>"),
        ("for<'a,'b:'a> fn(&'a u8,&'b u8)", "for<'a, 'b: 'a> fn(&'a u8, &'b u8)"),
        ("dyn for<'a> Tr<'a,Item:Copy>", "dyn for<'a> Tr<'a, Item: Copy>"),
    ] {
        assert_eq!(bound(src), ty, "{src}");
    }
}

/// A keyword written raw is a name: `r#mut` has none of the spaces `mut` has.
#[test]
fn a_keyword_written_raw_is_a_name_and_has_none_of_the_spaces_a_keyword_has() {
    let written = |src: &str| crate::text(&crate::lex(src).unwrap());
    for (src, ty) in [
        ("mut::Foo", "mut ::Foo"),
        ("r#mut::Foo", "r#mut::Foo"),
        ("*const r#mut::Foo", "*const r#mut::Foo"),
        ("r#const<A>", "r#const<A>"),
        ("dyn(A)", "dyn (A)"),
        ("r#dyn(A)", "r#dyn(A)"),
        ("&mut r#dyn::Foo", "&mut r#dyn::Foo"),
        ("r#impl[A]", "r#impl[A]"),
        ("Foo<r#as>", "Foo<r#as>"),
        ("Foo<r#as,r#mut>", "Foo<r#as, r#mut>"),
    ] {
        assert_eq!(written(src), ty, "{src}");
    }
}

/// The `>` before a `=` closes the arguments of a name, so the two are no `>=`: an associated type with
/// arguments is written `Item<X> = u8`, as rustfmt writes it, whether the source packs it, spaces it as rustfmt
/// does, or spaces one side of the `=`. And `as` has a space after it before a path from the root,
/// `<A as ::b::C>::D`.
#[test]
fn an_associated_type_with_arguments_and_a_path_from_the_root_after_as_are_written_as_rust_writes_them() {
    let written = |src: &str| crate::text(&crate::lex(src).unwrap());
    for ty in [
        "dyn Tr<Item<X> = u8>",
        "dyn Tr<Item<X> = u8, Other<Y, Z> = &u8>",
        "<A as ::b::C>::D",
        "<A as ::b::C<u8>>::D<u16>",
    ] {
        assert_eq!(written(ty), ty);
    }
    assert_eq!(written("<A as::b::C>::D"), "<A as ::b::C>::D");
    for (src, ty) in [
        ("dyn Tr<Item<X>=u8>", "dyn Tr<Item<X> = u8>"),
        ("dyn Tr<Item<X> =u8>", "dyn Tr<Item<X> = u8>"),
        ("dyn Tr<Item<X>= u8>", "dyn Tr<Item<X> = u8>"),
        ("dyn Tr<Item<X>=u8,Other<Y,Z>=&u8>", "dyn Tr<Item<X> = u8, Other<Y, Z> = &u8>"),
        ("dyn Tr<Item<Vec<X>>=u8>", "dyn Tr<Item<Vec<X>> = u8>"),
        ("dyn Tr<Item<X>=Vec<u8>>", "dyn Tr<Item<X> = Vec<u8>>"),
        ("dyn Tr<Item<fn()->u8>=u8>", "dyn Tr<Item<fn() -> u8> = u8>"),
        ("dyn Tr<Item<X>=fn()->u8>", "dyn Tr<Item<X> = fn() -> u8>"),
        ("dyn Tr<Item<X>={3}>", "dyn Tr<Item<X> = {3}>"),
        ("Box<dyn Tr<Item<X>=u8>+Send>", "Box<dyn Tr<Item<X> = u8> + Send>"),
        ("*const Foo<Item<X>=u8>", "*const Foo<Item<X> = u8>"),
    ] {
        assert_eq!(written(src), ty, "{src}");
    }
    let bound = |src: &str| crate::text(&lexed_with_lifetimes(src));
    for src in ["dyn Tr<Item<'a>=&'a u8>", "dyn Tr<Item<'a> =&'a u8>", "dyn Tr<Item<'a>= &'a u8>"] {
        assert_eq!(bound(src), "dyn Tr<Item<'a> = &'a u8>", "{src}");
    }
    assert_eq!(bound("dyn for<'a> Tr<Item<'a>=&'a u8>"), "dyn for<'a> Tr<Item<'a> = &'a u8>");
}

/// The lexer says, of each token, whether the source has a space before it: a space, a line break or a
/// comment; and of each group, whether it has one before its first token and before the bracket that
/// closes it.
#[test]
fn the_lexer_says_where_the_source_has_a_space() {
    let spaces = |src: &str| crate::lex(src).unwrap().iter().map(Tree::space).collect::<Vec<_>>();
    assert_eq!(spaces("a+b"), [false, false, false]);
    assert_eq!(spaces("a + b"), [false, true, true]);
    assert_eq!(spaces("a+ b"), [false, false, true]);
    assert_eq!(spaces("a +b"), [false, true, false]);
    assert_eq!(spaces("a  +\n\t b"), [false, true, true]);
    assert_eq!(spaces("a/**/+// c\nb"), [false, true, true]);
    assert_eq!(spaces("a/**/+b"), [false, true, false]);
    assert_eq!(spaces("a<<b a< <b"), [false, false, false, false, true, false, true, false]);
    let group = |src: &str| match crate::lex(src).unwrap().last() {
        Some(Tree::Group { trees, space, space_close, .. }) => {
            (*space, trees.first().is_some_and(Tree::space), *space_close)
        }
        _ => panic!("{src} does not end in a group"),
    };
    assert_eq!(group("f(a)"), (false, false, false));
    assert_eq!(group("f (a)"), (true, false, false));
    assert_eq!(group("f( a)"), (false, true, false));
    assert_eq!(group("f(a )"), (false, false, true));
    assert_eq!(group("f( a )"), (false, true, true));
    assert_eq!(group("f{}"), (false, false, false));
    assert_eq!(group("f{ }"), (false, false, true));
    assert_eq!(group("f{/**/}"), (false, false, true));
    assert_eq!(group("f{\n}"), (false, false, true));
}

/// Expressions, each as rustfmt writes it. The spaces that do not stand between two word characters are the
/// optional ones, as the ones around a `+` are, and the one in `as usize` is not.
const EXPRESSIONS: &[&str] = &[
    "3",
    "!0usize & 3",
    "A + B",
    "A - B",
    "A * B",
    "A / B",
    "A % B",
    "A ^ B",
    "A & B",
    "A | B",
    "A << 2",
    "A >> 2",
    "A == B",
    "A != B",
    "A < B",
    "A <= B",
    "A > B",
    "A >= B",
    "A && B",
    "A || B",
    "A += 1",
    "A -= 1",
    "A *= 1",
    "A /= 1",
    "A %= 1",
    "A ^= 1",
    "A &= 1",
    "A |= 1",
    "A <<= 1",
    "A >>= 1",
    "A = 1",
    "!A",
    "-A",
    "-1",
    "A - -1",
    "A & !B",
    "A & &B",
    "A < -1",
    "A * *B",
    "A && &B",
    "A + *B",
    "A == *B",
    "A * -1",
    "A == -1",
    "*A",
    "&A",
    "&mut A",
    "&mut *A",
    "*A + 1",
    "(A + B) * C",
    "f(-1)",
    "f(1, 2)",
    "f(A + 1)",
    "[1, 2][0]",
    "A[0] + A[1]",
    "A.b.c",
    "A.f().g()",
    "A::B + 1",
    "A as usize",
    "A as u8 as usize",
    "A as *const u8 as usize",
    "f as fn() -> u8",
    "f as fn(u8) -> u8 as usize",
    "A.0 - 1",
    "0..1",
    "0..N - 1",
    "A?",
    "1.5e-3",
    "\"a\".len()",
    "'a' as u32",
    "S { a: 1, b: -2 }",
    "m! { 1 }",
    "m!(1)",
    "if A > 1 { 1 } else { 2 }",
    "if !A { 1 } else { 2 }",
    "if !(A > 1) { 1 } else { 2 }",
    "let x = A + 1; x * 2",
    "let x: u8 = 1; x",
    "let mut i = 0; while i < N { i += 1; } i",
    "loop { break -1 }",
    "while !A {}",
    "let (a, b) = (1, 2); a",
    "match (A) {}",
    "return !A",
    "unsafe { A }",
    "{ A + 1 }",
    "size_of::<u8>()",
    "size_of::<[u8; 4]>() * 2",
    "size_of::<Vec<u8>>()",
    "size_of::<fn() -> u8>()",
    "size_of::<fn(u8) -> u8>() + 1",
    "f::<fn() -> u8>() > 2",
    "f::<Vec<u8>, Vec<u8>>()",
    "f::<A<B<C>>, D<E>>() > 2",
    "N as usize & 3",
    "f::<u8>(1) > 2",
    "f::<u8>(1) >> 2",
    "f::<{ 3 }>(1)",
    "<u8 as T>::C",
    "<u8>::C > 1",
    "<Cfg<4> as Tile>::N",
    "<Vec<u8> as Tr>::N",
    "<u8 as Tr<3>>::N",
    "<Foo<u8>>::N",
    "<Foo<u8>>::N < 3",
    "<<u8 as Tr>::A as Tr>::B",
    "A && <u8>::MAX > 3",
    "<u8>::MAX << 1",
    "let v: Option<usize> = Some(3); match v { Some(n) => n, None => 0 }",
    "match A { 1 => 2, _ => 3 }",
    "match A { n @ 1..=3 => n, _ => 0 }",
    "match A { 1 | 2 => 3, _ => 4 }",
    "if A { 1 } else { 2 }",
    "f::<u8>() < 3",
    "A < B && C > D",
    "(A < B) as usize",
    "A as i32 - 1",
    "x as *const u8 as usize",
    "[0u8; 4].len()",
    "A - (-1)",
    "S { a: 1 }.a",
    "Foo::<u8>::N + 1",
    "size_of::<u8>() * 2",
    "{ 3 }",
    "A::B",
    "*&A",
    "A.0",
    "A + B * C",
    "A < B || f::<u8>() > 2",
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
    "loop { break 3 }",
    "(3, 4).0",
    "true as usize",
    "A | B | C",
    "0 as *const Vec<u8> as usize",
    "|x: Vec<u8>| x",
    "|x| x",
    "let f = |x: Vec<u8>| x; 3",
    "let v: Option<usize> = Some(3); 3",
    "(0 as *const Vec<u8>).is_null() as usize",
    "N * 2",
    "<Cfg as Tile>::N",
    "A < -B",
    "1..=3",
    "f::<{ 3 }>()",
    "size_of::<[u8; 4]>()",
    "a + b",
    "a: b",
    "a & b",
    "A >> 1",
    "unsafe { 3 }",
    "(3)",
];

/// `row` with its optional spaces kept or taken out in each way that tells one from another: as written, all
/// taken out, and each one alone kept, and alone taken out.
fn spaced_both_ways(row: &str) -> Vec<String> {
    let chars: Vec<char> = row.chars().collect();
    let word = |c: char| c.is_alphanumeric() || c == '_';
    let optional: Vec<usize> =
        (1..chars.len() - 1).filter(|&i| chars[i] == ' ' && !(word(chars[i - 1]) && word(chars[i + 1]))).collect();
    let without = |gone: &[usize]| {
        chars.iter().enumerate().filter(|(i, _)| !gone.contains(i)).map(|(_, c)| *c).collect::<String>()
    };
    let mut out = vec![row.to_string(), without(&optional)];
    for &gap in &optional {
        out.push(without(&[gap]));
        out.push(without(&optional.iter().copied().filter(|&o| o != gap).collect::<Vec<_>>()));
    }
    out
}

/// A block is written with the spaces its source has, as the messages of rustc write one they do not work out:
/// `Marker<{N*2}>` from `{N*2}`, `Marker<{ N * 2 }>` from `{ N * 2 }`. So it is in the packed form, in the form
/// rustfmt writes, and with one space of the form, or one space less, at a time. The type around it is spaced as
/// before.
#[test]
fn a_block_is_written_as_its_source_has_it() {
    let written = |src: &str| crate::text(&crate::lex(src).unwrap());
    for expr in EXPRESSIONS {
        for src in spaced_both_ways(&format!("Marker<{{ {expr} }}>")) {
            assert_eq!(written(&src), src);
        }
    }
    // Rows of the review: the operators of a comparison are not the brackets of a type, and the other way round.
    assert_eq!(written("Marker<{!0usize&3}>"), "Marker<{!0usize&3}>");
    assert_eq!(written("Marker<{ !0usize & 3 }>"), "Marker<{ !0usize & 3 }>");
    assert_eq!(written("Marker<{<Vec<u8>as Tr>::N}>"), "Marker<{<Vec<u8>as Tr>::N}>");
    assert_eq!(written("Marker<{let v:Option<usize>=Some(3);v}>"), "Marker<{let v:Option<usize>=Some(3);v}>");
    assert_eq!(written("Marker<{N*2}>"), "Marker<{N*2}>");
    assert_eq!(written("Marker<{ N * 2 }>"), "Marker<{ N * 2 }>");
    assert_eq!(written("Marker<{N * 2}>"), "Marker<{N * 2}>");
    assert_eq!(written("Marker<{ N*2 }>"), "Marker<{ N*2 }>");
    // An empty block has none, or the one the source has.
    assert_eq!(written("Marker<{}>"), "Marker<{}>");
    assert_eq!(written("Marker<{ }>"), "Marker<{ }>");
    // A run of spaces, line breaks and comments is one space, and a comment is a space where it stands.
    assert_eq!(written("Marker<{  N   *\t2  }>"), "Marker<{ N * 2 }>");
    assert_eq!(written("Marker<{\n    N *\n    2\n}>"), "Marker<{ N * 2 }>");
    assert_eq!(written("Marker<{ // twice\n    N *\n    2 // the last\n}>"), "Marker<{ N * 2 }>");
    assert_eq!(written("Marker<{ /* a */ N /* b */ * /* c */ 2 /* d */ }>"), "Marker<{ N * 2 }>");
    assert_eq!(written("Marker<{N/* b */*2}>"), "Marker<{N *2}>");
    assert_eq!(written("Marker<{N*/* c */2}>"), "Marker<{N* 2}>");
    assert_eq!(written("Marker<{/* a */N*2/* d */}>"), "Marker<{ N*2 }>");
    // The block inside a block, and the array inside it, are as the source has them too.
    assert_eq!(written("Marker<{{N*2}}>"), "Marker<{{N*2}}>");
    assert_eq!(written("Marker<{ { N * 2 } }>"), "Marker<{ { N * 2 } }>");
    assert_eq!(written("Marker<{size_of::<[u8;4]>()}>"), "Marker<{size_of::<[u8;4]>()}>");
    assert_eq!(written("Marker<{size_of::<[u8; 4]>()}>"), "Marker<{size_of::<[u8; 4]>()}>");
}

/// The type's own tokens around a block are spaced as ever, whatever the source has; the block is not.
#[test]
fn the_type_around_a_block_is_spaced_as_before() {
    let written = |src: &str| crate::text(&crate::lex(src).unwrap());
    for (src, ty) in [
        ("Pair<u8,{3}>", "Pair<u8, {3}>"),
        ("Pair<u8 , { 3 }>", "Pair<u8, { 3 }>"),
        ("Pair<{3},{4}>", "Pair<{3}, {4}>"),
        ("Pair<{ 3 } , { 4 }>", "Pair<{ 3 }, { 4 }>"),
        ("[Arr<{3}>;1]", "[Arr<{3}>; 1]"),
        ("[ Arr < { 3 } > ; 1 ]", "[Arr<{ 3 }>; 1]"),
        ("fn()->Arr<{3}>", "fn() -> Arr<{3}>"),
        ("<A as B>::C<{3}>", "<A as B>::C<{3}>"),
        ("<A  as  B> :: C <  { 3 } >", "<A as B>::C<{ 3 }>"),
        ("dyn Tr<N={3}>", "dyn Tr<N = {3}>"),
        ("dyn Tr<N={ 3 }>", "dyn Tr<N = { 3 }>"),
        ("*  const Marker < { N*2 } >", "*const Marker<{ N*2 }>"),
        ("*mut*const Marker<{N*2}>", "*mut *const Marker<{N*2}>"),
        ("*const Marker<{!0usize&3}>", "*const Marker<{!0usize&3}>"),
        ("*const Marker<{ !0usize & 3 }>", "*const Marker<{ !0usize & 3 }>"),
        ("(Marker<{N*2}>,u8)", "(Marker<{N*2}>, u8)"),
        ("&mut Marker<{N*2}>", "&mut Marker<{N*2}>"),
        ("Marker<{N*2},{ N * 2 }>", "Marker<{N*2}, { N * 2 }>"),
    ] {
        assert_eq!(written(src), ty, "{src}");
    }
}

/// The length of an array, all that follows its `;`, is written as its source has it, one space after the
/// `;` and none before the `]`: `[u8; N*2]` from `[u8;N*2]` and `[u8; N * 2]` from `[u8 ; N * 2 ]`.
#[test]
fn an_arrays_length_is_written_as_its_source_has_it() {
    let written = |src: &str| crate::text(&crate::lex(src).unwrap());
    // A length is an expression; statements, `let x = 1; x`, are only a block's.
    for expr in EXPRESSIONS.iter().filter(|e| !e.starts_with("let ")) {
        for src in spaced_both_ways(&format!("[u8; {expr}]")) {
            let length = src["[u8;".len()..src.len() - 1].trim_start();
            assert_eq!(written(&src), format!("[u8; {length}]"), "{src}");
        }
    }
    for (src, ty) in [
        ("[u8;N*2]", "[u8; N*2]"),
        ("[u8; N * 2]", "[u8; N * 2]"),
        ("[u8 ;N * 2 ]", "[u8; N * 2]"),
        ("[ u8 ; N*2 ]", "[u8; N*2]"),
        ("[u8;N * 2]", "[u8; N * 2]"),
        ("[u8;N* 2]", "[u8; N* 2]"),
        ("[u8;{3}]", "[u8; {3}]"),
        ("[u8;{ 3 }]", "[u8; { 3 }]"),
        ("[u8;<Cfg<4> as Tile>::N]", "[u8; <Cfg<4> as Tile>::N]"),
        ("[u8;<Cfg<4>as Tile>::N]", "[u8; <Cfg<4>as Tile>::N]"),
        ("[u8;size_of::<u8>()*2]", "[u8; size_of::<u8>()*2]"),
        ("[u8;\n    N *\n    2\n]", "[u8; N * 2]"),
        ("[u8;N/* c */*2]", "[u8; N *2]"),
        // The element is the type's own tokens, the length is not.
        ("[Arr<{3}>;N*2]", "[Arr<{3}>; N*2]"),
        ("[*const u8;N*2]", "[*const u8; N*2]"),
        ("*const [u8;N*2]", "*const [u8; N*2]"),
        ("*const [u8;N * 2]", "*const [u8; N * 2]"),
        ("[[u8;N*2];M+1]", "[[u8; N*2]; M+1]"),
        ("[[u8; N * 2]; M + 1]", "[[u8; N * 2]; M + 1]"),
        ("Pair<[u8;N*2],[u8;N * 2]>", "Pair<[u8; N*2], [u8; N * 2]>"),
        ("fn([u8;N*2])->[u8;N * 2]", "fn([u8; N*2]) -> [u8; N * 2]"),
        // A `;` with nothing after it, as the token a message says it found, is the `;` alone.
        (";", ";"),
        ("[u8;]", "[u8;]"),
        // A `;` is the end of the element only in an array: a slice, a tuple and a group hold none.
        ("&[u8]", "&[u8]"),
        ("&mut [Arr<{3}>]", "&mut [Arr<{3}>]"),
    ] {
        assert_eq!(written(src), ty, "{src}");
    }
}

/// The arguments of a macro, in any of its brackets, are written as their source has them; the type around
/// them is spaced as ever, and so is the space before the braces of a macro, `byte! {..}`.
#[test]
fn a_macros_arguments_are_written_as_their_source_has_them() {
    let written = |src: &str| crate::text(&crate::lex(src).unwrap());
    for expr in EXPRESSIONS {
        for (open, close) in [("(", ")"), ("[", "]")] {
            for src in spaced_both_ways(&format!("m!{open} {expr} {close}")) {
                assert_eq!(written(&src), src);
            }
        }
        for src in spaced_both_ways(&format!("m!{{ {expr} }}")) {
            assert_eq!(written(&src), src.replacen("!{", "! {", 1));
        }
    }
    for (src, ty) in [
        ("m!(a+b)", "m!(a+b)"),
        ("m!(a + b)", "m!(a + b)"),
        ("m!(a +b)", "m!(a +b)"),
        ("m!(a+ b)", "m!(a+ b)"),
        ("m!( a + b )", "m!( a + b )"),
        ("m!(a:b)", "m!(a:b)"),
        ("m!(a: b)", "m!(a: b)"),
        ("m!(a,b)", "m!(a,b)"),
        ("m!(a, b)", "m!(a, b)"),
        ("m!{a&b}", "m! {a&b}"),
        ("m! { a & b }", "m! { a & b }"),
        ("m ! ( a+b )", "m!( a+b )"),
        ("m!(\n    a +\n    b\n)", "m!( a + b )"),
        ("m!(a/* c */+b)", "m!(a +b)"),
        ("m!()", "m!()"),
        ("m!( )", "m!( )"),
        ("m!{}", "m! {}"),
        ("m!{ }", "m! { }"),
        // The type around a macro, and a macro in the arguments of one.
        ("*const m!(a+b)", "*const m!(a+b)"),
        ("Wrap<m!(a+b),u8>", "Wrap<m!(a+b), u8>"),
        ("[m!(a+b);N*2]", "[m!(a+b); N*2]"),
        ("fn(u8)->m!(a+b)", "fn(u8) -> m!(a+b)"),
        ("crate::m!(a+b)", "crate::m!(a+b)"),
        ("m!(n!(a+b),c)", "m!(n!(a+b),c)"),
        ("m!(n!{a+b},c)", "m!(n!{a+b},c)"),
        ("m!(n! { a + b }, c)", "m!(n! { a + b }, c)"),
        ("m!([a;3],{b})", "m!([a;3],{b})"),
        // A group that follows a name but no `!` is the type's own, as `Fn(u8)` is.
        ("Box<dyn Fn( u8 )>", "Box<dyn Fn(u8)>"),
        ("Box<dyn Fn(a:u8)>", "Box<dyn Fn(a: u8)>"),
        // The `!` of a never type is no macro's, and the tuple beside it is the type's own.
        ("fn()->!", "fn() -> !"),
        ("Foo<!,(a:u8)>", "Foo<!, (a: u8)>"),
    ] {
        assert_eq!(written(src), ty, "{src}");
    }
}

/// A type that begins with a binder, `for<'a> fn(&'a u8)`, is read whole, as any other type is: the
/// `<` after `for` opens the binder, not the arguments of a type named `for`. So it is one type as a
/// parameter, which starts at its `for`, as the argument of another type, `Option<for<'a> fn(&'a u8)>`,
/// and as a result.
#[test]
fn a_type_that_begins_with_a_binder_is_read_whole() {
    let src = "pub unsafe extern \"sysv64\" fn k(x: for<'a> fn(&'a u8), y: Option<for<'a> fn(&'a u8)>, \
               z: Option<&dyn for<'a> Fn(&'a u8)>) -> for<'a> fn(&'a u8) {}";
    let module = crate::module(&lexed_with_lifetimes(src), 0).unwrap();
    let crate::ItemKind::Fn(f) = &module.items[0].kind else { panic!("a function") };
    let types: Vec<String> = f.params.iter().map(|p| shown(&p.ty)).collect();
    assert_eq!(types, ["for<'a> fn(&'a u8)", "Option<for<'a> fn(&'a u8)>", "Option<&dyn for<'a> Fn(&'a u8)>"]);
    // No `'` comes before the first `for`, so the lexer's offset is the source's.
    assert_eq!(f.params[0].ty.span(), Span(src.find("for<").unwrap() as u32));
    assert_eq!(f.ret.as_ref().map(|r| r.ty.as_str()), Some("for<'a> fn(&'a u8)"));
}

/// A name written raw is the name Rust reads: `r#u64` is `u64`. A type is read so wherever it is written: a
/// parameter's, a result's, a const's, a static's, a variable's, an array's element, an argument of a type, and
/// the names of a type read as text, `*const [r#f16; 4]`, at any depth. `text` writes a type as it was written.
#[test]
fn a_name_written_raw_is_the_name_rust_reads() {
    assert_eq!((crate::unraw("r#u64"), crate::unraw("u64"), crate::unraw("r#type")), ("u64", "u64", "type"));
    let read = |src: &str| crate::read_type(&crate::lex(src).unwrap());
    for (written, name) in [
        ("r#u64", "u64"),
        ("*const r#u8", "*const u8"),
        ("*const [r#f16; 4]", "*const [f16; 4]"),
        ("(r#u8, Pair<r#u16, 2>, fn(r#i8) -> r#i16)", "(u8, Pair<u16, 2>, fn(i8) -> i16)"),
        ("&mut r#str", "&mut str"),
        ("*const u8", "*const u8"),
    ] {
        assert_eq!(read(written), name, "{written}");
        assert_eq!(crate::text(&crate::lex(written).unwrap()), written, "{written}");
    }
    // A block among the arguments of a type holds names too, which `text` writes as the source has them.
    assert_eq!((read("Arr<{ r#N }>"), read("Arr<{r#N}>")), ("Arr<{ N }>".into(), "Arr<{N}>".into()));
    let src = r#"
        const N: r#usize = 4;
        static S: r#global<[r#u32; 4]>;
        pub unsafe extern "sysv64" fn k(x: r#u64, ys: [r#f32; 2], p: *const r#str) -> r#u64 {
            let v: r#ymm;
            let w: [r#ymm; 2];
            let n: r#u32 = 3;
        }
    "#;
    let module = parse(src).unwrap();
    let kinds: Vec<_> = module.items.iter().map(|item| &item.kind).collect();
    let [crate::ItemKind::Const(c), crate::ItemKind::Static(s), crate::ItemKind::Fn(f)] = kinds[..] else {
        panic!("a const, a static and a function")
    };
    assert_eq!((shown(&c.ty), shown(&s.ty)), ("usize".into(), "global<[u32; ..]>".into()));
    let params: Vec<String> = f.params.iter().map(|p| shown(&p.ty)).collect();
    assert_eq!(params, ["u64", "[f32; ..]", "*const str"]);
    assert_eq!(f.ret.as_ref().map(|r| r.ty.as_str()), Some("u64"));
    let lets = f.body.as_ref().unwrap().stmts.iter();
    let lets: Vec<String> = lets
        .filter_map(|s| match s {
            crate::Stmt::Let { ty: Some(ty), .. } => Some(shown(ty)),
            _ => None,
        })
        .collect();
    assert_eq!(lets, ["ymm", "[ymm; ..]", "u32"]);
    // A pointer to a `str` written raw is a pointer to a `str`, which has two words.
    let words = "a pointer to `str` has two words, an address and a length or a table, and a kernel's pointer has one";
    assert_eq!(crate::pointer(&params[2]), Err(words.into()));
}

/// A keyword written raw is a name, and no keyword: `read_type` leaves its `r#`, at any depth of a type and for every
/// keyword Rust lets be written raw, so that no reader of the type takes the name for the keyword. A pointer to
/// `r#dyn::Foo`, a struct, has one word, where a pointer to `dyn Foo` has two; `r#mut<u8>` is a name with an argument,
/// written close, where `mut <u8>` is not; `Arr<{r#true}>` holds a name, and not the literal. A word that is no
/// keyword loses its `r#`, a weak keyword too, and a generic's name is read so as the rest is.
#[test]
fn a_keyword_written_raw_stays_a_name_in_a_type() {
    // The keywords Rust lets be written raw, as the Rust Reference lists them: the strict ones, then the reserved ones.
    let keywords = [
        "as", "async", "await", "break", "const", "continue", "dyn", "else", "enum", "extern", "false", "fn", "for",
        "if", "impl", "in", "let", "loop", "match", "mod", "move", "mut", "pub", "ref", "return", "static", "struct",
        "trait", "true", "type", "unsafe", "use", "where", "while", "abstract", "become", "box", "do", "final", "gen",
        "macro", "override", "priv", "try", "typeof", "unsized", "virtual", "yield",
    ];
    let read = |src: &str| crate::read_type(&crate::lex(src).unwrap());
    for word in keywords {
        for ty in [
            format!("r#{word}"),
            format!("*const r#{word}::Foo"),
            format!("*mut r#{word}<u8>"),
            format!("[r#{word}; 2]"),
            format!("(r#{word}, r#{word}::Foo)"),
            format!("fn(r#{word}) -> r#{word}"),
            format!("Arr<{{r#{word}}}>"),
        ] {
            assert_eq!(read(&ty), ty, "{ty}");
        }
        // A pointer to a type that the word names has one word.
        for pointer in [format!("*const r#{word}::Foo"), format!("*mut r#{word}<u64>")] {
            assert_eq!(crate::pointer(&read(&pointer)), Ok(true), "{pointer}");
        }
    }
    let words = |ty: &str| crate::pointer(ty).unwrap_err();
    assert!(words(&read("*const dyn Foo")).starts_with("a pointer to `dyn Foo` has two words"));
    assert!(words(&read("*const r#str")).starts_with("a pointer to `str` has two words"));
    assert!(words(&read("*const [r#mut]")).starts_with("a pointer to `[r#mut]` has two words"));
    assert!(words(&read("*const (u8, r#str)")).starts_with("a pointer to `(u8, str)` has two words"));
    for word in ["union", "macro_rules", "raw", "safe", "auto", "default", "u64", "str", "f16", "gpr", "Foo", "dyn_"] {
        assert_eq!(read(&format!("*const r#{word}::Foo")), format!("*const {word}::Foo"), "{word}");
    }
    // The parser holds a name that is a keyword written raw as `read_type` gives it,
    // in a generic's name as in the rest.
    let module = parse("static S: r#mut<[r#dyn; 4], r#u32>; static T: r#global<r#true>;").unwrap();
    let kinds: Vec<_> = module.items.iter().map(|item| &item.kind).collect();
    let [crate::ItemKind::Static(s), crate::ItemKind::Static(t)] = kinds[..] else { panic!("two statics") };
    assert_eq!((shown(&s.ty), shown(&t.ty)), ("r#mut<[r#dyn; ..], u32>".into(), "global<r#true>".into()));
}

/// An attribute of kernite's written raw is the attribute, as Rust reads `r#cfg` as `cfg`: a function holds its
/// `#[r#instances(..)]`, its `#[r#cooperative]` and its directives, each under the name Rust reads, and
/// `#[r#allow(..)]`, which is Rust's, is no directive.
#[test]
fn an_attribute_written_raw_is_the_attribute() {
    let src = r#"
        #[r#instances(1, 2)]
        #[r#cooperative]
        #[r#maxntid(16, 16)]
        #[r#reqntid(8)]
        #[r#minnctapersm(2)]
        #[r#maxnreg(32)]
        #[r#allow(dead_code)]
        pub unsafe extern "ptx-kernel" fn k<const N: u32>() {}
    "#;
    let module = parse(src).unwrap();
    let crate::ItemKind::Fn(f) = &module.items[0].kind else { panic!("a function") };
    assert_eq!((f.instances.is_some(), f.cooperative.is_some()), (true, true));
    let directives: Vec<&str> = f.directives.iter().map(|d| d.name.as_str()).collect();
    assert_eq!(directives, ["maxntid", "reqntid", "minnctapersm", "maxnreg"]);
    // Each is where it is written, at its `r#`, for an error to point there.
    let at = |name: &str| Span(src.find(name).unwrap() as u32);
    assert_eq!(f.instances.as_ref().map(|i| i.span), Some(at("r#instances")));
    assert_eq!(f.cooperative.as_ref().map(|c| c.span), Some(at("r#cooperative")));
    let spans: Vec<Span> = f.directives.iter().map(|d| d.span).collect();
    assert_eq!(spans, ["r#maxntid", "r#reqntid", "r#minnctapersm", "r#maxnreg"].map(at));
}

/// A path is names joined by `::`, with a `::` before the first or not, and `a::b` is four trees: a name, two colons
/// and a name. The words that a type or a bound follows, `as`, `const`, `dyn`, `impl` and `mut`, are no names of one,
/// but the same written raw is a name. Two colons make a `::` only when no space is between them; a name is a path
/// whether a `::` comes after it or not, and a `::` with no name after it is no part of one.
#[test]
fn a_path_is_names_joined_by_double_colons() {
    let count = |src: &str| crate::token::path(&crate::lex(src).unwrap());
    for (src, trees) in [
        ("a", 1),
        ("a::b", 4),
        ("a::b::c", 7),
        ("::a", 3),
        ("::a::b", 6),
        (" :: a :: b ", 6),
        ("a::b + c", 4),
        ("a::b<c>", 4),
        ("r#a::r#b", 4),
        ("a::", 1),
        ("a::b::", 4),
        ("::", 0),
        ("::::a", 0),
        (":a", 0),
        ("a:b", 1),
        ("a: :b", 1),
        ("a : : b", 1),
        ("1::a", 0),
        ("(a)::b", 0),
        ("<a>::b", 0),
        ("", 0),
    ] {
        assert_eq!(count(src), trees, "{src}");
    }
    for word in ["as", "const", "dyn", "impl", "mut"] {
        for src in [format!("{word}::a"), format!("::{word}::a"), word.to_string(), format!("a::{word}::b")] {
            let want = if src.starts_with("a::") { 1 } else { 0 };
            assert_eq!(count(&src), want, "{src}");
        }
        assert_eq!(count(&format!("r#{word}::a")), 4, "r#{word}");
        assert_eq!(count(&format!("::r#{word}::a")), 6, "::r#{word}");
    }
}

/// `primitive` reads the third name of `core::primitive::NAME`, or of `std::primitive::NAME`, with a `::` before the
/// first or not, when `core::primitive` has a type of that name. It takes seven trees, or nine with the `::`. `r#` on a
/// name is Rust's way to write it, so it is dropped from each name of the path, `::core::primitive::r#u64` and
/// `r#core::primitive::u64` as the others. The path may go on past the primitive's, and other trees may follow it;
/// any other path begins with no primitive's.
#[test]
fn a_path_begins_with_a_primitive_when_core_or_std_primitive_names_one() {
    let begins = |src: &str| crate::token::primitive(&crate::lex(src).unwrap());
    for (src, want) in [
        ("core::primitive::u64", Some(("u64", 7))),
        ("std::primitive::str", Some(("str", 7))),
        ("::core::primitive::bool", Some(("bool", 9))),
        ("::std::primitive::f32", Some(("f32", 9))),
        (" :: std :: primitive :: u128 ", Some(("u128", 9))),
        ("r#core::r#primitive::r#i8", Some(("i8", 7))),
        ("::r#std::r#primitive::r#usize", Some(("usize", 9))),
        ("::core::primitive::r#u64", Some(("u64", 9))),
        ("r#core::primitive::u64", Some(("u64", 7))),
        ("core::primitive::u64::MAX", Some(("u64", 7))),
        ("::std::primitive::u8::MAX::X", Some(("u8", 9))),
        ("core::primitive::u64 + 1", Some(("u64", 7))),
        ("::std::primitive::f32::from_bits(1)", Some(("f32", 9))),
        ("core::primitive::u64, std::primitive::u8", Some(("u64", 7))),
        ("core::primitive::f16", None),
        ("core::primitive::f128", None),
        ("core::primitive::U64", None),
        ("core::primitive::u65", None),
        ("core::primitive", None),
        ("::core::primitive", None),
        ("core::primitive::", None),
        ("my::core::primitive::u64", None),
        ("::my::std::primitive::u64", None),
        ("alloc::primitive::u64", None),
        ("core::prim::u64", None),
        ("core::u64::MAX", None),
        ("::core::u64::MAX", None),
        ("::primitive::u64", None),
        (": :core::primitive::u64", None),
        ("-core::primitive::u64", None),
        ("(core::primitive::u64)", None),
        ("core: :primitive::u64", None),
        ("core::primitive: :u64", None),
        ("u64", None),
        ("u64::MAX", None),
        ("core", None),
        ("", None),
    ] {
        assert_eq!(begins(src), want, "{src}");
    }
    // Fewer trees than a primitive's path has are the path of none.
    assert_eq!(crate::token::primitive(&[]), None);
    assert_eq!(crate::token::primitive(&crate::lex("core::primitive").unwrap()), None);
}

/// Rust names each primitive by a path as well as by its name, and the language reads the name: a type written with
/// `core::primitive::u64`, `::core::primitive::u64`, `std::primitive::u64` or `::std::primitive::u64` is the type
/// written with `u64`, wherever in it the path is, and so is a compile-time argument's type and a cast's. So it is
/// when a name of the path is written raw, `::core::primitive::r#u64` and `r#core::primitive::u64`: one rule reads
/// the names written raw and the paths of the primitives. A lifetime's name begins no path, so
/// `&'static ::core::primitive::u64` is `&'static u64`. A path that is no primitive's of `core::primitive` stays as
/// written but for the names written raw in it, which are the names Rust reads: one that goes on, one that begins
/// elsewhere, one to a type `core::primitive` has not, and a module's.
#[test]
fn a_primitive_is_read_by_its_name_whether_it_is_written_by_a_path_or_not() {
    let read = |src: &str| crate::read_type(&crate::lex(src).unwrap());
    let names = [
        "bool", "char", "f32", "f64", "i8", "i16", "i32", "i64", "i128", "isize", "str", "u8", "u16", "u32", "u64",
        "u128", "usize",
    ];
    for name in names {
        for root in ["core", "::core", "std", "::std", "r#core", " core ", ":: std"] {
            let path = format!("{root}::primitive::{name}");
            assert_eq!(read(&path), name, "{path}");
            assert_eq!(read(&format!("{root}::r#primitive::r#{name}")), name, "{path} written raw");
        }
    }
    for (src, want) in [
        ("*const core::primitive::u64", "*const u64"),
        ("*mut ::std::primitive::f32", "*mut f32"),
        ("*const *mut core::primitive::u8", "*const *mut u8"),
        ("[core::primitive::u8; 4]", "[u8; 4]"),
        ("*const [::core::primitive::u8; 4]", "*const [u8; 4]"),
        ("(core::primitive::u32, ::core::primitive::f64)", "(u32, f64)"),
        ("&mut [std::primitive::u16]", "&mut [u16]"),
        ("global<[core::primitive::u32; 4]>", "global<[u32; 4]>"),
        ("fn(core::primitive::u8) -> ::core::primitive::u8", "fn(u8) -> u8"),
        ("Pair<core::primitive::u8, 2>", "Pair<u8, 2>"),
        ("core :: primitive :: u64", "u64"),
        // The words a type is spaced by, `as`, `const`, `dyn`, `impl` and `mut`, are no names of a path, so a `::`
        // after one begins a path of its own.
        ("dyn core::primitive::u8", "dyn u8"),
        ("&mut core::primitive::u8", "&mut u8"),
        ("<core::primitive::u8 as Tr>::T", "<u8 as Tr>::T"),
        ("dyn ::core::primitive::u8", "dyn u8"),
        ("impl ::std::primitive::u8", "impl u8"),
        ("*const ::core::primitive::u8", "*const u8"),
        ("*mut ::std::primitive::u8", "*mut u8"),
        ("&mut ::core::primitive::u8", "&mut u8"),
        ("<::core::primitive::u8 as ::std::primitive::u16>::T", "<u8 as u16>::T"),
        // A block is written as its source has it, so the name that stands for a path has the space the path had.
        ("Marker<{ N as core::primitive::u8 }>", "Marker<{ N as u8 }>"),
        ("Marker<{N as ::std::primitive::u8}>", "Marker<{N as u8}>"),
        ("Marker<{ N as(core::primitive::u8) }>", "Marker<{ N as(u8) }>"),
        // A keyword written raw is a name, so it can begin a path, and be one beside it.
        ("Foo<r#mut, ::core::primitive::u8>", "Foo<r#mut, u8>"),
        ("r#dyn<::std::primitive::u8>", "r#dyn<u8>"),
        // A name written raw is the name Rust reads, whichever name of the path it is. One rule reads the names
        // written raw and the primitive's path, so a path that both touch is the primitive's name.
        ("::core::primitive::r#u64", "u64"),
        ("r#core::primitive::u64", "u64"),
        ("*const ::std::primitive::r#f32", "*const f32"),
        ("[r#std::r#primitive::u8; 4]", "[u8; 4]"),
        ("(r#core::primitive::u32, ::core::r#primitive::f64)", "(u32, f64)"),
    ] {
        assert_eq!(read(src), want, "{src}");
    }
    // A lifetime is a `'` and a name, and its name begins no path: `'static ::core::primitive::u8` is a lifetime and a
    // path. The tokens are made as the macro is given them, for the lexer of a string refuses a lifetime.
    let bound = |src: &str| crate::read_type(&lexed_with_lifetimes(src));
    for (src, want) in [
        ("&'static ::core::primitive::u8", "&'static u8"),
        ("&'static core::primitive::u8", "&'static u8"),
        ("&'a ::std::primitive::str", "&'a str"),
        ("&'a mut ::core::primitive::u8", "&'a mut u8"),
        ("&'_ ::std::primitive::u16", "&'_ u16"),
        ("Foo<'a, ::core::primitive::u8>", "Foo<'a, u8>"),
        ("(&'a ::core::primitive::u8, &'b std::primitive::i8)", "(&'a u8, &'b i8)"),
        ("for<'a> fn(&'a ::core::primitive::u8) -> &'a ::std::primitive::u8", "for<'a> fn(&'a u8) -> &'a u8"),
        ("&'a u8", "&'a u8"),
    ] {
        assert_eq!(bound(src), want, "{src}");
    }
    // A `'` with no name after it is no lifetime, and nothing to take the tokens that follow it for.
    let quote = Tree::Punct { ch: '\'', space: false, span: Span(0) };
    assert_eq!(crate::read_type(std::slice::from_ref(&quote)), "'");
    // Any other path stays: the types that `core::primitive` has not on this Rust, `f16` and `f128` say, a path that
    // goes on, one that begins elsewhere, one that is only a part, and the types of the module.
    for path in [
        "core::primitive::f16",
        "core::primitive::f128",
        "core::primitive::bf16",
        "core::primitive::s32",
        "core::primitive::U64",
        "core::primitive::u64::MAX",
        "my::core::primitive::u64",
        "crate::core::primitive::u64",
        "super::std::primitive::u64",
        "r#mut::core::primitive::u64",
        "r#dyn::std::primitive::u64",
        "core::primitive",
        "core::primitive::",
        // Two colons are a `::` only when no space is between them.
        "core: :primitive: :u64",
        "primitive::u64",
        "core::u64",
        "std::u64",
        "core::f32::consts::PI",
        "crate::Word",
        "u64::MAX",
        "*const my::core::primitive::u64",
        "[my::std::primitive::u8; 4]",
    ] {
        assert_eq!(read(path), crate::text(&crate::lex(path).unwrap()), "{path}");
    }
    // A path that is no primitive's is a path still, but for the names written raw in it, which are the names Rust
    // reads. A keyword written raw stays so.
    for (src, want) in [
        ("my::r#core::primitive::u64", "my::core::primitive::u64"),
        ("core::primitive::r#f16", "core::primitive::f16"),
        ("::r#core::primitive::r#u64::r#MAX", "::core::primitive::u64::MAX"),
        ("my::r#mut::r#Word", "my::r#mut::Word"),
    ] {
        assert_eq!(read(src), want, "{src}");
    }
    // A type is read as a path where the parser reads one: a function's parameters, its result, its compile-time
    // arguments and a cast.
    let src = "pub unsafe extern \"sysv64\" fn k<const N: core::primitive::usize, const R: r#u16, \
               const M: ::core::primitive::r#u64, const L: r#core::primitive::u8>(\
               x: ::core::primitive::u64, p: *mut std::primitive::u8, \
               q: [::std::primitive::f32; core::primitive::u8::BITS as usize]) \
               -> ::std::primitive::u32 { let y = x as core::primitive::u8; let z = y as r#u32; }";
    let module = crate::module(&crate::lex(src).unwrap(), 0).unwrap();
    let crate::ItemKind::Fn(f) = &module.items[0].kind else { panic!("a function") };
    // A name written raw is the name, as it is anywhere else.
    assert_eq!(f.generics.iter().map(|g| g.ty.as_str()).collect::<Vec<_>>(), ["usize", "u16", "u64", "u8"]);
    let types: Vec<String> = f.params[..2].iter().map(|p| shown(&p.ty)).collect();
    assert_eq!(types, ["u64", "*mut u8"]);
    let crate::Type::Array(elem, len, _) = &f.params[2].ty else { panic!("an array") };
    assert_eq!(shown(elem), "f32");
    // The length is a value, and its path is read from the primitive's name too.
    let crate::Expr::Cast(bits, ty, _) = len else { panic!("a cast") };
    assert!(matches!(&**bits, crate::Expr::Path(names, _) if names == &["u8", "BITS"]) && ty == "usize");
    let at = |path: &str| Span(src.find(path).unwrap() as u32);
    assert_eq!(f.generics[0].span, at("N:"));
    assert_eq!(f.params[0].ty.span(), at("::core::primitive::u64"));
    assert_eq!(f.params[1].ty.span(), at("*mut std::primitive::u8"));
    assert_eq!(f.ret.as_ref().map(|r| (r.ty.as_str(), r.span)), Some(("u32", at("::std::primitive::u32"))));
    let body = f.body.as_ref().unwrap();
    let crate::Stmt::Let { value: Some(crate::Expr::Cast(_, ty, _)), .. } = &body.stmts[0] else { panic!("a cast") };
    assert_eq!(ty, "u8");
    let crate::Stmt::Let { value: Some(crate::Expr::Cast(_, ty, _)), .. } = &body.stmts[1] else { panic!("a cast") };
    assert_eq!(ty, "u32");
    // The same with a lifetime before the path, in a parameter, in a type's arguments and in the result.
    let src = "pub unsafe extern \"sysv64\" fn k(x: &'static ::core::primitive::u8, y: Foo<'a, ::std::primitive::str>) \
               -> &'a ::std::primitive::str {}";
    let module = crate::module(&lexed_with_lifetimes(src), 0).unwrap();
    let crate::ItemKind::Fn(f) = &module.items[0].kind else { panic!("a function") };
    let types: Vec<String> = f.params.iter().map(|p| shown(&p.ty)).collect();
    assert_eq!(types, ["&'static u8", "Foo<'a, str>"]);
    assert_eq!(f.ret.as_ref().map(|r| r.ty.as_str()), Some("&'a str"));
    // No `'` comes before the first parameter's type, so the lexer's offset is the source's.
    assert_eq!(f.params[0].ty.span(), Span(src.find("&'static").unwrap() as u32));
    // A path that is no name is the parser's error: it wants a name, not a `::` with none after it.
    for (bad, said) in
        [("k<const N: ::>()", "expected a name, found `:`"), ("k<const N: 5>()", "expected a name, found `5`")]
    {
        let src = format!("pub unsafe extern \"sysv64\" fn {bad} {{}}");
        let err = crate::module(&crate::lex(&src).unwrap(), 0).unwrap_err();
        assert_eq!(err.message, said, "{src}");
    }
}

/// A path in a value that begins with a primitive's path, `core::primitive::u32::MAX`, is read as the one that begins
/// with the primitive's name, `u32::MAX`, by the rule that reads a type: in the four ways, with `r#` on a name or not,
/// alone and in a call, `core::primitive::f32::from_bits(..)`. It is at the place where the path begins. Any other
/// path stays as written: one to a type that `core::primitive` has not, one that begins elsewhere, and a module's,
/// `core::u32::MAX`. A path from the root that is no primitive's is no value: the parser wants an expression, not a
/// `::`.
#[test]
fn a_value_path_that_begins_with_a_primitives_path_is_read_from_its_name() {
    let parsed = |src: &str| {
        let module = crate::module(&crate::lex(&format!("const C: u64 = {src};")).unwrap(), 0);
        module.map(|m| match &m.items[0].kind {
            crate::ItemKind::Const(c) => c.value.clone(),
            _ => panic!("a const"),
        })
    };
    let at = Span("const C: u64 = ".len() as u32);
    let primitives = [
        "bool", "char", "f32", "f64", "i8", "i16", "i32", "i64", "i128", "isize", "str", "u8", "u16", "u32", "u64",
        "u128", "usize",
    ];
    for name in primitives {
        for root in ["core", "::core", "std", "::std", "r#core", ":: std"] {
            for (path, rest) in [
                (format!("{root}::primitive::{name}::MAX"), vec!["MAX"]),
                (format!("{root}::r#primitive::r#{name}::r#MAX"), vec!["MAX"]),
                (format!("{root}::primitive::{name}::NOT::HERE"), vec!["NOT", "HERE"]),
                (format!("{root}::primitive::{name}"), vec![]),
            ] {
                let Ok(crate::Expr::Path(names, span)) = parsed(&path) else { panic!("a path: {path}") };
                assert_eq!(names, [vec![name], rest].concat(), "{path}");
                assert_eq!(span, at, "{path}");
            }
        }
    }
    // A call, and what is called: `f32::from_bits`.
    for root in ["core", "::core", "std", "::std"] {
        let src = format!("{root}::primitive::f32::from_bits(1)");
        let Ok(crate::Expr::Call { func, args, span, .. }) = parsed(&src) else { panic!("a call: {src}") };
        let crate::Expr::Path(names, _) = *func else { panic!("a path: {src}") };
        assert_eq!((names, args.len(), span), (vec!["f32".to_string(), "from_bits".to_string()], 1, at), "{src}");
    }
    // A name written raw is the name Rust reads, whichever name of the path it is.
    for src in ["::core::primitive::r#u64::MAX", "r#core::primitive::u64::MAX", "::r#std::r#primitive::r#u64::r#MAX"] {
        let Ok(crate::Expr::Path(names, span)) = parsed(src) else { panic!("a path: {src}") };
        assert_eq!((names, span), (vec!["u64".to_string(), "MAX".to_string()], at), "{src}");
    }
    // A path goes on as a path does after a name: a cast, a method.
    let Ok(crate::Expr::Cast(inner, ty, _)) = parsed("::core::primitive::u8::MAX as ::std::primitive::u64") else {
        panic!("a cast")
    };
    let crate::Expr::Path(names, _) = *inner else { panic!("a path") };
    assert_eq!((names, ty.as_str()), (vec!["u8".to_string(), "MAX".to_string()], "u64"));
    assert!(matches!(parsed("core::primitive::u32::MAX.count_ones()"), Ok(crate::Expr::Method { .. })));
    // Any other path is as it was written: one to a type `core::primitive` has not, one that begins elsewhere, a
    // module's.
    for path in [
        "core::primitive::f16::MAX",
        "std::primitive::bf16::MAX",
        "core::primitive::s32::MAX",
        "core::primitive",
        "core::primitive::MAX",
        "core::u32::MAX",
        "std::f32::consts::PI",
        "my::core::primitive::u32::MAX",
        "crate::core::primitive::u32::MAX",
        "primitive::u32::MAX",
        "u32::MAX",
    ] {
        let Ok(crate::Expr::Path(names, span)) = parsed(path) else { panic!("a path: {path}") };
        assert_eq!(names.join("::"), path, "{path}");
        assert_eq!(span, at, "{path}");
    }
    // A path from the root that is no primitive's is no value: the parser wants an expression, not a `::`.
    for path in ["::my::u32::MAX", "::core::u32::MAX", "::core::primitive::f16::MAX", "::"] {
        let err = parsed(path).unwrap_err();
        assert_eq!((err.message.as_str(), err.span), ("expected an expression, found `:`", at), "{path}");
    }
}

/// The `!` of never follows the arrow, a keyword or a lifetime, and is no macro's: the braces after it are the
/// body. A lifetime is a `'` and a name, and that name is no macro's, so the result of `&'static !` is the four tokens
/// before the braces. A macro after a lifetime, `&'static byte!{ _ }`, takes its braces, as it does anywhere in a
/// result. The tokens are made as the macro is given them, for the lexer of a string refuses a lifetime.
#[test]
fn the_braces_after_a_never_that_follows_a_lifetime_are_the_body() {
    for (result, tokens) in [("&'static !", 4), ("&'a !", 4), ("&'a mut !", 5), ("&'static byte!{ _ }", 6)] {
        let src = format!("pub unsafe extern \"sysv64\" fn k() -> {result} {{ ud2(); }}");
        let module = crate::module(&lexed_with_lifetimes(&src), 0).expect(result);
        let crate::ItemKind::Fn(f) = &module.items[0].kind else { panic!("a function") };
        assert_eq!(f.ret.as_ref().map(|r| r.tokens.len()), Some(tokens), "{result}");
        assert_eq!(f.body.as_ref().expect(result).stmts.len(), 1, "{result}");
    }
}

/// A pointer is one word, unless it is to a slice, a `str` or a `dyn`: those are an address and a
/// length or a table. Parentheses around a type do not hide it, and a tuple that ends in one has no
/// size either. A pointer to what the lexer refuses, a lifetime, which the macro hands on, is
/// refused in the lexer's words, not taken for one word.
#[test]
fn a_pointer_has_one_word_unless_it_is_to_what_has_no_size() {
    for ty in [
        "*const f32",
        "*mut *const f32",
        "*const [usize; 3]",
        "*const [[u8; 4]; 2]",
        "*const [*const u8; 2]",
        "*const *const [u8]",
        "*const (u32, u32)",
        "*const ()",
        "*const (u32,)",
        "*const ((u32, [f32; 2]), u8)",
        "*const (Pair<u8, 2>)",
        "*const (Foo<A, dyn B>)",
        "*const (Foo<A, [f32]>, u32)",
        "*const fn(u32)",
        "*const fn(u32) -> u32",
        "*const Pair<fn() -> u8, 2>",
        "*const (u8, fn() -> u8)",
        "*const (u8, Foo<fn() -> u8, dyn A>)",
        "*mut Foo<u8>",
    ] {
        assert_eq!(crate::pointer(ty), Ok(true), "{ty}");
    }
    for ty in ["f32", "usize", "[usize; 3]", "(u32, u32)", "&mut [f32]", "&f32", "Foo<*const u8>"] {
        assert_eq!(crate::pointer(ty), Ok(false), "{ty}");
    }
    for to in [
        "[f32]",
        "[[u8; 4]]",
        "[*const u8]",
        "str",
        "dyn Foo",
        "dyn Fn(u32)",
        "(dyn A + Send)",
        "([f32])",
        "(str)",
        "((dyn Foo))",
        "(dyn Foo<A, B>)",
        "(u32, [f32])",
        "(u32, dyn Foo<A, B>)",
        "(dyn Foo,)",
        "([f32],)",
        "dyn Fn(u32) -> u32",
        "(fn() -> u8, dyn Foo)",
        "(Foo<fn() -> u8, 2>, [f32])",
    ] {
        let why = format!(
            "a pointer to `{to}` has two words, an address and a length or a table, and a kernel's pointer has one"
        );
        for kind in ["const", "mut"] {
            assert_eq!(crate::pointer(&format!("*{kind} {to}")), Err(why.clone()));
        }
    }
    for to in ["[&'static str]", "(dyn Debug + 'static)", "&'a u8", "Foo<'a>"] {
        for kind in ["const", "mut"] {
            let lifetimes = "lifetimes and labels have no place in a kernel";
            assert_eq!(crate::pointer(&format!("*{kind} {to}")), Err(lifetimes.into()), "{to}");
        }
    }
}

/// Every 16-bit float reads back as itself, and a number between two of them rounds to the
/// nearer, or at the middle to the one whose last bit is 0, as IEEE rounds: so on to the
/// largest, past which the middle rounds to infinity.
#[test]
fn sixteen_bit_floats_round_to_nearest_even() {
    for (ty, inf) in [("f16", 0x7c00), ("bf16", 0x7f80)] {
        let make = |h: u16| if ty == "f16" { Value::F16(h) } else { Value::Bf16(h) };
        let wide = |h: u16| match cast(&make(h), "f64") {
            Ok(Value::F64(x)) => x,
            other => panic!("{other:?}"),
        };
        let round = |x: f64| cast(&Value::F64(x), ty).unwrap();
        for h in 0..inf {
            let (x, next) = (wide(h), wide(h + 1));
            assert_eq!(round(x), make(h), "{ty} {h:#x}");
            assert_eq!(round(-x), make(h | 0x8000), "{ty} -{h:#x}");
            // Past the largest, the middle is where the next would be, were it not infinite.
            let next = if h + 1 == inf { x + (x - wide(h - 1)) } else { next };
            let mid = (x + next) / 2.0;
            assert_eq!(round(mid), make(if h % 2 == 0 { h } else { h + 1 }), "{ty} between {h:#x} and the next");
            assert_eq!(round(mid.next_down()), make(h), "{ty} under the middle above {h:#x}");
            assert_eq!(round(mid.next_up()), make(h + 1), "{ty} over the middle above {h:#x}");
        }
        assert_eq!(round(f64::INFINITY), make(inf));
        assert!(matches!(round(f64::NAN), Value::F16(h) | Value::Bf16(h) if h & inf == inf && h != inf));
    }
}

/// A value known now that is not of the type wanted is told by what it is. An integer where an integer of another
/// type is wanted is told by its type, `a u16` for a `u8`, as a float is by its own, `an f32`: `an integer` would
/// say no fault, for a `u8` is one too. Its article is how its name is read, `an i8` to `an isize` and `a u8` to
/// `a usize`. Where the type wanted is no integer's, what is wrong is that the value is an integer, and it is
/// `an integer`, whatever its type. A number of no type of its own takes the type wanted.
#[test]
fn an_integer_of_another_type_is_told_by_its_type() {
    let types = [
        ("i8", "an i8"),
        ("i16", "an i16"),
        ("i32", "an i32"),
        ("i64", "an i64"),
        ("isize", "an isize"),
        ("u8", "a u8"),
        ("u16", "a u16"),
        ("u32", "a u32"),
        ("u64", "a u64"),
        ("usize", "a usize"),
    ];
    for (name, said) in types {
        let int = Value::Int(1, IntTy::parse(name));
        for (wanted, _) in types {
            let want = if wanted == name { Ok(int.clone()) } else { Err(format!("this is {said}, not {wanted}")) };
            assert_eq!(typed(int.clone(), wanted), want, "{name} as {wanted}");
        }
        for wanted in ["f32", "bool", "Word", "[u8; 2]"] {
            let want = Err(format!("this is an integer, not {wanted}"));
            assert_eq!(typed(int.clone(), wanted), want, "{name} as {wanted}");
        }
        assert_eq!(typed(Value::Int(1, None), name), Ok(int), "no type as {name}");
    }
    assert_eq!(typed(Value::Int(1, None), "bool"), Err("this is an integer, not bool".into()));
    assert_eq!(typed(Value::F32(1.5), "u8"), Err("this is an f32, not u8".into()));
    assert_eq!(typed(Value::Bool(true), "u8"), Err("this is a bool, not u8".into()));
    assert_eq!(typed(Value::Bf16(0), "f16"), Err("this is a bf16, not f16".into()));
}

/// A type's name is read with `an` before `f`, `i` and `s`, which are read `ef`, `eye` and `es`, and with `a` before
/// any other letter: `an f32`, `an i8` and `an s16`, but `a u64`, `a b32` and `a pred`.
#[test]
fn a_type_name_is_read_with_its_article() {
    for name in ["f16", "f16x2", "f32", "f64", "i8", "i64", "isize", "s8", "s64"] {
        assert_eq!(article(name), "an", "{name}");
    }
    for name in ["u8", "u64", "usize", "b8", "b64", "bf16", "tf32", "pred"] {
        assert_eq!(article(name), "a", "{name}");
    }
}

#[test]
fn a_function_says_whether_it_is_unsafe() {
    let src = r#"
        pub unsafe extern "sysv64" fn k() {}
        extern "ptx-func" fn f() {}
        unsafe fn h() {}
        pub(crate) unsafe extern "C" fn c() {}
    "#;
    let module = crate::parse(src).unwrap();
    let said: Vec<(&str, bool)> = module
        .items
        .iter()
        .filter_map(|item| match &item.kind {
            crate::ItemKind::Fn(f) => Some((f.name.as_str(), f.unsafe_fn)),
            _ => None,
        })
        .collect();
    assert_eq!(said, [("k", true), ("f", false), ("h", true), ("c", true)]);
}

/// A place is in the item that starts last before it, whatever it is nested in: the brackets of a
/// static's type, the body of a helper.
#[test]
fn a_place_is_in_the_item_that_writes_it() {
    let src = "const A: u64 = 1;\nstatic S: global<[u32; A]>;\nfn help(x: gpr) { add(x, A); }\n\
               unsafe extern \"sysv64\" fn k(x: u64) { help(x); }";
    let module = parse(src).unwrap();
    let item_of = |at: &str| module.item_at(Span(src.rfind(at).unwrap() as u32));
    assert_eq!([item_of("A: u64"), item_of("A]"), item_of("A); }"), item_of("help(x); }")], [0, 1, 2, 3]);
}

/// Mentions come in the order of the source, however they were recorded, and a place has one.
#[test]
fn a_function_has_its_mentions_in_the_order_of_the_source_each_place_once() {
    let mut mentions = vec![];
    for (name, at) in [("B", 20), ("A", 5), ("B", 20), ("C", 12), ("A", 5)] {
        Mention::record(&mut mentions, name, Span(at));
    }
    let told: Vec<_> = mentions.iter().map(|m| (m.name.as_str(), m.span.0)).collect();
    assert_eq!(told, [("A", 5), ("C", 12), ("B", 20)]);
}

/// A module kernite cannot read has the items it reads, named in order, and `_` for each that is none of a const, a
/// function or a static: every item but those it cannot read, the head of a function with a generic type say, each of
/// which ends at its `;` if it is a `const` or a `static`, whose value may hold a brace group, and else at its first
/// `;` or at its first brace group outside `<..>`, a function's body; and, of items that share a name, those that are
/// not in the way: the static is in the way of the function, the function of the const, else the second. Of the
/// statics, it has none if it cannot read an item and reads no function that names an ABI. The error is the first
/// head's if there is one, else the first item in the way, as for a module it cannot read at all; the consts a module
/// is given are none of its own.
#[test]
fn a_module_kernite_cannot_read_has_the_items_it_reads() {
    let read = |src: &str, given: &str| {
        let (module, error) = crate::read(&crate::lex(src).unwrap(), crate::lex(given).unwrap().len());
        let names: Vec<_> = module.items.iter().map(|item| item.name().unwrap_or("_").to_string()).collect();
        (names, error.map(|e| e.at(src)))
    };
    let cases = [
        ("fn a() {} const b: u32 = 1; static c: global<u32>;", "", vec!["a", "b", "c"], None),
        (
            "fn a() {} fn b<T>() {} fn c() {} fn d<U>() {} fn e() {}",
            "",
            vec!["a", "c", "e"],
            Some("1:16: a function's generics are values, as in `<const N: usize>`"),
        ),
        (
            "fn a() {} fn b() {} fn a() {} fn c() {}",
            "",
            vec!["a", "b", "c"],
            Some("1:24: `a` names another function of the module as well"),
        ),
        (
            "fn k() {} static k: global<u32>; fn m() {}",
            "",
            vec!["k", "m"],
            Some("1:18: `k` names a function of the module as well"),
        ),
        (
            "fn h() {} const h: u32 = 1; fn j() {}",
            "",
            vec!["h", "j"],
            Some("1:4: `h` names a constant of the module as well"),
        ),
        (
            "fn a() {} fn a() {} fn b<T>() {}",
            "",
            vec!["a"],
            Some("1:26: a function's generics are values, as in `<const N: usize>`"),
        ),
        (
            "const A: u64 = 1; fn A() {} fn A() {}",
            "const A: u64 = 1;",
            vec!["A", "A"],
            Some("1:32: `A` names another function of the module as well"),
        ),
        (
            "static s: global<u32>; fn h() {} fn b<T>() {}",
            "",
            vec!["h"],
            Some("1:39: a function's generics are values, as in `<const N: usize>`"),
        ),
        (
            "static s: global<u32>; extern \"sysv64\" fn k() {} fn b<T>() {}",
            "",
            vec!["s", "k"],
            Some("1:55: a function's generics are values, as in `<const N: usize>`"),
        ),
        (
            "static s: global<u32>; fn b<T>() {} extern \"sysv64\" fn k() {}",
            "",
            vec!["s", "k"],
            Some("1:29: a function's generics are values, as in `<const N: usize>`"),
        ),
        (
            "const a: u64 = 1 fn b() {} const c: u64 = 3; fn d() {}",
            "",
            vec!["d"],
            Some("1:18: expected `;`, found `fn`"),
        ),
        ("const a: u64 = { 1 }; fn b() {}", "", vec!["b"], Some("1:16: a block is a statement, not a value")),
        ("const a: u64 = { 1 } + 2; fn b() {}", "", vec!["b"], Some("1:16: a block is a statement, not a value")),
        ("const a: u64 = if c { 1 } else { 2 }; fn b() {}", "", vec!["b"], Some("1:19: expected `;`, found `c`")),
        (
            "#[doc = \"x\"] pub(crate) static s: global<u32> = Pair { a: 1 }.a; fn b() {}",
            "",
            vec!["b"],
            Some("1:54: expected `;`, found `{ a: 1 }`"),
        ),
        (
            "const fn a<T>() -> u32 { 1 } fn b() {}",
            "",
            vec!["b"],
            Some("1:12: a function's generics are values, as in `<const N: usize>`"),
        ),
        (
            "fn a<T>() -> Foo<{ 3 }> {} fn b() {}",
            "",
            vec!["b"],
            Some("1:6: a function's generics are values, as in `<const N: usize>`"),
        ),
        (
            "fn a<T: Fn() -> u32>() -> Foo<{ 3 }> {} fn b() {}",
            "",
            vec!["b"],
            Some("1:6: a function's generics are values, as in `<const N: usize>`"),
        ),
        (
            "fn a<T>() -> Foo<fn() -> u32, { 3 }> {} fn b() {}",
            "",
            vec!["b"],
            Some("1:6: a function's generics are values, as in `<const N: usize>`"),
        ),
        (
            "fn a<T: Tr<{ 3 }>>() {} fn b() {}",
            "",
            vec!["b"],
            Some("1:6: a function's generics are values, as in `<const N: usize>`"),
        ),
        (
            "fn a<T>() where Foo<{ 3 }>: Sized {} fn b() {}",
            "",
            vec!["b"],
            Some("1:6: a function's generics are values, as in `<const N: usize>`"),
        ),
        ("impl Foo<{ 3 }> { fn x() {} } fn b() {}", "", vec!["_", "b"], None),
        (
            "fn a<T>() {} struct S; fn b() {}",
            "",
            vec!["_", "b"],
            Some("1:6: a function's generics are values, as in `<const N: usize>`"),
        ),
        (
            "static s: global<u32>; fn a() {} fn a() {}",
            "",
            vec!["s", "a"],
            Some("1:37: `a` names another function of the module as well"),
        ),
    ];
    for (src, given, names, error) in cases {
        let want = (names.into_iter().map(String::from).collect::<Vec<_>>(), error.map(String::from));
        assert_eq!(read(src, given), want, "{src}");
    }
}
