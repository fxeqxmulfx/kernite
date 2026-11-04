//! The lexer, how a type is read and written and which types are pointers, the rounding of 16-bit
//! floats, whether a function is `unsafe`, and where a place is. The rest of the language is tested
//! where it meets the targets it is compiled for, in `kernite`'s `tests/lang.rs`.

use crate::token::{Span, Tree};
use crate::value::{Value, cast};
use crate::{Mention, parse};

#[test]
fn the_lexer_reads_what_rust_writes() {
    let src = "a.b(0..n, 1.5e-3, 0x1F_u8, r#type, \"s\\\"q\", 1..=2) /* c /* nested */ */ // end";
    let trees = crate::lex(src).unwrap();
    assert_eq!(crate::text(&trees), r#"a.b(0..n, 1.5e-3, 0x1F_u8, r#type, "s\"q", 1..=2)"#);
}

/// The tokens of `src`, a source with lifetimes in it, as `for<'a> fn(&'a u8)` has. The lexer of a
/// string refuses a lifetime, but the macro is given one as `proc_macro` hands it over: a `'` joint
/// to the name that follows it. So each `'` of `src` is lexed as the start of a word, `lt_a`, which
/// is then made into the two tokens.
fn lexed_with_lifetimes(src: &str) -> Vec<Tree> {
    fn made(trees: Vec<Tree>) -> Vec<Tree> {
        trees
            .into_iter()
            .flat_map(|tree| match tree {
                Tree::Ident { name, span } if name.starts_with("lt_") => vec![
                    Tree::Punct { ch: '\'', joint: true, span },
                    Tree::Ident { name: name["lt_".len()..].to_string(), span },
                ],
                Tree::Group { delim, trees, span } => vec![Tree::Group { delim, trees: made(trees), span }],
                tree => vec![tree],
            })
            .collect()
    }
    made(crate::lex(&src.replace('\'', "lt_")).unwrap())
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

/// A type that begins with a binder, `for<'a> fn(&'a u8)`, is read whole, as any other type is: the
/// `<` after `for` opens the binder, not the arguments of a type named `for`. So it is one type as a
/// parameter, which starts at its `for`, as the argument of another type, `Option<for<'a> fn(&'a u8)>`,
/// and as a result.
#[test]
fn a_type_that_begins_with_a_binder_is_read_whole() {
    // A type as the parser holds it, a name or a name with arguments, written out.
    fn shown(ty: &crate::Type) -> String {
        match ty {
            crate::Type::Name(name, _) => name.clone(),
            crate::Type::Generic(name, args, _) => {
                format!("{name}<{}>", args.iter().map(shown).collect::<Vec<_>>().join(", "))
            }
            other => panic!("{other:?}"),
        }
    }
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
