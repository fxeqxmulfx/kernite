//! The lexer, the rounding of 16-bit floats, and whether a function is `unsafe`. The rest of
//! the language is tested where it meets the targets it is compiled for, in `kernite`'s
//! `tests/lang.rs`.

use crate::value::{Value, cast};

#[test]
fn the_lexer_reads_what_rust_writes() {
    let src = "a.b(0..n, 1.5e-3, 0x1F_u8, r#type, \"s\\\"q\", 1..=2) /* c /* nested */ */ // end";
    let trees = crate::lex(src).unwrap();
    assert_eq!(crate::text(&trees), r#"a.b(0..n,1.5e-3,0x1F_u8,r#type,"s\"q",1..=2)"#);
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
