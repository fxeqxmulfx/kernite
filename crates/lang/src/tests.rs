//! The lexer. The rest of the language is tested where it meets the targets it is compiled
//! for, in `kgen`'s `tests/lang.rs`.

#[test]
fn the_lexer_reads_what_rust_writes() {
    let src = "a.b(0..n, 1.5e-3, 0x1F_u8, r#type, \"s\\\"q\", 1..=2) /* c /* nested */ */ // end";
    let trees = crate::lex(src).unwrap();
    assert_eq!(crate::text(&trees), r#"a.b(0..n,1.5e-3,0x1F_u8,r#type,"s\"q",1..=2)"#);
}
