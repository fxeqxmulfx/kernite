//! The language, run by the targets it is compiled for: helpers inlined, loops unrolled,
//! labels numbered afresh, values computed as Rust computes them, memory in brackets,
//! predicates guarding, and errors where the trouble is.

/// Kernel `name` of `src`, for x86: its listing, as a snapshot holds it.
fn x86(src: &str, name: &str) -> String {
    kernite::x86_str(src, name).unwrap_or_else(|e| panic!("{e}")).text()
}

/// Kernel `name` of `src`, for x86: its instructions alone, without the listing's comments.
fn code(src: &str, name: &str) -> Vec<String> {
    kernite::x86_str(src, name).unwrap_or_else(|e| panic!("{e}")).code().map(String::from).collect()
}

/// The `.entry` of the one kernel in `src`, in PTX, from its name on.
fn ptx(src: &str) -> String {
    let module = kernite::ptx_str(src).unwrap_or_else(|e| panic!("{e}"));
    module.split(".visible .entry ").nth(1).expect("an entry").to_string()
}

/// Kernel `k`, whose body is `body` from its second line on.
fn kernel(body: &str) -> String {
    format!("unsafe extern \"sysv64\" fn k() {{\n{body}\n}}\n")
}

/// Why kernel `k`, whose body is `body` from its second line on, does not compile.
fn refused(body: &str) -> String {
    kernite::x86_str(&kernel(body), "k").unwrap_err()
}

/// What `src` compiles to, or why it does not: for x86, the listing of kernel `name`; for a GPU, which `src` says by
/// `ptx-kernel`, the PTX of all of it.
fn compile(src: &str, name: &str) -> Result<String, String> {
    match src.contains("ptx-kernel") {
        true => kernite::ptx_str(src),
        false => kernite::x86_str(src, name).map(|listing| listing.text()),
    }
}

#[test]
fn a_kernel_runs_its_helpers_in_place() {
    let src = r#"
        /// The fifth of the eight floats at `xs`, the first of their high half.
        pub unsafe extern "sysv64" fn fifth(xs: *const f32) -> f32 {
            let v: ymm;
            vmovups(v, [xs]);
            ret_high(v);
        }

        fn ret_high(v: ymm) {
            note("the high lanes");
            let high: ymm;
            vextractf128(high.xmm, v, 1);
            ret(high.xmm);
        }
    "#;
    let want = "# fifth(gpr) -> vec
# needs: avx
# frame: 0 bytes aligned to 0, saves []
# xs=rdi v=ymm0 high=ymm0
    vmovups ymm0, ymmword ptr [rdi]
    # the high lanes
    vextractf128 xmm0, ymm0, 1
    vzeroupper
    ret
";
    assert_eq!(x86(src, "fifth"), want);
}

/// A note is one line, on x86 and on a GPU: each line break in it, `\r\n`, `\n` or `\r`, is a space, so that no text of
/// it is code to an assembler, as `int3` would be on a line of its own. Each is written by an escape, and `\n` and
/// `\r\n` in the string itself too, plain or raw: Rust refuses a lone `\r` there.
#[test]
fn a_line_break_in_a_note_is_a_space_on_every_machine() {
    for (string, shown) in [
        (r#""a\nint3""#, "a int3"),
        (r#""a\r\nint3""#, "a int3"),
        (r#""a\rint3""#, "a int3"),
        ("\"a\nint3\"", "a int3"),
        ("\"a\r\nint3\"", "a int3"),
        ("r\"a\nint3\"", "a int3"),
        ("r#\"a\r\nint3\"#", "a int3"),
        // Each line break is a space of its own, one at either end too, and `\r\n` is one line break and not two.
        (r#""\na\n\nint3\r\n""#, " a  int3 "),
        (r#""\r\n\r\n\r\r\n\n""#, "     "),
        (r#""\ra\r\r\n\nint3\n""#, " a   int3 "),
    ] {
        let note = format!("note({string});");
        let listing = format!("# k() -> ()\n# frame: 0 bytes aligned to 0, saves []\n    # {shown}\n    ret\n");
        assert_eq!(x86(&kernel(&note), "k"), listing, "{string:?}");
        assert_eq!(code(&kernel(&note), "k"), ["ret"], "{string:?}");
        let gpu = format!("unsafe extern \"ptx-kernel\" fn k() {{\n{note}\n}}\n");
        assert_eq!(ptx(&gpu), format!("k(\n)\n{{\n\n\t// {shown}\n\tret;\n}}\n"), "{string:?}");
    }
}

/// A NUL in a note is refused at the string, on x86 and on a GPU, for `ptxas` reads a NUL as the end of the module: by
/// an escape, `\0`, `\x00` or `\u{0}`, and as the character, in a plain string and in a raw one. The error is at the
/// string, wherever the NUL is in it and whatever else the note holds. Another control character is no trouble.
#[test]
fn a_nul_in_a_note_is_refused_at_the_string_on_every_machine() {
    let message = "a NUL is not allowed in a note, on any machine: `ptxas` reads it as the end of the module";
    for string in [
        r#""a\0b""#,
        r#""a\x00b""#,
        r#""a\u{0}b""#,
        r#""a\u{0_0}b""#,
        r#""\0""#,
        "\"a\0b\"",
        "\"\0\"",
        "r\"a\0b\"",
        "r#\"a \"\0\" b\"#",
        // The note's text is read whole, so the NUL is refused after a name, and after a long text, at the string's
        // start.
        r#""{x} {{ \0""#,
        r#""the high lanes, the low lanes, the lanes of the next tile, the lanes of the last one: \0""#,
    ] {
        let note = format!("let x = 1;\nnote({string});");
        let want = format!("3:6: {message}");
        assert_eq!(refused(&note), want, "{string:?}");
        let gpu = format!("unsafe extern \"ptx-kernel\" fn k() {{\n{note}\n}}\n");
        assert_eq!(kernite::ptx_str(&gpu).unwrap_err(), want, "{string:?}");
    }
    // A note keeps any other control character as it is: U+0001, the NUL's neighbour, DEL, a tab and NEL.
    let listing = "# k() -> ()\n# frame: 0 bytes aligned to 0, saves []\n    # a\u{1}\u{7f}\tb\u{85}\n    ret\n";
    assert_eq!(x86(&kernel(r#"note("a\u{1}\x7f\tb\u{85}");"#), "k"), listing);
}

#[test]
fn a_function_that_names_an_abi_is_not_const_and_no_function_is_async() {
    let constant = "a function that names an ABI runs on its machine, never while Rust compiles, so it is not `const`";
    let asynchronous = "a function runs when it is called, so it is not `async`";
    for (src, at, message) in [
        (r#"pub const unsafe extern "sysv64" fn k() {}"#, "1:5", constant),
        (r#"pub async unsafe extern "sysv64" fn k() {}"#, "1:5", asynchronous),
        // Of two words, the first is the one refused.
        (r#"pub const async unsafe extern "sysv64" fn k() {}"#, "1:5", constant),
        (r#"pub async const unsafe extern "sysv64" fn k() {}"#, "1:5", asynchronous),
    ] {
        assert_eq!(kernite::x86_str(src, "k").unwrap_err(), format!("{at}: {message}"), "{src}");
    }
    for (helper, at) in [("async fn h() {}", "2:1"), ("const async fn h() {}", "2:7")] {
        let src = format!("pub unsafe extern \"sysv64\" fn k() {{ h(); }}\n{helper}");
        assert_eq!(kernite::x86_str(&src, "k").unwrap_err(), format!("{at}: {asynchronous}"), "{src}");
    }
    for (src, at, message) in [
        (r#"pub const unsafe extern "ptx-kernel" fn k() {}"#, "1:5", constant),
        (r#"pub async unsafe extern "ptx-kernel" fn k() {}"#, "1:5", asynchronous),
        (r#"const extern "ptx-func" fn f() {}"#, "1:1", constant),
    ] {
        assert_eq!(kernite::ptx_str(src).unwrap_err(), format!("{at}: {message}"), "{src}");
    }
    // A helper may be `const`: kernite runs it while the kernel compiles, as Rust may run a
    // `const fn`.
    let src = "pub unsafe extern \"sysv64\" fn k() { h(); }\nconst fn h() { note(\"it runs now\"); }";
    assert_eq!(x86(src, "k"), "# k() -> ()\n# frame: 0 bytes aligned to 0, saves []\n    # it runs now\n    ret\n");
}

/// A raw string is a string, as in Rust, wherever kernite takes one: a note's text by `r"hi"`, `r#"hi"#` or
/// `r##"a "# b"##` is the text of the plain string, with no escape read in it, and an ABI by `r"sysv64"` or
/// `r#"ptx-kernel"#` is the ABI. A raw string that is never closed is refused where it starts, as a plain one is. So is
/// one of more than 255 `#`, as Rust refuses it.
#[test]
fn a_raw_string_is_a_string_for_a_note_and_for_an_abi() {
    let raw = |hashes: usize, text: &str| format!("r{0}\"{text}\"{0}", "#".repeat(hashes));
    // (the `#`s of the raw string, its text, what a note shows of it), with `k` as 7.
    let notes = [
        (0, "hi", "hi"),
        (1, "hi", "hi"),
        (2, "a \"# b", "a \"# b"),
        (0, "a\\nb\\\\", "a\\nb\\\\"),
        (0, "a\\", "a\\"),
        (1, "{k} \"{k}\" {{k}}", "7 \"7\" {k}"),
        (255, "hi", "hi"),
    ];
    for (hashes, text, shown) in notes {
        let (raw, plain) = (raw(hashes, text), format!("{text:?}"));
        let on_x86 =
            |lit: &str| x86(&format!("unsafe extern \"sysv64\" fn k() {{\nlet k = 7;\nnote({lit});\n}}\n"), "k");
        let want = format!("# k() -> ()\n# frame: 0 bytes aligned to 0, saves []\n    # {shown}\n    ret\n");
        assert_eq!(on_x86(&plain), want, "{plain}");
        assert_eq!(on_x86(&raw), want, "{raw}");
        let on_gpu =
            |lit: &str| ptx(&format!("unsafe extern \"ptx-kernel\" fn k() {{\nlet k = 7;\nnote({lit});\n}}\n"));
        assert!(on_gpu(&plain).contains(&format!("\t// {shown}\n")), "{plain}");
        assert_eq!(on_gpu(&raw), on_gpu(&plain), "{raw}");
    }
    // Each ABI by a raw string is the ABI by the plain one, and an ABI that kernite does not know is told as the plain
    // one is, at the string, with no escape read in it: `sysv\64` is no ABI.
    let abis = ["sysv64", "ptx-kernel", "ptx-func", "sysv\\64"];
    for (abi, hashes) in abis.into_iter().flat_map(|abi| (0..3).map(move |hashes| (abi, hashes))) {
        let with = |lit: &str| match abi.starts_with("ptx-") {
            true => kernite::ptx_str(&format!("pub unsafe extern {lit} fn f() {{}}\n")),
            false => compile(&format!("pub unsafe extern {lit} fn f() {{}}\n"), "f"),
        };
        let plain = with(&format!("{abi:?}"));
        assert_eq!(plain.is_ok(), abi != "sysv\\64", "{abi}: {plain:?}");
        assert_eq!(with(&raw(hashes, abi)), plain, "{abi}, {hashes} `#`s");
    }
    // A raw string that is never closed is refused where it starts, as a plain one is.
    let never = "2:6: this string is never closed";
    for lit in ["\"hi);", "r\"hi);", "r#\"hi\");", "r##\"hi\"#);"] {
        assert_eq!(refused(&format!("note({lit}")), never, "{lit}");
    }
    let gpu = "pub unsafe extern \"ptx-kernel\" fn k() {\nnote(r#\"hi\");\n}\n";
    assert_eq!(kernite::ptx_str(gpu).unwrap_err(), never);
    // So is a raw string of more than 255 `#`, as Rust refuses it.
    assert_eq!(refused(&format!("note({});", raw(256, "hi"))), "2:6: a raw string has 255 `#` at most, as in Rust");
}

/// A CR that no LF follows is refused at the CR, as Rust refuses it, in a string, plain or raw, a note's or an ABI's,
/// and in a doc comment, `///`, `//!`, `/**` and `/*!`, on x86 and on a GPU. A comment that is no doc comment takes it,
/// as Rust does, and the escape `\r` is one.
#[test]
fn a_cr_on_its_own_is_refused_in_a_string_and_a_doc_comment_and_taken_in_any_other_comment() {
    let string = "a CR on its own is not allowed in a string, as in Rust: write `\\r`";
    let raw = "a CR on its own is not allowed in a raw string, as in Rust";
    let doc = "a CR on its own is not allowed in a doc comment, as in Rust";
    // The kernel `k` of each machine, with `top` before it and `body` in it.
    let kernels = |top: &str, body: &str| {
        ["sysv64", "ptx-kernel"].map(|abi| format!("{top}unsafe extern \"{abi}\" fn k() {{\n{body}\n}}\n"))
    };
    for (top, body, at, message) in [
        ("", "note(\"a\rb\");", "2:8", string),
        ("", "note(\"a\r\");", "2:8", string),
        ("", "note(\"a\nb\rc\");", "3:2", string),
        ("", "note(\"a\r\r\n\");", "2:8", string),
        ("", "note(r\"a\rb\");", "2:9", raw),
        ("", "note(r#\"a\r\"#);", "2:10", raw),
        ("", "note(r\"a\r\nb\rc\");", "3:2", raw),
        ("/// a\rb\n", "", "1:6", doc),
        ("//! a\rb\n", "", "1:6", doc),
        ("/** a\rb */\n", "", "1:6", doc),
        ("/*! a\rb */\n", "", "1:6", doc),
        ("/// a\r\n/// b\rc\n", "", "2:6", doc),
        ("", "/// a\rb\nnote(\"c\");", "2:6", doc),
    ] {
        for src in kernels(top, body) {
            assert_eq!(compile(&src, "k"), Err(format!("{at}: {message}")), "{src:?}");
        }
    }
    // An ABI is a string too, refused as a note's is, before it is looked up.
    for (src, at, message) in [
        ("unsafe extern \"sysv64\r\" fn k() {}\n", "1:22", string),
        ("unsafe extern r\"sysv64\r\" fn k() {}\n", "1:23", raw),
        ("unsafe extern \"ptx-kernel\r\" fn k() {}\n", "1:26", string),
        ("unsafe extern r#\"ptx-kernel\r\"# fn k() {}\n", "1:28", raw),
    ] {
        assert_eq!(compile(src, "k"), Err(format!("{at}: {message}")), "{src:?}");
    }
    // Any other comment takes a CR, and so does the escape `\r`, which a note shows as a space.
    let compiled = |top: &str, body: &str| kernels(top, body).map(|src| compile(&src, "k"));
    let want = compiled("", "note(\"a\");");
    assert!(want.iter().all(Result::is_ok), "{want:?}");
    for (top, body) in [
        ("// a\rb\n", "note(\"a\");"),
        ("//// a\rb\n", "note(\"a\");"),
        ("/* a\rb */\n", "note(\"a\");"),
        ("/*** a\rb ***/\n", "note(\"a\");"),
        ("", "note(\"a\"); // b\rc"),
        ("", "/* b\rc */ note(\"a\");"),
    ] {
        assert_eq!(compiled(top, body), want, "{top:?} {body:?}");
    }
    assert_eq!(compiled("", "note(\"a\\r\");"), compiled("", "note(\"a \");"));
}

/// A CRLF in a string, plain or raw, is a LF, as Rust reads it. A note shows each line break as a space, so an ABI is
/// the string that tells it: `extern "a<CRLF>b"` is none that kernite knows, and the error has it as `a<LF>b`, on x86
/// and on a GPU. A source whose lines end in CRLF is the source whose lines end in LF, with doc comments and strings of
/// more than a line in it.
#[test]
fn a_crlf_in_a_string_is_a_lf_and_a_source_of_crlf_lines_is_the_source_of_lf_lines() {
    for (crlf, lf, text) in [
        ("\"a\r\nb\"", "\"a\nb\"", "a\nb"),
        ("r\"a\r\nb\"", "r\"a\nb\"", "a\nb"),
        ("r#\"a\r\n\r\nb\"#", "r#\"a\n\nb\"#", "a\n\nb"),
    ] {
        for top in ["", "unsafe extern \"ptx-kernel\" fn g() {}\n"] {
            let with = |lit: &str| compile(&format!("{top}unsafe extern {lit} fn k() {{}}\n"), "k");
            assert_eq!(with(crlf), with(lf), "{top:?} {crlf:?}");
            let message = with(crlf).unwrap_err();
            assert!(message.ends_with(&format!("`extern \"{text}\"` is none of these")), "{message:?}");
        }
    }
    let (x86, gpu) = ("    # a b\n    # c d\n    ret\n", "\t// a b\n\t// c d\n\tret;\n");
    for (abi, notes) in [("sysv64", x86), ("ptx-kernel", gpu)] {
        let lf = [
            "/// A kernel.",
            "/** Its doc,",
            " over two lines. */",
            &format!("unsafe extern \"{abi}\" fn k() {{"),
            "    // A comment.",
            "    note(\"a",
            "b\");",
            "    note(r#\"c",
            "d\"#);",
            "}",
            "",
        ]
        .join("\n");
        let listing = compile(&lf, "k").unwrap();
        assert!(listing.contains(notes), "{abi}: {listing:?}");
        assert_eq!(compile(&lf.replace('\n', "\r\n"), "k"), Ok(listing), "{abi}");
    }
}

/// A plain string takes the escapes that Rust reads, wherever kernite takes one. A note's text, `note("\x41")`,
/// `note("caf\u{e9}")` and `note("a\` LF `    b")`, shows `A`, `café` and `ab`, on x86 and on a GPU, and an escape is
/// read before the `{k}` of a note is, as Rust reads a format string. An ABI, `extern "sysv\x364"`, is `sysv64`, and
/// one that kernite does not know is told as the string reads.
#[test]
fn a_string_takes_the_escapes_that_rust_reads_for_a_note_and_for_an_abi() {
    // (the text of a plain string, what a note shows of it), with `k` as 7.
    let notes = [
        (r"\x41", "A"),
        (r"\x41\x7e\x7E\x20\x2a", "A~~ *"),
        (r"caf\u{e9}", "café"),
        (r"\u{E9}\u{e_9}\u{e9_}\u{0e9}\u{000e9}\u{0_0_0_e_9}", "éééééé"),
        (r"\u{1f600}\u{1F_600}\u{10ffff}", "😀😀\u{10ffff}"),
        ("a\\\n    b", "ab"),
        ("a\\\r\n\t b", "ab"),
        ("a\\\n\n \t\r\n  b", "ab"),
        ("a\\\n\r\tb", "ab"),
        ("a\\\n    ", "a"),
        ("a\\\n\u{a0}b", "a\u{a0}b"),
        (r#"\\ \" \'"#, r#"\ " '"#),
        (r"\x7bk\x7d {k} \u{7b}k}", "7 7 7"),
        (r"\u{7b}\u{7b}k}}", "{k}"),
        (r"\n\r\t", "  \t"),
    ];
    for (text, shown) in notes {
        let lit = format!("\"{text}\"");
        let on_x86 =
            |lit: &str| x86(&format!("unsafe extern \"sysv64\" fn k() {{\nlet k = 7;\nnote({lit});\n}}\n"), "k");
        let want = format!("# k() -> ()\n# frame: 0 bytes aligned to 0, saves []\n    # {shown}\n    ret\n");
        assert_eq!(on_x86(&lit), want, "{text:?}");
        let on_gpu = ptx(&format!("unsafe extern \"ptx-kernel\" fn k() {{\nlet k = 7;\nnote({lit});\n}}\n"));
        assert!(on_gpu.contains(&format!("\t// {shown}\n")), "{text:?}: {on_gpu}");
    }
    // An ABI is read as a note's text is: each of these is the ABI it spells, and one told as the string says is one
    // that kernite does not know. A raw string has no escapes, so `r"ptx-\x6bernel"` is the text it shows.
    let with = |lit: &str| format!("pub unsafe extern {lit} fn f() {{}}\n");
    for (written, abi) in [("sysv\\x364", "sysv64"), ("sysv\\u{36}4", "sysv64"), ("\\x73ysv64", "sysv64")] {
        let escaped = compile(&with(&format!("\"{written}\"")), "f");
        assert!(escaped.is_ok(), "{written}");
        assert_eq!(escaped, compile(&with(&format!("{abi:?}")), "f"), "{written}");
    }
    for (written, abi) in
        [("ptx-\\x6bernel", "ptx-kernel"), ("ptx-\\u{66}unc", "ptx-func"), ("\\x70tx-func", "ptx-func")]
    {
        let escaped = kernite::ptx_str(&with(&format!("\"{written}\"")));
        assert!(escaped.is_ok(), "{written}");
        assert_eq!(escaped, kernite::ptx_str(&with(&format!("{abi:?}"))), "{written}");
    }
    let unknown = compile(&with("\"a\\x41\\u{e9}\\\n   b\""), "f").unwrap_err();
    assert!(unknown.ends_with("`extern \"aAéb\"` is none of these"), "{unknown}");
    let raw = kernite::ptx_str(&with(r#"r"ptx-\x6bernel""#)).unwrap_err();
    assert!(raw.ends_with(r#"`extern "ptx-\x6bernel"` is none of these"#), "{raw}");
}

/// An escape that Rust refuses is refused at its `\`, with `line:column`, in a note's text and in an ABI, on x86 and on
/// a GPU, and the words say which of three it is: an escape that is none, `\x` that has not two hex digits, 7f at most,
/// or `\u` that has not one to six in braces that make a character.
#[test]
fn a_string_refuses_the_escapes_that_rust_refuses_at_the_escape() {
    let none = |c: &str| {
        format!(
            "`\\{c}` is no escape, as in Rust: a string has `\\n`, `\\r`, `\\t`, `\\0`, `\\\\`, `\\'`, `\\\"`, \
             `\\x41`, `\\u{{e9}}` and a `\\` that ends a line"
        )
    };
    let x = "`\\x` takes two hex digits, 7f at most, as in Rust: write a greater character as `\\u{80}`";
    let u = "`\\u` takes one to six hex digits in braces, as `\\u{e9}`, of a character: not d800 to dfff, and \
             10ffff at most, as in Rust";
    // The kernel `k` of each machine, with `body` in it, from its second line on.
    let kernels =
        |body: &str| ["sysv64", "ptx-kernel"].map(|abi| format!("unsafe extern \"{abi}\" fn k() {{\n{body}\n}}\n"));
    for (lit, at, message) in [
        (r#""\x80""#, "2:7", x.to_string()),
        (r#""\u{d800}""#, "2:7", u.to_string()),
        (r#""\u{110000}""#, "2:7", u.to_string()),
        (r#""\u{}""#, "2:7", u.to_string()),
        (r#""\q""#, "2:7", none("q")),
        (r#""\xff""#, "2:7", x.to_string()),
        (r#""\x4""#, "2:7", x.to_string()),
        (r#""\xg0""#, "2:7", x.to_string()),
        (r#""\u00e9""#, "2:7", u.to_string()),
        (r#""\u{e9""#, "2:7", u.to_string()),
        (r#""\u{_e9}""#, "2:7", u.to_string()),
        (r#""\u{1234567}""#, "2:7", u.to_string()),
        (r#""\u{dfff}""#, "2:7", u.to_string()),
        (r#""\{k}""#, "2:7", none("{")),
        (r#""\é""#, "2:7", none("é")),
        (r#""\ ""#, "2:7", none(" ")),
        // The first is the one told, and a column counts characters, `é` as one. A string later in the line is read as
        // the first is.
        (r#""é\x41\q\x80""#, "2:12", none("q")),
        (r#""a"); note("b\u{110000}""#, "2:19", u.to_string()),
    ] {
        for src in kernels(&format!("note({lit});")) {
            assert_eq!(compile(&src, "k"), Err(format!("{at}: {message}")), "{src:?}");
        }
    }
    // On a later line, after a CRLF, and in a helper.
    for src in kernels("note(\"a\r\nb\\\n  c\r\n\\x80\");") {
        assert_eq!(compile(&src, "k"), Err(format!("5:1: {x}")), "{src:?}");
    }
    let src = "unsafe extern \"sysv64\" fn k() { h(); }\nfn h() { note(\"\\q\"); }\n";
    assert_eq!(compile(src, "k"), Err(format!("2:16: {}", none("q"))));
    // An ABI is a string too, refused before it is looked up.
    for (src, at, message) in [
        ("unsafe extern \"sysv\\x364\\q\" fn k() {}\n", "1:25", none("q")),
        ("unsafe extern \"sysv\\x80\" fn k() {}\n", "1:20", x.to_string()),
        ("unsafe extern \"ptx-kernel\\u{d800}\" fn k() {}\n", "1:26", u.to_string()),
    ] {
        assert_eq!(compile(src, "k"), Err(format!("{at}: {message}")), "{src:?}");
    }
    // A raw string has no escapes to refuse.
    for src in kernels(r#"note(r"\q\x80\ud800");"#) {
        assert!(compile(&src, "k").unwrap().contains(r"\q\x80\ud800"), "{src:?}");
    }
}

/// The spaces of Rust's lexer, Pattern_White_Space: a tab, the line breaks, U+000B and U+000C, U+0085, a space, the
/// marks that say which way text runs, U+200E and U+200F, and U+2028 and U+2029.
const SPACES: [char; 11] =
    ['\t', '\n', '\u{b}', '\u{c}', '\r', ' ', '\u{85}', '\u{200e}', '\u{200f}', '\u{2028}', '\u{2029}'];

/// The characters that Unicode calls a space and Rust does not, each of which `char::is_whitespace` takes.
const LOOKALIKES: [char; 16] = [
    '\u{a0}', '\u{1680}', '\u{2000}', '\u{2001}', '\u{2002}', '\u{2003}', '\u{2004}', '\u{2005}', '\u{2006}',
    '\u{2007}', '\u{2008}', '\u{2009}', '\u{200a}', '\u{202f}', '\u{205f}', '\u{3000}',
];

/// A space is what Rust reads as one, Pattern_White_Space, on x86 and on a GPU: a kernel with any of the eleven between
/// its tokens, before its first and after its last, and inside its brackets, compiles to the listing of the kernel with
/// a plain space there. U+200E and U+200F are among them, which `char::is_whitespace` does not take.
#[test]
fn a_space_is_what_rust_reads_as_one_on_every_machine() {
    // The kernel with a `$` for each space.
    let kernel = |abi: &str, space: char| {
        let spaced = r#"$unsafe$extern$"ABI"$fn$k$($)${$let$a$=$1$+$2$;$note$($"a {a}"$)$;$}$"#;
        spaced.replace("ABI", abi).replace('$', &space.to_string())
    };
    for abi in ["sysv64", "ptx-kernel"] {
        let plain = compile(&kernel(abi, ' '), "k");
        assert!(plain.as_ref().is_ok_and(|listing| listing.contains("a 3")), "{abi}: {plain:?}");
        for space in SPACES {
            assert_eq!(compile(&kernel(abi, space), "k"), plain, "{abi}: {space:?}");
        }
    }
}

/// No other character is a space. Each other that Unicode calls one is refused at the character on x86 and on a GPU, in
/// words that show which it is, as Rust's do: "unexpected `\u{a0}`". So is each other that no token starts with: shown
/// so if it is not printable ASCII, and as it is if it is.
#[test]
fn a_character_that_unicode_calls_a_space_and_rust_does_not_is_refused_at_the_character_on_every_machine() {
    // The two lists are each character that `char::is_whitespace` takes, and U+200E and U+200F besides.
    let whitespace: Vec<char> = (0..=0x10ffff).filter_map(char::from_u32).filter(|c| c.is_whitespace()).collect();
    let mut listed: Vec<char> =
        SPACES.into_iter().chain(LOOKALIKES).filter(|c| !matches!(c, '\u{200e}' | '\u{200f}')).collect();
    listed.sort();
    assert_eq!(whitespace, listed);
    let others = [
        ('\\', "\\"),
        ('`', "`"),
        ('\u{1}', "\\u{1}"),
        ('\u{7f}', "\\u{7f}"),
        ('§', "\\u{a7}"),
        ('\u{ad}', "\\u{ad}"),
        ('\u{200b}', "\\u{200b}"),
        ('\u{201c}', "\\u{201c}"),
        ('\u{feff}', "\\u{feff}"),
    ]
    .map(|(c, shown)| (c, shown.to_string()));
    let lookalikes = LOOKALIKES.map(|c| (c, format!("\\u{{{:x}}}", u32::from(c))));
    for abi in ["sysv64", "ptx-kernel"] {
        // The character is on line 2, right after the ABI.
        let head = format!("unsafe extern \"{abi}\"");
        for (c, shown) in lookalikes.iter().chain(&others) {
            let src = format!("// a\n{head}{c}fn k() {{}}\n");
            assert_eq!(compile(&src, "k"), Err(format!("2:{}: unexpected `{shown}`", head.len() + 1)), "{abi}: {c:?}");
        }
    }
}

/// A kernel for each machine, `sysv64` and `ptx-kernel`, whose parameter, variable, helper and const are named `ap`,
/// `av`, `ah` and `ak`, with `$` between the two letters of each name where it is declared and `%` where it is used.
const NAMED: [&str; 2] = [
    r#"
    const a$k: u64 = 7;
    pub unsafe extern "sysv64" fn k(a$p: u64) {
        let a$v: gpr;
        mov(a%v, a%p);
        a%h(a%v);
    }
    fn a$h(y: gpr) { add(y, a%k); }
    "#,
    r#"
    const a$k: u32 = 7;
    pub unsafe extern "ptx-kernel" fn k(a$p: *mut u32) {
        let a$v: u32;
        mov.u32(a%v, a%k);
        a%h(a%v);
        st.global.u32([a%p], a%v);
    }
    fn a$h(y: u32) { add.u32(y, y, a%k); }
    "#,
];

/// What each machine makes of `NAMED` with `declared` for each `$` and `used` for each `%`: the listing on x86, and the
/// PTX of the module on a GPU.
fn named(declared: &str, used: &str) -> [Result<String, String>; 2] {
    NAMED.map(|src| compile(&src.replace('$', declared).replace('%', used), "k"))
}

/// A name goes on with each character that Rust takes in one, on x86 and on a GPU, and not only with those that
/// `char::is_alphanumeric` takes: U+00B7 and U+203F, and U+0301 after a letter. A kernel whose names have one between
/// two letters compiles to the listing of the kernel whose names are ASCII, but for the names, which each machine
/// writes as it does: x86 as they are, in NFC, and a GPU's PTX with `$u` and the code point in hex and `$` for each
/// character beyond ASCII.
#[test]
fn a_name_goes_on_with_each_character_rust_takes_in_one_on_every_machine() {
    let [x86, gpu] = named("", "").map(Result::unwrap);
    // The character, and how each machine writes the name's first two letters, `a` and it, which are `á` for U+0301.
    for (c, on_x86, on_gpu) in [("·", "a·", "a$ub7$"), ("‿", "a‿", "a$u203f$"), ("\u{301}", "á", "$ue1$")] {
        let [on_x86_listing, on_gpu_listing] =
            named(c, c).map(|listing| listing.unwrap_or_else(|e| panic!("{c:?}: {e}")));
        assert_eq!(on_x86_listing.replace(on_x86, "a"), x86, "{c:?}");
        assert_eq!(on_gpu_listing.replace(on_gpu, "a"), gpu, "{c:?}");
    }
}

/// A name starts with each character that Rust takes to start one, and not only with those that `char::is_alphabetic`
/// takes: U+2118 is no letter. A kernel whose parameter is named so compiles, on x86 and on a GPU, to the listing of
/// the one named `a`, but for the name, which each machine writes as it does.
#[test]
fn a_name_starts_with_each_character_rust_takes_to_start_one_on_every_machine() {
    for (abi, on_machine) in [("sysv64", "℘"), ("ptx-kernel", "$u2118$")] {
        let listing = |name: &str| compile(&format!("unsafe extern \"{abi}\" fn k({name}: u64) {{}}\n"), "k").unwrap();
        assert_eq!(listing("℘").replace(on_machine, "a"), listing("a"), "{abi}");
    }
}

/// No other character is in a name. `a½` and `a²` have one that `char::is_alphanumeric` takes and Rust does not, and
/// `·`, `‿`, U+0301 and U+0345 go on a name and start none, U+0345 though `char::is_alphabetic` takes it. Each is
/// refused at the character, on x86 and on a GPU, in words that show which it is.
#[test]
fn no_other_character_is_in_a_name_on_every_machine() {
    for abi in ["sysv64", "ptx-kernel"] {
        let head = format!("unsafe extern \"{abi}\" fn k(");
        // The name, the character of it that is refused, and how many characters come before that.
        for (name, c, before) in [
            ("a½", '½', 1),
            ("a²", '²', 1),
            ("·a", '·', 0),
            ("‿a", '‿', 0),
            ("\u{301}a", '\u{301}', 0),
            ("\u{345}a", '\u{345}', 0),
        ] {
            let src = format!("{head}{name}: u64) {{}}\n");
            let column = head.chars().count() + before + 1;
            let want = format!("1:{column}: unexpected `\\u{{{:x}}}`", u32::from(c));
            assert_eq!(compile(&src, "k"), Err(want), "{abi}: {name:?}");
        }
    }
}

/// A name is one name in each of its spellings, as Rust reads each in NFC: `é` as U+00E9, and as `e` and U+0301. A
/// kernel that declares a name in one spelling and uses it in the other compiles to the listing of the kernel that
/// spells it alike, which writes the name in NFC. Two helpers so named are two of one name, refused where the second
/// is, as two of one spelling are. `x86_str` finds a kernel by its name in either spelling.
#[test]
fn a_name_in_each_of_its_spellings_is_one_name_on_every_machine() {
    let alike = named("é", "é").map(Result::unwrap);
    for (declared, used) in [("é", "e\u{301}"), ("e\u{301}", "é"), ("e\u{301}", "e\u{301}")] {
        assert_eq!(named(declared, used), alike.clone().map(Ok), "declared {declared:?}, used {used:?}");
    }
    let twice = |second: &str| format!("fn caf\u{e9}() {{}} fn {second}() {{}}");
    let want = "1:17: `caf\u{e9}` names another function of the module as well";
    for second in ["caf\u{e9}", "cafe\u{301}"] {
        assert_eq!(kernite::x86_str(&twice(second), "k").unwrap_err(), want, "{second:?}");
        assert_eq!(kernite::ptx_str(&twice(second)).unwrap_err(), want, "{second:?}");
    }
    let kernel = |declared: &str, given: &str| {
        compile(&format!("unsafe extern \"sysv64\" fn caf{declared}() {{}}\n"), &format!("caf{given}"))
    };
    let own = kernel("\u{e9}", "\u{e9}").unwrap();
    for (declared, given) in [("\u{e9}", "e\u{301}"), ("e\u{301}", "\u{e9}"), ("e\u{301}", "e\u{301}")] {
        assert_eq!(kernel(declared, given), Ok(own.clone()), "declared {declared:?}, given {given:?}");
    }
}

#[test]
fn each_parameter_and_generic_of_a_function_has_a_name_of_its_own() {
    // Rust refuses a name bound twice among a function's generics and parameters, E0403 and E0415;
    // without the check the body's `x` is the second, and the first goes without a word.
    let twice = |name: &str| format!("`{name}` names another parameter of the function as well");
    for (src, kernel, at, name) in [
        (r#"pub unsafe extern "sysv64" fn k(x: u64, x: u64) {}"#, "k", "1:41", "x"),
        // The twin need not be next to it.
        (r#"pub unsafe extern "sysv64" fn k(x: u64, y: u64, x: u64) {}"#, "k", "1:49", "x"),
        // A name written raw is the name.
        (r#"pub unsafe extern "sysv64" fn k(x: u64, r#x: u64) {}"#, "k", "1:41", "x"),
        // Only `_` alone names none.
        (r#"pub unsafe extern "sysv64" fn k(_x: u64, _x: u64) {}"#, "k", "1:42", "_x"),
        // A helper is never written out as Rust, so only kernite refuses one, called or not.
        ("pub unsafe extern \"sysv64\" fn k() {}\nfn h(x: ymm, x: ymm) { vaddps(x, x, x); }", "k", "2:14", "x"),
        // A generic kernel is compiled for the instances it lists, and each is named by its values.
        (
            "#[instances((1, 2))]\npub unsafe extern \"sysv64\" fn k<const N: u32, const N: u32>() {}",
            "k::<1, 2>",
            "2:53",
            "N",
        ),
        ("#[instances(1)]\npub unsafe extern \"sysv64\" fn k<const N: u64>(N: u64) {}", "k::<1>", "2:47", "N"),
    ] {
        assert_eq!(kernite::x86_str(src, kernel).unwrap_err(), format!("{at}: {}", twice(name)), "{src}");
    }
    // A GPU kernel and a GPU function are read as an x86 kernel is.
    for (src, at) in [
        (r#"pub unsafe extern "ptx-kernel" fn k(x: u64, x: u64) {}"#, "1:45"),
        (r#"pub unsafe extern "ptx-func" fn f(x: u64, x: u64) {}"#, "1:43"),
    ] {
        assert_eq!(kernite::ptx_str(src).unwrap_err(), format!("{at}: {}", twice("x")), "{src}");
    }
    // `_` names none, so two stay, as Rust takes them.
    let src = r#"pub unsafe extern "sysv64" fn k(_: u64, _: u64) {}"#;
    assert!(x86(src, "k").starts_with("# k(gpr, gpr) -> ()\n"), "{src}");
}

/// A `mut` before a parameter's name is taken and means nothing, in a function of every kind: a kernel compiles to the
/// listing, or the PTX, of the same kernel without it. So it is for an x86 kernel, plain, of `#[instances(..)]` and
/// `#[cooperative]`, and a helper; for a GPU kernel the same, and an `extern "ptx-func"` function; before one name and
/// before each.
#[test]
fn a_mut_before_a_parameters_name_is_taken_and_means_nothing_in_a_function_of_every_kind() {
    for (src, kernel) in [
        (r#"pub unsafe extern "sysv64" fn k(MUTx: u64) -> u64 { add(x, 1); ret(x); }"#, "k"),
        (r#"pub unsafe extern "sysv64" fn k(MUTa: u64, MUTb: u64) -> u64 { add(a, b); ret(a); }"#, "k"),
        (r#"pub unsafe extern "sysv64" fn k(a: u64, MUTb: u64) -> u64 { add(a, b); ret(a); }"#, "k"),
        (
            r#"
            #[instances(1, 2)]
            pub unsafe extern "sysv64" fn k<const N: u64>(MUTx: u64) -> u64 { add(x, N); ret(x); }
            "#,
            "k::<2>",
        ),
        (
            r#"
            #[cooperative]
            pub unsafe extern "sysv64" fn k(MUTx: u64, p: *mut u64) { grid.sync(); mov(qword[p], x); }
            "#,
            "k",
        ),
        (
            r#"
            pub unsafe extern "sysv64" fn k(x: u64) -> u64 { h(x); ret(x); }
            fn h(MUTy: gpr) { add(y, 1); }
            "#,
            "k",
        ),
        (
            r#"
            pub unsafe extern "ptx-kernel" fn k(p: *mut u32, MUTx: u32) {
                add.u32(x, x, 1);
                st.global.u32([p], x);
            }
            "#,
            "k",
        ),
        (
            r#"
            #[instances(1, 2)]
            pub unsafe extern "ptx-kernel" fn k<const N: u32>(p: *mut u32, MUTx: u32) {
                add.u32(x, x, N);
                st.global.u32([p], x);
            }
            "#,
            "k",
        ),
        (
            r#"
            #[cooperative]
            pub unsafe extern "ptx-kernel" fn k(p: *mut u32, MUTx: u32) { grid.sync(); st.global.u32([p], x); }
            "#,
            "k",
        ),
        (
            r#"
            pub unsafe extern "ptx-kernel" fn k(p: *mut u32, x: u32) { h(x); st.global.u32([p], x); }
            fn h(MUTy: u32) { add.u32(y, y, 1); }
            "#,
            "k",
        ),
        (
            r#"
            extern "ptx-func" fn f(MUTa: u32, MUTb: u32) -> u32 { add.u32(a, a, b); ret(a); }
            pub unsafe extern "ptx-kernel" fn k(p: *mut u32, a: u32, b: u32) {
                let r: u32;
                call(r, f, (a, b));
                st.global.u32([p], r);
            }
            "#,
            "k",
        ),
    ] {
        let without = compile(&src.replace("MUT", ""), kernel).unwrap_or_else(|e| panic!("{e}\n{src}"));
        assert_eq!(compile(&src.replace("MUT", "mut "), kernel), Ok(without), "{src}");
    }
}

/// `src` with each of `names` written raw, `u64` as `r#u64`, but not where it is a part of an instruction's name,
/// `ld.global.u32`.
fn raw(src: &str, names: &[&str]) -> String {
    let chars: Vec<char> = src.chars().collect();
    let in_word = |c: char| c.is_alphanumeric() || c == '_';
    let (mut out, mut at) = (String::new(), 0);
    while at < chars.len() {
        let end = (at..chars.len()).find(|&i| !in_word(chars[i])).unwrap_or(chars.len());
        if end == at {
            out.push(chars[at]);
            at += 1;
            continue;
        }
        let word: String = chars[at..end].iter().collect();
        let dotted = (at > 0 && chars[at - 1] == '.') || chars.get(end) == Some(&'.');
        out += if names.contains(&word.as_str()) && !dotted { "r#" } else { "" };
        out += &word;
        at = end;
    }
    out
}

/// A name written raw is the name Rust reads, so a type written so is the type written plain: a kernel compiles to
/// the listing, or the PTX, of the same kernel with its types written plain. So it is for a parameter's type and a
/// result's, a number's, a float's or a pointer's, in a kernel of instances too; for a variable's, alone or in an
/// array; for a helper's parameter's; and for a const's and a static's.
#[test]
fn a_type_written_raw_is_the_type_written_plain() {
    let x86_names = ["u64", "usize", "u32", "f32", "f64", "u8", "gpr", "ymm", "global"];
    for (src, kernel) in [
        (r#"pub unsafe extern "sysv64" fn k(x: u64, n: usize) -> u64 { add(x, n); ret(x); }"#, "k"),
        (r#"pub unsafe extern "sysv64" fn k(x: u64) -> u32 { add(x, 1); ret(x); }"#, "k"),
        (r#"pub unsafe extern "sysv64" fn k(a: f32, b: f32) -> f32 { vaddss(a, a, b); ret(a); }"#, "k"),
        (r#"pub unsafe extern "sysv64" fn k(a: f64, b: f64) -> f64 { vaddsd(a, a, b); ret(a); }"#, "k"),
        (r#"pub unsafe extern "sysv64" fn k(p: *const u8, q: *mut u64) { mov(qword[q], p); }"#, "k"),
        (r#"pub unsafe extern "sysv64" fn k(p: *const f32) -> *const f32 { ret(p); }"#, "k"),
        (
            "#[instances(1, 2)]\npub unsafe extern \"sysv64\" fn k<const N: u64>(x: u64) -> u64 { add(x, N); ret(x); }",
            "k::<2>",
        ),
        (
            r#"
            const N: usize = 4;
            pub unsafe extern "sysv64" fn k(x: u64) -> u64 {
                let t: gpr;
                let v: ymm;
                let w: [gpr; 2];
                mov(t, x);
                mov(w[0], t);
                mov(w[1], N);
                vmovups(v, [x]);
                twice(t);
                add(x, t);
                ret(x);
            }
            fn twice(x: gpr) { add(x, x); }
            "#,
            "k",
        ),
        (
            r#"
            static S: global<[u32; 4]>;
            pub unsafe extern "sysv64" fn k() { let at: gpr; lea(at, [S]); }
            "#,
            "k",
        ),
    ] {
        let written = raw(src, &x86_names);
        assert_ne!(written, src);
        assert_eq!(x86(&written, kernel), x86(src, kernel), "{written}");
    }
    let gpu_names = ["u32", "u64", "f32", "f16", "bf16", "pred", "global"];
    for src in [
        r#"pub unsafe extern "ptx-kernel" fn g(n: u32, w: u64, x: f32, out: *mut u32) { st.global.u32([out], n); }"#,
        r#"pub unsafe extern "ptx-kernel" fn g(h: f16, b: bf16, out: *mut f16) { st.global.b16([out], h); }"#,
        r#"pub unsafe extern "ptx-kernel" fn g(out: *mut bf16, b: bf16) { st.global.b16([out], b); }"#,
        r#"
        extern "ptx-func" fn f(a: u64, h: f16) -> u32 { let y: u32; ld.global.u32(y, [a]); ret(y); }
        pub unsafe extern "ptx-kernel" fn g(p: u64, h: f16) { let r: u32; call(r, f, (p, h)); st.global.u32([p], r); }
        "#,
        r#"
        pub unsafe extern "ptx-kernel" fn g(n: u32, out: *mut u32) {
            let i: u32;
            mov.u32(i, tid.x);
            let p: pred;
            setp.ge.u32(p, i, n);
            if p {
                ret();
            }
            let r: [u32; 2];
            add.u32(r[0], n, i);
            add.u32(r[1], n, i);
            st.global.u32([out], r[1]);
        }
        "#,
        r#"
        static S: global<[u32; 4]>;
        pub unsafe extern "ptx-kernel" fn g(out: *mut u32) {
            let at: u64;
            mov.u64(at, S);
            red.global.add.u32([at], 1);
            st.global.u32([out], 1);
        }
        "#,
    ] {
        let written = raw(src, &gpu_names);
        assert_ne!(written, src);
        let module = |src: &str| kernite::ptx_str(src).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(module(&written), module(src), "{written}");
    }
}

/// An error that quotes a type written raw quotes it as Rust reads it, `f16` and not `r#f16`, at the place it has
/// for the type written plain: a parameter's type, a result's, a variable's, and a pointer's, whole, in an x86
/// kernel, a cooperative one and a GPU kernel.
#[test]
fn an_error_quotes_a_type_written_raw_as_rust_reads_it() {
    let takes = "a kernel takes pointers, 64-bit integers and floats, not `f16`";
    let returns = "a kernel returns a float, an integer or a pointer, or never, `!`; not `f16`";
    let nothing = "a cooperative kernel's threads return nothing: they write what they compute to memory";
    let variable = "x86 variables are `gpr`, `ymm`, `xmm` and stack memory, `local<..>`; not `f16`";
    let words = "a pointer to `str` has two words, an address and a length or a table, and a kernel's pointer has one";
    let gpu_takes =
        "a GPU kernel takes pointers, integers and the floats `f16`, `bf16`, `f32` and `f64`, not `[u32; 2]`";
    let gpu_nothing = "a GPU kernel returns nothing, not `*const [f32; 2]`: it writes what it works out to memory";
    let gpu_variable = "PTX variables are `pred`, `b8` to `b128`, `u8` to `u64`, `s8` to `s64`, `f16`, `f16x2`, \
                        `bf16`, `bf16x2`, `tf32`, `f32` and `f64`, not `f128`";
    for (plain, name, at, said) in [
        (r#"unsafe extern "sysv64" fn k(x: f16) {}"#, "f16", "1:32", takes),
        (r#"unsafe extern "sysv64" fn k() -> f16 {}"#, "f16", "1:34", returns),
        ("#[cooperative]\nunsafe extern \"sysv64\" fn k() -> u64 {}", "u64", "2:34", nothing),
        (r#"unsafe extern "sysv64" fn k() { let v: [f16; 2]; }"#, "f16", "1:41", variable),
        (r#"unsafe extern "sysv64" fn k(p: *const str) {}"#, "str", "1:32", words),
        (r#"unsafe extern "ptx-kernel" fn k(x: [u32; 2]) {}"#, "u32", "1:36", gpu_takes),
        (r#"unsafe extern "ptx-kernel" fn k() -> *const [f32; 2] {}"#, "f32", "1:38", gpu_nothing),
        (r#"unsafe extern "ptx-kernel" fn k() { let v: f128; }"#, "f128", "1:44", gpu_variable),
    ] {
        let err = |src: &str| compile(src, "k").unwrap_err();
        let written = raw(plain, &[name]);
        assert_ne!(written, plain);
        assert_eq!(err(plain), format!("{at}: {said}"), "{plain}");
        assert_eq!(err(&written), format!("{at}: {said}"), "{written}");
    }
}

/// A keyword written raw is a name in a type: Rust reads `*const r#dyn::Foo`, a pointer to a struct of a module named
/// `dyn`, and `*const r#dyn<u64>`, a pointer to a generic struct named `dyn`, as pointers of one word, where a pointer
/// to `dyn Foo` has two. So a kernel that names a type by a keyword written raw compiles to the listing, or the PTX,
/// of the kernel that names types of the same layout by words that are no keywords, `dyn_::Foo` say, whichever keyword
/// it is: `dyn`, which makes a pointer two words; `mut` and `const`, which are spaced; `for`, which opens a binder;
/// `impl`, which a parameter may not have; `true`, which is a literal; and `type`, which no reader asks of.
#[test]
fn a_keyword_written_raw_is_a_name_in_a_type() {
    let x86_kernel = |name: &str| {
        format!(
            r#"pub unsafe extern "sysv64" fn k(
                p: *const {name}::Foo,
                q: *mut {name}<u64>,
                r: *const Arr<{{ {name} }}>,
            ) -> *const {name}::Foo {{
                ret(p);
            }}"#
        )
    };
    let gpu_kernel = |name: &str| {
        format!(
            r#"pub unsafe extern "ptx-kernel" fn k(p: *mut {name}::Foo, q: *const {name}<u64>, n: u64) {{
                st.global.u64([p], n);
            }}"#
        )
    };
    for word in ["dyn", "mut", "const", "for", "impl", "true", "type"] {
        // The kernel it is compared with names the keyword and a `_`, `dyn_`, which is no keyword.
        let (raw, no_keyword) = (format!("r#{word}"), format!("{word}_"));
        assert_eq!(x86(&x86_kernel(&raw), "k"), x86(&x86_kernel(&no_keyword), "k"), "{}", x86_kernel(&raw));
        assert_eq!(ptx(&gpu_kernel(&raw)), ptx(&gpu_kernel(&no_keyword)), "{}", gpu_kernel(&raw));
    }
}

/// A const named by a keyword written raw is found where a type takes it as an argument, as `ALIGN` is:
/// `shared<[u32; 4], r#type>` is `shared<[u32; 4], 16>`, and so is `shared<[u32; 4], ALIGN>`, whether `ALIGN` is
/// written raw where it is declared, where it is used, or both.
#[test]
fn a_const_named_by_a_keyword_is_an_argument_of_a_type() {
    let kernel = |declared: &str, align: &str| {
        format!(
            "const {declared}: usize = 16;\n\
             pub unsafe extern \"ptx-kernel\" fn g(out: *mut u32) {{\n\
                 let tile: shared<[u32; 4], {align}>;\n\
                 let at: u32;\n\
                 mov.u32(at, tile);\n\
                 st.global.u32([out], at);\n\
             }}"
        )
    };
    let literal = "pub unsafe extern \"ptx-kernel\" fn g(out: *mut u32) { let tile: shared<[u32; 4], 16>; let at: u32; \
                   mov.u32(at, tile); st.global.u32([out], at); }";
    for (declared, align) in
        [("r#type", "r#type"), ("r#ALIGN", "ALIGN"), ("ALIGN", "r#ALIGN"), ("r#ALIGN", "r#ALIGN"), ("ALIGN", "ALIGN")]
    {
        assert_eq!(ptx(&kernel(declared, align)), ptx(literal), "{declared}, {align}");
    }
}

/// An error that quotes a keyword written raw quotes it as written, `r#dyn::Foo` and `r#mut`: without its `r#` it is
/// the keyword, a type of another kind, and the error would name it. A name that is no keyword is quoted as Rust
/// reads it, `str` for `r#str`, and a pointer to it has two words as one to `str` has.
#[test]
fn an_error_quotes_a_keyword_written_raw_as_written() {
    let takes = |ty: &str| format!("a kernel takes pointers, 64-bit integers and floats, not `{ty}`");
    let returns = |ty: &str| format!("a kernel returns a float, an integer or a pointer, or never, `!`; not `{ty}`");
    let variable =
        |ty: &str| format!("x86 variables are `gpr`, `ymm`, `xmm` and stack memory, `local<..>`; not `{ty}`");
    let words = |ty: &str| {
        format!("a pointer to `{ty}` has two words, an address and a length or a table, and a kernel's pointer has one")
    };
    let gpu_takes = |ty: &str| {
        format!("a GPU kernel takes pointers, integers and the floats `f16`, `bf16`, `f32` and `f64`, not `{ty}`")
    };
    let gpu_variable = |ty: &str| {
        format!(
            "PTX variables are `pred`, `b8` to `b128`, `u8` to `u64`, `s8` to `s64`, `f16`, `f16x2`, `bf16`, \
             `bf16x2`, `tf32`, `f32` and `f64`, not `{ty}`"
        )
    };
    let gpu_nothing =
        |ty: &str| format!("a GPU kernel returns nothing, not `{ty}`: it writes what it works out to memory");
    for (src, at, said) in [
        (r#"unsafe extern "sysv64" fn k(x: r#dyn::Foo) {}"#, "1:32", takes("r#dyn::Foo")),
        (r#"unsafe extern "sysv64" fn k(x: r#mut) {}"#, "1:32", takes("r#mut")),
        (r#"unsafe extern "sysv64" fn k(x: r#mut<u8>) {}"#, "1:32", takes("r#mut<u8>")),
        (r#"unsafe extern "sysv64" fn k(x: Arr<{ r#true }>) {}"#, "1:32", takes("Arr<{ r#true }>")),
        (r#"unsafe extern "sysv64" fn k(x: fn(r#mut) -> r#dyn) {}"#, "1:32", takes("fn(r#mut) -> r#dyn")),
        (r#"unsafe extern "sysv64" fn k(x: *r#const u64) {}"#, "1:32", takes("*r#const u64")),
        (r#"unsafe extern "sysv64" fn k() -> r#mut {}"#, "1:34", returns("r#mut")),
        (r#"unsafe extern "sysv64" fn k() { let v: [r#mut; 2]; }"#, "1:41", variable("r#mut")),
        (r#"unsafe extern "sysv64" fn k(p: *const [r#mut]) {}"#, "1:32", words("[r#mut]")),
        (r#"unsafe extern "sysv64" fn k(p: *const dyn r#mut) {}"#, "1:32", words("dyn r#mut")),
        (r#"unsafe extern "sysv64" fn k(p: *const r#str) {}"#, "1:32", words("str")),
        (r#"unsafe extern "sysv64" fn k(p: *const (u8, r#str)) {}"#, "1:32", words("(u8, str)")),
        (r#"unsafe extern "ptx-kernel" fn k(x: r#dyn::Foo) {}"#, "1:36", gpu_takes("r#dyn::Foo")),
        (r#"unsafe extern "ptx-kernel" fn k(x: r#mut) {}"#, "1:36", gpu_takes("r#mut")),
        (r#"unsafe extern "ptx-kernel" fn k() -> *const r#mut {}"#, "1:38", gpu_nothing("*const r#mut")),
        (r#"unsafe extern "ptx-kernel" fn k() { let v: r#mut<u8>; }"#, "1:44", gpu_variable("r#mut<u8>")),
        (r#"unsafe extern "ptx-kernel" fn k(p: *const [r#mut]) {}"#, "1:36", words("[r#mut]")),
        (r#"unsafe extern "ptx-kernel" fn k(p: *const r#str) {}"#, "1:36", words("str")),
    ] {
        assert_eq!(compile(src, "k").unwrap_err(), format!("{at}: {said}"), "{src}");
    }
}

/// A name written raw is the name in a value's place as in a type's: `r#x` and `x` are one value, whether a
/// kernel's name, a parameter, a variable, a const, a compile-time argument or a helper, and whether the name is
/// written raw where it is declared, where it is used, or both.
#[test]
fn a_value_named_raw_is_the_value_named_plain() {
    let src = r#"
        const STEP: u64 = 3;
        #[instances(2)]
        pub unsafe extern "sysv64" fn k<const N: u64>(x: u64) -> u64 {
            let t: gpr;
            mov(t, x);
            bump(t);
            add(x, t);
            add(x, STEP);
            add(x, N);
            ret(x);
        }
        fn bump(g: gpr) { add(g, 1); }
    "#;
    let written = raw(src, &["STEP", "N", "x", "t", "g", "k", "bump"]);
    assert_ne!(written, src);
    assert_eq!(x86(&written, "k::<2>"), x86(src, "k::<2>"), "{written}");
    // A name raw where it is declared and plain where it is used, and the other way about, in one kernel.
    let plain = r#"pub unsafe extern "sysv64" fn k(x: u64) -> u64 { add(x, 1); ret(x); }"#;
    for mixed in [
        r#"pub unsafe extern "sysv64" fn k(r#x: u64) -> u64 { add(x, 1); ret(r#x); }"#,
        r#"pub unsafe extern "sysv64" fn k(x: u64) -> u64 { add(r#x, 1); ret(x); }"#,
    ] {
        assert_eq!(x86(mixed, "k"), x86(plain, "k"), "{mixed}");
    }
}

/// The attributes of kernite's: a function's `#[instances(..)]` and `#[cooperative]`, and a GPU kernel's directives,
/// `#[maxnctapersm(..)]`, which PTX no longer has, among them.
const ATTRIBUTES: [&str; 7] =
    ["instances", "cooperative", "maxntid", "reqntid", "minnctapersm", "maxnreg", "maxnctapersm"];

/// An attribute of kernite's written raw is the attribute, as Rust reads `r#cfg` as `cfg`: an x86 kernel with
/// `#[r#instances(..)]` or `#[r#cooperative]` compiles to the listing of the one written plain, and a GPU kernel with
/// those and with each directive to its PTX.
#[test]
fn an_attribute_written_raw_is_the_attribute() {
    let tile = r#"
        #[instances(1, 2)]
        pub unsafe extern "sysv64" fn k<const N: u64>(x: u64) -> u64 {
            add(x, N);
            ret(x);
        }
    "#;
    let together = r#"
        #[cooperative]
        pub unsafe extern "sysv64" fn k(p: *mut u64) {
            grid.sync();
            mov(qword[p], 1);
        }
    "#;
    for (src, kernel) in [(tile, "k::<1>"), (tile, "k::<2>"), (together, "k")] {
        let written = raw(src, &ATTRIBUTES);
        assert_ne!(written, src);
        assert_eq!(x86(&written, kernel), x86(src, kernel), "{written}");
    }
    assert_ne!(x86(tile, "k::<1>"), x86(tile, "k::<2>"));
    assert!(x86(together, "k").contains("lock xadd"));
    let gpu = r#"
        #[instances(1, 2)]
        #[maxntid(256)]
        pub unsafe extern "ptx-kernel" fn tile<const N: u32>(x: *mut u32) {
            st.global.u32([x], N);
        }

        #[cooperative]
        #[reqntid(8, 8, 4)]
        #[minnctapersm(2)]
        #[maxnreg(32)]
        pub unsafe extern "ptx-kernel" fn together(x: *mut u32) {
            grid.sync();
            st.global.u32([x], 1);
        }
    "#;
    let ptx = kernite::ptx_str(gpu).unwrap();
    let want = [".maxntid 256", ".maxnreg 32", ".minnctapersm 2", ".reqntid 8, 8, 4", "atom.add.u32 %old, [%barrier]"];
    assert!(want.iter().all(|line| ptx.contains(line)), "{ptx}");
    assert_eq!(kernite::ptx_str(&raw(gpu, &ATTRIBUTES)).unwrap(), ptx);
}

/// An attribute of kernite's written raw is wrong as the plain one is: the same error, in the same words, at the same
/// `line:column`. Each attribute is on a line of its own, so that its name starts where the plain one's does.
#[test]
fn an_attribute_written_raw_is_wrong_as_the_plain_one_is() {
    let kernel = "unsafe extern \"ptx-kernel\" fn k() {}";
    let generic = "unsafe extern \"ptx-kernel\" fn k<const N: u32>() {}";
    let gives =
        |name: &str, what: &str, example: &str| format!("`#[{name}(..)]` gives {what}, as in `#[{name}({example})]`");
    let cases = [
        (
            format!("#[instances]\n{generic}"),
            "1:3: `#[instances(..)]` lists what a function is compiled for, as in `#[instances(4, 8)]`".to_string(),
        ),
        (
            format!("#[instances(1)]\n#[instances(2)]\n{generic}"),
            "2:3: a function has one `#[instances(..)]`, which lists each instance it is compiled for".into(),
        ),
        (
            format!("#[instances(1)]\n{kernel}"),
            "1:3: `k` takes no compile-time arguments, so it is compiled once, with no `#[instances(..)]`".into(),
        ),
        (
            "#[instances(1)]\nconst C: u32 = 1;".into(),
            "1:3: `#[instances(..)]` goes on a function that names an ABI, which it lists the instances of".into(),
        ),
        (
            format!("#[cooperative(1)]\n{kernel}"),
            "1:3: `#[cooperative]` takes nothing: it says that a kernel's blocks or threads wait for each other".into(),
        ),
        (format!("#[cooperative]\n#[cooperative]\n{kernel}"), "2:3: a kernel has one `#[cooperative]`".into()),
        (
            "#[cooperative]\nfn h() {}".into(),
            "1:3: `#[cooperative]` goes on a kernel whose blocks or threads wait for each other, `extern \
             \"ptx-kernel\"` or `extern \"sysv64\"`; `h` is a helper, inlined where it is called"
                .into(),
        ),
        (
            format!("#[maxntid]\n{kernel}"),
            format!("1:3: {}", gives("maxntid", "the most threads a block has, along x, then y and z", "16, 16")),
        ),
        (
            format!("#[reqntid]\n{kernel}"),
            format!("1:3: {}", gives("reqntid", "the threads a block has, along x, then y and z", "8, 8, 4")),
        ),
        (
            format!("#[minnctapersm]\n{kernel}"),
            format!("1:3: {}", gives("minnctapersm", "the blocks an SM should hold at once, at least", "2")),
        ),
        (
            format!("#[maxnreg]\n{kernel}"),
            format!("1:3: {}", gives("maxnreg", "the most registers a thread may use", "32")),
        ),
        (format!("#[maxnreg(32)]\n#[maxnreg(32)]\n{kernel}"), "2:3: a kernel has one `#[maxnreg(..)]`".into()),
        (
            format!("#[maxnctapersm(2)]\n{kernel}"),
            "1:3: PTX has no `.maxnctapersm` from 2.1 on; `#[minnctapersm(..)]` gives the blocks an SM should hold at \
             once, at least"
                .into(),
        ),
        (
            "#[maxntid(256)]\nunsafe extern \"sysv64\" fn k() {}".into(),
            "1:3: `#[maxntid(..)]` goes on a GPU kernel, `extern \"ptx-kernel\"`; `k` is a kernel for x86-64, which \
             the process runs"
                .into(),
        ),
    ];
    for (src, want) in cases {
        let written = raw(&src, &ATTRIBUTES);
        assert_ne!(written, src);
        for src in [&src, &written] {
            assert_eq!(kernite::ptx_str(src).unwrap_err(), want, "{src}");
        }
    }
}

/// What kernite says of an `inline` on a function or a static.
const INLINED: &str = "a helper is always inlined, and nothing else is: `#[inline]` changes nothing kernite compiles";

/// Each way to write `#[inline]`, inside the brackets, and the word where the error stands. In `cfg_attr(..)` it is the
/// `inline` it gives, whatever its condition: `any()` is false, `all()` true.
const INLINES: [(&str, &str); 14] = [
    ("inline", "inline"),
    ("inline(always)", "inline"),
    ("inline(never)", "inline"),
    ("inline = \"never\"", "inline"),
    ("r#inline", "r#inline"),
    ("r#inline(never)", "r#inline"),
    ("cfg_attr(all(), inline)", "inline"),
    ("cfg_attr(any(), inline(never))", "inline"),
    ("cfg_attr(target_arch = \"x86_64\", inline(always))", "inline"),
    ("cfg_attr(all(), cfg_attr(any(), inline))", "inline"),
    ("cfg_attr(all(), doc = \"x\", r#inline)", "r#inline"),
    ("r#cfg_attr(all(), inline)", "inline"),
    ("cfg_attr(all(), r#inline = \"always\")", "r#inline"),
    ("cfg_attr(any(), cfg_attr(any(), doc = \"x\", inline(always)))", "inline"),
];

/// Each item that refuses `#[inline]`, with `ATTR` where it goes, on a line of its own, and the kernel to ask of the
/// source: a helper; an `extern "ptx-func"` function; a kernel for x86-64, plain, of `#[instances(..)]` and
/// `#[cooperative]`, the attribute before kernite's own and after; a GPU kernel the same; and a static of each machine.
/// Each compiles without the attribute.
const INLINED_ITEMS: [(&str, &str); 18] = [
    ("unsafe extern \"sysv64\" fn k() { h(); }\nATTR\nfn h() {}", "k"),
    ("unsafe extern \"sysv64\" fn k() { h(); }\nATTR\nconst fn h() {}", "k"),
    ("unsafe extern \"ptx-kernel\" fn k() { h(); }\nATTR\nfn h() {}", "k"),
    ("ATTR\nextern \"ptx-func\" fn f(a: u32) -> u32 { ret(a); }\nunsafe extern \"ptx-kernel\" fn k() {}", "k"),
    ("ATTR\npub unsafe extern \"sysv64\" fn k() {}", "k"),
    ("ATTR\n#[instances(1, 2)]\nunsafe extern \"sysv64\" fn k<const N: u64>() {}", "k::<1>"),
    ("#[instances(1, 2)]\nATTR\nunsafe extern \"sysv64\" fn k<const N: u64>() {}", "k::<2>"),
    ("ATTR\n#[cooperative]\nunsafe extern \"sysv64\" fn k(p: *mut u64) { grid.sync(); mov(qword[p], 1); }", "k"),
    ("#[cooperative]\nATTR\nunsafe extern \"sysv64\" fn k(p: *mut u64) { grid.sync(); mov(qword[p], 1); }", "k"),
    ("ATTR\npub unsafe extern \"ptx-kernel\" fn k() {}", "k"),
    ("ATTR\n#[instances(1, 2)]\nunsafe extern \"ptx-kernel\" fn k<const N: u32>() {}", "k"),
    ("#[instances(1, 2)]\nATTR\nunsafe extern \"ptx-kernel\" fn k<const N: u32>() {}", "k"),
    (
        "ATTR\n#[cooperative]\nunsafe extern \"ptx-kernel\" fn k(p: *mut u32) { grid.sync(); st.global.u32([p], 1); }",
        "k",
    ),
    (
        "#[cooperative]\nATTR\nunsafe extern \"ptx-kernel\" fn k(p: *mut u32) { grid.sync(); st.global.u32([p], 1); }",
        "k",
    ),
    ("ATTR\nstatic S: global<[u32; 4]>;\nunsafe extern \"sysv64\" fn k() {}", "k"),
    ("ATTR\npub static C: constant<[f32; 2]> = [0.5, 1.5];\nunsafe extern \"sysv64\" fn k() {}", "k"),
    ("ATTR\nstatic S: global<[u32; 4]>;\nunsafe extern \"ptx-kernel\" fn k() {}", "k"),
    ("ATTR\npub static C: constant<[f32; 2]> = [0.5, 1.5];\nunsafe extern \"ptx-kernel\" fn k() {}", "k"),
];

/// A helper is always inlined and nothing else is, so `#[inline]` changes nothing kernite compiles, and kernite refuses
/// it, in every form, on a function or a static: one error, at the `inline`. `kernite::x86_str` and `kernite::ptx_str`
/// say it of every source, whatever machine its kernels are for, for kernite reads the attribute before it asks one.
#[test]
fn an_inline_attribute_on_a_function_or_a_static_is_refused_at_its_inline() {
    for (item, kernel) in INLINED_ITEMS {
        let line = item.lines().position(|line| line == "ATTR").expect("a line for the attribute") + 1;
        compile(&item.replace("ATTR\n", ""), kernel).unwrap_or_else(|e| panic!("{e}\n{item}"));
        for (form, word) in INLINES {
            let src = item.replace("ATTR", &format!("#[{form}]"));
            let column = format!("#[{form}]").find(word).expect("the word of the error") + 1;
            let want = format!("{line}:{column}: {INLINED}");
            assert_eq!(kernite::x86_str(&src, kernel).unwrap_err(), want, "{src}");
            assert_eq!(kernite::ptx_str(&src).unwrap_err(), want, "{src}");
        }
    }
}

/// An item that is neither a function nor a static keeps its `#[inline]` for Rust to read: a const kernite works out
/// compiles as it does without the attribute, on x86 and on a GPU. The macros hand the const to Rust as written, and
/// Rust says what is wrong of it.
#[test]
fn an_inline_attribute_on_a_const_is_left_to_rust() {
    for (src, kernel) in [
        ("#[inline]\nconst N: u32 = 1;\nunsafe extern \"sysv64\" fn k() { let r: gpr; mov(r, N); }", "k"),
        (
            "#[inline]\nconst N: u32 = 1;\nunsafe extern \"ptx-kernel\" fn k(p: *mut u32) { st.global.u32([p], N); }",
            "k",
        ),
    ] {
        let without = compile(&src.replace("#[inline]\n", ""), kernel).unwrap_or_else(|e| panic!("{e}\n{src}"));
        assert_eq!(compile(src, kernel), Ok(without), "{src}");
    }
}

/// The places a name is written, a kernel for each, `NAME` where the name goes, and the kernel to ask of it: a kernel's
/// name, a parameter's, a variable's, a type, and the start of a type's path, for x86 and then for a GPU. Each is one
/// line, so the column of an error at its first `r#` is where that `r#` stands, and each compiles with a plain name
/// for `NAME`.
const PLACES: [(&str, &str); 10] = [
    (r#"unsafe extern "sysv64" fn NAME(x: u64) -> u64 { ret(x); }"#, "NAME"),
    (r#"unsafe extern "sysv64" fn k(NAME: u64) -> u64 { ret(NAME); }"#, "k"),
    (r#"unsafe extern "sysv64" fn k(x: u64) -> u64 { let NAME: gpr; mov(NAME, x); ret(NAME); }"#, "k"),
    (r#"unsafe extern "sysv64" fn k(p: *const NAME) -> *const NAME { ret(p); }"#, "k"),
    (r#"unsafe extern "sysv64" fn k(p: *const NAME::Foo) -> *const NAME::Foo { ret(p); }"#, "k"),
    (r#"unsafe extern "ptx-kernel" fn NAME(p: *mut u64) { st.global.u64([p], 1); }"#, "NAME"),
    (r#"unsafe extern "ptx-kernel" fn k(NAME: *mut u64) { st.global.u64([NAME], 1); }"#, "k"),
    (
        r#"unsafe extern "ptx-kernel" fn k(p: *mut u8) { let NAME: u64; mov.u64(NAME, 1); st.global.u64([p], NAME); }"#,
        "k",
    ),
    (r#"unsafe extern "ptx-kernel" fn k(p: *mut NAME) { st.global.u64([p], 1); }"#, "k"),
    (r#"unsafe extern "ptx-kernel" fn k(p: *mut NAME::Foo) { st.global.u64([p], 1); }"#, "k"),
];

/// An attribute's name, one more place a name is written, on x86 and then on a GPU, as `PLACES` has them. No name that
/// `PLACES` takes is an attribute of kernite's, so these are for the refusals alone.
const ATTRIBUTE_PLACES: [&str; 2] =
    [r#"#[NAME] unsafe extern "sysv64" fn k() {}"#, r#"#[NAME(256)] unsafe extern "ptx-kernel" fn k() {}"#];

/// The sources that have `NAME` in a place where a name is written: those of `PLACES`, and an attribute's.
fn places() -> impl Iterator<Item = &'static str> {
    PLACES.iter().map(|&(place, _)| place).chain(ATTRIBUTE_PLACES)
}

/// Five names Rust lets be written raw in no place, `self`, `Self`, `super`, `crate` and `_`, are refused at their `r#`
/// wherever a name is written: a kernel's, a parameter's, a variable's, an attribute's, a type, and the start of a
/// type's path, on x86 and on a GPU. The error says the name as Rust reads it.
#[test]
fn a_name_rust_lets_be_written_raw_in_no_place_is_refused_at_its_r() {
    for name in ["self", "Self", "super", "crate", "_"] {
        for place in places() {
            let src = place.replace("NAME", &format!("r#{name}"));
            let at = src.find("r#").expect("a raw name") + 1;
            assert_eq!(compile(&src, "k"), Err(format!("1:{at}: `{name}` cannot be written raw")), "{src}");
        }
    }
}

/// An `r#` that no name follows is refused at the `r#`, wherever a name is written, whether a bracket, a colon, a
/// dot or a `#` comes after it, or a space or a digit: none of these starts a name. A quote starts a raw string.
#[test]
fn an_r_that_no_name_follows_is_refused_at_its_r() {
    let said = "`r#` comes before a name, as in `r#u64`";
    let mut sources: Vec<String> = places().map(|place| place.replace("NAME", "r#")).collect();
    for after in [" x", "9", "9x", ".x", "#x"] {
        sources.push(format!(r#"unsafe extern "sysv64" fn k(r#{after}: u64) -> u64 {{ ret(x); }}"#));
    }
    for src in sources {
        let at = src.find("r#").expect("a raw name") + 1;
        assert_eq!(compile(&src, "k"), Err(format!("1:{at}: {said}")), "{src}");
    }
}

/// Every other name written raw is a name wherever one is written: `r#u64`, a type's; `r#dyn`, a keyword; `r#union`, a
/// weak one; and `r#gen` and `r#try`, reserved words. A kernel with one compiles to what the kernel with `xxx` in its
/// place compiles to, with the name where `xxx` was, on x86 and on a GPU.
#[test]
fn every_other_name_written_raw_is_a_name_wherever_a_name_is_written() {
    for name in ["u64", "dyn", "union", "gen", "try"] {
        let plain = "x".repeat(name.len());
        for (place, kernel) in PLACES {
            let compiles = |written: &str| {
                let src = place.replace("NAME", written);
                let kernel = kernel.replace("NAME", written.trim_start_matches("r#"));
                compile(&src, &kernel).unwrap_or_else(|e| panic!("{e}\n{src}"))
            };
            assert_eq!(compiles(&format!("r#{name}")), compiles(&plain).replace(&plain, name), "{place}");
        }
    }
}

#[test]
fn a_generic_is_a_value_and_a_type_is_refused_at_its_token() {
    // `<T>` is a generic that is a type, and so is an `impl` in a parameter's type or the result's: in a
    // parameter it makes the function generic over a type, and in the result it hides one. A kernel's generics
    // are values, so each is refused, at the `T` or the first `impl`, in one set of words, wherever the type
    // holds it. A pointer to `impl ?Sized` is among them: Rust passes one to a type with no size in two words,
    // so the next parameter was read from the wrong register.
    let generics = "a function's generics are values, as in `<const N: usize>`";
    for (src, at) in [
        (r#"pub unsafe extern "sysv64" fn k<T>(p: *const T, n: usize) {}"#, "1:33"),
        // The first generic that is no value.
        (r#"pub unsafe extern "sysv64" fn k<const N: usize, T: Copy>() {}"#, "1:49"),
        // In a parameter by value, and in a type that holds one.
        (r#"pub unsafe extern "sysv64" fn k(x: impl Copy) {}"#, "1:36"),
        (r#"pub unsafe extern "sysv64" fn k(x: Option<impl Copy>) {}"#, "1:43"),
        (r#"pub unsafe extern "sysv64" fn k(x: Option<*const impl Copy>) {}"#, "1:50"),
        (r#"pub unsafe extern "sysv64" fn k(x: [*const impl Copy; 2]) {}"#, "1:44"),
        (r#"pub unsafe extern "sysv64" fn k(x: (*const impl Copy, u8)) {}"#, "1:44"),
        // In a pointer's type.
        (r#"pub unsafe extern "sysv64" fn k(p: *const impl Copy, n: usize) { mov(qword[p], n); }"#, "1:43"),
        (r#"pub unsafe extern "sysv64" fn k(p: *mut impl Copy, n: usize) { mov(qword[p], n); }"#, "1:41"),
        // A kernel of instances, a cooperative kernel, and a helper, whose types are read as a kernel's are.
        (
            "#[instances(1, 2)]\npub unsafe extern \"sysv64\" fn k<const N: usize>(p: *const impl Copy, n: usize) {}",
            "2:59",
        ),
        ("#[instances(1, 2)]\npub unsafe extern \"sysv64\" fn k<const N: usize>(x: impl Copy) {}", "2:52"),
        ("#[instances(1, 2)]\npub unsafe extern \"sysv64\" fn k<const N: usize>(x: Option<impl Copy>) {}", "2:59"),
        ("#[cooperative]\npub unsafe extern \"sysv64\" fn k(p: *const impl Copy, n: usize) {}", "2:43"),
        ("#[cooperative]\npub unsafe extern \"sysv64\" fn k(x: impl Copy) {}", "2:36"),
        ("pub unsafe extern \"sysv64\" fn k() { h(); }\nfn h(p: *const impl Copy) {}", "2:16"),
        ("pub unsafe extern \"sysv64\" fn k() { h(); }\nfn h(x: impl Copy) {}", "2:9"),
        ("pub unsafe extern \"sysv64\" fn k() { h(); }\nfn h() -> impl Copy {}", "2:11"),
        // In the result.
        (r#"pub unsafe extern "sysv64" fn k(p: *mut u8) -> *mut impl Copy { ret(p); }"#, "1:53"),
        (r#"pub unsafe extern "sysv64" fn k(x: u64) -> impl Copy { ret(x); }"#, "1:44"),
        (r#"pub unsafe extern "sysv64" fn k(x: u64) -> Option<impl Copy> { ret(x); }"#, "1:51"),
        (r#"pub unsafe extern "sysv64" fn k(x: u64) -> Option<*const impl Copy> { ret(x); }"#, "1:58"),
        // In a type the pointer is to, however deep, and in a group of it.
        (r#"pub unsafe extern "sysv64" fn k(p: *const Option<impl Copy>, n: usize) {}"#, "1:50"),
        (r#"pub unsafe extern "sysv64" fn k(p: *const [impl Copy; 2], n: usize) {}"#, "1:44"),
        (r#"pub unsafe extern "sysv64" fn k(p: *const (u8, impl Copy), n: usize) {}"#, "1:48"),
        (r#"pub unsafe extern "sysv64" fn k(p: *const (impl ?Sized + core::fmt::Debug), n: usize) {}"#, "1:44"),
        (r#"pub unsafe extern "sysv64" fn k(p: *mut *const impl Copy, n: usize) {}"#, "1:48"),
        // Of two, the first.
        (r#"pub unsafe extern "sysv64" fn k(p: *const (impl A, impl B)) {}"#, "1:44"),
    ] {
        assert_eq!(kernite::x86_str(src, "k").unwrap_err(), format!("{at}: {generics}"), "{src}");
    }
    // A GPU kernel and a GPU function are read as an x86 kernel is.
    for (src, at) in [
        (r#"pub unsafe extern "ptx-kernel" fn g<T>(p: *mut T, n: u32) {}"#, "1:37"),
        (r#"pub unsafe extern "ptx-kernel" fn g(x: impl Copy) {}"#, "1:40"),
        (r#"pub unsafe extern "ptx-kernel" fn g(x: Option<impl Copy>) {}"#, "1:47"),
        (r#"pub unsafe extern "ptx-kernel" fn g(x: Option<*mut impl Copy>) {}"#, "1:52"),
        (r#"pub unsafe extern "ptx-kernel" fn g(x: [*mut impl Copy; 2]) {}"#, "1:46"),
        (r#"pub unsafe extern "ptx-kernel" fn g(x: (*mut impl Copy, u8)) {}"#, "1:46"),
        (r#"pub unsafe extern "ptx-kernel" fn g(p: *mut impl Copy, n: u32) {}"#, "1:45"),
        ("#[instances(1, 2)]\npub unsafe extern \"ptx-kernel\" fn g<const N: u32>(p: *mut impl Copy) {}", "2:59"),
        ("#[instances(1, 2)]\npub unsafe extern \"ptx-kernel\" fn g<const N: u32>(x: impl Copy) {}", "2:54"),
        (r#"pub unsafe extern "ptx-func" fn f(x: impl Copy) {}"#, "1:38"),
        (r#"pub unsafe extern "ptx-func" fn f(p: *mut impl Copy) {}"#, "1:43"),
        (r#"pub unsafe extern "ptx-func" fn f(p: *mut u64) -> *mut impl Copy {}"#, "1:56"),
        (r#"pub unsafe extern "ptx-func" fn f(x: u64) -> impl Copy {}"#, "1:46"),
        (r#"pub unsafe extern "ptx-func" fn f(x: u64) -> Option<impl Copy> {}"#, "1:53"),
        (r#"pub unsafe extern "ptx-func" fn f(x: u64) -> Option<*mut impl Copy> {}"#, "1:58"),
    ] {
        assert_eq!(kernite::ptx_str(src).unwrap_err(), format!("{at}: {generics}"), "{src}");
    }
    // A type named `r#impl` is a name, not the keyword.
    let src = r#"pub unsafe extern "sysv64" fn k(p: *const r#impl, n: usize) { mov(qword[p], n); }"#;
    assert!(x86(src, "k").starts_with("# k(gpr, gpr) -> ()\n"), "{src}");
}

/// The 17 types of `core::primitive`, which Rust names by a path as well as by a name.
const PRIMITIVES: [&str; 17] = [
    "bool", "char", "f32", "f64", "i8", "i16", "i32", "i64", "i128", "isize", "str", "u8", "u16", "u32", "u64", "u128",
    "usize",
];

/// The paths by which Rust names a type of `core::primitive`, with `{}` where the type's name goes: four
/// spellings, and two that write a name raw, `r#core::primitive::u64` and `::core::primitive::r#u64`.
const PATHS: [&str; 6] = [
    "core::primitive::{}",
    "::core::primitive::{}",
    "std::primitive::{}",
    "::std::primitive::{}",
    "r#core::primitive::{}",
    "::core::primitive::r#{}",
];

/// `src` with each primitive written as `path`, `u64` as `core::primitive::u64`, and `u32::MAX` as
/// `core::primitive::u32::MAX`, but where it is a part of an instruction's name, `ld.global.u32`.
fn pathed(src: &str, path: &str) -> String {
    let chars: Vec<char> = src.chars().collect();
    let in_word = |c: char| c.is_alphanumeric() || c == '_';
    let (mut out, mut at) = (String::new(), 0);
    while at < chars.len() {
        let end = (at..chars.len()).find(|&i| !in_word(chars[i])).unwrap_or(chars.len());
        if end == at {
            out.push(chars[at]);
            at += 1;
            continue;
        }
        let word: String = chars[at..end].iter().collect();
        let part = (at > 0 && chars[at - 1] == '.') || chars.get(end) == Some(&'.');
        out += &if PRIMITIVES.contains(&word.as_str()) && !part { path.replace("{}", &word) } else { word };
        at = end;
    }
    out
}

/// Rust names a primitive by a path as well as by its name, and a path is read as the name: a kernel typed by
/// `core::primitive::u64`, `::core::primitive::u64`, `std::primitive::u64`, `::std::primitive::u64`,
/// `r#core::primitive::u64` or `::core::primitive::r#u64` compiles to the listing, or the PTX, of the same kernel typed
/// by `u64`. So it is wherever a type is written: the parameters and the result of a kernel, of a pointer to a type, of
/// a kernel of instances, and of an `extern "ptx-func"` function; a compile-time argument's type, a kernel's, a
/// helper's and a function's; a variable's, alone, in an array and in memory; a const's and a static's; a helper's
/// parameter; and the type of a cast.
#[test]
fn a_primitive_written_by_its_path_is_the_primitive_written_by_its_name() {
    for (src, kernels) in [
        // A kernel's parameters and result: integers, floats, and pointers to a type, in an array and nested.
        (
            r#"pub unsafe extern "sysv64" fn k(x: u64, n: usize, m: i64, i: isize) -> u64 { add(x, n); ret(x); }"#,
            &["k"][..],
        ),
        (r#"pub unsafe extern "sysv64" fn k(a: f32, b: f64) -> f64 { ret(b); }"#, &["k"]),
        (
            r#"pub unsafe extern "sysv64" fn k(p: *const u8, q: *mut u64, r: *const *mut f32) { mov(qword[q], p); }"#,
            &["k"],
        ),
        (r#"pub unsafe extern "sysv64" fn k(p: *const [usize; 3]) -> *mut [u8; 4] { ret(p); }"#, &["k"]),
        (r#"pub unsafe extern "sysv64" fn k(p: *const (u8, u32)) -> *const (f32, *const u64) { ret(p); }"#, &["k"]),
        (r#"#[cooperative] pub unsafe extern "sysv64" fn k(p: *mut u32, n: u64) { mov(dword[p], 1); }"#, &["k"]),
        // The compile-time arguments of a kernel, of two, and of a helper.
        (
            "#[instances(1, 2)] pub unsafe extern \"sysv64\" fn k<const N: u64>(x: u64) -> u64 { add(x, N); ret(x); }",
            &["k::<1>", "k::<2>"],
        ),
        (
            "#[instances(true, false)] pub unsafe extern \"sysv64\" fn k<const B: bool>(x: u64) -> u64 {\n\
             if B { add(x, 1); }\nret(x);\n}",
            &["k::<true>", "k::<false>"],
        ),
        (
            "#[instances((1, 2))] pub unsafe extern \"sysv64\" fn k<const A: usize, const B: u8>(x: u64) -> u64 {\n\
             add(x, A);\nadd(x, B);\nret(x);\n}",
            &["k::<1, 2>"],
        ),
        (
            "fn h<const N: i32, const B: bool>(x: gpr) { add(x, N); }\n\
             #[instances(1)] pub unsafe extern \"sysv64\" fn k<const M: i32>(x: u64) -> u64 {\n\
             h::<3, true>(x);\nret(x);\n}",
            &["k::<1>"],
        ),
        // Variables: in an array, in stack memory; a helper that takes the memory.
        (
            r#"pub unsafe extern "sysv64" fn k(p: *mut u32) {
                let t: [gpr; 2];
                let m: local<[u32; 4], 16>;
                let q: local<f32>;
                let a: gpr;
                lea(a, [m]);
                mov(dword[p], 1);
            }"#,
            &["k"],
        ),
        (
            "fn h(m: local<[u32; 4]>, g: gpr) { mov(qword[m], g); }\n\
             pub unsafe extern \"sysv64\" fn k(x: u64) { let m: local<[u32; 4]>; let g: gpr; mov(g, x); h(m, g); }",
            &["k"],
        ),
        // Consts, with a cast; statics; a loop over a cast.
        (
            "const N: usize = 4; const F: f32 = 1.5; const B: bool = true; const A: [u32; 2] = [1, 2];\n\
             const M: u8 = 5 as u8; const G: f64 = 2 as f64; const T: (u32, f32) = (1, 2.0);\n\
             pub unsafe extern \"sysv64\" fn k(x: u64) -> u64 { add(x, N); add(x, A[1]); add(x, M); ret(x); }",
            &["k"],
        ),
        (
            "static S: global<[u32; 4]>; static C: constant<[f32; 2]> = [0.5, 1.5]; static G: global<u8>;\n\
             pub unsafe extern \"sysv64\" fn k() { let at: gpr; lea(at, [S]); lea(at, [C]); lea(at, [G]); }",
            &["k"],
        ),
        (r#"pub unsafe extern "sysv64" fn k(x: u64) { for i in 0..(3 as usize) { add(x, i); } }"#, &["k"]),
    ] {
        for path in PATHS {
            let written = pathed(src, path);
            assert_ne!(written, src);
            for kernel in kernels {
                assert_eq!(x86(&written, kernel), x86(src, kernel), "{written}");
            }
        }
    }
    let module = |src: &str| kernite::ptx_str(src).unwrap_or_else(|e| panic!("{e}\n{src}"));
    for src in [
        // A kernel's parameters: integers, floats and a pointer.
        "pub unsafe extern \"ptx-kernel\" fn g(\n\
         n: u32, w: u64, x: f32, d: f64, i: i32, j: i64, b: u8, c: u16, s: usize, t: isize, out: *mut u32,\n\
         ) { st.global.u32([out], n); }",
        r#"pub unsafe extern "ptx-kernel" fn g(p: *const *mut [f32; 2], q: *mut (u8, u16), out: *mut u32) {
            st.global.u32([out], 1);
        }"#,
        // An `extern "ptx-func"` function's parameters and result.
        r#"
        extern "ptx-func" fn f(a: u64, b: u32, c: f32, d: f64) -> u32 { ret(b); }
        pub unsafe extern "ptx-kernel" fn g(p: *mut u32) {
            let r: u32;
            call(r, f, (1u64, 2u32, 3.0f32, 4.0f64));
            st.global.u32([p], r);
        }
        "#,
        // The compile-time arguments of a kernel, of two, and of a function; and a helper's.
        "#[instances(4, 8)] pub unsafe extern \"ptx-kernel\" fn g<const N: u32>(p: *mut u32) {\n\
         st.global.u32([p], N);\n}",
        "#[instances((4, true))] pub unsafe extern \"ptx-kernel\" fn g<const N: u32, const B: bool>(p: *mut u32) {\n\
         st.global.u32([p], N);\n}",
        "#[instances(2)] extern \"ptx-func\" fn f<const K: usize>(a: u32) -> u32 { add.u32(a, a, K); ret(a); }\n\
         pub unsafe extern \"ptx-kernel\" fn g(p: *mut u32) {\n\
         let r: u32; call(r, f::<2>, (1u32,)); st.global.u32([p], r);\n}",
        "fn h<const K: u64>(x: u32) { add.u32(x, x, K); }\n\
         #[instances(3)] pub unsafe extern \"ptx-kernel\" fn g<const N: u8>(p: *mut u32) {\n\
         let v: u32; mov.u32(v, N); h::<7>(v); st.global.u32([p], v);\n}",
        // Variables: alone, in an array, in shared and local memory; a helper that takes them; a cast.
        r#"
        pub unsafe extern "ptx-kernel" fn g(p: *mut u32) {
            let r: [u32; 2];
            let x: f32;
            let d: f64;
            let t: shared<[f32; 8], 16>;
            let l: local<[u8; 4]>;
            let h: u8;
            let q: pred;
            mov.u32(r[0], 1);
            st.global.u32([p], r[0]);
        }
        "#,
        r#"
        fn h(x: u32, y: f32, t: shared<[u32; 4], 16>, q: pred) { add.u32(x, x, 1); }
        pub unsafe extern "ptx-kernel" fn g(p: *mut u32) {
            let v: u32;
            mov.u32(v, 1);
            let y: f32;
            mov.f32(y, 1.0);
            let t: shared<[u32; 4], 16>;
            let q: pred;
            setp.eq.u32(q, v, 1);
            h(v, y, t, q);
            st.global.u32([p], v);
        }
        "#,
        r#"pub unsafe extern "ptx-kernel" fn g(p: *mut u32) {
            let v: u32; mov.u32(v, 3 as u32);
            let w: u64; mov.u64(w, 4 as u64);
            st.global.u32([p], v);
        }"#,
        // Consts, with a cast; statics.
        "const N: usize = 4; const F: f32 = 1.5; const B: bool = true; const A: [u32; 2] = [1, 2];\n\
         const M: u8 = 5 as u8; const G: f64 = 2 as f64;\n\
         pub unsafe extern \"ptx-kernel\" fn g(p: *mut u32) { st.global.u32([p], A[1]); }",
        "static S: global<[u32; 4]>; static C: constant<[f32; 2]> = [0.5, 1.5]; static G: global<u8>;\n\
         pub unsafe extern \"ptx-kernel\" fn g(p: *mut u32) {\n\
         let at: u64; mov.u64(at, S); mov.u64(at, C); mov.u64(at, G); st.global.u32([p], 1);\n}",
    ] {
        for path in PATHS {
            let written = pathed(src, path);
            assert_ne!(written, src);
            assert_eq!(module(&written), module(src), "{written}");
        }
    }
}

/// A primitive's constants and `from_bits` are named by the path of the primitive as well as by its name, and a path is
/// read as the name: a kernel that names `core::primitive::u32::MAX`, `::core::primitive::u32::MAX`,
/// `std::primitive::u32::MAX`, `::std::primitive::u32::MAX`, `r#core::primitive::u32::MAX` or
/// `::core::primitive::r#u32::MAX` compiles to the listing, or the PTX, of the same kernel that names `u32::MAX`. So it
/// is wherever a value is known now: a const, a static's data, a kernel's operand, a compile-time argument and the
/// value an instance lists, a cast's, and the length of an array.
#[test]
fn a_constant_named_by_the_path_of_its_primitive_is_the_constant_named_by_its_name() {
    for (src, kernels) in [
        // A const, a static, and a kernel's operands: an integer, a float, a float made of bits.
        (
            "const TOP: u64 = u32::MAX as u64; const LOW: i32 = i16::MIN as i32; const BITS: u32 = u64::BITS;\n\
             const INF: f32 = f32::INFINITY; const ONE: f32 = f32::from_bits(0x3f800000);\n\
             static S: constant<[f32; 3]> = [INF, f32::NEG_INFINITY, f32::from_bits(0x40000000)];\n\
             pub unsafe extern \"sysv64\" fn k(x: u64) -> u64 {\n\
             let at: gpr;\nlea(at, [S]);\nmov(at, TOP);\nadd(x, u16::MAX);\n\
             add(x, BITS);\nadd(x, i8::MIN);\nret(x);\n}",
            &["k"][..],
        ),
        // A compile-time argument, a helper's, and the values an instance lists.
        (
            "fn h<const N: u8>(x: gpr) { add(x, N); }\n\
             #[instances(u8::MAX, u8::MIN)] pub unsafe extern \"sysv64\" fn k<const N: u8>(x: u64) -> u64 {\n\
             h::<u8::MAX>(x);\nadd(x, N);\nret(x);\n}",
            &["k::<255>", "k::<0>"],
        ),
        // A cast of a constant, a loop over a range that ends with one, and the length of an array.
        (
            "pub unsafe extern \"sysv64\" fn k(x: u64) {\n\
             add(x, u8::MAX as u64);\nfor i in 0..(u8::BITS as usize) { add(x, i); }\n\
             let m: local<[u32; u8::BITS as usize]>;\nlet a: gpr;\nlea(a, [m]);\nadd(x, a);\n}",
            &["k"],
        ),
    ] {
        for path in PATHS {
            let written = pathed(src, path);
            assert_ne!(written, src);
            for kernel in kernels {
                assert_eq!(x86(&written, kernel), x86(src, kernel), "{written}");
            }
        }
    }
    let module = |src: &str| kernite::ptx_str(src).unwrap_or_else(|e| panic!("{e}\n{src}"));
    for src in [
        "const TOP: u32 = u16::MAX as u32; const INF: f32 = f32::INFINITY;\n\
         const HALF: f64 = f64::from_bits(0x3fe0000000000000);\n\
         static C: constant<[f32; 2]> = [f32::NEG_INFINITY, f32::MAX];\n\
         #[instances(u32::MAX)] pub unsafe extern \"ptx-kernel\" fn g<const N: u32>(p: *mut u32) {\n\
         let v: u32;\nmov.u32(v, TOP);\nadd.u32(v, v, u32::MAX);\nadd.u32(v, v, N);\nadd.u32(v, v, u8::BITS);\n\
         let x: f32;\nmov.f32(x, INF);\nmov.f32(x, f32::MIN_POSITIVE);\nmov.f32(x, f32::from_bits(0x3f800000));\n\
         let d: f64;\nmov.f64(d, HALF);\nmov.f64(d, f64::MIN);\nmov.f64(d, f64::from_bits(0x3ff0000000000000));\n\
         st.global.u32([p], v);\n}",
        "fn h<const K: u64>(x: u64) { add.u64(x, x, K); }\n\
         extern \"ptx-func\" fn f(a: u32) -> u32 { add.u32(a, a, i32::MIN as u32); ret(a); }\n\
         pub unsafe extern \"ptx-kernel\" fn g(p: *mut u32) {\n\
         let w: u64;\nmov.u64(w, u64::MAX);\nh::<u64::MAX>(w);\n\
         let r: u32;\ncall(r, f, (usize::BITS,));\nst.global.u32([p], r);\n\
         let t: [u32; u8::BITS as usize / 4];\n}",
    ] {
        for path in PATHS {
            let written = pathed(src, path);
            assert_ne!(written, src);
            assert_eq!(module(&written), module(src), "{written}");
        }
    }
}

/// The constants of `core::f32::consts` and `core::f64::consts` that Rust has stable, by name, each with Rust's own
/// value as the bits of an `f32` and of an `f64`.
macro_rules! rust_consts {
    ($($name:ident),*) => {
        [$((stringify!($name), std::f32::consts::$name.to_bits(), std::f64::consts::$name.to_bits())),*]
    };
}

const RUST_CONSTS: [(&str, u32, u64); 21] = rust_consts! {
    PI, TAU, E, EULER_GAMMA, GOLDEN_RATIO, FRAC_PI_2, FRAC_PI_3, FRAC_PI_4, FRAC_PI_6, FRAC_PI_8, FRAC_1_PI, FRAC_2_PI,
    FRAC_2_SQRT_PI, SQRT_2, FRAC_1_SQRT_2, LN_2, LN_10, LOG2_E, LOG10_E, LOG2_10, LOG10_2
};

/// The six constants of the two modules that Rust has unstable, `more_float_constants`, which stable Rust cannot name
/// to take: the bits of an `f32` and of an `f64` that Rust's nightly gives each.
const UNSTABLE_CONSTS: [(&str, u32, u64); 6] = [
    ("SQRT_3", 0x3fddb3d7, 0x3ffbb67ae8584caa),
    ("SQRT_5", 0x400f1bbd, 0x4001e3779b97f4a8),
    ("FRAC_1_SQRT_2PI", 0x3ecc422a, 0x3fd9884533d43651),
    ("FRAC_1_SQRT_3", 0x3f13cd3a, 0x3fe279a74590331c),
    ("FRAC_1_SQRT_5", 0x3ee4f92e, 0x3fdc9f25c5bfedd9),
    ("FRAC_1_SQRT_PI", 0x3f106ebb, 0x3fe20dd750429b6d),
];

/// The paths by which Rust names a module of `core` or `std`, `core::f32::consts`: four spellings, one that writes a
/// name raw, and one with a space after the `::`. A path is a root and the rest, `f32::consts::PI`.
const ROOTS: [&str; 6] = ["core::", "::core::", "std::", "::std::", "::r#std::", ":: core::"];

/// What the kernels over the operands `single` and `double`, an `f32` and an `f64`, compile to: for x86 each loads its
/// operand from memory, as a table of constants in the listing; for a GPU one moves both into registers.
fn compiled(single: &str, double: &str) -> (String, String, String) {
    let x86s = format!(r#"unsafe extern "sysv64" fn s() {{ let v: ymm; vmovups(v, [[{single}; 8]]); }}"#);
    let x86d = format!(r#"unsafe extern "sysv64" fn d() {{ let v: ymm; vmovupd(v, [[{double}; 4]]); }}"#);
    let ptx_src = format!(
        r#"unsafe extern "ptx-kernel" fn g(a: *mut f32, b: *mut f64) {{
            let (x, y): (f32, f64);
            mov.f32(x, {single});
            mov.f64(y, {double});
            st.global.f32([a], x);
            st.global.f64([b], y);
        }}"#
    );
    (x86(&x86s, "s"), x86(&x86d, "d"), ptx(&ptx_src))
}

/// A constant of `core::f32::consts` or `core::f64::consts` is named by its path from the root of `core` or `std`,
/// `std::f32::consts::LOG2_E`, with a `::` before it or not, with a name raw or a space after the `::`, and has Rust's
/// own value, bit for bit: a kernel that names it compiles to the listing, or the PTX, of the kernel that makes the
/// same bits, `f32::from_bits(0x3fb8aa3b)`, for x86 and for a GPU. So is each of the 21 that Rust has stable, whose
/// values are Rust's own constants, and each of the six that it has unstable, whose values are the bits its nightly
/// gives; of `f32` and of `f64`.
#[test]
fn a_constant_of_core_consts_is_rusts_own_by_a_path_from_its_root() {
    for (name, single, double) in RUST_CONSTS.into_iter().chain(UNSTABLE_CONSTS) {
        let bits = (format!("f32::from_bits({single:#x})"), format!("f64::from_bits({double:#x})"));
        let want = compiled(&bits.0, &bits.1);
        for root in ROOTS {
            let named = compiled(&format!("{root}f32::consts::{name}"), &format!("{root}f64::consts::{name}"));
            assert_eq!(named, want, "{root}{name}");
        }
        // A name written raw is the name Rust reads, at each place it is written.
        let raw = compiled(&format!("::r#core::r#f32::r#consts::r#{name}"), &format!("std::r#f64::consts::r#{name}"));
        assert_eq!(raw, want, "r#{name}");
    }
}

/// An integer's constants, `MIN`, `MAX` and `BITS`, are Rust's own for each of the ten integers kernite has, `u8` to
/// `u64`, `i8` to `i64`, `usize` and `isize`: a const that holds each, or a note that says it, has Rust's value, and
/// `BITS` is a `u32` whatever the integer is.
#[test]
fn an_integer_s_constants_are_rusts_own() {
    macro_rules! own {
        ($($ty:ident),*) => {
            [$((stringify!($ty), $ty::MIN.to_string(), $ty::MAX.to_string(), $ty::BITS.to_string())),*]
        };
    }
    for (ty, min, max, bits) in own!(u8, u16, u32, u64, usize, i8, i16, i32, i64, isize) {
        let src = format!(
            "const LOW: {ty} = {ty}::MIN; const HIGH: {ty} = {ty}::MAX; const WIDTH: u32 = {ty}::BITS;\n\
             pub unsafe extern \"sysv64\" fn k() {{ note(\"{{LOW}} {{HIGH}} {{WIDTH}}\"); }}"
        );
        let listing = x86(&src, "k");
        assert!(listing.contains(&format!("# {min} {max} {bits}\n")), "{ty}: {listing}");
        // The value is the integer's own type, so that a use of another is refused, `u64::MAX` as a `u32`.
        let wrong =
            format!("const WIDTH: u8 = {ty}::BITS;\npub unsafe extern \"sysv64\" fn k() {{ note(\"{{WIDTH}}\"); }}");
        assert_eq!(kernite::x86_str(&wrong, "k").unwrap_err(), "1:19: this is a u32, not u8", "{ty}");
    }
}

/// A float's own constants are named by the float's path: `INFINITY`, `NEG_INFINITY`, `MAX`, `MIN`, `MIN_POSITIVE` and
/// `EPSILON` of `f32` and of `f64` have Rust's own value, bit for bit. `f16` and `bf16` have the same six, with the
/// bits that IEEE 754's rules give a float of their widths, 5 bits of exponent and 10 of fraction, and 8 and 7, for
/// Rust has no such floats to take them from. The primitive's path names them too, `core::primitive::f64::EPSILON`.
#[test]
fn a_float_s_own_constants_are_rusts_own_by_the_floats_path() {
    macro_rules! own {
        ($($name:ident),*) => {
            [$((stringify!($name), f32::$name.to_bits(), f64::$name.to_bits())),*]
        };
    }
    let floats: [(&str, u32, u64); 6] = own!(INFINITY, NEG_INFINITY, MAX, MIN, MIN_POSITIVE, EPSILON);
    for (name, single, double) in floats {
        let bits = (format!("f32::from_bits({single:#x})"), format!("f64::from_bits({double:#x})"));
        let want = compiled(&bits.0, &bits.1);
        for path in PATHS {
            let (single, double) = (path.replace("{}", "f32"), path.replace("{}", "f64"));
            assert_eq!(compiled(&format!("{single}::{name}"), &format!("{double}::{name}")), want, "{path}::{name}");
        }
        assert_eq!(compiled(&format!("f32::{name}"), &format!("f64::{name}")), want, "{name}");
    }
    // The bits of the halves: a lane of a vector for x86, a register for a GPU.
    let halves = [
        ("INFINITY", 0x7c00, 0x7f80),
        ("NEG_INFINITY", 0xfc00, 0xff80),
        ("MAX", 0x7bff, 0x7f7f),
        ("MIN", 0xfbff, 0xff7f),
        ("MIN_POSITIVE", 0x0400, 0x0080),
        ("EPSILON", 0x1400, 0x3c00),
    ];
    let halved = |half: &str, brain: &str| {
        let x86_src = format!(
            r#"unsafe extern "sysv64" fn k() {{ let v: ymm; vpbroadcastw(v, [{half}]); vpbroadcastw(v, [{brain}]); }}"#
        );
        let ptx_src = format!(
            r#"unsafe extern "ptx-kernel" fn g(a: *mut f16, b: *mut bf16) {{
                let (x, y): (f16, bf16);
                mov.b16(x, {half});
                mov.b16(y, {brain});
                st.global.b16([a], x);
                st.global.b16([b], y);
            }}"#
        );
        (x86(&x86_src, "k"), ptx(&ptx_src))
    };
    for (name, half, brain) in halves {
        let want = halved(&format!("f16::from_bits({half:#x})"), &format!("bf16::from_bits({brain:#x})"));
        assert_eq!(halved(&format!("f16::{name}"), &format!("bf16::{name}")), want, "{name}");
    }
}

/// Why a constant's path is refused in text, which no Rust reads: the path as it is written, and what kernite says of
/// it at its first token.
fn refused_constants() -> Vec<(String, String)> {
    let unknown = |p: &str| (p.to_string(), format!("`{p}` is not a constant kernite knows"));
    let mut rows = vec![];
    // A path without its root is Rust's after a `use std::f32;`, which kernite does not read.
    for read in ["f32::consts::LOG2_E", "f64::consts::PI", "f64::consts::FRAC_2_SQRT_PI"] {
        let ty = read.split("::").next().unwrap();
        let error = format!(
            "`{read}` is Rust's only after `use std::{ty};`, which kernite does not read: write `core::{read}`"
        );
        rows.push((read.to_string(), error));
    }
    // Types that have no `consts`, names that the modules have not, and paths of another shape: through a primitive to
    // a module, which is no path of Rust's, `core::primitive::f32::consts::PI`, as it is written.
    for p in [
        "core::primitive::f32::consts::PI",
        "std::f16::consts::PI",
        "core::f128::consts::PI",
        "std::bf16::consts::E",
        "f16::consts::PI",
        "bf16::consts::E",
        "std::u32::consts::MAX",
        "std::f32::consts::NOPE",
        "std::f64::consts::pi",
        "std::f32::consts::Pi",
        "core::f32::consts::PI_2",
        "std::f32::consts",
        "std::f32::consts::PI::X",
        "std::consts::PI",
        "std::f32::PI",
        "consts::PI",
        "my::f32::consts::PI",
        "crate::std::f32::consts::PI",
        "self::f32::consts::PI",
        "::std::f32::consts::PI::X",
        "::core::f32::consts",
    ] {
        rows.push(unknown(p));
    }
    // A name written raw is the name, in a path as anywhere: it is quoted so.
    rows.push((
        "::std::primitive::r#f64::consts::E".to_string(),
        "`::std::primitive::f64::consts::E` is not a constant kernite knows".to_string(),
    ));
    // A type's constant that Rust has by another path or name, or has not: the modules' constants, which Rust has
    // deprecated, `std::u32::MAX`; the floats' other constants; a name that no type has.
    for p in [
        "std::u32::MAX",
        "core::f32::MAX",
        "std::f64::EPSILON",
        "core::i8::MIN",
        "::std::f32::INFINITY",
        "f32::NAN",
        "f64::NAN",
        "f32::RADIX",
        "f64::DIGITS",
        "f32::MANTISSA_DIGITS",
        "f64::MAX_EXP",
        "u8::NOPE",
        "u128::MAX",
        "i128::MIN",
        "u128::BITS",
        "char::MAX",
        "f32::consts",
        "core::MAX",
    ] {
        rows.push(unknown(p));
    }
    rows
}

/// A path that names no constant kernite has a value for is refused where it is written, at its first token, in
/// kernite's words, for x86 and for a GPU, wherever a value is known now: an operand, a const, a static's data and a
/// helper. In text no Rust reads the path, so kernite says why: a constant of `core::f32::consts` or
/// `core::f64::consts` without `std::` or `core::` before it needs one, as kernite reads no `use`; and every other
/// path is no constant kernite knows: a name or a type the modules have not, a path of another shape, which includes
/// one through a primitive to a module, `core::primitive::f32::consts::PI`, quoted as it is written, a constant that
/// Rust has deprecated in a module of its type, `std::u32::MAX`, or has not, `f32::NAN`, `f32::RADIX` or `f64::DIGITS`.
/// A path with a `::` before another name than `core` or `std` is no expression.
#[test]
fn a_path_that_names_no_constant_kernite_knows_is_refused_at_its_first_token_in_text() {
    let at = |src: &str, path: &str| 1 + src.find(path).unwrap_or_else(|| panic!("`{path}` in {src}"));
    for (path, error) in refused_constants() {
        for src in [
            format!(r#"unsafe extern "sysv64" fn k() {{ let g: gpr; mov(g, {path}); }}"#),
            format!("const C: f32 = {path};\nunsafe extern \"sysv64\" fn k() {{ let g: gpr; mov(g, C); }}"),
            format!(
                "static S: constant<[f32; 1]> = [{path}];\n\
                 unsafe extern \"sysv64\" fn k() {{ let g: gpr; lea(g, [S]); }}"
            ),
            format!("fn h() {{ let g: gpr; mov(g, {path}); }}\nunsafe extern \"sysv64\" fn k() {{ h(); }}"),
        ] {
            let want = format!("{}:{}: {error}", 1, at(&src, &path));
            assert_eq!(kernite::x86_str(&src, "k").unwrap_err(), want, "{src}");
        }
        let src = format!(r#"unsafe extern "ptx-kernel" fn g() {{ let v: f32; mov.f32(v, {path}); }}"#);
        let want = format!("1:{}: {error}", at(&src, &path));
        assert_eq!(kernite::ptx_str(&src).unwrap_err(), want, "{src}");
    }
    // The `::` before a name other than `core` or `std` begins no path.
    for path in ["::f32::consts::PI", "::my::f32::consts::PI", "::u32::MAX"] {
        let src = format!(r#"unsafe extern "sysv64" fn k() {{ let g: gpr; mov(g, {path}); }}"#);
        assert_eq!(kernite::x86_str(&src, "k").unwrap_err(), "1:52: expected an expression, found `:`", "{path}");
        let src = format!(r#"unsafe extern "ptx-kernel" fn g() {{ let v: f32; mov.f32(v, {path}); }}"#);
        assert_eq!(kernite::ptx_str(&src).unwrap_err(), "1:60: expected an expression, found `:`", "{path}");
    }
}

/// Each place a type is written, with `{ty}` where it is, and for x86 the kernel to compile: a kernel's parameter,
/// pointee and result, a variable in an array and in stack memory, a compile-time argument's type, a helper's
/// parameter and result, a static, and a cast.
const X86_PLACES: [(&str, &str); 14] = [
    (r#"unsafe extern "sysv64" fn k(x: {ty}) {}"#, "k"),
    (r#"unsafe extern "sysv64" fn k(p: *const {ty}) {}"#, "k"),
    (r#"unsafe extern "sysv64" fn k() -> {ty} {}"#, "k"),
    ("#[cooperative]\nunsafe extern \"sysv64\" fn k() -> {ty} {}", "k"),
    (r#"unsafe extern "sysv64" fn k() { let v: [{ty}; 2]; }"#, "k"),
    (r#"unsafe extern "sysv64" fn k() { let v: local<{ty}>; }"#, "k"),
    ("#[instances(1)]\nunsafe extern \"sysv64\" fn k<const M: {ty}>() {}", "k::<1>"),
    ("fn h<const M: {ty}>() {}\nunsafe extern \"sysv64\" fn k() { h::<1>(); }", "k"),
    ("fn h() -> {ty} {}\nunsafe extern \"sysv64\" fn k() { h(); }", "k"),
    ("fn h(x: {ty}) {}\nunsafe extern \"sysv64\" fn k() { let v: gpr; h(v); }", "k"),
    ("static S: global<{ty}>;\nunsafe extern \"sysv64\" fn k() {}", "k"),
    ("static S: constant<[{ty}; 2]> =\n1;\nunsafe extern \"sysv64\" fn k() {}", "k"),
    ("const C: {ty} = 1;\nunsafe extern \"sysv64\" fn k() {}", "k"),
    (r#"unsafe extern "sysv64" fn k() { let v = 1 as {ty}; }"#, "k"),
];

/// Each place a type is written, with `{ty}` where it is, for a GPU: the same, and an `extern "ptx-func"` function's
/// parameter, result and compile-time argument, and memory that is shared.
const GPU_PLACES: [&str; 17] = [
    r#"unsafe extern "ptx-kernel" fn g(x: {ty}) {}"#,
    r#"unsafe extern "ptx-kernel" fn g(p: *mut {ty}) {}"#,
    r#"unsafe extern "ptx-kernel" fn g() -> {ty} {}"#,
    r#"unsafe extern "ptx-kernel" fn g() { let v: [{ty}; 2]; }"#,
    r#"unsafe extern "ptx-kernel" fn g() { let v: shared<{ty}>; }"#,
    r#"unsafe extern "ptx-kernel" fn g() { let v: local<{ty}>; }"#,
    "#[instances(1)]\nunsafe extern \"ptx-kernel\" fn g<const M: {ty}>() {}",
    r#"extern "ptx-func" fn f(a: {ty}) {}"#,
    r#"extern "ptx-func" fn f() -> {ty} {}"#,
    "#[instances(1)]\nextern \"ptx-func\" fn f<const M: {ty}>() {}",
    "fn h<const M: {ty}>() {}\nunsafe extern \"ptx-kernel\" fn g() { h::<1>(); }",
    "fn h() -> {ty} {}\nunsafe extern \"ptx-kernel\" fn g() { h(); }",
    "fn h(x: {ty}) {}\nunsafe extern \"ptx-kernel\" fn g() { let v: u32; h(v); }",
    "static S: global<{ty}>;\nunsafe extern \"ptx-kernel\" fn g() {}",
    "static S: constant<[{ty}; 2]> =\n[1, 2];\nunsafe extern \"ptx-kernel\" fn g() {}",
    "const C: {ty} = 1;\nunsafe extern \"ptx-kernel\" fn g() {}",
    r#"unsafe extern "ptx-kernel" fn g() { let v = 1 as {ty}; }"#,
];

/// What every place says of `ty`: the listing, or the PTX, or why not.
fn said(ty: &str) -> Vec<Result<String, String>> {
    let x86s =
        X86_PLACES.iter().map(|(src, kernel)| kernite::x86_str(&src.replace("{ty}", ty), kernel).map(|l| l.text()));
    let ptxs = GPU_PLACES.iter().map(|src| kernite::ptx_str(&src.replace("{ty}", ty)));
    x86s.chain(ptxs).collect()
}

/// A type that a place does not take is refused where the type is written, in kernite's words, and is the same type
/// whether Rust's path names it or not: `core::primitive::str` is refused as `str` is, and so is every primitive in
/// every place, in all six spellings. kernite reads no other path as a primitive. A place takes it or refuses it as
/// it does `Word`, and quotes it as written: a path into the crate, a path that is only a part of the primitive's,
/// `core::primitive::u64` after another name, or before another, `core::primitive::u64::MAX`; and one to a type that
/// `core::primitive` has not, `core::primitive::f16`.
#[test]
fn a_path_is_refused_as_the_type_it_names_is_where_the_place_does_not_take_it() {
    for name in PRIMITIVES {
        let by_name = said(name);
        assert!(by_name.iter().any(Result::is_err), "{name}");
        for path in PATHS {
            let written = path.replace("{}", name);
            assert_eq!(said(&written), by_name, "{written}");
        }
    }
    // `Word` is a type of the program's own: every place that is not a pointer or a const says so, naming it.
    let word = said("Word");
    assert!(word.iter().filter(|r| r.is_err()).count() > 20);
    for path in [
        "crate::Word",
        "self::Word",
        "my::core::primitive::u64",
        "crate::core::primitive::u64",
        "core::primitive::u64::MAX",
        "core::primitive::f16",
        "core::primitive::f128",
        "core::primitive::bf16",
        "core::primitive::s32",
        "core::primitive",
        "core::u64",
    ] {
        let by_name: Vec<_> = word
            .iter()
            .map(|r| r.as_ref().map(|ok| ok.replace("Word", path)).map_err(|e| e.replace("Word", path)))
            .collect();
        assert_eq!(said(path), by_name, "{path}");
    }
}

/// The error for a primitive written by a path names it by its name, at the type, and the error for any other path
/// names the path as written. So does the error for a constant of a primitive: `core::primitive::u32::NOPE` is
/// `u32::NOPE` in it, at the path's first token, and `my::core::primitive::u32::MAX` is as written. A name written
/// raw is the name Rust reads in it, as in any path, but for a keyword, `r#mut`, which stays as written, in a
/// compile-time argument's type and a cast's too.
#[test]
fn an_error_names_a_primitive_by_its_name_and_any_other_path_as_written() {
    let takes = "a kernel takes pointers, 64-bit integers and floats, not";
    let returns = "a kernel returns a float, an integer or a pointer, or never, `!`; not";
    let compile_time = "a function that names an ABI takes integers and `bool`s at compile time, as Rust's const \
                        generics are; ";
    let variables = "x86 variables are `gpr`, `ymm`, `xmm` and stack memory, `local<..>`; not";
    let statics = "a static holds Rust's numbers, `u8` to `u64`, `i8` to `i64`, `usize`, `isize`, `f32` and `f64`; not";
    for (src, kernel, error) in [
        // A parameter, a pointer to a type, a result, a compile-time argument, and a variable.
        (r#"unsafe extern "sysv64" fn k(x: core::primitive::str) {}"#, "k", format!("1:32: {takes} `str`")),
        (r#"unsafe extern "sysv64" fn k(x: ::std::primitive::char) {}"#, "k", format!("1:32: {takes} `char`")),
        (r#"unsafe extern "sysv64" fn k(x: r#core::primitive::str) {}"#, "k", format!("1:32: {takes} `str`")),
        (r#"unsafe extern "sysv64" fn k(x: ::core::primitive::r#char) {}"#, "k", format!("1:32: {takes} `char`")),
        (
            r#"unsafe extern "sysv64" fn k(x: my::r#core::primitive::u64) {}"#,
            "k",
            format!("1:32: {takes} `my::core::primitive::u64`"),
        ),
        (r#"unsafe extern "sysv64" fn k(x: crate::Word) {}"#, "k", format!("1:32: {takes} `crate::Word`")),
        (
            r#"unsafe extern "sysv64" fn k(x: core::primitive::f16) {}"#,
            "k",
            format!("1:32: {takes} `core::primitive::f16`"),
        ),
        (
            r#"unsafe extern "sysv64" fn k(x: my::core::primitive::u64) {}"#,
            "k",
            format!("1:32: {takes} `my::core::primitive::u64`"),
        ),
        (
            r#"unsafe extern "sysv64" fn k(x: core::primitive::u64::MAX) {}"#,
            "k",
            format!("1:32: {takes} `core::primitive::u64::MAX`"),
        ),
        (
            r#"unsafe extern "sysv64" fn k(p: *const core::primitive::str) {}"#,
            "k",
            "1:32: a pointer to `str` has two words, an address and a length or a table, and a kernel's pointer has \
             one"
            .into(),
        ),
        (r#"unsafe extern "sysv64" fn k() -> ::core::primitive::i128 {}"#, "k", format!("1:34: {returns} `i128`")),
        (
            "#[instances(1)]\nunsafe extern \"sysv64\" fn k<const M: core::primitive::f32>() {}",
            "k::<1>",
            format!("2:35: {compile_time}`f32` is neither"),
        ),
        (
            "#[instances(1)]\nunsafe extern \"sysv64\" fn k<const M: crate::Word>() {}",
            "k::<1>",
            format!("2:35: {compile_time}`crate::Word` is neither"),
        ),
        (
            "#[instances(1)]\nunsafe extern \"sysv64\" fn k<const M: r#mut>() {}",
            "k::<1>",
            format!("2:35: {compile_time}`r#mut` is neither"),
        ),
        (
            "fn h<const M: ::std::primitive::str>() {}\nunsafe extern \"sysv64\" fn k() { h::<1>(); }",
            "k",
            "2:37: this is an integer, not str".into(),
        ),
        (
            r#"unsafe extern "sysv64" fn k() { let v: [std::primitive::u8; 2]; }"#,
            "k",
            format!("1:41: {variables} `u8`"),
        ),
        // A static, and a cast.
        (
            "static S: global<::core::primitive::str>;\nunsafe extern \"sysv64\" fn k() {}",
            "k",
            format!("1:11: {statics} `str`"),
        ),
        (
            r#"unsafe extern "sysv64" fn k() { let v = 1 as core::primitive::str; }"#,
            "k",
            "1:43: an integer cannot be cast to str".into(),
        ),
        (
            r#"unsafe extern "sysv64" fn k() { let v = 1 as r#mut; }"#,
            "k",
            "1:43: an integer cannot be cast to r#mut".into(),
        ),
        // A constant named by the path of its primitive is the primitive's name in the error, where the constant is
        // none, at the path's first token, `core` or the `::` before it; any other path is as written.
        (
            r#"unsafe extern "sysv64" fn k() { let g: gpr; mov(g, core::primitive::u32::NOPE); }"#,
            "k",
            "1:52: `u32::NOPE` is not a constant kernite knows".into(),
        ),
        (
            r#"unsafe extern "sysv64" fn k() { let g: gpr; mov(g, ::std::primitive::f64::NOPE); }"#,
            "k",
            "1:52: `f64::NOPE` is not a constant kernite knows".into(),
        ),
        (
            "const C: u64 = ::std::primitive::u64::NOPE;\nunsafe extern \"sysv64\" fn k() { let g: gpr; mov(g, C); }",
            "k",
            "1:16: `u64::NOPE` is not a constant kernite knows".into(),
        ),
        (
            r#"unsafe extern "sysv64" fn k() { let g: gpr; mov(g, core::primitive::f16::MAX); }"#,
            "k",
            "1:52: `core::primitive::f16::MAX` is not a constant kernite knows".into(),
        ),
        (
            r#"unsafe extern "sysv64" fn k() { let g: gpr; mov(g, my::core::primitive::u32::MAX); }"#,
            "k",
            "1:52: `my::core::primitive::u32::MAX` is not a constant kernite knows".into(),
        ),
        (
            r#"unsafe extern "sysv64" fn k() { let g: gpr; mov(g, ::my::u32::MAX); }"#,
            "k",
            "1:52: expected an expression, found `:`".into(),
        ),
        (
            r#"unsafe extern "sysv64" fn k() { let g: gpr; core::primitive::u32::MAX(g); }"#,
            "k",
            "1:45: `u32::MAX` names nothing kernite calls".into(),
        ),
        (
            r#"unsafe extern "sysv64" fn k() { let g: gpr; mov(g, core::primitive::u32::from_bits(1)); }"#,
            "k",
            "1:52: `from_bits` makes an f32, f64, f16 or bf16 of an integer".into(),
        ),
    ] {
        assert_eq!(kernite::x86_str(src, kernel).unwrap_err(), error, "{src}");
    }
    let takes = "a GPU kernel takes pointers, integers and the floats `f16`, `bf16`, `f32` and `f64`, not";
    let memory = "memory holds `b8` to `b128`, `u8` to `u64`, `s8` to `s64`, `f16`, `f16x2`, `bf16`, `bf16x2`, `tf32`, \
                  `f32` and `f64`; not";
    let function = "a function takes and returns PTX's types, as its variables are: `u32`, `f32`, `b64`; not";
    for (src, error) in [
        (r#"unsafe extern "ptx-kernel" fn g(x: core::primitive::str) {}"#, format!("1:36: {takes} `str`")),
        (r#"unsafe extern "ptx-kernel" fn g(x: crate::Word) {}"#, format!("1:36: {takes} `crate::Word`")),
        (
            "#[instances(1)]\nunsafe extern \"ptx-kernel\" fn g<const M: core::primitive::f32>() {}",
            format!("2:39: {compile_time}`f32` is neither"),
        ),
        (
            r#"unsafe extern "ptx-kernel" fn g() { let v: shared<std::primitive::str>; }"#,
            format!("1:44: {memory} `str`"),
        ),
        (
            r#"extern "ptx-func" fn f(a: ::core::primitive::i64) {}"#,
            "1:27: a function takes and returns PTX's types, PTX writes it `s64`; not `i64`".into(),
        ),
        (r#"extern "ptx-func" fn f() -> std::primitive::str {}"#, format!("1:29: {function} `str`")),
        (
            "static S: constant<[core::primitive::str; 2]> = [1, 2];\nunsafe extern \"ptx-kernel\" fn g() {}",
            format!("1:11: {memory} `str`"),
        ),
        (
            r#"unsafe extern "ptx-kernel" fn g() { let v: u32; mov.u32(v, core::primitive::u32::NOPE); }"#,
            "1:60: `u32::NOPE` is not a constant kernite knows".into(),
        ),
        (
            r#"unsafe extern "ptx-kernel" fn g() { let v: f32; mov.f32(v, ::std::primitive::f32::NOPE); }"#,
            "1:60: `f32::NOPE` is not a constant kernite knows".into(),
        ),
        (
            r#"unsafe extern "ptx-kernel" fn g() { let v: u32; mov.u32(v, ::core::primitive::f16::MAX); }"#,
            "1:60: `::core::primitive::f16::MAX` is not a constant kernite knows".into(),
        ),
    ] {
        assert_eq!(kernite::ptx_str(src).unwrap_err(), error, "{src}");
    }
}

/// A result is a type, and is read whole as a parameter's is, up to the body or the `where` or `;` before it: not
/// up to a block among the arguments of a name in it, `Pair<u8, { 3 }>`, which is a const argument. A kernel
/// gives back a pointer to such a type, plain and of compile-time arguments, whatever else the type holds. A
/// result that is no number or pointer, and one that a cooperative kernel or a GPU kernel may not have, is
/// refused: every error is at the result, and the x86 kernel's and the GPU kernel's quote the type whole.
#[test]
fn a_result_that_holds_a_block_is_read_whole_up_to_its_body() {
    for to in [
        "Arr<{ 3 }>",
        "crate::Arr<{ 3 }>",
        "Pair<u8, { 3 }>",
        "Pair<{ 3 }, u8>",
        "Pair<{ 3 }, { 4 }>",
        "Wrap<Arr<{ let _ = 1; 3 }>>",
        "[Arr<{ 3 }>; 1]",
        "fn() -> Arr<{ 3 }>",
        "<A as B>::C<{ 3 }>",
    ] {
        let src = format!("unsafe extern \"sysv64\" fn k(p: *const u8) -> *const {to} {{ ret(p); }}");
        assert_eq!(code(&src, "k"), ["mov rax, rdi", "ret"], "{to}");
    }
    let src = "#[instances(1, 2)]\nunsafe extern \"sysv64\" fn k<const N: u64>(p: *const u8) -> *const Arr<{ 3 }> \
               { add(p, N); ret(p); }";
    assert_eq!(code(src, "k::<1>"), ["add rdi, 1", "mov rax, rdi", "ret"]);
    assert_eq!(code(src, "k::<2>"), ["add rdi, 2", "mov rax, rdi", "ret"]);
    // Every error is at the result, and the x86 kernel's and the GPU kernel's quote the type whole: a kernel
    // returns a pointer or a number, a cooperative kernel's threads return nothing, and so does a GPU kernel.
    let src = "unsafe extern \"sysv64\" fn k() -> Arr<{ 3 }> {}";
    let returns = "a kernel returns a float, an integer or a pointer, or never, `!`; not `Arr<{ 3 }>`";
    assert_eq!(kernite::x86_str(src, "k").unwrap_err(), format!("1:34: {returns}"));
    let src = "#[cooperative]\nunsafe extern \"sysv64\" fn k() -> *const Arr<{ 3 }> {}";
    let nothing = "a cooperative kernel's threads return nothing: they write what they compute to memory";
    assert_eq!(kernite::x86_str(src, "k").unwrap_err(), format!("2:34: {nothing}"));
    let src = "unsafe extern \"ptx-kernel\" fn k() -> *const Arr<{ 3 }> {}";
    let nothing = "a GPU kernel returns nothing, not `*const Arr<{ 3 }>`: it writes what it works out to memory";
    assert_eq!(kernite::ptx_str(src).unwrap_err(), format!("1:38: {nothing}"));
    // The result still ends at its `where`, which a kernel has none of, and at the `;` of a function with no
    // body, with a block in it or without.
    let no_body = "a kernel is written in its body, and `k` has none";
    for (result, at, message) in [
        ("u8 where u8: Sized {}", "1:37", "a kernel has no `where`"),
        ("*const Arr<{ 3 }> where u8: Sized {}", "1:52", "a kernel has no `where`"),
        ("u8;", "1:36", no_body),
        ("*const Arr<{ 3 }>;", "1:51", no_body),
    ] {
        let src = format!("unsafe extern \"sysv64\" fn k() -> {result}");
        assert_eq!(kernite::x86_str(&src, "k").unwrap_err(), format!("{at}: {message}"), "{result}");
    }
}

/// A result that is a type macro in braces, `byte!{ _ }`, is read whole, as one in parentheses is: the braces right
/// after the `!` of a macro's name are its argument, and the body is the braces after them. The `!` of never follows
/// the arrow, a keyword or a lifetime, `-> !`, `*const !` or `&'static !`, and is no macro's: the braces after it are
/// the body. The lexer of a string refuses a lifetime, so that one is tried where its tokens can be made, in
/// kernite-lang's tests and kernite-macros's. A kernel gives back a pointer to such a type, plain and of compile-time
/// arguments. A result that is no number or pointer, and one that a cooperative kernel or a GPU kernel may not have,
/// is refused at the result, as it is with any type.
#[test]
fn a_result_that_holds_a_macro_in_braces_is_read_whole_up_to_its_body() {
    for to in [
        "byte!{ _ }",
        "crate::byte!{ _ }",
        "fn(u8) -> byte!{ _ }",
        // A macro's argument in parentheses or brackets, and its braces in angles or brackets, are not the body.
        "byte!(_)",
        "byte![_]",
        "Wrap<byte!{ _ }>",
        "[byte!{ _ }; 1]",
    ] {
        let src = format!("unsafe extern \"sysv64\" fn k(p: *const u8) -> *const {to} {{ ret(p); }}");
        assert_eq!(code(&src, "k"), ["mov rax, rdi", "ret"], "{to}");
    }
    // The `!` of never follows the arrow, `-> !`, a keyword, `*const !`, or a lifetime, `&'static !`, and is no macro's:
    // the braces after it are the body. A string holds no lifetime, so the first two are tried here.
    assert_eq!(code("unsafe extern \"sysv64\" fn k() -> ! { ud2(); }", "k"), ["ud2"]);
    for to in ["*const fn() -> !", "*const !", "*mut !"] {
        let src = format!("unsafe extern \"sysv64\" fn k(p: *const u8) -> {to} {{ ret(p); }}");
        assert_eq!(code(&src, "k"), ["mov rax, rdi", "ret"], "{to}");
    }
    let src = "#[instances(1, 2)]\nunsafe extern \"sysv64\" fn k<const N: u64>(p: *const u8) -> *const byte!{ _ } \
               { add(p, N); ret(p); }";
    assert_eq!(code(src, "k::<1>"), ["add rdi, 1", "mov rax, rdi", "ret"]);
    assert_eq!(code(src, "k::<2>"), ["add rdi, 2", "mov rax, rdi", "ret"]);
    // Every error is at the result, and the x86 kernel's and the GPU kernel's quote the type whole.
    let src = "unsafe extern \"sysv64\" fn k() -> byte!{ _ } {}";
    let returns = "a kernel returns a float, an integer or a pointer, or never, `!`; not `byte! { _ }`";
    assert_eq!(kernite::x86_str(src, "k").unwrap_err(), format!("1:34: {returns}"));
    let src = "#[cooperative]\nunsafe extern \"sysv64\" fn k() -> *const byte!{ _ } {}";
    let nothing = "a cooperative kernel's threads return nothing: they write what they compute to memory";
    assert_eq!(kernite::x86_str(src, "k").unwrap_err(), format!("2:34: {nothing}"));
    let src = "unsafe extern \"ptx-kernel\" fn k() -> *const byte!{ _ } {}";
    let nothing = "a GPU kernel returns nothing, not `*const byte! { _ }`: it writes what it works out to memory";
    assert_eq!(kernite::ptx_str(src).unwrap_err(), format!("1:38: {nothing}"));
    // The result still ends at its `where`, which a kernel has none of, and at the `;` of a function with no
    // body, after a macro in braces too.
    let no_body = "a kernel is written in its body, and `k` has none";
    for (result, at, message) in [
        ("*const byte!{ _ } where u8: Sized {}", "1:52", "a kernel has no `where`"),
        ("*const byte!{ _ };", "1:51", no_body),
    ] {
        let src = format!("unsafe extern \"sysv64\" fn k() -> {result}");
        assert_eq!(kernite::x86_str(&src, "k").unwrap_err(), format!("{at}: {message}"), "{result}");
    }
}

#[test]
fn loops_unroll_and_number_their_labels_afresh() {
    let src = r#"
        const STEP: usize = 32;

        pub unsafe extern "sysv64" fn k(x: *const f32, n: usize) {
            dot::<2>(x, n);
        }

        fn dot<const U: usize>(x: gpr, n: gpr) {
            let acc: [ymm; U];
            for k in 0..U {
                zero(acc[k]);
            }
            label(top);
            for k in 0..U {
                vfmadd231ps(acc[k], acc[k], [x + STEP * k]);
            }
            add(x, STEP * U);
            sub(n, 8 * U);
            jnz(top);
            for k in (0..4).step_by(2).rev() {
                jmp(skip);
                label(skip);
                note("k = {k}, {{in braces}}");
            }
            for a in acc {
                vaddps(a, a, acc[0]);
            }
        }
    "#;
    // The language's labels 0, 1 and 2, which x86 writes from 2: `0b` and `1b` would be binary.
    let want = "# k(gpr, gpr) -> ()
# needs: avx, fma
# frame: 0 bytes aligned to 0, saves []
# x=rdi n=rsi acc0=ymm0 acc1=ymm1
    vxorps xmm0, xmm0, xmm0
    vxorps xmm1, xmm1, xmm1
2:
    vfmadd231ps ymm0, ymm0, ymmword ptr [rdi]
    vfmadd231ps ymm1, ymm1, ymmword ptr [rdi + 32]
    add rdi, 64
    sub rsi, 16
    jnz 2b
    jmp 3f
3:
    # k = 2, {in braces}
    jmp 4f
4:
    # k = 0, {in braces}
    vaddps ymm0, ymm0, ymm0
    vaddps ymm1, ymm1, ymm0
    vzeroupper
    ret
";
    assert_eq!(x86(src, "k"), want);
}

#[test]
fn brackets_are_memory_at_an_address_or_holding_data() {
    let src = r#"
        pub unsafe extern "sysv64" fn m(x: *const f32, i: usize) {
            let v: ymm;
            vmovups(v, [x + i * 4 + 16]);
            vmovups(v, [i * 8 - 4 + x]);
            vmovups(v, [x + (i + 2) * 4]);
            vmovups(v, [x - 8]);
            vmovups(v, [x + i]);
            vbroadcastss(v, [1.0]);
            vbroadcastsd(v, [[0.5; 2]]);
            vpbroadcastw(v, [[1u8, 2]]);
            vpbroadcastq(v, [[-1, 2]]);
        }
    "#;
    let want = "# m(gpr, gpr) -> ()
# needs: avx, avx2
# frame: 0 bytes aligned to 0, saves []
# x=rdi i=rsi v=ymm0
    vmovups ymm0, ymmword ptr [rdi + rsi*4 + 16]
    vmovups ymm0, ymmword ptr [rdi + rsi*8 - 4]
    vmovups ymm0, ymmword ptr [rdi + rsi*4 + 8]
    vmovups ymm0, ymmword ptr [rdi - 8]
    vmovups ymm0, ymmword ptr [rdi + rsi*1]
    vbroadcastss ymm0, dword ptr [rip + {c0}]
    vbroadcastsd ymm0, qword ptr [rip + {c1}]
    vpbroadcastw ymm0, word ptr [rip + {c2}]
    vpbroadcastq ymm0, qword ptr [rip + {c3}]
    vzeroupper
    ret
# c0, aligned to 4:
#   00 00 80 3f
# c1, aligned to 8:
#   00 00 00 3f 00 00 00 3f
# c2, aligned to 2:
#   01 02
# c3, aligned to 8:
#   ff ff ff ff 02 00 00 00
";
    assert_eq!(x86(src, "m"), want);
}

#[test]
fn a_predicate_guards_what_its_if_holds() {
    let src = r#"
        pub unsafe extern "ptx-kernel" fn g(n: u32, out: *mut u32) {
            let i: u32;
            mov.u32(i, tid.x);
            let p: pred;
            setp.ge.u32(p, i, n);
            if p {
                ret();
            }
            if !p {
                add.u32(n, n, 1);
            } else {
                sub.u32(n, n, 1);
            }
            if 1 + 1 == 2 {
                note("now");
            } else {
                note("never");
            }
            st.global.u32([out], n);
        }
    "#;
    let want = "g(
\t.param .u32 g_param_0,
\t.param .u64 g_param_1
)
{
\t.reg .pred %p;
\t.reg .u32 %n, %i;
\t.reg .u64 %out;

\tld.param.u32 %n, [g_param_0];
\tld.param.u64 %out, [g_param_1];
\tcvta.to.global.u64 %out, %out;
\tmov.u32 %i, %tid.x;
\tsetp.ge.u32 %p, %i, %n;
\t@%p ret;
\t@!%p add.u32 %n, %n, 1;
\t@%p sub.u32 %n, %n, 1;
\t// now
\tst.global.u32 [%out], %n;
\tret;
}
";
    assert_eq!(ptx(src), want);
}

/// An `if` on a variable guards each instruction inside, and the target checks the variable as
/// it guards one. A block that runs none would check nothing, so it is refused at the `if`, an
/// `else` too, whatever the variable is. Both blocks run before either count is read, so what is
/// wrong inside either is said first.
#[test]
fn an_if_that_guards_no_instruction_is_refused() {
    let block = "an `if` on a variable guards each instruction inside, and its block runs none";
    // x86 takes no predicate, and a note is no instruction.
    for body in [r#"let g: gpr; zero(g); if g { note("x"); }"#, "let g: gpr; zero(g); if g {}"] {
        assert_eq!(refused(body), format!("2:22: {block}"), "{body}");
    }
    // PTX takes a `pred` alone, and `r` is a `u32`. A block counts what it runs, not what it
    // holds: a `for` over nothing runs none, and so does a call of `h`, which is empty. A block
    // that runs none is refused though its `else` runs one, and an `else` that runs none is
    // refused too. A store in the `else` that `r` guards is refused first, at the store.
    let cases = [
        ("let r: u32; mov.u32(r, n); if r {}", format!("2:28: {block}")),
        ("let r: u32; mov.u32(r, n); if r { for i in 0..0 { st.global.u32([out], n); } }", format!("2:28: {block}")),
        ("let p: pred; setp.ne.u32(p, n, 0); if p { h(); }", format!("2:36: {block}")),
        ("let p: pred; setp.ne.u32(p, n, 0); if p {} else { st.global.u32([out], n); }", format!("2:36: {block}")),
        (
            r#"let p: pred; setp.ne.u32(p, n, 0); if p { st.global.u32([out], n); } else { note("x"); }"#,
            "2:36: an `if` on a variable guards each instruction inside, and its `else` runs none".to_string(),
        ),
        (
            "let r: u32; mov.u32(r, n); if r {} else { st.global.u32([out], n); }",
            "2:53: an instruction is guarded by a `pred`; this is a `u32`".to_string(),
        ),
    ];
    for (body, want) in cases {
        let src = format!("unsafe extern \"ptx-kernel\" fn k(n: u32, out: *mut u32) {{\n{body}\n}}\nfn h() {{}}\n");
        assert_eq!(kernite::ptx_str(&src).unwrap_err(), want, "{body}");
    }
    // An `if` that guards an instruction takes a note beside it, and counts one that a loop, an
    // `if` on a bool or a helper runs.
    let src = r#"
        pub unsafe extern "ptx-kernel" fn k(n: u32, out: *mut u32) {
            let p: pred;
            setp.ne.u32(p, n, 0);
            if p {
                note("x");
                st.global.u32([out], n);
            }
            if p {
                for k in 0..2 {
                    if k == 0 {
                        s(out, n);
                    }
                }
            }
        }

        fn s(o: u64, v: u32) {
            st.global.u32([o], v);
        }
    "#;
    let entry = ptx(src);
    assert!(entry.contains("\t// x\n\t@%p st.global.u32 [%out], %n;\n\t@%p st.global.u32 [%out], %n;\n"), "{entry}");
}

#[test]
fn values_are_computed_as_rust_computes_them() {
    let src = r#"
        const BITS: u32 = 1 << 31;
        const HALF: f64 = 1.0 / 2.0;
        const TABLE: [i32; 3] = [1, -2, 3];
        const SIGNED: i32 = BITS as i32;
        const TRUNC: u8 = 300 as u8;
        const ROUND: i32 = -2.7 as i32;
        const NEG_INF: f32 = f32::NEG_INFINITY;
        const LAST: i32 = TABLE[2];
        const BITS_OF: u32 = 1.0f32.to_bits();
        const ONE: f32 = f32::from_bits(0x3f80_0000);

        pub unsafe extern "sysv64" fn v() {
            let k: usize = 3;
            note("{BITS} {HALF} {SIGNED} {TRUNC} {ROUND} {NEG_INF} {LAST} {BITS_OF} {ONE} {k} {TABLE}");
            let g: gpr;
            mov(g, -LAST);
            add(g, 0xff_u8);
            sub(g, k);
            let h: ymm;
            vbroadcastsd(h, [HALF]);
            vbroadcastss(h, [ONE]);
        }
    "#;
    // `HALF` is held as the f64 it is declared, `ONE` as an f32.
    let want = "# v() -> ()
# needs: avx
# frame: 0 bytes aligned to 0, saves []
# g=rax h=ymm0
    # 2147483648 0.5 -2147483648 44 -2 -inf 3 1065353216 1 3 [1, -2, 3]
    mov rax, -3
    add rax, 255
    sub rax, 3
    vbroadcastsd ymm0, qword ptr [rip + {c0}]
    vbroadcastss ymm0, dword ptr [rip + {c1}]
    vzeroupper
    ret
# c0, aligned to 8:
#   00 00 00 00 00 00 e0 3f
# c1, aligned to 4:
#   00 00 80 3f
";
    assert_eq!(x86(src, "v"), want);
}

/// A shift or a `!` is checked for overflow, as `+` is. One of an integer with a type is wrapped
/// to the type's width, as Rust does, so it stays in range, as the note says. One of an integer
/// with no type is held to the range every other is, `i64::MIN` to `u64::MAX`, and refused past
/// it, at the operator: past it, a value would reach the ends of `i128`, and the third would
/// divide `i128::MIN` by -1.
#[test]
fn a_shift_and_a_not_are_checked_for_overflow() {
    let src = r#"
        const HIGH: u64 = 3 << 63;
        const MASK: u8 = !0;

        pub unsafe extern "sysv64" fn k() {
            note("{HIGH} {MASK}");
        }
    "#;
    let want = "# k() -> ()
# frame: 0 bytes aligned to 0, saves []
    # 9223372036854775808 255
    ret
";
    assert_eq!(x86(src, "k"), want);
    let cases = [
        ("const X: u64 = (3 << 63) as u64;", "2:19: 27670116110564327424 is out of range for an integer"),
        ("const X: u64 = !18446744073709551615 as u64;", "2:16: -18446744073709551616 is out of range for an integer"),
        (
            "const X: u64 = ((((-1 << 63) << 63) << 1) / -1) as u64;",
            "2:30: -85070591730234615865843651857942052864 is out of range for an integer",
        ),
    ];
    for (body, want) in cases {
        assert_eq!(refused(body), want, "{body}");
    }
}

/// A shift known now by a negative count, by as many bits as its type has, or by more, is an error at its operator
/// that says which counts a shift has, from 0 to the last bit of the type, 31 for a `u32`: the words are true of each
/// count, the same for `<<` and `>>`, for each integer type and for one with no type, held to 64 bits. A shift by the
/// last bit works out as in Rust: `1 << 7` of an `i8` is -128, and a `>>` of a signed type keeps its sign: `-8 >> 1`
/// is -4.
#[test]
fn a_shift_known_now_by_a_count_that_is_not_from_0_to_the_last_bit_of_its_type_is_refused_at_its_operator() {
    // Each integer type, its bits, and what Rust makes of `1 << (bits - 1)` of it.
    macro_rules! own {
        ($($ty:ident),*) => {
            [$((stringify!($ty), $ty::BITS as i32, ((1 as $ty) << ($ty::BITS - 1)).to_string())),*]
        };
    }
    let types = own!(u8, i8, u16, i16, u32, i32, u64, i64, usize, isize);
    let mut consts = String::from("const SIGNED: i32 = -8 >> 1;\n");
    for (ty, bits, _) in &types {
        consts += &format!("const {}: {ty} = 1 << {};\n", ty.to_uppercase(), bits - 1);
        for op in ["<<", ">>"] {
            for count in [-1, -3, *bits, bits + 8] {
                let body = format!("const X: {ty} = 1 {op} {count};");
                let want =
                    format!("2:{}: a shift by {count} is not from 0 to {}", body.find(op).unwrap() + 1, bits - 1);
                assert_eq!(refused(&body), want, "{body}");
            }
        }
    }
    assert_eq!(refused("const X: u64 = (1 << 64) as u64;"), "2:19: a shift by 64 is not from 0 to 63");
    assert_eq!(refused("const X: u64 = (1 >> -1) as u64;"), "2:19: a shift by -1 is not from 0 to 63");
    let names: Vec<_> = types.iter().map(|(ty, ..)| format!("{{{}}}", ty.to_uppercase())).collect();
    let src = format!("{consts}unsafe extern \"sysv64\" fn k() {{ note(\"{{SIGNED}} {}\"); }}\n", names.join(" "));
    let values: Vec<_> = types.iter().map(|(.., last)| last.as_str()).collect();
    let listing = x86(&src, "k");
    assert!(listing.contains(&format!("# {} {}\n", -8_i32 >> 1, values.join(" "))), "{listing}");
}

/// A const is what kernels read: a number, a float, a `bool`, or an array of them. The kernel
/// reads one of each, and adds only if each is what it was declared. One of another type is no
/// value known now, and the error says what one is.
#[test]
fn a_const_is_a_number_a_float_a_bool_or_an_array_of_them() {
    let src = |ty: &str| {
        format!(
            "const N: u64 = 1;\nconst F: f32 = 2.5;\nconst B: bool = true;\nconst XS: {ty} = [1, 2];\n\
             unsafe extern \"sysv64\" fn k(x: u64) -> u64 {{\nif B && F > 2.0 {{\nadd(x, N + XS[1]);\n}}\n\
             ret(x);\n}}\n"
        )
    };
    let want = "# k(gpr) -> gpr
# frame: 0 bytes aligned to 0, saves []
# x=rdi,rax
    add rdi, 3
    mov rax, rdi
    ret
";
    assert_eq!(x86(&src("[u64; 2]"), "k"), want);
    let e = kernite::x86_str(&src("[u64]"), "k").unwrap_err();
    assert_eq!(e, "4:11: a value known now is a number, a bool, or an array of them, `[T; N]`");
}

#[test]
fn sixteen_bit_floats_are_values_and_data() {
    let src = r#"
        const THIRD: f16 = 1.0 / 3.0;

        pub unsafe extern "sysv64" fn h() {
            const SUM: f16 = 0.1f16 + 0.2f16;
            const B: bf16 = 1.0bf16 / 3.0;
            const HUGE: f16 = f16::MAX * 2.0;
            const TIE: bf16 = 1.00390625;
            const LEAST: f16 = f16::from_bits(1);
            const BITS: u16 = THIRD.to_bits();
            const WIDE: f32 = THIRD as f32;
            let k: f16 = -2.5;
            note("{THIRD} {SUM} {B} {HUGE} {TIE} {LEAST} {BITS} {WIDE} {k}");
            let v: ymm;
            vpbroadcastw(v, [THIRD]);
            vpbroadcastw(v, [[1.5bf16, -2.0bf16]]);
        }
    "#;
    // Each is rounded once, to the nearest, the middle to even: 0.1 + 0.2 lands on a middle.
    // Past the largest, a product is infinite, as a literal may not be.
    let want = "# h() -> ()
# needs: avx2
# frame: 0 bytes aligned to 0, saves []
# v=ymm0
    # 0.3333 0.2998 0.334 inf 1 0.00000006 13653 0.33325195 -2.5
    vpbroadcastw ymm0, word ptr [rip + {c0}]
    vpbroadcastw ymm0, word ptr [rip + {c1}]
    vzeroupper
    ret
# c0, aligned to 2:
#   55 35
# c1, aligned to 4:
#   c0 3f 00 c0
";
    assert_eq!(x86(src, "h"), want);
}

#[test]
fn errors_point_where_the_trouble_is() {
    let cases: &[(&str, &str)] = &[
        ("let k = nope;", "1:9: `nope` is not defined"),
        ("let x;", "1:5: `let x;` declares nothing: give it a type, as in `let x: ymm;`, or a value"),
        ("label(a); label(a);", "1:17: label `a` is placed twice in this block"),
        ("two(1);", "1:1: `two` takes 2 arguments; this gives 1"),
        ("deep();", "5:13: helpers call helpers 64 deep here; does `deep` call itself?"),
        ("const X: u8 = 255 + 1;", "1:19: 256 is out of range for u8"),
        // A literal negated is one number, as Rust reads it: `-128i8` is in range.
        ("const X: i8 = -129;", "1:15: -129 is out of range for i8"),
        ("const X: i8 = -(128);", "1:17: 128 is out of range for i8"),
        ("const X: u8 = -0;", "1:15: a u8 cannot be negated"),
        // A float too big for its type is out of range, as Rust has it, and not infinite.
        ("const F: f32 = 1e39;", "1:16: 1e39 is out of range for f32"),
        ("const F: f64 = 2e308f64;", "1:16: 2e308f64 is out of range for f64"),
        ("const F: f16 = 65520.0;", "1:16: 65520.0 is out of range for f16"),
        ("const F: bf16 = 3.4e38bf16;", "1:17: 3.4e38bf16 is out of range for bf16"),
        ("let k = A;", "7:18: `A` is defined in terms of itself"),
        (
            "let a: [ymm; 2]; vaddps(a, a, a);",
            "1:25: x86 takes a variable here, and these are 2 together: name one, as in `a[0]`",
        ),
        (
            "let v: ymm; vaddps(v, v, (v, v));",
            "1:26: x86 takes a variable here, and these are 2 together: name one, as in `a[0]`",
        ),
        ("let v: ymm; vaddps(v, v, !v);", "1:26: x86 reads no operand inverted or negated; `!p` and `-a` are PTX's"),
        ("let v: ymm; vaddps(v, v | v, v);", "1:25: x86 has no instruction that writes `a | b`, two at once"),
        ("let v: ymm; vmovups(v, [v, 4]);", "1:24: memory is one address in brackets, as in `[x + 4]`"),
        ("let g: gpr; mov(g, 1.5f16);", "1:20: x86 takes floats from memory: `[1.5f16]`"),
        (
            "const K: usize = 4; let t: shared<[f32; 2 * K], 16>;",
            "1:28: `shared<..>` is a GPU's memory; an x86 kernel has stack memory, `local<[f32; 8], 16>`",
        ),
        ("let t: [f32];", "1:8: variables are declared so many at once: `let t: [u32; 4];`"),
        // The machine is given a name as it writes it, not as Rust must.
        ("let v: ymm; vaddps._2d__x(v);", "1:20: x86 has no instruction `vaddps.2d::x` that kernite knows"),
        ("let a: [ymm; 2]; zero(a[2]);", "1:24: `a` has 2 variables; there is no [2]"),
        ("let v: ymm; vmovups(v, [v * v]);", "1:27: an address multiplies a variable by a number, not by a variable"),
        (
            "let v: ymm; let k = v + 1;",
            "1:21: `v` is a variable, known only when the kernel runs; this wants a value known now",
        ),
        ("two::<1>(1, 2);", "1:1: `two` takes 0 values in `::<..>`; this gives 1"),
        ("note(\"{gone}\");", "1:6: `gone` is not defined"),
        ("loop {}", "1:1: a kernel has no `loop`: loops are labels and jumps, and `for` is unrolled"),
        (
            "let g: gpr; mov(g, two);",
            "1:20: `two` is a helper, inlined where it is called, `two(..)`: no instruction calls it",
        ),
        (
            "let g: gpr; mov(g, k);",
            "1:20: `mov` takes no function; `call(r, k, (a, b))` calls `k`, and `lea(p, [k])` puts its address in `p`",
        ),
        (
            "k();",
            "1:1: `k` names an ABI, so it is compiled on its own, not inlined as a helper is: an instruction calls \
             it, `call(r, k, (a, b))`",
        ),
    ];
    for (body, want) in cases {
        let src = format!(
            "unsafe extern \"sysv64\" fn k() {{\n{body}\n}}\nfn two(a: ymm, b: ymm) {{}}\nfn deep() {{ deep(); }}\n\
             const A: usize = B;\nconst B: usize = A;\n"
        );
        let got = kernite::x86_str(&src, "k").unwrap_err();
        // The body is on the second line.
        let want = want.replacen("1:", "2:", 1);
        assert_eq!(got, want, "{body}");
    }
    // Predicates are PTX's.
    let cases: &[(&str, &str)] = &[
        (
            "let p: pred; if p { if p { ret(); } }",
            "2:21: a predicate's `if` inside another's: one predicate guards an instruction",
        ),
        (
            "let p: pred; if p { label(x); }",
            "2:21: a label takes no predicate: control arrives there whatever it holds",
        ),
        // Rust has no word that starts with a digit, nor `::` after a field: the source a
        // macro is given says `._2d` and `__`, and a string is told so.
        (
            "tex.2d.v4.f32.f32(a, [t, (x, y)]);",
            "2:5: a word after a dot starts with a letter or `_`: `.2d` is written `._2d`",
        ),
        (
            "ld.global.L2::64B.b32(a, [x]);",
            "2:13: a word of a name is joined to the next by `__`, which PTX writes `::`: `L2__64B`",
        ),
    ];
    for (body, want) in cases {
        let got = kernite::ptx_str(&format!("unsafe extern \"ptx-kernel\" fn k() {{\n{body}\n}}\n")).unwrap_err();
        assert_eq!(got, *want, "{body}");
    }
}

/// What the machine finds wrong with a variable once the body is written is placed where that
/// variable is declared, though others of its name are declared before it or after: a helper's,
/// an array's, or one declared again.
#[test]
fn an_error_is_placed_at_its_own_variable_s_declaration() {
    let cases: &[(&str, &str)] = &[
        ("let v: ymm; h(); let v: ymm;", "4:14: `v` is read before anything writes it"),
        // An array's variables are called `v0` and `v1`.
        ("let v0: ymm; let v: [ymm; 2]; vaddps(v[0], v[0], v[0]);", "2:18: `v0` is read before anything writes it"),
        ("let v1: ymm; let v: [ymm; 2]; vaddps(v[1], v[1], v[1]);", "2:18: `v1` is read before anything writes it"),
    ];
    for (body, want) in cases {
        let src =
            format!("unsafe extern \"sysv64\" fn k() {{\n{body}\n}}\nfn h() {{ let v: ymm; vaddps(v, v, v); }}\n");
        let got = kernite::x86_str(&src, "k").unwrap_err();
        assert_eq!(got, *want, "{body}");
    }
    let body = "let v: f32; let v: f32; add.rn.f32(v, v, 1.5); let v: f32;";
    let got = kernite::ptx_str(&format!("unsafe extern \"ptx-kernel\" fn k() {{\n{body}\n}}\n")).unwrap_err();
    assert_eq!(got, "2:17: `v` is read before anything writes it");
}

#[test]
fn a_kernel_with_compile_time_arguments_is_compiled_for_each_instance_it_lists() {
    // Each case comes before these, so its lines are its own.
    let after = r#"
const WIDE: u32 = 8;
#[instances(1, WIDE)]
unsafe extern "ptx-kernel" fn shift<const S: u32>(x: *mut u32) {}
unsafe extern "ptx-kernel" fn plain() {}
fn h<const N: u32>() {}
"#;
    let k = "unsafe extern \"ptx-kernel\" fn k";
    let cases: &[(&str, &str)] = &[
        (
            "unsafe extern \"ptx-kernel\" fn k<const N: u32>() {}",
            "1:39: `k` takes compile-time arguments, so it is compiled for each instance its `#[instances(..)]` lists, \
             as in `#[instances(4, 8)]`",
        ),
        (
            "#[instances(1)]\nK() {}",
            "1:3: `k` takes no compile-time arguments, so it is compiled once, with no `#[instances(..)]`",
        ),
        (
            "#[instances(1)]\nfn g<const N: u32>() {}",
            "1:3: a helper is compiled where it is called, for what `::<..>` gives it there: `#[instances(..)]` lists \
             what a function that names an ABI is compiled for",
        ),
        (
            "#[instances(1, (2, 3))]\nK<const A: u32, const B: u32>() {}",
            "1:13: an instance of `k` gives its 2 compile-time arguments in parentheses, as `(A, B)`",
        ),
        (
            "#[instances((1, 2, 3))]\nK<const A: u32, const B: u32>() {}",
            "1:13: an instance of `k` gives its 2 compile-time arguments in parentheses, as `(A, B)`",
        ),
        ("#[instances(4, 2 + 2)]\nK<const N: u32>() {}", "1:18: `k::<4>` is listed twice"),
        ("#[instances()]\nK<const N: u32>() {}", "1:3: `#[instances(..)]` lists each instance, and this lists none"),
        (
            "#[instances(1.0)]\nK<const X: f32>() {}",
            "2:39: a function that names an ABI takes integers and `bool`s at compile time, as Rust's const generics \
             are; `f32` is neither",
        ),
        ("#[instances(256)]\nK<const N: u8>() {}", "1:13: 256 is out of range for u8"),
        ("#[instances(1)]\nK<const N: bool>() {}", "1:13: this is an integer, not bool"),
        ("#[instances(WIDE)]\nK<const N: u8>() {}", "1:13: this is a u32, not u8"),
        ("#[instances((1u8, 2usize))]\nK<const N: u8, const M: isize>() {}", "1:19: this is a usize, not isize"),
        ("#[instances(MISSING)]\nK<const N: u32>() {}", "1:13: `MISSING` is not defined"),
        (
            "#[instances(1)]\nconst C: u32 = 1;",
            "1:3: `#[instances(..)]` goes on a function that names an ABI, which it lists the instances of",
        ),
        (
            "#[instances]\nK<const N: u32>() {}",
            "1:3: `#[instances(..)]` lists what a function is compiled for, as in `#[instances(4, 8)]`",
        ),
        (
            "#[instances(1)]\n#[instances(2)]\nK<const N: u32>() {}",
            "2:3: a function has one `#[instances(..)]`, which lists each instance it is compiled for",
        ),
        // An instruction names an instance that is listed, by its values, however written.
        (
            "K() { let a: u64; mov.u64(a, shift::<3>); }",
            "1:60: `shift::<3>` is not compiled: `shift`'s `#[instances(..)]` lists `shift::<1>`, `shift::<8>`",
        ),
        (
            "K() { let a: u64; mov.u64(a, shift); }",
            "1:60: `shift` is compiled for each of its instances: an instruction names one, `shift::<..>`",
        ),
        (
            "K() { let a: u64; mov.u64(a, plain::<1>); }",
            "1:60: `plain` takes no compile-time arguments: an instruction names it, `plain`",
        ),
        (
            "K() { let a: u64; mov.u64(a, h::<1>); }",
            "1:60: `h` is a helper, inlined where it is called: an instruction names a function that names an ABI",
        ),
        ("K() { let a: u64; mov.u64(a, shift::<1, 2>); }", "1:60: `shift` takes 1 values in `::<..>`; this gives 2"),
        (
            "K() { let n = shift::<1>; }",
            "1:45: an instance of a function is for an instruction to name, as `call` does, not a value",
        ),
        // What goes wrong in one instance alone says which.
        (
            "#[instances(1, 16)]\nK<const B: u32>() { bar.sync(B); }",
            "2:60: in `k::<16>`: `bar.sync` takes 0 to 15 here; 16 is not one of them",
        ),
    ];
    for (case, want) in cases {
        let src = case.replacen('K', k, 1) + after;
        assert_eq!(kernite::ptx_str(&src).unwrap_err(), *want, "{case}");
    }
    let named = "K() { let a: u64; mov.u64(a, shift::<{ WIDE / 8 }>); mov.u64(a, shift::<8u32>); }";
    let ptx = kernite::ptx_str(&(named.replacen('K', k, 1) + after)).unwrap();
    assert!(ptx.contains("\tmov.u64 %a, _Z5shiftILj1EE;\n\tmov.u64 %a, _Z5shiftILj8EE;\n"), "{ptx}");
}

#[test]
fn a_helper_sees_none_of_its_callers_names() {
    let src = r#"
        pub unsafe extern "sysv64" fn k() {
            let secret = 3;
            h();
        }
        fn h() {
            note("{secret}");
        }
    "#;
    assert_eq!(kernite::x86_str(src, "k").unwrap_err(), "7:18: `secret` is not defined");
}

/// Each const, function and static of a module has a name of its own: of two of one name, the
/// static is refused, else the function, else the second, at its name.
#[test]
fn each_const_function_and_static_has_a_name_of_its_own() {
    let cases = [
        ("fn h() {} fn h() {}", "1:14: `h` names another function of the module as well"),
        (r#"fn k() {} pub unsafe extern "sysv64" fn k() {}"#, "1:41: `k` names another function of the module as well"),
        // A kernel of compile-time arguments and one of none are two kernels of a name.
        (
            r#"#[instances(4, 2)] pub unsafe extern "sysv64" fn tile<const N: usize>() {} pub unsafe extern "sysv64" fn tile() {}"#,
            "1:106: `tile` names another function of the module as well",
        ),
        // Two under a `#[cfg]` each are two as well: kernite reads every item.
        (
            r#"#[cfg(a)] pub unsafe extern "sysv64" fn k() {} #[cfg(not(a))] pub unsafe extern "sysv64" fn k() {}"#,
            "1:93: `k` names another function of the module as well",
        ),
        // A const and a function: the function is refused, whichever is first.
        ("const h: u32 = 1; fn h() {}", "1:22: `h` names a constant of the module as well"),
        ("fn h() {} const h: u32 = 1;", "1:4: `h` names a constant of the module as well"),
        ("const N: u32 = 1; const N: u32 = 2;", "1:25: `N` names another constant of the module as well"),
        // Of two items refused, the first in the module is the one: here `b`, though the const
        // it is refused for comes last.
        ("fn a() {} fn b() {} fn a() {} const b: u32 = 1;", "1:14: `b` names a constant of the module as well"),
        (
            r#"pub unsafe extern "sysv64" fn k() {} static k: global<u32>;"#,
            "1:45: `k` names a function of the module as well",
        ),
    ];
    for (src, want) in cases {
        assert_eq!(kernite::x86_str(src, "k").unwrap_err(), want, "{src}");
    }
    let src = r#"pub unsafe extern "ptx-kernel" fn k() {} pub unsafe extern "ptx-kernel" fn k() {}"#;
    assert_eq!(kernite::ptx_str(src).unwrap_err(), "1:76: `k` names another function of the module as well");
}

#[test]
fn array_parameters_take_variables_declared_together() {
    let src = r#"
        pub unsafe extern "sysv64" fn k(x: *const f32) -> f32 {
            let acc: [ymm; 3];
            for k in 0..3 {
                vmovups(acc[k], [x + 32 * k]);
            }
            sum::<3>(acc);
        }
        fn sum<const N: usize>(acc: [xmm; N]) {
            for k in 1..N {
                vaddps(acc[0], acc[0], acc[k]);
            }
            ret(acc[0]);
        }
    "#;
    let want = "# k(gpr) -> vec
# needs: avx
# frame: 0 bytes aligned to 0, saves []
# x=rdi acc0=ymm0 acc1=ymm1 acc2=ymm2
    vmovups ymm0, ymmword ptr [rdi]
    vmovups ymm1, ymmword ptr [rdi + 32]
    vmovups ymm2, ymmword ptr [rdi + 64]
    vaddps xmm0, xmm0, xmm1
    vaddps xmm0, xmm0, xmm2
    vzeroupper
    ret
";
    assert_eq!(x86(src, "k"), want);
}

/// What each const of the module `src` works out to on its own: its value, or its error at its place.
fn constants(src: &str) -> impl Fn(&str) -> Result<String, String> + '_ {
    let module = kernite::lang::parse(src).unwrap();
    move |name| {
        let c = module
            .items
            .iter()
            .find_map(|item| match &item.kind {
                kernite::lang::ItemKind::Const(c) if c.name == name => Some(c),
                _ => None,
            })
            .unwrap();
        kernite::lang::constant(&module, c).map(|v| v.to_string()).map_err(|e| e.at(src))
    }
}

/// A constant of a module works out on its own, in terms of the module's others, before or
/// after it, as a module of consts has each checked; or says why it is no value known now.
#[test]
fn a_constant_works_out_on_its_own() {
    let src = "const D: usize = E / H;\nconst E: usize = 16;\nconst H: usize = 4;\nconst W: [f32; 2] = [0.5, -1.0];\n\
               const LOOP: u32 = LOOP + 1;\nconst BAD: u32 = 1.5;\nconst GONE: u32 = NOPE;\n\
               const DIV: u32 = 1 / 0;\nconst REM: u32 = 1 % 0;\nunsafe extern \"sysv64\" fn k() {}\n";
    let value = constants(src);
    let cases = [
        ("D", Ok("4")),
        ("W", Ok("[0.5, -1]")),
        ("LOOP", Err("5:19: `LOOP` is defined in terms of itself")),
        ("BAD", Err("6:18: this is an f32, not u32")),
        ("GONE", Err("7:19: `NOPE` is not defined")),
        ("DIV", Err("8:20: 1 / 0 divides by zero")),
        ("REM", Err("9:20: 1 % 0 divides by zero")),
    ];
    for (name, want) in cases {
        assert_eq!(value(name), want.map(String::from).map_err(String::from), "{name}");
    }
}

/// A `/` or a `%` of integers known now by zero says that it divides by zero, at its operator, in a const that a kernel
/// reads, on x86 and on a GPU, as it does where the const is worked out on its own. By one, the same kernels compile.
#[test]
fn a_kernel_that_reads_a_division_by_zero_is_told_that_it_divides_by_zero() {
    let src = |gpu: bool, op: &str, divisor: u32| {
        let kernel = match gpu {
            true => r#"unsafe extern "ptx-kernel" fn k(p: *mut u32) { st.global.u32([p], N); }"#,
            false => r#"unsafe extern "sysv64" fn k() { let r: gpr; mov(r, N); }"#,
        };
        format!("const N: u32 = 1 {op} {divisor};\n{kernel}\n")
    };
    for (op, want) in [("/", "1:18: 1 / 0 divides by zero"), ("%", "1:18: 1 % 0 divides by zero")] {
        for gpu in [false, true] {
            assert_eq!(compile(&src(gpu, op, 0), "k").unwrap_err(), want, "{op}, gpu: {gpu}");
            compile(&src(gpu, op, 1), "k").unwrap_or_else(|e| panic!("{op} by 1, gpu: {gpu}: {e}"));
        }
    }
}

/// The remainder of the least of a signed type by -1, known now, overflows, as in Rust, though it is 0: it is an error
/// at its `%`, in a const that a kernel reads, on x86 and on a GPU. By 1, the same kernels compile.
#[test]
fn a_kernel_that_reads_the_remainder_of_the_least_of_a_signed_type_by_minus_one_is_told_that_it_overflows() {
    let src = |gpu: bool, ty: &str, store: &str, least: &str, divisor: i32| {
        let kernel = match gpu {
            true => format!(r#"unsafe extern "ptx-kernel" fn k(p: *mut {ty}) {{ st.global.{store}([p], N); }}"#),
            false => r#"unsafe extern "sysv64" fn k() { let r: gpr; mov(r, N); }"#.to_string(),
        };
        format!("const N: {ty} = {least} % {divisor};\n{kernel}\n")
    };
    for (ty, store, least, at) in
        [("i64", "s64", "-9223372036854775808", "1:37"), ("i32", "s32", "-2147483648", "1:28")]
    {
        let want = format!("{at}: {least} % -1 overflows");
        for gpu in [false, true] {
            assert_eq!(compile(&src(gpu, ty, store, least, -1), "k").unwrap_err(), want, "{ty}, gpu: {gpu}");
            compile(&src(gpu, ty, store, least, 1), "k").unwrap_or_else(|e| panic!("{ty} by 1, gpu: {gpu}: {e}"));
        }
    }
}

/// The least of each signed type by -1 overflows under `%` as under `/`, at the operator: the `%` says that it
/// overflows, where the `/` says that its quotient is out of range. The least by 1, and the one above it by -1, are 0,
/// as they are in Rust. A product past i128, in which kernite works an integer out, says that it overflows too:
/// `u64::MAX * u64::MAX`.
#[test]
fn the_least_of_each_signed_type_overflows_by_minus_one_under_a_remainder_as_under_a_quotient() {
    let types = [
        ("i8", "-128"),
        ("i16", "-32768"),
        ("i32", "-2147483648"),
        ("i64", "-9223372036854775808"),
        ("isize", "-9223372036854775808"),
    ];
    let (mut src, mut cases) = (String::new(), vec![]);
    // A const of the type `ty` that is `sum`, which works out to `want`, or is the error `want` at its `op`.
    let mut add = |ty: &str, sum: &str, op: char, want: Result<&str, String>| {
        let line = format!("const N{}: {ty} = {sum};\n", cases.len());
        let at = format!("{}:{}", src.lines().count() + 1, line.find(op).unwrap() + 1);
        cases.push((format!("N{}", cases.len()), want.map(String::from).map_err(|message| format!("{at}: {message}"))));
        src += &line;
    };
    for (ty, least) in types {
        let magnitude = -least.parse::<i128>().unwrap();
        add(ty, &format!("{least} % -1"), '%', Err(format!("{least} % -1 overflows")));
        add(ty, &format!("({least} + 1) % -1"), '%', Ok("0"));
        add(ty, &format!("{least} % 1"), '%', Ok("0"));
        add(ty, &format!("{least} / -1"), '/', Err(format!("{magnitude} is out of range for {ty}")));
    }
    let product = "18446744073709551615 * 18446744073709551615";
    add("u64", product, '*', Err(format!("{product} overflows")));
    let value = constants(&src);
    for (name, want) in cases {
        assert_eq!(value(&name), want, "{name}");
    }
}

/// An integer where an integer of another type is wanted is told by its type, `a u16` for a `u8`: `an integer` would
/// say no fault, for a `u8` is one too. So it is for a const, an element of an array const, a const named by another
/// and a `let`, on x86; and for a compile-time argument of `#[instances(..)]`, on x86 and on a GPU. Where the type
/// wanted is no integer's, what is wrong is that the value is an integer, and it is `an integer`, whatever its type.
#[test]
fn an_integer_of_another_type_is_told_by_its_type() {
    let x86 = |items: &str| {
        let src = format!("{items}\nunsafe extern \"sysv64\" fn k() -> u64 {{ let r: gpr; mov(r, A); ret(r); }}\n");
        kernite::x86_str(&src, "k").unwrap_err()
    };
    let consts = [
        ("const A: u8 = 1u16;", "1:15: this is a u16, not u8"),
        ("const A: i32 = 2usize;", "1:16: this is a usize, not i32"),
        ("const A: u64 = 3i64;", "1:16: this is an i64, not u64"),
        ("const A: isize = 4usize;", "1:18: this is a usize, not isize"),
        ("const A: usize = 5isize;", "1:18: this is an isize, not usize"),
        // The type of a sum is that of its terms, told at the `+`, and a const named brings its own.
        ("const A: u8 = 1u16 + 2;", "1:20: this is a u16, not u8"),
        ("const B: u16 = 3;\nconst A: u8 = B;", "2:15: this is a u16, not u8"),
        // An element of an array is told at the array.
        ("const A: [u8; 2] = [1u8, 2u16];", "1:20: this is a u16, not u8"),
        ("const A: [[u8; 2]; 2] = [[1, 2], [3, 4u32]];", "1:25: this is a u32, not u8"),
        // A float and a bool are told by their types; a number of no type, and an integer where the type wanted is
        // no integer's, are `an integer`; and a number too big for its type is told by its value.
        ("const A: u8 = 1.5;", "1:15: this is an f32, not u8"),
        ("const A: u32 = true;", "1:16: this is a bool, not u32"),
        ("const A: bool = 1;", "1:17: this is an integer, not bool"),
        ("const A: bool = 1u16;", "1:17: this is an integer, not bool"),
        ("const A: f32 = 1u16;", "1:16: this is an integer, not f32"),
        ("const A: u8 = 300;", "1:15: 300 is out of range for u8"),
    ];
    for (items, want) in consts {
        assert_eq!(x86(items), want, "{items}");
    }
    assert_eq!(refused("let n: u8 = 1u16;"), "2:13: this is a u16, not u8");
    assert_eq!(refused("let n: u8 = 1;\nlet m: bool = 1u16;"), "3:15: this is an integer, not bool");
    let instances = [
        ("#[instances(1u16)]\nK<const N: u8>() {}", "1:13: this is a u16, not u8"),
        ("#[instances(1, 2i32)]\nK<const N: u8>() {}", "1:16: this is an i32, not u8"),
        ("#[instances(1u16)]\nK<const N: bool>() {}", "1:13: this is an integer, not bool"),
    ];
    for (case, want) in instances {
        let on_x86 = case.replacen('K', "unsafe extern \"sysv64\" fn k", 1);
        assert_eq!(kernite::x86_str(&on_x86, "k").unwrap_err(), want, "{on_x86}");
        let on_a_gpu = case.replacen('K', "unsafe extern \"ptx-kernel\" fn k", 1);
        assert_eq!(kernite::ptx_str(&on_a_gpu).unwrap_err(), want, "{on_a_gpu}");
    }
}

/// A range unrolls to no more numbers than a value holds, 1048576 at most.
#[test]
fn a_range_of_more_than_a_value_holds_is_refused() {
    assert_eq!(refused("for i in 0..1048577 {}"), "2:11: 1048577 elements are too many: a value holds 1048576 at most");
}

/// A repeat of more elements than a value holds is refused there, before it is built, and so is one
/// of empty arrays or of arrays of one, which hold no more than one each but cost memory of their
/// own.
#[test]
fn a_repeat_of_more_than_a_value_holds_is_refused() {
    let cases = [
        (
            "const X: [u32; 1048577] = [0; 1048577];",
            "2:27: 1048577 elements are too many: a value holds 1048576 at most",
        ),
        (
            "const X: [[u32; 0]; 1048577] = [[]; 1048577];",
            "2:32: 1048577 elements are too many: a value holds 1048576 at most",
        ),
        (
            "const X: [[[u32; 1]; 1]; 524288] = [[[0]]; 524288];",
            "2:36: 1572864 elements are too many: a value holds 1048576 at most",
        ),
    ];
    for (body, want) in cases {
        assert_eq!(refused(body), want, "{body}");
    }
}

/// A repeat of repeats, each within the most a value holds, is refused at the outer one when they
/// hold more in all, however deep they nest, before it is built: the first would hold
/// 1099511627776 numbers.
#[test]
fn a_repeat_of_repeats_holding_more_in_all_is_refused() {
    let cases = [
        (
            "const X: [[u32; 1 << 20]; 1 << 20] = [[0; 1 << 20]; 1 << 20];",
            "2:38: 1099511627776 elements are too many: a value holds 1048576 at most",
        ),
        (
            "const X: [[[u32; 1 << 10]; 1 << 10]; 2] = [[[0; 1 << 10]; 1 << 10]; 2];",
            "2:43: 2097152 elements are too many: a value holds 1048576 at most",
        ),
    ];
    for (body, want) in cases {
        assert_eq!(refused(body), want, "{body}");
    }
}

/// An array of repeats, each within the most a value holds, is refused at the array when they hold
/// more in all, and its error counts what every item holds; so is an array of one that holds the
/// most, which counts as one more.
#[test]
fn an_array_of_repeats_holding_more_in_all_is_refused() {
    let cases = [
        (
            "const X: [[u32; 1 << 20]; 3] = [[0; 1 << 20], [0; 1 << 20], [0; 1 << 20]];",
            "2:32: 3145728 elements are too many: a value holds 1048576 at most",
        ),
        (
            "const X: [[u32; 1 << 20]; 1] = [[0; 1 << 20]];",
            "2:32: 1048577 elements are too many: a value holds 1048576 at most",
        ),
    ];
    for (body, want) in cases {
        assert_eq!(refused(body), want, "{body}");
    }
}

/// A value that holds the most, 1048576 elements, keeps every item, and a range unrolls to as many.
#[test]
fn a_value_holding_the_most_keeps_every_item() {
    let src = r#"pub unsafe extern "sysv64" fn k(x: u64) -> u64 {
    for i in 0..1048576 {}
    for half in [[0; 1 << 19], [0; 1 << 19]] {
        add(x, 1);
    }
    ret(x);
}
"#;
    let want = "# k(gpr) -> gpr
# frame: 0 bytes aligned to 0, saves []
# x=rdi,rax
    add rdi, 1
    add rdi, 1
    mov rax, rdi
    ret
";
    assert_eq!(x86(src, "k"), want);
}

/// A kernel declares at most 65536 variables by `let`, however its loops repeat them. The `let`
/// that would pass the most is refused where it stands, before it makes any variable: an array of
/// 2^40 would abort the process.
#[test]
fn a_let_that_would_pass_the_most_variables_is_refused_before_any_is_made() {
    let cases = [
        ("let v: [gpr; 1099511627776];", "2:14: 1099511627776 variables are too many: a kernel declares 65536 at most"),
        (
            "let v: [gpr; 18446744073709551615];",
            "2:14: 18446744073709551615 variables are too many: a kernel declares 65536 at most",
        ),
        ("for i in 0..262144 { let x: gpr; }", "2:26: 65537 variables are too many: a kernel declares 65536 at most"),
        (
            "let a: [gpr; 32768]; let b: [gpr; 32769];",
            "2:35: 65537 variables are too many: a kernel declares 65536 at most",
        ),
    ];
    for (body, want) in cases {
        assert_eq!(refused(body), want, "{body}");
    }
}

/// The variables a helper declares count for the kernel that calls it: here `t` makes 65537.
#[test]
fn a_helpers_variables_count_for_the_kernel_that_calls_it() {
    let src = "unsafe extern \"sysv64\" fn k() {\nlet a: [gpr; 65535]; h();\n}\nfn h() { let t: [gpr; 2]; }\n";
    assert_eq!(
        kernite::x86_str(src, "k").unwrap_err(),
        "4:23: 65537 variables are too many: a kernel declares 65536 at most"
    );
}

/// A kernel that declares exactly the most compiles, its parameters not counted among them. PTX
/// builds no graph of its variables, so this takes no time to speak of.
#[test]
fn a_kernel_that_declares_exactly_the_most_variables_compiles() {
    kernite::ptx_str("unsafe extern \"ptx-kernel\" fn k(x: *mut u32) {\nlet v: [u32; 65536];\n}\n").unwrap();
}

/// A row of operators, of brackets, of `else if`s or of arrays nests 64 deep at most. A row of the
/// most compiles, and the level past it is refused where it starts. So is a row of 5000, which
/// the parser or the evaluator would follow, each level a call, until a thread of 2 MiB ran out
/// of stack.
#[test]
fn what_nests_past_64_deep_is_refused_where_the_level_starts() {
    // A name, the body of a row of `n`, the most `n` that compiles, and where one more is refused.
    type Shape = (&'static str, fn(usize) -> String, usize, &'static str);
    let shapes: [Shape; 11] = [
        ("a sum", |n| format!("let y = 1{};", " + 1".repeat(n)), 63, "2:263"),
        ("a difference", |n| format!("let y = 1{};", " - 1".repeat(n)), 63, "2:263"),
        ("casts", |n| format!("let y = 1{};", " as u32".repeat(n)), 63, "2:452"),
        ("methods", |n| format!("let y = 1{};", ".min(1)".repeat(n)), 63, "2:452"),
        ("minus signs", |n| format!("let y = {}1;", "-".repeat(n)), 63, "2:72"),
        ("nots", |n| format!("let y = {}true;", "!".repeat(n)), 63, "2:72"),
        ("parentheses", |n| format!("let y = {}1{};", "(".repeat(n), ")".repeat(n)), 63, "2:72"),
        ("blocks", |n| format!("{}{}", "{".repeat(n), "}".repeat(n)), 63, "2:64"),
        ("`else if`s", |n| format!("if false {{}}{}", " else if false {}".repeat(n)), 62, "2:1072"),
        ("arrays of an array", |n| format!("let w = 0;{}", "\nlet w = [w];".repeat(n)), 64, "67:9"),
        ("arrays of a repeat", |n| format!("let w = 0;{}", "\nlet w = [w; 1];".repeat(n)), 64, "67:9"),
    ];
    for (name, body, most, at) in shapes {
        x86(&kernel(&body(most)), "k");
        let why = format!("{at}: this nests 64 deep at most");
        assert_eq!(refused(&body(most + 1)), why, "{name}");
        assert_eq!(refused(&body(5000)), why, "{name}, 5000 of them");
    }
}

/// Consts that name consts nest like the expressions that name them: a chain of 64 works out, and
/// one of 65, or of 5000, is refused where the 65th const is written.
#[test]
fn consts_that_name_consts_nest_64_deep_at_most() {
    let module = |n: usize| {
        let consts: String = (0..n).map(|i| format!("const A{i}: u32 = A{};\n", i + 1)).collect();
        format!("{consts}const A{n}: u32 = 1;\n{}", kernel("let y = A0;"))
    };
    x86(&module(63), "k");
    for n in [64, 5000] {
        assert_eq!(kernite::x86_str(&module(n), "k").unwrap_err(), "65:18: this nests 64 deep at most", "{n}");
    }
}

/// A type nests 64 deep at most, and `Name<T>` and `[T; N]` are a level each, in a pointer to them
/// as well: the parser refuses the level past it where it starts, however long the row, before it
/// reads the rest. A type of 64 is read, and then it is the machine that has no such variable. A
/// result is a type, and nests as far.
#[test]
fn a_type_nests_64_deep_at_most() {
    for (pointer, at) in [("", "2:134"), ("*const ", "2:141")] {
        let body = |n: usize| format!("let x: {pointer}{}u32{};", "a<".repeat(n), ">".repeat(n));
        assert!(refused(&body(63)).starts_with("2:8: x86 variables are "), "{pointer}");
        for n in [64, 5000] {
            assert_eq!(refused(&body(n)), format!("{at}: this nests 64 deep at most"), "{pointer}{n}");
        }
    }
    let result = |n: usize| {
        let ty = format!("{}u32{}", "a<".repeat(n), ">".repeat(n));
        format!("unsafe extern \"sysv64\" fn k(p: *const u8) -> *const {ty} {{ ret(p); }}")
    };
    kernite::x86_str(&result(63), "k").unwrap();
    for n in [64, 5000] {
        assert_eq!(kernite::x86_str(&result(n), "k").unwrap_err(), "1:179: this nests 64 deep at most", "-> {n}");
    }
}

/// A row inside brackets inside a row is counted whole: 62 fields in parentheses in 62 fields, and
/// so on, is 3906 deep, which the evaluator would follow, with no count of its own for a field,
/// until a stack of 6 MiB ran out. It is refused at the first level past 64.
#[test]
fn a_row_inside_brackets_inside_a_row_is_counted_whole() {
    let deep = (0..62).fold("x".to_string(), |e, _| format!("({e}){}", ".f".repeat(62)));
    assert_eq!(refused(&format!("let y = {deep};")), "2:194: this nests 64 deep at most");
}

/// A block, a `for` and an `if` are a level for what they hold, and count with the expression in
/// them: 62 of them around `1 + 1` compile, and 63 are refused where the 65th level starts.
#[test]
fn blocks_loops_and_ifs_nest_with_their_expressions_64_deep_at_most() {
    for (open, at) in [("{", "2:72"), ("if true {", "2:576"), ("for i in 0..1 {", "2:940")] {
        let body = |n: usize| format!("{}let y = 1 + 1;{}", open.repeat(n), "}".repeat(n));
        x86(&kernel(&body(62)), "k");
        assert_eq!(refused(&body(63)), format!("{at}: this nests 64 deep at most"), "{open}");
    }
}

/// A helper's blocks count with its caller's, for the stack is one: 62 helpers, each calling the
/// next from a `for`, compile, and the 63rd `for` is refused at its range.
#[test]
fn the_loops_of_helpers_that_call_helpers_nest_64_deep_at_most() {
    let module = |n: usize| {
        let helpers: String = (0..n)
            .map(|i| {
                let inside = if i + 1 < n { format!("h{}();", i + 1) } else { "let y = 1;".to_string() };
                format!("fn h{i}() {{ for i in 0..1 {{ {inside} }} }}\n")
            })
            .collect();
        format!("{helpers}{}", kernel("h0();"))
    };
    x86(&module(62), "k");
    assert_eq!(kernite::x86_str(&module(63), "k").unwrap_err(), "63:21: this nests 64 deep at most");
}

/// The deepest kernel the rules allow, helpers 64 deep and a sum of 64 terms in the last, compiles
/// on the stack of a test, 2 MiB: 64 is what fits it, with room to spare.
#[test]
fn the_deepest_kernel_fits_the_stack_of_a_test() {
    let helpers: String = (0..64)
        .map(|i| {
            let inside = if i < 63 { format!("h{}();", i + 1) } else { format!("let y = 1{};", " + 1".repeat(63)) };
            format!("fn h{i}() {{ {inside} }}\n")
        })
        .collect();
    x86(&format!("{helpers}{}", kernel("h0();")), "k");
}

/// The step past the most, 4194304, is refused where it is taken, in seconds. Without the count,
/// loops that nest take 2^40 turns, a helper that calls itself twice takes 2^40 calls, and a const
/// of 2^20 elements shown 2^20 times takes 2^40 elements: days, or an abort out of memory.
#[test]
fn work_past_the_most_steps_is_refused_where_the_step_past_it_is_taken() {
    let twice =
        "fn h<const N: usize>() {\n    if N > 0 {\n        h::<{ N - 1 }>();\n        h::<{ N - 1 }>();\n    }\n}\n";
    let table = "const A: [u32; 1 << 20] = [0; 1 << 20]; ";
    let rows = "const M: [[u32; 1024]; 1024] = [[0; 1024]; 1024]; ";
    let cases = [
        ("loops that nest", kernel("for i in 0..1 << 20 { for j in 0..1 << 20 {} }"), "2:33"),
        ("a note of a const", kernel(&format!("{table}for i in 0..1 << 20 {{ note(\"{{A}}\"); }}")), "2:68"),
        ("a helper that calls itself twice", format!("{twice}{}", kernel("h::<40>();")), "4:9"),
        ("an array built in a loop", kernel("for i in 0..1 << 20 { let a = [0; 1 << 20]; }"), "2:31"),
        (
            "a const copied in a loop",
            kernel(&format!("{table}for i in 0..1 << 20 {{ let b: [u32; 1 << 20] = A; }}")),
            "2:87",
        ),
        ("a method's array in a loop", kernel(&format!("{table}for i in 0..1 << 20 {{ let b = A.rev(); }}")), "2:73"),
        (
            "the rows of a const in a loop",
            kernel(&format!("{rows}for i in 0..1 << 20 {{ for row in M {{}} }}")),
            "2:84",
        ),
        ("a row indexed in a loop", kernel(&format!("{rows}for i in 0..1 << 20 {{ let row = M[0]; }}")), "2:84"),
        (
            "data in memory in a loop",
            kernel(&format!("{table}let x: gpr; for i in 0..1 << 20 {{ lea(x, [A]); }}")),
            "2:83",
        ),
    ];
    for (name, src, at) in cases {
        let want = format!("{at}: this takes more than 4194304 steps to work out");
        assert_eq!(kernite::x86_str(&src, "k").unwrap_err(), want, "{name}");
    }
}

/// Work of exactly the most steps compiles, and a note after it is the step past, refused. A loop
/// of `n` turns is `2n` steps, its range and its turns, so two loops fill what each case leaves, a
/// range holding 2^20 at most. A case would pass the most if it counted one step more than GUIDE.md
/// says, and fall short if it counted one less: loops add, and do not multiply; a number read by
/// name or worked out costs none; an array listed costs its items, not its elements; an `[x; n]`,
/// an array a method makes and one copied from a name, by a typed `let` or by `for`, cost their
/// elements; and a label costs one, a number an index takes or a note shows costs one, and an array
/// there costs its elements.
#[test]
fn work_within_the_most_steps_compiles() {
    let cases = [
        ("a loop's range and turns", "for i in 0..1048576 {}", 2 * 1048576),
        ("a number read by name or worked out", "for i in 0..1000 { let b = i + 1; }", 2 * 1000),
        ("an array listed", "let a = [[0, 0], [0, 0]];", 2 + 2 + 2),
        ("an `[x; n]`", "let a = [0; 1000];", 1000),
        ("an array a method makes", "let a = [0; 1000]; let b = a.rev();", 1000 + 1000),
        ("an array copied by a typed `let`", "let a = [0; 1000]; let b: [u32; 1000] = a;", 1000 + 1000),
        ("an array copied by `for`", "let a = [0; 1000]; for x in a {}", 1000 + 1000 + 1000),
        ("a label", "for i in 0..1000 { label(a); }", 3 * 1000),
        ("a number an index takes", "let a = [0, 0]; for i in 0..1000 { let b = a[0]; }", 2 + 3 * 1000),
        ("an array an index takes", "let a = [[0; 1000]; 2]; let b = a[0];", 1000 + 2 * 1000 + 1000),
        ("a number a note shows", "let a = 7; for i in 0..1000 { note(\"{a}\"); }", 4 * 1000),
        ("an array a note shows", "let a = [0; 1000]; note(\"{a}\"); note(\"{a}\");", 1000 + 2 * (1 + 1000)),
    ];
    for (name, work, steps) in cases {
        let turns = (4194304 - steps) / 2;
        let body = format!("{work} for k in 0..{} {{}} for l in 0..{} {{}}", turns / 2, turns - turns / 2);
        if let Err(e) = kernite::x86_str(&kernel(&body), "k") {
            panic!("{name}: {e}");
        }
        let past = kernite::x86_str(&kernel(&format!("{body} note(\"\");")), "k");
        let want = format!("2:{}: this takes more than 4194304 steps to work out", body.len() + 2);
        assert_eq!(past.err(), Some(want), "{name}");
    }
}

/// An operator of two characters, `<<`, `&&`, `==`, `!=`, `<=` and `..`, is one token only if no space, line
/// break or comment is between its characters: with one, they are two, and no expression begins with the
/// second.
#[test]
fn an_operator_of_two_characters_has_nothing_between_them() {
    let constant = |op: &str| {
        let src = format!(
            "const A: u32 = 8;\nconst B: u32 = A {op} 1;\n\
             unsafe extern \"sysv64\" fn k() -> u64 {{ let r: gpr; mov(r, B); ret(r); }}\n"
        );
        kernite::x86_str(&src, "k").map(|l| l.code().collect::<Vec<_>>().join("; "))
    };
    assert_eq!(constant("<<").unwrap(), "mov rax, 16; ret");
    assert_eq!(constant(">>").unwrap(), "mov rax, 4; ret");
    for op in ["&&", "||"] {
        assert_eq!(constant(op).unwrap_err(), format!("2:18: `{op}` has no meaning between an integer and an integer"));
    }
    for op in ["==", "!=", "<=", ">="] {
        assert_eq!(constant(op).unwrap_err(), "2:18: this is a bool, not u32");
    }
    assert_eq!(constant("..").unwrap_err(), "2:18: this is an array, not u32");
    for (op, at, found) in [
        ("< <", "2:20", "`<`"),
        ("<\n<", "3:1", "`<`"),
        ("< /* a comment */ <", "2:36", "`<`"),
        ("> >", "2:20", "`>`"),
        ("& &", "2:20", "`&`"),
        ("| |", "2:20", "`|`"),
        ("< =", "2:20", "`=`"),
        ("> =", "2:20", "`=`"),
    ] {
        assert_eq!(constant(op).unwrap_err(), format!("{at}: expected an expression, found {found}"), "{op}");
    }
    assert_eq!(constant(". .").unwrap_err(), "2:20: expected a name, found `.`");
    assert_eq!(constant("= =").unwrap_err(), "2:18: expected `;`, found `=`");
    assert_eq!(constant("! =").unwrap_err(), "2:18: expected `;`, found `!`");
}
