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
        let err = |src: &str| match src.contains("ptx-kernel") {
            true => kernite::ptx_str(src).unwrap_err(),
            false => kernite::x86_str(src, "k").unwrap_err(),
        };
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
        let err = match src.contains("ptx-kernel") {
            true => kernite::ptx_str(src).unwrap_err(),
            false => kernite::x86_str(src, "k").unwrap_err(),
        };
        assert_eq!(err, format!("{at}: {said}"), "{src}");
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
            "1:60: expected an expression, found `:`".into(),
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

/// A constant of a module works out on its own, in terms of the module's others, before or
/// after it, as a module of consts has each checked; or says why it is no value known now.
#[test]
fn a_constant_works_out_on_its_own() {
    let src = "const D: usize = E / H;\nconst E: usize = 16;\nconst H: usize = 4;\nconst W: [f32; 2] = [0.5, -1.0];\n\
               const LOOP: u32 = LOOP + 1;\nconst BAD: u32 = 1.5;\nconst GONE: u32 = NOPE;\n\
               unsafe extern \"sysv64\" fn k() {}\n";
    let module = kernite::lang::parse(src).unwrap();
    let value = |name: &str| {
        let c = module
            .items
            .iter()
            .find_map(|item| match &item.kind {
                kernite::lang::ItemKind::Const(c) if c.name == name => Some(c),
                _ => None,
            })
            .unwrap();
        kernite::lang::constant(&module, c).map(|v| v.to_string()).map_err(|e| e.at(src))
    };
    let cases = [
        ("D", Ok("4")),
        ("W", Ok("[0.5, -1]")),
        ("LOOP", Err("5:19: `LOOP` is defined in terms of itself")),
        ("BAD", Err("6:18: this is an f32, not u32")),
        ("GONE", Err("7:19: `NOPE` is not defined")),
    ];
    for (name, want) in cases {
        assert_eq!(value(name), want.map(String::from).map_err(String::from), "{name}");
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
/// range holding 2^20 at most. A case would pass the most if it counted one step more than README
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
