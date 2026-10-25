//! The language, run by the targets it is compiled for: helpers inlined, loops unrolled,
//! labels numbered afresh, values computed as Rust computes them, memory in brackets,
//! predicates guarding, and errors where the trouble is.

/// Kernel `name` of `src`, for x86: its listing, as a snapshot holds it.
fn x86(src: &str, name: &str) -> String {
    kgen::x86_str(src, name).unwrap_or_else(|e| panic!("{e}")).text()
}

/// The `.entry` of the one kernel in `src`, in PTX, from its name on.
fn ptx(src: &str) -> String {
    let module = kgen::ptx_str(src).unwrap_or_else(|e| panic!("{e}"));
    module.split(".visible .entry ").nth(1).expect("an entry").to_string()
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

#[test]
fn sixteen_bit_floats_are_values_and_data() {
    let src = r#"
        const THIRD: f16 = 1.0 / 3.0;

        pub unsafe extern "sysv64" fn h() {
            const SUM: f16 = 0.1f16 + 0.2f16;
            const B: bf16 = 1.0bf16 / 3.0;
            const HUGE: f16 = 65520.0;
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
    let want = "# h() -> ()
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
            "1:28: x86 variables are `gpr`, `ymm` and `xmm`, not `shared<[f32; 8], 16>`",
        ),
        ("let t: [f32];", "1:8: variables are declared so many at once: `let t: [u32; 4];`"),
        // The machine is given a name as it writes it, not as Rust must.
        ("let v: ymm; vaddps._2d__x(v);", "1:20: x86 has no instruction `vaddps.2d::x` that kgen knows"),
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
        ("let g: gpr; mov(g, k);", "1:20: an x86 kernel calls no function; `k` names an ABI, so it is one"),
        (
            "k();",
            "1:1: `k` names an ABI, so it is compiled on its own, not inlined as a helper is: an instruction calls \
             it, as `call(r, k, (a, b))` does in PTX",
        ),
    ];
    for (body, want) in cases {
        let src = format!(
            "unsafe extern \"sysv64\" fn k() {{\n{body}\n}}\nfn two(a: ymm, b: ymm) {{}}\nfn deep() {{ deep(); }}\n\
             const A: usize = B;\nconst B: usize = A;\n"
        );
        let got = kgen::x86_str(&src, "k").unwrap_err();
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
        let got = kgen::ptx_str(&format!("unsafe extern \"ptx-kernel\" fn k() {{\n{body}\n}}\n")).unwrap_err();
        assert_eq!(got, *want, "{body}");
    }
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
    assert_eq!(kgen::x86_str(src, "k").unwrap_err(), "7:18: `secret` is not defined");
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
