//! Kernels in the language, compiled for x86: what they come to, and what the errors say; and
//! what the builder refuses.

use kgen_x86::Asm;

/// The kernels of `examples/my-app`, and the same dot product with sixteen accumulators.
const KERNELS: &str = r#"
    /// The sum of the eight floats at `xs`.
    pub unsafe extern "sysv64" fn sum8(xs: *const f32) -> f32 {
        let v: ymm;
        vmovups(v, [xs]);
        ret_sum(v);
    }

    pub unsafe extern "sysv64" fn dot_unroll1(x: *const f32, y: *const f32, n: usize) -> f32 {
        dot::<1>(x, y, n);
    }

    pub unsafe extern "sysv64" fn dot_unroll4(x: *const f32, y: *const f32, n: usize) -> f32 {
        dot::<4>(x, y, n);
    }

    pub unsafe extern "sysv64" fn dot_unroll16(x: *const f32, y: *const f32, n: usize) -> f32 {
        dot::<16>(x, y, n);
    }

    fn dot<const UNROLL: usize>(x: gpr, y: gpr, n: gpr) {
        test(n, n);
        jz(empty);
        let acc: [ymm; UNROLL];
        for k in 0..UNROLL {
            zero(acc[k]);
        }
        label(top);
        for k in 0..UNROLL {
            let xs: ymm;
            vmovups(xs, [x + 32 * k]);
            vfmadd231ps(acc[k], xs, [y + 32 * k]);
        }
        add(x, 32 * UNROLL);
        add(y, 32 * UNROLL);
        sub(n, 8 * UNROLL);
        jnz(top);

        note("the accumulators, in order");
        for k in 1..UNROLL {
            vaddps(acc[0], acc[0], acc[k]);
        }
        ret_sum(acc[0]);

        label(empty);
        let zero: ymm;
        zero(zero);
        ret(zero.xmm);
    }

    fn ret_sum(v: ymm) {
        note("the lanes");
        let high: ymm;
        vextractf128(high.xmm, v, 1);
        let sum: ymm;
        vaddps(sum.xmm, v.xmm, high.xmm);
        let pair: ymm;
        vmovhlps(pair.xmm, sum.xmm, sum.xmm);
        vaddps(sum.xmm, sum.xmm, pair.xmm);
        vmovshdup(pair.xmm, sum.xmm);
        vaddss(sum.xmm, sum.xmm, pair.xmm);
        ret(sum.xmm);
    }
"#;

/// `tests/builder` holds the listings `kgen-x86`'s builder made of these kernels, before the
/// language: what the language writes must compile to the same instructions in the same registers.
#[test]
fn the_examples_kernels_come_out_as_the_builder_wrote_them() {
    let snapshots = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/builder");
    for name in ["sum8", "dot_unroll1", "dot_unroll4"] {
        let listing = kgen::x86_str(KERNELS, name).unwrap();
        let want = std::fs::read_to_string(snapshots.join(format!("{name}.s"))).unwrap();
        assert_eq!(listing.text(), want, "{name}");
    }
}

#[test]
fn sixteen_accumulators_and_a_load_do_not_fit_and_the_error_says_where() {
    let e = kgen::x86_str(KERNELS, "dot_unroll16").unwrap_err();
    assert!(e.starts_with("30:17: no Vec register left for `xs`: all 16 are held by"), "{e}");
}

#[test]
fn up_to_fifteen_accumulators_fit() {
    let mut src = KERNELS.to_string();
    for k in 1..=15 {
        src += &format!(
            "unsafe extern \"sysv64\" fn dot{k}(x: *const f32, y: *const f32, n: usize) -> f32 {{ dot::<{k}>(x, y, n); }}\n"
        );
    }
    for k in 1..=15 {
        kgen::x86_str(&src, &format!("dot{k}")).unwrap_or_else(|e| panic!("{k} accumulators: {e}"));
    }
}

/// The kernel `k` around `body`, compiled, or the error.
fn compile(params: &str, body: &str) -> Result<String, String> {
    let src = format!("unsafe extern \"sysv64\" fn k({params}) {{\n{body}\n}}\n");
    kgen::x86_str(&src, "k").map(|l| l.code().collect::<Vec<_>>().join("\n"))
}

#[test]
fn memory_operands_take_every_form_x86_has() {
    let code = compile(
        "x: *const f32, i: usize",
        "let v: ymm;
         vmovups(v, [x + i * 4 + 16]);
         vmovups([x - 32], v);
         vbroadcastss(v, [2.0]);
         vmulps(v, v, [[0.5; 8]]);
         let g: gpr;
         mov(g.r32, [x + i * 4]);
         lea(g, [x + i * 8]);",
    )
    .unwrap();
    let want = "vmovups ymm0, ymmword ptr [rdi + rsi*4 + 16]
vmovups ymmword ptr [rdi - 32], ymm0
vbroadcastss ymm0, dword ptr [rip + {c0}]
vmulps ymm0, ymm0, ymmword ptr [rip + {c1}]
mov eax, dword ptr [rdi + rsi*4]
lea rax, [rdi + rsi*8]
vzeroupper
ret";
    assert_eq!(code, want);
}

#[test]
fn a_wrong_operand_is_named_by_what_the_instruction_wants() {
    let cases = [
        ("let v: ymm; vaddps(v, v.xmm, v);", "2:25: `vaddps` wants a ymm; this is an xmm"),
        ("let v: ymm; vaddps(v, v);", "2:13: `vaddps` takes 3 operands, not 2"),
        ("let v: ymm; vmovups(v, [1.0]);", "2:24: this reads 32 bytes of a constant of 4"),
        ("let v: ymm; vmovups(v, [x + x * 3]);", "2:24: x86 wants a base beside a scaled index"),
        ("let i: gpr; let v: ymm; vmovups(v, [x + i * 3]);", "2:36: x86 scales an index by 1, 2, 4 or 8, not 3"),
        ("let v: ymm; vaddps(v, v, 1.5);", "2:26: x86 takes floats from memory: `[1.5]`"),
        ("vfrobps(x);", "2:1: x86 has no instruction `vfrobps` that kgen knows"),
        ("let v: ymm; vaddps(v, v, w);", "2:26: `w` is not defined"),
        ("let v: zmm;", "2:8: x86 variables are `gpr`, `ymm` and `xmm`, not `zmm`"),
        ("let v: ymm; vaddps(v, v, v);", "2:5: `v` is read before anything writes it"),
        ("shl(x, 64);", "2:8: `shl` wants an integer, 0 to 63; this is the integer 64"),
        ("sar(x, -1);", "2:8: `sar` wants an integer, 0 to 63; this is the integer -1"),
        ("let v: ymm; vpslld(v, v, 256);", "2:26: `vpslld` wants an integer, 0 to 255; this is the integer 256"),
        (
            "let v: ymm; vextractf128(v.xmm, v, 2);",
            "2:36: `vextractf128` wants an integer, 0 to 1; this is the integer 2",
        ),
        (
            "let v: ymm; vinsertf128(v, v, v.xmm, 2);",
            "2:38: `vinsertf128` wants an integer, 0 to 1; this is the integer 2",
        ),
        ("let v: ymm; vcmpps(v, v, v, 32);", "2:29: `vcmpps` wants a control byte, 0 to 31; this is the integer 32"),
        ("let v: ymm; vroundps(v, v, 16);", "2:28: `vroundps` wants a control byte, 0 to 15; this is the integer 16"),
        (
            "let v: ymm; vpblendd(v.xmm, v.xmm, v.xmm, 16);",
            "2:43: `vpblendd` wants a control byte, 0 to 15; this is the integer 16",
        ),
        (
            "let v: ymm; vpblendd(v, v, v, 256);",
            "2:31: `vpblendd` wants a control byte, 0 to 255; this is the integer 256",
        ),
    ];
    for (body, want) in cases {
        assert_eq!(compile("x: *const f32", body).unwrap_err(), want, "{body}");
    }
}

/// An immediate x86 reads in part is no more than it reads: 6 bits of a shift's count, 5 of
/// `vcmpps`'s control byte, 1 of a ymm's half, and a bit for each lane `vpblendd` blends.
#[test]
fn an_immediate_is_what_x86_reads_of_it() {
    let code = compile(
        "x: *const f32",
        "let v: ymm; vmovups(v, [x]); shl(x, 63); vcmpps(v, v, v, 31); vextractf128(v.xmm, v, 1); vpslld(v, v, \
         255); vpblendd(v, v, v, 0xff); vpblendd(v.xmm, v.xmm, v.xmm, 0xf);",
    )
    .unwrap();
    let want = "vmovups ymm0, ymmword ptr [rdi]
shl rdi, 63
vcmpps ymm0, ymm0, ymm0, 0x1f
vextractf128 xmm0, ymm0, 1
vpslld ymm0, ymm0, 255
vpblendd ymm0, ymm0, ymm0, 0xff
vpblendd xmm0, xmm0, xmm0, 0x0f
vzeroupper
ret";
    assert_eq!(code, want);
}

/// The builder takes no more of an immediate than x86 reads, as the language does not.
#[test]
fn the_builder_refuses_what_x86_would_not_read() {
    let refused = |want: &str, build: fn(&mut Asm)| {
        let e = std::panic::catch_unwind(|| build(&mut Asm::new("k"))).expect_err(want);
        assert_eq!(e.downcast_ref::<String>().map(String::as_str), Some(want));
    };
    refused("shl by 64: x86 shifts a gpr by 0 to 63", |a| {
        let g = a.gpr("g");
        a.shl(g, 64);
    });
    refused("shr by 255: x86 shifts a gpr by 0 to 63", |a| {
        let g = a.gpr("g");
        a.shr(g, 255);
    });
    refused("vextractf128 of lane 2: a ymm has 2", |a| {
        let v = a.ymm("v");
        a.vextractf128(v.xmm(), v, 2);
    });
    refused("vinserti128 of lane 3: a ymm has 2", |a| {
        let v = a.ymm("v");
        a.vinserti128(v, v, v.xmm(), 3);
    });
    refused("vpblendd of 4 lanes by 0x10", |a| {
        let v = a.ymm("v").xmm();
        a.vpblendd(v, v, v, 0x10);
    });
    let mut a = Asm::new("k");
    let (g, v) = (a.gpr("g"), a.ymm("v"));
    a.shl(g, 63);
    a.vextractf128(v.xmm(), v, 1);
    a.vpblendd(v, v, v, 0xff);
    a.vpblendd(v.xmm(), v.xmm(), v.xmm(), 0x0f);
}

#[test]
fn a_kernel_takes_what_sysv_passes_in_registers() {
    let e = compile("a: u32", "").unwrap_err();
    assert_eq!(e, "1:32: SysV leaves the high bits of a `u32` as they were; take a usize or a u64");
    let code = compile("a: f32, p: *mut f32, b: f32", "vmovss([p], b.xmm); vaddss(b.xmm, a.xmm, b.xmm);").unwrap();
    assert_eq!(code, "vmovss dword ptr [rdi], xmm1\nvaddss xmm1, xmm0, xmm1\nvzeroupper\nret");
}

/// Rust never sees a kernel's body run, so kgen says which of the module's constants it read:
/// what the macro marks as used, so that only the constants nothing reads are dead code.
#[test]
fn a_kernel_names_the_constants_it_reads() {
    let src = r#"
        const ADD: usize = 1;
        const STORE: usize = 0;
        const OFFSET: usize = 8 * LANE;
        const LANE: usize = 4;
        const SPARE: usize = 2;
        const UNREAD: usize = 3;

        pub unsafe extern "sysv64" fn stores(x: *mut f32, y: *const f32) {
            let v: ymm;
            vmovups(v, [y]);
            then::<STORE>(x, v);
        }

        pub unsafe extern "sysv64" fn adds(x: *mut f32, y: *const f32) {
            let v: ymm;
            vmovups(v, [y]);
            then::<ADD>(x, v);
        }

        fn then<const HOW: usize>(x: gpr, v: ymm) {
            if HOW == ADD {
                vaddps(v, v, [x + OFFSET]);
            }
            vmovups([x], v);
        }

        fn spare(x: gpr) {
            add(x, SPARE);
        }
    "#;
    let module = kgen::lang::parse(src).unwrap();
    let reads = |name| kgen::x86(&module, kgen::function(&module, name).unwrap()).unwrap().consts;
    // `STORE` as a helper's value and `ADD` in the `if` that chooses, in the module's order;
    // not `OFFSET`, in the branch not taken, nor `SPARE`, in a helper not called.
    assert_eq!(reads("stores"), ["ADD", "STORE"]);
    // `OFFSET` in the branch taken, and `LANE` through it.
    assert_eq!(reads("adds"), ["ADD", "OFFSET", "LANE"]);
}
