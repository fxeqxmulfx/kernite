//! Kernels in the language, compiled for x86: what they come to, and what the errors say; every
//! form of every instruction kgen knows, which `llvm-mc` assembles and GNU as takes on a machine
//! of its feature; and what the builder refuses.

use std::path::PathBuf;
use std::process::Command;
use std::sync::Mutex;

use kgen_x86::forms::{self, Form, Kind};
use kgen_x86::{Asm, Feature};

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

/// Sixteen accumulators and the load beside them take 17 of the 16 vector registers: the two
/// that cost least to keep on the stack, of those that hold one through the loop, spill.
#[test]
fn sixteen_accumulators_and_a_load_spill_two_and_the_listing_says_where() {
    let text = kgen::x86_str(KERNELS, "dot_unroll16").unwrap().text();
    let header = "# dot_unroll16(gpr, gpr, gpr) -> vec
# needs: avx, fma
# frame: 64 bytes aligned to 32, saves []
# acc15, spilled: 32 bytes at rsp
# acc14, spilled: 32 bytes at rsp + 32
";
    assert!(text.starts_with(header), "{text}");
    let last = "    vmovups ymm14, ymmword ptr [rdi + 480]
    # acc15 reloaded
    vmovups ymm15, ymmword ptr [rsp]
    vfmadd231ps ymm15, ymm14, ymmword ptr [rsi + 480]
    # acc15 spilled
    vmovups ymmword ptr [rsp], ymm15
    add rdi, 512
";
    assert!(text.contains(last), "{text}");
}

#[test]
fn up_to_fifteen_accumulators_fit_without_spilling() {
    let mut src = KERNELS.to_string();
    for k in 1..=15 {
        src += &format!(
            "unsafe extern \"sysv64\" fn dot{k}(x: *const f32, y: *const f32, n: usize) -> f32 {{ dot::<{k}>(x, y, n); }}\n"
        );
    }
    for k in 1..=15 {
        let listing = kgen::x86_str(&src, &format!("dot{k}")).unwrap_or_else(|e| panic!("{k} accumulators: {e}"));
        assert!(listing.frame.slots.is_empty(), "{k} accumulators:\n{}", listing.text());
    }
}

/// Thirteen sums, the pointer and the count take 15 of the 14 general registers the allocator
/// has. The count spills, a parameter stored from where it arrives; then a sum, for the count to
/// be loaded beside the others. Its store goes between the `sub` and the `jnz` that reads its
/// flags, which a `mov` leaves as they were.
#[test]
fn a_general_register_spills_with_mov_a_parameter_from_where_it_arrives() {
    let src = "unsafe extern \"sysv64\" fn k(p: *mut u64, n: usize) {
        let g: [gpr; 13];
        for k in 0..13 { mov(g[k], [p + 8 * k]); }
        label(top);
        for k in 0..13 { add(g[k], [p + 8 * k]); }
        sub(n, 1);
        jnz(top);
        for k in 0..13 { mov([p + 8 * k], g[k]); }
    }";
    let text = kgen::x86_str(src, "k").unwrap().text();
    let parts = [
        "# frame: 16 bytes aligned to 8, saves [rbx, r12, r13, r14, r15]
# n, spilled: 8 bytes at rsp
# g12, spilled: 8 bytes at rsp + 8
# p=rdi n=rsi,r15 g0=rax",
        "    sub rsp, 16
    # n spilled
    mov qword ptr [rsp], rsi
    mov rax, qword ptr [rdi]",
        "    # n reloaded
    mov r15, qword ptr [rsp]
    sub r15, 1
    # n spilled
    mov qword ptr [rsp], r15
    jnz 2b",
    ];
    for part in parts {
        assert!(text.contains(part), "{part}\nin\n{text}");
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

/// Memory is as much as the other operands say, or, where an instruction's forms differ by that
/// alone, as Intel's word before the brackets says. A write to `r16` or `r8` keeps the rest of
/// the register, so it needs a value there first.
#[test]
fn memory_is_as_much_as_the_operands_or_a_word_before_it_say() {
    let code = compile(
        "x: *mut u64",
        "let g: gpr; zero(g); movzx(g.r32, byte[x]); add(dword[x], 1); mov(g.r16, word[x]); mov(g.r8, [[7u8]]); \
         add(qword[x], g); add([x], g);",
    )
    .unwrap();
    let want = "xor eax, eax
movzx eax, byte ptr [rdi]
add dword ptr [rdi], 1
mov ax, word ptr [rdi]
mov al, byte ptr [rip + {c0}]
add qword ptr [rdi], rax
add qword ptr [rdi], rax
ret";
    assert_eq!(code, want);
}

/// A gather reads memory at a vector of indices, `[p + v * 4]`, under a mask, and needs AVX2.
#[test]
fn a_gather_reads_memory_at_a_vector_of_indices() {
    let src = "unsafe extern \"sysv64\" fn k(x: *const f32) {
        let v: ymm; let m: ymm; let i: ymm;
        zero(v); ones(m); zero(i);
        vpgatherdd(v, [x + i * 4], m);
        vpgatherdq(v, [x + i.xmm * 8], m);
        vgatherqps(v.xmm, [x + i * 4], m.xmm);
    }";
    let listing = kgen::x86_str(src, "k").unwrap();
    assert_eq!(listing.features, [Feature::Avx, Feature::Avx2]);
    let code: Vec<&str> = listing.code().collect();
    let want = [
        "vxorps xmm0, xmm0, xmm0",
        "vpcmpeqd ymm1, ymm1, ymm1",
        "vxorps xmm2, xmm2, xmm2",
        "vpgatherdd ymm0, dword ptr [rdi + ymm2*4], ymm1",
        "vpgatherdq ymm0, qword ptr [rdi + xmm2*8], ymm1",
        "vgatherqps xmm0, dword ptr [rdi + ymm2*4], xmm1",
        "vzeroupper",
        "ret",
    ];
    assert_eq!(code, want);
}

/// SysV leaves MXCSR's control bits as the caller set them, so a kernel that loads MXCSR stores
/// the caller's first, in its frame, and loads it back before it returns.
#[test]
fn a_kernel_that_loads_mxcsr_gives_the_callers_back() {
    let src = "unsafe extern \"sysv64\" fn k(x: *const u32) {\nvldmxcsr([x]);\n}\n";
    let want = "# k(gpr) -> ()
# needs: avx
# frame: 4 bytes aligned to 4, saves []
# mxcsr: 4 bytes at rsp
# x=rdi
    push rbp
    mov rbp, rsp
    sub rsp, 4
    # the caller's mxcsr, which it gets back
    vstmxcsr dword ptr [rsp]
    vldmxcsr dword ptr [rdi]
    vldmxcsr dword ptr [rsp]
    mov rsp, rbp
    pop rbp
    ret
";
    assert_eq!(kgen::x86_str(src, "k").unwrap().text(), want);
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
        ("let v: zmm;", "2:8: x86 variables are `gpr`, `ymm`, `xmm` and stack memory, `local<..>`; not `zmm`"),
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
        ("let v: ymm; vbroadcastsd(v.xmm, [x]);", "2:28: `vbroadcastsd` wants a ymm; this is an xmm"),
        ("let v: ymm; vpermps(v, v, v.xmm);", "2:29: `vpermps` wants a ymm; this is an xmm"),
        ("let v: ymm; vpermq(v, v.xmm, 1);", "2:25: `vpermq` wants a ymm; this is an xmm"),
        (
            "let g: gpr; zero(g); vmovd(g.r16, g);",
            "2:30: `vmovd` wants a gpr (whole or as r32) or an xmm; this is a gpr's r16",
        ),
        (
            "let g: gpr; zero(g); movzx(g, 1);",
            "2:31: `movzx` wants a gpr (as r16 or r8) or memory of 1 or 2 bytes; this is the integer 1",
        ),
        ("let g: gpr; mov(g.r8, 1);", "2:5: `g` is read before anything writes it"),
        (
            "let g: gpr; movzx(g.r32, [x]);",
            "2:26: `movzx` reaches 1 or 2 bytes of memory here: say which, `byte[..]` or `word[..]`",
        ),
        (
            "add([x], 1);",
            "2:5: `add` reaches 1, 2, 4 or 8 bytes of memory here: say which, `byte[..]`, `word[..]`, `dword[..]` or \
             `qword[..]`",
        ),
        (
            "let g: gpr; mov(g, dwrd[x]);",
            "2:24: `dwrd` is not defined; before brackets, x86 names how much memory an instruction reaches: `byte`, \
             `word`, `dword`, `qword`, `xmmword` or `ymmword`",
        ),
        ("let g: gpr; zero(g); mov(g, byte[x]);", "2:33: `mov` wants 8 bytes of memory; this is 1 byte of memory"),
        ("let v: ymm; vmovups(v, qword[x]);", "2:29: `vmovups` wants 32 bytes of memory; this is 8 bytes of memory"),
        (
            "let g: gpr; lea(g, qword[x]);",
            "2:25: `lea` takes an address and reaches no memory: no size before the brackets",
        ),
        (
            "let v: ymm; zero(v); vmovups([[1.0; 8]], v);",
            "2:30: `vmovups` writes this memory, which holds data known now and is read-only",
        ),
        (
            "let v: ymm; let m: ymm; zero(v); ones(m); vpgatherdd(v, [x + v * 4], m);",
            "2:57: `vpgatherdd` faults unless its result, its indices and its mask are three registers; `v` is its \
             result and its indices",
        ),
        (
            "let v: ymm; let m: ymm; zero(v); ones(m); vpgatherdd(v, [x + m * 4], m);",
            "2:70: `vpgatherdd` faults unless its result, its indices and its mask are three registers; `m` is its \
             indices and its mask",
        ),
        (
            "let v: ymm; let m: ymm; zero(v); ones(m); vpgatherdd(v, [x + x * 4], m);",
            "2:57: `vpgatherdd` wants 4 bytes of memory at a ymm of indices; this is memory at an index in a gpr",
        ),
        (
            "let v: ymm; let m: ymm; let i: ymm; zero(v); ones(m); zero(i); vpgatherdd(v, [i * 4], m);",
            "2:78: x86 adds a gpr to a vector of indices, as in `[p + v * 4]`",
        ),
        ("jmp(x);", "2:5: `jmp` wants a label; this is a gpr"),
        ("let g: gpr; add(g, top); label(top);", "2:20: a label is for a jump"),
        ("rdtscp();", "2:1: x86 has no instruction `rdtscp` that kgen knows"),
        ("let v: ymm; addps(v.xmm, v.xmm);", "2:13: kgen writes `addps` as AVX does: `vaddps`"),
        ("vzeroupper();", "2:1: kgen does not yet write `vzeroupper`, which clears the upper half of every vector"),
        ("pcmpestri(x, x, 0);", "2:1: kgen writes `pcmpestri` as AVX does: `vpcmpestri`"),
        ("mul(x);", "2:1: kgen does not yet write `mul`, which multiplies `rax` into `rdx:rax`"),
        ("imul(x);", "2:1: kgen does not yet write `imul` of 1 operand, which multiplies `rax` into `rdx:rax`"),
        ("shld(x, x);", "2:1: kgen does not yet write `shld` of 2 operands, which shifts by `cl`"),
        (
            "shl(x, cl);",
            "2:8: kgen does not yet write `shl` by `cl`, a register of its own: shift by a number, or by a gpr with \
             `shlx`, `shrx` or `sarx`",
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
    let reads = |name| kgen::x86(&module, kgen::function(&module, name).unwrap(), &[]).unwrap().consts;
    // `STORE` as a helper's value and `ADD` in the `if` that chooses, in the module's order;
    // not `OFFSET`, in the branch not taken, nor `SPARE`, in a helper not called.
    assert_eq!(reads("stores"), ["ADD", "STORE"]);
    // `OFFSET` in the branch taken, and `LANE` through it.
    assert_eq!(reads("adds"), ["ADD", "OFFSET", "LANE"]);
}

/// Memory of the module's, which its x86 kernels name: a histogram they keep from call to
/// call, and what they read.
const STATICS: &str = r#"
    const N: usize = 16;

    /// How many of each value `count` has seen.
    pub static COUNTS: global<[u32; N]>;
    /// What `weigh` multiplies by, 32 bytes at once.
    static WEIGHTS: constant<[f32; 8], 32> = [1.0, 0.5, 0.25, 2.0, -1.0, 0.0, 3.0, 1.5];
    static SEEN: global<u64> = 7;
    static SIGNED: global<[i8; 3]> = [-1, 2, -128];

    /// Counts `x[i]`, each below 16, for each `i < n`, and how many it has seen in all.
    pub unsafe extern "sysv64" fn count(x: *const u8, n: usize) {
        let (p, v, c): (gpr, gpr, gpr);
        mov(c, [SEEN]);
        add(c, n);
        mov([SEEN], c);
        test(n, n);
        jz(done);
        lea(p, [COUNTS]);
        label(top);
        movzx(v.r32, byte[x]);
        mov(c.r32, [p + v * 4]);
        inc(c);
        mov([p + v * 4], c.r32);
        inc(x);
        dec(n);
        jnz(top);
        label(done);
    }

    /// `x[0..8] * WEIGHTS` twice, the twos `count` has seen in `x[8]`, and the last of
    /// `SIGNED` in `x[9]`.
    pub unsafe extern "sysv64" fn weigh(x: *mut f32) {
        let (v, w): (ymm, ymm);
        vmovups(v, [x]);
        vmovaps(w, [WEIGHTS]);
        vmulps(v, v, w);
        vmulps(v, v, [WEIGHTS]);
        vmovups([x], v);
        let c: gpr;
        mov(c.r32, [COUNTS + 8]);
        mov([x + 32], c.r32);
        movzx(c.r32, byte[SIGNED + 2]);
        mov([x + 36], c.r32);
    }
"#;

/// A kernel reaches a static at a displacement from `rip`, with no register beside it, and
/// takes its address for an index; its listing names each static it names, `{COUNTS}`, for the
/// macro to pass by `sym`.
#[test]
fn a_static_is_reached_at_a_displacement_from_rip() {
    let count = kgen::x86_str(STATICS, "count").unwrap();
    let want = "# count(gpr, gpr) -> ()
# frame: 0 bytes aligned to 0, saves []
# x=rdi n=rsi p=rcx v=rdx c=rax
    mov rax, qword ptr [rip + {SEEN}]
    add rax, rsi
    mov qword ptr [rip + {SEEN}], rax
    test rsi, rsi
    jz 3f
    lea rcx, [rip + {COUNTS}]
2:
    movzx edx, byte ptr [rdi]
    mov eax, dword ptr [rcx + rdx*4]
    inc rax
    mov dword ptr [rcx + rdx*4], eax
    inc rdi
    dec rsi
    jnz 2b
3:
    ret
";
    assert_eq!(count.text(), want);
    assert_eq!(count.statics, ["SEEN", "COUNTS"]);
    let weigh = kgen::x86_str(STATICS, "weigh").unwrap();
    let want = "# weigh(gpr) -> ()
# needs: avx
# frame: 0 bytes aligned to 0, saves []
# x=rdi v=ymm0 w=ymm1 c=rax
    vmovups ymm0, ymmword ptr [rdi]
    vmovaps ymm1, ymmword ptr [rip + {WEIGHTS}]
    vmulps ymm0, ymm0, ymm1
    vmulps ymm0, ymm0, ymmword ptr [rip + {WEIGHTS}]
    vmovups ymmword ptr [rdi], ymm0
    mov eax, dword ptr [rip + {COUNTS} + 8]
    mov dword ptr [rdi + 32], eax
    movzx eax, byte ptr [rip + {SIGNED} + 2]
    mov dword ptr [rdi + 36], eax
    vzeroupper
    ret
";
    assert_eq!(weigh.text(), want);
    assert_eq!(weigh.statics, ["WEIGHTS", "COUNTS", "SIGNED"]);
}

/// Each static is an array of one of Rust's numbers, at the multiple its type says, holding
/// what it says, its bytes the lowest first, or zeros; and the constants it read.
#[test]
fn a_static_is_rusts_numbers_holding_what_it_says() {
    let module = kgen::lang::parse(STATICS).unwrap();
    let statics: Vec<_> = kgen::x86_statics(&module).into_iter().map(Result::unwrap).collect();
    let shape = |s: &kgen::X86Static| (s.name.clone(), s.elem, s.size, s.len, s.align, s.constant);
    let shapes: Vec<_> = statics.iter().map(|s| shape(&s.code)).collect();
    assert_eq!(
        shapes,
        [
            ("COUNTS".to_string(), "u32", 4, 16, 4, false),
            ("WEIGHTS".to_string(), "f32", 4, 8, 32, true),
            ("SEEN".to_string(), "u64", 8, 1, 8, false),
            ("SIGNED".to_string(), "i8", 1, 3, 1, false),
        ]
    );
    let floats: Vec<u8> = [1.0f32, 0.5, 0.25, 2.0, -1.0, 0.0, 3.0, 1.5].iter().flat_map(|x| x.to_le_bytes()).collect();
    assert_eq!(statics[0].code.init, []);
    assert_eq!(statics[1].code.init, floats);
    assert_eq!(statics[2].code.init, [7, 0, 0, 0, 0, 0, 0, 0]);
    assert_eq!(statics[3].code.init, [255, 2, 128]);
    assert_eq!(statics[0].consts, ["N"]);
    assert!(statics[1..].iter().all(|s| s.consts.is_empty()));
}

/// A helper takes memory of the module's as it takes stack memory: of the type it is declared.
#[test]
fn a_helper_takes_a_static_of_its_type() {
    let src = |param: &str| {
        format!(
            "static C: global<[u32; 16]>;\nunsafe extern \"sysv64\" fn k() {{\nbump(C);\n}}\nfn bump(c: {param}) \
             {{\nlet t: gpr;\nmov(t.r32, [c + 8]);\ninc(t);\nmov([c + 8], t.r32);\n}}\n"
        )
    };
    let code = kgen::x86_str(&src("global<[u32; 16]>"), "k").unwrap();
    assert!(code.code().any(|l| l == "mov dword ptr [rip + {C} + 8], eax"), "{}", code.text());
    let e = kgen::x86_str(&src("global<[u32; 8]>"), "k").unwrap_err();
    assert_eq!(e, "3:6: `c`: `C` is memory, `global<[u32; 16]>`; it is no `global<[u32; 8]>`");
    let e = kgen::x86_str(&src("gpr"), "k").unwrap_err();
    assert_eq!(e, "3:6: `c`: `C` is memory, `global<[u32; 16]>`; it is no `gpr`");
    let src = "unsafe extern \"sysv64\" fn k(x: *const f32) {\nload(x);\n}\nfn load(b: constant<u64>) {}\n";
    assert_eq!(kgen::x86_str(src, "k").unwrap_err(), "2:6: `b`: a gpr is no memory, `constant<u64>`");
}

/// What kgen can check of an address in a static, it does, as it does for stack memory; and
/// it refuses a write to constant memory, and an index beside a static's name, which x86 has
/// no address for.
#[test]
fn a_static_is_reached_within_it() {
    let cases = [
        (
            "let g: gpr; zero(g); mov(g.r32, [C + g * 4]);",
            "4:33: x86 reaches memory of the module's at a displacement from `rip`, with no register beside it: \
             `lea(p, [C])` puts the address of `C` in `p`, for `[p + g * 4]`",
        ),
        (
            "let g: gpr; zero(g); mov(g.r32, [g + C]);",
            "4:33: x86 reaches memory of the module's at a displacement from `rip`, with no register beside it: \
             `lea(p, [C])` puts the address of `C` in `p`, for `[p + g]`",
        ),
        ("let g: gpr; mov(g.r32, [C + 64]);", "4:24: `mov` reads 4 bytes at `[C + 64]`, past the end of `C`, 64 bytes"),
        ("let g: gpr; mov(g.r32, [C + 62]);", "4:24: `mov` reads 4 bytes at `[C + 62]`, past the end of `C`, 64 bytes"),
        ("let g: gpr; mov(g.r32, [C - 4]);", "4:24: `mov` reads at `[C - 4]`, before `C` starts"),
        (
            "let g: gpr; mov(g, [C + 5000000000]);",
            "4:20: `mov` reads 8 bytes at `[C + 5000000000]`, past the end of `C`, 64 bytes",
        ),
        (
            "let g: gpr; lea(g, [C + 68]);",
            "4:20: `lea` takes the address `[C + 68]`, more than one past the end of `C`, 64 bytes",
        ),
        (
            "let g: gpr; zero(g); mov([W], g.r32);",
            "4:26: `mov` writes `W`, which is constant memory: the kernels only read it",
        ),
        (
            "let v: ymm; zero(v); vmovaps([W + 32], v);",
            "4:30: `vmovaps` writes `W`, which is constant memory: the kernels only read it",
        ),
        (
            "add(dword[W + 4], 1);",
            "4:10: `add` reads and writes `W`, which is constant memory: the kernels only read it",
        ),
        (
            "let v: ymm; vmovaps(v, [C]);",
            "4:24: `vmovaps` reads 32 bytes at once, at an address that is a multiple of 32; `C` is at a multiple of \
             4: `global<[u32; 16], 32>`, or `vmovups` at any",
        ),
        (
            "let v: ymm; vmovaps(v, [W + 16]);",
            "4:24: `vmovaps` reads 32 bytes at once, at an address that is a multiple of 32; `[W + 16]` is not one",
        ),
        ("let g: gpr; mov(g, [C + W]);", "4:20: an address names one memory at most; `C` and `W` are two"),
        (
            "let g: gpr; mov(g, [x + C * 4]);",
            "4:20: an address adds memory as it is, as in `[C + 8]`; this multiplies `C` by 4",
        ),
        (
            "let g: gpr; mov(g, C);",
            "4:20: `mov` takes no memory as an operand: `lea(p, [C])` puts the address of `C` in `p`, and `[C]` is \
             what it holds",
        ),
        (
            "let x: global<[u32; 4]>;",
            "4:8: global and constant memory are the module's, which a static declares, as `static x: global<[u32; \
             4]>;` beside the kernels: a kernel declares stack memory, `local<..>`",
        ),
        (
            "let x: constant<f32>;",
            "4:8: global and constant memory are the module's, which a static declares, as `static x: \
             constant<f32>;` beside the kernels: a kernel declares stack memory, `local<..>`",
        ),
    ];
    let statics = "static C: global<[u32; 16]>;\nstatic W: constant<[f32; 16], 32> = [1.0; 16];\n";
    for (body, want) in cases {
        let src = format!("{statics}unsafe extern \"sysv64\" fn k(x: *const f32) {{\n{body}\n}}\n");
        assert_eq!(kgen::x86_str(&src, "k").unwrap_err(), want, "{body}");
    }
    // As far as one past the end for `lea`, which reaches nothing, of constant memory as well;
    // and an instruction that reads and writes memory of the module's, which is no constant.
    let src = format!(
        "{statics}unsafe extern \"sysv64\" fn k() {{\nlet g: gpr;\nlea(g, [C + 64]);\nlea(g, [W]);\nadd(dword[C + 8], \
         1);\n}}\n"
    );
    let code = kgen::x86_str(&src, "k").unwrap();
    let want = ["lea rax, [rip + {C} + 64]", "lea rax, [rip + {W}]", "add dword ptr [rip + {C} + 8], 1", "ret"];
    assert_eq!(code.code().collect::<Vec<_>>(), want);
}

/// A static of an x86 kernel's module is of Rust's numbers, as the host reads them, of a size
/// a displacement from `rip` reaches; it holds what it says, and has a name of its own.
#[test]
fn a_static_is_memory_of_rusts_numbers_holding_what_it_says() {
    let cases = [
        (
            "static mut A: global<u32>;",
            "1:8: a static of the module is memory, and its type says what writes it, as in `static COUNTS: \
             global<[u32; 16]>;`: it takes no `mut`",
        ),
        ("static A: global<u32>; static A: global<u32>;", "1:31: `A` names another static of the module as well"),
        ("const A: u32 = 1; static A: global<u32>;", "1:26: `A` names a constant of the module as well"),
        ("static k: global<u32>;", "1:8: `k` names a function of the module as well"),
        (
            "static c0: global<u32>;",
            "1:8: a kernel's listing names a constant `{c0}` and a function `{f0}`, and a static by its name, so one \
             is not named `c0`",
        ),
        (
            "static f12: global<u32>;",
            "1:8: a kernel's listing names a constant `{c0}` and a function `{f0}`, and a static by its name, so one \
             is not named `f12`",
        ),
        (
            "static A: shared<[u32; 4]>;",
            "1:11: a static of the module is memory: `global<[u32; 16]>`, which its kernels read and write, or \
             `constant<[f32; 4]>`, which they only read; not `shared<[u32; 4]>`",
        ),
        (
            "static A: u32;",
            "1:11: a static of the module is memory: `global<[u32; 16]>`, which its kernels read and write, or \
             `constant<[f32; 4]>`, which they only read; not `u32`",
        ),
        ("static A: global<[u32]>;", "1:11: `global<[u32]>` has no size: a static has one, as in `global<[u32; 64]>`"),
        ("static A: global<[u32; 0]>;", "1:11: `global<[u32; 0]>` holds nothing"),
        (
            "static A: global<[f16; 4]>;",
            "1:11: a static holds Rust's numbers, `u8` to `u64`, `i8` to `i64`, `usize`, `isize`, `f32` and `f64`; \
             not `f16`",
        ),
        ("static A: global<[u32; 4], 3>;", "1:11: what an address is a multiple of is a power of two, not 3"),
        ("static A: global<[u32; 4], 2>;", "1:11: an array of `u32` is aligned to 4 at least, not 2"),
        ("static A: constant<[u32; 4], 8192>;", "1:11: a static is aligned to a page at most, 4096; not 8192"),
        (
            "static A: global<[u32; 4], 16, 2>;",
            "1:11: `global<..>` is an array of Rust's numbers and what its address is a multiple of, `global<[f32; \
             64], 32>`, or one, `global<u64>`; not `global<[u32; 4], 16, 2>`",
        ),
        ("static A: global<[u32; M]>;", "1:24: `M` is not defined"),
        (
            "static A: global<[u8; 2147483648]>;",
            "1:11: `global<[u8; 2147483648]>` is more than 2147483647 bytes: a kernel reaches memory of the module's \
             at a displacement from `rip`, of 32 bits, signed",
        ),
        (
            "static A: global<[u64; 18446744073709551615]>;",
            "1:11: `global<[u64; 18446744073709551615]>` is more than 2147483647 bytes: a kernel reaches memory of \
             the module's at a displacement from `rip`, of 32 bits, signed",
        ),
        ("static A: global<[u32; 2]> = [1, 2, 3];", "1:30: `global<[u32; 2]>` holds 2 elements; this has 3"),
        ("static A: global<[u32; 2]> = 1;", "1:30: `global<[u32; 2]>` holds an array of `u32`; this is an integer"),
        ("static A: global<[u32; 2]> = [1.5, 2];", "1:30: element 0 is an f32, not a `u32`"),
        ("static A: global<[u32; 2]> = [1, -1];", "1:34: a u32 cannot be negated"),
        ("static A: global<u32> = [1];", "1:25: this is an array, not a `u32`"),
        ("static A: global<u32> = true;", "1:25: this is a bool, not a `u32`"),
        ("static A: global<u32> = 1u8;", "1:25: this is a `u8`, not a `u32`"),
        ("static A: global<f32> = 1;", "1:25: this is an integer, and an `f32` a float: `1.0`"),
        ("static A: global<f32> = 1.5f64;", "1:25: this is an f64, not an `f32`"),
        ("static A: global<u8> = 256;", "1:24: 256 is out of range for u8"),
        ("static A: global<i8> = 128;", "1:24: 128 is out of range for i8"),
        ("static A: global<u32> = M;", "1:25: `M` is not defined"),
        // A kernel that names one that does not compile says so, and why, where it is declared.
        (
            "static A: global<[u32; 2]> = [1]; unsafe extern \"sysv64\" fn j() { let g: gpr; mov(g.r32, [A]); }",
            "1:30: `global<[u32; 2]>` holds 2 elements; this has 1",
        ),
    ];
    let k = "unsafe extern \"sysv64\" fn k(x: *mut u32) {}";
    for (items, want) in cases {
        assert_eq!(kgen::x86_str(&format!("{items}\n{k}"), "k").unwrap_err(), want, "{items}");
    }
    // The kernel itself says that it names one that does not compile.
    let src =
        "static A: global<[u32; 2]> = [1];\nunsafe extern \"sysv64\" fn j() {\nlet g: gpr;\nmov(g.r32, [A]);\n}\n";
    let module = kgen::lang::parse(src).unwrap();
    let e = kgen::x86(&module, kgen::function(&module, "j").unwrap(), &[]).unwrap_err();
    assert_eq!(e.at(src), "4:13: `A` does not compile, so no kernel names it");
    // Every number of each type, and a name like a listing's that is none of its.
    for items in [
        "static A: global<[i8; 2]> = [-128, 127];",
        "static A: global<[i64; 2]> = [-9223372036854775808, 9223372036854775807];",
        "static A: global<[isize; 2]> = [-0x8000_0000_0000_0000, -1];",
        "static A: global<[u64; 2]> = [0, 18446744073709551615];",
        "static A: global<[f64; 2]> = [1.5, -0.0];",
        "static A: global<[u8; 2147483647]>;",
        "static c: global<u32>; static cf1: global<u32>; static c0x: global<u32>;",
    ] {
        kgen::x86_str(&format!("{items}\n{k}"), "k").unwrap();
    }
}

/// A module's statics are its kernels' memory: the process's for x86 kernels, and the GPU's
/// for PTX ones, or for none. A module of both kinds of kernel and a static is two modules,
/// which its first static says, whatever comes before it.
#[test]
fn a_static_is_its_kernels_memory() {
    use kgen::Owner::{Gpu, Process};
    let x86 = "unsafe extern \"sysv64\" fn k(x: u64) -> u64 { ret(x); }";
    let gpu = "unsafe extern \"ptx-kernel\" fn g(x: *mut u32) { st.global.u32([x], 1); }";
    let func = "extern \"ptx-func\" fn f(x: u32) -> u32 { ret(x); }";
    let helper = "fn h(x: gpr) { inc(x); }";
    let other = "extern \"C\" fn c() {}";
    let (s, t) = ("static S: global<u32>;", "static T: constant<u32> = 1;");
    let whose = |items: &[&str]| {
        let src = items.join("\n");
        kgen::owner(&kgen::lang::parse(&src).unwrap()).map_err(|e| e.at(&src))
    };
    let split = "a static is memory of the module's kernels: the GPU's, for PTX kernels, or the process's, for x86 \
                 ones; and this module has both, so it is two modules, one of each kind of kernel and the statics they \
                 name";
    let cases: [(&[&str], _); 11] = [
        (&[s, x86, t], Ok(Process)),
        (&[x86], Ok(Process)),
        (&[helper, s, x86, other], Ok(Process)),
        (&[s, t], Ok(Gpu)),
        (&[gpu, s], Ok(Gpu)),
        (&[func, s, other], Ok(Gpu)),
        (&[gpu, x86], Ok(Gpu)),
        (&[gpu, x86, func, helper], Ok(Gpu)),
        (&[gpu, s, x86, t], Err(format!("2:8: {split}"))),
        (&[x86, t, s, func], Err(format!("2:8: {split}"))),
        (&[helper, x86, func, s], Err(format!("4:8: {split}"))),
    ];
    for (items, want) in cases {
        assert_eq!(whose(items), want, "{items:?}");
    }
}

/// A cooperative kernel's thread reads its place in the grid from the register after its
/// parameters, and `grid.sync()` is code of the kernel's own, in its body and in a helper's:
/// each thread counts itself in with `lock xadd`, and the last lets the others go, who spin
/// on the generation with `pause`. Its labels are the kernel's own, beside the body's.
#[test]
fn a_cooperative_kernel_waits_for_its_grid_at_a_barrier_of_its_own() {
    let src = r#"
        #[cooperative]
        unsafe extern "sysv64" fn k(x: *mut u64) {
            let (t, u): (gpr, gpr);
            mov(t, grid.thread);
            mov([x + t * 8], t);
            grid.sync();
            note("the next thread's index");
            inc(t);
            cmp(t, grid.threads);
            jnz(read);
            zero(t);
            label(read);
            mov(u, [x + t * 8]);
            wait();
            mov([x + t * 8], u);
        }

        fn wait() {
            grid.sync();
        }
    "#;
    let want = "# k(gpr, gpr) -> ()
# frame: 0 bytes aligned to 0, saves []
# x=rdi t=rax u=rcx grid=rsi barrier=rcx,rdx generation=rdx,r8
# arrived=r8,r9
    mov rax, qword ptr [rsi]
    mov qword ptr [rdi + rax*8], rax
    # grid.sync()
    mov rcx, qword ptr [rsi + 16]
    mov rdx, qword ptr [rcx + 8]
    mov r8, 1
    lock xadd qword ptr [rcx], r8
    inc r8
    cmp r8, qword ptr [rsi + 8]
    jnz 2f
    # the last to come lets the others go
    mov qword ptr [rcx], 0
    inc rdx
    mov qword ptr [rcx + 8], rdx
    jmp 3f
2:
    # the others wait for it
    pause
    cmp rdx, qword ptr [rcx + 8]
    jz 2b
3:
    # the next thread's index
    inc rax
    cmp rax, qword ptr [rsi + 8]
    jnz 4f
    xor eax, eax
4:
    mov rcx, qword ptr [rdi + rax*8]
    # grid.sync()
    mov rdx, qword ptr [rsi + 16]
    mov r8, qword ptr [rdx + 8]
    mov r9, 1
    lock xadd qword ptr [rdx], r9
    inc r9
    cmp r9, qword ptr [rsi + 8]
    jnz 5f
    # the last to come lets the others go
    mov qword ptr [rdx], 0
    inc r8
    mov qword ptr [rdx + 8], r8
    jmp 6f
5:
    # the others wait for it
    pause
    cmp r8, qword ptr [rdx + 8]
    jz 5b
6:
    mov qword ptr [rdi + rax*8], rcx
    ret
";
    assert_eq!(kgen::x86_str(src, "k").unwrap().text(), want);
}

/// The cooperative kernel `k` around `body`, compiled, or the error.
fn cooperative(params: &str, body: &str) -> Result<String, String> {
    let src = format!("#[cooperative]\nunsafe extern \"sysv64\" fn k({params}) {{\n{body}\n}}\n");
    kgen::x86_str(&src, "k").map(|l| l.code().collect::<Vec<_>>().join("\n"))
}

/// What a thread may do with the grid: read where it is, which is memory, and wait for every
/// thread, in a kernel that runs on a grid, `#[cooperative]`, which takes five integers at
/// most and returns nothing.
#[test]
fn the_grid_is_a_cooperative_kernels_to_read_and_wait_for() {
    let cases = [
        (
            "let t: gpr; mov(t, grid.block);",
            "3:25: a thread reads `grid.thread`, its index in the grid, and `grid.threads`, how many there are; not \
             `grid.block`",
        ),
        ("mov(grid.thread, x);", "3:10: `grid.thread` is read, as in `mov(t, grid.thread)`; the launcher writes it"),
        ("inc(grid.threads);", "3:10: `grid.threads` is read, as in `mov(t, grid.threads)`; the launcher writes it"),
        (
            "let t: gpr; lea(t, grid.thread);",
            "3:25: `grid.thread` is read, as in `mov(t, grid.thread)`; `lea` takes the address of the kernel's memory",
        ),
        ("grid.wait();", "3:6: kgen's barrier for the grid is `grid.sync()`; not `grid.wait`"),
        ("grid();", "3:1: kgen's barrier for the grid is `grid.sync()`; not `grid`"),
        ("grid.sync(x);", "3:6: `grid.sync` takes no operands, not 1"),
        (
            "let g: gpr; zero(g); if g { grid.sync(); }",
            "3:34: `grid.sync()` is for every thread of the grid, under no `if`: the grid waits for ever for one that \
             does not come to it",
        ),
    ];
    for (body, want) in cases {
        assert_eq!(cooperative("x: usize", body).unwrap_err(), want, "{body}");
    }
    let not = [
        (
            "let t: gpr; mov(t, grid.thread);",
            "2:25: `grid.thread` is of the grid a kernel runs on when it is `#[cooperative]`, and `k` is not",
        ),
        (
            "grid.sync();",
            "2:6: `grid.sync()` waits for every thread of the grid, which runs all at once only if `k` is \
             `#[cooperative]`",
        ),
    ];
    for (body, want) in not {
        assert_eq!(compile("x: usize", body).unwrap_err(), want, "{body}");
    }
    // The place takes the sixth integer register; floats go in registers of their own.
    let five = "a: usize, b: usize, c: usize, d: usize, e: usize";
    assert_eq!(cooperative(&format!("{five}, x: f32"), "").unwrap(), "ret");
    assert_eq!(
        cooperative(&format!("{five}, f: usize"), "").unwrap_err(),
        "2:82: a cooperative kernel takes five integers at most: SysV passes six in registers, and the sixth is the \
         thread's place in the grid"
    );
    let src = "#[cooperative]\nunsafe extern \"sysv64\" fn k() -> f32 {}\n";
    assert_eq!(
        kgen::x86_str(src, "k").unwrap_err(),
        "2:34: a cooperative kernel's threads return nothing: they write what they compute to memory"
    );
}

/// Stack memory is the kernel's frame: the header says where each slot lies, and brackets reach
/// it from `rsp`, an index beside it.
#[test]
fn stack_memory_is_part_of_the_frame() {
    let src = r#"
        unsafe extern "sysv64" fn k(x: *const f32, i: usize) -> f32 {
            let n: local<u64>;
            let buf: local<[f32; 16], 32>;
            let v: ymm;
            vmovups(v, [x]);
            vmovaps([buf + 32], v);
            vmovaps([buf], v);
            mov([n], i);
            let p: gpr;
            lea(p, [buf + 64]);
            vmovss(v.xmm, [i + buf]);
            vaddss(v.xmm, v.xmm, [buf + i * 4 + 60]);
            add(i, [n]);
            ret(v.xmm);
        }
    "#;
    let want = "# k(gpr, gpr) -> vec
# needs: avx
# frame: 96 bytes aligned to 32, saves []
# n: 8 bytes at rsp
# buf: 64 bytes at rsp + 32
# x=rdi i=rsi v=ymm0 p=rax
    push rbp
    mov rbp, rsp
    sub rsp, 96
    and rsp, -32
    vmovups ymm0, ymmword ptr [rdi]
    vmovaps ymmword ptr [rsp + 64], ymm0
    vmovaps ymmword ptr [rsp + 32], ymm0
    mov qword ptr [rsp], rsi
    lea rax, [rsp + 96]
    vmovss xmm0, dword ptr [rsp + rsi*1 + 32]
    vaddss xmm0, xmm0, dword ptr [rsp + rsi*4 + 92]
    add rsi, qword ptr [rsp]
    mov rsp, rbp
    pop rbp
    vzeroupper
    ret
";
    assert_eq!(kgen::x86_str(src, "k").unwrap().text(), want);
}

/// The stack pointer moves no more than a page below memory the kernel has touched, as Rust's
/// functions do, so that a frame past the end of the stack faults at the guard page there
/// rather than step over it: a bigger frame is touched a page at a time, in a loop past three
/// pages. What the alignment may add counts, and the stack pointer is a multiple of 8 already.
#[test]
fn a_frame_of_more_than_a_page_is_touched_a_page_at_a_time() {
    let prologue = |ty: &str| {
        let code = compile("", &format!("let b: {ty}; let g: gpr; mov(g, [b]);")).unwrap();
        let end = code.find("\nmov rax").unwrap();
        code[..end].to_string()
    };
    let page = "sub rsp, 4096\nor qword ptr [rsp], 0";
    let cases = [
        ("local<u64>", "push rbp\nmov rbp, rsp\nsub rsp, 8".to_string()),
        ("local<[u8; 4096]>", "push rbp\nmov rbp, rsp\nsub rsp, 4096".into()),
        ("local<[u8; 4097]>", format!("push rbp\nmov rbp, rsp\n{page}\nsub rsp, 1")),
        ("local<[u8; 4064], 32>", "push rbp\nmov rbp, rsp\nsub rsp, 4064\nand rsp, -32".into()),
        ("local<[u8; 4096], 16>", format!("push rbp\nmov rbp, rsp\n{page}\nand rsp, -16")),
        ("local<[u8; 4096], 4096>", format!("push rbp\nmov rbp, rsp\n{page}\nand rsp, -4096")),
        ("local<[u8; 16384]>", format!("push rbp\nmov rbp, rsp\n{page}\n{page}\n{page}\nsub rsp, 4096")),
        (
            "local<[u8; 16385]>",
            format!("push rbp\nmov rbp, rsp\nlea r11, [rsp - 16384]\n2:\n{page}\ncmp rsp, r11\njne 2b\nsub rsp, 1"),
        ),
    ];
    for (ty, want) in cases {
        assert_eq!(prologue(ty), want, "{ty}");
    }
}

/// The prologue's loop takes the first label, and the body's come after it.
#[test]
fn the_prologues_loop_comes_before_the_bodys_labels() {
    let code =
        compile("n: usize", "let b: local<[u64; 4096]>;\nlabel(top);\nmov([b], n);\ndec(n);\njnz(top);").unwrap();
    let want = "push rbp
mov rbp, rsp
lea r11, [rsp - 28672]
2:
sub rsp, 4096
or qword ptr [rsp], 0
cmp rsp, r11
jne 2b
sub rsp, 4096
3:
mov qword ptr [rsp], rdi
dec rdi
jnz 3b
mov rsp, rbp
pop rbp
ret";
    assert_eq!(code, want);
}

/// A helper takes stack memory as a parameter of the type it was declared.
#[test]
fn a_helper_takes_stack_memory_of_its_type() {
    let src = |param: &str| {
        format!(
            "unsafe extern \"sysv64\" fn k() {{\nlet buf: local<[f32; 16], 32>;\nlet v: ymm;\nzero(v);\nstore(buf, \
             v);\n}}\nfn store(b: {param}, v: ymm) {{\nvmovaps([b + 32], v);\n}}\n"
        )
    };
    let code = kgen::x86_str(&src("local<[f32; 16], 32>"), "k").unwrap();
    assert!(code.code().any(|l| l == "vmovaps ymmword ptr [rsp + 32], ymm0"), "{}", code.text());
    let e = kgen::x86_str(&src("local<[f32; 16]>"), "k").unwrap_err();
    assert_eq!(e, "5:7: `b`: `buf` is memory, `local<[f32; 16], 32>`; it is no `local<[f32; 16]>`");
    let e = kgen::x86_str(&src("ymm"), "k").unwrap_err();
    assert_eq!(e, "5:7: `b`: `buf` is memory, `local<[f32; 16], 32>`; it is no `ymm`");
    let src = "unsafe extern \"sysv64\" fn k(x: *const f32) {\nload(x);\n}\nfn load(b: local<u64>) {}\n";
    assert_eq!(kgen::x86_str(src, "k").unwrap_err(), "2:6: `b`: a gpr is no memory, `local<u64>`");
}

/// What kgen can check of stack memory, it does: its type, and an address where the index is 0.
#[test]
fn stack_memory_is_checked_where_kgen_can() {
    let cases = [
        ("let b: local<[f32]>;", "2:8: `local<[f32]>` has no size: stack memory has one, as in `local<[f32; 64]>`"),
        ("let b: local<[f32; 0]>;", "2:8: `local<[f32; 0]>` holds nothing"),
        ("let b: local<[f32; 4], 2>;", "2:8: an array of `f32` is aligned to 4 at least, not 2"),
        ("let b: local<[f32; 4], 3>;", "2:8: what an address is a multiple of is a power of two, not 3"),
        ("let b: local<[f32; 4], 8192>;", "2:8: stack memory is aligned to a page at most, 4096; not 8192"),
        (
            "let b: local<[bool; 4]>;",
            "2:8: stack memory holds Rust's numbers, `u8` to `u64`, `i8` to `i64`, `usize`, `isize`, `f32` and `f64`; \
             not `bool`",
        ),
        (
            "let b: local<[f32; 4], 16, 2>;",
            "2:8: `local<..>` is an array of Rust's numbers and what its address is a multiple of, `local<[f32; 64], \
             32>`, or one, `local<u64>`; not `local<[f32; 4], 16, 2>`",
        ),
        (
            "let b: shared<[f32; 4]>;",
            "2:8: `shared<..>` is a GPU's memory; an x86 kernel has stack memory, `local<[f32; 4]>`",
        ),
        (
            "let b: local<[u64; 300000000]>;",
            "2:8: this makes the kernel's stack memory more than 2147483647 bytes, the most an address reaches past \
             the stack pointer",
        ),
        (
            "let b: local<[u8; 2147483647]>; let c: local<u8>;",
            "2:40: this makes the kernel's stack memory more than 2147483647 bytes, the most an address reaches past \
             the stack pointer",
        ),
        (
            "let b: local<[f32; 16]>; let v: ymm; vmovups(v, [b - 4]);",
            "2:49: `vmovups` reads at `[b - 4]`, before `b` starts",
        ),
        (
            "let b: local<[f32; 16]>; let v: ymm; vmovups(v, [b + 40]);",
            "2:49: `vmovups` reads 32 bytes at `[b + 40]`, past the end of `b`, 64 bytes",
        ),
        (
            "let b: local<[f32; 16]>; let v: ymm; zero(v); vmovups([b + 36], v);",
            "2:55: `vmovups` writes 32 bytes at `[b + 36]`, past the end of `b`, 64 bytes",
        ),
        (
            "let b: local<u8>; let g: gpr; movzx(g.r32, byte[b]); mov([b], g.r32);",
            "2:58: `mov` writes 4 bytes at `[b]`, past the end of `b`, 1 byte",
        ),
        (
            "let b: local<[f32; 16]>; let v: ymm; vmovups(v, [b + 5000000000]);",
            "2:49: `vmovups` reads 32 bytes at `[b + 5000000000]`, past the end of `b`, 64 bytes",
        ),
        (
            "let b: local<[f32; 16]>; let v: ymm; vmovaps(v, [b]);",
            "2:49: `vmovaps` reads 32 bytes at once, at an address that is a multiple of 32; `b` is at a multiple of \
             4: `local<[f32; 16], 32>`, or `vmovups` at any",
        ),
        (
            "let b: local<[f32; 16], 32>; let v: ymm; vmovaps(v, [b + 16]);",
            "2:53: `vmovaps` reads 32 bytes at once, at an address that is a multiple of 32; `[b + 16]` is not one",
        ),
        (
            "let b: local<[f32; 16], 32>; let v: ymm; vmovdqa(v.xmm, [b + 8]);",
            "2:57: `vmovdqa` reads 16 bytes at once, at an address that is a multiple of 16; `[b + 8]` is not one",
        ),
        (
            "let b: local<[f32; 16]>; let i: gpr; zero(i); let v: ymm; vmovss(v.xmm, [b + i * 4 + 64]);",
            "2:73: `vmovss` reads 4 bytes at `[b + i * 4 + 64]`, past the end of `b`, 64 bytes, when `i` is 0",
        ),
        (
            "let b: local<[f32; 16]>; let i: gpr; zero(i); let v: ymm; vmovss(v.xmm, [b + i * 4 - 4]);",
            "2:73: `vmovss` reads at `[b + i * 4 - 4]`, before `b` starts, when `i` is 0",
        ),
        (
            "let b: local<[f32; 16]>; let p: gpr; lea(p, [b + 68]);",
            "2:45: `lea` takes the address `[b + 68]`, more than one past the end of `b`, 64 bytes",
        ),
        (
            "let b: local<[f32; 16]>; let p: gpr; lea(p, [b - 4]);",
            "2:45: `lea` takes the address `[b - 4]`, before `b` starts",
        ),
        (
            "let b: local<[f32; 16]>; let v: ymm; vaddps(v, v, b);",
            "2:51: `vaddps` takes no memory as an operand: `lea(p, [b])` puts the address of `b` in `p`, and `[b]` is \
             what it holds",
        ),
        (
            "let b: local<[f32; 16]>; let c: local<u64>; let g: gpr; mov(g, [b + c]);",
            "2:64: an address names one memory at most; `b` and `c` are two",
        ),
        (
            "let b: local<[f32; 16]>; let g: gpr; mov(g, [x + b * 4]);",
            "2:45: an address adds memory as it is, as in `[b + i * 4]`; this multiplies `b` by 4",
        ),
        (
            "let b: local<[f32; 16]>; let g: gpr; zero(g); mov(g, [b + g.r32 * 4]);",
            "2:54: an address is made of gprs seen whole; this is a gpr's r32",
        ),
    ];
    for (body, want) in cases {
        assert_eq!(compile("x: *const f32", body).unwrap_err(), want, "{body}");
    }
}

/// Functions that call: `gcd` itself, and the rest `gcd`, each other and the functions they
/// take the address of.
const CALLERS: &str = r#"
    /// The greatest common divisor of `a` and `b`, by subtraction.
    unsafe extern "sysv64" fn gcd(a: u64, b: u64) -> u64 {
        let (t, g): (gpr, gpr);
        test(b, b);
        jnz(more);
        ret(a);
        label(more);
        cmp(a, b);
        jae(ordered);
        mov(t, a);
        mov(a, b);
        mov(b, t);
        label(ordered);
        sub(a, b);
        call(g, gcd, (a, b));
        ret(g);
    }

    /// `x` and the gcd of `x` and 6.
    unsafe extern "sysv64" fn plus(x: u64) -> u64 {
        let y: gpr;
        call(y, gcd, (x, 6));
        add(y, x);
        ret(y);
    }

    /// The eight floats at `p`, kept across a call.
    unsafe extern "sysv64" fn keep(p: *mut f32) {
        let v: ymm;
        vmovups(v, [p]);
        call(nothing);
        vmovups([p], v);
    }

    unsafe extern "sysv64" fn nothing() {}

    unsafe extern "sysv64" fn sum(a: u64, b: u64) -> u64 {
        add(a, b);
        ret(a);
    }

    /// The gcd of `a` and itself, called at the address in a gpr.
    unsafe extern "sysv64" fn through(a: u64) -> u64 {
        let (fp, r): (gpr, gpr);
        lea(fp, [gcd]);
        call(r, fp, (a, a), (gcd, sum));
        ret(r);
    }

    /// Stops the program at `ud2`, which raises SIGILL.
    unsafe extern "sysv64" fn fail() -> ! {
        ud2();
    }

    /// The `i`th of the `n` integers at `xs`; past them, a stop, at `fail`.
    unsafe extern "sysv64" fn at(xs: *const u64, n: usize, i: usize) -> u64 {
        let x: gpr;
        cmp(i, n);
        jae(past);
        mov(x, [xs + i * 8]);
        ret(x);
        label(past);
        call(fail);
    }
"#;

/// A call passes its arguments where SysV does, the integers in rdi, rsi, rdx, rcx, r8 and r9,
/// and its callee gives back in rax: here, where the arguments were, but for the swap. The
/// stack pointer is a multiple of 16 at the call, 8 below where it was at the kernel's.
#[test]
fn a_function_calls_itself_with_its_arguments_where_sysv_passes_them() {
    let want = "# gcd(gpr, gpr) -> gpr
# frame: 0 bytes aligned to 0, saves [], and 8 more so that it calls at a multiple of 16
# a=rdi,rax b=rsi t=rax g=rax
    sub rsp, 8
    test rsi, rsi
    jnz 2f
    mov rax, rdi
    add rsp, 8
    ret
2:
    cmp rdi, rsi
    jae 3f
    mov rax, rdi
    mov rdi, rsi
    mov rsi, rax
3:
    sub rdi, rsi
    call {f0}
    add rsp, 8
    ret
# f0: gcd
";
    assert_eq!(kgen::x86_str(CALLERS, "gcd").unwrap().text(), want);
}

/// A call changes the registers SysV lets it, all but rbx, rbp and r12 to r15: an integer
/// live across one is kept in rbx, which the kernel saves for its own caller; a vector, of
/// which the call may change any, is spilled to the frame.
#[test]
fn a_value_live_across_a_call_is_kept_where_the_call_leaves_it() {
    let want = "# plus(gpr) -> gpr
# frame: 0 bytes aligned to 0, saves [rbx]
# x=rbx,rdi y=rax b=rsi
    push rbx
    mov rbx, rdi
    mov rdi, rbx
    mov rsi, 6
    call {f0}
    add rax, rbx
    pop rbx
    ret
# f0: gcd
";
    assert_eq!(kgen::x86_str(CALLERS, "plus").unwrap().text(), want);
    let want = "# keep(gpr) -> ()
# needs: avx
# frame: 32 bytes aligned to 32, saves [rbx]
# v, spilled: 32 bytes at rsp
# p=rbx,rdi v=ymm0
    push rbx
    push rbp
    mov rbp, rsp
    sub rsp, 32
    and rsp, -32
    mov rbx, rdi
    vmovups ymm0, ymmword ptr [rbx]
    # v spilled
    vmovups ymmword ptr [rsp], ymm0
    call {f0}
    # v reloaded
    vmovups ymm0, ymmword ptr [rsp]
    vmovups ymmword ptr [rbx], ymm0
    mov rsp, rbp
    pop rbp
    pop rbx
    vzeroupper
    ret
# f0: nothing
";
    assert_eq!(kgen::x86_str(CALLERS, "keep").unwrap().text(), want);
}

/// `lea` takes the address of a function of the module, rip-relative, and a call through the
/// gpr that holds it calls one of the functions taken together last, which take what it
/// passes.
#[test]
fn a_call_through_a_gpr_calls_one_of_the_functions_taken_together_last() {
    let want = "# through(gpr) -> gpr
# frame: 0 bytes aligned to 0, saves [], and 8 more so that it calls at a multiple of 16
# a=rsi,rdi fp=rax r=rax
    sub rsp, 8
    mov rsi, rdi
    lea rax, [rip + {f0}]
    mov rdi, rsi
    call rax
    add rsp, 8
    ret
# f0: gcd
";
    assert_eq!(kgen::x86_str(CALLERS, "through").unwrap().text(), want);
}

/// A function that never returns, `-> !`, ends each way out in `ud2` or in a call of such a
/// function, which no epilogue follows.
#[test]
fn a_function_that_never_returns_ends_in_ud2_or_a_call_of_one() {
    let want = "# fail() -> !
# frame: 0 bytes aligned to 0, saves []
    ud2
";
    assert_eq!(kgen::x86_str(CALLERS, "fail").unwrap().text(), want);
    let want = "# at(gpr, gpr, gpr) -> gpr
# frame: 0 bytes aligned to 0, saves [], and 8 more so that it calls at a multiple of 16
# xs=rdi n=rsi i=rdx x=rax
    sub rsp, 8
    cmp rdx, rsi
    jae 2f
    mov rax, qword ptr [rdi + rdx*8]
    add rsp, 8
    ret
2:
    call {f0}
# f0: fail
";
    assert_eq!(kgen::x86_str(CALLERS, "at").unwrap().text(), want);
}

/// Functions of the module for `k` to call, and some it may not.
const CALLEES: &str = r#"
unsafe extern "sysv64" fn gcd(a: u64, b: u64) -> u64 { ret(a); }
unsafe extern "sysv64" fn half(x: f32) -> f32 { ret(x); }
unsafe extern "sysv64" fn scale(x: f64, n: usize) -> f64 { ret(x); }
unsafe extern "sysv64" fn nothing() {}
unsafe extern "sysv64" fn stop() -> ! { ud2(); }
#[instances(1, 2)]
unsafe extern "sysv64" fn tile<const N: usize>(p: *mut f32) {}
#[cooperative]
unsafe extern "sysv64" fn together(x: *mut u64) {}
extern "ptx-func" fn on_gpu(a: u32) -> u32 { ret(a); }
unsafe extern "sysv64" fn narrow(a: u32) {}
"#;

/// The kernel `k` around `body`, returning `ret`, in a module of [`CALLEES`], compiled, or
/// the error.
fn calling(params: &str, ret: &str, body: &str) -> Result<String, String> {
    let src = format!("unsafe extern \"sysv64\" fn k({params}){ret} {{\n{body}\n}}\n{CALLEES}");
    kgen::x86_str(&src, "k").map(|l| l.code().collect::<Vec<_>>().join("\n"))
}

/// The stack pointer is a multiple of 16 plus 8 on entry, and each push and the frame move
/// it: a kernel that calls moves it 8 more where that leaves it short of a multiple of 16.
#[test]
fn the_stack_pointer_is_a_multiple_of_16_at_a_call() {
    let cases = [
        ("x: u64", "call(nothing);", "sub rsp, 8\ncall {f0}\nadd rsp, 8\nret"),
        ("x: u64", "call(nothing); add(x, x);", "push rbx\nmov rbx, rdi\ncall {f0}\nadd rbx, rbx\npop rbx\nret"),
        (
            "x: u64, y: u64",
            "call(nothing); add(x, y);",
            "push rbx\npush r12\nsub rsp, 8\nmov rbx, rsi\nmov r12, rdi\ncall {f0}\nadd r12, rbx\nadd rsp, 8\npop \
             r12\npop rbx\nret",
        ),
        (
            "x: u64",
            "let n: local<u64>; mov([n], x); call(nothing); mov(x, [n]);",
            "push rbp\nmov rbp, rsp\nsub rsp, 16\nmov qword ptr [rsp], rdi\ncall {f0}\nmov rdi, qword ptr [rsp]\nmov \
             rsp, rbp\npop rbp\nret",
        ),
        (
            "x: u64",
            "let n: local<[u64; 2], 16>; mov([n], x); call(nothing); mov(x, [n]);",
            "push rbp\nmov rbp, rsp\nsub rsp, 16\nand rsp, -16\nmov qword ptr [rsp], rdi\ncall {f0}\nmov rdi, qword \
             ptr [rsp]\nmov rsp, rbp\npop rbp\nret",
        ),
    ];
    for (params, body, want) in cases {
        assert_eq!(calling(params, "", body).unwrap(), want, "{body}");
    }
}

/// A gpr's r32 passes its 32 bits, the rest clear, and takes the result's; a float known now
/// is read from memory, as x86 takes floats.
#[test]
fn a_call_passes_what_its_callee_takes() {
    let code = calling("x: u64", " -> u64", "let r: gpr; call(r.r32, gcd, (x.r32, x)); ret(r);").unwrap();
    assert_eq!(code, "sub rsp, 8\nmov rsi, rdi\nmov edi, esi\ncall {f0}\nmov eax, eax\nadd rsp, 8\nret");
    let code = calling("x: *mut f32", "", "let v: ymm; call(v.xmm, half, 2.5); vmovss([x], v.xmm);").unwrap();
    let want = "push rbx\nmov rbx, rdi\nvmovss xmm0, dword ptr [rip + {c0}]\ncall {f0}\nvmovss dword ptr [rbx], \
                xmm0\npop rbx\nvzeroupper\nret";
    assert_eq!(code, want);
    let code = calling("x: *mut f64", "", "let v: ymm; call(v.xmm, scale, (2.5f64, x)); vmovq([x], v.xmm);").unwrap();
    let want = "push rbx\nmov rbx, rdi\nvmovq xmm0, qword ptr [rip + {c0}]\nmov rdi, rbx\ncall {f0}\nvmovq qword ptr \
                [rbx], xmm0\npop rbx\nvzeroupper\nret";
    assert_eq!(code, want);
}

/// A call takes the module's functions SysV calls, which pass what it passes and give back
/// what it takes, and a gpr's address of one.
#[test]
fn a_call_is_checked_against_its_callee() {
    let cases = [
        ("let r: gpr; call(r, gcd, (x));", "2:26: `gcd` takes 2 arguments, not 1"),
        (
            "let (r, v): (gpr, ymm); zero(v); call(r, gcd, (x, v));",
            "2:47: `gcd` takes an integer `b` as argument 2; this is a ymm",
        ),
        (
            "call(gcd, (x, x));",
            "2:1: `gcd` returns an integer: `call(r, gcd, ..)` writes it to `r`, or `call(_, gcd, ..)` to nothing",
        ),
        ("let r: gpr; call(r, nothing);", "2:18: `nothing` returns nothing: `call(nothing, ..)` calls it"),
        ("let r: gpr; call(r, stop);", "2:18: `stop` never returns: `call(stop, ..)` calls it"),
        ("let v: ymm; call(v, gcd, (x, x));", "2:18: `gcd` returns an integer, to a gpr or `_`; this is a ymm"),
        ("let v: ymm; call(v, half, 1.5f64);", "2:27: `half` takes an `f32` `x`; this is the f64 1.5"),
        (
            "let v: ymm; call(v, scale, (2.5, x));",
            "2:28: `scale` takes an `f64` `x` as argument 1; this is the f32 2.5: `2.5f64` is an `f64`",
        ),
        ("let v: ymm; call(v, half, 2);", "2:27: `half` takes an `f32` `x`; this is the integer 2: `2.0` is an `f32`"),
        (
            "let v: ymm; call(v, half, gcd);",
            "2:27: `half` takes an `f32` `x`; this is the function `gcd`: `lea(p, [gcd])` puts its address in `p`",
        ),
        (
            "call(together, x);",
            "2:6: `together` is `#[cooperative]`: its threads run on a grid, which its launcher starts, so no kernel \
             calls it",
        ),
        (
            "call(on_gpu, x);",
            "2:6: `on_gpu` is no x86 function: an x86 kernel calls those of its module `extern \"sysv64\"`",
        ),
        ("call(tile, x);", "2:6: `tile` is compiled for each of its instances: an instruction names one, `tile::<..>`"),
        (
            "call(tile::<3>, x);",
            "2:6: `tile::<3>` is not compiled: `tile`'s `#[instances(..)]` lists `tile::<1>`, `tile::<2>`",
        ),
        (
            "let v: ymm; vmovups(v, [gcd]);",
            "2:24: `vmovups` takes no function's code: `lea(p, [gcd])` puts the address of `gcd` in `p`",
        ),
        (
            "let v: ymm; vmovups(v, ymmword[gcd]);",
            "2:31: `vmovups` takes no function's code: `lea(p, [gcd])` puts the address of `gcd` in `p`",
        ),
        (
            "let p: gpr; lea(p, qword[gcd]);",
            "2:25: `lea` takes an address and reaches no memory: no size before the brackets",
        ),
        (
            "let p: gpr; lea(p, [together]);",
            "2:20: `together` is `#[cooperative]`: its threads run on a grid, which its launcher starts, so no \
             kernel calls it",
        ),
        (
            "let p: gpr; mov(p, gcd);",
            "2:20: `mov` takes no function; `call(r, gcd, (a, b))` calls `gcd`, and `lea(p, [gcd])` puts its \
             address in `p`",
        ),
        (
            "let p: gpr; lea(p, [gcd]); call(p, (x, 1));",
            "2:36: a call through a gpr passes variables, each in the file it is of: this is the integer 1; taking \
             the functions it may call last, `call(r, fp, (a, b), (f, g))`, lets it be more",
        ),
        (
            "let (p, r): (gpr, gpr); lea(p, [gcd]); call(r, p, (x, x), (gcd, half));",
            "2:45: `half` returns an `f32`, to a vector or `_`; this is a gpr",
        ),
        (
            "let r: gpr; call(r, gcd, (x, x), (gcd, half));",
            "2:21: `call` names the function it calls, `gcd`, or takes its address from a gpr, of one of the \
             functions taken together last; not both",
        ),
        (
            "let v: ymm; zero(v); call(v, v, (x));",
            "2:30: `call` calls a function by its name, or at the address in a gpr; this is a ymm",
        ),
        (
            "call();",
            "2:1: `call` is `call(f, (a, b))`, or `call(r, f, (a, b))` for a function that returns a value, `f` a \
             function of the module or a gpr that holds the address of one",
        ),
        (
            "let p: gpr; lea(p, [gcd]); call(p, (x, x, x, x, x, x, x));",
            "2:36: this passes too many: SysV passes a seventh integer on the stack, which kgen does not read",
        ),
        (
            "let v: ymm; zero(v); let p: gpr; lea(p, [gcd]); call(p, (v, v, v, v, v, v, v, v, v));",
            "2:57: this passes too many: SysV passes a ninth float on the stack, which kgen does not read",
        ),
        ("call(narrow, x);", "2:6: `narrow` does not compile, so no kernel calls it"),
        ("ud2(x);", "2:1: `ud2` takes no operands, not 1"),
    ];
    for (body, want) in cases {
        assert_eq!(calling("x: u64", "", body).unwrap_err(), want, "{body}");
    }
}

/// A function that never returns, `-> !`, may not, and a cooperative kernel returns.
#[test]
fn a_function_that_never_returns_is_checked() {
    let cases = [
        (
            "unsafe extern \"sysv64\" fn k() -> ! {\nret();\n}",
            "2:1: `k` never returns, `-> !`: it ends each way out in `ud2()` or in a call of a function that never \
             returns",
        ),
        (
            "unsafe extern \"sysv64\" fn k(x: u64) -> ! {\ntest(x, x);\njz(out);\nud2();\nlabel(out);\n}",
            "1:27: control reaches the end of the body, and `k` never returns, `-> !`: it ends each way out in \
             `ud2()` or in a call of a function that never returns",
        ),
        (
            "unsafe extern \"sysv64\" fn k() -> ! {\ncall(nothing);\n}\nunsafe extern \"sysv64\" fn nothing() {}",
            "1:27: control reaches the end of the body, and `k` never returns, `-> !`: it ends each way out in \
             `ud2()` or in a call of a function that never returns",
        ),
        (
            "#[cooperative]\nunsafe extern \"sysv64\" fn k() -> ! {\nud2();\n}",
            "2:34: a cooperative kernel's threads return nothing: they write what they compute to memory",
        ),
    ];
    for (src, want) in cases {
        assert_eq!(kgen::x86_str(src, "k").unwrap_err(), want, "{src}");
    }
}

/// A program of the toolchain: `$var`, or `name` on the path, if it runs.
fn tool(var: &str, name: &str) -> Option<PathBuf> {
    if let Some(p) = std::env::var_os(var) {
        return Some(p.into());
    }
    let here = Command::new(name).arg("--version").output().is_ok_and(|o| o.status.success());
    here.then(|| name.into())
}

/// The kernel `k` around a call of `name` in `form`: each operand a variable of its own, written
/// before and read after by an instruction of another name, so that no two share a register and
/// kgen leaves out no move of one to another, as it leaves out a move of a register to itself;
/// memory at `x` of the size the form reaches, a vector of indices beside it; and each integer
/// the least the form takes there, or the most.
fn call(name: &str, form: &Form, most: bool) -> String {
    let mut body = String::new();
    let mut after = String::new();
    let mut args = vec![];
    let store = if name == "vmovups" { "vmovdqu" } else { "vmovups" };
    let test = if name == "test" { "cmp" } else { "test" };
    for (i, spec) in form.operands.iter().enumerate() {
        let mut vector = |bytes: u32| {
            body += &format!("let v{i}: ymm; zero(v{i});\n");
            after += &format!("\n{store}([x], v{i});");
            if bytes == 32 { format!("v{i}") } else { format!("v{i}.xmm") }
        };
        args.push(match spec.kind {
            Kind::Gpr(w) => {
                body += &format!("let g{i}: gpr; zero(g{i});\n");
                after += &format!("\n{test}(g{i}, g{i});");
                if w.bytes() == 8 { format!("g{i}") } else { format!("g{i}.r{}", w.bytes() * 8) }
            }
            Kind::Vec(w) => vector(w.bytes()),
            Kind::Mem(w) | Kind::Aligned(w) => format!("{}[x]", forms::word(w)),
            Kind::Addr => "[x]".into(),
            Kind::Vsib(elem, index) => format!("{}[x + {} * 4]", forms::word(elem), vector(index.bytes())),
            Kind::Label => "end".into(),
            kind => {
                let (least, most_) = kind.ints().expect("an integer");
                (if most { most_ } else { least }).to_string()
            }
        });
    }
    let end = if form.operands.iter().any(|s| s.kind == Kind::Label) { "\nlabel(end);" } else { "" };
    format!("{body}{name}({});{after}{end}", args.join(", "))
}

/// `lines` as a file the assemblers read, Intel's syntax, and a jump's label after it: the file,
/// and the line of it each of `lines` is, counted from 1.
fn source(lines: &[&str]) -> (String, Vec<usize>) {
    let mut text = String::from(".intel_syntax noprefix\n");
    let (mut at, mut n) = (vec![], 1);
    for line in lines {
        text += &format!("{line}\n");
        n += 1;
        at.push(n);
        // A jump names its label ahead, `jz 2f`.
        let label = line.rsplit(' ').next().and_then(|t| t.strip_suffix('f')).filter(|t| t.parse::<u32>().is_ok());
        if let Some(label) = label {
            text += &format!("{label}:\n");
            n += 1;
        }
    }
    (text, at)
}

/// `program` run with `args` on `text`, written to the test's directory as `name`: whether it
/// succeeded, what it wrote, and the lines of `text` it says something of, by number, with what.
fn assemble(program: &PathBuf, args: &[&str], name: &str, text: &str) -> (bool, String, Vec<(usize, String)>) {
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    std::fs::write(&path, text).unwrap();
    let run = Command::new(program).args(args).arg(&path).output().unwrap();
    let file = path.to_string_lossy().into_owned();
    // `path:line: Error: why` from GNU as, `path:line:column: error: why` from llvm-mc.
    let said = String::from_utf8_lossy(&run.stderr)
        .lines()
        .filter_map(|e| {
            let rest = e.strip_prefix(&file)?.strip_prefix(':')?;
            let (line, why) = rest.split_once(':')?;
            Some((line.parse().ok()?, why.trim().to_string()))
        })
        .collect();
    (run.status.success(), String::from_utf8_lossy(&run.stdout).into_owned(), said)
}

/// GNU as's name for `f`, in `-march`.
fn gas(f: Feature) -> &'static str {
    match f {
        Feature::Bmi1 => "bmi",
        f => f.name(),
    }
}

/// Every form of every instruction kgen knows comes out of the language as a line `llvm-mc`
/// assembles, which assembles a kernel's `naked_asm!` as rustc does, with each integer the
/// least the form takes and the most; and encodes none with EVEX, which only AVX-512 reads,
/// and which `llvm-mc` assembles for any machine. GNU as, which knows what each feature adds,
/// takes each line on a machine of the CPU feature its form names, and refuses it on a machine
/// of every feature before that one; and the listing names the feature.
#[test]
fn every_form_kgen_knows_assembles_on_a_machine_of_its_feature() {
    let (Some(mc), Some(gnu)) = (tool("LLVM_MC", "llvm-mc"), tool("GNU_AS", "as")) else {
        eprintln!("no llvm-mc or GNU as here: skipped; set LLVM_MC and GNU_AS to run it");
        return;
    };
    let all: Vec<&Form> = forms::names().flat_map(|n| forms::of(n).unwrap()).collect();
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let written = Mutex::new(vec![]);
    std::thread::scope(|s| {
        for chunk in all.chunks(all.len().div_ceil(threads)) {
            let written = &written;
            s.spawn(move || {
                for form in chunk {
                    let ints = form.operands.iter().any(|s| s.kind.ints().is_some());
                    for most in [false, true].into_iter().take(if ints { 2 } else { 1 }) {
                        let call = call(form.op.name(), form, most);
                        let src = format!("unsafe extern \"sysv64\" fn k(x: *mut f32) {{\n{call}\n}}\n");
                        let listing = kgen::x86_str(&src, "k").unwrap_or_else(|e| panic!("{e}, of\n{call}"));
                        if let Some(f) = form.feature {
                            assert!(listing.features.contains(&f), "no {f:?} in {:?}, of\n{call}", listing.features);
                        }
                        let mnemonic = form.op.mnemonic();
                        let line =
                            listing.code().filter(|l| *l == mnemonic || l.starts_with(&format!("{mnemonic} "))).last();
                        let line = line.unwrap_or_else(|| panic!("no `{mnemonic}` in\n{}", listing.text()));
                        written.lock().unwrap().push((form.feature, line.to_string(), call));
                    }
                }
            });
        }
    });
    let mut written = written.into_inner().unwrap();
    written.sort();

    let lines: Vec<&str> = written.iter().map(|(_, line, _)| line.as_str()).collect();
    let (text, at) = source(&lines);
    let of = |n: usize| at.iter().position(|&a| a == n).map(|i| &written[i]);
    let (ok, out, said) = assemble(&mc, &["-triple=x86_64", "-x86-asm-syntax=intel", "-show-encoding"], "x86.s", &text);
    let refused: Vec<String> = said
        .iter()
        .map(|(n, why)| match of(*n) {
            Some((_, line, call)) => format!("{line}: {why}, of\n{call}"),
            None => format!("line {n}: {why}"),
        })
        .collect();
    assert!(refused.is_empty() && ok, "llvm-mc refuses what kgen writes:\n\n{}", refused.join("\n\n"));
    let encodings: Vec<&str> = out.lines().filter_map(|l| l.split_once("encoding: [").map(|(_, e)| e)).collect();
    assert_eq!(encodings.len(), lines.len());
    let evex: Vec<_> = lines.iter().zip(&encodings).filter(|(_, e)| e.starts_with("0x62")).map(|(l, _)| l).collect();
    assert!(evex.is_empty(), "kgen writes AVX-512's EVEX: {evex:#?}");

    let features = [None].into_iter().chain(Feature::ALL.map(Some));
    for (k, feature) in features.enumerate() {
        let group: Vec<_> = written.iter().filter(|(f, ..)| *f == feature).collect();
        let lines: Vec<&str> = group.iter().map(|(_, line, _)| line.as_str()).collect();
        let (text, at) = source(&lines);
        let name = format!("x86-{}.s", feature.map_or("x86-64", Feature::name));
        let of = |n: usize| at.iter().position(|&a| a == n).map(|i| group[i]);
        let march = |fs: &[Feature]| {
            format!("-march=generic64{}", fs.iter().map(|&f| format!("+{}", gas(f))).collect::<String>())
        };
        let on = march(feature.as_slice());
        let (ok, _, said) = assemble(&gnu, &["--64", &on, "-o", "/dev/null"], &name, &text);
        let refused: Vec<String> = said
            .iter()
            .map(|(n, why)| match of(*n) {
                Some((_, line, call)) => format!("{line}: {why}, of\n{call}"),
                None => format!("line {n}: {why}"),
            })
            .collect();
        assert!(refused.is_empty() && ok, "GNU as {on} says, of what kgen writes:\n\n{}", refused.join("\n\n"));
        if feature.is_none() {
            continue;
        }
        // Every feature before this one, which builds on none after it.
        let before = march(&Feature::ALL[..k - 1]);
        let (_, _, said) = assemble(&gnu, &["--64", &before, "-o", "/dev/null"], &name, &text);
        let taken: Vec<&str> = at
            .iter()
            .zip(&lines)
            .filter(|(n, _)| !said.iter().any(|(m, why)| m == *n && why.starts_with("Error")))
            .map(|(_, l)| *l)
            .collect();
        assert!(taken.is_empty(), "GNU as {before} takes what kgen says needs {feature:?}: {taken:#?}");
    }
}

/// Every instruction of x86-64-v3 that `llvm-mc` takes without AVX-512, kgen knows, or names as
/// one it does not yet write, and why; and kgen knows no other. They are x86-64's own on
/// integers, SSE's to SSE4.2's as AVX writes them, and those of AVX, AVX2, FMA, F16C, BMI1,
/// BMI2, LZCNT, POPCNT and MOVBE; not the stack's, the strings', `loop`, x87's, MMX's or the
/// system's.
#[test]
fn kgen_knows_every_instruction_of_x86_64_v3() {
    const V3: &str = "
        adc add and andn bextr blsi blsmsk blsr bsf bsr bswap bt btc btr bts bzhi cbw cdq cdqe clc clflush cmc cmp
        cmpxchg cmpxchg16b cmpxchg8b cpuid cqo crc32 cwd cwde dec div idiv imul inc int3 jmp lahf lea lfence lock.adc
        lock.add lock.and lock.btc lock.btr lock.bts lock.cmpxchg lock.dec lock.inc lock.neg lock.not lock.or lock.sbb
        lock.sub lock.xadd lock.xchg lock.xor lzcnt mfence mov movbe movnti movsx movsxd movzx mul mulx neg nop not or
        pause pdep pext popcnt prefetchnta prefetcht0 prefetcht1 prefetcht2 rcl rcr rdtsc rol ror rorx sahf sal sar sarx
        sbb sfence shl shld shlx shr shrd shrx stc sub test tzcnt ud2 vaddpd vaddps vaddsd vaddss vaddsubpd vaddsubps
        vandnpd vandnps vandpd vandps vblendpd vblendps vblendvpd vblendvps vbroadcastf128 vbroadcasti128 vbroadcastsd
        vbroadcastss vcmppd vcmpps vcmpsd vcmpss vcomisd vcomiss vcvtdq2pd vcvtdq2ps vcvtpd2dq vcvtpd2ps vcvtph2ps
        vcvtps2dq vcvtps2pd vcvtps2ph vcvtsd2si vcvtsd2ss vcvtsi2sd vcvtsi2ss vcvtss2sd vcvtss2si vcvttpd2dq vcvttps2dq
        vcvttsd2si vcvttss2si vdivpd vdivps vdivsd vdivss vdppd vdpps vextractf128 vextracti128 vextractps vfmadd132pd
        vfmadd132ps vfmadd132sd vfmadd132ss vfmadd213pd vfmadd213ps vfmadd213sd vfmadd213ss vfmadd231pd vfmadd231ps
        vfmadd231sd vfmadd231ss vfmaddsub132pd vfmaddsub132ps vfmaddsub213pd vfmaddsub213ps vfmaddsub231pd
        vfmaddsub231ps vfmsub132pd vfmsub132ps vfmsub132sd vfmsub132ss vfmsub213pd vfmsub213ps vfmsub213sd vfmsub213ss
        vfmsub231pd vfmsub231ps vfmsub231sd vfmsub231ss vfmsubadd132pd vfmsubadd132ps vfmsubadd213pd vfmsubadd213ps
        vfmsubadd231pd vfmsubadd231ps vfnmadd132pd vfnmadd132ps vfnmadd132sd vfnmadd132ss vfnmadd213pd vfnmadd213ps
        vfnmadd213sd vfnmadd213ss vfnmadd231pd vfnmadd231ps vfnmadd231sd vfnmadd231ss vfnmsub132pd vfnmsub132ps
        vfnmsub132sd vfnmsub132ss vfnmsub213pd vfnmsub213ps vfnmsub213sd vfnmsub213ss vfnmsub231pd vfnmsub231ps
        vfnmsub231sd vfnmsub231ss vgatherdpd vgatherdps vgatherqpd vgatherqps vhaddpd vhaddps vhsubpd vhsubps
        vinsertf128 vinserti128 vinsertps vlddqu vldmxcsr vmaskmovdqu vmaskmovpd vmaskmovps vmaxpd vmaxps vmaxsd vmaxss
        vminpd vminps vminsd vminss vmovapd vmovaps vmovd vmovddup vmovdqa vmovdqu vmovhlps vmovhpd vmovhps vmovlhps
        vmovlpd vmovlps vmovmskpd vmovmskps vmovntdq vmovntdqa vmovntpd vmovntps vmovq vmovsd vmovshdup vmovsldup vmovss
        vmovupd vmovups vmpsadbw vmulpd vmulps vmulsd vmulss vorpd vorps vpabsb vpabsd vpabsw vpackssdw vpacksswb
        vpackusdw vpackuswb vpaddb vpaddd vpaddq vpaddsb vpaddsw vpaddusb vpaddusw vpaddw vpalignr vpand vpandn vpavgb
        vpavgw vpblendd vpblendvb vpblendw vpbroadcastb vpbroadcastd vpbroadcastq vpbroadcastw vpcmpeqb vpcmpeqd
        vpcmpeqq vpcmpeqw vpcmpestri vpcmpestrm vpcmpgtb vpcmpgtd vpcmpgtq vpcmpgtw vpcmpistri vpcmpistrm vperm2f128
        vperm2i128 vpermd vpermilpd vpermilps vpermpd vpermps vpermq vpextrb vpextrd vpextrq vpextrw vpgatherdd
        vpgatherdq vpgatherqd vpgatherqq vphaddd vphaddsw vphaddw vphminposuw vphsubd vphsubsw vphsubw vpinsrb vpinsrd
        vpinsrq vpinsrw vpmaddubsw vpmaddwd vpmaskmovd vpmaskmovq vpmaxsb vpmaxsd vpmaxsw vpmaxub vpmaxud vpmaxuw
        vpminsb vpminsd vpminsw vpminub vpminud vpminuw vpmovmskb vpmovsxbd vpmovsxbq vpmovsxbw vpmovsxdq vpmovsxwd
        vpmovsxwq vpmovzxbd vpmovzxbq vpmovzxbw vpmovzxdq vpmovzxwd vpmovzxwq vpmuldq vpmulhrsw vpmulhuw vpmulhw vpmulld
        vpmullw vpmuludq vpor vpsadbw vpshufb vpshufd vpshufhw vpshuflw vpsignb vpsignd vpsignw vpslld vpslldq vpsllq
        vpsllvd vpsllvq vpsllw vpsrad vpsravd vpsraw vpsrld vpsrldq vpsrlq vpsrlvd vpsrlvq vpsrlw vpsubb vpsubd vpsubq
        vpsubsb vpsubsw vpsubusb vpsubusw vpsubw vptest vpunpckhbw vpunpckhdq vpunpckhqdq vpunpckhwd vpunpcklbw
        vpunpckldq vpunpcklqdq vpunpcklwd vpxor vrcpps vrcpss vroundpd vroundps vroundsd vroundss vrsqrtps vrsqrtss
        vshufpd vshufps vsqrtpd vsqrtps vsqrtsd vsqrtss vstmxcsr vsubpd vsubps vsubsd vsubss vtestpd vtestps vucomisd
        vucomiss vunpckhpd vunpckhps vunpcklpd vunpcklps vxorpd vxorps vzeroall vzeroupper xadd xchg xgetbv xor
    ";
    let conditions = "a ae b be c e g ge l le na nae nb nbe nc ne ng nge nl nle no np ns nz o p pe po s z";
    let predicates = "eq lt le unord neq nlt nle ord eq_uq nge ngt false neq_oq ge gt true eq_os lt_oq le_oq unord_s \
                      neq_us nlt_uq nle_uq ord_s eq_us nge_uq ngt_uq false_os neq_os ge_oq gt_oq true_us";
    let mut v3: Vec<String> = V3.split_whitespace().map(String::from).collect();
    for c in conditions.split(' ') {
        v3.extend([format!("j{c}"), format!("cmov{c}"), format!("set{c}")]);
    }
    for p in predicates.split(' ') {
        v3.extend(["ps", "pd", "ss", "sd"].map(|t| format!("vcmp{p}{t}")));
    }
    for name in &v3 {
        let later = forms::LATER.iter().any(|(n, _)| n == name);
        assert!(forms::of(name).is_some() != later, "`{name}` is known or later, and not both");
    }
    for name in forms::names() {
        assert!(v3.iter().any(|n| n == name), "`{name}` is not of x86-64-v3");
    }
}
