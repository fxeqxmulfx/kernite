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

/// Sixteen accumulators and the load beside them take 17 of the 16 vector registers: the two
/// that cost least to keep on the stack, of those that hold one through the loop, spill.
#[test]
fn sixteen_accumulators_and_a_load_spill_two_and_the_listing_says_where() {
    let text = kgen::x86_str(KERNELS, "dot_unroll16").unwrap().text();
    let header = "# dot_unroll16(gpr, gpr, gpr) -> vec
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

/// A static is memory of a GPU's module, which an x86 kernel has none of: it takes memory by
/// pointer.
#[test]
fn a_static_is_no_memory_of_an_x86_kernel() {
    let src =
        "static COUNTS: global<[u32; 8]>;\nunsafe extern \"sysv64\" fn k() {\nlet v: ymm;\nvmovups(v, [COUNTS]);\n}\n";
    assert_eq!(
        kgen::x86_str(src, "k").unwrap_err(),
        "4:13: `COUNTS` is memory of the module's, and this machine has none: its kernels take memory by pointer"
    );
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
        ("inc(grid.threads);", "3:10: `inc` wants a gpr; this is memory"),
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
            "let b: local<u8>; let g: gpr; movzx(g.r32, [b]); mov([b], g.r32);",
            "2:54: `mov` writes 4 bytes at `[b]`, past the end of `b`, 1 byte",
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
