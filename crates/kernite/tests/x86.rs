//! Kernels in the language, compiled for x86: what they come to, and what the errors say; every
//! form of every instruction kernite knows, which `llvm-mc` assembles and GNU as takes on a machine
//! of its feature; and what the builder refuses.

use std::path::PathBuf;
use std::process::Command;
use std::sync::{Mutex, PoisonError};

use kernite_core::Access;
use kernite_x86::forms::{self, Form, Kind};
use kernite_x86::{Asm, Feature, SysV, intel};

#[path = "common/quotes.rs"]
mod quotes;

type Result<T = (), E = Box<dyn std::error::Error + Send + Sync>> = std::result::Result<T, E>;

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

/// `tests/builder` holds the listings `kernite-x86`'s builder made of these kernels, before the
/// language: what the language writes must compile to the same instructions in the same registers.
#[test]
fn the_examples_kernels_come_out_as_the_builder_wrote_them() -> Result {
    let snapshots = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/builder");
    for name in ["sum8", "dot_unroll1", "dot_unroll4"] {
        let listing = kernite::x86_str(KERNELS, name)?;
        let want = std::fs::read_to_string(snapshots.join(format!("{name}.s")))?;
        assert_eq!(listing.to_string(), want, "{name}");
    }
    Ok(())
}

/// A kernel's assembly changes only when the kernel does: it is the same each time it compiles,
/// and the same whatever else its module holds.
#[test]
fn a_kernels_assembly_changes_only_when_it_does() -> Result {
    let k = "unsafe extern \"sysv64\" fn k(x: *mut u64) {\nlet g: gpr;\nmov(g, [x]);\nadd(g, STEP);\nmov([x], g);\n}\n";
    let alone = format!("const STEP: u64 = 3;\n{k}");
    let among = format!(
        "static S: global<u64>;\nconst STEP: u64 = 3;\nconst FAR: u64 = 9;\nfn far(g: gpr) {{\nadd(g, FAR);\n}}\n\
         unsafe extern \"sysv64\" fn j(x: *mut u64) {{\nlet g: gpr;\nmov(g, [x]);\nfar(g);\nmov([x], g);\n}}\n{k}"
    );
    let listing = |src: &str| -> Result<_> { Ok(kernite::x86_str(src, "k")?.to_string()) };
    assert_eq!(listing(&alone)?, listing(&alone)?);
    assert_eq!(listing(&among)?, listing(&alone)?);
    Ok(())
}

/// Sixteen accumulators and the load beside them take 17 of the 16 vector registers: the two
/// that cost least to keep on the stack, of those that hold one through the loop, spill.
#[test]
fn sixteen_accumulators_and_a_load_spill_two_and_the_listing_says_where() -> Result {
    let text = kernite::x86_str(KERNELS, "dot_unroll16")?.to_string();
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
    Ok(())
}

#[test]
fn up_to_fifteen_accumulators_fit_without_spilling() -> Result {
    let mut src = KERNELS.to_string();
    for k in 1..=15 {
        src += &format!(
            "unsafe extern \"sysv64\" fn dot{k}(x: *const f32, y: *const f32, n: usize) -> f32 {{ dot::<{k}>(x, y, n); }}\n"
        );
    }
    for k in 1..=15 {
        let listing = kernite::x86_str(&src, &format!("dot{k}")).map_err(|e| format!("{k} accumulators: {e}"))?;
        assert!(listing.frame.slots.is_empty(), "{k} accumulators:\n{listing}");
    }
    Ok(())
}

/// Thirteen sums, the pointer and the count take 15 of the 14 general registers the allocator
/// has. The count spills, a parameter stored from where it arrives; then a sum, for the count to
/// be loaded beside the others. Its store goes between the `sub` and the `jnz` that reads its
/// flags, which a `mov` leaves as they were. The count is in rsi at the top of the loop on the
/// way in, and in r15 on the jump back, so no register holds it there, and it is loaded.
#[test]
fn a_general_register_spills_with_mov_a_parameter_from_where_it_arrives() -> Result {
    let src = "unsafe extern \"sysv64\" fn k(p: *mut u64, n: usize) {
        let g: [gpr; 13];
        for k in 0..13 { mov(g[k], [p + 8 * k]); }
        label(top);
        for k in 0..13 { add(g[k], [p + 8 * k]); }
        sub(n, 1);
        jnz(top);
        for k in 0..13 { mov([p + 8 * k], g[k]); }
    }";
    let text = kernite::x86_str(src, "k")?.to_string();
    let parts = [
        "# frame: 16 bytes aligned to 8, saves [rbx, r12, r13, r14, r15]
# n, spilled: 8 bytes at rsp - 16
# g12, spilled: 8 bytes at rsp - 8
# p=rdi n=rsi,r15 g0=rax",
        "    # n spilled
    mov qword ptr [rsp - 16], rsi
    mov rax, qword ptr [rdi]",
        "    # n reloaded
    mov r15, qword ptr [rsp - 16]
    sub r15, 1
    # n spilled
    mov qword ptr [rsp - 16], r15
    jnz 2b",
    ];
    for part in parts {
        assert!(text.contains(part), "{part}\nin\n{text}");
    }
    Ok(())
}

/// A spilled variable is read in the register that holds it, with no load, where one does. A store leaves what it
/// stores in the register it stores from, and a load in the register it loads to, until an instruction writes the
/// register. Thirteen sums, the pointer and the count are 15 values for the 14 registers, and the count spills, stored
/// on entry from rsi, where it arrives: the first `add` reads it there, and the sum that takes rsi ends that. The first
/// of the two `add`s at the end loads it to rcx, and the second reads it there.
#[test]
fn a_spilled_variable_is_read_in_the_register_a_store_or_a_load_left_it_in() -> Result<(), String> {
    let src = "unsafe extern \"sysv64\" fn k(p: *mut u64, n: u64) {
        let g: [gpr; 13];
        mov(g[0], [p]);
        add(g[0], n);
        for k in 1..13 { mov(g[k], [p + 8 * k]); }
        for k in 1..13 { add(g[0], g[k]); }
        add(g[0], n);
        add(g[0], n);
        mov([p], g[0]);
    }";
    let text = kernite::x86_str(src, "k")?.to_string();
    let parts = [
        "# n, spilled: 8 bytes at rsp - 8
# p=rdi n=rsi,rcx g0=rax g1=rcx g2=rdx g3=rsi",
        "    # n spilled
    mov qword ptr [rsp - 8], rsi
    mov rax, qword ptr [rdi]
    add rax, rsi
    mov rcx, qword ptr [rdi + 8]",
        "    mov rsi, qword ptr [rdi + 24]",
        "    # n reloaded
    mov rcx, qword ptr [rsp - 8]
    add rax, rcx
    add rax, rcx
    mov qword ptr [rdi], rax",
    ];
    for part in parts {
        assert!(text.contains(part), "{part}\nin\n{text}");
    }
    assert_eq!(text.matches("reloaded").count(), 1, "{text}");
    Ok(())
}

/// A call leaves in a register what it does not change, and a spilled variable is read there after it: the count, in
/// rsi, after `nothing`, which writes no register. A callee that writes rsi leaves nothing there, and the count is
/// loaded after the call.
#[test]
fn a_call_leaves_a_spilled_variable_in_a_register_it_does_not_change() -> Result<(), String> {
    let kernel = |callee: &str| {
        let src = format!(
            "unsafe extern \"sysv64\" fn k(p: *mut u64, n: u64) {{
                let g: [gpr; 13];
                mov(g[0], [p]);
                call({callee});
                add(g[0], n);
                for k in 1..13 {{ mov(g[k], [p + 8 * k]); }}
                for k in 1..13 {{ add(g[0], g[k]); }}
                add(g[0], n);
                mov([p], g[0]);
            }}
            unsafe extern \"sysv64\" fn nothing() {{}}
            unsafe extern \"sysv64\" fn writes_rsi() {{
                let (a, b, c, d): (gpr, gpr, gpr, gpr);
                mov(a, 1); mov(b, 1); mov(c, 1); mov(d, 1);
                add(a, b); add(a, c); add(a, d);
            }}"
        );
        kernite::x86_str(&src, "k").map(|listing| listing.to_string())
    };
    let text = kernel("nothing")?;
    let want = "    # n spilled
    mov qword ptr [rsp], rsi
    mov rax, qword ptr [rdi]
    call {f0}
    add rax, rsi
";
    assert!(text.contains(want), "{text}");
    assert_eq!(text.matches("reloaded").count(), 1, "{text}");
    let text = kernel("writes_rsi")?;
    let want = "    # n spilled
    mov qword ptr [rsp], rsi
    mov r8, qword ptr [rdi]
    call {f0}
    # n reloaded
    mov rax, qword ptr [rsp]
    add r8, rax
";
    assert!(text.contains(want), "{text}");
    assert_eq!(text.matches("reloaded").count(), 2, "{text}");
    Ok(())
}

/// An instruction that reads a spilled variable and writes it too takes a copy of the register that holds it, a `mov`
/// between registers, and not a load: the count is stored on entry from rsi, and `add(n, 1)` copies it to rcx, adds
/// there, and stores it from there. The `add` that reads it once, at the end, loads it, for no register holds it then.
#[test]
fn a_spilled_variable_that_an_instruction_writes_too_is_copied_from_the_register_that_holds_it() -> Result<(), String> {
    let src = "unsafe extern \"sysv64\" fn k(p: *mut u64, n: u64) {
        let g: [gpr; 13];
        mov(g[0], [p]);
        add(n, 1);
        for k in 1..13 { mov(g[k], [p + 8 * k]); }
        for k in 1..13 { add(g[0], g[k]); add(g[0], g[k]); }
        add(g[0], n);
        mov([p], g[0]);
    }";
    let text = kernite::x86_str(src, "k")?.to_string();
    let parts = [
        "# n, spilled: 8 bytes at rsp - 8\n",
        "    mov rax, qword ptr [rdi]
    mov rcx, rsi
    add rcx, 1
    # n spilled
    mov qword ptr [rsp - 8], rcx
",
        "    # n reloaded
    mov rcx, qword ptr [rsp - 8]
    add rax, rcx
    mov qword ptr [rdi], rax
",
    ];
    for part in parts {
        assert!(text.contains(part), "{part}\nin\n{text}");
    }
    assert_eq!(text.matches("reloaded").count(), 1, "{text}");
    Ok(())
}

/// A call of a function that runs `vzeroupper` clears the upper half of every vector register, so that a ymm the kernel
/// reads whole after it is kept in the frame, and loaded once after the call: `vaddps` and the store to `[q]` read the
/// register `w` was loaded to, with no store or load of `w` between them. `s`, an xmm that the kernel reads in its
/// lower half alone, stays in its register. A register that the callee does not write is cleared as well: `w` of
/// `vpast` is in ymm1, `vtouch` writes ymm0, and `w` is loaded after the call all the same.
#[test]
fn a_vector_the_call_clears_is_loaded_once_after_it_and_read_in_the_register_it_was_loaded_to() -> Result<(), String> {
    let src = r#"
        unsafe extern "sysv64" fn vtouch(p: *mut f32) -> f32 {
            let v: ymm;
            vmovups(v, [p]);
            vaddps(v, v, v);
            vmovups([p], v);
            ret(v.xmm);
        }
        unsafe extern "sysv64" fn vkeeps(p: *mut f32, q: *mut f32) -> f32 {
            let (w, s, r): (ymm, ymm, ymm);
            vmovups(w, [q]);
            vmovss(s.xmm, [q + 32]);
            call(r.xmm, vtouch, (p));
            vaddps(w, w, w);
            vmovups([q], w);
            vaddss(r.xmm, r.xmm, s.xmm);
            ret(r.xmm);
        }
        unsafe extern "sysv64" fn vpast(p: *mut f32, q: *mut f32) -> f32 {
            let (a, w, r): (ymm, ymm, ymm);
            vmovups(a, [q + 64]);
            vmovups(w, [q]);
            vmovups([q + 64], a);
            call(r.xmm, vtouch, (p));
            vaddps(w, w, w);
            vmovups([q], w);
            ret(r.xmm);
        }"#;
    let want = "# vkeeps(gpr, gpr) -> vec
# needs: avx
# frame: 32 bytes aligned to 32, saves []
# w, spilled: 32 bytes at rsp
# p=rdi q=rsi s=ymm1 r=ymm0 w=ymm0,ymm2
    push rbp
    mov rbp, rsp
    sub rsp, 32
    and rsp, -32
    vmovups ymm0, ymmword ptr [rsi]
    # w spilled
    vmovups ymmword ptr [rsp], ymm0
    vmovss xmm1, dword ptr [rsi + 32]
    call {f0}
    # w reloaded
    vmovups ymm2, ymmword ptr [rsp]
    vaddps ymm2, ymm2, ymm2
    vmovups ymmword ptr [rsi], ymm2
    vaddss xmm0, xmm0, xmm1
    mov rsp, rbp
    pop rbp
    vzeroupper
    ret
# f0: vtouch
";
    assert_eq!(kernite::x86_str(src, "vkeeps")?.to_string(), want);
    let want = "# vpast(gpr, gpr) -> vec
# needs: avx
# frame: 32 bytes aligned to 32, saves []
# w, spilled: 32 bytes at rsp
# p=rdi q=rsi a=ymm0 r=ymm0 w=ymm1
    push rbp
    mov rbp, rsp
    sub rsp, 32
    and rsp, -32
    vmovups ymm0, ymmword ptr [rsi + 64]
    vmovups ymm1, ymmword ptr [rsi]
    # w spilled
    vmovups ymmword ptr [rsp], ymm1
    vmovups ymmword ptr [rsi + 64], ymm0
    call {f0}
    # w reloaded
    vmovups ymm1, ymmword ptr [rsp]
    vaddps ymm1, ymm1, ymm1
    vmovups ymmword ptr [rsi], ymm1
    mov rsp, rbp
    pop rbp
    vzeroupper
    ret
# f0: vtouch
";
    assert_eq!(kernite::x86_str(src, "vpast")?.to_string(), want);
    Ok(())
}

/// A register holds a spilled variable at a label if each way to it brings it there. The count is stored on entry
/// from rsi, and the way that falls through leaves rsi as it was, as the way that jumps does, so that the `add` after
/// the label reads it there. Where the way that falls through writes the count, a store to its slot from rcx ends what
/// rsi holds of it, the ways bring it in two registers, and the count is loaded after the label.
#[test]
fn a_register_holds_a_spilled_variable_at_a_label_if_every_way_to_it_brings_it_there() -> Result<(), String> {
    let kernel = |way: &str| {
        let src = format!(
            "unsafe extern \"sysv64\" fn k(p: *mut u64, n: u64) {{
                let g: [gpr; 13];
                mov(g[0], [p]);
                test(g[0], g[0]);
                jz(skip);
                {way}
                label(skip);
                add(g[0], n);
                for k in 1..13 {{ mov(g[k], [p + 8 * k]); }}
                for k in 1..13 {{ add(g[0], g[k]); }}
                add(g[0], n);
                mov([p], g[0]);
            }}"
        );
        kernite::x86_str(&src, "k").map(|listing| listing.to_string())
    };
    let text = kernel("add(g[0], 1);")?;
    let want = "    jz 2f
    add rax, 1
2:
    add rax, rsi
";
    assert!(text.contains(want), "{text}");
    assert_eq!(text.matches("reloaded").count(), 1, "{text}");
    let text = kernel("mov(n, 5);")?;
    let want = "    jz 2f
    mov rcx, 5
    # n spilled
    mov qword ptr [rsp - 8], rcx
2:
    # n reloaded
    mov rcx, qword ptr [rsp - 8]
    add rax, rcx
";
    assert!(text.contains(want), "{text}");
    assert_eq!(text.matches("reloaded").count(), 2, "{text}");
    Ok(())
}

/// A register holds a spilled variable at a label though each way to it writes the variable: `c` is written to rcx on
/// each way of an `if`, and stored from there, and the `add` after the label reads it in rcx, with no load. The last
/// `add` loads it, for the sums between take rcx.
#[test]
fn a_register_holds_a_spilled_variable_at_a_label_though_each_way_writes_it() -> Result<(), String> {
    let src = "unsafe extern \"sysv64\" fn k(p: *mut u64) {
        let g: [gpr; 13];
        let c: gpr;
        mov(g[0], [p]);
        test(g[0], g[0]);
        jz(other);
        mov(c, 5);
        jmp(join);
        label(other);
        mov(c, 6);
        label(join);
        add(g[0], c);
        for k in 1..13 { mov(g[k], [p + 8 * k]); }
        for k in 1..13 { add(g[0], g[k]); }
        add(g[0], c);
        mov([p], g[0]);
    }";
    let text = kernite::x86_str(src, "k")?.to_string();
    let parts = [
        "    jz 2f
    mov rcx, 5
    # c spilled
    mov qword ptr [rsp - 8], rcx
    jmp 3f
2:
    mov rcx, 6
    # c spilled
    mov qword ptr [rsp - 8], rcx
3:
    add rax, rcx
    mov rcx, qword ptr [rdi + 8]
",
        "    # c reloaded
    mov rcx, qword ptr [rsp - 8]
    add rax, rcx
",
    ];
    for part in parts {
        assert!(text.contains(part), "{part}\nin\n{text}");
    }
    assert_eq!(text.matches("reloaded").count(), 1, "{text}");
    Ok(())
}

/// A call of a function that runs `vzeroupper` leaves the lower half of each vector register as it was, so that an xmm
/// kept in the frame stays in its register over it: `x` is stored from xmm1, `vtouch` writes ymm0 alone, and the
/// `vaddss` after the call reads `x` in xmm1, with no load. The `vaddss` at the end loads it, for the sums between take
/// xmm1.
#[test]
fn a_call_that_runs_vzeroupper_leaves_a_spilled_xmm_in_its_register() -> Result<(), String> {
    let src = r#"
        unsafe extern "sysv64" fn vtouch(p: *mut f32) -> f32 {
            let v: ymm;
            vmovups(v, [p]);
            vaddps(v, v, v);
            vmovups([p], v);
            ret(v.xmm);
        }
        unsafe extern "sysv64" fn k(p: *mut f32, q: *mut f32) -> f32 {
            let (a, x, r): (ymm, ymm, ymm);
            let g: [ymm; 15];
            vmovss(a.xmm, [q + 64]);
            vmovss(x.xmm, [q]);
            vmovss([q + 64], a.xmm);
            call(r.xmm, vtouch, (p));
            vaddss(r.xmm, r.xmm, x.xmm);
            for k in 0..15 { vmovss(g[k].xmm, [q + 4 * k + 4]); }
            for k in 0..15 { vaddss(r.xmm, r.xmm, g[k].xmm); }
            vaddss(r.xmm, r.xmm, x.xmm);
            ret(r.xmm);
        }"#;
    let text = kernite::x86_str(src, "k")?.to_string();
    let parts = [
        "    vmovss xmm1, dword ptr [rsi]
    # x spilled
    vmovups xmmword ptr [rsp], xmm1
    vmovss dword ptr [rsi + 64], xmm0
    call {f0}
    vaddss xmm0, xmm0, xmm1
    vmovss xmm1, dword ptr [rsi + 4]
",
        "    # x reloaded
    vmovups xmm1, xmmword ptr [rsp]
    vaddss xmm0, xmm0, xmm1
",
    ];
    for part in parts {
        assert!(text.contains(part), "{part}\nin\n{text}");
    }
    assert_eq!(text.matches("reloaded").count(), 1, "{text}");
    Ok(())
}

/// A store that no load follows is left out, and its slot with it. Seventeen vectors alive at once are one too many for
/// the 16 registers: kernite spills the last, `v16`, and then `v15`, to make room for the last to be written. The last
/// is read in the register it was written to, so that no load follows its store, and the frame has the slot of `v15`
/// alone, and the last `vaddps` reads `v16` in ymm15.
#[test]
fn a_store_that_no_load_follows_is_left_out_with_its_slot() -> Result<(), String> {
    let loads: String = (0..17).map(|i| format!("let v{i}: ymm;\nvmovups(v{i}, [x + {}]);\n", 32 * i)).collect();
    let sums: String = (1..17).map(|i| format!("vaddps(v0, v0, v{i});\n")).collect();
    let src = format!("unsafe extern \"sysv64\" fn k(x: *mut f32) {{\n{loads}{sums}vmovups([x], v0);\n}}\n");
    let listing = kernite::x86_str(&src, "k")?;
    let text = listing.to_string();
    let parts = [
        "# frame: 32 bytes aligned to 32, saves []\n# v15, spilled: 32 bytes at rsp\n",
        "    vmovups ymm15, ymmword ptr [rdi + 480]
    # v15 spilled
    vmovups ymmword ptr [rsp], ymm15
    vmovups ymm15, ymmword ptr [rdi + 512]
",
        "    # v15 reloaded
    vmovups ymm1, ymmword ptr [rsp]
    vaddps ymm0, ymm0, ymm1
    vaddps ymm0, ymm0, ymm15
",
    ];
    for part in parts {
        assert!(text.contains(part), "{part}\nin\n{text}");
    }
    assert_eq!(listing.frame.slots.iter().filter(|slot| slot.spill).count(), 1, "{text}");
    assert_eq!(text.matches("spilled").count(), 2, "{text}");
    Ok(())
}

/// A store is left out where another store of the variable follows before any load, though a load comes after the
/// second: `c` is written, read in rcx where it was written, and written again, and the `add` at the end loads it. Only
/// the second store stays, and the listing names the spill twice, in its header and at that store.
#[test]
fn a_store_that_another_store_of_the_variable_follows_before_any_load_is_left_out() -> Result<(), String> {
    let src = "unsafe extern \"sysv64\" fn k(p: *mut u64) {
        let g: [gpr; 13];
        let c: gpr;
        mov(g[0], [p]);
        mov(c, [p + 8]);
        add(g[0], c);
        mov(c, [p + 16]);
        for k in 1..13 { mov(g[k], [p + 8 * k]); }
        for k in 1..13 { add(g[0], g[k]); }
        add(g[0], c);
        mov([p], g[0]);
    }";
    let text = kernite::x86_str(src, "k")?.to_string();
    let want = "    mov rcx, qword ptr [rdi + 8]
    add rax, rcx
    mov rcx, qword ptr [rdi + 16]
    # c spilled
    mov qword ptr [rsp - 8], rcx
";
    assert!(text.contains(want), "{text}");
    assert_eq!(text.matches("spilled").count(), 2, "{text}");
    Ok(())
}

/// An instruction that reads two spilled variables reads each in the register that holds it, with neither load: `n` and
/// `m` arrive in rsi and rdx and are stored on entry, and `cmp` reads them there, though the load of `m` is followed by
/// the load of `n`, and not by the `cmp`, and the load of `n` goes to rdx, where `m` is: it is left out as well, and
/// writes no register. The `add`s at the end load each once, for the sums between take rsi and rdx.
#[test]
fn an_instruction_reads_two_spilled_variables_in_the_registers_that_hold_them() -> Result<(), String> {
    let src = "unsafe extern \"sysv64\" fn k(p: *mut u64, n: u64, m: u64) {
        let g: [gpr; 13];
        mov(g[0], [p]);
        cmp(n, m);
        jb(skip);
        add(g[0], 1);
        label(skip);
        for k in 1..13 { mov(g[k], [p + 8 * k]); }
        for k in 1..13 { add(g[0], g[k]); }
        add(g[0], n);
        add(g[0], m);
        mov([p], g[0]);
    }";
    let text = kernite::x86_str(src, "k")?.to_string();
    let parts = [
        "    # n spilled
    mov qword ptr [rsp - 8], rsi
    # m spilled
    mov qword ptr [rsp - 16], rdx
    mov rax, qword ptr [rdi]
    cmp rsi, rdx
    jb 2f
",
        "    # n reloaded
    mov rcx, qword ptr [rsp - 8]
    add rax, rcx
    # m reloaded
    mov rcx, qword ptr [rsp - 16]
    add rax, rcx
    mov qword ptr [rdi], rax
",
    ];
    for part in parts {
        assert!(text.contains(part), "{part}\nin\n{text}");
    }
    assert_eq!(text.matches("reloaded").count(), 2, "{text}");
    Ok(())
}

/// The kernel `k` around `body`, compiled, or the error.
fn compile(params: &str, body: &str) -> Result<String, String> {
    let src = format!("unsafe extern \"sysv64\" fn k({params}) {{\n{body}\n}}\n");
    kernite::x86_str(&src, "k").map(|l| l.code().collect::<Vec<_>>().join("\n"))
}

#[test]
fn memory_operands_take_every_form_x86_has() -> Result {
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
    )?;
    let want = "vmovups ymm0, ymmword ptr [rdi + rsi*4 + 16]
vmovups ymmword ptr [rdi - 32], ymm0
vbroadcastss ymm0, dword ptr [rip + {c0}]
vmulps ymm0, ymm0, ymmword ptr [rip + {c1}]
mov eax, dword ptr [rdi + rsi*4]
lea rax, [rdi + rsi*8]
vzeroupper
ret";
    assert_eq!(code, want);
    Ok(())
}

/// x86 scales an index by 1, 2, 4 or 8, so an index with no base beside it, times 3, 5 or 9, is
/// its own base: `[n + n * 2]`, which `[n * 3]` is too, since a variable that comes twice in an
/// address has its scales added. `llvm-mc` and GNU as assemble each as x86 encodes it.
#[test]
fn an_index_with_no_base_times_3_5_or_9_is_its_own_base() -> Result {
    let cases = [
        ("lea(x, [n + n * 2]);", "lea rdi, [rsi + rsi*2]", "48 8d 3c 76"),
        ("lea(x, [n * 3]);", "lea rdi, [rsi + rsi*2]", "48 8d 3c 76"),
        ("lea(x, [n * 5]);", "lea rdi, [rsi + rsi*4]", "48 8d 3c b6"),
        (
            "let v: ymm; vmovups(v, [n + n * 4 + 8]);",
            "vmovups ymm0, ymmword ptr [rsi + rsi*4 + 8]",
            "c5 fc 10 44 b6 08",
        ),
        ("lea(x, [n + n * 8]);", "lea rdi, [rsi + rsi*8]", "48 8d 3c f6"),
        ("lea(x, [n * 9]);", "lea rdi, [rsi + rsi*8]", "48 8d 3c f6"),
    ];
    addresses_encode("own-base", &cases)?;
    Ok(())
}

/// A gpr whose scales come to 0 is left out: `[x + n * 0]`, `[x + n - n]` and `[x + n * N]` where
/// the const `N` is 0 are all `[x]`. `llvm-mc` and GNU as assemble each as x86 encodes it.
#[test]
fn a_gpr_whose_scales_come_to_0_is_left_out() -> Result {
    let cases = [
        ("lea(x, [x + n * 0]);", "lea rdi, [rdi]", "48 8d 3f"),
        ("lea(x, [x + n - n]);", "lea rdi, [rdi]", "48 8d 3f"),
        ("const N: usize = 0; lea(x, [x + n * N]);", "lea rdi, [rdi]", "48 8d 3f"),
    ];
    addresses_encode("left-out", &cases)?;
    Ok(())
}

/// A base takes an index times 1, 2, 4 or 8, and a number. `llvm-mc` and GNU as assemble each as
/// x86 encodes it.
#[test]
fn a_base_takes_an_index_times_1_2_4_or_8_and_a_number() -> Result {
    let cases = [
        ("lea(x, [x + n]);", "lea rdi, [rdi + rsi*1]", "48 8d 3c 37"),
        ("lea(x, [x + n * 2]);", "lea rdi, [rdi + rsi*2]", "48 8d 3c 77"),
        ("lea(x, [x + n * 4 + 8]);", "lea rdi, [rdi + rsi*4 + 8]", "48 8d 7c b7 08"),
        ("lea(x, [x + n * 8 - 8]);", "lea rdi, [rdi + rsi*8 - 8]", "48 8d 7c f7 f8"),
    ];
    addresses_encode("scales", &cases)?;
    Ok(())
}

/// x86 has an index with no base, `[rcx*4]`, and writes it with a 32-bit displacement, 0 where
/// there is none. `[x + x * 3]` is `[x * 4]`, since a variable that comes twice in an address has
/// its scales added. A gather's vector of indices is an index too, `[ymm2*4]`, of a ymm or an xmm.
/// `llvm-mc` and GNU as assemble each as x86 encodes it.
#[test]
fn an_index_with_no_base_has_a_32_bit_displacement() -> Result {
    let cases = [
        ("lea(x, [n * 4]);", "lea rdi, [rsi*4]", "48 8d 3c b5 00 00 00 00"),
        ("lea(x, [n * 2]);", "lea rdi, [rsi*2]", "48 8d 3c 75 00 00 00 00"),
        ("lea(x, [n * 4 + 8]);", "lea rdi, [rsi*4 + 8]", "48 8d 3c b5 08 00 00 00"),
        ("lea(x, [n * 4 - 8]);", "lea rdi, [rsi*4 - 8]", "48 8d 3c b5 f8 ff ff ff"),
        (
            "let v: ymm; vmovups(v, [n * 8 + 32]);",
            "vmovups ymm0, ymmword ptr [rsi*8 + 32]",
            "c5 fc 10 04 f5 20 00 00 00",
        ),
        ("let v: ymm; vmovups(v, [x + x * 3]);", "vmovups ymm0, ymmword ptr [rdi*4]", "c5 fc 10 04 bd 00 00 00 00"),
        (
            "let v: ymm; let m: ymm; let i: ymm; zero(v); ones(m); zero(i); vpgatherdd(v, [i * 4], m);",
            "vpgatherdd ymm0, dword ptr [ymm2*4], ymm1",
            "c4 e2 75 90 04 95 00 00 00 00",
        ),
        (
            "let v: ymm; let m: ymm; let i: ymm; zero(v); ones(m); zero(i); vpgatherdd(v, [i * 8 - 8], m);",
            "vpgatherdd ymm0, dword ptr [ymm2*8 - 8], ymm1",
            "c4 e2 75 90 04 d5 f8 ff ff ff",
        ),
        (
            "let v: ymm; let m: ymm; let i: ymm; zero(v); ones(m); zero(i); vpgatherdd(v, [i], m);",
            "vpgatherdd ymm0, dword ptr [ymm2*1], ymm1",
            "c4 e2 75 90 04 15 00 00 00 00",
        ),
        (
            "let v: ymm; let m: ymm; let i: ymm; zero(v); ones(m); zero(i); vpgatherdq(v, [i.xmm * 8], m);",
            "vpgatherdq ymm0, qword ptr [xmm2*8], ymm1",
            "c4 e2 f5 90 04 d5 00 00 00 00",
        ),
    ];
    addresses_encode("no-base", &cases)?;
    Ok(())
}

/// Each of `cases`, a kernel's body, a line of its listing and the bytes x86 encodes that line
/// as, holds: the kernel compiles to a listing with that line, and `llvm-mc` and GNU as, where
/// they are here, make those bytes of it. `name` is for the files they read.
fn addresses_encode(name: &str, cases: &[(&str, &str, &str)]) -> Result {
    for &(body, line, _) in cases {
        let code = compile("x: *const f32, n: usize", body).map_err(|e| format!("{e}, of {body}"))?;
        assert!(code.lines().any(|l| l == line), "{line} is not a line of\n{code}\nof {body}");
    }
    let encoded: Vec<_> = cases.iter().map(|&(_, line, bytes)| (line, bytes)).collect();
    encode(name, &encoded)?;
    Ok(())
}

/// Each of `cases`, a line of a listing and the bytes x86 encodes it as, holds: `llvm-mc` and GNU
/// as, where they are here, make those bytes of it. `name` is for the files they read.
fn encode(name: &str, cases: &[(&str, &str)]) -> Result {
    let (Some(mc), Some(gnu)) = (tool("LLVM_MC", "llvm-mc"), tool("GNU_AS", "as")) else {
        eprintln!("no llvm-mc or GNU as here: skipped; set LLVM_MC and GNU_AS to run it");
        return Ok(());
    };
    let lines: Vec<&str> = cases.iter().map(|&(line, _)| line).collect();
    let (text, at) = source(&lines);
    let (ok, out, said) =
        assemble(&mc, &["-triple=x86_64", "-x86-asm-syntax=intel", "-show-encoding"], &format!("{name}-mc.s"), &text)?;
    assert!(ok && said.is_empty(), "llvm-mc says {said:?}, of\n{text}");
    // `leaq (,%rsi,4), %rdi  # encoding: [0x48,0x8d,0x3c,0xb5,0x00,0x00,0x00,0x00]`
    let made = out.lines().filter_map(|l| l.split_once("encoding: [")).map(|(_, e)| {
        e.trim_end_matches(']').split(',').map(|b| b.trim_start_matches("0x")).collect::<Vec<_>>().join(" ")
    });
    for ((line, want), made) in cases.iter().zip(made) {
        assert_eq!(made, *want, "llvm-mc, of {line}");
    }
    // GNU as's listing has a line's number, its offset and up to four of its bytes in hex in a
    // row, and more rows below it for the rest of them.
    let (ok, out, said) = assemble(&gnu, &["--64", "-al", "-o", "/dev/null"], &format!("{name}-gnu.s"), &text)?;
    assert!(ok && said.is_empty(), "GNU as says {said:?}, of\n{text}");
    let mut made: Vec<Vec<&str>> = vec![vec![]; cases.len()];
    for row in out.lines() {
        let n = row.get(..4).and_then(|n| n.trim().parse().ok());
        let hex = row.get(10..).and_then(|r| r.split('\t').next()).map_or("", str::trim);
        if let Some(bytes) = at.iter().position(|&a| Some(a) == n).and_then(|i| made.get_mut(i)) {
            bytes.extend(hex.as_bytes().chunks(2).filter_map(|pair| std::str::from_utf8(pair).ok()));
        }
    }
    for ((line, want), made) in cases.iter().zip(made) {
        assert_eq!(made.join(" ").to_lowercase(), *want, "GNU as, of {line}");
    }
    Ok(())
}

/// Memory is as much as the other operands say, or, where an instruction's forms differ by that
/// alone, as Intel's word before the brackets says. A write to `r16` or `r8` keeps the rest of
/// the register, so it needs a value there first.
#[test]
fn memory_is_as_much_as_the_operands_or_a_word_before_it_say() -> Result {
    let code = compile(
        "x: *mut u64",
        "let g: gpr; zero(g); movzx(g.r32, byte[x]); add(dword[x], 1); mov(g.r16, word[x]); mov(g.r8, [[7u8]]); \
         add(qword[x], g); add([x], g);",
    )?;
    let want = "xor eax, eax
movzx eax, byte ptr [rdi]
add dword ptr [rdi], 1
mov ax, word ptr [rdi]
mov al, byte ptr [rip + {c0}]
add qword ptr [rdi], rax
add qword ptr [rdi], rax
ret";
    assert_eq!(code, want);
    Ok(())
}

/// The word before the brackets is checked wherever it is given: on stack memory, on memory that
/// holds data, on a static, and on memory at a vector of indices, as on memory at a pointer.
#[test]
fn the_word_before_the_brackets_is_checked_wherever_it_is_given() -> Result {
    let cases = [
        (
            "let b: local<u64>; let g: gpr; zero(g); mov(qword[b], g); mov(g, dword[b]);",
            "3:71: `mov` wants 8 bytes of memory; this is 4 bytes of memory",
        ),
        (
            "let g: gpr; mov(g, dword[[1u64]]);",
            "3:25: `mov` wants 8 bytes of memory; this is 4 bytes of memory holding data",
        ),
        ("let g: gpr; mov(g, dword[S]);", "3:25: `mov` wants 8 bytes of memory; this is 4 bytes of memory"),
        (
            "let v: ymm; let m: ymm; let i: ymm; zero(v); ones(m); zero(i); vpgatherdd(v, qword[x + i * 4], m);",
            "3:83: `vpgatherdd` wants 4 bytes of memory at a ymm of indices; this is 8 bytes of memory at a ymm of \
             indices",
        ),
    ];
    for (body, want) in cases {
        let src = format!("static S: global<u64>;\nunsafe extern \"sysv64\" fn k(x: *const u32) {{\n{body}\n}}\n");
        assert_eq!(kernite::x86_str(&src, "k").err().ok_or("x86_str gave Ok")?, want, "{body}");
    }
    Ok(())
}

/// A gpr is `g`, or `g.r32`, `g.r16` or `g.r8`, its low 32, 16 or 8 bits, and nothing else.
#[test]
fn a_gpr_is_g_or_g_r32_g_r16_or_g_r8() -> Result {
    let code =
        compile("x: *mut u64", "let g: gpr; mov(g, [x]); mov(g.r32, 1); mov(g.r16, 2); mov(g.r8, 3); mov([x], g);");
    let want = "mov rax, qword ptr [rdi]
mov eax, 1
mov ax, 2
mov al, 3
mov qword ptr [rdi], rax
ret";
    assert_eq!(code?, want);
    let e = compile("x: *mut u64", "let g: gpr; mov(g.r64, 1);").err().ok_or("compile gave Ok")?;
    assert_eq!(e, "2:19: a gpr is seen as `gpr`, `r32`, `r16` or `r8`, not `r64`");
    Ok(())
}

/// A write to `r16` or `r8` keeps the rest of the gpr, as x86 does, so it reads the gpr, which
/// must hold a value first. One to `r32` clears the rest, as x86 does, and reads nothing.
#[test]
fn a_write_to_r16_or_r8_keeps_the_rest_of_the_gpr() -> Result {
    for body in ["let g: gpr; mov(g.r16, 1); mov([x], g);", "let g: gpr; mov(g.r8, 1); mov([x], g);"] {
        assert_eq!(
            compile("x: *mut u64", body).err().ok_or("compile gave Ok")?,
            "2:5: `g` is read before anything writes it",
            "{body}"
        );
    }
    let code = compile("x: *mut u64", "let g: gpr; mov(g.r32, 1); mov([x], g);");
    assert_eq!(code?, "mov eax, 1\nmov qword ptr [rdi], rax\nret");
    Ok(())
}

/// A gather reads memory at a vector of indices, `[p + v * 4]`, under a mask, and needs AVX2.
#[test]
fn a_gather_reads_memory_at_a_vector_of_indices() -> Result {
    let src = "unsafe extern \"sysv64\" fn k(x: *const f32) {
        let v: ymm; let m: ymm; let i: ymm;
        zero(v); ones(m); zero(i);
        vpgatherdd(v, [x + i * 4], m);
        vpgatherdq(v, [x + i.xmm * 8], m);
        vgatherqps(v.xmm, [x + i * 4], m.xmm);
    }";
    let listing = kernite::x86_str(src, "k")?;
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
    Ok(())
}

/// A gather takes its result, its indices and its mask in three registers, as x86 faults
/// otherwise: kernite keeps them apart where nothing reads the indices or the mask again, and
/// refuses a variable that is two of them, here the result and the mask.
#[test]
fn a_gather_takes_its_result_indices_and_mask_in_three_registers() -> Result {
    let body = "let v: ymm; let m: ymm; let i: ymm; zero(v); ones(m); zero(i); vpgatherdd(v, [x + i * 4], m); \
                vmovups([x], v);";
    let want = "vxorps xmm0, xmm0, xmm0
vpcmpeqd ymm1, ymm1, ymm1
vxorps xmm2, xmm2, xmm2
vpgatherdd ymm0, dword ptr [rdi + ymm2*4], ymm1
vmovups ymmword ptr [rdi], ymm0
vzeroupper
ret";
    assert_eq!(compile("x: *mut u32", body)?, want);
    let e = compile("x: *mut u32", "let v: ymm; let i: ymm; zero(v); zero(i); vpgatherdd(v, [x + i * 4], v);");
    assert_eq!(
        e.err().ok_or("e gave Ok")?,
        "2:70: `vpgatherdd` faults unless its result, its indices and its mask are three registers; `v` is its \
         result and its mask"
    );
    Ok(())
}

/// SysV leaves MXCSR's control bits as the caller set them, so a kernel that loads MXCSR stores
/// the caller's first, in its frame, and loads it back before it returns.
#[test]
fn a_kernel_that_loads_mxcsr_gives_the_callers_back() -> Result {
    let src = "unsafe extern \"sysv64\" fn k(x: *const u32) {\nvldmxcsr([x]);\n}\n";
    let want = "# k(gpr) -> ()
# needs: avx
# frame: 4 bytes aligned to 4, saves []
# mxcsr: 4 bytes at rsp - 4
# x=rdi
    # the caller's mxcsr, which it gets back
    vstmxcsr dword ptr [rsp - 4]
    vldmxcsr dword ptr [rdi]
    vldmxcsr dword ptr [rsp - 4]
    ret
";
    assert_eq!(kernite::x86_str(src, "k")?.to_string(), want);
    Ok(())
}

/// A kernel that loads MXCSR twice and returns in two places stores the caller's once, before
/// its first load, and loads it back before each return.
#[test]
fn a_kernel_that_loads_mxcsr_loads_the_callers_back_before_each_return() -> Result {
    let src = "unsafe extern \"sysv64\" fn k(x: *const u32, n: u64) {\nvldmxcsr([x]);\ntest(n, n);\njz(done);\n\
               vldmxcsr([x + 4]);\nret();\nlabel(done);\n}\n";
    let want = "# k(gpr, gpr) -> ()
# needs: avx
# frame: 4 bytes aligned to 4, saves []
# mxcsr: 4 bytes at rsp - 4
# x=rdi n=rsi
    # the caller's mxcsr, which it gets back
    vstmxcsr dword ptr [rsp - 4]
    vldmxcsr dword ptr [rdi]
    test rsi, rsi
    jz 2f
    vldmxcsr dword ptr [rdi + 4]
    vldmxcsr dword ptr [rsp - 4]
    ret
2:
    vldmxcsr dword ptr [rsp - 4]
    ret
";
    assert_eq!(kernite::x86_str(src, "k")?.to_string(), want);
    Ok(())
}

#[test]
fn a_wrong_operand_is_named_by_what_the_instruction_wants() -> Result {
    let cases = [
        ("let v: ymm; vaddps(v, v.xmm, v);", "2:25: `vaddps` wants a ymm; this is an xmm"),
        ("let v: ymm; vaddps(v, v);", "2:13: `vaddps` takes 3 operands, not 2"),
        ("let v: ymm; vmovups(v, [1.0]);", "2:24: this reads 32 bytes of a constant of 4"),
        ("let i: gpr; let v: ymm; vmovups(v, [x + i * 3]);", "2:36: x86 scales an index by 1, 2, 4 or 8, not 3"),
        ("let i: gpr; let v: ymm; vmovups(v, [i * 3 + x]);", "2:36: x86 scales an index by 1, 2, 4 or 8, not 3"),
        (
            "let b: local<[f32; 16]>; let i: gpr; let p: gpr; lea(p, [b + i * 3]);",
            "2:57: x86 scales an index by 1, 2, 4 or 8, not 3",
        ),
        ("let i: gpr; let v: ymm; vmovups(v, [x + i * 16]);", "2:36: x86 scales an index by 1, 2, 4 or 8, not 16"),
        ("let i: gpr; let v: ymm; vmovups(v, [i * 6]);", "2:36: x86 scales an index by 1, 2, 4 or 8, not 6"),
        ("let i: gpr; let v: ymm; vmovups(v, [i + i * 5]);", "2:36: x86 scales an index by 1, 2, 4 or 8, not 6"),
        ("let v: ymm; vaddps(v, v, 1.5);", "2:26: x86 takes floats from memory: `[1.5]`"),
        ("vfrobps(x);", "2:1: x86 has no instruction `vfrobps` that kernite knows"),
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
            "let v: ymm; let m: ymm; zero(v); ones(m); vpgatherdd(v, [x * 4], m);",
            "2:57: `vpgatherdd` wants 4 bytes of memory at a ymm of indices; this is memory at an index in a gpr",
        ),
        (
            "let v: ymm; let m: ymm; zero(v); ones(m); vpgatherdd(v, [x], m);",
            "2:57: `vpgatherdd` wants 4 bytes of memory at a ymm of indices; this is memory",
        ),
        (
            "let v: ymm; let m: ymm; let i: ymm; zero(v); ones(m); zero(i); vpgatherdd(v, [i * 3], m);",
            "2:78: x86 scales an index by 1, 2, 4 or 8, not 3",
        ),
        ("jmp(x);", "2:5: `jmp` wants a label; this is a gpr"),
        ("let g: gpr; add(g, top); label(top);", "2:20: a label is for a jump"),
        ("rdtscp();", "2:1: x86 has no instruction `rdtscp` that kernite knows"),
        ("let v: ymm; addps(v.xmm, v.xmm);", "2:13: kernite writes `addps` as AVX does: `vaddps`"),
        ("vzeroupper();", "2:1: kernite does not yet write `vzeroupper`, which clears the upper half of every vector"),
        ("pcmpestri(x, x, 0);", "2:1: kernite writes `pcmpestri` as AVX does: `vpcmpestri`"),
        ("mul(x);", "2:1: kernite does not yet write `mul`, which multiplies `rax` into `rdx:rax`"),
        ("imul(x);", "2:1: kernite does not yet write `imul` of 1 operand, which multiplies `rax` into `rdx:rax`"),
        ("shld(x, x);", "2:1: kernite does not yet write `shld` of 2 operands, which shifts by `cl`"),
        (
            "shl(x, cl);",
            "2:8: kernite does not yet write `shl` by `cl`, a register of its own: shift by a number, or by a gpr with \
             `shlx`, `shrx` or `sarx`",
        ),
    ];
    for (body, want) in cases {
        assert_eq!(compile("x: *const f32", body).err().ok_or("compile gave Ok")?, want, "{body}");
    }
    Ok(())
}

/// An immediate x86 reads in part is no more than it reads: 6 bits of a shift's count, 5 of
/// `vcmpps`'s control byte, 1 of a ymm's half, and a bit for each lane `vpblendd` blends.
#[test]
fn an_immediate_is_what_x86_reads_of_it() -> Result {
    let code = compile(
        "x: *const f32",
        "let v: ymm; vmovups(v, [x]); shl(x, 63); vcmpps(v, v, v, 31); vextractf128(v.xmm, v, 1); vpslld(v, v, \
         255); vpblendd(v, v, v, 0xff); vpblendd(v.xmm, v.xmm, v.xmm, 0xf);",
    )?;
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
    Ok(())
}

/// A program built in Rust writes an index with no base as the language does: `n.times(4)` is
/// `[rsi*4]`, and `plus` puts a displacement after it, in the bytes of the language's `[n * 4]` and
/// `[n * 4 + 8]`. A gather's vector of indices is written so too: `i.times(4)` is `[ymm2*4]`, and
/// `i.xmm().times(8)` is `[xmm2*8]`, in the bytes of the language's `vpgatherdd(v, [i * 4], m)` and
/// `vpgatherdq(v, [i.xmm * 8], m)`. Only the builder writes a scale of 1 of a gpr, `[rsi*1]`, where
/// the language writes a base, `[n]`.
#[test]
fn the_builder_writes_an_index_with_no_base() -> Result {
    let mut a = Asm::new("k");
    let (x, n) = (a.param("x"), a.param("n"));
    let (v, m, i) = (a.ymm("v"), a.ymm("m"), a.ymm("i"));
    a.zero(v);
    a.ones(m);
    a.zero(i);
    a.lea(x, n.times(1)?);
    a.lea(x, n.times(2)?);
    a.lea(x, n.times(4)?);
    a.lea(x, n.times(4)?.plus(8));
    a.lea(x, n.times(8)?.plus(-8));
    a.vpgatherdd(v, i.times(4)?, m);
    a.ones(m);
    a.vpgatherdd(v, i.times(8)?.plus(-8), m);
    a.ones(m);
    a.vpgatherdq(v, i.xmm().times(8)?, m);
    a.ret();
    let listing = intel::print(&kernite_compile::compile(&a.finish()?, &SysV)?, vec![])?;
    let cases = [
        ("lea rdi, [rsi*1]", "48 8d 3c 35 00 00 00 00"),
        ("lea rdi, [rsi*2]", "48 8d 3c 75 00 00 00 00"),
        ("lea rdi, [rsi*4]", "48 8d 3c b5 00 00 00 00"),
        ("lea rdi, [rsi*4 + 8]", "48 8d 3c b5 08 00 00 00"),
        ("lea rdi, [rsi*8 - 8]", "48 8d 3c f5 f8 ff ff ff"),
        ("vpgatherdd ymm0, dword ptr [ymm2*4], ymm1", "c4 e2 75 90 04 95 00 00 00 00"),
        ("vpgatherdd ymm0, dword ptr [ymm2*8 - 8], ymm1", "c4 e2 75 90 04 d5 f8 ff ff ff"),
        ("vpgatherdq ymm0, qword ptr [xmm2*8], ymm1", "c4 e2 f5 90 04 d5 00 00 00 00"),
    ];
    for (line, _) in cases {
        assert!(listing.code().any(|l| l == line), "{line} is not a line of\n{listing}");
    }
    encode("builder", &cases)?;
    Ok(())
}

/// The builder takes no more of an immediate than x86 reads, nor a scale x86 has not, as the
/// language does not: an address is refused where it is made, and an instruction at `finish`, which
/// says the first it was given.
#[test]
fn the_builder_refuses_what_x86_would_not_read() -> Result {
    let finish = |build: fn(&mut Asm)| {
        let mut a = Asm::new("k");
        build(&mut a);
        a.finish().err()
    };
    let refused = |want: &str, build: fn(&mut Asm)| assert_eq!(finish(build).as_deref(), Some(want));
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
    // The first of two wrong values is the one it says.
    refused("shl by 64: x86 shifts a gpr by 0 to 63", |a| {
        let g = a.gpr("g");
        a.shl(g, 64);
        a.shr(g, 65);
    });
    refused("a seventh integer parameter, where SysV passes six in registers", |a| {
        for name in ["a", "b", "c", "d", "e", "f", "g"] {
            a.param(name);
        }
    });
    refused("`slot` is aligned to 3, where an alignment is a power of two, a page at most", |a| {
        a.slot("slot", 8, 3);
    });
    refused("`slot` is aligned to 8192, where an alignment is a power of two, a page at most", |a| {
        a.slot("slot", 8, 8192);
    });
    refused("a constant is aligned to 6, which is no power of two", |a| {
        a.constant(&[1], 6);
    });
    refused("a label in a captured stream", |a| {
        let label = a.label();
        a.capture(|a| a.place(label));
    });
    refused("`jmp` in a captured stream", |a| {
        let label = a.label();
        a.capture(|a| a.jmp(label));
    });
    refused("`vpgatherdd` reads memory at a vector of indices", |a| {
        let (g, v) = (a.gpr("g"), a.ymm("v"));
        a.vpgatherdd(v, g.at(0), v);
    });
    refused("`vpgatherdd` faults unless its result, indices and mask are three registers", |a| {
        let (i, v) = (a.ymm("i"), a.ymm("v"));
        if let Ok(from) = i.times(4) {
            a.vpgatherdd(v, from, v);
        }
    });
    let mut a = Asm::new("k");
    let (g, v) = (a.gpr("g"), a.ymm("v"));
    assert_eq!(g.times(3).err().as_deref(), Some("scale 3"));
    assert_eq!(v.times(16).err().as_deref(), Some("scale 16"));
    assert_eq!(v.xmm().times(0).err().as_deref(), Some("scale 0"));
    assert_eq!(g.at(0).index(g, 4)?.index_ymm(v, 4).err().as_deref(), Some("one index to an address"));
    assert_eq!(a.u32s(&[1]).index(g, 4).err().as_deref(), Some("a constant's address takes no index"));
    a.shl(g, 63);
    a.vextractf128(v.xmm(), v, 1);
    a.vpblendd(v, v, v, 0xff);
    a.vpblendd(v.xmm(), v.xmm(), v.xmm(), 0x0f);
    assert!(a.finish().is_ok());
    Ok(())
}

#[test]
fn a_kernel_takes_what_sysv_passes_in_registers() -> Result {
    let e = compile("a: u32", "").err().ok_or("compile gave Ok")?;
    assert_eq!(e, "1:32: SysV leaves the high bits of a `u32` as they were; take a usize or a u64");
    let code = compile("a: f32, p: *mut f32, b: f32", "vmovss([p], b.xmm); vaddss(b.xmm, a.xmm, b.xmm);")?;
    assert_eq!(code, "vmovss dword ptr [rdi], xmm1\nvaddss xmm1, xmm0, xmm1\nret");
    Ok(())
}

/// What kernite says of the instruction `name` that reads `reads` bits of `x`, of which only the low `has` are the
/// kernel's.
fn past_the_float(name: &str, reads: u32, has: u32, x: &str) -> String {
    format!(
        "`{name}` reads {reads} bits of `{x}`, of which only the low {has} are the kernel's: SysV leaves the rest of \
         the register of a float as the caller had it, or the callee, of one it gave back. Read the float alone, as a \
         scalar instruction does, or write all of `{x}` first"
    )
}

/// SysV passes a float in the low bits of a vector register, 32 of an `f32` and 64 of an `f64`, and leaves the rest as
/// the caller had it. A call gives one back so, and leaves the rest as the callee had it. A variable has as many of
/// the low bits of its register for its own as it has been given: a float's, to begin with, and all of the register
/// where the kernel has written it. An instruction that reads more of it is an error at the instruction, which names
/// the variable, on any way to it. One that carries the caller's bits to the same place of its variable, as `vaddss`
/// and `vmovaps` do, hands them on to it, whatever it is.
#[test]
fn a_kernel_reads_no_more_of_a_float_it_takes_than_the_float() {
    // Each case its parameters and body, where the error is, and what it says: the instruction, the bits it reads,
    // the bits the variable has, and its name.
    let cases = [
        // All of an xmm, and of a ymm, of an `f32` and of an `f64`; 64 bits of an `f32`; and what a gpr or the flags
        // take.
        ("x: f32, p: *mut f32", "vmovups([p], x);", "2:1", ("vmovups", 128, 32, "x")),
        ("x: f32, p: *mut f32", "vmovups([p], x.ymm);", "2:1", ("vmovups", 256, 32, "x")),
        ("x: f64, p: *mut f64", "vmovupd([p], x);", "2:1", ("vmovupd", 128, 64, "x")),
        ("x: f64, p: *mut f64", "vmovupd([p], x.ymm);", "2:1", ("vmovupd", 256, 64, "x")),
        ("x: f32, p: *mut f64", "vmovsd([p], x);", "2:1", ("vmovsd", 64, 32, "x")),
        ("x: f32, n: usize", "vmovq(n, x);", "2:1", ("vmovq", 64, 32, "x")),
        ("x: f32, n: usize", "vmovmskps(n.r32, x);", "2:1", ("vmovmskps", 128, 32, "x")),
        ("x: f32", "vptest(x, x);", "2:1", ("vptest", 128, 32, "x")),
        // An instruction that reads and writes it, and each float the kernel takes, by the bits it has.
        ("x: f32, y: f32, z: f32", "vfmadd231ps(x, y, z);", "2:1", ("vfmadd231ps", 128, 32, "x")),
        ("x: f32, y: f32, p: *mut f32", "zero(x);\nvmovups([p], y);", "3:1", ("vmovups", 128, 32, "y")),
        ("x: f32, y: f64, p: *mut f64", "vmovsd([p], y);\nvmovupd([p], y);", "3:1", ("vmovupd", 128, 64, "y")),
        // A scalar instruction writes its own bits, 32 of `vaddss`, and passes the rest of its first source on, so `x`
        // has no more than the instruction writes and the source had: `vcvtss2sd` writes 64.
        ("x: f32, y: f32, p: *mut f32", "vaddss(x, x, y);\nvmovups([p], x);", "3:1", ("vmovups", 128, 32, "x")),
        ("x: f32, y: f32, p: *mut f32", "vmovss(x, y, y);\nvmovups([p], x);", "3:1", ("vmovups", 128, 32, "x")),
        ("x: f32, n: usize, p: *mut f32", "vcvtsi2ss(x, x, n);\nvmovups([p], x);", "3:1", ("vmovups", 128, 32, "x")),
        (
            "x: f32, y: f32, z: f32, p: *mut f32",
            "vfmadd231ss(x, y, z);\nvmovups([p], x);",
            "3:1",
            ("vmovups", 128, 32, "x"),
        ),
        ("x: f32, y: f32, p: *mut f64", "vcvtss2sd(x, x, y);\nvmovupd([p], x);", "3:1", ("vmovupd", 128, 64, "x")),
        // An instruction that carries a register carries it as it was: `vmovaps(x, x)` leaves `x` as it was, and so
        // does `vxorps(x, x, x)`, which computes zero but does not read `x`. `zero(x)` writes it.
        ("x: f32, p: *mut f32", "vmovaps(x, x);\nvmovups([p], x);", "3:1", ("vmovups", 128, 32, "x")),
        ("x: f32, p: *mut f32", "vxorps(x, x, x);\nvmovups([p], x);", "3:1", ("vmovups", 128, 32, "x")),
        // The bits go to the variable the instruction writes, whatever it is: one made with `let`, and a ymm, which
        // a move of a whole register fills with them.
        (
            "x: f32, y: f32, p: *mut f32",
            "let z: xmm;\nvaddss(z, x, y);\nvmovups([p], z);",
            "4:1",
            ("vmovups", 128, 32, "z"),
        ),
        ("x: f32, p: *mut f32", "let v: ymm;\nvmovaps(v, x.ymm);\nvmovups([p], v);", "4:1", ("vmovups", 256, 32, "v")),
    ];
    for (params, body, at, (name, reads, has, x)) in cases {
        let want = format!("{at}: {}", past_the_float(name, reads, has, x));
        assert_eq!(compile(params, body), Err(want), "{body}");
    }
    // A function of the module is held to it as a kernel is, whoever calls it.
    let held = "unsafe extern \"sysv64\" fn held(x: f32, p: *mut f32) {\nvmovups([p], x);\n}\n";
    let want = format!("2:1: {}", past_the_float("vmovups", 128, 32, "x"));
    assert_eq!(kernite::x86_str(held, "held").map(|_| ()), Err(want));
    // `ret` reads as far as the kernel gives back, 64 bits of an `f64`.
    let want = format!("2:1: {}", past_the_float("ret", 64, 32, "x"));
    assert_eq!(calling("x: f32", " -> f64", "ret(x);"), Err(want));
}

/// A float is read as a float, and read whole after the kernel has written all of it, as it writes any variable:
/// `zero` and `ones`, a load, a broadcast, an instruction that writes all of an xmm or a ymm, and a scalar one that
/// takes the rest from a variable that is the kernel's own.
#[test]
fn a_kernel_reads_a_float_it_takes_as_a_float_and_whole_once_it_has_written_it() {
    let cases = [
        ("x: f32, p: *mut f32", "vmovss([p], x);", "vmovss dword ptr [rdi], xmm0\nret"),
        ("x: f64, p: *mut f64", "vmovsd([p], x);", "vmovsd qword ptr [rdi], xmm0\nret"),
        ("x: f64, n: usize", "vmovq(n, x);", "vmovq rdi, xmm0\nret"),
        ("x: f32, y: f64, p: *mut f64", "vmovsd([p], y);", "vmovsd qword ptr [rdi], xmm1\nret"),
        (
            "x: f32, p: *mut f32",
            "zero(x);\nvmovups([p], x);",
            "vxorps xmm0, xmm0, xmm0\nvmovups xmmword ptr [rdi], xmm0\nret",
        ),
        (
            "x: f32, p: *mut f32",
            "ones(x);\nvmovups([p], x);",
            "vpcmpeqd xmm0, xmm0, xmm0\nvmovups xmmword ptr [rdi], xmm0\nret",
        ),
        (
            "x: f32, p: *mut f32, q: *const f32",
            "vmovups(x, [q]);\nvmovups([p], x);",
            "vmovups xmm0, xmmword ptr [rsi]\nvmovups xmmword ptr [rdi], xmm0\nret",
        ),
        (
            "x: f32, p: *mut f32, q: *const f32",
            "vmovss(x, [q]);\nvmovups([p], x);",
            "vmovss xmm0, dword ptr [rsi]\nvmovups xmmword ptr [rdi], xmm0\nret",
        ),
        (
            "x: f32, p: *mut f32",
            "vbroadcastss(x.ymm, x);\nvmovups([p], x.ymm);",
            "vbroadcastss ymm0, xmm0\nvmovups ymmword ptr [rdi], ymm0\nvzeroupper\nret",
        ),
        // An instruction of three operands that is no scalar one writes all of its first, and so does one that unpacks
        // the low halves of its sources, which is all it reads of them: two `f64`s fill the register of an `f32`.
        (
            "x: f32, p: *mut f32",
            "let y: xmm;\nzero(y);\nvaddps(x, y, y);\nvmovups([p], x);",
            "vxorps xmm0, xmm0, xmm0\nvaddps xmm0, xmm0, xmm0\nvmovups xmmword ptr [rdi], xmm0\nret",
        ),
        (
            "x: f32, y: f64, p: *mut f32",
            "vunpcklps(x, y, y);\nvmovups([p], x);",
            "vunpcklps xmm0, xmm1, xmm1\nvmovups xmmword ptr [rdi], xmm0\nret",
        ),
        (
            "x: f32, y: f64, p: *mut f32",
            "vmovlhps(x, y, y);\nvmovups([p], x);",
            "vmovlhps xmm0, xmm1, xmm1\nvmovups xmmword ptr [rdi], xmm0\nret",
        ),
        // A scalar instruction takes the rest of its result from its first source, which is the kernel's own where
        // the kernel wrote it: all of `x` is.
        (
            "x: f32, y: f32, p: *mut f32",
            "let z: xmm;\nzero(z);\nvaddss(x, z, y);\nvmovups([p], x);",
            "vxorps xmm0, xmm0, xmm0\nvaddss xmm0, xmm0, xmm1\nvmovups xmmword ptr [rdi], xmm0\nret",
        ),
    ];
    for (params, body, want) in cases {
        assert_eq!(compile(params, body).as_deref(), Ok(want), "{body}");
    }
    // Each scalar instruction reads the float alone, and `x` is passed on and given back as it is.
    let scalar = "vaddss(y, y, x);\nvsqrtss(y, y, x);\nvcvtss2sd(y, y, x);\nvucomiss(x, y);\nvmovd(n.r32, x);\n\
                  vbroadcastss(w.ymm, x);\nvfmadd231ss(y, x, x);";
    assert_eq!(compile("x: f32, y: f32, w: f32, n: usize", scalar).map(|_| ()), Ok(()));
    assert_eq!(calling("x: f32", " -> f32", "ret(x);").as_deref(), Ok("ret"));
    // The ymm of a float is given back as the float: `ret` reads what the type of the result is.
    assert_eq!(calling("x: f32", " -> f32", "ret(x.ymm);").as_deref(), Ok("ret"));
}

/// A float is read as far as every way to the read has written it. The ways are the program's: at a label a variable
/// has the least that any way to it brings, a jump back among them, and what no way comes to reads nothing.
#[test]
fn a_float_is_read_as_far_as_every_way_to_the_read_has_written_it() {
    let params = "x: f32, n: usize, p: *mut f32";
    // Each body, and where the read of `x` that no way writes before is.
    let refused = [
        // One way writes it, and one does not.
        ("test(n, n);\njz(skip);\nzero(x);\nlabel(skip);\nvmovups([p], x);", "6:1", 128),
        // The first round of a loop reads it before the loop writes it.
        ("label(top);\nvmovups([p], x);\nzero(x);\ndec(n);\njnz(top);", "3:1", 128),
        // A jump back to the read from a way that did not write it, where the way before the label did.
        (
            "test(n, n);\njz(skip);\nzero(x);\nlabel(top);\nvmovups([p], x.ymm);\nlabel(skip);\ndec(n);\njnz(top);",
            "6:1",
            256,
        ),
        // Two jumps to a label, and the way before it: the least of the three.
        ("test(n, n);\njz(skip);\nzero(x);\ntest(n, n);\njz(skip);\nlabel(skip);\nvmovups([p], x);", "8:1", 128),
        // A label that only a jump comes to: the code before it, which `jmp` leaves, comes to it by no way.
        ("jmp(done);\nlabel(other);\nzero(x);\nlabel(done);\nvmovups([p], x);", "6:1", 128),
        // A jump back that brings `x` narrowed to a label, and on past a branch after it to the read, which a pass over
        // the blocks in order reaches first.
        (
            "let t: xmm;\nvmovaps(t, x);\nzero(x);\nlabel(top);\ntest(n, n);\njz(done);\nvmovups([p], x);\n\
             vmovaps(x, t);\ndec(n);\njmp(top);\nlabel(done);",
            "8:1",
            128,
        ),
    ];
    for (body, at, reads) in refused {
        let want = format!("{at}: {}", past_the_float("vmovups", reads, 32, "x"));
        assert_eq!(compile(params, body), Err(want), "{body}");
    }
    // Each way writes it, or no way comes to the read: a loop entered at its test, the shape of a `while`, whose way
    // in writes all of what the loop reads; a jump back from ways that did not write it, to a label that the code
    // after it writes; and code after `ret`, which no way comes to, though the code before it did not write `x`.
    let params_rotated = "a: f32, x: *const f32, y: *mut f32, n: u64";
    let rotated = "let acc: ymm;\nvbroadcastss(a.ymm, a);\nzero(acc);\njmp(test);\nlabel(top);\n\
                   vfmadd231ps(acc, a.ymm, [x]);\nadd(x, 32);\nsub(n, 8);\nlabel(test);\ntest(n, n);\njnz(top);\n\
                   vmovups([y], acc);";
    let want = "vbroadcastss ymm0, xmm0\nvxorps xmm1, xmm1, xmm1\njmp 3f\n2:\nvfmadd231ps ymm1, ymm0, ymmword ptr \
                [rdi]\nadd rdi, 32\nsub rdx, 8\n3:\ntest rdx, rdx\njnz 2b\nvmovups ymmword ptr [rsi], ymm1\n\
                vzeroupper\nret";
    assert_eq!(compile(params_rotated, rotated).as_deref(), Ok(want));
    let again = "test(n, n);\njz(skip);\nzero(x);\nlabel(top);\nvbroadcastss(x.ymm, [p]);\nvmovups([p], x.ymm);\n\
                 label(skip);\ndec(n);\njnz(top);";
    let want = "test rdi, rdi\njz 3f\nvxorps xmm0, xmm0, xmm0\n2:\nvbroadcastss ymm0, dword ptr [rsi]\nvmovups \
                ymmword ptr [rsi], ymm0\n3:\ndec rdi\njnz 2b\nvzeroupper\nret";
    assert_eq!(compile(params, again).as_deref(), Ok(want));
    for body in ["ret();\nlabel(top);\nvmovups([p], x);\ndec(n);\njnz(top);", "zero(x);\nret();\nvmovups([p], x);"] {
        assert_eq!(compile(params, body).map(|_| ()), Ok(()), "{body}");
    }
    let body = "zero(x);\ncall(stop);\nlabel(top);\nvmovups([p], x);\njmp(top);";
    assert_eq!(calling(params, "", body).map(|_| ()), Ok(()), "{body}");
    // A cooperative kernel's `grid.sync()` leaves what the kernel wrote of `x` as it was, and refuses what it did not.
    let want = past_the_float("vmovups", 128, 32, "x");
    assert_eq!(cooperative("x: f32, p: *mut f32", "grid.sync();\nvmovups([p], x);"), Err(format!("4:1: {want}")));
    assert_eq!(cooperative("x: f32, p: *mut f32", "zero(x);\ngrid.sync();\nvmovups([p], x);").map(|_| ()), Ok(()));
}

/// What rustc writes for a float, written as kernels, compiles to what rustc writes: each instruction carries the bits
/// past the float to the same place of its result, as `vandps` does in `x.abs()`, and reads none of them.
#[test]
fn rustcs_code_for_a_float_compiles_as_it_is() {
    // Its parameters, what it returns, its body, and the code: `abs`, `neg`, `copysign`, `round`, `min`, `max`, a
    // select, `_mm_set_ss`, `_mm_setr_ps` and `_mm_setr_pd`, `*p = x as f64`.
    let cases = [
        (
            "x: f32, m: *const f32",
            " -> f32",
            "let t: xmm; vbroadcastss(t, [m]); vandps(x, x, t); ret(x);",
            "vbroadcastss xmm1, dword ptr [rdi]\nvandps xmm0, xmm0, xmm1\nret",
        ),
        (
            "x: f32, m: *const f32",
            " -> f32",
            "let t: xmm; vbroadcastss(t, [m]); vxorps(x, x, t); ret(x);",
            "vbroadcastss xmm1, dword ptr [rdi]\nvxorps xmm0, xmm0, xmm1\nret",
        ),
        (
            "x: f32, y: f32, s: *const f32, a: *const f32",
            " -> f32",
            "let t: xmm; vbroadcastss(t, [s]); vandps(y, y, t); vbroadcastss(t, [a]); vandps(x, x, t); vorps(x, x, y); \
             ret(x);",
            "vbroadcastss xmm2, dword ptr [rdi]\nvandps xmm1, xmm1, xmm2\nvbroadcastss xmm2, dword ptr [rsi]\nvandps \
             xmm0, xmm0, xmm2\nvorps xmm0, xmm0, xmm1\nret",
        ),
        (
            "x: f32, s: *const f32, h: *const f32",
            " -> f32",
            "let (t, u): (xmm, xmm); vbroadcastss(t, [s]); vandps(t, x, t); vbroadcastss(u, [h]); vorps(t, t, u); \
             vaddss(x, x, t); vroundss(x, x, x, 11); ret(x);",
            "vbroadcastss xmm1, dword ptr [rdi]\nvandps xmm1, xmm0, xmm1\nvbroadcastss xmm2, dword ptr [rsi]\nvorps \
             xmm1, xmm1, xmm2\nvaddss xmm0, xmm0, xmm1\nvroundss xmm0, xmm0, xmm0, 0x0b\nret",
        ),
        (
            "x: f32, y: f32",
            " -> f32",
            "let t: xmm; vminss(t, x, y); vcmpunordss(y, y, y); vblendvps(x, t, x, y); ret(x);",
            "vminss xmm2, xmm0, xmm1\nvcmpunordss xmm1, xmm1, xmm1\nvblendvps xmm0, xmm2, xmm0, xmm1\nret",
        ),
        (
            "x: f64, y: f64",
            " -> f64",
            "let t: xmm; vmaxsd(t, x, y); vcmpunordsd(y, y, y); vblendvpd(x, t, x, y); ret(x);",
            "vmaxsd xmm2, xmm0, xmm1\nvcmpunordsd xmm1, xmm1, xmm1\nvblendvpd xmm0, xmm2, xmm0, xmm1\nret",
        ),
        (
            "c: u64, x: f32, y: f32",
            " -> f32",
            "test(c, c); jnz(done); vmovaps(x, y); label(done); ret(x);",
            "test rdi, rdi\njnz 2f\nvmovaps xmm0, xmm1\n2:\nret",
        ),
        (
            "x: f32, p: *mut f32",
            "",
            "let z: xmm; zero(z); vblendps(x, z, x, 1); vmovups([p], x);",
            "vxorps xmm1, xmm1, xmm1\nvblendps xmm0, xmm1, xmm0, 0x01\nvmovups xmmword ptr [rdi], xmm0\nret",
        ),
        (
            "a: f32, b: f32, c: f32, d: f32, p: *mut f32",
            "",
            "vinsertps(a, a, b, 16); vinsertps(a, a, c, 32); vinsertps(a, a, d, 48); vmovups([p], a);",
            "vinsertps xmm0, xmm0, xmm1, 0x10\nvinsertps xmm0, xmm0, xmm2, 0x20\nvinsertps xmm0, xmm0, xmm3, \
             0x30\nvmovups xmmword ptr [rdi], xmm0\nret",
        ),
        (
            "a: f64, b: f64, p: *mut f64",
            "",
            "vmovlhps(a, a, b); vmovupd([p], a);",
            "vmovlhps xmm0, xmm0, xmm1\nvmovupd xmmword ptr [rdi], xmm0\nret",
        ),
        (
            "x: f32, p: *mut f64",
            "",
            "vcvtss2sd(x, x, x); vmovsd([p], x);",
            "vcvtss2sd xmm0, xmm0, xmm0\nvmovsd qword ptr [rdi], xmm0\nret",
        ),
    ];
    for (params, ret, body, want) in cases {
        assert_eq!(calling(params, ret, body).as_deref(), Ok(want), "{body}");
    }
    // What the instruction takes as it is: `vcvtss2sd` writes 64 bits, which the store reads; a scalar instruction
    // whose first source is not read, `vsqrtsd(y, x, y)` and `vroundsd(y, x, y, 0)`, as `vmovsd(y, x, y)` is; the
    // low lane, by a number.
    for (body, want) in [
        ("vsqrtsd(y, x, y);\nvmovsd([p], y);", "vsqrtsd xmm1, xmm0, xmm1\nvmovsd qword ptr [rdi], xmm1\nret"),
        (
            "vroundsd(y, x, y, 0);\nvmovsd([p], y);",
            "vroundsd xmm1, xmm0, xmm1, 0x00\nvmovsd qword ptr [rdi], xmm1\nret",
        ),
        ("vmovsd(y, x, y);\nvmovsd([p], y);", "vmovsd xmm1, xmm0, xmm1\nvmovsd qword ptr [rdi], xmm1\nret"),
        ("vextractps([p], x, 0);", "vextractps dword ptr [rdi], xmm0, 0\nret"),
        (
            "let w: xmm;\nvpshufd(w, x, 0);\nvmovss([p], w);",
            "vpshufd xmm0, xmm0, 0x00\nvmovss dword ptr [rdi], xmm0\nret",
        ),
    ] {
        assert_eq!(compile("x: f32, y: f64, p: *mut f64", body).as_deref(), Ok(want), "{body}");
    }
    // The bits that `vxorps(x, x, x)` carries are the caller's still, and `zero(x)` writes them.
    assert_eq!(compile("x: f32, p: *mut f32", "zero(x);\nvxorps(x, x, x);\nvmovups([p], x);").map(|_| ()), Ok(()));
}

/// A call reads what it passes as far as its callee takes it, and gives back what its callee gives: 32 bits of an
/// `f32` and 64 of an `f64`, of a function of the module; of one at an address with no functions given, what the call
/// names, a float the kernel took as its type, an xmm as 128 bits and `.ymm` as all 256. The rest of the register of a
/// result is the callee's, which the kernel may not read, unless the call names an xmm: the move of the result to its
/// register clears the rest of its ymm.
#[test]
fn a_call_reads_and_gives_back_no_more_than_the_float_its_callee_takes() {
    let cases = [
        // An `f32` passed to a function that takes an `f64`, and all of a ymm passed at an address, which no callee
        // takes of a float.
        ("x: f32, n: usize", "let r: xmm;\ncall(r, scale, (x, n));", "3:1", ("call", 64, 32, "x")),
        ("x: f32, fp: u64", "call(fp, (x.ymm));", "2:1", ("call", 256, 32, "x")),
        // The result of a function that gives back an `f32` and of one that gives back an `f64`, and of a call at an
        // address that names the float the kernel took: all of its xmm is read after.
        ("x: f32, y: f32, p: *mut f32", "call(x, half, (y));\nvmovups([p], x);", "3:1", ("vmovups", 128, 32, "x")),
        (
            "x: f64, n: usize, p: *mut f64",
            "call(x, scale, (x, n));\nvmovupd([p], x);",
            "3:1",
            ("vmovupd", 128, 64, "x"),
        ),
        ("x: f32, fp: u64, p: *mut f32", "call(x, fp, (x));\nvmovups([p], x);", "3:1", ("vmovups", 128, 32, "x")),
        // A variable made with `let` is held as a float is: it has what the call gave it.
        (
            "x: f32, y: f32, p: *mut f32",
            "let z: xmm;\ncall(z, half, (y));\nvmovups([p], z);",
            "4:1",
            ("vmovups", 128, 32, "z"),
        ),
    ];
    for (params, body, at, (name, reads, has, x)) in cases {
        let want = format!("{at}: {}", past_the_float(name, reads, has, x));
        assert_eq!(calling(params, "", body), Err(want), "{body}");
    }
    // What a call may be given and may give back: the float it takes, as the kernel took it; an xmm at an address; a
    // ymm where the call names one, and an xmm result, which the move after the call fills to all of its ymm.
    let cases = [
        (
            "x: f32, y: f32, p: *mut f32",
            "call(x, half, (y));\nvmovss([p], x);",
            "sub rsp, 8\nvmovaps xmm0, xmm1\ncall {f0}\nvmovss dword ptr [rdi], xmm0\nadd rsp, 8\nret",
        ),
        ("x: f32, fp: u64", "call(fp, (x));", "jmp rdi"),
        ("x: f64, fp: u64", "call(fp, (x.xmm));", "jmp rdi"),
        (
            "x: *mut f32, fp: u64",
            "let v: ymm;\ncall(v.xmm, fp, ());\nvmovups([x], v);",
            "push rbx\nmov rbx, rdi\ncall rsi\nvmovaps xmm0, xmm0\nvmovups ymmword ptr [rbx], ymm0\npop rbx\n\
             vzeroupper\nret",
        ),
        (
            "x: *mut f32, fp: u64",
            "let v: ymm;\ncall(v, fp, ());\nvmovups([x], v);",
            "push rbx\nmov rbx, rdi\ncall rsi\nvmovups ymmword ptr [rbx], ymm0\npop rbx\nvzeroupper\nret",
        ),
    ];
    for (params, body, want) in cases {
        assert_eq!(calling(params, "", body).as_deref(), Ok(want), "{body}");
    }
}

/// How much of its register each instruction that reads a float as data reads, as Intel's manual says: an `f32` is 32
/// bits of its register, so an instruction that reads 64 is an error at the float, and one that reads 32 or fewer is
/// not; an `f64` is 64, and none of 64 or fewer is. Where the instruction picks what it reads by a number, it reads the
/// lanes the number names and no more.
#[test]
fn an_instruction_reads_as_much_of_a_float_as_intels_manual_says() {
    // The bits each instruction reads of `fl`; `y` and `w` are `f64`s and `z` an xmm, written first so that an
    // instruction may read all of them, `n` an integer, and `p` memory.
    let cases = [
        (64, "vmovq(n, fl);"),
        (64, "vmovd(n, fl);"),
        (64, "vmovsd([p], fl);"),
        (64, "vmovlps([p], fl);"),
        (64, "vaddsd(y, fl, y);"),
        (64, "vsqrtsd(y, y, fl);"),
        (64, "vucomisd(fl, y);"),
        (64, "vcomisd(y, fl);"),
        (64, "vcmpsd(y, y, fl, 0);"),
        (64, "vcmpsd(y, fl, y, 0);"),
        (64, "vcmpeqsd(y, y, fl);"),
        (64, "vcmpeqsd(y, fl, y);"),
        (64, "vroundsd(y, y, fl, 0);"),
        (64, "vcvtsd2ss(y, y, fl);"),
        (64, "vcvtsd2si(n, fl);"),
        (64, "vcvtdq2pd(y, fl);"),
        (64, "vcvtps2pd(y, fl);"),
        (64, "vcvtph2ps(y, fl);"),
        (64, "vpmovsxbw(y, fl);"),
        (64, "vpmovsxbd(w.ymm, fl);"),
        (64, "vpmovzxdq(y, fl);"),
        (64, "vmovddup(y, fl);"),
        (64, "vbroadcastsd(w.ymm, fl);"),
        (64, "vpbroadcastq(w.ymm, fl);"),
        (64, "vfmadd231sd(y, y, fl);"),
        (64, "vfmadd231sd(y, fl, y);"),
        (64, "vfmadd231sd(fl, y, y);"),
        (32, "vmovd(n.r32, fl);"),
        (32, "vmovss([p], fl);"),
        (32, "vaddss(y, y, fl);"),
        (32, "vcvtss2sd(y, y, fl);"),
        (32, "vcvtss2si(n, fl);"),
        (32, "vucomiss(fl, y);"),
        (32, "vbroadcastss(w.ymm, fl);"),
        (32, "vpbroadcastd(w.ymm, fl);"),
        (32, "vpmovsxbd(y, fl);"),
        (32, "vpmovsxbq(w.ymm, fl);"),
        (32, "vfmadd231ss(y, y, fl);"),
        (32, "vroundss(y, y, fl, 0);"),
        (32, "vcmpss(y, y, fl, 0);"),
        (32, "vcmpeqss(y, y, fl);"),
        (32, "vcmpeqss(y, fl, y);"),
        (16, "vpbroadcastw(w.ymm, fl);"),
        (16, "vpmovsxbq(y, fl);"),
        (8, "vpbroadcastb(w.ymm, fl);"),
        // An instruction that unpacks the low halves of two sources reads 64 bits of each, and of each half of a ymm,
        // and a shift 64 of its count; one that unpacks the high halves reads all of its first source, and of its
        // second all but the last element, which stays in place; and a shift reads all of what it shifts.
        (64, "vunpcklps(w, y, fl);"),
        (64, "vunpcklps(w, fl, y);"),
        (192, "vunpcklps(w.ymm, fl.ymm, y.ymm);"),
        (64, "vmovlhps(w, y, fl);"),
        (64, "vunpcklpd(w, y, fl);"),
        (192, "vunpcklpd(w.ymm, y.ymm, fl.ymm);"),
        (64, "vpunpcklbw(w, fl, y);"),
        (192, "vpunpcklbw(w.ymm, y.ymm, fl.ymm);"),
        (64, "vpunpcklqdq(w, y, fl);"),
        (64, "vpsllw(w, y, fl);"),
        (64, "vpsrad(w, y, fl);"),
        (96, "vunpckhps(w, y, fl);"),
        (128, "vunpckhps(w, fl, y);"),
        (224, "vunpckhps(w.ymm, y.ymm, fl.ymm);"),
        (128, "vmovhlps(w, y, fl);"),
        (128, "vpunpckhbw(w, fl, y);"),
        (120, "vpunpckhbw(w, y, fl);"),
        (112, "vpunpckhwd(w, y, fl);"),
        (96, "vpunpckhdq(w, y, fl);"),
        (128, "vpsllq(w, fl, y);"),
        // What takes part of each half reads as far as that part: the low quadword of `vmovddup`, the even lanes of
        // `vmovsldup`, `vpmuludq` and `vpmuldq`.
        (192, "vmovddup(w.ymm, fl.ymm);"),
        (96, "vmovsldup(w, fl);"),
        (224, "vmovsldup(w.ymm, fl.ymm);"),
        (96, "vpmuludq(w, y, fl);"),
        (96, "vpmuldq(w, fl, y);"),
        // What a gpr, the flags, and a store of the whole register take.
        (128, "vmovmskps(n.r32, fl);"),
        (128, "vptest(fl, z);"),
        (128, "vmovups([p], fl);"),
        (256, "vmovups([p], fl.ymm);"),
        // An instruction that picks what it reads by a number reads the lanes it names, and the highest of them counts:
        // `vextractps` and `vpextrd` the lane; `vpextrb` the byte and `vpextrw` the word; `vpshufd` and `vpermilps` the
        // lanes of their fields, `vshufps` of its two, `vinsertps` its source lane, none where it zeroes the lane it
        // writes, and `vpermilpd` and `vshufpd` the halves; `vpalignr` and the shifts by bytes what they shift in; a
        // dot product the lanes it multiplies, and `vmpsadbw` the bytes it sums. Of a ymm, the high half counts, which
        // picks by the same fields or by its own, and `vextractf128`, `vpermq` and `vperm2f128` pick from all of it.
        (32, "vextractps(n.r32, fl, 0);"),
        (64, "vextractps(n.r32, fl, 1);"),
        (128, "vextractps(n.r32, fl, 3);"),
        (32, "vpextrd(n.r32, fl, 0);"),
        (64, "vpextrd(n.r32, fl, 1);"),
        (64, "vpextrq(n, fl, 0);"),
        (128, "vpextrq(n, fl, 1);"),
        (8, "vpextrb(n.r32, fl, 0);"),
        (32, "vpextrb(n.r32, fl, 3);"),
        (48, "vpextrw(n.r32, fl, 2);"),
        (32, "vpshufd(w, fl, 0);"),
        (64, "vpshufd(w, fl, 1);"),
        (64, "vpshufd(w, fl, 0x55);"),
        (128, "vpshufd(w, fl, 0xe4);"),
        (32, "vpermilps(w, fl, 0);"),
        (128, "vpermilps(w, fl, 0xff);"),
        (64, "vpermilpd(w, fl, 0);"),
        (128, "vpermilpd(w, fl, 2);"),
        (32, "vshufps(w, fl, fl, 0);"),
        (64, "vshufps(w, fl, y, 4);"),
        (128, "vshufps(w, y, fl, 0xc0);"),
        (64, "vshufpd(w, fl, fl, 0);"),
        (128, "vshufpd(w, fl, y, 1);"),
        (128, "vshufpd(w, y, fl, 2);"),
        (32, "vinsertps(w, z, fl, 0);"),
        (64, "vinsertps(w, z, fl, 0x40);"),
        (128, "vinsertps(w, z, fl, 0xc0);"),
        (0, "vinsertps(w, z, fl, 0x41);"),
        (16, "vpshuflw(w, fl, 0);"),
        (64, "vpshuflw(w, fl, 0xe4);"),
        (80, "vpshufhw(w, fl, 0);"),
        (128, "vpshufhw(w, fl, 0xe4);"),
        (128, "vpalignr(w, z, fl, 4);"),
        (32, "vpalignr(w, fl, z, 4);"),
        (0, "vpalignr(w, fl, z, 32);"),
        (0, "vpslldq(w, fl, 16);"),
        (32, "vpslldq(w, fl, 12);"),
        (96, "vpslldq(w, fl, 4);"),
        (128, "vpsrldq(w, fl, 4);"),
        (0, "vpsrldq(w, fl, 16);"),
        (32, "vdpps(w, fl, y, 0x11);"),
        (64, "vdpps(w, y, fl, 0x21);"),
        (128, "vdpps(w, fl, y, 0xf1);"),
        (64, "vdppd(w, fl, y, 0x11);"),
        (128, "vdppd(w, y, fl, 0x21);"),
        (32, "vmpsadbw(w, y, fl, 0);"),
        (64, "vmpsadbw(w, y, fl, 1);"),
        (88, "vmpsadbw(w, fl, y, 0);"),
        (120, "vmpsadbw(w, fl, y, 4);"),
        (160, "vpshufd(w.ymm, fl.ymm, 0);"),
        (192, "vpermilpd(w.ymm, fl.ymm, 0);"),
        (0, "vdpps(w.ymm, fl.ymm, y.ymm, 0x0f);"),
        (160, "vdpps(w.ymm, fl.ymm, y.ymm, 0x11);"),
        (192, "vmpsadbw(w.ymm, y.ymm, fl.ymm, 8);"),
        (248, "vmpsadbw(w.ymm, fl.ymm, y.ymm, 0x20);"),
        (128, "vextractf128(z, fl.ymm, 0);"),
        (256, "vextractf128(z, fl.ymm, 1);"),
        (64, "vpermq(w.ymm, fl.ymm, 0);"),
        (256, "vpermq(w.ymm, fl.ymm, 0xc0);"),
        (128, "vperm2f128(w.ymm, fl.ymm, y.ymm, 9);"),
        (256, "vperm2f128(w.ymm, fl.ymm, y.ymm, 1);"),
    ];
    for (reads, body) in cases {
        let name = body.split_once('(').map_or(body, |(name, _)| name);
        for has in [32, 64] {
            let params = format!("fl: f{has}, y: f64, w: f64, n: usize, p: *mut f32");
            let got = compile(&params, &format!("let z: xmm; zero(y); zero(w); zero(z);\n{body}")).map(|_| ());
            let want =
                if reads <= has { Ok(()) } else { Err(format!("3:1: {}", past_the_float(name, reads, has, "fl"))) };
            assert_eq!(got, want, "an f{has} in {body}");
        }
    }
}

/// How much of its result each instruction that carries a float gives the kernel, as Intel's manual says: the bits of
/// the float that it moves to the same place of its result are not read, and are the kernel's in the result where they
/// are the float's, so the result has as many low bits of its own as the least of what it carries and writes allow.
/// `fl` is the float, 32 bits or 64; `y` is an `f64` and `z` a vector, both written first, `n` an integer, and `p`
/// memory. Each case gives how many of the low bits of `w`, which the instruction writes, are the kernel's after it,
/// for an `f32` and for an `f64`: all of an xmm is 128 and of a ymm 256, which is what the store of `w` after it reads.
/// A selector that picks lanes of 64 bits counts in whole lanes, so an `f32` of 32 bits makes none of them its own.
#[test]
fn an_instruction_carries_a_float_as_far_as_intels_manual_says() {
    let xmm = [
        // Moves of a whole register and the logic of bits carry all they take, from whichever source it is.
        ((32, 64), "vmovaps(w, fl);"),
        ((32, 64), "vmovups(w, fl);"),
        ((32, 64), "vmovdqa(w, fl);"),
        ((32, 64), "vandps(w, fl, z);"),
        ((32, 64), "vandpd(w, z, fl);"),
        ((32, 64), "vxorps(w, fl, fl);"),
        ((32, 64), "vpor(w, z, fl);"),
        ((32, 64), "vpandn(w, fl, z);"),
        // `vmovq` of an xmm carries its low 64 bits and writes the rest; `vmovss` and `vmovsd` of registers carry the
        // low bits of the third operand, and the rest of the second.
        ((32, 128), "vmovq(w, fl);"),
        ((32, 128), "vmovsd(w, z, fl);"),
        ((128, 128), "vmovss(w, z, fl);"),
        ((64, 64), "vmovsd(w, fl, y);"),
        ((32, 64), "vmovss(w, fl, y);"),
        // A blend by a number takes each lane from one source, and carries it; one by a register carries both sources,
        // and its selector, in whole lanes of its own size.
        ((128, 128), "vblendps(w, z, fl, 1);"),
        ((32, 128), "vblendps(w, z, fl, 3);"),
        ((32, 64), "vblendps(w, z, fl, 7);"),
        ((32, 64), "vblendps(w, fl, z, 0);"),
        ((128, 128), "vblendps(w, fl, z, 14);"),
        ((32, 128), "vblendpd(w, z, fl, 1);"),
        ((128, 128), "vpblendd(w, z, fl, 1);"),
        ((128, 128), "vpblendw(w, z, fl, 1);"),
        ((32, 64), "vblendvps(w, fl, z, z);"),
        ((32, 64), "vblendvpd(w, z, fl, z);"),
        ((32, 64), "vpblendvb(w, z, z, fl);"),
        ((32, 64), "vblendvps(w, z, z, fl);"),
        ((0, 64), "vblendvpd(w, z, z, fl);"),
        // What a merge keeps: the rest of the first source of a scalar instruction, the half of `vmovhps` and
        // `vmovlps`, the low half of the first source of the unpacks, and the lanes that an insert leaves.
        ((64, 64), "vsqrtsd(w, fl, y);"),
        ((32, 64), "vsqrtss(w, fl, y);"),
        ((64, 64), "vroundsd(w, fl, y, 0);"),
        ((32, 64), "vcvtsd2ss(w, fl, y);"),
        ((64, 64), "vcvtss2sd(w, fl, y);"),
        ((32, 64), "vcvtsi2ss(w, fl, n);"),
        ((32, 128), "vmovhps(w, fl, [p]);"),
        ((64, 64), "vmovlps(w, fl, [p]);"),
        ((32, 128), "vmovlhps(w, fl, y);"),
        ((32, 128), "vunpcklpd(w, fl, y);"),
        ((32, 128), "vpunpcklqdq(w, fl, y);"),
        ((64, 64), "vinsertps(w, fl, z, 0x10);"),
        ((128, 128), "vinsertps(w, z, fl, 0x10);"),
        ((32, 128), "vpinsrq(w, fl, n, 1);"),
        ((64, 64), "vpinsrd(w, fl, n.r32, 1);"),
        ((32, 64), "vpinsrw(w, fl, n.r32, 0);"),
        ((32, 64), "vpinsrb(w, fl, n.r32, 0);"),
        ((128, 128), "vinsertps(w, fl, z, 0x0e);"),
        ((64, 64), "vpshuflw(w, fl, 0);"),
        ((64, 64), "vmovhlps(w, fl, z);"),
        ((64, 64), "vunpckhpd(w, z, fl);"),
        ((64, 64), "vpunpckhqdq(w, z, fl);"),
        ((96, 96), "vblendps(w, z, fl, 8); vunpckhps(w, z, w);"),
    ];
    let ymm = [
        ((32, 64), "vmovaps(w, fl.ymm);"),
        ((32, 64), "vandps(w, z, fl.ymm);"),
        ((32, 64), "vinsertf128(w, fl.ymm, z.xmm, 1);"),
        ((128, 128), "vinsertf128(w, fl.ymm, z.xmm, 0);"),
        ((32, 64), "vinserti128(w, fl.ymm, z.xmm, 1);"),
        ((32, 64), "vblendps(w, z, fl.ymm, 15);"),
        ((256, 256), "vblendps(w, z, fl.ymm, 1);"),
        ((128, 128), "vpblendw(w, z, fl.ymm, 1);"),
        ((32, 128), "vunpcklpd(w, fl.ymm, z);"),
        ((192, 192), "vinsertf128(w, fl.ymm, z.xmm, 0); vunpckhpd(w, z, w);"),
        ((224, 224), "vblendps(w, z, fl.ymm, 0x80); vunpckhps(w, z, w);"),
    ];
    for (vector, whole, cases) in [("xmm", 128, &xmm[..]), ("ymm", 256, &ymm[..])] {
        for &((a32, a64), body) in cases {
            for (has, after) in [(32, a32), (64, a64)] {
                let params = format!("fl: f{has}, y: f64, n: usize, p: *mut f32");
                let src = format!("let (w, z): ({vector}, {vector}); zero(z); zero(y);\n{body}\nvmovups([p], w);");
                let want = match after {
                    after if after >= whole => Ok(()),
                    after => Err(format!("4:1: {}", past_the_float("vmovups", whole, after, "w"))),
                };
                assert_eq!(compile(&params, &src).map(|_| ()), want, "an f{has} in {body}");
            }
        }
    }
}

/// What each instruction that writes a float leaves of it, as Intel's manual says: one that writes an xmm or a ymm
/// writes all of it; a scalar one writes the float alone and carries the rest of its first source, so `x` has what that
/// source has beyond the float, or all of it where the source is a variable that the kernel wrote. Each case gives how
/// many bits of `x` the instruction reads, and how many of its low bits are the kernel's after it, for an `f32` and for
/// an `f64`: all of an xmm is 128, which the store of `x` after it reads. An `f32` of a case that reads 64 bits of it
/// is the other test's. `y` is an `f64` and `z` an xmm written first, `n` an integer, and `p` and `q` memory.
#[test]
fn an_instruction_writes_as_much_of_a_float_as_intels_manual_says() {
    let cases = [
        // What writes all of an xmm: a load, a move from a gpr, a conversion, a broadcast, and what takes all of its
        // sources from variables the kernel wrote.
        (0, (128, 128), "vmovss(x, [q]);"),
        (0, (128, 128), "vmovsd(x, [q]);"),
        (0, (128, 128), "vmovd(x, n.r32);"),
        (0, (128, 128), "vmovq(x, n);"),
        (0, (128, 128), "vmovq(x, y);"),
        (0, (128, 128), "vcvtdq2pd(x, y);"),
        (0, (128, 128), "vcvtps2pd(x, y);"),
        (0, (128, 128), "vcvtph2ps(x, y);"),
        (0, (128, 128), "vcvtpd2ps(x, z);"),
        (0, (128, 128), "vpmovsxbw(x, y);"),
        (0, (128, 128), "vmovddup(x, y);"),
        (0, (128, 128), "vbroadcastss(x.ymm, y);"),
        (0, (128, 128), "vpbroadcastb(x, y);"),
        (0, (128, 128), "vmovaps(x, z);"),
        (0, (128, 128), "vsqrtps(x, z);"),
        (0, (128, 128), "vaddps(x, z, z);"),
        (0, (128, 128), "vpsllw(x, z, y);"),
        (0, (128, 128), "vpslldq(x, z, 3);"),
        (0, (128, 128), "vpinsrd(x, z, n.r32, 0);"),
        (0, (128, 128), "vinsertps(x, z, z, 0);"),
        (0, (128, 128), "vblendps(x, z, z, 1);"),
        (0, (128, 128), "vandps(x, z, z);"),
        // The low halves that an instruction unpacks are all it reads of the floats it takes.
        (0, (128, 128), "vunpcklps(x, y, y);"),
        (0, (128, 128), "vunpcklpd(x, y, y);"),
        (0, (128, 128), "vmovlhps(x, y, y);"),
        (0, (128, 128), "vpunpcklbw(x, y, y);"),
        (0, (128, 128), "vpunpcklqdq(x, y, y);"),
        (0, (128, 128), "vmovhps(x, y, [q]);"),
        (0, (128, 128), "vmovlps(x, z, [q]);"),
        (0, (128, 128), "vmovhlps(x, z, z);"),
        // A scalar instruction carries the rest of its first source: all of `x` where that is `z`, which the kernel
        // wrote.
        (0, (128, 128), "vaddss(x, z, y);"),
        (0, (128, 128), "vsqrtsd(x, z, y);"),
        (0, (128, 128), "vroundsd(x, z, y, 0);"),
        (0, (128, 128), "vcvtss2sd(x, z, y);"),
        (0, (128, 128), "vmovss(x, z, y);"),
        // And no more than `x` has where that is `x`: the float it writes, 32 bits or 64, and the rest that it had.
        (32, (32, 64), "vaddss(x, x, y);"),
        (64, (0, 64), "vaddsd(x, x, y);"),
        (0, (32, 64), "vsqrtss(x, x, y);"),
        (0, (64, 64), "vsqrtsd(x, x, y);"),
        (0, (32, 64), "vrcpss(x, x, y);"),
        (0, (32, 64), "vrsqrtss(x, x, y);"),
        (32, (32, 64), "vcmpss(x, x, y, 0);"),
        (32, (32, 64), "vcmpeqss(x, x, y);"),
        (64, (0, 64), "vcmpsd(x, x, y, 0);"),
        (64, (0, 64), "vcmpeqsd(x, x, y);"),
        (0, (32, 64), "vroundss(x, x, y, 0);"),
        (0, (64, 64), "vroundsd(x, x, y, 0);"),
        (0, (64, 64), "vcvtss2sd(x, x, y);"),
        (0, (32, 64), "vcvtsd2ss(x, x, y);"),
        (0, (32, 64), "vcvtsi2ss(x, x, n);"),
        (0, (64, 64), "vcvtsi2sd(x, x, n);"),
        (0, (32, 64), "vmovss(x, x, y);"),
        (0, (64, 64), "vmovsd(x, x, y);"),
        (32, (32, 64), "vfmadd231ss(x, y, y);"),
        (64, (0, 64), "vfmadd231sd(x, y, y);"),
    ];
    for (reads, (a32, a64), body) in cases {
        for (has, after) in [(32, a32), (64, a64)].into_iter().filter(|&(has, _)| has >= reads) {
            let params = format!("x: f{has}, y: f64, n: usize, p: *mut f32, q: *const f32");
            let src = format!("let z: xmm; zero(z);\n{body}\nvmovups([p], x);");
            let want = match after {
                after if after >= 128 => Ok(()),
                after => Err(format!("4:1: {}", past_the_float("vmovups", 128, after, "x"))),
            };
            assert_eq!(compile(&params, &src).map(|_| ()), want, "an f{has} in {body}");
        }
    }
}

/// An operand of a form that reads the low bits of an xmm as data and no more, `v32` and `v64` among them, takes a
/// float parameter of that many bits or more as it is, and refuses one of fewer at the instruction, saying how much it
/// reads. The instructions that carry such an operand, or pick what they read of it by a number, do not read it as
/// data: another test holds them. `call` writes the other operands of the form.
#[test]
fn each_form_that_reads_the_low_bits_of_an_xmm_takes_a_float_of_so_many() -> Result {
    let carries = ["vmovq", "vmovss", "vmovsd", "vmovhps", "vmovhpd", "vunpcklpd", "vpunpcklqdq", "vmovlhps"];
    let mut cases = 0;
    for form in forms::names().filter(|n| !carries.contains(n)).flat_map(|n| forms::of(n).unwrap_or_default()) {
        let name = form.op.name();
        for (i, spec) in form.operands.iter().enumerate() {
            let (Kind::Vec(w), Access::Read | Access::ReadWrite) = (spec.kind, spec.access) else { continue };
            let reads = w.bytes() * 8;
            if reads >= 128 {
                continue;
            }
            let call = call(name, form, false, Some(i))?;
            // The instruction is a line of its own, in column 1.
            let row = call.lines().position(|l| l.starts_with(&format!("{name}("))).map_or(0, |r| r + 2);
            for has in [32, 64] {
                let src = format!("unsafe extern \"sysv64\" fn k(f: f{has}, x: *mut f32) {{\n{call}\n}}\n");
                let got = kernite::x86_str(&src, "k").map(|_| ());
                let want = if reads <= has {
                    Ok(())
                } else {
                    Err(format!("{row}:1: {}", past_the_float(name, reads, has, "f")))
                };
                assert_eq!(got, want, "an f{has} at operand {i} of\n{call}");
                cases += 1;
            }
        }
    }
    // Every form that reads part of an xmm, for each float: neither a form nor an operand is left out.
    assert!(cases >= 850, "{cases} cases");
    Ok(())
}

/// An operand of a form that writes an xmm or a ymm takes a float parameter, which the kernel reads all of after. A
/// form that writes the low bits of an xmm, `=v32` and `=v64`, passes the rest of its first source on, which `call`
/// writes first, so the kernel reads all of the parameter then too; one that reads and writes them, `+v32`, keeps what
/// the parameter has, and a read of all of it after is an error at the read, where the instruction itself is not. A
/// gather keeps the lanes of its result that its mask leaves out, and reads none of them, so its result too keeps what
/// the parameter has.
#[test]
fn each_form_that_writes_a_vector_writes_a_float_whole_or_alone() -> Result {
    let mut cases = 0;
    for form in forms::names().flat_map(|n| forms::of(n).unwrap_or_default()) {
        let name = form.op.name();
        for (i, spec) in form.operands.iter().enumerate() {
            let (Kind::Vec(w), Access::Write | Access::ReadWrite) = (spec.kind, spec.access) else { continue };
            let bits = w.bytes() * 8;
            let call = call(name, form, false, Some(i))?;
            let at = call.lines().position(|l| l.starts_with(&format!("{name}("))).map_or(0, |r| r + 2);
            // The kernel's last line, after the call and the stores of its other operands.
            let last = call.lines().count() + 2;
            let gather = i == 0 && form.operands.iter().any(|s| matches!(s.kind, Kind::Vsib(..)));
            for has in [32, 64] {
                let src =
                    format!("unsafe extern \"sysv64\" fn k(f: f{has}, x: *mut f32) {{\n{call}\nvmovups([x], f);\n}}\n");
                let got = kernite::x86_str(&src, "k").map(|_| ());
                let want = match (spec.access, bits > has && !gather) {
                    (Access::Write, _) => Ok(()),
                    (_, true) => Err(format!("{at}:1: {}", past_the_float(name, bits, has, "f"))),
                    (_, false) => Err(format!("{last}:1: {}", past_the_float("vmovups", 128, has, "f"))),
                };
                assert_eq!(got, want, "an f{has} at operand {i} of\n{call}");
                cases += 1;
            }
        }
    }
    // Every form that writes a vector, for each float: neither a form nor an operand is left out.
    assert!(cases >= 2900, "{cases} cases");
    Ok(())
}

/// A pointer is one register, whatever it points to: an array, a pointer, a tuple, a generic type.
/// A pointer to what has no size, a slice, a `str` or a `dyn`, in parentheses or not, or a tuple that
/// ends in one, is two words in Rust, and an error.
#[test]
fn a_kernel_takes_a_pointer_to_any_type_in_a_register() -> Result {
    let params = "shape: *const [usize; 3], out: *mut *const f32, pair: *const (u32, u32), \
                  cell: *const Cell<u8, 2>, n: usize";
    let code = compile(params, "mov(n, [shape + 8]); mov([out], pair); mov([cell], n);")?;
    assert_eq!(code, "mov r8, qword ptr [rdi + 8]\nmov qword ptr [rsi], rdx\nmov qword ptr [rcx], r8\nret");
    let src = "unsafe extern \"sysv64\" fn k(p: *const [usize; 3]) -> *mut [usize; 3] { ret(p); }";
    assert_eq!(kernite::x86_str(src, "k")?.code().collect::<Vec<_>>(), ["mov rax, rdi", "ret"]);
    for to in ["()", "(u32,)", "(u32, u32)"] {
        assert_eq!(compile(&format!("p: *const {to}"), "")?, "ret", "{to}");
    }
    for to in [
        "[f32]",
        "[[u8; 4]]",
        "str",
        "dyn Foo",
        "(dyn A + Send)",
        "([f32])",
        "(str)",
        "((dyn Foo))",
        "(dyn Foo<A, B>)",
        "(u32, [f32])",
    ] {
        let why = format!(
            "a pointer to `{to}` has two words, an address and a length or a table, and a kernel's pointer has one"
        );
        assert_eq!(
            compile(&format!("p: *const {to}"), "").err().ok_or("compile gave Ok")?,
            format!("1:32: {why}"),
            "{to}"
        );
        let src = format!("unsafe extern \"sysv64\" fn k() -> *mut {to} {{}}");
        assert_eq!(kernite::x86_str(&src, "k").err().ok_or("x86_str gave Ok")?, format!("1:34: {why}"), "-> {to}");
    }
    Ok(())
}

/// A type may hold an arrow, `->`, one operator whose `>` ends no type and closes no `<`: a pointer to
/// a function's pointer, or to a type with one among its arguments, is one register as any pointer is,
/// and the parameter after it is read from the next.
#[test]
fn a_kernel_takes_a_pointer_to_a_type_that_holds_an_arrow() -> Result {
    for params in [
        "call: *const fn(u32) -> u32, table: *mut Pair<fn() -> u8, 2>, \
         boxed: *const Box<dyn Fn(u8) -> u8 + Send>, n: usize",
        "call: *const fn(u32)->u32, table: *mut Pair<fn()->&u8,2>, boxed: *const Box<dyn Fn(u8)->u8+Send>,n: usize",
    ] {
        let code = compile(params, "mov([call], n); mov([table], n); mov([boxed], n);")?;
        assert_eq!(code, "mov qword ptr [rdi], rcx\nmov qword ptr [rsi], rcx\nmov qword ptr [rdx], rcx\nret");
    }
    let src = "unsafe extern \"sysv64\" fn k(p: *const fn(u32) -> u32) -> *mut fn(u32) -> u32 { ret(p); }";
    assert_eq!(kernite::x86_str(src, "k")?.code().collect::<Vec<_>>(), ["mov rax, rdi", "ret"]);
    let to = "dyn Fn(u8) -> u8";
    let why = format!(
        "a pointer to `{to}` has two words, an address and a length or a table, and a kernel's pointer has one"
    );
    assert_eq!(
        compile("p: *const dyn Fn(u8)->u8, n: usize", "").err().ok_or("compile gave Ok")?,
        format!("1:32: {why}")
    );
    Ok(())
}

/// A type in a message is written as Rust writes it, whatever the source spaced: a parameter by
/// value, a result, a variable. A block and a macro's arguments are written as their source has them, packed
/// or spaced, as the messages of rustc write them. So is the length of an array, but where kernite takes the type
/// of a parameter or a variable apart, and works the length out.
#[test]
fn a_type_in_a_message_is_written_as_rust_writes_it() -> Result {
    let takes = "a kernel takes pointers, 64-bit integers and floats, not";
    for (params, ty) in [
        ("a: [usize;3]", "[usize; 3]"),
        ("a: (u32 ,u32)", "(u32, u32)"),
        ("a: &mut[f32]", "&mut [f32]"),
        ("a: Box<dyn A+B>", "Box<dyn A + B>"),
        ("a: Box<dyn Iterator<Item=u8>>", "Box<dyn Iterator<Item = u8>>"),
        ("a: Box<dyn Iterator<Item = u8> + Send>", "Box<dyn Iterator<Item = u8> + Send>"),
        ("f: fn(u32) -> u32", "fn(u32) -> u32"),
        ("f: fn(u32)->u32, n: usize", "fn(u32) -> u32"),
        ("a: fn(fn()->u8)->&u8", "fn(fn() -> u8) -> &u8"),
        ("a: fn()->!", "fn() -> !"),
        ("a: Pair<fn()->u8,2>", "Pair<fn() -> u8, 2>"),
        ("a: Box<dyn Fn(u8)->u8+Send>", "Box<dyn Fn(u8) -> u8 + Send>"),
        ("a: fn(f16:u8)", "fn(f16: u8)"),
        ("a: fn(f16: ::core::primitive::u8)", "fn(f16: u8)"),
        ("a: fn(f16: ::my::Word)", "fn(f16: ::my::Word)"),
        ("a: Box<dyn Iterator<Item:Copy>>", "Box<dyn Iterator<Item: Copy>>"),
        ("a: Box<dyn Tr<Item<X>=u8>>", "Box<dyn Tr<Item<X> = u8>>"),
        ("a: Box<dyn Tr<Item<X> =u8>>", "Box<dyn Tr<Item<X> = u8>>"),
        ("a: Box<dyn Tr<Item<X>= u8>>", "Box<dyn Tr<Item<X> = u8>>"),
        ("a: Box<dyn Tr<Item<X> = u8>>", "Box<dyn Tr<Item<X> = u8>>"),
        ("a: Box<dyn Tr<Item<Vec<X>>=u8,Other<Y,Z>=&u8>>", "Box<dyn Tr<Item<Vec<X>> = u8, Other<Y, Z> = &u8>>"),
        ("a: Marker<{!0usize&3}>", "Marker<{!0usize&3}>"),
        ("a: Marker<{ !0usize & 3 }>", "Marker<{ !0usize & 3 }>"),
        ("a: Marker<{\n    N * // twice\n    2\n}>", "Marker<{ N * 2 }>"),
        ("a: m!(a+b)", "m!(a+b)"),
        ("a: m!(a + b)", "m!(a + b)"),
        ("a: m!{a&b}", "m! {a&b}"),
        ("a: Pair<u8,{3}>", "Pair<u8, {3}>"),
        ("a: Pair<u8,{ 3 }>", "Pair<u8, { 3 }>"),
        ("a: Marker<{}>", "Marker<{}>"),
        ("a: Marker<{ }>", "Marker<{ }>"),
    ] {
        assert_eq!(compile(params, "").err().ok_or("compile gave Ok")?, format!("1:32: {takes} `{ty}`"), "{params}");
    }
    // The rows of the review, each packed and spaced, in a block and as a macro's arguments.
    for (ty, quoted) in quotes::quoted() {
        assert_eq!(
            compile(&format!("a: {ty}"), "").err().ok_or("compile gave Ok")?,
            format!("1:32: {takes} `{quoted}`"),
            "{ty}"
        );
    }
    // The length of an array is worked out in the type of a parameter that is an array, a reference to one or a name
    // with arguments, and in those of its elements and arguments that are, `[usize; 6]`; its names are the module's.
    // In any other type, a tuple, a reference to another type or a path, it is written as its source has it. A
    // reference to an array of numbers is one a kernel takes, so it is no row here: `ptx.rs` has it.
    for (ty, quoted) in [
        ("(u8,[usize;N*2])", "(u8, [usize; N*2])"),
        ("(u8,[usize; N * 2])", "(u8, [usize; N * 2])"),
        ("(u8,[usize ;(N+1)/2 ])", "(u8, [usize; (N+1)/2])"),
        ("(u8,[usize;!0usize&N])", "(u8, [usize; !0usize&N])"),
        ("(u8,[usize;\n    N * // twice\n    2\n])", "(u8, [usize; N * 2])"),
        ("&(u8,[usize;N*2])", "&(u8, [usize; N*2])"),
        ("&mut (u8,[usize;N * 2])", "&mut (u8, [usize; N * 2])"),
        ("a::Wrap<[usize;N*2]>", "a::Wrap<[usize; N*2]>"),
        ("Wrap<(u8,[usize;N*2])>", "Wrap<(u8, [usize; N*2])>"),
        ("[(u8,[usize;N*2]);2]", "[(u8, [usize; N*2]); 2]"),
        ("[usize;N*2]", "[usize; 6]"),
        ("Wrap<[usize;N*2]>", "Wrap<[usize; 6]>"),
        ("[[usize;N*2];2]", "[[usize; 6]; 2]"),
        ("Wrap<Wrap<[usize;N*2]>>", "Wrap<Wrap<[usize; 6]>>"),
    ] {
        let src = format!("const N: usize = 3;\nunsafe extern \"sysv64\" fn k(a: {ty}) {{}}\n");
        assert_eq!(
            kernite::x86_str(&src, "k").err().ok_or("x86_str gave Ok")?,
            format!("2:32: {takes} `{quoted}`"),
            "{ty}"
        );
    }
    let two = "has two words, an address and a length or a table, and a kernel's pointer has one";
    assert_eq!(
        compile("p: *const dyn A+Send", "").err().ok_or("compile gave Ok")?,
        format!("1:32: a pointer to `dyn A + Send` {two}")
    );
    for (to, quoted) in [
        ("dyn Tr<N={3}>", "dyn Tr<N = {3}>"),
        ("dyn Tr<N={ 3 }>", "dyn Tr<N = { 3 }>"),
        ("dyn m!(a+b)", "dyn m!(a+b)"),
        ("dyn Tr<Item<X>=u8>", "dyn Tr<Item<X> = u8>"),
        ("dyn Tr<Item<X> =u8>", "dyn Tr<Item<X> = u8>"),
    ] {
        let why = format!("1:32: a pointer to `{quoted}` {two}");
        assert_eq!(compile(&format!("p: *const {to}"), "").err().ok_or("compile gave Ok")?, why, "{to}");
    }
    let src = "#[instances(2)]\nunsafe extern \"sysv64\" fn k<const N: usize>(a: [usize; N]) {}";
    assert_eq!(
        kernite::x86_str(src, "k::<2>").err().ok_or("x86_str gave Ok")?,
        format!("2:48: in `k::<2>`: {takes} `[usize; 2]`")
    );
    let src = "unsafe extern \"sysv64\" fn k() -> (u32,u32) {}";
    let returns = "a kernel returns a float, an integer or a pointer, or never, `!`; not `(u32, u32)`";
    assert_eq!(kernite::x86_str(src, "k").err().ok_or("x86_str gave Ok")?, format!("1:34: {returns}"));
    let src = "unsafe extern \"sysv64\" fn k() -> fn(u32)->u32 {}";
    let returns = "a kernel returns a float, an integer or a pointer, or never, `!`; not `fn(u32) -> u32`";
    assert_eq!(kernite::x86_str(src, "k").err().ok_or("x86_str gave Ok")?, format!("1:34: {returns}"));
    let src = "unsafe extern \"sysv64\" fn k() -> (u8,Marker<{N*2}>) {}";
    let returns = "a kernel returns a float, an integer or a pointer, or never, `!`; not `(u8, Marker<{N*2}>)`";
    assert_eq!(kernite::x86_str(src, "k").err().ok_or("x86_str gave Ok")?, format!("1:34: {returns}"));
    let src = "unsafe extern \"sysv64\" fn k() -> (u8,Marker<{ N * 2 }>) {}";
    let returns = "a kernel returns a float, an integer or a pointer, or never, `!`; not `(u8, Marker<{ N * 2 }>)`";
    assert_eq!(kernite::x86_str(src, "k").err().ok_or("x86_str gave Ok")?, format!("1:34: {returns}"));
    // In a result the length of an array is as the source has it; in the type of a variable, worked out.
    let src = "const N: usize = 3;\nunsafe extern \"sysv64\" fn k() -> [usize;N*2] {}";
    let returns = "a kernel returns a float, an integer or a pointer, or never, `!`; not `[usize; N*2]`";
    assert_eq!(kernite::x86_str(src, "k").err().ok_or("x86_str gave Ok")?, format!("2:34: {returns}"));
    let vars = "x86 variables are `gpr`, `ymm`, `xmm` and stack memory, `local<..>`; not";
    let src = "const N: usize = 3;\nunsafe extern \"sysv64\" fn k() { let v: Wrap<[usize;N*2]>; }";
    assert_eq!(kernite::x86_str(src, "k").err().ok_or("x86_str gave Ok")?, format!("2:40: {vars} `Wrap<[usize; 6]>`"));
    let src = "const N: usize = 3;\nunsafe extern \"sysv64\" fn k() { let v: a::Wrap<[usize;N*2]>; }";
    assert_eq!(
        kernite::x86_str(src, "k").err().ok_or("x86_str gave Ok")?,
        format!("2:40: {vars} `a::Wrap<[usize; N*2]>`")
    );
    assert_eq!(
        compile("", "let v: *const[u8;4];").err().ok_or("compile gave Ok")?,
        format!("2:8: {vars} `*const [u8; 4]`")
    );
    assert_eq!(
        compile("", "let v: fn(u32)->u32;").err().ok_or("compile gave Ok")?,
        format!("2:8: {vars} `fn(u32) -> u32`")
    );
    assert_eq!(compile("", "let v: m!(a+b);").err().ok_or("compile gave Ok")?, format!("2:8: {vars} `m!(a+b)`"));
    assert_eq!(compile("", "let v: m!(a + b);").err().ok_or("compile gave Ok")?, format!("2:8: {vars} `m!(a + b)`"));
    assert_eq!(
        compile("", "let v: (u8,Marker<{N*2}>);").err().ok_or("compile gave Ok")?,
        format!("2:8: {vars} `(u8, Marker<{{N*2}}>)`")
    );
    Ok(())
}

/// Rust never sees a kernel's body run, so kernite says which of the module's items it named, and
/// where, in the order of the source: what the macro writes for Rust at each name, so that Rust finds
/// only the constants nothing reads dead, and says what it cannot find where it is written. A name
/// in a helper the kernel calls is the kernel's, and one in the definition of a constant is not.
#[test]
fn a_kernel_names_the_items_it_names() -> Result {
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
    let module = kernite::lang::parse(src)?;
    let named = |name| -> Result<_> {
        let k = kernite::x86(&module, kernite::function(&module, name).ok_or("function gave None")?, &[])?;
        k.mentions
            .into_iter()
            .map(|m| -> Result<_> {
                Ok((m.name, src.get(..m.span.0 as usize).ok_or("get gave None")?.lines().count()))
            })
            .collect::<Result<Vec<_>>>()
    };
    let at = |name: &str, line: usize| (name.to_string(), line);
    // The helper `then`, `STORE` as its value, and `ADD` in the `if` that chooses, in its body;
    // not `OFFSET`, in the branch not taken, nor `SPARE`, in a helper not called.
    assert_eq!(named("stores")?, [at("then", 12), at("STORE", 12), at("ADD", 22)]);
    // `OFFSET` in the branch taken, and not `LANE`, which `OFFSET`'s definition names.
    assert_eq!(named("adds")?, [at("then", 18), at("ADD", 18), at("ADD", 22), at("OFFSET", 23)]);
    Ok(())
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
fn a_static_is_reached_at_a_displacement_from_rip() -> Result {
    let count = kernite::x86_str(STATICS, "count")?;
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
    assert_eq!(count.to_string(), want);
    assert_eq!(count.statics, ["SEEN", "COUNTS"]);
    let weigh = kernite::x86_str(STATICS, "weigh")?;
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
    assert_eq!(weigh.to_string(), want);
    assert_eq!(weigh.statics, ["WEIGHTS", "COUNTS", "SIGNED"]);
    Ok(())
}

/// Each static is an array of one of Rust's numbers, at the multiple its type says, holding
/// what it says, its bytes the lowest first, or zeros; and the constants it read.
#[test]
fn a_static_is_rusts_numbers_holding_what_it_says() -> Result {
    let module = kernite::lang::parse(STATICS)?;
    let statics: Vec<_> = kernite::x86_statics(&module).into_iter().map(Result::unwrap).collect();
    let shape = |s: &kernite_x86::lang::Static| (s.name.clone(), s.elem, s.size, s.len, s.align, s.constant);
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
    assert_eq!(statics[0].mentions.iter().map(|m| m.name.as_str()).collect::<Vec<_>>(), ["N"]);
    assert!(statics[1..].iter().all(|s| s.mentions.is_empty()));
    Ok(())
}

/// A helper takes memory of the module's as it takes stack memory: of the type it is declared.
#[test]
fn a_helper_takes_a_static_of_its_type() -> Result {
    let src = |param: &str| {
        format!(
            "static C: global<[u32; 16]>;\nunsafe extern \"sysv64\" fn k() {{\nbump(C);\n}}\nfn bump(c: {param}) \
             {{\nlet t: gpr;\nmov(t.r32, [c + 8]);\ninc(t);\nmov([c + 8], t.r32);\n}}\n"
        )
    };
    let code = kernite::x86_str(&src("global<[u32; 16]>"), "k")?;
    assert!(code.code().any(|l| l == "mov dword ptr [rip + {C} + 8], eax"), "{}", code.to_string());
    let e = kernite::x86_str(&src("global<[u32; 8]>"), "k").err().ok_or("x86_str gave Ok")?;
    assert_eq!(e, "3:6: `c`: `C` is memory, `global<[u32; 16]>`; it is no `global<[u32; 8]>`");
    let e = kernite::x86_str(&src("gpr"), "k").err().ok_or("x86_str gave Ok")?;
    assert_eq!(e, "3:6: `c`: `C` is memory, `global<[u32; 16]>`; it is no `gpr`");
    let src = "unsafe extern \"sysv64\" fn k(x: *const f32) {\nload(x);\n}\nfn load(b: constant<u64>) {}\n";
    assert_eq!(
        kernite::x86_str(src, "k").err().ok_or("x86_str gave Ok")?,
        "2:6: `b`: a gpr is no memory, `constant<u64>`"
    );
    Ok(())
}

/// What kernite can check of an address in a static, it does, as it does for stack memory; and
/// it refuses a write to constant memory, and an index beside a static's name, which x86 has
/// no address for.
#[test]
fn a_static_is_reached_within_it() -> Result {
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
        assert_eq!(kernite::x86_str(&src, "k").err().ok_or("x86_str gave Ok")?, want, "{body}");
    }
    // As far as one past the end for `lea`, which reaches nothing, of constant memory as well;
    // and an instruction that reads and writes memory of the module's, which is no constant.
    let src = format!(
        "{statics}unsafe extern \"sysv64\" fn k() {{\nlet g: gpr;\nlea(g, [C + 64]);\nlea(g, [W]);\nadd(dword[C + 8], \
         1);\n}}\n"
    );
    let code = kernite::x86_str(&src, "k")?;
    let want = ["lea rax, [rip + {C} + 64]", "lea rax, [rip + {W}]", "add dword ptr [rip + {C} + 8], 1", "ret"];
    assert_eq!(code.code().collect::<Vec<_>>(), want);
    Ok(())
}

/// A static of an x86 kernel's module is of Rust's numbers, as the host reads them, of a size
/// a displacement from `rip` reaches; it holds what it says, and has a name of its own.
#[test]
fn a_static_is_memory_of_rusts_numbers_holding_what_it_says() -> Result {
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
        // The name is checked before the type.
        (
            "static c0: u32;",
            "1:8: a kernel's listing names a constant `{c0}` and a function `{f0}`, and a static by its name, so one \
             is not named `c0`",
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
        ("static A: global<i32> = 1i8;", "1:25: this is an `i8`, not an `i32`"),
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
        assert_eq!(kernite::x86_str(&format!("{items}\n{k}"), "k").err().ok_or("x86_str gave Ok")?, want, "{items}");
    }
    // The kernel itself says that it names one that does not compile.
    let src =
        "static A: global<[u32; 2]> = [1];\nunsafe extern \"sysv64\" fn j() {\nlet g: gpr;\nmov(g.r32, [A]);\n}\n";
    let module = kernite::lang::parse(src)?;
    let e = kernite::x86(&module, kernite::function(&module, "j").ok_or("function gave None")?, &[])
        .err()
        .ok_or("x86 gave Ok")?;
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
        kernite::x86_str(&format!("{items}\n{k}"), "k")?;
    }
    Ok(())
}

/// A module's statics are its kernels' memory: the process's for x86 kernels, and the GPU's
/// for PTX ones, or for none. A module of both kinds of kernel and a static is two modules,
/// which its first static says, whatever comes before it.
#[test]
fn a_static_is_its_kernels_memory() -> Result {
    use kernite::Owner::{Gpu, Process};
    let x86 = "unsafe extern \"sysv64\" fn k(x: u64) -> u64 { ret(x); }";
    let gpu = "unsafe extern \"ptx-kernel\" fn g(x: *mut u32) { st.global.u32([x], 1); }";
    let func = "extern \"ptx-func\" fn f(x: u32) -> u32 { ret(x); }";
    let helper = "fn h(x: gpr) { inc(x); }";
    let other = "extern \"C\" fn c() {}";
    let (s, t) = ("static S: global<u32>;", "static T: constant<u32> = 1;");
    let whose = |items: &[&str]| -> Result<_> {
        let src = items.join("\n");
        Ok(kernite::owner(&kernite::lang::parse(&src)?).map_err(|e| e.at(&src)))
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
        assert_eq!(whose(items)?, want, "{items:?}");
    }
    Ok(())
}

/// A cooperative kernel's thread reads its place in the grid from the register after its
/// parameters, and `grid.sync()` is code of the kernel's own, in its body and in a helper's:
/// each thread counts itself in with `lock xadd`, and the last lets the others go, who spin
/// on the generation with `pause`. Its labels are the kernel's own, beside the body's.
#[test]
fn a_cooperative_kernel_waits_for_its_grid_at_a_barrier_of_its_own() -> Result {
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
    assert_eq!(kernite::x86_str(src, "k")?.to_string(), want);
    Ok(())
}

/// The cooperative kernel `k` around `body`, compiled, or the error.
fn cooperative(params: &str, body: &str) -> Result<String, String> {
    let src = format!("#[cooperative]\nunsafe extern \"sysv64\" fn k({params}) {{\n{body}\n}}\n");
    kernite::x86_str(&src, "k").map(|l| l.code().collect::<Vec<_>>().join("\n"))
}

/// What a thread may do with the grid: read where it is, which is memory, and wait for every
/// thread, in a kernel that runs on a grid, `#[cooperative]`, which takes five integers at
/// most and returns nothing.
#[test]
fn the_grid_is_a_cooperative_kernels_to_read_and_wait_for() -> Result {
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
        ("grid.wait();", "3:6: kernite's barrier for the grid is `grid.sync()`; not `grid.wait`"),
        ("grid();", "3:1: kernite's barrier for the grid is `grid.sync()`; not `grid`"),
        ("grid.sync(x);", "3:6: `grid.sync` takes no operands, not 1"),
        (
            "let g: gpr; zero(g); if g { grid.sync(); }",
            "3:34: `grid.sync()` is for every thread of the grid, under no `if`: kernite writes it with no guard, and a \
             kernel that wants it under a condition the same in every thread jumps around it",
        ),
    ];
    for (body, want) in cases {
        assert_eq!(cooperative("x: usize", body).err().ok_or("cooperative gave Ok")?, want, "{body}");
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
        assert_eq!(compile("x: usize", body).err().ok_or("compile gave Ok")?, want, "{body}");
    }
    // The place takes the sixth integer register; floats go in registers of their own.
    let five = "a: usize, b: usize, c: usize, d: usize, e: usize";
    assert_eq!(cooperative(&format!("{five}, x: f32"), "")?, "ret");
    assert_eq!(
        cooperative(&format!("{five}, f: usize"), "").err().ok_or("cooperative gave Ok")?,
        "2:82: a cooperative kernel takes five integers at most: SysV passes six in registers, and the sixth is the \
         thread's place in the grid"
    );
    let src = "#[cooperative]\nunsafe extern \"sysv64\" fn k() -> f32 {}\n";
    assert_eq!(
        kernite::x86_str(src, "k").err().ok_or("x86_str gave Ok")?,
        "2:34: a cooperative kernel's threads return nothing: they write what they compute to memory"
    );
    Ok(())
}

/// A GPU's tuning directives are the GPU kernel's: an x86 kernel says none, however many
/// instances it has, and the error is one, at the directive.
#[test]
fn an_x86_kernel_says_no_gpu_directive() -> Result {
    let not = |name: &str| {
        format!(
            "`#[{name}(..)]` goes on a GPU kernel, `extern \"ptx-kernel\"`; `k` is a kernel for x86-64, which the \
             process runs"
        )
    };
    for (attr, name) in [
        ("#[maxntid(256)]", "maxntid"),
        ("#[reqntid(8, 8)]", "reqntid"),
        ("#[minnctapersm(2)]", "minnctapersm"),
        ("#[maxnreg(32)]", "maxnreg"),
    ] {
        let src = format!("{attr}\nunsafe extern \"sysv64\" fn k() {{}}\n");
        assert_eq!(
            kernite::x86_str(&src, "k").err().ok_or("x86_str gave Ok")?,
            format!("1:3: {}", not(name)),
            "{attr}"
        );
    }
    // What PTX has not is said by its name, whatever the kernel.
    let src = "#[maxnctapersm(2)]\nunsafe extern \"sysv64\" fn k() {}\n";
    assert_eq!(
        kernite::x86_str(src, "k").err().ok_or("x86_str gave Ok")?,
        "1:3: PTX has no `.maxnctapersm` from 2.1 on; `#[minnctapersm(..)]` gives the blocks an SM should hold at \
         once, at least"
    );
    let src = "#[maxnreg(32)] #[instances(1, 2)]\nunsafe extern \"sysv64\" fn k<const N: u32>() {}\n";
    for name in ["k::<1>", "k::<2>"] {
        assert_eq!(
            kernite::x86_str(src, name).err().ok_or("x86_str gave Ok")?,
            format!("1:3: {}", not("maxnreg")),
            "{name}"
        );
    }
    Ok(())
}

/// Stack memory is the kernel's frame: the header says where each slot lies, and brackets reach
/// it from `rsp`, an index beside it.
#[test]
fn stack_memory_is_part_of_the_frame() -> Result {
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
    assert_eq!(kernite::x86_str(src, "k")?.to_string(), want);
    Ok(())
}

/// Until written, stack memory holds whatever was there: kernite clears none of it, and takes a
/// read of it before any write, as it does not of a variable.
#[test]
fn stack_memory_holds_whatever_was_there_until_written() -> Result {
    let src = "unsafe extern \"sysv64\" fn k() -> u64 {\nlet b: local<u64>;\nlet g: gpr;\nmov(g, [b]);\nret(g);\n}\n";
    let want = "# k() -> gpr
# frame: 8 bytes aligned to 8, saves []
# b: 8 bytes at rsp - 8
# g=rax
    mov rax, qword ptr [rsp - 8]
    ret
";
    assert_eq!(kernite::x86_str(src, "k")?.to_string(), want);
    Ok(())
}

/// The stack pointer is 8 past a multiple of 16 on entry, as SysV says, so a frame aligned to 16 or less keeps no `rbp`
/// and takes no `and`: it lies below the stack pointer, up to 128 bytes of it, and `sub` makes the rest, which `add`
/// takes down. One aligned to 16 lies at a multiple of 16 whether the kernel calls or not, and the listing says why it
/// is as long as it is. One aligned to more keeps the caller's stack pointer in `rbp`, aligns its own with `and`, and
/// takes the frame down from `rbp`.
#[test]
fn only_a_frame_aligned_past_16_keeps_the_stack_pointer_in_rbp() -> Result {
    let cases = [
        ("local<u64>", "mov rax, qword ptr [rsp - 8]\nret"),
        ("local<[u64; 2], 16>", "mov rax, qword ptr [rsp - 24]\nret"),
        (
            "local<[u8; 4096], 16>",
            "sub rsp, 4096\nor qword ptr [rsp], 0\nmov rax, qword ptr [rsp - 8]\nadd rsp, 4096\nret",
        ),
        (
            "local<[u8; 32], 32>",
            "push rbp\nmov rbp, rsp\nsub rsp, 32\nand rsp, -32\nmov rax, qword ptr [rsp]\nmov rsp, rbp\npop rbp\nret",
        ),
    ];
    for (ty, want) in cases {
        assert_eq!(compile("", &format!("let b: {ty}; let g: gpr; mov(g, [b]);"))?, want, "{ty}");
    }
    let src = "unsafe extern \"sysv64\" fn k() {\nlet b: local<[u64; 2], 16>;\nlet g: gpr;\nmov(g, [b]);\n}\n";
    let text = kernite::x86_str(src, "k")?.to_string();
    assert!(
        text.contains("\n# frame: 16 bytes aligned to 16, saves [], and 8 more so that it is aligned to 16\n"),
        "{text}"
    );
    Ok(())
}

/// An `add` takes 2^31 - 1 at most, and the stack pointer may move by 2^31 bytes: a frame of 2^31 bytes and the 8 more
/// that align it to 16 keeps 8 bytes below the stack pointer, and the kernel takes the rest down by two. Eighteen xmm
/// alive at once spill two, 16 bytes each, after stack memory of 2^31 - 32 bytes.
#[test]
fn a_frame_of_2_to_the_31_bytes_is_taken_down_by_two_adds() -> Result {
    let loads: String = (0..18).map(|i| format!("let v{i}: xmm;\nvmovups(v{i}, [x + {}]);\n", 16 * i)).collect();
    let sums: String = (1..18).map(|i| format!("vaddps(v0, v0, v{i});\n")).collect();
    let src = format!(
        "unsafe extern \"sysv64\" fn k(x: *mut f32) {{\nlet big: local<[u8; 2147483616], 16>;\n{loads}{sums}\
         vmovups([big], v0);\n}}\n"
    );
    let text = kernite::x86_str(&src, "k")?.to_string();
    let frame = "# frame: 2147483648 bytes aligned to 16, saves [], and 8 more so that it is aligned to 16\n";
    assert!(text.contains(frame), "{text}");
    assert!(text.contains("\n    lea r11, [rsp - 2147483648]\n"), "{text}");
    assert!(text.ends_with("\n    add rsp, 2147483647\n    add rsp, 1\n    ret\n"), "{text}");
    Ok(())
}

/// A frame of more than a page is touched a page at a time on the way down, as Rust's functions are, so that a frame
/// past the end of the stack faults at the guard page there rather than step over it: no byte of the frame lies more
/// than a page below memory the kernel has touched, those below the stack pointer counted. Past three pages it is
/// touched in a loop. What the alignment may add counts.
#[test]
fn a_frame_of_more_than_a_page_is_touched_a_page_at_a_time() -> Result {
    let prologue = |ty: &str| -> Result<_> {
        let code = compile("", &format!("let b: {ty}; let g: gpr; mov(g, [b]);"))?;
        let (prologue, _) = code.split_once("\nmov rax").ok_or("split_once gave None")?;
        Ok(prologue.to_string())
    };
    let page = "sub rsp, 4096\nor qword ptr [rsp], 0";
    let cases = [
        ("local<[u8; 129]>", "sub rsp, 1".to_string()),
        ("local<[u8; 4096]>", "sub rsp, 3968".into()),
        ("local<[u8; 4097]>", page.into()),
        ("local<[u8; 4224]>", page.into()),
        ("local<[u8; 4225]>", format!("{page}\nsub rsp, 1")),
        ("local<[u8; 4064], 32>", "push rbp\nmov rbp, rsp\nsub rsp, 4064\nand rsp, -32".into()),
        ("local<[u8; 4096], 16>", page.into()),
        ("local<[u8; 4096], 4096>", format!("push rbp\nmov rbp, rsp\n{page}\nand rsp, -4096")),
        ("local<[u8; 16384]>", format!("{page}\n{page}\n{page}\nsub rsp, 3968")),
        ("local<[u8; 16385]>", format!("lea r11, [rsp - 16384]\n2:\n{page}\ncmp rsp, r11\njne 2b")),
    ];
    for (ty, want) in cases {
        assert_eq!(prologue(ty)?, want, "{ty}");
    }
    Ok(())
}

/// A frame of more than 2^31 bytes, its stack memory and the variables that spilled, is more than
/// an address reaches past the stack pointer, and the kernel is refused at its name. Eighteen
/// vectors alive at once spill two, after stack memory of `N` bytes. A frame of 2^31 bytes
/// compiles, its last spill at the furthest an address reaches and its loop's end at the furthest
/// down; a byte more of stack memory makes it 32 bytes more, as a spill is aligned to 32.
#[test]
fn a_frame_past_what_an_address_reaches_is_refused() -> Result {
    let loads: String = (0..18).map(|i| format!("let v{i}: ymm;\nvmovups(v{i}, [x + {}]);\n", 32 * i)).collect();
    let sums: String = (1..18).map(|i| format!("vaddps(v0, v0, v{i});\n")).collect();
    let src = format!(
        "#[instances(2147483584, 2147483585)]\nunsafe extern \"sysv64\" fn k<const N: usize>(x: *mut f32) {{\n\
         let big: local<[u8; N], 32>;\n{loads}{sums}vmovups([big], v0);\n}}\n"
    );
    let text = kernite::x86_str(&src, "k::<2147483584>")?.to_string();
    let frame = "# frame: 2147483648 bytes aligned to 32, saves []
# big: 2147483584 bytes at rsp
# v16, spilled: 32 bytes at rsp + 2147483584
# v15, spilled: 32 bytes at rsp + 2147483616
";
    assert!(text.contains(frame), "{text}");
    assert!(text.contains("\n    lea r11, [rsp - 2147483648]\n"), "{text}");
    let want = "2:27: in `k::<2147483585>`: the kernel's frame, its stack memory and the variables that spilled, \
                reaches 2147483680 bytes past the stack pointer: more than an address does";
    assert_eq!(kernite::x86_str(&src, "k::<2147483585>").err().ok_or("x86_str gave Ok")?, want);
    Ok(())
}

/// The prologue's loop takes the first label, and the body's come after it.
#[test]
fn the_prologues_loop_comes_before_the_bodys_labels() -> Result {
    let code = compile("n: usize", "let b: local<[u64; 4096]>;\nlabel(top);\nmov([b], n);\ndec(n);\njnz(top);")?;
    let want = "lea r11, [rsp - 28672]
2:
sub rsp, 4096
or qword ptr [rsp], 0
cmp rsp, r11
jne 2b
sub rsp, 3968
3:
mov qword ptr [rsp - 128], rdi
dec rdi
jnz 3b
add rsp, 32640
ret";
    assert_eq!(code, want);
    Ok(())
}

/// A helper takes stack memory as a parameter of the type it was declared.
#[test]
fn a_helper_takes_stack_memory_of_its_type() -> Result {
    let src = |param: &str| {
        format!(
            "unsafe extern \"sysv64\" fn k() {{\nlet buf: local<[f32; 16], 32>;\nlet v: ymm;\nzero(v);\nstore(buf, \
             v);\n}}\nfn store(b: {param}, v: ymm) {{\nvmovaps([b + 32], v);\n}}\n"
        )
    };
    let code = kernite::x86_str(&src("local<[f32; 16], 32>"), "k")?;
    assert!(code.code().any(|l| l == "vmovaps ymmword ptr [rsp + 32], ymm0"), "{}", code.to_string());
    let e = kernite::x86_str(&src("local<[f32; 16]>"), "k").err().ok_or("x86_str gave Ok")?;
    assert_eq!(e, "5:7: `b`: `buf` is memory, `local<[f32; 16], 32>`; it is no `local<[f32; 16]>`");
    let e = kernite::x86_str(&src("ymm"), "k").err().ok_or("x86_str gave Ok")?;
    assert_eq!(e, "5:7: `b`: `buf` is memory, `local<[f32; 16], 32>`; it is no `ymm`");
    let src = "unsafe extern \"sysv64\" fn k(x: *const f32) {\nload(x);\n}\nfn load(b: local<u64>) {}\n";
    assert_eq!(
        kernite::x86_str(src, "k").err().ok_or("x86_str gave Ok")?,
        "2:6: `b`: a gpr is no memory, `local<u64>`"
    );
    Ok(())
}

/// What kernite can check of stack memory, it does: its type, and an address where the index is 0.
#[test]
fn stack_memory_is_checked_where_kernite_can() -> Result {
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
        assert_eq!(compile("x: *const f32", body).err().ok_or("compile gave Ok")?, want, "{body}");
    }
    Ok(())
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
/// and its callee gives back in rax: here, where the arguments were, but for the swap. A `ret` follows the call, so it
/// is a jump, and `gcd` goes on in the stack it has, as a loop does.
#[test]
fn a_function_calls_itself_with_its_arguments_where_sysv_passes_them() -> Result {
    let want = "# gcd(gpr, gpr) -> gpr
# frame: 0 bytes aligned to 0, saves []
# a=rdi,rax b=rsi t=rax g=rax
    test rsi, rsi
    jnz 2f
    mov rax, rdi
    ret
2:
    cmp rdi, rsi
    jae 3f
    mov rax, rdi
    mov rdi, rsi
    mov rsi, rax
3:
    sub rdi, rsi
    jmp {f0}
# f0: gcd
";
    assert_eq!(kernite::x86_str(CALLERS, "gcd")?.to_string(), want);
    Ok(())
}

/// A parameter that a call needs the register of moves on entry to the register it waits out the call in, and an
/// argument in the register it arrived in is not moved back there, for the register holds it still: `x` is moved once,
/// to `rcx`, and `1` is moved to `rsi`.
#[test]
fn an_integer_passed_in_its_own_register_and_alive_after_the_call_moves_once() -> Result<(), String> {
    let src = "unsafe extern \"sysv64\" fn k(x: u64) -> u64 { let r: gpr; call(r, g, (x, 1)); add(r, x); ret(r); }
unsafe extern \"sysv64\" fn g(a: u64, b: u64) -> u64 { sub(a, b); ret(a); }";
    let want = "# k(gpr) -> gpr
# frame: 0 bytes aligned to 0, saves [], and 8 more so that it calls at a multiple of 16
# x=rcx,rdi r=rax b=rsi
    sub rsp, 8
    mov rcx, rdi
    mov rsi, 1
    call {f0}
    add rax, rcx
    add rsp, 8
    ret
# f0: g
";
    assert_eq!(kernite::x86_str(src, "k")?.to_string(), want);
    Ok(())
}

/// A float passed twice is moved twice where the callee writes both registers it is passed in: to the register it waits
/// out the call in and to `xmm1`, both from `xmm0`, the register it arrived in, so that neither move waits for the
/// other, and not back to `xmm0`.
#[test]
fn a_float_passed_twice_and_alive_after_the_call_moves_twice() -> Result<(), String> {
    let src =
        "unsafe extern \"sysv64\" fn k(x: f32) -> f32 { let r: xmm; call(r, h, (x, x)); vaddss(r, r, x); ret(r); }
unsafe extern \"sysv64\" fn h(a: f32, b: f32) -> f32 { vsubss(a, a, b); vaddss(b, b, a); ret(a); }";
    let want = "# k(vec) -> vec
# needs: avx
# frame: 0 bytes aligned to 0, saves [], and 8 more so that it calls at a multiple of 16
# x=ymm2,ymm0,ymm1 r=ymm0
    sub rsp, 8
    vmovaps xmm2, xmm0
    vmovaps xmm1, xmm0
    call {f0}
    vaddss xmm0, xmm0, xmm2
    add rsp, 8
    ret
# f0: h
";
    assert_eq!(kernite::x86_str(src, "k")?.to_string(), want);
    Ok(())
}

/// An integer passed twice is moved twice too, both from `rdi`, where the callee writes both registers it is passed in.
#[test]
fn an_integer_passed_twice_moves_twice_too() -> Result<(), String> {
    let src = "unsafe extern \"sysv64\" fn k(x: u64) -> u64 { let r: gpr; call(r, g, (x, x)); add(r, x); ret(r); }
unsafe extern \"sysv64\" fn g(a: u64, b: u64) -> u64 { sub(a, b); add(b, 1); ret(a); }";
    let want = "# k(gpr) -> gpr
# frame: 0 bytes aligned to 0, saves [], and 8 more so that it calls at a multiple of 16
# x=rcx,rdi,rsi r=rax
    sub rsp, 8
    mov rcx, rdi
    mov rsi, rdi
    call {f0}
    add rax, rcx
    add rsp, 8
    ret
# f0: g
";
    assert_eq!(kernite::x86_str(src, "k")?.to_string(), want);
    Ok(())
}

/// An integer passed twice and read no more after the call stays in `rdi`, where it arrived, and one move passes it the
/// second time, from `rdi`, once `y` has been read from `rsi`: it is not moved on entry to a register of its own, and
/// from there to `rsi`.
#[test]
fn an_integer_passed_twice_and_read_no_more_moves_once() -> Result<(), String> {
    let src = "unsafe extern \"sysv64\" fn k(x: u64, y: u64) -> u64 { let (r, t): (gpr, gpr); mov(t, y); add(t, 1); \
               call(r, g, (x, x)); add(r, t); ret(r); }
unsafe extern \"sysv64\" fn g(a: u64, b: u64) -> u64 { sub(a, b); ret(a); }";
    let want = "# k(gpr, gpr) -> gpr
# frame: 0 bytes aligned to 0, saves [], and 8 more so that it calls at a multiple of 16
# x=rdi,rsi y=rsi r=rax t=rcx
    sub rsp, 8
    mov rcx, rsi
    add rcx, 1
    mov rsi, rdi
    call {f0}
    add rax, rcx
    add rsp, 8
    ret
# f0: g
";
    assert_eq!(kernite::x86_str(src, "k")?.to_string(), want);
    Ok(())
}

/// A value passed twice, and alive after the call, waits it out in the register that passes it the second time where
/// the callee leaves that register as it was, an integer in `rsi` and a float in `xmm1`: it is moved there once, from
/// the register it arrived in, and not to another register as well.
#[test]
fn a_value_passed_twice_waits_out_the_call_in_the_register_that_passes_it_again() -> Result<(), String> {
    let src = "unsafe extern \"sysv64\" fn k(x: u64) -> u64 { let r: gpr; call(r, g, (x, x)); add(r, x); ret(r); }
unsafe extern \"sysv64\" fn g(a: u64, b: u64) -> u64 { sub(a, b); ret(a); }";
    let want = "# k(gpr) -> gpr
# frame: 0 bytes aligned to 0, saves [], and 8 more so that it calls at a multiple of 16
# x=rsi,rdi r=rax
    sub rsp, 8
    mov rsi, rdi
    call {f0}
    add rax, rsi
    add rsp, 8
    ret
# f0: g
";
    assert_eq!(kernite::x86_str(src, "k")?.to_string(), want);
    let src =
        "unsafe extern \"sysv64\" fn k(x: f32) -> f32 { let r: xmm; call(r, h, (x, x)); vaddss(r, r, x); ret(r); }
unsafe extern \"sysv64\" fn h(a: f32, b: f32) -> f32 { vsubss(a, a, b); ret(a); }";
    let want = "# k(vec) -> vec
# needs: avx
# frame: 0 bytes aligned to 0, saves [], and 8 more so that it calls at a multiple of 16
# x=ymm1,ymm0 r=ymm0
    sub rsp, 8
    vmovaps xmm1, xmm0
    call {f0}
    vaddss xmm0, xmm0, xmm1
    add rsp, 8
    ret
# f0: h
";
    assert_eq!(kernite::x86_str(src, "k")?.to_string(), want);
    Ok(())
}

/// A register holds at a label what every way to it brings. A jump and the code before the label both leave `x` in
/// `rdi`, where it arrived, so that the call after the label passes it with no move; and where the code before it calls
/// a function that writes `rdi`, the call after the label moves `x` to `rdi` again, from the register it waits in.
#[test]
fn a_move_after_a_label_is_left_out_where_every_way_to_it_leaves_the_value_there() -> Result<(), String> {
    let src = "unsafe extern \"sysv64\" fn k(x: u64, y: u64) -> u64 { let r: gpr; test(y, y); jz(out); add(y, 1); \
               label(out); call(r, g, (x, y)); add(r, x); ret(r); }
unsafe extern \"sysv64\" fn g(a: u64, b: u64) -> u64 { sub(a, b); ret(a); }";
    let want = "# k(gpr, gpr) -> gpr
# frame: 0 bytes aligned to 0, saves [], and 8 more so that it calls at a multiple of 16
# x=rcx,rdi y=rsi r=rax
    sub rsp, 8
    mov rcx, rdi
    test rsi, rsi
    jz 2f
    add rsi, 1
2:
    call {f0}
    add rax, rcx
    add rsp, 8
    ret
# f0: g
";
    assert_eq!(kernite::x86_str(src, "k")?.to_string(), want);
    let src = "unsafe extern \"sysv64\" fn k(x: u64, y: u64) -> u64 { let (r, q): (gpr, gpr); test(y, y); jz(out); \
               call(q, g, (y, y)); label(out); call(r, g, (x, y)); add(r, x); ret(r); }
unsafe extern \"sysv64\" fn g(a: u64, b: u64) -> u64 { sub(a, b); ret(a); }";
    let want = "# k(gpr, gpr) -> gpr
# frame: 0 bytes aligned to 0, saves [], and 8 more so that it calls at a multiple of 16
# x=rcx,rdi y=rsi,rdi r=rax q=rax
    sub rsp, 8
    mov rcx, rdi
    test rsi, rsi
    jz 2f
    mov rdi, rsi
    call {f0}
2:
    mov rdi, rcx
    call {f0}
    add rax, rcx
    add rsp, 8
    ret
# f0: g
";
    assert_eq!(kernite::x86_str(src, "k")?.to_string(), want);
    Ok(())
}

/// A register holds in a loop what the way into it and the jump back both bring, so that a call in the loop of a
/// function that leaves `rdi` as it was passes `x` with no move, whether the loop is entered at its top or at its test;
/// and one of a function that writes `rdi` has the move each time round. `x` is read after a call of a function that
/// writes `rdi`, so that it waits that call out in `rcx`, where a move on entry puts it.
#[test]
fn a_move_in_a_loop_is_left_out_where_the_way_in_and_the_jump_back_leave_the_value_there() -> Result<(), String> {
    let h = "unsafe extern \"sysv64\" fn h(a: u64) -> u64 { let r: gpr; mov(r, a); add(r, 1); ret(r); }";
    let g = "unsafe extern \"sysv64\" fn g(a: u64, b: u64) -> u64 { sub(a, b); ret(a); }";
    let src = format!(
        "unsafe extern \"sysv64\" fn k(x: u64, n: u64) -> u64 {{ let (r, s): (gpr, gpr); mov(s, 0); label(top); \
         call(r, h, (x)); add(s, r); dec(n); jnz(top); call(r, g, (s, 1)); add(r, x); ret(r); }}\n{h}\n{g}"
    );
    let want = "# k(gpr, gpr) -> gpr
# frame: 0 bytes aligned to 0, saves [], and 8 more so that it calls at a multiple of 16
# x=rcx,rdi n=rsi r=rax s=rdx,rdi b=rsi
    sub rsp, 8
    mov rcx, rdi
    mov rdx, 0
2:
    call {f0}
    add rdx, rax
    dec rsi
    jnz 2b
    mov rdi, rdx
    mov rsi, 1
    call {f1}
    add rax, rcx
    add rsp, 8
    ret
# f0: h
# f1: g
";
    assert_eq!(kernite::x86_str(&src, "k")?.to_string(), want);
    let src = format!(
        "unsafe extern \"sysv64\" fn k(x: u64, n: u64) -> u64 {{ let (r, s): (gpr, gpr); mov(s, 0); jmp(test); \
         label(top); call(r, h, (x)); add(s, r); label(test); dec(n); jnz(top); call(r, g, (s, 1)); add(r, x); \
         ret(r); }}\n{h}\n{g}"
    );
    let want = "# k(gpr, gpr) -> gpr
# frame: 0 bytes aligned to 0, saves [], and 8 more so that it calls at a multiple of 16
# x=rcx,rdi n=rsi r=rax s=rdx,rdi b=rsi
    sub rsp, 8
    mov rcx, rdi
    mov rdx, 0
    jmp 3f
2:
    call {f0}
    add rdx, rax
3:
    dec rsi
    jnz 2b
    mov rdi, rdx
    mov rsi, 1
    call {f1}
    add rax, rcx
    add rsp, 8
    ret
# f0: h
# f1: g
";
    assert_eq!(kernite::x86_str(&src, "k")?.to_string(), want);
    let src = "unsafe extern \"sysv64\" fn k(x: u64, n: u64) -> u64 { let (r, s): (gpr, gpr); mov(s, 0); label(top); \
               call(r, g, (x, 1)); add(s, r); dec(n); jnz(top); add(s, x); ret(s); }
unsafe extern \"sysv64\" fn g(a: u64, b: u64) -> u64 { sub(a, b); ret(a); }";
    let want = "# k(gpr, gpr) -> gpr
# frame: 0 bytes aligned to 0, saves [], and 8 more so that it calls at a multiple of 16
# x=rcx,rdi n=rdx,rsi r=rax s=r8,rax b=rsi
    sub rsp, 8
    mov rcx, rdi
    mov rdx, rsi
    mov r8, 0
2:
    mov rdi, rcx
    mov rsi, 1
    call {f0}
    add r8, rax
    dec rdx
    jnz 2b
    add r8, rcx
    mov rax, r8
    add rsp, 8
    ret
# f0: g
";
    assert_eq!(kernite::x86_str(src, "k")?.to_string(), want);
    Ok(())
}

/// A move into a register that holds another value stays: `y` takes `x`, then `z`, then `x` again, and the third move
/// is made; and so does one into a register that holds only part of the value, as an xmm clears the rest of its ymm:
/// `y` takes the low half of `x`, and `z`, in the same register, takes all of it.
#[test]
fn a_move_into_a_register_that_holds_another_value_or_part_of_this_one_stays() -> Result<(), String> {
    let src = "unsafe extern \"sysv64\" fn k(x: u64, z: u64) -> u64 { let (y, w): (gpr, gpr); mov(y, x); mov(w, y); \
               mov(y, z); add(w, y); mov(y, x); add(w, y); add(w, x); add(w, z); ret(w); }";
    let want = "# k(gpr, gpr) -> gpr
# frame: 0 bytes aligned to 0, saves []
# x=rdi z=rsi y=rcx w=rax
    mov rcx, rdi
    mov rax, rcx
    mov rcx, rsi
    add rax, rcx
    mov rcx, rdi
    add rax, rcx
    add rax, rdi
    add rax, rsi
    ret
";
    assert_eq!(kernite::x86_str(src, "k")?.to_string(), want);
    let src = "unsafe extern \"sysv64\" fn k(p: *mut f32) { let (x, y, z): (ymm, xmm, ymm); vmovups(x, [p]); \
               vmovaps(y, x.xmm); vmovss([p], y); vmovaps(z, x); vmovups([p + 32], z); vmovups([p + 64], x); }";
    let want = "# k(gpr) -> ()
# needs: avx
# frame: 0 bytes aligned to 0, saves []
# p=rdi x=ymm0 y=ymm1 z=ymm1
    vmovups ymm0, ymmword ptr [rdi]
    vmovaps xmm1, xmm0
    vmovss dword ptr [rdi], xmm1
    vmovaps ymm1, ymm0
    vmovups ymmword ptr [rdi + 32], ymm1
    vmovups ymmword ptr [rdi + 64], ymm0
    vzeroupper
    ret
";
    assert_eq!(kernite::x86_str(src, "k")?.to_string(), want);
    Ok(())
}

/// A call changes the registers its callee writes, and those that each function the callee calls writes: an integer
/// live across one stays in a register none of them writes, here rcx, for `gcd` writes rdi, rsi and rax alone, and
/// calls itself; and a vector stays in the register it is in, for `nothing` writes none.
#[test]
fn a_value_live_across_a_call_is_kept_where_the_call_leaves_it() -> Result {
    let want = "# plus(gpr) -> gpr
# frame: 0 bytes aligned to 0, saves [], and 8 more so that it calls at a multiple of 16
# x=rcx,rdi y=rax b=rsi
    sub rsp, 8
    mov rcx, rdi
    mov rsi, 6
    call {f0}
    add rax, rcx
    add rsp, 8
    ret
# f0: gcd
";
    assert_eq!(kernite::x86_str(CALLERS, "plus")?.to_string(), want);
    let want = "# keep(gpr) -> ()
# needs: avx
# frame: 0 bytes aligned to 0, saves [], and 8 more so that it calls at a multiple of 16
# p=rdi v=ymm0
    sub rsp, 8
    vmovups ymm0, ymmword ptr [rdi]
    call {f0}
    vmovups ymmword ptr [rdi], ymm0
    add rsp, 8
    vzeroupper
    ret
# f0: nothing
";
    assert_eq!(kernite::x86_str(CALLERS, "keep")?.to_string(), want);
    Ok(())
}

/// A vector `v` stays in a register across a call, though not in one the call changes. A call changes the registers its
/// callee writes, and each function the callee calls, and the upper half of every vector register where one of them
/// runs `vzeroupper`: `v` then spills to the frame if the kernel reads all of it after the call, and keeps its register
/// if it reads only its lower half, `v.xmm`, which `vzeroupper` leaves as it was, or writes `v.xmm` first, which clears
/// the upper half. A call at an address with the functions it may reach given is of each; with none given, or of the
/// kernel itself, which is not compiled yet, it changes every vector register, and `v` spills whatever it reads.
#[test]
fn a_vector_live_across_a_call_spills_only_where_the_call_may_clear_a_half_the_kernel_reads() -> Result {
    let at = "let fp: gpr; lea(fp, [nothing]);";
    // Each case a call, what the kernel reads of `v` after it, and where `v` is: its registers, and `spilled` if the
    // frame keeps it.
    let cases = [
        // `nothing` writes no register: `v` stays in `ymm0`.
        ("", "call(nothing);", "v", "ymm0"),
        // `writes_ymm0` writes `ymm0`: `v` moves to `ymm1`.
        ("", "call(writes_ymm0);", "v", "ymm1"),
        // `runs_vzeroupper` writes `ymm0` too, and clears the upper half of each vector register: `v` spills if the
        // kernel reads all of it after the call, and moves to `ymm1` if it reads the lower half alone.
        ("", "call(runs_vzeroupper);", "v", "ymm0 spilled"),
        ("", "call(runs_vzeroupper);", "v.xmm", "ymm1"),
        // An instruction that writes `v.xmm` clears the upper half of `v`, though it reads `v.xmm`, as an FMA does: `v`
        // moves to `ymm1` though the kernel reads all of it after.
        ("", "call(runs_vzeroupper); vfmadd231ps(v.xmm, v.xmm, v.xmm);", "v", "ymm1"),
        // `calls_vzeroupper` writes none, and calls it.
        ("", "call(calls_vzeroupper);", "v", "ymm0 spilled"),
        ("", "call(calls_vzeroupper);", "v.xmm", "ymm1"),
        // At an address, one of the functions given: each is as above.
        (at, "call(fp, (nothing, writes_ymm0));", "v", "ymm1"),
        (at, "call(fp, (nothing, runs_vzeroupper));", "v", "ymm0 spilled"),
        (at, "call(fp, (nothing, runs_vzeroupper));", "v.xmm", "ymm1"),
        // At an address, one of any function, which may write each register and clear each upper half: `v` spills
        // whatever the kernel reads of it.
        (at, "call(fp);", "v.xmm", "ymm0 spilled"),
        // The kernel itself, `k`, which is not compiled yet: as one of any function.
        ("", "call(k, p);", "v.xmm", "ymm0 spilled"),
    ];
    for (setup, call, read, want) in cases {
        let body = format!("let v: ymm; {setup} vmovups(v, [p]); {call} vmovups([p], {read});");
        let listing = listing("p: *mut f32", "", &body)?;
        let (_, regs) = listing.regs.iter().find(|(name, _)| name == "v").ok_or("find gave None")?;
        let spilled = if listing.frame.slots.iter().any(|s| s.spill) { " spilled" } else { "" };
        assert_eq!(format!("{regs}{spilled}"), want, "{call} {read}\n{}", listing.to_string());
    }
    Ok(())
}

/// A vector that an address takes its indices from, as a gather's are, counts as read whole after a call of a function
/// that runs `vzeroupper`, though a gather of an xmm of indices reads its lower half alone: kernite does not know how
/// much of the vector an address reads. So the vector stays in the frame, and in its register where the call clears no
/// half.
#[test]
fn a_vector_used_as_the_indices_of_an_address_counts_as_read_whole_after_a_call_that_clears_a_half() -> Result {
    let cases = [
        ("nothing", "vpgatherdd(d, [p + v * 4], m)", false),
        ("runs_vzeroupper", "vpgatherdd(d, [p + v * 4], m)", true),
        ("runs_vzeroupper", "vpgatherdq(d, [p + v.xmm * 8], m)", true),
    ];
    for (callee, gather, spills) in cases {
        let body = format!(
            "let (v, d, m): (ymm, ymm, ymm); vmovups(v, [p]); call({callee}); ones(m); zero(d); {gather}; \
             vmovups([p], d);"
        );
        let listing = listing("p: *mut f32", "", &body)?;
        assert_eq!(listing.frame.slots.iter().any(|s| s.spill), spills, "{callee} {gather}\n{listing}");
    }
    Ok(())
}

/// The registers every call leaves as they were are rbx and r12 to r15: an integer alive across a call of a function
/// that changes all the others is kept in them alone, five, each saved for the kernel's caller, and a sixth is spilled
/// to the frame. Across a call of one that changes none, all six stay where they are.
#[test]
fn a_value_alive_across_a_call_that_changes_every_register_is_kept_in_rbx_or_r12_to_r15() -> Result {
    let params = "a: u64, b: u64, c: u64, d: u64, e: u64, f: u64";
    let body = |callee| format!("call({callee}); add(a, b); add(a, c); add(a, d); add(a, e); add(a, f); ret(a);");
    let code = calling(params, " -> u64", &body("wipe"))?;
    let want = "push rbx\npush r12\npush r13\npush r14\npush r15\nsub rsp, 16\nmov qword ptr [rsp], r9\nmov rbx, \
                r8\nmov r12, rcx\nmov r13, rdx\nmov r14, rsi\nmov r15, rdi\ncall {f0}\nadd r15, r14\nadd r15, \
                r13\nadd r15, r12\nadd r15, rbx\nmov rax, qword ptr [rsp]\nadd r15, rax\nmov rax, r15\nadd rsp, \
                16\npop r15\npop r14\npop r13\npop r12\npop rbx\nret";
    assert_eq!(code, want);
    let code = calling(params, " -> u64", &body("nothing"))?;
    let want = "sub rsp, 8\ncall {f0}\nadd rdi, rsi\nadd rdi, rdx\nadd rdi, rcx\nadd rdi, r8\nadd rdi, r9\nmov rax, \
                rdi\nadd rsp, 8\nret";
    assert_eq!(code, want);
    Ok(())
}

/// A call changes the registers its callee writes, and those of each function the callee calls, by a jump or a call, by
/// name or at an address with the functions it may reach given, each counted once though two call each other: an
/// integer alive across it stays in a register none of them writes, rcx or rdx here, for each writes rax. A call at an
/// address with no functions given may reach any function, a call of the kernel itself reaches code that is not
/// compiled yet, and a call of a function that does not compile reaches code kernite does not have: each changes every
/// register a callee may, and the integer stays in rbx, which every call leaves as it was and the kernel saves.
#[test]
fn a_call_changes_the_registers_of_each_function_it_may_reach() -> Result {
    let kernel = |reg: &str, calls: &str| match reg {
        "rbx" => format!("push rbx\nmov rbx, 5\n{calls}\nadd rbx, 1\nmov rax, rbx\npop rbx\nret"),
        _ => format!("sub rsp, 8\nmov {reg}, 5\n{calls}\nadd {reg}, 1\nmov rax, {reg}\nadd rsp, 8\nret"),
    };
    let at = "lea rax, [rip + {f0}]\ncall rax";
    let cases = [
        // `writes_rax`, `jumps`, which jumps to it, and `calls`, which calls it and goes on, each change rax.
        ("call(writes_rax);", "rcx", "call {f0}"),
        ("call(jumps);", "rcx", "call {f0}"),
        ("call(calls);", "rcx", "call {f0}"),
        // `reaches` writes rax alone, and calls at an address one of `jumps` and `writes_rax_rcx`.
        ("call(reaches);", "rdx", "call {f0}"),
        // The first call changes rax, and the second rax and rcx.
        ("call(writes_rax); call(writes_rax_rcx);", "rdx", "call {f0}\ncall {f1}"),
        // `ping` writes rax and jumps to `pong`, which writes rax and rcx, and jumps back.
        ("call(ping);", "rdx", "call {f0}"),
        // At an address, one of `jumps` and `writes_rax_rcx`, which together change rax and rcx.
        ("let fp: gpr; lea(fp, [jumps]); call(fp, (jumps, writes_rax_rcx));", "rdx", at),
        // At an address, one of any function.
        ("let fp: gpr; lea(fp, [jumps]); call(fp);", "rbx", at),
        // The kernel itself, `k`, which is not compiled yet.
        ("call(_, k, ());", "rbx", "call {f0}"),
        // `broken` does not compile, so kernite has no code of it.
        ("call(broken);", "rbx", "call {f0}"),
    ];
    for (call, reg, calls) in cases {
        let body = format!("let x: gpr; mov(x, 5); {call} add(x, 1); ret(x);");
        assert_eq!(calling("", " -> u64", &body)?, kernel(reg, calls), "{call}");
    }
    Ok(())
}

/// `lea` takes the address of a function of the module, rip-relative, and a call through the
/// gpr that holds it calls one of the functions taken together last, which take what it
/// passes. A `ret` follows it, so it is a jump to the address.
#[test]
fn a_call_through_a_gpr_calls_one_of_the_functions_taken_together_last() -> Result {
    let want = "# through(gpr) -> gpr
# frame: 0 bytes aligned to 0, saves []
# a=rdi,rsi fp=rax r=rax
    lea rax, [rip + {f0}]
    mov rsi, rdi
    jmp rax
# f0: gcd
";
    assert_eq!(kernite::x86_str(CALLERS, "through")?.to_string(), want);
    Ok(())
}

/// A function that never returns, `-> !`, ends each way out in `ud2` or in a call of such a
/// function, which no epilogue follows.
#[test]
fn a_function_that_never_returns_ends_in_ud2_or_a_call_of_one() -> Result {
    let want = "# fail() -> !
# frame: 0 bytes aligned to 0, saves []
    ud2
";
    assert_eq!(kernite::x86_str(CALLERS, "fail")?.to_string(), want);
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
    assert_eq!(kernite::x86_str(CALLERS, "at")?.to_string(), want);
    Ok(())
}

/// Functions of the module for `k` to call, and some it may not. `broken` does not compile.
const CALLEES: &str = r#"
unsafe extern "sysv64" fn gcd(a: u64, b: u64) -> u64 { ret(a); }
unsafe extern "sysv64" fn half(x: f32) -> f32 { ret(x); }
unsafe extern "sysv64" fn scale(x: f64, n: usize) -> f64 { ret(x); }
unsafe extern "sysv64" fn nothing() {}
unsafe extern "sysv64" fn wipe() {
    let (a, b, c, d, e, f, g, h, i): (gpr, gpr, gpr, gpr, gpr, gpr, gpr, gpr, gpr);
    mov(a, 1); mov(b, 1); mov(c, 1); mov(d, 1); mov(e, 1); mov(f, 1); mov(g, 1); mov(h, 1); mov(i, 1);
    add(a, b); add(a, c); add(a, d); add(a, e); add(a, f); add(a, g); add(a, h); add(a, i);
}
unsafe extern "sysv64" fn writes_rax() { let t: gpr; mov(t, 1); add(t, t); }
unsafe extern "sysv64" fn writes_rax_rcx() { let (t, u): (gpr, gpr); mov(t, 1); mov(u, 2); add(t, u); }
unsafe extern "sysv64" fn writes_ymm0() { let v: ymm; zero(v); }
unsafe extern "sysv64" fn runs_vzeroupper() { let v: ymm; zero(v); vaddps(v, v, v); }
unsafe extern "sysv64" fn calls_vzeroupper() { call(runs_vzeroupper); pause(); }
unsafe extern "sysv64" fn calls_unknown(fp: u64) { call(fp, ()); }
unsafe extern "sysv64" fn calls_via(fp: u64) { call(calls_unknown, (fp)); }
unsafe extern "sysv64" fn passes_ymm(fp: u64) { let v: ymm; zero(v); call(fp, (v)); }
unsafe extern "sysv64" fn jumps() { call(writes_rax); }
unsafe extern "sysv64" fn calls() { call(writes_rax); pause(); }
unsafe extern "sysv64" fn reaches() {
    let fp: gpr; lea(fp, [jumps]); call(fp, (jumps, writes_rax_rcx)); pause();
}
unsafe extern "sysv64" fn ping() { let t: gpr; mov(t, 1); add(t, t); call(pong); }
unsafe extern "sysv64" fn pong() { let (t, u): (gpr, gpr); mov(t, 1); mov(u, 2); add(t, u); call(ping); }
unsafe extern "sysv64" fn broken() { let t: gpr; add(t, t); }
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
fn listing(params: &str, ret: &str, body: &str) -> Result<intel::Listing, String> {
    let src = format!("unsafe extern \"sysv64\" fn k({params}){ret} {{\n{body}\n}}\n{CALLEES}");
    kernite::x86_str(&src, "k")
}

/// The code of [`listing`].
fn calling(params: &str, ret: &str, body: &str) -> Result<String, String> {
    listing(params, ret, body).map(|l| l.code().collect::<Vec<_>>().join("\n"))
}

/// The stack pointer is a multiple of 16 plus 8 on entry, and each push and the frame move it: a kernel that calls
/// moves it 8 more where that leaves it short of a multiple of 16, and so does one with a frame aligned to 16, which
/// lies at a multiple of 16. A frame aligned to 16 or less keeps no `rbp`; one aligned to more keeps the caller's stack
/// pointer in `rbp`, and aligns the stack pointer with `and`.
#[test]
fn the_stack_pointer_is_a_multiple_of_16_at_a_call() -> Result {
    let cases = [
        ("x: u64", "call(nothing); pause();", "sub rsp, 8\ncall {f0}\npause\nadd rsp, 8\nret"),
        ("x: u64", "call(wipe); add(x, x);", "push rbx\nmov rbx, rdi\ncall {f0}\nadd rbx, rbx\npop rbx\nret"),
        (
            "x: u64, y: u64",
            "call(wipe); add(x, y);",
            "push rbx\npush r12\nsub rsp, 8\nmov rbx, rsi\nmov r12, rdi\ncall {f0}\nadd r12, rbx\nadd rsp, 8\npop \
             r12\npop rbx\nret",
        ),
        (
            "x: u64",
            "let n: local<u64>; mov([n], x); call(nothing); mov(x, [n]);",
            "sub rsp, 8\nmov qword ptr [rsp], rdi\ncall {f0}\nmov rdi, qword ptr [rsp]\nadd rsp, 8\nret",
        ),
        (
            "x: u64",
            "let n: local<[u64; 2], 16>; mov([n], x); call(nothing); mov(x, [n]);",
            "sub rsp, 24\nmov qword ptr [rsp], rdi\ncall {f0}\nmov rdi, qword ptr [rsp]\nadd rsp, 24\nret",
        ),
        (
            "x: u64",
            "let n: local<[u64; 2], 16>; mov([n], x); call(wipe); add(x, x);",
            "push rbx\nsub rsp, 16\nmov rbx, rdi\nmov qword ptr [rsp], rbx\ncall {f0}\nadd rbx, rbx\nadd rsp, 16\n\
             pop rbx\nret",
        ),
        (
            "x: u64",
            "let n: local<[u8; 4096], 16>; mov([n], x); call(wipe); add(x, x);",
            "push rbx\nsub rsp, 4096\nmov rbx, rdi\nmov qword ptr [rsp], rbx\ncall {f0}\nadd rbx, rbx\nadd rsp, 4096\n\
             pop rbx\nret",
        ),
        (
            "x: u64",
            "let n: local<u64>; mov([n], x); call(wipe); add(x, x);",
            "push rbx\nsub rsp, 16\nmov rbx, rdi\nmov qword ptr [rsp], rbx\ncall {f0}\nadd rbx, rbx\nadd rsp, 16\n\
             pop rbx\nret",
        ),
        (
            "x: u64",
            "let n: local<[u64; 4], 32>; mov([n], x); call(nothing); mov(x, [n]);",
            "push rbp\nmov rbp, rsp\nsub rsp, 32\nand rsp, -32\nmov qword ptr [rsp], rdi\ncall {f0}\nmov rdi, qword \
             ptr [rsp]\nmov rsp, rbp\npop rbp\nret",
        ),
    ];
    for (params, body, want) in cases {
        assert_eq!(calling(params, "", body)?, want, "{body}");
    }
    Ok(())
}

/// A kernel that calls nothing keeps its frame below the stack pointer, in the red zone that SysV gives a function, and
/// does not set it up or take it down, if the frame, aligned to 16 or less, and the bytes more that align it are 128 or
/// less: a slot is at its offset from as far below the stack pointer as they come to, past the registers the kernel
/// pushes. A frame of more bytes, a page at most, keeps 128 of them there, and the stack pointer moves by the rest, as
/// rustc's moves it. A call that is a jump is no call. A call that stays, even one that a jump would do but for the
/// address the kernel has of its frame, which the callee may use, makes the frame as before, as an alignment past 16
/// does in `only_a_frame_aligned_past_16_keeps_the_stack_pointer_in_rbp`.
#[test]
fn a_kernel_that_calls_nothing_keeps_up_to_128_bytes_of_its_frame_below_the_stack_pointer() {
    // A word of `ty` at `at` written and read back, with the code of a frame below the stack pointer and of one made.
    let word = |ty: &str, at: u32| {
        let body = format!("let b: {ty}; let g: gpr; mov([b + {at}], x); mov(g, [b + {at}]); ret(g);");
        calling("x: u64", " -> u64", &body)
    };
    let below = |disp: &str| format!("mov qword ptr [rsp{disp}], rdi\nmov rax, qword ptr [rsp{disp}]\nret");
    let made = |moved: u32, disp: &str| {
        format!(
            "sub rsp, {moved}\nmov qword ptr [rsp{disp}], rdi\nmov rax, qword ptr [rsp{disp}]\nadd rsp, {moved}\nret"
        )
    };
    let cases = [
        ("local<u64>", 0, below(" - 8")),
        ("local<[u64; 16]>", 0, below(" - 128")),
        ("local<[u64; 16]>", 120, below(" - 8")),
        ("local<[u8; 128]>", 120, below(" - 8")),
        ("local<[u8; 129]>", 120, made(1, " - 8")),
        ("local<[u64; 17]>", 128, made(8, "")),
        ("local<[u64; 32]>", 0, made(128, " - 128")),
        ("local<[u64; 32]>", 248, made(128, " + 120")),
    ];
    for (ty, at, want) in cases {
        assert_eq!(word(ty, at), Ok(want), "{ty} at {at}");
    }

    // A frame aligned to 16 lies at a multiple of 16, with the bytes more that make it so counted in the 128.
    let aligned = |ty: &str, at: u32| {
        let body = format!(
            "let b: {ty}; let v: xmm; vmovups(v, [p]); vmovaps([b + {at}], v); vmovaps(v, [b + {at}]); vmovups([p], v);"
        );
        calling("p: *mut f32", "", &body)
    };
    let moves = |disp: &str| {
        format!(
            "vmovups xmm0, xmmword ptr [rdi]\nvmovaps xmmword ptr [rsp{disp}], xmm0\nvmovaps xmm0, xmmword ptr \
             [rsp{disp}]\nvmovups xmmword ptr [rdi], xmm0"
        )
    };
    let cases = [
        ("local<[f32; 4], 16>", 0, format!("{}\nret", moves(" - 24"))),
        ("local<[f32; 28], 16>", 96, format!("{}\nret", moves(" - 24"))),
        ("local<[f32; 32], 16>", 112, format!("sub rsp, 8\n{}\nadd rsp, 8\nret", moves(" - 16"))),
    ];
    for (ty, at, want) in cases {
        assert_eq!(aligned(ty, at), Ok(want), "{ty} at {at}");
    }

    // The registers the kernel saves are pushed first, below which the red zone lies, and each moves the frame 8 down.
    for (n, pushes, pops, disp) in
        [(9, "push rbx\n", "pop rbx\nret", " - 16"), (10, "push rbx\npush r12\n", "pop r12\npop rbx\nret", " - 24")]
    {
        let body = format!(
            "let b: local<[u64; 2], 16>; let g: [gpr; {n}]; for i in 0..{n} {{ mov(g[i], [p + 8 * i]); }} \
             mov([b], g[0]); for i in 1..{n} {{ add(g[0], g[i]); }} add(g[0], [b]); mov([p], g[0]);"
        );
        let code = calling("p: *mut u64", "", &body);
        let rsp =
            code.as_ref().map(|c| c.lines().filter(|l| l.contains("rsp")).map(str::to_string).collect::<Vec<_>>());
        let want = vec![format!("mov qword ptr [rsp{disp}], rax"), format!("add rax, qword ptr [rsp{disp}]")];
        assert_eq!(rsp, Ok(want), "{n} sums");
        assert!(code.as_ref().is_ok_and(|c| c.starts_with(pushes) && c.ends_with(pops)), "{n} sums: {code:?}");
    }

    let cases = [
        // A call that is a jump is no call: the callee is jumped to, and the frame it may write over is dead.
        (
            "x: u64, fp: u64",
            "",
            "let b: local<u64>; let g: gpr; mov([b], x); mov(g, [b]); call(fp, (g));",
            "mov qword ptr [rsp - 8], rdi\nmov rdi, qword ptr [rsp - 8]\njmp rsi",
        ),
        // A call that stays writes its return address in the red zone.
        (
            "x: u64, fp: u64",
            " -> u64",
            "let b: local<u64>; let g: gpr; mov([b], x); call(fp); mov(g, [b]); ret(g);",
            "sub rsp, 8\nmov qword ptr [rsp], rdi\ncall rsi\nmov rax, qword ptr [rsp]\nadd rsp, 8\nret",
        ),
        // So does one that would be a jump but for the address the kernel gives of its frame.
        (
            "x: u64, fp: u64",
            "",
            "let b: local<u64>; let q: gpr; mov([b], x); lea(q, [b]); call(fp, (q));",
            "sub rsp, 8\nmov qword ptr [rsp], rdi\nlea rdi, [rsp]\ncall rsi\nadd rsp, 8\nret",
        ),
        // An index is added to the slot's address as it is to any.
        (
            "x: u64, i: u64",
            " -> u64",
            "let b: local<[u64; 8]>; let g: gpr; mov([b + i * 8], x); mov(g, [b + i * 8]); ret(g);",
            "mov qword ptr [rsp + rsi*8 - 64], rdi\nmov rax, qword ptr [rsp + rsi*8 - 64]\nret",
        ),
    ];
    for (params, ret, body, want) in cases {
        assert_eq!(calling(params, ret, body), Ok(want.to_string()), "{body}");
    }

    // The listing says where each slot is: the first at the bottom of the frame, with the bytes of alignment above.
    let body = "let a: local<u64>; let b: local<[f32; 4], 16>; let v: xmm; mov([a], x); vmovups(v, [p]); \
                vmovaps([b], v); vmovaps(v, [b]); vmovups([p], v);";
    let text = listing("p: *mut f32, x: u64", "", body).map(|l| l.to_string());
    let frame = "# frame: 32 bytes aligned to 16, saves [], and 8 more so that it is aligned to 16\n\
                 # a: 8 bytes at rsp - 40\n# b: 16 bytes at rsp - 24\n";
    assert!(text.as_ref().is_ok_and(|text| text.contains(frame)), "{text:?}");

    // It says how many bytes of the frame are below the stack pointer: all of a small one, 128 of a bigger, none of one
    // there is not.
    let below = |body: &str| listing("x: u64", "", body).map(|l| l.below);
    assert_eq!(below("let b: local<u64>; mov([b], x);"), Ok(8));
    assert_eq!(below("let b: local<[u8; 129]>; mov([b], x);"), Ok(128));
    assert_eq!(below("pause();"), Ok(0));
}

/// A call that `ret` follows at once is a jump: the kernel takes its frame down as `ret` does, the registers it saved
/// restored and the stack pointer where it was on entry, 8 past a multiple of 16 as a callee has it after a call, and
/// jumps to the callee, which returns to the kernel's caller. It does so by name and at an address, the arguments in
/// their registers before the frame goes, a float's in its xmm register, through the `vzeroupper` of a kernel that may
/// have set an upper half and with none of a kernel that has not, by name and at an address with the functions given;
/// at an address with no functions given, where it passes a float or takes one back, or passes all of a ymm with no
/// `vzeroupper` before the jump, as the callee clears the upper halves before it returns; at an address in a register
/// the kernel saves, which goes to `r11` before the frame goes; with a note between the call and the `ret`; and with a
/// label between, whose `ret` stays for the way that comes to it. A kernel whose every call is a jump does not pad its
/// frame for a call; one with a call that stays pads it, and takes the padding down before each jump.
#[test]
fn a_call_that_ret_follows_is_a_jump_after_the_frame_is_taken_down() -> Result {
    let saving = "let (r, v, fp): (gpr, ymm, gpr); let t: local<[f32; 8], 32>; vmovups(v, [p]); vmovups([t], v); \
                  call(wipe); vmovups(v, [t]); vmovups([p], v);";
    let before = "push rbx\npush r12\npush rbp\nmov rbp, rsp\nsub rsp, 32\nand rsp, -32\nmov rbx, rsi\nmov r12, \
                  rdi\nvmovups ymm0, ymmword ptr [r12]\nvmovups ymmword ptr [rsp], ymm0\ncall {f0}\nvmovups ymm0, \
                  ymmword ptr [rsp]\nvmovups ymmword ptr [r12], ymm0\n";
    let after = "mov rsp, rbp\npop rbp\npop r12\npop rbx\nvzeroupper\n";
    let cases = [
        ("x: u64", "", "call(nothing);".to_string(), "jmp {f0}".to_string()),
        ("x: u64", "", "call(nothing); note(\"it returns to the caller\");".to_string(), "jmp {f0}".to_string()),
        (
            "p: *mut f32, n: u64",
            " -> u64",
            format!("{saving} call(r, gcd, (n, n)); ret(r);"),
            format!("{before}mov rdi, rbx\nmov rsi, rbx\n{after}jmp {{f1}}"),
        ),
        (
            "p: *mut f32, n: u64",
            " -> u64",
            format!("{saving} lea(fp, [gcd]); call(r, fp, (n, n), (gcd, gcd)); ret(r);"),
            format!("{before}lea rax, [rip + {{f1}}]\nmov rdi, rbx\nmov rsi, rbx\n{after}jmp rax"),
        ),
        (
            "x: u64",
            " -> f32",
            "let v: ymm; call(v.xmm, half, 2.5); ret(v.xmm);".to_string(),
            "vmovss xmm0, dword ptr [rip + {c0}]\njmp {f0}".to_string(),
        ),
        (
            "x: u64",
            " -> f32",
            "let v: ymm; zero(v); vaddps(v, v, v); call(v.xmm, half, 2.5); ret(v.xmm);".to_string(),
            "vxorps xmm0, xmm0, xmm0\nvaddps ymm0, ymm0, ymm0\nvmovss xmm0, dword ptr [rip + {c0}]\nvzeroupper\njmp \
             {f0}"
                .to_string(),
        ),
        (
            "x: u64",
            " -> f32",
            "let (v, fp): (ymm, gpr); zero(v); lea(fp, [half]); call(v, fp, (v), (half, half)); ret(v.xmm);"
                .to_string(),
            "vxorps xmm0, xmm0, xmm0\nlea rax, [rip + {f0}]\njmp rax".to_string(),
        ),
        ("fp: u64, x: f32", "", "call(fp, (x));".to_string(), "jmp rdi".to_string()),
        ("fp: u64", " -> f32", "let v: xmm; call(v, fp, ()); ret(v);".to_string(), "jmp rdi".to_string()),
        (
            "fp: u64, p: *const f32",
            "",
            "let v: ymm; vmovups(v, [p]); call(fp, (v));".to_string(),
            "vmovups ymm0, ymmword ptr [rsi]\njmp rdi".to_string(),
        ),
        (
            "fp: u64, p: *const f32, n: u64",
            "",
            "let v: ymm; vmovups(v, [p]); test(n, n); jz(out); call(fp, (v)); label(out); ret();".to_string(),
            "vmovups ymm0, ymmword ptr [rsi]\ntest rdx, rdx\njz 2f\njmp rdi\n2:\nvzeroupper\nret".to_string(),
        ),
        (
            "x: u64",
            " -> u64",
            "let (r, fp): (gpr, gpr); lea(fp, [gcd]); call(r, fp, (x, x)); call(r, fp, (r, r), (gcd, gcd)); ret(r);"
                .to_string(),
            "push rbx\nlea rbx, [rip + {f0}]\nmov rsi, rdi\ncall rbx\nmov rdi, rax\nmov rsi, rax\nmov r11, rbx\npop \
             rbx\njmp r11"
                .to_string(),
        ),
        (
            "x: u64",
            "",
            "test(x, x); jz(out); call(nothing); label(out); ret();".to_string(),
            "test rdi, rdi\njz 2f\njmp {f0}\n2:\nret".to_string(),
        ),
        (
            "x: u64",
            " -> u64",
            "let r: gpr; test(x, x); jz(out); call(r, gcd, (x, x)); ret(r); label(out); call(r, gcd, (x, 1)); \
             add(r, 1); ret(r);"
                .to_string(),
            "sub rsp, 8\ntest rdi, rdi\njz 2f\nmov rsi, rdi\nadd rsp, 8\njmp {f0}\n2:\nmov rsi, 1\ncall {f0}\nadd \
             rax, 1\nadd rsp, 8\nret"
                .to_string(),
        ),
    ];
    for (params, ret, body, want) in cases {
        assert_eq!(calling(params, ret, &body)?, want, "{body}");
    }
    Ok(())
}

/// A call stays one where a jump would not do what it does: where an instruction comes between it and the `ret`, to
/// change what the callee gives back, to return another value, or to restore the MXCSR the kernel keeps; where the
/// kernel has taken the address of its stack memory, which the callee may use, and the frame the jump takes down holds;
/// and where it calls at an address with no functions given and takes all of a ymm back: the kernel's `vzeroupper` runs
/// before the callee sets the upper half of what it gives back. A gpr's `r32` result is one more:
/// `a_call_passes_what_its_callee_takes` has its `mov eax, eax`.
#[test]
fn a_call_stays_one_where_a_jump_would_not_do_what_it_does() -> Result {
    let cases = [
        (
            "x: u64",
            " -> u64",
            "let r: gpr; call(r, gcd, (x, x)); add(r, 1); ret(r);",
            "sub rsp, 8\nmov rsi, rdi\ncall {f0}\nadd rax, 1\nadd rsp, 8\nret",
        ),
        (
            "x: u64",
            " -> u64",
            "call(_, gcd, (x, x)); ret(x);",
            "sub rsp, 8\nmov rsi, rdi\ncall {f0}\nmov rax, rdi\nadd rsp, 8\nret",
        ),
        (
            "x: u64, m: *mut u32",
            " -> u64",
            "let r: gpr; vldmxcsr([m]); call(r, gcd, (x, x)); ret(r);",
            "sub rsp, 8\nvstmxcsr dword ptr [rsp]\nvldmxcsr dword ptr [rsi]\nmov rsi, rdi\ncall {f0}\nvldmxcsr dword \
             ptr [rsp]\nadd rsp, 8\nret",
        ),
        (
            "x: u64",
            " -> u64",
            "let n: local<u64>; let (p, r): (gpr, gpr); lea(p, [n]); call(r, gcd, (p, x)); ret(r);",
            "sub rsp, 8\nmov rsi, rdi\nlea rdi, [rsp]\ncall {f0}\nadd rsp, 8\nret",
        ),
        (
            "x: u64",
            " -> f32",
            "let (v, fp): (ymm, gpr); zero(v); lea(fp, [half]); call(v, fp, (v)); ret(v.xmm);",
            "sub rsp, 8\nvxorps xmm0, xmm0, xmm0\nlea rax, [rip + {f0}]\ncall rax\nadd rsp, 8\nvzeroupper\nret",
        ),
        (
            "fp: u64",
            " -> f32",
            "let v: ymm; call(v, fp, ()); ret(v.xmm);",
            "sub rsp, 8\ncall rdi\nadd rsp, 8\nvzeroupper\nret",
        ),
    ];
    for (params, ret, body, want) in cases {
        assert_eq!(calling(params, ret, body)?, want, "{body}");
    }
    Ok(())
}

/// A gpr's r32 passes its 32 bits, the rest clear, and takes the result's. A float comes back as an xmm, the rest of
/// its ymm clear, as an instruction that writes an xmm leaves it, by name and at an address with no functions given. A
/// function of the module gives back a float, 32 or 64 bits of its xmm, and the kernel reads no more of it, unless it
/// writes the rest first: the move of the result to the register it is in stays only where the kernel reads all of the
/// ymm after, at an address with no functions given, where the call names an xmm. A float known now is read from
/// memory, as x86 takes floats.
#[test]
fn a_call_passes_what_its_callee_takes() -> Result {
    let code = calling("x: u64", " -> u64", "let r: gpr; call(r.r32, gcd, (x.r32, x)); ret(r);")?;
    assert_eq!(code, "sub rsp, 8\nmov rsi, rdi\nmov edi, esi\ncall {f0}\nmov eax, eax\nadd rsp, 8\nret");
    let body = "let v: ymm; vmovups(v, [x]); call(v.xmm, half, (v.xmm)); vbroadcastss(v, v.xmm); vaddps(v, v, v); \
                vmovups([x], v);";
    let want = "sub rsp, 8\nvmovups ymm0, ymmword ptr [rdi]\ncall {f0}\nvbroadcastss ymm0, xmm0\nvaddps ymm0, ymm0, \
                ymm0\nvmovups ymmword ptr [rdi], ymm0\nadd rsp, 8\nvzeroupper\nret";
    assert_eq!(calling("x: *mut f32", "", body)?, want);
    let body = "let v: ymm; call(v.xmm, fp, ()); vaddps(v, v, v); vmovups([x], v);";
    let want = "push rbx\nmov rbx, rdi\ncall rsi\nvmovaps xmm0, xmm0\nvaddps ymm0, ymm0, ymm0\nvmovups ymmword ptr \
                [rbx], ymm0\npop rbx\nvzeroupper\nret";
    assert_eq!(calling("x: *mut f32, fp: u64", "", body)?, want);
    let code = calling("x: *mut f32", "", "let v: ymm; call(v.xmm, half, 2.5); vmovss([x], v.xmm);")?;
    let want = "sub rsp, 8\nvmovss xmm0, dword ptr [rip + {c0}]\ncall {f0}\nvmovss dword ptr [rdi], xmm0\nadd rsp, \
                8\nret";
    assert_eq!(code, want);
    let code = calling("x: *mut f64", "", "let v: ymm; call(v.xmm, scale, (2.5f64, x)); vmovq([x], v.xmm);")?;
    let want = "sub rsp, 8\nvmovq xmm0, qword ptr [rip + {c0}]\ncall {f0}\nvmovq qword ptr [rdi], xmm0\nadd rsp, \
                8\nret";
    assert_eq!(code, want);
    Ok(())
}

/// A call takes the module's functions SysV calls, which pass what it passes and give back
/// what it takes, and a gpr's address of one.
#[test]
fn a_call_is_checked_against_its_callee() -> Result {
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
            "2:36: this passes too many: SysV passes a seventh integer on the stack, which kernite does not read",
        ),
        (
            "let v: ymm; zero(v); let p: gpr; lea(p, [gcd]); call(p, (v, v, v, v, v, v, v, v, v));",
            "2:57: this passes too many: SysV passes a ninth float on the stack, which kernite does not read",
        ),
        ("call(narrow, x);", "2:6: `narrow` does not compile, so no kernel calls it"),
        ("ud2(x);", "2:1: `ud2` takes no operands, not 1"),
    ];
    for (body, want) in cases {
        assert_eq!(calling("x: u64", "", body).err().ok_or("calling gave Ok")?, want, "{body}");
    }
    Ok(())
}

/// A function that never returns, `-> !`, may not, and a cooperative kernel returns. Control does not reach the end of
/// a body that ends in a jump, as one that spins for ever does.
#[test]
fn a_function_that_never_returns_is_checked() -> Result {
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
        assert_eq!(kernite::x86_str(src, "k").err().ok_or("x86_str gave Ok")?, want, "{src}");
    }
    assert_eq!(calling("", " -> !", "label(spin); pause(); jmp(spin);")?, "2:\npause\njmp 2b");
    Ok(())
}

/// A kernel gives back its result as SysV does: an integer, a `bool` or a pointer by `ret(g)` of a gpr, a float by
/// `ret(v)` of a vector, and nothing by `ret()`. Any other `ret` is refused at the `ret`, in a kernel and in a function
/// of the module alike: a vector of any width, or a float, for an integer, a `bool` or a pointer; a gpr for a float; a
/// value for nothing; and none for a value.
#[test]
fn a_ret_of_the_wrong_kind_of_value_is_refused_at_the_ret() {
    let integer = "this kernel returns an integer: `ret(g)`, `g` a gpr";
    let float = "this kernel returns a float: `ret(v)`, `v` a vector";
    let nothing = "this kernel returns nothing: `ret()`";
    // The parameters, the result, a line before the `ret`, what it gives back, and what is said of it.
    let cases = [
        ("", " -> u64", "let v: xmm; zero(v);", "v", integer),
        ("", " -> u64", "let v: ymm; zero(v);", "v", integer),
        ("", " -> u64", "let v: ymm; zero(v);", "v.xmm", integer),
        ("", " -> bool", "let v: ymm; zero(v);", "v", integer),
        ("x: f32", " -> *mut f32", "", "x", integer),
        ("x: f64", " -> u32", "", "x", integer),
        ("", " -> u64", "", "", integer),
        ("n: u64", " -> f32", "", "n", float),
        ("", " -> f64", "let g: gpr; mov(g, 1);", "g", float),
        ("", " -> f32", "", "", float),
        ("", "", "let g: gpr; mov(g, 1);", "g", nothing),
        ("n: u64", "", "", "n", nothing),
        ("x: f32", "", "", "x", nothing),
    ];
    for (params, result, before, given, say) in cases {
        let kernel = format!("unsafe extern \"sysv64\" fn k({params}){result} {{\n{before}\nret({given});\n}}");
        assert_eq!(kernite::x86_str(&kernel, "k").err(), Some(format!("3:1: {say}")), "{kernel}");
        // A function that `k` calls, which takes no parameter for the call to pass, and whose result the call drops.
        if params.is_empty() {
            let call = if result.is_empty() { "call(f);" } else { "call(_, f);" };
            let function = format!(
                "unsafe extern \"sysv64\" fn k() {{\n{call}\n}}\n\
                 unsafe extern \"sysv64\" fn f(){result} {{\n{before}\nret({given});\n}}"
            );
            assert_eq!(kernite::x86_str(&function, "f").err(), Some(format!("6:1: {say}")), "{function}");
        }
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
/// kernite leaves out no move of one to another, as it leaves out a move of a register to itself;
/// memory at `x` of the size the form reaches, a vector of indices beside it; each integer the least the form takes
/// there, or the most; and operand `float`, if there is one, the float parameter `f`, which the kernel does not write
/// first, and with it no vector is read after, as a result may hold the caller's bits of `f`.
fn call(name: &str, form: &Form, most: bool, float: Option<usize>) -> Result<String, String> {
    let mut body = String::new();
    let mut after = String::new();
    let mut args = vec![];
    let store = if name == "vmovups" { "vmovdqu" } else { "vmovups" };
    let test = if name == "test" { "cmp" } else { "test" };
    for (i, spec) in form.operands.iter().enumerate() {
        let mut vector = |bytes: u32| {
            body += &format!("let v{i}: ymm; zero(v{i});\n");
            if float.is_none() {
                after += &format!("\n{store}([x], v{i});");
            }
            if bytes == 32 { format!("v{i}") } else { format!("v{i}.xmm") }
        };
        args.push(match spec.kind {
            Kind::Vec(w) if float == Some(i) => if w.bytes() == 32 { "f.ymm" } else { "f" }.into(),
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
                let (least, most_) =
                    kind.ints().ok_or_else(|| format!("`{name}` has {kind:?}, which has no integers"))?;
                (if most { most_ } else { least }).to_string()
            }
        });
    }
    let end = if form.operands.iter().any(|s| s.kind == Kind::Label) { "\nlabel(end);" } else { "" };
    Ok(format!("{body}{name}({});{after}{end}", args.join(", ")))
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

/// What a program of the toolchain did with a file: whether it succeeded, what it wrote, and the lines of the file it
/// says something of, by number, with what.
type Assembled = (bool, String, Vec<(usize, String)>);

/// `program` run with `args` on `text`, written to the test's directory as `name`.
fn assemble(program: &PathBuf, args: &[&str], name: &str, text: &str) -> Result<Assembled> {
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    std::fs::write(&path, text)?;
    let run = Command::new(program).args(args).arg(&path).output()?;
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
    Ok((run.status.success(), String::from_utf8_lossy(&run.stdout).into_owned(), said))
}

/// GNU as's name for `f`, in `-march`.
fn gas(f: Feature) -> &'static str {
    match f {
        Feature::Bmi1 => "bmi",
        f => f.name(),
    }
}

/// Every form of every instruction kernite knows comes out of the language as a line `llvm-mc`
/// assembles, which assembles a kernel's `naked_asm!` as rustc does, with each integer the
/// least the form takes and the most; and encodes none with EVEX, which only AVX-512 reads,
/// and which `llvm-mc` assembles for any machine. GNU as, which knows what each feature adds,
/// takes each line on a machine of the CPU feature its form names, and refuses it on a machine
/// of every feature before that one; and the listing names the feature.
#[test]
fn every_form_kernite_knows_assembles_on_a_machine_of_its_feature() -> Result {
    let (Some(mc), Some(gnu)) = (tool("LLVM_MC", "llvm-mc"), tool("GNU_AS", "as")) else {
        eprintln!("no llvm-mc or GNU as here: skipped; set LLVM_MC and GNU_AS to run it");
        return Ok(());
    };
    let all: Vec<&Form> = forms::names().flat_map(|n| forms::of(n).unwrap_or_default()).collect();
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let written = Mutex::new(vec![]);
    let done: Vec<_> = std::thread::scope(|s| {
        let spawned: Vec<_> = all
            .chunks(all.len().div_ceil(threads))
            .map(|chunk| {
                let written = &written;
                s.spawn(move || -> Result<(), String> {
                    for form in chunk {
                        let ints = form.operands.iter().any(|s| s.kind.ints().is_some());
                        for most in [false, true].into_iter().take(if ints { 2 } else { 1 }) {
                            let call = call(form.op.name(), form, most, None)?;
                            let src = format!("unsafe extern \"sysv64\" fn k(x: *mut f32) {{\n{call}\n}}\n");
                            let listing = kernite::x86_str(&src, "k").map_err(|e| format!("{e}, of\n{call}"))?;
                            if let Some(f) = form.feature {
                                let features = &listing.features;
                                assert!(features.contains(&f), "no {f:?} in {features:?}, of\n{call}");
                            }
                            let mnemonic = form.op.mnemonic();
                            let line = listing
                                .code()
                                .filter(|l| *l == mnemonic || l.starts_with(&format!("{mnemonic} ")))
                                .last()
                                .ok_or_else(|| format!("no `{mnemonic}` in\n{listing}"))?;
                            written.lock().unwrap_or_else(PoisonError::into_inner).push((
                                form.feature,
                                line.to_string(),
                                call,
                            ));
                        }
                    }
                    Ok(())
                })
            })
            .collect();
        spawned.into_iter().map(|thread| thread.join()).collect()
    });
    for thread in done {
        thread.map_err(|_| "a thread panicked")??;
    }
    let mut written = written.into_inner().unwrap_or_else(PoisonError::into_inner);
    written.sort();

    let lines: Vec<&str> = written.iter().map(|(_, line, _)| line.as_str()).collect();
    let (text, at) = source(&lines);
    let of = |n: usize| at.iter().position(|&a| a == n).map(|i| &written[i]);
    let (ok, out, said) =
        assemble(&mc, &["-triple=x86_64", "-x86-asm-syntax=intel", "-show-encoding"], "x86.s", &text)?;
    let refused: Vec<String> = said
        .iter()
        .map(|(n, why)| match of(*n) {
            Some((_, line, call)) => format!("{line}: {why}, of\n{call}"),
            None => format!("line {n}: {why}"),
        })
        .collect();
    assert!(refused.is_empty() && ok, "llvm-mc refuses what kernite writes:\n\n{}", refused.join("\n\n"));
    let encodings: Vec<&str> = out.lines().filter_map(|l| l.split_once("encoding: [").map(|(_, e)| e)).collect();
    assert_eq!(encodings.len(), lines.len());
    let evex: Vec<_> = lines.iter().zip(&encodings).filter(|(_, e)| e.starts_with("0x62")).map(|(l, _)| l).collect();
    assert!(evex.is_empty(), "kernite writes AVX-512's EVEX: {evex:#?}");

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
        let (ok, _, said) = assemble(&gnu, &["--64", &on, "-o", "/dev/null"], &name, &text)?;
        let refused: Vec<String> = said
            .iter()
            .map(|(n, why)| match of(*n) {
                Some((_, line, call)) => format!("{line}: {why}, of\n{call}"),
                None => format!("line {n}: {why}"),
            })
            .collect();
        assert!(refused.is_empty() && ok, "GNU as {on} says, of what kernite writes:\n\n{}", refused.join("\n\n"));
        if feature.is_none() {
            continue;
        }
        // Every feature before this one, which builds on none after it.
        let before = march(&Feature::ALL[..k - 1]);
        let (_, _, said) = assemble(&gnu, &["--64", &before, "-o", "/dev/null"], &name, &text)?;
        let taken: Vec<&str> = at
            .iter()
            .zip(&lines)
            .filter(|(n, _)| !said.iter().any(|(m, why)| m == *n && why.starts_with("Error")))
            .map(|(_, l)| *l)
            .collect();
        assert!(taken.is_empty(), "GNU as {before} takes what kernite says needs {feature:?}: {taken:#?}");
    }
    Ok(())
}

/// Every instruction of x86-64-v3 that `llvm-mc` takes without AVX-512, kernite knows, or names
/// as one it does not yet write, and why; and kernite knows no other. They are x86-64's own on
/// integers, SSE's to SSE4.2's as AVX writes them, and those of AVX, AVX2, FMA, F16C, BMI1,
/// BMI2, LZCNT, POPCNT and MOVBE; not the stack's, the strings', `loop`, x87's, MMX's or the
/// system's.
#[test]
fn kernite_knows_every_instruction_of_x86_64_v3() {
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

/// A kernel clears the upper halves of the vector registers with `vzeroupper` only where it may have set one: where an
/// instruction of its own reads or writes all of a ymm, or it calls at an address with no functions given and passes
/// all of a ymm or takes all of one back. It clears them before it returns, and before a call that may run SSE code,
/// which pays for an upper half set, as rustc does: after the arguments are moved and what lives across the call is
/// spilled, just before the call. That is a call at an address with no functions given; a call that comes back to the
/// function it is in through another of the module's, which has no listing yet; a call of a function of the module that
/// makes such a call, or calls one that may, by name or at an address with the functions given, for none of the
/// module's clears the upper halves before such a call; and a call of the kernel itself, where one of its other calls
/// is. Any other call of the module's functions has none before it: their instructions are AVX's, and a vector alive
/// across it stays in its register, which `vzeroupper` would clear. Nor has a call that passes all of a ymm or takes
/// all of one back, nor does it give one to a call of the function it is in: its callee runs AVX instructions, and
/// `vzeroupper` would clear the upper half of a ymm it passes. A vector register moves and spills as wide as an
/// instruction names its variable, so a kernel of xmm registers alone writes none, though it moves and spills its
/// floats and calls the module's functions, by name or at an address with the functions given: each clears what it sets
/// before it returns. Nor does a call at an address with no functions given that passes and takes back a float, an xmm:
/// its callee takes and gives back no ymm, and a call that `ret` follows is a jump.
#[test]
fn a_kernel_clears_the_upper_halves_only_where_it_may_have_set_one() -> Result {
    let cases = [
        // No `vzeroupper`: nothing on all of a ymm, and no call of a function it does not know that passes or takes
        // back all of one.
        ("x: f32", " -> f32", "vaddss(x, x, x); ret(x);", "vaddss xmm0, xmm0, xmm0\nret"),
        ("x: u64", " -> f32", "let v: ymm; zero(v); ret(v.xmm);", "vxorps xmm0, xmm0, xmm0\nret"),
        (
            "p: *mut f32",
            "",
            "let v: xmm; vmovups(v, [p]); vaddps(v, v, v); vmovups([p], v);",
            "vmovups xmm0, xmmword ptr [rdi]\nvaddps xmm0, xmm0, xmm0\nvmovups xmmword ptr [rdi], xmm0\nret",
        ),
        (
            "x: f32",
            " -> f32",
            "let v: xmm; call(v, half, (x)); vaddss(v, v, v); ret(v);",
            "sub rsp, 8\ncall {f0}\nvaddss xmm0, xmm0, xmm0\nadd rsp, 8\nret",
        ),
        (
            "x: f32",
            " -> f32",
            "let (v, fp): (xmm, gpr); lea(fp, [half]); call(v, fp, (x), (half, half)); vaddss(v, v, v); ret(v);",
            "sub rsp, 8\nlea rax, [rip + {f0}]\ncall rax\nvaddss xmm0, xmm0, xmm0\nadd rsp, 8\nret",
        ),
        (
            "fp: u64, n: u64",
            " -> u64",
            "let r: gpr; call(r, fp, (n)); add(r, 1); ret(r);",
            "sub rsp, 8\nmov rax, rdi\nmov rdi, rsi\ncall rax\nadd rax, 1\nadd rsp, 8\nret",
        ),
        // A float moves as an xmm: to the register it is passed in, or kept in across a call, and not at all where it
        // is in the register already. The move the kernel writes stays, where it clears an upper half.
        ("a: f32, b: f32", " -> f32", "let v: xmm; call(v, half, (b)); ret(v);", "vmovaps xmm0, xmm1\njmp {f0}"),
        ("x: f32", " -> f32", "let v: xmm; call(v, half, (x)); ret(v);", "jmp {f0}"),
        (
            "x: u64",
            " -> f32",
            "let (a, b): (xmm, xmm); call(a, half, 1.5); call(b, half, 2.5); vaddss(b, a, b); ret(b);",
            "sub rsp, 8\nvmovss xmm0, dword ptr [rip + {c0}]\ncall {f0}\nvmovaps xmm1, xmm0\nvmovss xmm0, dword ptr \
             [rip + {c1}]\ncall {f0}\nvaddss xmm0, xmm1, xmm0\nadd rsp, 8\nret",
        ),
        ("x: f32", " -> f32", "vmovaps(x, x); ret(x);", "vmovaps xmm0, xmm0\nret"),
        // A float spills as an xmm: 16 bytes in a frame aligned to 16.
        (
            "fp: u64, x: f32",
            " -> f32",
            "call(fp); ret(x);",
            "sub rsp, 24\nvmovups xmmword ptr [rsp], xmm0\ncall rdi\nvmovups xmm0, xmmword ptr [rsp]\nadd rsp, 24\nret",
        ),
        (
            "fp: u64, x: f64",
            " -> f64",
            "call(fp); ret(x);",
            "sub rsp, 24\nvmovups xmmword ptr [rsp], xmm0\ncall rdi\nvmovups xmm0, xmmword ptr [rsp]\nadd rsp, 24\nret",
        ),
        // A call at an address with no functions given passes and takes back a float as an xmm, as it moves it: its
        // callee takes and gives back no ymm. A call that `ret` follows is a jump; one that is not writes no
        // `vzeroupper` after it; and the xmm of a ymm is one more.
        ("fp: u64, x: f32", "", "call(fp, (x));", "jmp rdi"),
        ("fp: u64", " -> f32", "let v: xmm; call(v, fp, ()); ret(v);", "jmp rdi"),
        ("fp: u64", " -> f32", "let v: ymm; call(v.xmm, fp, ()); ret(v.xmm);", "jmp rdi"),
        (
            "fp: u64",
            " -> f32",
            "let v: xmm; call(v, fp, ()); vaddss(v, v, v); ret(v);",
            "sub rsp, 8\ncall rdi\nvaddss xmm0, xmm0, xmm0\nadd rsp, 8\nret",
        ),
        ("fp: u64, a: f32, b: f32", " -> f32", "let v: xmm; call(v, fp, (b)); ret(v);", "vmovaps xmm0, xmm1\njmp rdi"),
        (
            "fp: u64, a: f32, b: f32",
            " -> f32",
            "let v: xmm; call(v, fp, (b)); vaddss(v, v, a); ret(v);",
            "sub rsp, 24\nvmovups xmmword ptr [rsp], xmm0\nvmovaps xmm0, xmm1\ncall rdi\nvmovups xmm1, xmmword ptr \
             [rsp]\nvaddss xmm0, xmm0, xmm1\nadd rsp, 24\nret",
        ),
        // `vzeroupper`: an instruction on all of a ymm, which its variable is moved and spilled as; a call at an
        // address with no functions given that gives back all of a ymm, or is passed one and has more after it; and the
        // jump of a kernel that set the upper half of a ymm to a callee that is passed none.
        (
            "p: *mut f32",
            "",
            "let v: ymm; vmovups(v, [p]); vaddps(v, v, v); vmovups([p], v);",
            "vmovups ymm0, ymmword ptr [rdi]\nvaddps ymm0, ymm0, ymm0\nvmovups ymmword ptr [rdi], ymm0\nvzeroupper\n\
             ret",
        ),
        // A read of all of a ymm counts as much as a write, as for rustc: the store of a ymm whose xmm was zeroed.
        (
            "p: *mut f32",
            "",
            "let v: ymm; zero(v); vmovups([p], v);",
            "vxorps xmm0, xmm0, xmm0\nvmovups ymmword ptr [rdi], ymm0\nvzeroupper\nret",
        ),
        (
            "fp: u64, p: *mut f32",
            "",
            "let v: ymm; vmovups(v, [p]); call(fp); vmovups([p], v);",
            "push rbx\npush rbp\nmov rbp, rsp\nsub rsp, 32\nand rsp, -32\nmov rbx, rsi\nvmovups ymm0, ymmword ptr \
             [rbx]\nvmovups ymmword ptr [rsp], ymm0\nvzeroupper\ncall rdi\nvmovups ymm0, ymmword ptr [rsp]\n\
             vmovups ymmword ptr [rbx], ymm0\nmov rsp, rbp\npop rbp\npop rbx\nvzeroupper\nret",
        ),
        (
            "fp: u64",
            " -> f32",
            "let v: ymm; call(v, fp, ()); ret(v.xmm);",
            "sub rsp, 8\ncall rdi\nadd rsp, 8\nvzeroupper\nret",
        ),
        (
            "fp: u64, p: *const f32",
            " -> u64",
            "let (v, r): (ymm, gpr); vmovups(v, [p]); call(r, fp, (v)); add(r, 1); ret(r);",
            "sub rsp, 8\nvmovups ymm0, ymmword ptr [rsi]\ncall rdi\nadd rax, 1\nadd rsp, 8\nvzeroupper\nret",
        ),
        (
            "fp: u64, p: *mut f32, x: f32",
            "",
            "let v: ymm; vmovups(v, [p]); vaddps(v, v, v); vmovups([p], v); call(fp, (x));",
            "vmovups ymm1, ymmword ptr [rsi]\nvaddps ymm1, ymm1, ymm1\nvmovups ymmword ptr [rsi], ymm1\nvzeroupper\n\
             jmp rdi",
        ),
        (
            "fp: u64, p: *mut f32",
            "",
            "let v: ymm; vmovups(v, [p]); vaddps(v, v, v); vmovups([p], v); call(fp, (v.xmm));",
            "vmovups ymm0, ymmword ptr [rsi]\nvaddps ymm0, ymm0, ymm0\nvmovups ymmword ptr [rsi], ymm0\nvzeroupper\n\
             jmp rdi",
        ),
        // A call at an address with no functions given has `vzeroupper` just before it, after the arguments are moved,
        // where the kernel may have set an upper half. One that takes all of a ymm back has none, nor has a call of the
        // module's functions, by name or at an address with them given, the function it is in among them: the
        // `vzeroupper` comes before the `ret` after it.
        (
            "fp: u64, p: *mut f32",
            " -> u64",
            "let v: ymm; let r: gpr; vmovups(v, [p]); vaddps(v, v, v); vmovups([p], v); call(r, fp, (p)); add(r, 1); \
             ret(r);",
            "sub rsp, 8\nmov rax, rdi\nvmovups ymm0, ymmword ptr [rsi]\nvaddps ymm0, ymm0, ymm0\nvmovups ymmword ptr \
             [rsi], ymm0\nmov rdi, rsi\nvzeroupper\ncall rax\nadd rax, 1\nadd rsp, 8\nret",
        ),
        (
            "fp: u64, p: *mut f32, x: f32",
            " -> u64",
            "let (v, r): (ymm, gpr); vmovups(v, [p]); vaddps(v, v, v); vmovups([p], v); call(r, fp, (x)); add(r, 1); \
             ret(r);",
            "sub rsp, 8\nvmovups ymm1, ymmword ptr [rsi]\nvaddps ymm1, ymm1, ymm1\nvmovups ymmword ptr [rsi], \
             ymm1\nvzeroupper\ncall rdi\nadd rax, 1\nadd rsp, 8\nret",
        ),
        (
            "fp: u64, p: *mut f32",
            " -> f32",
            "let v: ymm; vmovups(v, [p]); vaddps(v, v, v); vmovups([p], v); call(v, fp, ()); ret(v.xmm);",
            "sub rsp, 8\nvmovups ymm0, ymmword ptr [rsi]\nvaddps ymm0, ymm0, ymm0\nvmovups ymmword ptr [rsi], \
             ymm0\ncall rdi\nadd rsp, 8\nvzeroupper\nret",
        ),
        (
            "p: *mut f32, n: u64",
            " -> u64",
            "let (v, r): (ymm, gpr); vmovups(v, [p]); vaddps(v, v, v); vmovups([p], v); call(r, gcd, (n, n)); add(r, \
             1); ret(r);",
            "sub rsp, 8\nvmovups ymm0, ymmword ptr [rdi]\nvaddps ymm0, ymm0, ymm0\nvmovups ymmword ptr [rdi], \
             ymm0\nmov rdi, rsi\ncall {f0}\nadd rax, 1\nadd rsp, 8\nvzeroupper\nret",
        ),
        (
            "p: *mut f32, n: u64",
            " -> u64",
            "let (v, r, fp): (ymm, gpr, gpr); vmovups(v, [p]); vaddps(v, v, v); vmovups([p], v); lea(fp, [gcd]); \
             call(r, fp, (n, n), (gcd, gcd)); add(r, 1); ret(r);",
            "sub rsp, 8\nvmovups ymm0, ymmword ptr [rdi]\nvaddps ymm0, ymm0, ymm0\nvmovups ymmword ptr [rdi], \
             ymm0\nlea rax, [rip + {f0}]\nmov rdi, rsi\ncall rax\nadd rax, 1\nadd rsp, 8\nvzeroupper\nret",
        ),
        (
            "p: *mut f32, n: u64",
            " -> u64",
            "let (v, r, fp): (ymm, gpr, gpr); vmovups(v, [p]); vaddps(v, v, v); vmovups([p], v); lea(fp, [k]); \
             call(r, fp, (p, n), (k, k)); add(r, 1); ret(r);",
            "sub rsp, 8\nvmovups ymm0, ymmword ptr [rdi]\nvaddps ymm0, ymm0, ymm0\nvmovups ymmword ptr [rdi], \
             ymm0\nlea rax, [rip + {f0}]\ncall rax\nadd rax, 1\nadd rsp, 8\nvzeroupper\nret",
        ),
        // `vzeroupper` before a call that may run SSE code: of a function of the module that calls at an address with
        // no functions given, or calls one that does, by name or at an address with them given; of one of two that call
        // each other, which have no listing yet; and of the kernel itself, where one of its other calls may. None of
        // the module's clears the upper halves before it does.
        (
            "fp: u64, p: *mut f32",
            "",
            "let v: ymm; vmovups(v, [p]); vaddps(v, v, v); vmovups([p], v); call(calls_unknown, (fp)); pause();",
            "sub rsp, 8\nvmovups ymm0, ymmword ptr [rsi]\nvaddps ymm0, ymm0, ymm0\nvmovups ymmword ptr [rsi], \
             ymm0\nvzeroupper\ncall {f0}\npause\nadd rsp, 8\nret",
        ),
        (
            "fp: u64, p: *mut f32",
            "",
            "let v: ymm; vmovups(v, [p]); vaddps(v, v, v); vmovups([p], v); call(calls_via, (fp)); pause();",
            "sub rsp, 8\nvmovups ymm0, ymmword ptr [rsi]\nvaddps ymm0, ymm0, ymm0\nvmovups ymmword ptr [rsi], \
             ymm0\nvzeroupper\ncall {f0}\npause\nadd rsp, 8\nret",
        ),
        (
            "fp: u64, p: *mut f32",
            "",
            "let (v, q): (ymm, gpr); vmovups(v, [p]); vaddps(v, v, v); vmovups([p], v); lea(q, [calls_unknown]); \
             call(q, (fp), (calls_unknown, calls_unknown)); pause();",
            "sub rsp, 8\nvmovups ymm0, ymmword ptr [rsi]\nvaddps ymm0, ymm0, ymm0\nvmovups ymmword ptr [rsi], \
             ymm0\nlea rax, [rip + {f0}]\nvzeroupper\ncall rax\npause\nadd rsp, 8\nret",
        ),
        (
            "p: *mut f32",
            "",
            "let v: ymm; vmovups(v, [p]); vaddps(v, v, v); vmovups([p], v); call(ping); pause();",
            "sub rsp, 8\nvmovups ymm0, ymmword ptr [rdi]\nvaddps ymm0, ymm0, ymm0\nvmovups ymmword ptr [rdi], \
             ymm0\nvzeroupper\ncall {f0}\npause\nadd rsp, 8\nret",
        ),
        (
            "p: *mut f32",
            "",
            "let (v, q, fp): (ymm, gpr, gpr); vmovups(v, [p]); vaddps(v, v, v); vmovups([p], v); lea(q, [k]); \
             call(q, (p), (k, k)); lea(fp, [gcd]); call(fp, ()); pause();",
            "sub rsp, 8\nvmovups ymm0, ymmword ptr [rdi]\nvaddps ymm0, ymm0, ymm0\nvmovups ymmword ptr [rdi], \
             ymm0\nlea rax, [rip + {f0}]\nvzeroupper\ncall rax\nlea rax, [rip + {f1}]\ncall rax\npause\nadd rsp, \
             8\nret",
        ),
        // None, where the kernel has set no upper half; where the function it calls passes all of a ymm to an address
        // with no functions given: that callee runs AVX instructions, so the function has none before its call; or
        // where a ymm is alive across a call of a function that writes no vector register, though another call of the
        // kernel may run SSE code: `vzeroupper` would clear the upper half of the register that keeps it.
        ("fp: u64", "", "call(calls_unknown, (fp)); pause();", "sub rsp, 8\ncall {f0}\npause\nadd rsp, 8\nret"),
        (
            "fp: u64, p: *mut f32",
            "",
            "let v: ymm; vmovups(v, [p]); vaddps(v, v, v); vmovups([p], v); call(passes_ymm, (fp)); pause();",
            "sub rsp, 8\nvmovups ymm0, ymmword ptr [rsi]\nvaddps ymm0, ymm0, ymm0\nvmovups ymmword ptr [rsi], \
             ymm0\ncall {f0}\npause\nadd rsp, 8\nvzeroupper\nret",
        ),
        (
            "fp: u64, p: *mut f32",
            "",
            "let v: ymm; vmovups(v, [p]); vaddps(v, v, v); call(writes_rax); vmovups([p], v); call(fp, ());",
            "sub rsp, 8\nvmovups ymm0, ymmword ptr [rsi]\nvaddps ymm0, ymm0, ymm0\ncall {f0}\nvmovups ymmword ptr \
             [rsi], ymm0\nadd rsp, 8\nvzeroupper\njmp rdi",
        ),
    ];
    for (params, ret, body, want) in cases {
        assert_eq!(calling(params, ret, body)?, want, "{body}");
    }
    Ok(())
}

/// Where a kernel clears the upper halves of the vector registers depends on the ways to the place: a `vzeroupper` goes
/// before a `ret`, and before a call at an address with no functions given, where some way to it may have set one and
/// none has run `vzeroupper` since. So neither a second call nor a jump to one of the module's functions has one after
/// the first, until the kernel sets one again; a loop that calls before its ymm work clears at the top, for the next
/// round; a `ret` that comes before any ymm work has none, nor has the `ret` that a jump after ymm work leaves for the
/// ways that skip that work; two ways that join, one of which may have set one, clear at the join; and a call that
/// `ret` follows clears after the frame is taken down, before the jump, as `ret` does, though the kernel reloads the
/// address of the call, which it spilled, before the frame goes. A call that is no jump clears after that reload, just
/// before it.
#[test]
fn vzeroupper_goes_where_a_way_to_it_may_have_set_an_upper_half() -> Result {
    let cases = [
        (
            "fp: u64, p: *mut f32",
            " -> u64",
            "let (v, r): (ymm, gpr); vmovups(v, [p]); vaddps(v, v, v); vmovups([p], v); call(fp, ()); call(r, fp, \
             ()); add(r, 1); ret(r);",
            "push rbx\nmov rbx, rdi\nvmovups ymm0, ymmword ptr [rsi]\nvaddps ymm0, ymm0, ymm0\nvmovups ymmword ptr \
             [rsi], ymm0\nvzeroupper\ncall rbx\ncall rbx\nadd rax, 1\npop rbx\nret",
        ),
        (
            "fp: u64, p: *mut f32, n: u64",
            " -> u64",
            "let (v, r): (ymm, gpr); vmovups(v, [p]); vaddps(v, v, v); vmovups([p], v); call(fp, ()); call(r, gcd, (n, \
             n)); ret(r);",
            "push rbx\nmov rbx, rdx\nvmovups ymm0, ymmword ptr [rsi]\nvaddps ymm0, ymm0, ymm0\nvmovups ymmword ptr \
             [rsi], ymm0\nvzeroupper\ncall rdi\nmov rdi, rbx\nmov rsi, rbx\npop rbx\njmp {f0}",
        ),
        (
            "fp: u64, p: *mut f32",
            " -> u64",
            "let (v, r): (ymm, gpr); call(v, fp, ()); vmovups([p], v); call(r, fp, ()); add(r, 1); ret(r);",
            "push rbx\npush r12\nsub rsp, 8\nmov rbx, rsi\nmov r12, rdi\ncall r12\nvmovups ymmword ptr [rbx], \
             ymm0\nvzeroupper\ncall r12\nadd rax, 1\nadd rsp, 8\npop r12\npop rbx\nret",
        ),
        (
            "fp: u64, p: *mut f32, n: u64",
            "",
            "let v: ymm; label(top); call(fp, ()); vmovups(v, [p]); vaddps(v, v, v); vmovups([p], v); dec(n); \
             jnz(top); ret();",
            "push rbx\npush r12\npush r13\nmov rbx, rdx\nmov r12, rsi\nmov r13, rdi\n2:\nvzeroupper\ncall \
             r13\nvmovups ymm0, ymmword ptr [r12]\nvaddps ymm0, ymm0, ymm0\nvmovups ymmword ptr [r12], ymm0\ndec \
             rbx\njnz 2b\npop r13\npop r12\npop rbx\nvzeroupper\nret",
        ),
        (
            "fp: u64, p: *mut f32, n: u64",
            "",
            "let v: ymm; label(top); vmovups(v, [p]); vaddps(v, v, v); vmovups([p], v); dec(n); jnz(top); call(fp, \
             ()); ret();",
            "2:\nvmovups ymm0, ymmword ptr [rsi]\nvaddps ymm0, ymm0, ymm0\nvmovups ymmword ptr [rsi], ymm0\ndec \
             rdx\njnz 2b\nvzeroupper\njmp rdi",
        ),
        (
            "p: *mut f32, n: u64",
            "",
            "let v: ymm; test(n, n); jnz(work); ret(); label(work); vmovups(v, [p]); vaddps(v, v, v); vmovups([p], \
             v); ret();",
            "test rsi, rsi\njnz 2f\nret\n2:\nvmovups ymm0, ymmword ptr [rdi]\nvaddps ymm0, ymm0, ymm0\nvmovups \
             ymmword ptr [rdi], ymm0\nvzeroupper\nret",
        ),
        (
            "p: *mut f32, a: u64, b: u64",
            " -> u64",
            "let (v, r): (ymm, gpr); mov(r, 7); test(a, a); jz(out); vmovups(v, [p]); vaddps(v, v, v); vmovups([p], \
             v); call(r, gcd, (a, b)); label(out); ret(r);",
            "mov rax, 7\ntest rsi, rsi\njz 2f\nvmovups ymm0, ymmword ptr [rdi]\nvaddps ymm0, ymm0, ymm0\nvmovups \
             ymmword ptr [rdi], ymm0\nmov rdi, rsi\nmov rsi, rdx\nvzeroupper\njmp {f0}\n2:\nret",
        ),
        (
            "p: *mut f32, n: u64",
            "",
            "let v: ymm; test(n, n); jz(out); vmovups(v, [p]); vaddps(v, v, v); vmovups([p], v); label(out); ret();",
            "test rsi, rsi\njz 2f\nvmovups ymm0, ymmword ptr [rdi]\nvaddps ymm0, ymm0, ymm0\nvmovups ymmword ptr \
             [rdi], ymm0\n2:\nvzeroupper\nret",
        ),
        (
            "fp: u64, p: *mut f32, n: u64",
            "",
            "let v: ymm; vmovups(v, [p]); vaddps(v, v, v); vmovups([p], v); test(n, n); jz(out); call(fp, ()); \
             label(out); ret();",
            "vmovups ymm0, ymmword ptr [rsi]\nvaddps ymm0, ymm0, ymm0\nvmovups ymmword ptr [rsi], ymm0\ntest rdx, \
             rdx\njz 2f\nvzeroupper\njmp rdi\n2:\nvzeroupper\nret",
        ),
        (
            "fp: u64, p: *mut f32",
            "",
            "let v: ymm; let t: local<[f32; 8], 32>; vmovups(v, [p]); vmovups([t], v); vaddps(v, v, v); vmovups([p], \
             v); call(fp, ());",
            "push rbp\nmov rbp, rsp\nsub rsp, 32\nand rsp, -32\nvmovups ymm0, ymmword ptr [rsi]\nvmovups ymmword ptr \
             [rsp], ymm0\nvaddps ymm0, ymm0, ymm0\nvmovups ymmword ptr [rsi], ymm0\nmov rsp, rbp\npop \
             rbp\nvzeroupper\njmp rdi",
        ),
        (
            "p: *mut f32, a: u64, b: u64, c: u64, d: u64, fp: u64",
            " -> u64",
            "let (v, r): (ymm, gpr); call(fp, ()); vmovups(v, [p]); vaddps(v, v, v); vmovups([p], v); add(a, b); \
             add(a, c); add(a, d); add(b, c); add(b, d); add(c, d); add(a, b); call(r, fp, (a)); ret(r);",
            "push rbx\npush r12\npush r13\npush r14\npush r15\nsub rsp, 16\nmov qword ptr [rsp], r9\nmov rbx, r8\n\
             mov r12, rcx\nmov r13, rdx\nmov r14, rsi\nmov r15, rdi\ncall r9\n\
             vmovups ymm0, ymmword ptr [r15]\nvaddps ymm0, ymm0, ymm0\nvmovups ymmword ptr [r15], ymm0\n\
             add r14, r13\nadd r14, r12\nadd r14, rbx\nadd r13, r12\nadd r13, rbx\nadd r12, rbx\nadd r14, r13\n\
             mov rdi, r14\nmov rax, qword ptr [rsp]\nadd rsp, 16\npop r15\npop r14\npop r13\npop r12\npop rbx\n\
             vzeroupper\njmp rax",
        ),
        (
            "p: *mut f32, a: u64, b: u64, c: u64, d: u64, fp: u64",
            " -> u64",
            "let (v, r): (ymm, gpr); vmovups(v, [p]); vaddps(v, v, v); vmovups([p], v); call(fp, ()); add(a, b); \
             add(a, c); add(a, d); add(b, c); add(b, d); add(c, d); add(a, p); call(r, fp, (a)); ret(r);",
            "push rbx\npush r12\npush r13\npush r14\npush r15\nsub rsp, 16\nmov qword ptr [rsp], r9\nmov rbx, r8\n\
             mov r12, rcx\nmov r13, rdx\nmov r14, rsi\nmov r15, rdi\nvmovups ymm0, ymmword ptr [r15]\n\
             vaddps ymm0, ymm0, ymm0\nvmovups ymmword ptr [r15], ymm0\nvzeroupper\n\
             call r9\nadd r14, r13\nadd r14, r12\nadd r14, rbx\nadd r13, r12\nadd r13, rbx\nadd r12, rbx\nadd r14, \
             r15\nmov rdi, r14\nmov rax, qword ptr [rsp]\nadd rsp, 16\npop r15\npop r14\npop r13\npop r12\npop rbx\n\
             jmp rax",
        ),
        (
            "p: *mut f32, a: u64, b: u64, c: u64, d: u64, fp: u64",
            " -> u64",
            "let v: ymm; call(fp, ()); vmovups(v, [p]); vaddps(v, v, v); vmovups([p], v); add(a, b); add(a, c); \
             add(a, d); add(b, c); add(b, d); add(c, d); call(fp, ()); add(a, p); ret(a);",
            "push rbx\npush r12\npush r13\npush r14\npush r15\nsub rsp, 16\nmov qword ptr [rsp], r9\nmov rbx, r8\n\
             mov r12, rcx\nmov r13, rdx\nmov r14, rsi\nmov r15, rdi\ncall r9\nvmovups ymm0, ymmword ptr [r15]\n\
             vaddps ymm0, ymm0, ymm0\nvmovups ymmword ptr [r15], ymm0\nadd r14, r13\nadd r14, r12\nadd r14, rbx\n\
             add r13, r12\nadd r13, rbx\nadd r12, rbx\nmov rax, qword ptr [rsp]\nvzeroupper\ncall rax\nadd r14, r15\n\
             mov rax, r14\nadd rsp, 16\npop r15\npop r14\npop r13\npop r12\npop rbx\nret",
        ),
    ];
    for (params, ret, body, want) in cases {
        assert_eq!(calling(params, ret, body)?, want, "{body}");
    }
    Ok(())
}

/// A kernel needs what the instructions its listing prints need, and `vzeroupper` is AVX's: a kernel that writes it
/// needs AVX though nothing else it runs does, unless it needs FMA, F16C or AVX2, which build on AVX. One that runs
/// nothing on all of a ymm, nor calls a function it does not know and passes it all of one or takes all of one back,
/// writes none, and needs only what its other instructions need. AVX takes its place among the other needs, which are
/// listed by name.
#[test]
fn vzeroupper_needs_avx_where_nothing_else_says_so() -> Result {
    // What each kernel's own instructions need, with no call, and with a call at an address with no functions given
    // that takes all of a ymm back, so the kernel may have set an upper half.
    let cases = [
        ("", "", "avx"),
        ("popcnt(n, n);", "popcnt", "avx, popcnt"),
        ("vfmadd231ss(x, y, z);", "fma", "fma"),
        ("vcvtph2ps(x, y);", "f16c", "f16c"),
        ("vpbroadcastd(x, y);", "avx2", "avx2"),
        ("popcnt(n, n); vcvtph2ps(x, x); popcnt(n, n);", "f16c, popcnt", "f16c, popcnt"),
    ];
    for (body, plain, called) in cases {
        for (call, needs) in [("", plain), (" let v: ymm; call(v, fp, ());", called)] {
            let src = format!(
                "pub unsafe extern \"sysv64\" fn f(fp: u64, n: u64, x: f64, y: f64, z: f64) {{ {body}{call} }}"
            );
            let listing = kernite::x86_str(&src, "f")?;
            let features: Vec<_> = listing.features.iter().map(|f| f.name()).collect();
            assert_eq!(features.join(", "), needs, "{src}");
            let cleared = listing.code().filter(|line| *line == "vzeroupper").count();
            assert_eq!(cleared, usize::from(!call.is_empty()), "{src}\n{listing}");
        }
    }
    Ok(())
}

/// A kernel needs what the instructions its listing prints need. A move of a register to itself
/// that the listing leaves out needs nothing: `never` passes `x` on in `ymm0`, the register it
/// arrives in, and does not return, so it runs no instruction of AVX's. A move of an xmm to
/// itself clears the upper lanes, so the listing prints it, and `kept` needs AVX for it.
#[test]
fn a_kernels_needs_are_those_of_the_instructions_it_prints() -> Result {
    let src = "pub unsafe extern \"sysv64\" fn die(x: f32) -> ! { ud2(); }
pub unsafe extern \"sysv64\" fn never(x: f32) -> ! { call(die, x); }
pub unsafe extern \"sysv64\" fn kept(x: f32) -> ! { vmovaps(x, x); ud2(); }";
    let want = "# never(vec) -> !
# frame: 0 bytes aligned to 0, saves [], and 8 more so that it calls at a multiple of 16
# x=ymm0
    sub rsp, 8
    call {f0}
# f0: die
";
    assert_eq!(kernite::x86_str(src, "never")?.to_string(), want);
    let want = "# kept(vec) -> !
# needs: avx
# frame: 0 bytes aligned to 0, saves []
# x=ymm0
    vmovaps xmm0, xmm0
    ud2
";
    assert_eq!(kernite::x86_str(src, "kept")?.to_string(), want);
    Ok(())
}
