//! Kernels in the language, compiled to PTX: what they come to, that `ptxas` takes them, and
//! what the errors say; and every form of every instruction kernite knows, which `ptxas` takes
//! at the version and for the GPU kernite says it needs, and not at one older, and the names of
//! them kernite refuses, which `ptxas` refuses too.

use std::collections::BTreeMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use kernite::lang::{IntTy, Mem as Address, Operand as Given, Target, Value};
use kernite::{Assembly, Kept};
use kernite_ptx::forms::{self, Bound, Check, Count, Form, Imm, Role, Sel, Space, Spec};
use kernite_ptx::lang::{Callee, Kernel, Var};
use kernite_ptx::{Entry, Kind, Needs, Prototype, Returns, Static, Ty};

#[path = "common/quotes.rs"]
mod quotes;

const KERNELS: &str = r#"
    /// `x[i] *= a` for each `i < n`, a thread each.
    pub unsafe extern "ptx-kernel" fn scale(x: *mut f32, a: f32, n: u32) {
        let i: u32;
        global_index(i);
        let p: pred;
        setp.ge.u32(p, i, n);
        if p {
            bra(done);
        }
        let at: u64;
        mad.wide.u32(at, i, 4, x);
        let v: f32;
        ld.global.f32(v, [at]);
        mul.rn.f32(v, v, a);
        st.global.f32([at], v);
        label(done);
    }

    /// `y[i] = max(x[i - 1], x[i]) + 0.5` for each `0 < i < n`, and `y[0] = 1`.
    pub unsafe extern "ptx-kernel" fn smooth(x: *const f32, y: *mut f32, n: u32) {
        let i: u32;
        global_index(i);
        let p: pred;
        setp.ge.u32(p, i, n);
        if p {
            ret();
        }
        let (at, out): (u64, u64);
        mad.wide.u32(at, i, 4, x);
        mad.wide.u32(out, i, 4, y);
        let v: f32;
        let first: pred;
        setp.eq.u32(first, i, 0);
        if first {
            mov.f32(v, 1);
        } else {
            let (a, b): (f32, f32);
            ld.global.f32(a, [at - 4]);
            ld.global.f32(b, [at]);
            max.f32(v, a, b);
            add.rn.f32(v, v, 0.5);
        }
        st.global.f32([out], v);
    }

    fn global_index(i: u32) {
        let (block, size, thread): (u32, u32, u32);
        mov.u32(block, ctaid.x);
        mov.u32(size, ntid.x);
        mov.u32(thread, tid.x);
        mad.lo.u32(i, block, size, thread);
    }
"#;

#[test]
fn a_kernel_comes_out_as_ptx_writes_it() {
    let ptx = kernite::ptx_str(KERNELS).unwrap();
    println!("{ptx}");
    let scale = ptx.split(".visible .entry ").nth(1).unwrap();
    let want = "scale(
\t.param .u64 scale_param_0,
\t.param .f32 scale_param_1,
\t.param .u32 scale_param_2
)
{
\t.reg .pred %p;
\t.reg .u32 %n, %i, %block, %size, %thread;
\t.reg .f32 %a, %v;
\t.reg .u64 %x, %at;

\tld.param.u64 %x, [scale_param_0];
\tcvta.to.global.u64 %x, %x;
\tld.param.f32 %a, [scale_param_1];
\tld.param.u32 %n, [scale_param_2];
\tmov.u32 %block, %ctaid.x;
\tmov.u32 %size, %ntid.x;
\tmov.u32 %thread, %tid.x;
\tmad.lo.u32 %i, %block, %size, %thread;
\tsetp.ge.u32 %p, %i, %n;
\t@%p bra $L0;
\tmad.wide.u32 %at, %i, 4, %x;
\tld.global.f32 %v, [%at];
\tmul.rn.f32 %v, %v, %a;
\tst.global.f32 [%at], %v;
$L0:
\tret;
}

";
    assert_eq!(scale, want);
}

/// `ptxas`, if CUDA is here: `$PTXAS`, or on the path, or in a toolkit under `/usr/local`.
fn ptxas() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("PTXAS") {
        return Some(p.into());
    }
    if Command::new("ptxas").arg("--version").output().is_ok_and(|o| o.status.success()) {
        return Some("ptxas".into());
    }
    let mut found: Vec<PathBuf> = std::fs::read_dir("/usr/local")
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path().join("bin/ptxas"))
        .filter(|p| p.exists())
        .collect();
    found.sort();
    found.pop()
}

#[test]
fn ptxas_takes_what_kernite_writes() {
    let Some(ptxas) = ptxas() else {
        eprintln!("no ptxas here: skipped; set PTXAS to run it");
        return;
    };
    for (name, kernels) in [
        ("kernels", KERNELS),
        ("shared", SHARED),
        ("tables", TABLES),
        ("textures", TEXTURES),
        ("surfaces", SURFACES),
        ("calls", CALLS),
        ("instances", INSTANCES),
        ("names", NAMES),
        ("statics", STATICS),
        ("cooperative", COOPERATIVE),
        // Every byte of constant memory, and what it holds, on one line.
        ("constant", "static ALL: constant<[u8; 65536]> = [1; 65536];"),
    ] {
        // ptxas 13.5 cannot size the stack of `calls`, which recurses, and warns of it. That warning is turned off
        // for `calls` alone: any other warning of `calls`, and every warning of the other kernels, is an error.
        let flags: &[&str] = match name {
            "calls" => &["--warning-as-error", "--suppress-stack-size-warning"],
            _ => &["--warning-as-error"],
        };
        let run = assembled(&ptxas, &kernite::ptx_str(kernels).unwrap(), "sm_75", name, flags);
        assert!(run.status.success(), "{name}: {}", String::from_utf8_lossy(&run.stderr));
    }
}

/// The kernel `k` around `body`, compiled, or the error.
fn compile(params: &str, body: &str) -> Result<String, String> {
    kernite::ptx_str(&format!("unsafe extern \"ptx-kernel\" fn k({params}) {{\n{body}\n}}\n"))
}

#[test]
fn a_wrong_operand_is_named_by_what_the_instruction_wants() {
    let cases = [
        ("let i: u32; add.rn.f32(i, x, x);", "2:24: `add.rn.f32` wants a `f32` variable; this is a `u32`"),
        (
            "let v: f32; add.f32(v, v, v);",
            "2:17: `add.f32` leaves ptxas to choose how it rounds, and to fuse it; say so: `add.rn.f32`",
        ),
        (
            "let i: u32; mul.u32(i, n, n);",
            "2:17: `mul.u32` keeps the low half of the product, the high or both: `mul.lo`, `mul.hi`, `mul.wide`",
        ),
        (
            "let v: f32; ld.global.f32(v, [x + 4 * n]);",
            "2:30: `ld.global.f32` takes an address of a variable and a displacement; work out `p + 4 * i` first, as `mad.wide.u32(a, i, 4, p)` does",
        ),
        ("let v: f32; ld.f32(v, [n]);", "2:23: `ld.f32` takes an address in a 64-bit variable; this is a `u32`"),
        ("let v: f32; ld.glob.f32(v, [x]);", "2:21: PTX's `ld` has no `.glob` that kernite knows"),
        ("let i: u32; add.u32(i, tid.x, 1);", "2:28: `add.u32` reads no special register; `mov.u32` does"),
        // `ptxas` 13.5 takes no special register in a `mov` of a float: not `envreg0`, a `b32`, for an `f32`, and not
        // `tid.x` or `gridid`.
        ("let f: f32; mov.f32(f, envreg0);", "2:24: `mov.f32` reads no special register; `mov.b32` does"),
        ("let f: f32; mov.f32(f, tid.x);", "2:28: `mov.f32` reads no special register; `mov.u32` does"),
        ("let d: f64; mov.f64(d, gridid);", "2:24: `mov.f64` reads no special register; `mov.u64` does"),
        // `ptxas` 13.5 takes no integer variable beside a float in a vector of floats, where it takes a special
        // register: "Vector with elements of different types".
        (
            "let (f, i): (f32, u32); mov.f32(f, 1.5); mov.u32(i, n); st.global.v2.f32([x], (f, i));",
            "2:79: `st.global.v2.f32` wants a `f32` variable; this is a `u32`",
        ),
        // A vector that takes no special register, or that the instruction writes, says so, whatever the bits of the
        // register.
        ("let w: b32; mov.b32(w, (tid.x, tid.x));", "2:24: `mov.b32` reads no special register; `mov.u32` does"),
        (
            "wmma.store.d.sync.aligned.row.m16n16k16.global.s32([x], (gridid, n, n, n, n, n, n, n));",
            "2:57: `wmma.store.d.sync.aligned.row.m16n16k16.global.s32` reads no special register; `mov.u64` does",
        ),
        (
            "let (z, b): (b64, b32); mov.b64(z, x); mov.b64((gridid, b), z);",
            "2:48: `mov.b64` writes a variable; `gridid` is a special register, which it reads",
        ),
        ("let i: u32; mov.u32(i, tod.x);", "2:28: `tod.x` is not defined"),
        ("let i: u32; if n { mov.u32(i, 1); }", "2:24: an instruction is guarded by a `pred`; this is a `u32`"),
        // ptxas 13.5 takes an `f16x2` for the predicate that `and`, `or`, `xor`, `not` and `mov` read, but reads its 32
        // bits, where a predicate is one bit: kernite takes it for no predicate.
        (
            "let (h, p): (f16x2, pred); mov.b32(h, n); and.pred(p, h, h);",
            "2:55: `and.pred` wants a `pred` variable; this is a `f16x2`",
        ),
        (
            "let (h, p): (f16x2, pred); mov.b32(h, n); or.pred(p, h, h);",
            "2:54: `or.pred` wants a `pred` variable; this is a `f16x2`",
        ),
        (
            "let (h, p): (f16x2, pred); mov.b32(h, n); xor.pred(p, h, h);",
            "2:55: `xor.pred` wants a `pred` variable; this is a `f16x2`",
        ),
        (
            "let (h, p): (f16x2, pred); mov.b32(h, n); not.pred(p, h);",
            "2:55: `not.pred` wants a `pred` variable; this is a `f16x2`",
        ),
        (
            "let (h, p): (f16x2, pred); mov.b32(h, n); mov.pred(p, h);",
            "2:55: `mov.pred` wants a `pred` variable; this is a `f16x2`",
        ),
        (
            "let (h, i): (f16x2, u32); mov.b32(h, n); if h { mov.u32(i, 1); }",
            "2:53: an instruction is guarded by a `pred`; this is a `f16x2`",
        ),
        // ptxas 13.5 takes an `f16x2` where an instruction wants a `u32` or an `s32`, and kernite does. It takes no
        // integer where an instruction wants an `f16x2`, no `f16x2` where one wants an `f32`, no `f16` where one
        // wants an integer of 16 bits and no integer of 16 bits where one wants an `f16`: kernite takes none.
        (
            "let h: f16x2; mov.b32(h, n); add.rn.f16x2(h, h, n);",
            "2:49: `add.rn.f16x2` wants a `f16x2` variable; this is a `u32`",
        ),
        (
            "let (h, s): (f16x2, s32); mov.b32(h, n); mov.s32(s, 1); add.rn.f16x2(h, h, s);",
            "2:76: `add.rn.f16x2` wants a `f16x2` variable; this is a `s32`",
        ),
        (
            "let (v, h): (f32, f16x2); mov.b32(h, n); add.rn.f32(v, h, h);",
            "2:56: `add.rn.f32` wants a `f32` variable; this is a `f16x2`",
        ),
        (
            "let (a, h): (u16, f16); mov.b16(h, 1); mov.u16(a, 1); add.u16(a, h, h);",
            "2:66: `add.u16` wants a `u16` variable; this is a `f16`",
        ),
        (
            "let (a, h): (u16, f16); mov.b16(h, 1); mov.u16(a, 1); add.rn.f16(h, a, a);",
            "2:69: `add.rn.f16` wants a `f16` variable; this is a `u16`",
        ),
        // ptxas 13.5 takes a `.b32` for an operand of `fma.rn.bf16x2`, which a `tf32` is, and refuses a `.u32`, an
        // `.f32` and an `.f16x2`: kernite takes the `tf32`, as `KEPT` has it, and refuses each of the others.
        (
            "let (b, x): (bf16x2, u32); mov.b32(b, n); mov.u32(x, n); fma.rn.bf16x2(b, x, b, b);",
            "2:75: `fma.rn.bf16x2` wants a `bf16x2` variable; this is a `u32`",
        ),
        (
            "let (b, x): (bf16x2, f32); mov.b32(b, n); mov.f32(x, 1.5); fma.rn.bf16x2(b, x, b, b);",
            "2:77: `fma.rn.bf16x2` wants a `bf16x2` variable; this is a `f32`",
        ),
        (
            "let (b, x): (bf16x2, f16x2); mov.b32(b, n); mov.b32(x, n); fma.rn.bf16x2(b, x, b, b);",
            "2:77: `fma.rn.bf16x2` wants a `bf16x2` variable; this is a `f16x2`",
        ),
        // ptxas 13.5 refuses an `f16x2` for the count of a shift, a result of `clz` or `popc`, an element of a vector
        // beside one of another type, an operand that is negated, the time of `nanosleep.u32`, an operand of
        // `redux.sync` of `.add`, `.min` or `.max`, and the last one of `vadd4.u32.u32.u32.add` and
        // `vsub4.u32.u32.u32.add`: kernite refuses it for each. `ptxas` checks each register of a vector against its
        // neighbour alone, and takes `st.global.v4.u32 [%rd1], {%h1, %b1, %r1, %r1};`, an `f16x2` and a `u32` with a
        // `b32` between, which looks like a bug of its own: kernite refuses that too.
        (
            "let (g, h): (u32, f16x2); mov.b32(h, n); mov.u32(g, 1); shr.u32(g, g, h);",
            "2:71: `shr.u32` wants a `u32` variable; this is a `f16x2`",
        ),
        (
            "let (g, h): (u32, f16x2); mov.b32(h, n); mov.u32(g, 1); clz.b32(h, g);",
            "2:65: `clz.b32` wants a `u32` variable; this is a `f16x2`",
        ),
        (
            "let (g, h): (u32, f16x2); mov.b32(h, n); mov.u32(g, 1); popc.b32(h, g);",
            "2:66: `popc.b32` wants a `u32` variable; this is a `f16x2`",
        ),
        (
            "let (g, h): (u32, f16x2); mov.b32(h, n); mov.u32(g, 1); ld.global.v2.u32((g, h), [x]);",
            "2:78: `ld.global.v2.u32` takes the operands of a vector of one type, bits beside any and an integer \
             beside an integer; this is a `f16x2` variable beside a `u32` variable",
        ),
        (
            "let (g, h): (u32, f16x2); mov.b32(h, n); mov.u32(g, 1); st.global.v2.u32([x], (g, h));",
            "2:83: `st.global.v2.u32` takes the operands of a vector of one type, bits beside any and an integer \
             beside an integer; this is a `f16x2` variable beside a `u32` variable",
        ),
        (
            "let (h, b): (f16x2, b32); mov.b32(h, n); mov.b32(b, n); st.global.v4.u32([x], (h, b, n, n));",
            "2:86: `st.global.v4.u32` takes the operands of a vector of one type, bits beside any and an integer \
             beside an integer; this is a `u32` variable beside a `f16x2` variable",
        ),
        (
            "let (g, h): (u32, f16x2); mov.b32(h, n); mov.u32(g, 1); vmad.u32.u32.u32(g, -h, n, n);",
            "2:77: `vmad.u32.u32.u32` wants a `u32` variable; this is a `f16x2`",
        ),
        (
            "let h: f16x2; mov.b32(h, n); nanosleep.u32(h);",
            "2:44: `nanosleep.u32` wants a `u32` variable; this is a `f16x2`",
        ),
        (
            "let (g, h): (u32, f16x2); mov.b32(h, n); mov.u32(g, 1); redux.sync.add.u32(g, h, 0xffffffff);",
            "2:79: `redux.sync.add.u32` wants a `u32` variable; this is a `f16x2`",
        ),
        (
            "let (g, h): (u32, f16x2); mov.b32(h, n); mov.u32(g, 1); redux.sync.min.s32(g, h, 0xffffffff);",
            "2:79: `redux.sync.min.s32` wants a `s32` variable; this is a `f16x2`",
        ),
        (
            "let (g, h): (u32, f16x2); mov.b32(h, n); mov.u32(g, 1); redux.sync.max.s32(h, g, 0xffffffff);",
            "2:76: `redux.sync.max.s32` wants a `s32` variable; this is a `f16x2`",
        ),
        (
            "let h: f16x2; mov.b32(h, n); bar.warp.sync(h);",
            "2:44: `bar.warp.sync` wants a `u32` variable; this is a `f16x2`",
        ),
        (
            "let (g, h): (u32, f16x2); mov.b32(h, n); mov.u32(g, 1); vadd4.u32.u32.u32.add(g, g, g, h);",
            "2:88: `vadd4.u32.u32.u32.add` wants a `u32` variable; this is a `f16x2`",
        ),
        (
            "let (g, h): (u32, f16x2); mov.b32(h, n); mov.u32(g, 1); vsub4.u32.u32.u32.add(g, g, g, h);",
            "2:88: `vsub4.u32.u32.u32.add` wants a `u32` variable; this is a `f16x2`",
        ),
        // ptxas 13.5 takes `_` for one result of `wmma.mma` of `f16.f32` and of the integers on Turing and on Ampere,
        // of `f32.f32` on Ampere alone, and of `tf32` on none: kernite takes it only where both do.
        (
            "let (a, b): ([f16x2; 8], [f16x2; 8]); let c: [f32; 8]; let d: f32; \
             wmma.load.a.sync.aligned.row.m16n16k16.f16(a, [x]); \
             wmma.load.b.sync.aligned.col.m16n16k16.f16(b, [x], 16); \
             wmma.load.c.sync.aligned.row.m16n16k16.f32(c, [x]); \
             wmma.mma.sync.aligned.row.col.m16n16k16.f32.f32((d, d, d, d, d, d, d, _), a, b, c);",
            "2:276: `wmma.mma.sync.aligned.row.col.m16n16k16.f32.f32` has no result here that it may leave \
             unwritten, `_`",
        ),
        (
            "let (a, b): ([tf32; 4], [tf32; 4]); let c: [f32; 8]; let d: f32; \
             wmma.load.a.sync.aligned.row.m16n16k8.tf32(a, [x]); \
             wmma.load.b.sync.aligned.col.m16n16k8.tf32(b, [x], 16); \
             wmma.load.c.sync.aligned.row.m16n16k8.f32(c, [x]); \
             wmma.mma.sync.aligned.row.col.m16n16k8.f32.tf32.tf32.f32((d, d, d, d, d, d, d, _), a, b, c);",
            "2:282: `wmma.mma.sync.aligned.row.col.m16n16k8.f32.tf32.tf32.f32` has no result here that it may leave \
             unwritten, `_`",
        ),
        ("let v: f32; add.rn.f32(v, v, 1.5);", "2:5: `v` is read before anything writes it"),
        (
            "let v: f32; brx.idx(n, (a, b)); label(a); mov.f32(v, 1); label(b); st.global.f32([x], v);",
            "2:5: `v` is read before anything writes it",
        ),
        (
            "let v: f32; let p: pred; setp.eq.u32(p, n, 0); if p { brx.idx(n, (a,)); } st.global.f32([x], v); \
             label(a); mov.f32(v, 1); st.global.f32([x], v);",
            "2:5: `v` is read before anything writes it",
        ),
        (
            "let v: f128;",
            "2:8: PTX variables are `pred`, `b8` to `b128`, `u8` to `u64`, `s8` to `s64`, `f16`, `f16x2`, `bf16`, \
             `bf16x2`, `tf32`, `f32` and `f64`, not `f128`",
        ),
        ("vfrob.f32(x);", "2:7: PTX has no instruction `vfrob` that kernite knows"),
    ];
    for (body, want) in cases {
        assert_eq!(compile("x: *mut f32, n: u32", body).unwrap_err(), want, "{body}");
    }
}

/// ptxas 13.5 takes an `f16x2` where an instruction of a 32-bit integer type wants a variable, and reads its 32 bits
/// as those of a `b32`, and kernite takes it: its code has the variable as the operand, with no `mov` beside it.
#[test]
fn an_f16x2_is_taken_where_an_instruction_wants_a_32_bit_integer() {
    let cases = [
        ("add.u32(g, h, n);", "add.u32 %g, %h, %n;"),
        ("add.s32(g, h, n);", "add.s32 %g, %h, %n;"),
        ("mul.lo.u32(g, h, n);", "mul.lo.u32 %g, %h, %n;"),
        ("shr.u32(g, h, n);", "shr.u32 %g, %h, %n;"),
        ("mov.u32(g, h);", "mov.u32 %g, %h;"),
        ("setp.lt.u32(p, h, n);", "setp.lt.u32 %p, %h, %n;"),
        // A part of one, as a video instruction selects it.
        ("vadd.u32.u32.u32(g, h.b0, n);", "vadd.u32.u32.u32 %g, %h.b0, %n;"),
        ("vadd2.u32.u32.u32(g.h1, n, h.h21, h);", "vadd2.u32.u32.u32 %g.h1, %n, %h.h21, %h;"),
        // The elements of a vector, beside an `f16x2`, a `b32`, `_` or a special register, as `ptxas` checks that by
        // its size alone, and beside no `u32`, `s32` or value known now.
        ("ld.global.v2.u32((g, h), [x]);", "ld.global.v2.u32 {%g, %h}, [%x];"),
        ("ld.global.v2.u32((h, b), [x]);", "ld.global.v2.u32 {%h, %b}, [%x];"),
        ("ld.global.v2.u32((h, _), [x]);", "ld.global.v2.u32 {%h, _}, [%x];"),
        ("st.global.v2.u32([x], (b, h));", "st.global.v2.u32 [%x], {%b, %h};"),
        ("st.global.v2.u32([x], (h, tid.x));", "st.global.v2.u32 [%x], {%h, %tid.x};"),
        ("st.global.v2.u32([x], (tid.x, h));", "st.global.v2.u32 [%x], {%tid.x, %h};"),
        ("st.global.v2.s32([x], (tid.x, h));", "st.global.v2.s32 [%x], {%tid.x, %h};"),
        ("st.global.v2.u32([x], (h, laneid));", "st.global.v2.u32 [%x], {%h, %laneid};"),
        // The last operand of `vadd4` and `vsub4`, which `ptxas` takes for each name but those of `dp4a`.
        ("vadd4.s32.s32.s32(g, h, n, h);", "vadd4.s32.s32.s32 %g, %h, %n, %h;"),
        ("vadd4.u32.u32.u32.sat(g, h, n, h);", "vadd4.u32.u32.u32.sat %g, %h, %n, %h;"),
        ("vsub4.u32.u32.s32.add(g, h, n, h);", "vsub4.u32.u32.s32.add %g, %h, %n, %h;"),
    ];
    for (inst, line) in cases {
        let body = format!("let (h, g): (f16x2, f16x2); let (b, p): (b32, pred); mov.b32(h, n); mov.b32(b, n); {inst}");
        let ptx = compile("x: *mut u32, n: u32", &body).unwrap();
        assert!(ptx.contains(&format!("\t{line}\n")), "{inst}: {ptx}");
        assert_eq!(ptx.matches("\tmov.").count(), body.matches("mov.").count(), "{inst}: a `mov` beside it: {ptx}");
    }
}

/// An `f16x2` where an instruction wants a 32-bit integer, and where a call wants one or an `f32`, which `ptxas` reads
/// as the 32 bits of a `b32`: passed to a function, written by one, and given back by one. And a `u32` where a call or
/// a `ret` wants an `f16x2`, a `bf16x2` or a `tf32`, whose `.param` is a `.b32`.
const BITS: &str = r#"
    pub unsafe extern "ptx-kernel" fn bits(x: *mut u32, n: u32) {
        let (h, g, i): (f16x2, f16x2, u32);
        let p: pred;
        mov.b32(h, n);
        add.u32(g, h, n);
        add.s32(g, g, n);
        mul.lo.u32(g, g, n);
        shr.u32(g, g, n);
        mov.u32(g, g);
        setp.lt.u32(p, g, n);
        vadd.u32.u32.u32(g, g.b0, n);
        vadd2.u32.u32.u32(g.h1, n, g.h21, g);
        call(g, count, (g));
        call(h, half, (g));
        call(g, pack, (n));
        call(g, pack_s, (g));
        mov.u32(i, n);
        call(i, keep_h, (i));
        call(i, keep_b, (i));
        call(i, keep_t, (i));
        call(i, give_h, (i));
        call(i, give_b, (i));
        call(i, give_t, (i));
        mov.u32(i, g);
        st.global.u32([x], i);
    }

    extern "ptx-func" fn count(a: u32) -> u32 { ret(a); }
    extern "ptx-func" fn half(a: f32) -> f32 { ret(a); }
    extern "ptx-func" fn pack(a: u32) -> u32 { let h: f16x2; mov.b32(h, a); ret(h); }
    extern "ptx-func" fn pack_s(a: s32) -> s32 { let h: f16x2; mov.b32(h, a); ret(h); }
    extern "ptx-func" fn keep_h(a: f16x2) -> f16x2 { ret(a); }
    extern "ptx-func" fn keep_b(a: bf16x2) -> bf16x2 { ret(a); }
    extern "ptx-func" fn keep_t(a: tf32) -> tf32 { ret(a); }
    extern "ptx-func" fn give_h(a: u32) -> f16x2 { ret(a); }
    extern "ptx-func" fn give_b(a: u32) -> bf16x2 { ret(a); }
    extern "ptx-func" fn give_t(a: u32) -> tf32 { ret(a); }
"#;

/// A call takes an `f16x2` for the `.param` of a `u32`, an `s32` or an `f32`, as it takes it for the result, and `ret`
/// gives one back for a `u32` or an `s32`: each writes the variable, with no `mov` beside it. A function of an
/// `f16x2`, a `bf16x2` or a `tf32` takes a `u32` for its argument, writes one with its result, and gives back one
/// with `ret`.
#[test]
fn an_f16x2_is_taken_where_a_call_wants_a_32_bit_integer_or_float() {
    let ptx = kernite::ptx_str(BITS).unwrap();
    for line in [
        "\tcall (%g), count, (%g);\n",
        "\tcall (%h), half, (%g);\n",
        "\tcall (%g), pack, (%n);\n",
        "\tcall (%g), pack_s, (%g);\n",
        "\tst.param.u32 [pack_retval], %h;\n",
        "\tst.param.s32 [pack_s_retval], %h;\n",
        "\tcall (%i), keep_h, (%i);\n",
        "\tcall (%i), keep_b, (%i);\n",
        "\tcall (%i), keep_t, (%i);\n",
        "\tcall (%i), give_h, (%i);\n",
        "\tcall (%i), give_b, (%i);\n",
        "\tcall (%i), give_t, (%i);\n",
        "\tst.param.b32 [give_h_retval], %a;\n",
        "\tst.param.b32 [give_b_retval], %a;\n",
        "\tst.param.b32 [give_t_retval], %a;\n",
    ] {
        assert!(ptx.contains(line), "{line}: {ptx}");
    }
    assert_eq!(ptx.matches("\tmov.").count(), BITS.matches("mov.").count(), "a `mov` beside a call: {ptx}");
}

/// What `ptxas` makes of [`BITS`] is what it makes of the same kernels with a `b32` in place of each `f16x2`,
/// `bf16x2` and `tf32`: the same bytes of a cubin, and so the same machine code.
#[test]
fn ptxas_assembles_an_f16x2_for_a_32_bit_integer_as_a_b32() {
    let Some(ptxas) = ptxas() else {
        eprintln!("no ptxas here: skipped; set PTXAS to run it");
        return;
    };
    let packed = kernite::ptx_str(BITS).unwrap();
    // A register of `b32` needs no PTX 4.2 and `sm_53`, as one of `f16x2` does: its module is put there.
    let twin = BITS.replace("bf16x2", "b32").replace("tf32", "b32").replace("f16x2", "b32");
    let bits = retarget(&kernite::ptx_str(&twin).unwrap(), 42, 53);
    assert_eq!(header(&packed), header(&bits));
    let cubin = |name: &str, text: &str| {
        let run = assembled(&ptxas, text, "sm_75", name, &["--warning-as-error"]);
        assert!(run.status.success(), "{name}: {}", String::from_utf8_lossy(&run.stderr));
        std::fs::read(Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("{name}.cubin"))).unwrap()
    };
    let (of_f16x2, of_b32) = (cubin("bits_f16x2", &packed), cubin("bits_b32", &bits));
    assert!(of_f16x2 == of_b32, "ptxas makes other bytes of an f16x2 than of a b32");
}

/// `ptxas` 13.5 wants the registers of a vector of one size and one type, bits beside any and an integer beside an
/// integer, in `st`, `ld`, `ldu`, `mov`, `suld.b`, `sust.b`, `sust.p`, `ldmatrix` and `mma`: "Vector with elements of
/// different types are not allowed in st instruction", and for registers of two sizes, "Incompatible elements of vector
/// expression". kernite refuses the others, at the first register that does not go with one before it, and names the
/// one it does not go with: `(f, b, i, i)` at the first `i`, which goes with `b` and not with `f`.
#[test]
fn the_registers_of_a_vector_are_of_one_size_and_one_type() {
    let cases = [
        // An `f32` beside a `u32`, in `st`, `ld`, `ldu` and `mov`, packed and unpacked.
        (
            "let (f, i): (f32, u32); mov.f32(f, 1.5); mov.u32(i, n); st.global.v2.b32([x], (f, i));",
            "2:83: `st.global.v2.b32` takes the operands of a vector of one type, bits beside any and an integer \
             beside an integer; this is a `u32` variable beside a `f32` variable",
        ),
        (
            "let (f, i): (f32, u32); mov.f32(f, 1.5); mov.u32(i, n); ld.global.v2.b32((f, i), [x]);",
            "2:78: `ld.global.v2.b32` takes the operands of a vector of one type, bits beside any and an integer \
             beside an integer; this is a `u32` variable beside a `f32` variable",
        ),
        (
            "let (f, i): (f32, u32); mov.f32(f, 1.5); mov.u32(i, n); ldu.global.v2.b32((f, i), [x]);",
            "2:79: `ldu.global.v2.b32` takes the operands of a vector of one type, bits beside any and an integer \
             beside an integer; this is a `u32` variable beside a `f32` variable",
        ),
        (
            "let (f, i): (f32, u32); mov.f32(f, 1.5); mov.u32(i, n); let z: b64; mov.b64(z, (f, i));",
            "2:84: `mov.b64` takes the operands of a vector of one type, bits beside any and an integer beside an \
             integer; this is a `u32` variable beside a `f32` variable",
        ),
        (
            "let (f, i): (f32, u32); mov.f32(f, 1.5); mov.u32(i, n); let z: b64; mov.b64(z, x); mov.b64((f, i), z);",
            "2:96: `mov.b64` takes the operands of a vector of one type, bits beside any and an integer beside an \
             integer; this is a `u32` variable beside a `f32` variable",
        ),
        (
            "let (f, i): (f32, u32); mov.f32(f, 1.5); mov.u32(i, n); st.global.v2.b32([x], (i, f));",
            "2:83: `st.global.v2.b32` takes the operands of a vector of one type, bits beside any and an integer \
             beside an integer; this is a `f32` variable beside a `u32` variable",
        ),
        // An `s32` beside an `f32`, an integer beside a float.
        (
            "let (g, s): (f32, s32); mov.f32(g, 1.5); mov.s32(s, 1); st.global.v2.b32([x], (s, g));",
            "2:83: `st.global.v2.b32` takes the operands of a vector of one type, bits beside any and an integer \
             beside an integer; this is a `f32` variable beside a `s32` variable",
        ),
        // Each register against each before it, not against its neighbour alone, which `ptxas` 13.5 does, and
        // so takes `(f, b, i, i)`, an `f32` and a `u32` with a `b32` between.
        (
            "let (f, i): (f32, u32); mov.f32(f, 1.5); mov.u32(i, n); let b: b32; mov.b32(b, n); st.global.v4.b32([x], \
             (f, b, i, i));",
            "2:113: `st.global.v4.b32` takes the operands of a vector of one type, bits beside any and an integer \
             beside an integer; this is a `u32` variable beside a `f32` variable",
        ),
        (
            "let (f, i): (f32, u32); mov.f32(f, 1.5); mov.u32(i, n); let b: b32; mov.b32(b, n); st.global.v4.b32([x], \
             (i, b, f, b));",
            "2:113: `st.global.v4.b32` takes the operands of a vector of one type, bits beside any and an integer \
             beside an integer; this is a `f32` variable beside a `u32` variable",
        ),
        (
            "let (f, i): (f32, u32); mov.f32(f, 1.5); mov.u32(i, n); let b: b32; mov.b32(b, n); ld.global.v4.b32((b, \
             f, b, i), [x]);",
            "2:111: `ld.global.v4.b32` takes the operands of a vector of one type, bits beside any and an integer \
             beside an integer; this is a `u32` variable beside a `f32` variable",
        ),
        // An `f16x2` beside an `f32`: two types of floats.
        (
            "let (h, f): (f16x2, f32); mov.b32(h, n); mov.f32(f, 1.5); st.global.v2.b32([x], (h, f));",
            "2:85: `st.global.v2.b32` takes the operands of a vector of one type, bits beside any and an integer \
             beside an integer; this is a `f32` variable beside a `f16x2` variable",
        ),
        // Registers of 16 bits and of 64 bits.
        (
            "let (h, a): (f16, u16); mov.b16(h, 1.5f16); mov.u16(a, 1); st.global.v2.b16([x], (h, a));",
            "2:86: `st.global.v2.b16` takes the operands of a vector of one type, bits beside any and an integer \
             beside an integer; this is a `u16` variable beside a `f16` variable",
        ),
        (
            "let (d, u): (f64, u64); mov.f64(d, 1.5f64); mov.u64(u, 1); st.global.v2.b64([x], (d, u));",
            "2:86: `st.global.v2.b64` takes the operands of a vector of one type, bits beside any and an integer \
             beside an integer; this is a `u64` variable beside a `f64` variable",
        ),
        // Registers of two sizes, which `ld` and `st` take for a type of fewer bits than either, wide, but not
        // together.
        (
            "let (a, b): (u16, u32); mov.u16(a, 1); mov.u32(b, n); st.global.v2.u16([x], (a, b));",
            "2:81: `st.global.v2.u16` takes the operands of a vector of one size; this is a `u32` variable beside a \
             `u16` variable",
        ),
        (
            "let (a, b): (u16, u32); ld.global.v2.u16((b, a), [x]);",
            "2:46: `ld.global.v2.u16` takes the operands of a vector of one size; this is a `u16` variable beside a \
             `u32` variable",
        ),
        (
            "let (b, c): (b32, b64); mov.b32(b, n); mov.b64(c, x); st.global.v2.b32([x], (b, c));",
            "2:81: `st.global.v2.b32` takes the operands of a vector of one size; this is a `b64` variable beside a \
             `b32` variable",
        ),
        // The vector of a surface, of `ldmatrix` and of `mma`, which `ptxas` checks as it does `st` and `ld`.
        (
            "let (f, i): (f32, u32); mov.f32(f, 1.5); mov.u32(i, n); sust.b._2d.v2.b32.trap([s, (n, n)], (f, i));",
            "2:97: `sust.b.2d.v2.b32.trap` takes the operands of a vector of one type, bits beside any and an integer \
             beside an integer; this is a `u32` variable beside a `f32` variable",
        ),
        (
            "let (f, i): (f32, u32); mov.f32(f, 1.5); mov.u32(i, n); sust.p._2d.v2.b32.trap([s, (n, n)], (f, i));",
            "2:97: `sust.p.2d.v2.b32.trap` takes the operands of a vector of one type, bits beside any and an integer \
             beside an integer; this is a `u32` variable beside a `f32` variable",
        ),
        (
            "let (f, i): (f32, u32); mov.f32(f, 1.5); mov.u32(i, n); suld.b._2d.v2.b32.trap((f, i), [s, (n, n)]);",
            "2:84: `suld.b.2d.v2.b32.trap` takes the operands of a vector of one type, bits beside any and an integer \
             beside an integer; this is a `u32` variable beside a `f32` variable",
        ),
        (
            "let (f, i): (f32, u32); mov.f32(f, 1.5); mov.u32(i, n); ldmatrix.sync.aligned.m8n8.x2.shared.b16((f, i), \
             [n]);",
            "2:102: `ldmatrix.sync.aligned.m8n8.x2.shared.b16` takes the operands of a vector of one type, bits \
             beside any and an integer beside an integer; this is a `u32` variable beside a `f32` variable",
        ),
        (
            "let (f, i): (f32, u32); mov.f32(f, 1.5); mov.u32(i, n); let (c0, c1, c2, c3): (s32, s32, s32, s32); let \
             b: b32; mov.b32(b, n); mov.s32(c0, 0); mov.s32(c1, 0); mov.s32(c2, 0); mov.s32(c3, 0); \
             mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.xor.popc((c0, c1, c2, c3), (f, i), b, (c0, c1, c2, c3));",
            "2:272: `mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.xor.popc` takes the operands of a vector of one \
             type, bits beside any and an integer beside an integer; this is a `u32` variable beside a `f32` variable",
        ),
    ];
    for (body, want) in cases {
        assert_eq!(compile("x: *mut u32, n: u32, s: u64", body).unwrap_err(), want, "{body}");
    }
}

/// Why the operands of a vector do not go together: what kernite says of it, as the words of its error, and what
/// `ptxas` 13.5 says of the same line of PTX, where it refuses it.
#[derive(Clone, Copy)]
enum Why {
    /// Operands of two sizes: registers, special registers and, in a vector of bits, floats known now, which are of
    /// the bits they are written in, 32 for `1.5` and 64 for `1.5f64`.
    Size,
    /// A float known now first in a vector of bits, before a register of 64 bits, where `ptxas` says it in other
    /// words.
    Order,
    /// Operands of types that do not go together.
    Type,
    /// A part of a vector's special register, as `tid.x` is of `tid`, beside a value known now.
    Part,
    /// A special register of more bits than the operands of a vector of 32 bits: `gridid`, which has 64, where `ptxas`
    /// says it of the instruction, as it does of an operand it does not take.
    Wide,
    /// [`Wide`] after a variable of another size, where `ptxas` says it as it does of a [`Size`]: of the elements of
    /// the vector, and not of the instruction.
    Beside,
}

use Why::{Beside, Order, Part, Size, Type, Wide};

/// What kernite says of the operand that is [`Wide`], as the end of its error.
const GRIDID: &str = "`gridid`, which has 64 bits, and `mov.u32` puts 32 of them in a variable";

impl Why {
    /// What kernite says, after the name of the instruction.
    fn kernite(self) -> &'static str {
        match self {
            Size | Order => "takes the operands of a vector of one size",
            Type => "takes the operands of a vector of one type, bits beside any and an integer beside an integer",
            Part => "takes no part of a special register beside a value known now, as `tid.x` is of `tid`",
            Wide | Beside => "takes the operands of a vector of 32 bits",
        }
    }

    /// What `ptxas` says, or the part of it that tells it from the rest.
    fn ptxas(self) -> &'static str {
        match self {
            Size | Beside => "Incompatible elements of vector expression",
            Order | Wide => "Arguments mismatch for instruction",
            Type => "Vector with elements of different types are not allowed in",
            Part => "must be register or constant",
        }
    }
}

/// A vector whose operands do not go together: the line of the kernel, with `@` where its error points, at the operand,
/// at the `x` of `tid.x`, as an error points at the field of a special register, and at the `[` of the coordinates in a
/// texture or a surface; the line of PTX that kernite would write; why; and the end of the error, after `this is`.
type Apart = (&'static str, &'static str, Why, &'static str);

/// Vectors that kernite and `ptxas` 13.5 both refuse, by the registers of [`TOGETHER_DECLARED`], in each instruction
/// that takes one: a value known now of another kind than a register beside it, or than another value, an integer or a
/// float; a special register of another size, in a vector of floats too; a float known now first in a vector of bits,
/// before a register of 64 bits; a part of a vector's special register, as `tid.x` is of `tid`, beside a value known
/// now, which a whole one, `laneid`, goes beside, in a vector of floats too; and `gridid`, a special register of 64
/// bits, in a vector of 32, in `mov`, `mma`, `wmma`, the coordinates of a surface, and the coordinates and the offsets
/// of a texture, the layer of an array too: before a value known now, beside another `gridid` or a variable, and in
/// `mma` after a value known now. The data of `st` and `sust` take one, [`TOGETHER`].
const BOTH_REFUSE: &[Apart] = &[
    (
        "st.global.v2.b32([x], (f, @1));",
        "st.global.v2.b32 [%x], {%f, 1};",
        Type,
        "the integer 1 beside a `f32` variable",
    ),
    (
        "st.global.v2.b32([x], (1, @f));",
        "st.global.v2.b32 [%x], {1, %f};",
        Type,
        "a `f32` variable beside the integer 1",
    ),
    (
        "st.global.v2.b32([x], (i, @1.5));",
        "st.global.v2.b32 [%x], {%i, 0f3FC00000};",
        Type,
        "the float 1.5 beside a `u32` variable",
    ),
    (
        "st.global.v2.b32([x], (1.5, @i));",
        "st.global.v2.b32 [%x], {0f3FC00000, %i};",
        Type,
        "a `u32` variable beside the float 1.5",
    ),
    (
        "st.global.v2.b32([x], (j, @1.5));",
        "st.global.v2.b32 [%x], {%j, 0f3FC00000};",
        Type,
        "the float 1.5 beside a `s32` variable",
    ),
    (
        "st.global.v2.b32([x], (h, @1.5));",
        "st.global.v2.b32 [%x], {%h, 0f3FC00000};",
        Type,
        "the float 1.5 beside a `f16x2` variable",
    ),
    (
        "st.global.v2.b32([x], (h, @1));",
        "st.global.v2.b32 [%x], {%h, 1};",
        Type,
        "the integer 1 beside a `f16x2` variable",
    ),
    (
        "st.global.v2.b32([x], (1, @1.5));",
        "st.global.v2.b32 [%x], {1, 0f3FC00000};",
        Type,
        "the float 1.5 beside the integer 1",
    ),
    (
        "st.global.v2.b64([x], (d, @1));",
        "st.global.v2.b64 [%x], {%d, 1};",
        Type,
        "the integer 1 beside a `f64` variable",
    ),
    (
        "st.global.v2.b64([x], (u, @1.5f64));",
        "st.global.v2.b64 [%x], {%u, 0d3FF8000000000000};",
        Type,
        "the f64 1.5 beside a `u64` variable",
    ),
    (
        "st.global.v2.b16([x], (l, @1));",
        "st.global.v2.b16 [%x], {%l, 1};",
        Type,
        "the integer 1 beside a `f16` variable",
    ),
    (
        "st.global.v2.b16([x], (l, @1.5f16));",
        "st.global.v2.b16 [%x], {%l, 0x3E00};",
        Type,
        "the f16 1.5, written as the integer 0x3E00, beside a `f16` variable",
    ),
    (
        "st.global.v2.b16([x], (1.5f16, @l));",
        "st.global.v2.b16 [%x], {0x3E00, %l};",
        Type,
        "a `f16` variable beside the f16 1.5, written as the integer 0x3E00",
    ),
    (
        "st.global.v2.b16([x], (l, @1.5bf16));",
        "st.global.v2.b16 [%x], {%l, 0x3FC0};",
        Type,
        "the bf16 1.5, written as the integer 0x3FC0, beside a `f16` variable",
    ),
    ("mov.b64(z, (f, @1));", "mov.b64 %z, {%f, 1};", Type, "the integer 1 beside a `f32` variable"),
    ("mov.b64(z, (1, @f));", "mov.b64 %z, {1, %f};", Type, "a `f32` variable beside the integer 1"),
    ("mov.b64(z, (i, @1.5));", "mov.b64 %z, {%i, 0f3FC00000};", Type, "the float 1.5 beside a `u32` variable"),
    (
        "st.global.v4.b32([x], (f, @1, 2, 3));",
        "st.global.v4.b32 [%x], {%f, 1, 2, 3};",
        Type,
        "the integer 1 beside a `f32` variable",
    ),
    (
        "st.global.v4.b32([x], (1, 2, 3, @f));",
        "st.global.v4.b32 [%x], {1, 2, 3, %f};",
        Type,
        "a `f32` variable beside the integer 1",
    ),
    (
        "sust.b._2d.v2.b32.trap([s, (n, n)], (f, @1));",
        "sust.b.2d.v2.b32.trap [%s, {%n, %n}], {%f, 1};",
        Type,
        "the integer 1 beside a `f32` variable",
    ),
    // An `f16x2` for an integer, which `ptxas` takes beside an `f16x2`, a `b32`, `_` and a special register alone.
    (
        "st.global.v2.u32([x], (h, @i));",
        "st.global.v2.u32 [%x], {%h, %i};",
        Type,
        "a `u32` variable beside a `f16x2` variable",
    ),
    (
        "st.global.v2.b32([x], (h, @i));",
        "st.global.v2.b32 [%x], {%h, %i};",
        Type,
        "a `u32` variable beside a `f16x2` variable",
    ),
    (
        "st.global.v2.u32([x], (h, @j));",
        "st.global.v2.u32 [%x], {%h, %j};",
        Type,
        "a `s32` variable beside a `f16x2` variable",
    ),
    (
        "st.global.v2.u32([x], (h, @1));",
        "st.global.v2.u32 [%x], {%h, 1};",
        Type,
        "the integer 1 beside a `f16x2` variable",
    ),
    // A special register of another size, which `ptxas` checks by its size alone.
    (
        "st.global.v2.b32([x], (u, tid.@x));",
        "st.global.v2.b32 [%x], {%u, %tid.x};",
        Size,
        "`tid.x` beside a `u64` variable",
    ),
    (
        "st.global.v2.b32([x], (tid.x, @u));",
        "st.global.v2.b32 [%x], {%tid.x, %u};",
        Size,
        "a `u64` variable beside `tid.x`",
    ),
    (
        "st.global.v2.u32([x], (u, tid.@x));",
        "st.global.v2.u32 [%x], {%u, %tid.x};",
        Size,
        "`tid.x` beside a `u64` variable",
    ),
    (
        "st.global.v2.b16([x], (a, tid.@x));",
        "st.global.v2.b16 [%x], {%a, %tid.x};",
        Size,
        "`tid.x` beside a `u16` variable",
    ),
    (
        "st.global.v2.b32([x], (d, @laneid));",
        "st.global.v2.b32 [%x], {%d, %laneid};",
        Size,
        "`laneid` beside a `f64` variable",
    ),
    (
        "st.global.v2.f32([x], (f, @gridid));",
        "st.global.v2.f32 [%x], {%f, %gridid};",
        Size,
        "`gridid` beside a `f32` variable",
    ),
    // A float known now first in a vector of bits, beside a register of 64 bits.
    (
        "st.global.v2.b32([x], (1.5, @d));",
        "st.global.v2.b32 [%x], {0f3FC00000, %d};",
        Order,
        "a `f64` variable beside the float 1.5",
    ),
    (
        "st.global.v2.b32([x], (1.5, @z));",
        "st.global.v2.b32 [%x], {0f3FC00000, %z};",
        Order,
        "a `b64` variable beside the float 1.5",
    ),
    // A part of a vector's special register beside a value known now, whichever of them is first and whichever the
    // part, in each instruction that takes a vector.
    (
        "st.global.v2.b32([x], (tid.x, @1));",
        "st.global.v2.b32 [%x], {%tid.x, 1};",
        Part,
        "the integer 1 beside `tid.x`",
    ),
    (
        "st.global.v2.b32([x], (1, tid.@x));",
        "st.global.v2.b32 [%x], {1, %tid.x};",
        Part,
        "`tid.x` beside the integer 1",
    ),
    (
        "st.global.v2.b32([x], (tid.x, @1.5));",
        "st.global.v2.b32 [%x], {%tid.x, 0f3FC00000};",
        Part,
        "the float 1.5 beside `tid.x`",
    ),
    (
        "st.global.v2.b32([x], (1.5, tid.@x));",
        "st.global.v2.b32 [%x], {0f3FC00000, %tid.x};",
        Part,
        "`tid.x` beside the float 1.5",
    ),
    (
        "st.global.v2.u32([x], (tid.x, @1));",
        "st.global.v2.u32 [%x], {%tid.x, 1};",
        Part,
        "the integer 1 beside `tid.x`",
    ),
    (
        "st.global.v2.b32([x], (ntid.x, @1));",
        "st.global.v2.b32 [%x], {%ntid.x, 1};",
        Part,
        "the integer 1 beside `ntid.x`",
    ),
    (
        "st.global.v2.b32([x], (ctaid.y, @1));",
        "st.global.v2.b32 [%x], {%ctaid.y, 1};",
        Part,
        "the integer 1 beside `ctaid.y`",
    ),
    (
        "st.global.v2.b32([x], (1, nctaid.@z));",
        "st.global.v2.b32 [%x], {1, %nctaid.z};",
        Part,
        "`nctaid.z` beside the integer 1",
    ),
    (
        "st.global.v4.b32([x], (tid.x, tid.y, tid.z, @1));",
        "st.global.v4.b32 [%x], {%tid.x, %tid.y, %tid.z, 1};",
        Part,
        "the integer 1 beside `tid.x`",
    ),
    ("mov.b64(z, (tid.x, @1));", "mov.b64 %z, {%tid.x, 1};", Part, "the integer 1 beside `tid.x`"),
    ("mov.b64(z, (tid.x, @1.5));", "mov.b64 %z, {%tid.x, 0f3FC00000};", Part, "the float 1.5 beside `tid.x`"),
    (
        "mma.sync.aligned.m16n8k16.row.col.s32.u8.u8.s32((j, j, j, j), (tid.x, @8), n, (0, 0, 0, 0));",
        "mma.sync.aligned.m16n8k16.row.col.s32.u8.u8.s32 {%j, %j, %j, %j}, {%tid.x, 8}, {%n}, {0, 0, 0, 0};",
        Part,
        "the integer 8 beside `tid.x`",
    ),
    (
        "sust.b._2d.v2.b32.trap([s, (n, n)], (tid.x, @1));",
        "sust.b.2d.v2.b32.trap [%s, {%n, %n}], {%tid.x, 1};",
        Part,
        "the integer 1 beside `tid.x`",
    ),
    (
        "sust.b._2d.v2.b32.trap(@[s, (tid.x, 5)], (f, b));",
        "sust.b.2d.v2.b32.trap [%s, {%tid.x, 5}], {%f, %b};",
        Part,
        "the integer 5 beside `tid.x`",
    ),
    (
        "tex._2d.v4.f32.s32((r0, r1, r2, r3), @[s, (tid.x, 5)]);",
        "tex.2d.v4.f32.s32 {%r0, %r1, %r2, %r3}, [%s, {%tid.x, 5}];",
        Part,
        "the integer 5 beside `tid.x`",
    ),
    (
        "st.global.v2.f32([x], (tid.x, @1.5));",
        "st.global.v2.f32 [%x], {%tid.x, 0f3FC00000};",
        Part,
        "the float 1.5 beside `tid.x`",
    ),
    (
        "st.global.v2.f32([x], (1.5, tid.@x));",
        "st.global.v2.f32 [%x], {0f3FC00000, %tid.x};",
        Part,
        "`tid.x` beside the float 1.5",
    ),
    (
        "tex._2d.v4.f32.f32((r0, r1, r2, r3), @[s, (tid.x, 1.5)]);",
        "tex.2d.v4.f32.f32 {%r0, %r1, %r2, %r3}, [%s, {%tid.x, 0f3FC00000}];",
        Part,
        "the float 1.5 beside `tid.x`",
    ),
    ("mov.b64(z, (@gridid, gridid));", "mov.b64 %z, {%gridid, %gridid};", Wide, GRIDID),
    ("mov.b64(z, (@gridid, 1));", "mov.b64 %z, {%gridid, 1};", Wide, GRIDID),
    ("mov.b64(z, (@gridid, -1));", "mov.b64 %z, {%gridid, -1};", Wide, GRIDID),
    (
        "mma.sync.aligned.m16n8k16.row.col.s32.u8.u8.s32((j, j, j, j), (@gridid, gridid), n, (0, 0, 0, 0));",
        "mma.sync.aligned.m16n8k16.row.col.s32.u8.u8.s32 {%j, %j, %j, %j}, {%gridid, %gridid}, {%n}, {0, 0, 0, 0};",
        Wide,
        GRIDID,
    ),
    (
        "mma.sync.aligned.m16n8k16.row.col.s32.u8.u8.s32((j, j, j, j), (1, @gridid), n, (0, 0, 0, 0));",
        "mma.sync.aligned.m16n8k16.row.col.s32.u8.u8.s32 {%j, %j, %j, %j}, {1, %gridid}, {%n}, {0, 0, 0, 0};",
        Wide,
        GRIDID,
    ),
    (
        "wmma.mma.sync.aligned.row.col.m16n16k16.s32.s8.s8.s32((j, j, j, j, j, j, j, j), (n, n), (n, n), \
         (@gridid, 1, 1, 1, 1, 1, 1, 1));",
        "wmma.mma.sync.aligned.row.col.m16n16k16.s32.s8.s8.s32 {%j, %j, %j, %j, %j, %j, %j, %j}, {%n, %n}, {%n, %n}, \
         {%gridid, 1, 1, 1, 1, 1, 1, 1};",
        Wide,
        GRIDID,
    ),
    (
        "sust.b._2d.v2.b32.trap(@[s, (gridid, gridid)], (f, b));",
        "sust.b.2d.v2.b32.trap [%s, {%gridid, %gridid}], {%f, %b};",
        Wide,
        GRIDID,
    ),
    (
        "sust.b._2d.v2.b32.trap(@[s, (gridid, 1)], (f, b));",
        "sust.b.2d.v2.b32.trap [%s, {%gridid, 1}], {%f, %b};",
        Wide,
        GRIDID,
    ),
    (
        "suld.b._2d.v2.b32.trap((b, b), @[s, (gridid, gridid)]);",
        "suld.b.2d.v2.b32.trap {%b, %b}, [%s, {%gridid, %gridid}];",
        Wide,
        GRIDID,
    ),
    (
        "sured.b.add._2d.u32.trap(@[s, (gridid, 1)], n);",
        "sured.b.add.2d.u32.trap [%s, {%gridid, 1}], %n;",
        Wide,
        GRIDID,
    ),
    (
        "tex._2d.v4.f32.s32((r0, r1, r2, r3), @[s, (gridid, gridid)]);",
        "tex.2d.v4.f32.s32 {%r0, %r1, %r2, %r3}, [%s, {%gridid, %gridid}];",
        Wide,
        GRIDID,
    ),
    (
        "tex._2d.v4.f32.s32((r0, r1, r2, r3), @[s, (gridid, 1)]);",
        "tex.2d.v4.f32.s32 {%r0, %r1, %r2, %r3}, [%s, {%gridid, 1}];",
        Wide,
        GRIDID,
    ),
    (
        "tex._1d.v4.f32.s32((r0, r1, r2, r3), @[s, gridid]);",
        "tex.1d.v4.f32.s32 {%r0, %r1, %r2, %r3}, [%s, {%gridid}];",
        Wide,
        GRIDID,
    ),
    (
        "tex._2d.v4.f32.s32((r0, r1, r2, r3), [s, (j, j)], (@gridid, 1));",
        "tex.2d.v4.f32.s32 {%r0, %r1, %r2, %r3}, [%s, {%j, %j}], {%gridid, 1};",
        Wide,
        GRIDID,
    ),
    ("mov.b64(z, (@gridid, i));", "mov.b64 %z, {%gridid, %i};", Wide, GRIDID),
    ("mov.b64(z, (i, @gridid));", "mov.b64 %z, {%i, %gridid};", Beside, GRIDID),
    (
        "tex.a1d.v4.f32.s32((r0, r1, r2, r3), @[s, (gridid, j)]);",
        "tex.a1d.v4.f32.s32 {%r0, %r1, %r2, %r3}, [%s, {%gridid, %j}];",
        Wide,
        GRIDID,
    ),
    (
        "tex.a1d.v4.f32.f32((r0, r1, r2, r3), @[s, (gridid, f)]);",
        "tex.a1d.v4.f32.f32 {%r0, %r1, %r2, %r3}, [%s, {%gridid, %f}];",
        Wide,
        GRIDID,
    ),
    (
        "tex.a1d.v4.f32.s32((r0, r1, r2, r3), @[s, (n, gridid)]);",
        "tex.a1d.v4.f32.s32 {%r0, %r1, %r2, %r3}, [%s, {%n, %gridid}];",
        Beside,
        GRIDID,
    ),
];

/// Vectors that `ptxas` 13.5 takes and kernite refuses, by one of two bugs of its own. It checks each operand against
/// the one before it alone: `(f, b, 1, 1)` has an `f32` and an integer with a `b32` between, as `(f, b, i, i)` has an
/// `f32` and a `u32`, and `(f, tid.x, i, i)` a special register, which `ptxas` checks by its size alone. A float known
/// now in a vector of bits beside a register of 64 bits, `ptxas` refuses first, [`Order`], and takes elsewhere, after
/// one and before one: kernite refuses it in every place. `gridid`, a special register of 64 bits, `ptxas` takes after
/// a value known now in a vector of 32 bits, `(1, gridid)`, in `mov`, in the accumulator of `mma`, and in coordinates
/// and offsets, as it checks the operand against its neighbour, `1`, alone: kernite refuses it, as it does
/// `(gridid, 1)`.
const ONLY_KERNITE_REFUSES: &[Apart] = &[
    (
        "st.global.v4.b32([x], (f, b, @1, 1));",
        "st.global.v4.b32 [%x], {%f, %b, 1, 1};",
        Type,
        "the integer 1 beside a `f32` variable",
    ),
    (
        "st.global.v4.b32([x], (f, tid.x, @i, i));",
        "st.global.v4.b32 [%x], {%f, %tid.x, %i, %i};",
        Type,
        "a `u32` variable beside a `f32` variable",
    ),
    (
        "st.global.v2.b32([x], (d, @1.5));",
        "st.global.v2.b32 [%x], {%d, 0f3FC00000};",
        Size,
        "the float 1.5 beside a `f64` variable",
    ),
    (
        "st.global.v2.b32([x], (z, @1.5));",
        "st.global.v2.b32 [%x], {%z, 0f3FC00000};",
        Size,
        "the float 1.5 beside a `b64` variable",
    ),
    (
        "st.global.v4.b32([x], (d, d, @1.5, d));",
        "st.global.v4.b32 [%x], {%d, %d, 0f3FC00000, %d};",
        Size,
        "the float 1.5 beside a `f64` variable",
    ),
    ("mov.b64(z, (1, @gridid));", "mov.b64 %z, {1, %gridid};", Wide, GRIDID),
    (
        "sust.b._2d.v2.b32.trap(@[s, (1, gridid)], (f, b));",
        "sust.b.2d.v2.b32.trap [%s, {1, %gridid}], {%f, %b};",
        Wide,
        GRIDID,
    ),
    (
        "tex._2d.v4.f32.s32((r0, r1, r2, r3), @[s, (1, gridid)]);",
        "tex.2d.v4.f32.s32 {%r0, %r1, %r2, %r3}, [%s, {1, %gridid}];",
        Wide,
        GRIDID,
    ),
    (
        "tex._2d.v4.f32.s32((r0, r1, r2, r3), [s, (j, j)], (1, @gridid));",
        "tex.2d.v4.f32.s32 {%r0, %r1, %r2, %r3}, [%s, {%j, %j}], {1, %gridid};",
        Wide,
        GRIDID,
    ),
    (
        "mma.sync.aligned.m16n8k16.row.col.s32.u8.u8.s32((j, j, j, j), (i, i), n, (1, @gridid, gridid, gridid));",
        "mma.sync.aligned.m16n8k16.row.col.s32.u8.u8.s32 {%j, %j, %j, %j}, {%i, %i}, {%n}, \
         {1, %gridid, %gridid, %gridid};",
        Wide,
        GRIDID,
    ),
];

/// The kernel `line`, a vector of a table such as [`BOTH_REFUSE`], without the `@` that marks where its error points:
/// the line of the body it is in, after the variables of [`TOGETHER_DECLARED`]; the column its error points at; and the
/// name of its instruction.
fn apart(line: &str) -> (String, usize, String) {
    let (before, after) = line.split_once('@').unwrap();
    let kernel = format!("{before}{after}");
    let body = format!("{TOGETHER_DECLARED} {kernel}");
    let name = kernel[..kernel.find('(').unwrap()].replace("._", ".");
    let col = body.len() - after.len() + 1;
    (body, col, name)
}

/// kernite refuses each vector of [`BOTH_REFUSE`] and [`ONLY_KERNITE_REFUSES`] at the operand that does not go in it,
/// in the words of why, and names it: beside the first operand before it that it does not go with, or with its bits.
#[test]
fn an_operand_that_does_not_go_in_its_vector_is_refused_where_ptxas_refuses_it() {
    for &(line, _, why, this) in BOTH_REFUSE.iter().chain(ONLY_KERNITE_REFUSES) {
        let (body, col, name) = apart(line);
        let want = format!("2:{col}: `{name}` {}; this is {this}", why.kernite());
        assert_eq!(compile("x: *mut u32, n: u32, s: u64", &body).unwrap_err(), want, "{line}");
    }
}

/// The module of [`TOGETHER_DECLARED`] alone, of the newest PTX for `sm_86`, with the line of PTX `line` in it.
fn with_line(line: &str) -> String {
    let ptx = compile("x: *mut u32, n: u32, s: u64", TOGETHER_DECLARED).unwrap();
    retarget(&ptx, NEWEST, 86).replace("\tret;\n", &format!("\t{line}\n\tret;\n"))
}

/// `ptxas` refuses each line of [`BOTH_REFUSE`], in the words that tell why, and takes the module without it.
#[test]
fn ptxas_refuses_each_vector_that_kernite_refuses_with_it() {
    let Some(ptxas) = ptxas() else {
        eprintln!("no ptxas here: skipped; set PTXAS to run it");
        return;
    };
    let alone = assembled(&ptxas, &with_line(""), "sm_86", "apart_alone", &[]);
    assert!(alone.status.success(), "{}", String::from_utf8_lossy(&alone.stderr));
    for (n, &(_, line, why, _)) in BOTH_REFUSE.iter().enumerate() {
        let run = assembled(&ptxas, &with_line(line), "sm_86", &format!("apart_{n}"), &[]);
        let said = String::from_utf8_lossy(&run.stderr);
        assert!(!run.status.success() && said.contains(why.ptxas()), "{line}: {said}");
    }
}

/// `ptxas` 13.5 takes each vector of [`ONLY_KERNITE_REFUSES`], by one of two bugs of its own: it checks each operand
/// of a vector against its neighbour alone, and it refuses a float known now beside a register of 64 bits in a vector
/// of bits only where the float is first. It fails the day `ptxas` refuses one, for GUIDE.md to say no more of that
/// bug.
#[test]
fn ptxas_takes_each_vector_that_only_kernite_refuses() {
    let Some(ptxas) = ptxas() else {
        eprintln!("no ptxas here: skipped; set PTXAS to run it");
        return;
    };
    for (n, &(_, line, _, _)) in ONLY_KERNITE_REFUSES.iter().enumerate() {
        let run = assembled(&ptxas, &with_line(line), "sm_86", &format!("only_kernite_{n}"), &[]);
        assert!(run.status.success(), "{line}: {}", String::from_utf8_lossy(&run.stderr));
    }
}

/// A special register of bits that a vector of floats does not take, beside a float known now, before it and after it:
/// `gridid`, of 64 bits, in the coordinates of `tex`, which take a register of as many bits as the float, and `laneid`,
/// of 32, in the data of `st.global.v2.f64`, which takes one of no fewer. Each is a kernel, with `@` where its error
/// points; the line of PTX kernite would write; whether `ptxas` 13.5 takes it; and the end of the error. `ptxas` checks
/// the register against its neighbour alone, a float, which goes beside any register, and takes it only where the float
/// is first, as it takes `gridid` after a value known now in a vector of 32 bits, `(1, gridid)`: kernite refuses both.
const FLOAT_FIRST: [(&str, &str, bool, &str); 4] = [
    (
        "tex._2d.v4.f32.f32((r0, r1, r2, r3), @[s, (1.5, gridid)]);",
        "tex.2d.v4.f32.f32 {%r0, %r1, %r2, %r3}, [%s, {0f3FC00000, %gridid}];",
        true,
        "reads a `f32`; `gridid` is a `u64`",
    ),
    (
        "tex._2d.v4.f32.f32((r0, r1, r2, r3), @[s, (gridid, 1.5)]);",
        "tex.2d.v4.f32.f32 {%r0, %r1, %r2, %r3}, [%s, {%gridid, 0f3FC00000}];",
        false,
        "reads a `f32`; `gridid` is a `u64`",
    ),
    (
        "st.global.v2.f64([x], @(1.5f64, laneid));",
        "st.global.v2.f64 [%x], {0d3FF8000000000000, %laneid};",
        true,
        "reads a `f64`; `laneid` is a `u32`",
    ),
    (
        "st.global.v2.f64([x], @(laneid, 1.5f64));",
        "st.global.v2.f64 [%x], {%laneid, 0d3FF8000000000000};",
        false,
        "reads a `f64`; `laneid` is a `u32`",
    ),
];

/// kernite refuses each vector of [`FLOAT_FIRST`], the float first or last, at the vector, or the `[` of the
/// coordinates: the register has bits that the instruction does not take there.
#[test]
fn a_special_register_of_bits_a_vector_of_floats_does_not_take_is_refused_beside_a_float_known_now() {
    for (line, _, _, said) in FLOAT_FIRST {
        let (body, col, name) = apart(line);
        let want = format!("2:{col}: `{name}` {said}");
        assert_eq!(compile("x: *mut u32, n: u32, s: u64", &body).unwrap_err(), want, "{line}");
    }
}

/// `ptxas` 13.5 takes each vector of [`FLOAT_FIRST`] that has the float first, and refuses the others, "Arguments
/// mismatch". It fails the day `ptxas` takes or refuses one the other way, for GUIDE.md to say no more of that bug.
#[test]
fn ptxas_takes_a_special_register_of_bits_the_vector_does_not_take_after_a_float_known_now_and_refuses_it_before() {
    let Some(ptxas) = ptxas() else {
        eprintln!("no ptxas here: skipped; set PTXAS to run it");
        return;
    };
    for (n, &(_, line, takes, _)) in FLOAT_FIRST.iter().enumerate() {
        let run = assembled(&ptxas, &with_line(line), "sm_86", &format!("float_first_{n}"), &[]);
        let said = String::from_utf8_lossy(&run.stderr);
        assert_eq!(run.status.success(), takes, "{line}: {said}");
        assert!(takes || said.contains("Arguments mismatch for instruction"), "{line}: {said}");
    }
}

/// An integer variable of the float's size in a vector of floats, beside no float: the data of `st`, the result of `ld`
/// and `ldu`, and the coordinates and the result of `tex` and `tld4`, with a special register before it or not. Each is
/// a kernel, with `@` where its error points; the line of PTX kernite would write; and the end of the error. `ptxas`
/// 13.5 takes each, as it checks the registers of a vector against each other, and not their type against the
/// instruction's, and refuses an integer variable alone, and one beside a float. kernite refuses all three.
const INTEGER_IN_FLOATS: [(&str, &str, &str); 12] = [
    ("st.global.v2.f32([x], @(i, i));", "st.global.v2.f32 [%x], {%i, %i};", "wants a `f32` variable; this is a `u32`"),
    (
        "st.global.v2.f32([x], @(tid.x, i));",
        "st.global.v2.f32 [%x], {%tid.x, %i};",
        "wants a `f32` variable; this is a `u32`",
    ),
    (
        "st.global.v2.f32([x], @(laneid, j));",
        "st.global.v2.f32 [%x], {%laneid, %j};",
        "wants a `f32` variable; this is a `s32`",
    ),
    ("st.global.v2.f64([x], @(u, u));", "st.global.v2.f64 [%x], {%u, %u};", "wants a `f64` variable; this is a `u64`"),
    (
        "st.global.v2.f64([x], @(gridid, u));",
        "st.global.v2.f64 [%x], {%gridid, %u};",
        "wants a `f64` variable; this is a `u64`",
    ),
    ("ld.global.v2.f32(@(i, i), [x]);", "ld.global.v2.f32 {%i, %i}, [%x];", "wants a `f32` variable; this is a `u32`"),
    (
        "ldu.global.v2.f32(@(i, i), [x]);",
        "ldu.global.v2.f32 {%i, %i}, [%x];",
        "wants a `f32` variable; this is a `u32`",
    ),
    (
        "tex._2d.v4.f32.f32((r0, r1, r2, r3), @[s, (i, i)]);",
        "tex.2d.v4.f32.f32 {%r0, %r1, %r2, %r3}, [%s, {%i, %i}];",
        "wants a `f32` variable; this is a `u32`",
    ),
    (
        "tex._1d.v4.f32.f32((r0, r1, r2, r3), @[s, i]);",
        "tex.1d.v4.f32.f32 {%r0, %r1, %r2, %r3}, [%s, {%i}];",
        "wants a `f32` variable; this is a `u32`",
    ),
    (
        "tex._2d.v4.f32.f32(@(i, i, i, i), [s, (f, f)]);",
        "tex.2d.v4.f32.f32 {%i, %i, %i, %i}, [%s, {%f, %f}];",
        "wants a `f32` variable; this is a `u32`",
    ),
    (
        "tld4.r._2d.v4.f32.f32((r0, r1, r2, r3), @[s, (i, i)]);",
        "tld4.r.2d.v4.f32.f32 {%r0, %r1, %r2, %r3}, [%s, {%i, %i}];",
        "wants a `f32` variable; this is a `u32`",
    ),
    (
        "tld4.r._2d.v4.f32.f32(@(i, i, i, i), [s, (f, f)]);",
        "tld4.r.2d.v4.f32.f32 {%i, %i, %i, %i}, [%s, {%f, %f}];",
        "wants a `f32` variable; this is a `u32`",
    ),
];

/// kernite refuses each vector of [`INTEGER_IN_FLOATS`], at the vector, or the `[` of the coordinates, though `ptxas`
/// takes it: an integer variable is not a float, whatever it is beside.
#[test]
fn an_integer_variable_in_a_vector_of_floats_is_refused_though_ptxas_takes_it() {
    for (line, _, said) in INTEGER_IN_FLOATS {
        let (body, col, name) = apart(line);
        let want = format!("2:{col}: `{name}` {said}");
        assert_eq!(compile("x: *mut u32, n: u32, s: u64", &body).unwrap_err(), want, "{line}");
    }
}

/// `ptxas` 13.5 takes each vector of [`INTEGER_IN_FLOATS`], and refuses an integer variable alone, in `st.global.f32`
/// and `ld.global.f32`: "Arguments mismatch". It fails the day `ptxas` refuses a vector or takes a variable alone, for
/// GUIDE.md to say no more of that bug.
#[test]
fn ptxas_takes_an_integer_variable_in_a_vector_of_floats_and_refuses_it_alone() {
    let Some(ptxas) = ptxas() else {
        eprintln!("no ptxas here: skipped; set PTXAS to run it");
        return;
    };
    for (n, &(_, line, _)) in INTEGER_IN_FLOATS.iter().enumerate() {
        let run = assembled(&ptxas, &with_line(line), "sm_86", &format!("integer_in_floats_{n}"), &[]);
        assert!(run.status.success(), "{line}: {}", String::from_utf8_lossy(&run.stderr));
    }
    for (n, line) in ["st.global.f32 [%x], %i;", "ld.global.f32 %i, [%x];"].into_iter().enumerate() {
        let run = assembled(&ptxas, &with_line(line), "sm_86", &format!("integer_alone_{n}"), &[]);
        let said = String::from_utf8_lossy(&run.stderr);
        assert!(!run.status.success() && said.contains("Arguments mismatch for instruction"), "{line}: {said}");
    }
}

/// A place where an instruction reads a special register by its bits: its kernel, the line of PTX kernite writes for
/// it, and the type that reads the register, and its bits. `R` stands for the register, and `@` marks where an error
/// points: at the whole of a vector, at the `[` of the coordinates in a texture, and at the register of `cvt`, or at
/// its field, the `x` of `tid.x`.
struct Place {
    kernel: String,
    line: String,
    ty: String,
    bits: u32,
    /// Whether the place takes a register of as many bits as it reads, and none of more.
    exact: bool,
}

impl Place {
    /// Whether the place takes a special register of `have` bits.
    fn takes(&self, have: u32) -> bool {
        if self.exact { have == self.bits } else { have >= self.bits }
    }
}

/// Where an instruction reads a special register by its bits, as `ptxas` does. It takes a register of more bits than
/// the type as the source of `cvt` and as the data of the vector of `st` and of `sust`, and one of as many bits as the
/// float in the coordinates of `tex` and `tld4`.
fn places() -> Vec<Place> {
    let place =
        |kernel: String, line: String, ty: &str, bits, exact| Place { kernel, line, ty: ty.into(), bits, exact };
    let mut out = vec![];
    for (ty, bits) in
        [("u8", 8), ("s8", 8), ("u16", 16), ("s16", 16), ("u32", 32), ("s32", 32), ("u64", 64), ("s64", 64)]
    {
        out.push(place(format!("cvt.u64.{ty}(u, @R);"), format!("cvt.u64.{ty} %u, %R;"), ty, bits, false));
    }
    for (ty, bits) in [("b8", 8), ("b16", 16), ("b32", 32), ("b64", 64), ("f32", 32), ("f64", 64)] {
        let st = format!("st.global.v2.{ty}");
        out.push(place(format!("{st}([x], @(R, R));"), format!("{st} [%x], {{%R, %R}};"), ty, bits, false));
        // `sust` writes bits alone.
        if ty.starts_with('b') {
            let sust = format!("sust.b.2d.v2.{ty}.trap");
            let kernel = format!("{}([s, (j, j)], @(R, R));", sust.replace(".2d", "._2d"));
            out.push(place(kernel, format!("{sust} [%s, {{%j, %j}}], {{%R, %R}};"), ty, bits, false));
        }
    }
    for name in ["tex._2d", "tld4.r._2d"] {
        let kernel = format!("{name}.v4.f32.f32((r0, r1, r2, r3), @[s, (R, R)]);");
        let line = format!("{}.v4.f32.f32 {{%r0, %r1, %r2, %r3}}, [%s, {{%R, %R}}];", name.replace("._", "."));
        out.push(place(kernel, line, "f32", 32, true));
    }
    out
}

/// A special register goes where its bits are those an instruction reads: as an integer, bits or a float of no more
/// bits than it has where the instruction takes a register wider than its type, `cvt.u32.u8(i, laneid)`,
/// `(laneid, laneid)` in a vector of 16 bits, and in one of `f32`; and of as many where it takes none wider,
/// `[s, (f, tid.x)]`, the coordinates of `tex` and `tld4`. kernite writes the line of each special register of the bits
/// the place takes, and refuses the rest where the place has `@`.
#[test]
fn a_special_register_goes_where_its_bits_are_those_the_instruction_reads() {
    for place in places() {
        let (before, after) = place.kernel.split_once('@').unwrap();
        for r in forms::sregs() {
            let after = after.replace('R', &r.name);
            let kernel = format!("{before}{after}");
            let body = format!("{TOGETHER_DECLARED} {kernel}");
            let got = compile("x: *mut u32, n: u32, s: u64", &body);
            if place.takes(r.ty.bits()) {
                let ptx = got.unwrap_or_else(|e| panic!("{kernel}: {e}"));
                assert!(ptx.contains(&format!("\t{}\n", place.line.replace('R', &r.name))), "{kernel}: {ptx}");
            } else {
                let name = before[..before.find('(').unwrap()].replace("._", ".");
                let field = if after.starts_with(['(', '[']) { 0 } else { r.name.find('.').map_or(0, |dot| dot + 1) };
                let col = body.len() - after.len() + 1 + field;
                let want = format!("2:{col}: `{name}` reads a `{}`; `{}` is a `{}`", place.ty, r.name, r.ty.name());
                assert_eq!(got.unwrap_err(), want, "{kernel}");
            }
        }
    }
}

/// What `ptxas` says of each of `lines` of PTX, each in a kernel of its own that declares `u`, `x` and `s`, of 64 bits,
/// `i`, a `u32`, `j`, an `s32`, `f` and `r0` to `r3`, of `f32`, and `d`, an `f64`, in one module of the newest PTX for
/// `sm_86`: its first error, or `None` where it takes the line.
fn said_of(ptxas: &Path, dir: &str, lines: &[String]) -> Vec<Option<String>> {
    const DECLARED: &str =
        ".reg .u64 %u, %x, %s;\n\t.reg .u32 %i;\n\t.reg .s32 %j;\n\t.reg .f32 %f, %r0, %r1, %r2, %r3;\n\t.reg .f64 %d;";
    let mut text = format!(".version {}\n.target sm_86\n.address_size 64\n\n", dotted(NEWEST));
    for (n, line) in lines.iter().enumerate() {
        text += &format!(".visible .entry k{n}()\n{{\n\t{DECLARED}\n\t{line}\n\tret;\n}}\n");
    }
    let (refused, _) = assemble(ptxas, dir, &[(text, 86)]).remove(0);
    (0..lines.len()).map(|n| refused.get(&format!("k{n}")).cloned()).collect()
}

/// `ptxas` 13.5 takes a special register as an integer, bits or a float of no more bits than it has, where an
/// instruction takes a register wider than its type, and as one of as many bits alone where it takes none wider, in the
/// coordinates of `tex` and `tld4`, and refuses one of more or of fewer bits: "Arguments mismatch". It fails the day
/// `ptxas` takes one it refuses here, for kernite to take it too, or refuses one it takes, for kernite and GUIDE.md to
/// say no more of it.
#[test]
fn ptxas_takes_a_special_register_where_its_bits_are_those_the_instruction_reads() {
    let Some(ptxas) = ptxas() else {
        eprintln!("no ptxas here: skipped; set PTXAS to run it");
        return;
    };
    let mut cases = vec![];
    for place in places() {
        cases.extend(forms::sregs().iter().map(|r| (place.line.replace('R', &r.name), place.takes(r.ty.bits()))));
    }
    let lines: Vec<String> = cases.iter().map(|(line, _)| line.clone()).collect();
    for ((line, takes), said) in cases.iter().zip(said_of(&ptxas, "wider", &lines)) {
        match said {
            None => assert!(takes, "ptxas takes `{line}`"),
            Some(said) => {
                let op = line.split(['.', ' ']).next().unwrap();
                assert!(!takes, "ptxas refuses `{line}`: {said}");
                // The quote is left open, as `ptxas` names `sust.b` where `op` is `sust`.
                assert!(said.contains(&format!("Arguments mismatch for instruction '{op}")), "{line}: {said}");
            }
        }
    }
}

/// `ptxas` 13.5 refuses each special register in `st` alone, which stores one only as one of a vector's: it says that a
/// special register is not allowed there. It fails the day `ptxas` takes one, for kernite to take it too.
#[test]
fn ptxas_refuses_each_special_register_in_st_alone() {
    let Some(ptxas) = ptxas() else {
        eprintln!("no ptxas here: skipped; set PTXAS to run it");
        return;
    };
    let lines: Vec<String> = forms::sregs().iter().map(|r| format!("st.global.b16 [%x], %{};", r.name)).collect();
    for (line, said) in lines.iter().zip(said_of(&ptxas, "alone", &lines)) {
        let said = said.unwrap_or_else(|| panic!("ptxas takes `{line}`"));
        assert!(said.ends_with("Special register argument not allowed for instruction 'st'"), "{line}: {said}");
    }
}

/// The variables of [`TOGETHER`]: each type of 16, 32 and 64 bits that tells `ptxas` registers apart in a vector, and
/// bits of each size, which are those of a `bf16`, a `bf16x2` and a `tf32` too, for PTX declares them as bits.
const TOGETHER_DECLARED: &str = "let (f, g, i, j, b, h): (f32, f32, u32, s32, b32, f16x2); \
     let (r0, r1, r2, r3): (f32, f32, f32, f32); let (a, c, e): (u16, s16, b16); let (d, u, q): (f64, u64, b64); \
     let (z, w): (b64, b32); let (m, o, k, l): (bf16x2, tf32, bf16, f16); mov.b32(m, n); mov.b32(o, n); mov.b16(k, 1); \
     mov.b16(l, 1); mov.f32(f, 1.5); mov.f32(g, 2.5); mov.u32(i, n); mov.s32(j, 1); mov.b32(b, n); \
     mov.b32(h, n); mov.u16(a, 1); mov.s16(c, 1); mov.b16(e, 1); mov.f64(d, 1.5f64); mov.u64(u, 1); mov.b64(q, x); \
     mov.b64(z, x); mov.b32(w, n);";

/// Vectors that `ptxas` 13.5 takes, and the line of PTX kernite writes for each: registers of one type; bits beside any
/// register of their size, and an integer beside an integer, `u32` and `s32`; registers of 16 bits and of 64; of 32
/// bits wider than an instruction of 16, where it takes them; a special register beside a register of any type of its
/// size; a value known now beside bits, and beside a register of its kind, an integer beside integers and a float
/// beside floats; a value known now beside any register of a vector of floats, a `b64` too; a whole special register
/// beside a value, which a part of a vector's does not go beside; a special register of more bits than the type, as the
/// source of `cvt` and the data of `st` and `sust`, which take a register of more bits: `laneid` in a vector of 16,
/// `clock64` in one of 32; the source of a `cvt` from an integer to one of another size or sign,
/// `cvt.u32.s32(i, envreg0)`, `cvt.u64.u32(u, tid.x)` and `cvt.u32.u64(i, gridid)`; `gridid`, of 64 bits, in a vector
/// of 64, and `mov.u32` reading 32 of its bits; `envreg0`, of `b32`, in a `mov` of bits or of an integer, and in a
/// vector of floats, where a `mov` of a float takes none; an integer special register in a vector of floats, by its
/// size: beside a variable, `(f, tid.x)`, beside itself, `(laneid, laneid)`, and `(gridid, gridid)` in the data of
/// `st`, which takes a register of more bits, and a whole one beside a value known now, `(1.5, laneid)` and
/// `(laneid, 1)`; `_`; and the layer of an array of textures, the first of its vector, a `u32` beside floats, which
/// `ptxas` checks apart.
const TOGETHER: [(&str, &str); 118] = [
    ("st.global.v2.b32([x], (f, g));", "st.global.v2.b32 [%x], {%f, %g};"),
    ("st.global.v2.b32([x], (i, j));", "st.global.v2.b32 [%x], {%i, %j};"),
    ("st.global.v2.u32([x], (i, j));", "st.global.v2.u32 [%x], {%i, %j};"),
    ("st.global.v2.b32([x], (h, h));", "st.global.v2.b32 [%x], {%h, %h};"),
    ("st.global.v2.b32([x], (b, i));", "st.global.v2.b32 [%x], {%b, %i};"),
    ("st.global.v2.b32([x], (i, b));", "st.global.v2.b32 [%x], {%i, %b};"),
    ("st.global.v2.b32([x], (b, f));", "st.global.v2.b32 [%x], {%b, %f};"),
    ("st.global.v2.b32([x], (f, b));", "st.global.v2.b32 [%x], {%f, %b};"),
    ("st.global.v2.b32([x], (b, h));", "st.global.v2.b32 [%x], {%b, %h};"),
    ("st.global.v2.b32([x], (h, b));", "st.global.v2.b32 [%x], {%h, %b};"),
    ("st.global.v4.b32([x], (f, b, g, b));", "st.global.v4.b32 [%x], {%f, %b, %g, %b};"),
    ("st.global.v4.u32([x], (i, b, j, b));", "st.global.v4.u32 [%x], {%i, %b, %j, %b};"),
    ("ld.global.v2.b32((f, g), [x]);", "ld.global.v2.b32 {%f, %g}, [%x];"),
    ("ld.global.v2.b32((i, j), [x]);", "ld.global.v2.b32 {%i, %j}, [%x];"),
    ("ld.global.v2.u32((b, i), [x]);", "ld.global.v2.u32 {%b, %i}, [%x];"),
    ("ld.global.v2.b32((b, f), [x]);", "ld.global.v2.b32 {%b, %f}, [%x];"),
    ("ld.global.v4.b32((f, b, g, b), [x]);", "ld.global.v4.b32 {%f, %b, %g, %b}, [%x];"),
    ("mov.b64(z, (f, g));", "mov.b64 %z, {%f, %g};"),
    ("mov.b64(z, (i, j));", "mov.b64 %z, {%i, %j};"),
    ("mov.b64(z, (b, f));", "mov.b64 %z, {%b, %f};"),
    ("mov.b64((f, g), z);", "mov.b64 {%f, %g}, %z;"),
    ("mov.b64((i, j), z);", "mov.b64 {%i, %j}, %z;"),
    ("st.global.v2.b16([x], (a, c));", "st.global.v2.b16 [%x], {%a, %c};"),
    ("st.global.v2.b16([x], (e, a));", "st.global.v2.b16 [%x], {%e, %a};"),
    ("st.global.v2.b16([x], (k, l));", "st.global.v2.b16 [%x], {%k, %l};"),
    ("st.global.v4.b32([x], (m, o, f, f));", "st.global.v4.b32 [%x], {%m, %o, %f, %f};"),
    ("mov.b32(w, (a, c));", "mov.b32 %w, {%a, %c};"),
    ("st.global.v2.b64([x], (d, d));", "st.global.v2.b64 [%x], {%d, %d};"),
    ("st.global.v2.b64([x], (q, d));", "st.global.v2.b64 [%x], {%q, %d};"),
    ("st.global.v2.u64([x], (u, q));", "st.global.v2.u64 [%x], {%u, %q};"),
    ("st.global.v2.u16([x], (i, b));", "st.global.v2.u16 [%x], {%i, %b};"),
    ("st.global.v2.b16([x], (f, b));", "st.global.v2.b16 [%x], {%f, %b};"),
    ("st.global.v2.b32([x], (tid.x, f));", "st.global.v2.b32 [%x], {%tid.x, %f};"),
    ("st.global.v2.u32([x], (i, 5));", "st.global.v2.u32 [%x], {%i, 5};"),
    ("st.global.v2.f32([x], (f, 1.5));", "st.global.v2.f32 [%x], {%f, 0f3FC00000};"),
    ("ld.global.v2.u32((i, _), [x]);", "ld.global.v2.u32 {%i, _}, [%x];"),
    (
        "tex.a2d.v4.f32.f32((r0, r1, r2, r3), [s, (n, f, g, f)]);",
        "tex.a2d.v4.f32.f32 {%r0, %r1, %r2, %r3}, [%s, {%n, %f, %g, %f}];",
    ),
    (
        "tex.a2d.v4.f32.s32((r0, r1, r2, r3), [s, (n, j, j, j)]);",
        "tex.a2d.v4.f32.s32 {%r0, %r1, %r2, %r3}, [%s, {%n, %j, %j, %j}];",
    ),
    ("sust.b._2d.v2.b32.trap([s, (n, n)], (f, b));", "sust.b.2d.v2.b32.trap [%s, {%n, %n}], {%f, %b};"),
    ("sust.b._2d.v2.b32.trap([s, (n, n)], (i, j));", "sust.b.2d.v2.b32.trap [%s, {%n, %n}], {%i, %j};"),
    ("suld.b._2d.v2.b32.trap((b, f), [s, (n, n)]);", "suld.b.2d.v2.b32.trap {%b, %f}, [%s, {%n, %n}];"),
    (
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16((f, g), [n]);",
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%f, %g}, [%n];",
    ),
    (
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16((b, i), [n]);",
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%b, %i}, [%n];",
    ),
    (
        "mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.xor.popc((j, j, j, j), (b, i), b, (j, j, j, j));",
        "mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.xor.popc {%j, %j, %j, %j}, {%b, %i}, {%b}, {%j, %j, %j, %j};",
    ),
    ("ld.global.v2.u16((i, j), [x]);", "ld.global.v2.u16 {%i, %j}, [%x];"),
    ("st.global.v2.b32([x], (f, 2.5));", "st.global.v2.b32 [%x], {%f, 0f40200000};"),
    ("st.global.v2.b32([x], (f, 1.5f32));", "st.global.v2.b32 [%x], {%f, 0f3FC00000};"),
    ("st.global.v2.b32([x], (i, 1));", "st.global.v2.b32 [%x], {%i, 1};"),
    ("st.global.v2.b32([x], (b, 1.5));", "st.global.v2.b32 [%x], {%b, 0f3FC00000};"),
    ("st.global.v2.b32([x], (1, 2));", "st.global.v2.b32 [%x], {1, 2};"),
    ("st.global.v2.b32([x], (1.5, 2.5));", "st.global.v2.b32 [%x], {0f3FC00000, 0f40200000};"),
    ("st.global.v2.b32([x], (f, tid.x));", "st.global.v2.b32 [%x], {%f, %tid.x};"),
    ("st.global.v2.b32([x], (h, tid.x));", "st.global.v2.b32 [%x], {%h, %tid.x};"),
    ("st.global.v2.b32([x], (tid.x, h));", "st.global.v2.b32 [%x], {%tid.x, %h};"),
    ("st.global.v2.u32([x], (h, tid.x));", "st.global.v2.u32 [%x], {%h, %tid.x};"),
    ("st.global.v2.u32([x], (tid.x, h));", "st.global.v2.u32 [%x], {%tid.x, %h};"),
    ("st.global.v2.b32([x], (tid.x, b));", "st.global.v2.b32 [%x], {%tid.x, %b};"),
    ("st.global.v2.b32([x], (tid.x, tid.y));", "st.global.v2.b32 [%x], {%tid.x, %tid.y};"),
    ("st.global.v2.b16([x], (a, 1.5f16));", "st.global.v2.b16 [%x], {%a, 0x3E00};"),
    ("st.global.v2.b16([x], (a, 1.5bf16));", "st.global.v2.b16 [%x], {%a, 0x3FC0};"),
    ("st.global.v2.b16([x], (k, 1.5bf16));", "st.global.v2.b16 [%x], {%k, 0x3FC0};"),
    ("st.global.v2.b64([x], (u, 1));", "st.global.v2.b64 [%x], {%u, 1};"),
    ("st.global.v2.b64([x], (d, 1.5f64));", "st.global.v2.b64 [%x], {%d, 0d3FF8000000000000};"),
    ("st.global.v2.b64([x], (1.5f64, d));", "st.global.v2.b64 [%x], {0d3FF8000000000000, %d};"),
    ("st.global.v2.f32([x], (f, 1));", "st.global.v2.f32 [%x], {%f, 0f3F800000};"),
    ("st.global.v2.f32([x], (q, 1.5));", "st.global.v2.f32 [%x], {%q, 0f3FC00000};"),
    ("st.global.v2.f32([x], (1.5, q));", "st.global.v2.f32 [%x], {0f3FC00000, %q};"),
    ("st.global.v2.f32([x], (q, 1));", "st.global.v2.f32 [%x], {%q, 0f3F800000};"),
    ("mov.b64(z, (f, tid.x));", "mov.b64 %z, {%f, %tid.x};"),
    ("mov.b64(z, (tid.x, f));", "mov.b64 %z, {%tid.x, %f};"),
    ("mov.b64(z, (laneid, 1));", "mov.b64 %z, {%laneid, 1};"),
    ("st.global.v2.b32([x], (laneid, 1));", "st.global.v2.b32 [%x], {%laneid, 1};"),
    ("st.global.v2.b32([x], (1, laneid));", "st.global.v2.b32 [%x], {1, %laneid};"),
    ("st.global.v2.b32([x], (laneid, 1.5));", "st.global.v2.b32 [%x], {%laneid, 0f3FC00000};"),
    ("st.global.v2.b64([x], (clock64, 1));", "st.global.v2.b64 [%x], {%clock64, 1};"),
    ("st.global.v2.b64([x], (clock64, d));", "st.global.v2.b64 [%x], {%clock64, %d};"),
    ("st.global.v2.b64([x], (u, clock64));", "st.global.v2.b64 [%x], {%u, %clock64};"),
    ("sust.b._2d.v2.b32.trap([s, (n, n)], (f, tid.x));", "sust.b.2d.v2.b32.trap [%s, {%n, %n}], {%f, %tid.x};"),
    ("sust.b._2d.v2.b32.trap([s, (n, n)], (laneid, 1));", "sust.b.2d.v2.b32.trap [%s, {%n, %n}], {%laneid, 1};"),
    (
        "tex._2d.v4.f32.s32((r0, r1, r2, r3), [s, (laneid, 5)]);",
        "tex.2d.v4.f32.s32 {%r0, %r1, %r2, %r3}, [%s, {%laneid, 5}];",
    ),
    (
        "mma.sync.aligned.m16n8k16.row.col.s32.u8.u8.s32((j, j, j, j), (laneid, 8), n, (0, 0, 0, 0));",
        "mma.sync.aligned.m16n8k16.row.col.s32.u8.u8.s32 {%j, %j, %j, %j}, {%laneid, 8}, {%n}, {0, 0, 0, 0};",
    ),
    ("st.global.v2.b32([x], (gridid, gridid));", "st.global.v2.b32 [%x], {%gridid, %gridid};"),
    ("st.global.v2.b32([x], (gridid, 1));", "st.global.v2.b32 [%x], {%gridid, 1};"),
    (
        "sust.b._2d.v2.b32.trap([s, (n, n)], (gridid, gridid));",
        "sust.b.2d.v2.b32.trap [%s, {%n, %n}], {%gridid, %gridid};",
    ),
    ("st.global.v2.b64([x], (gridid, d));", "st.global.v2.b64 [%x], {%gridid, %d};"),
    ("cvt.u32.u16(i, laneid);", "cvt.u32.u16 %i, %laneid;"),
    ("cvt.u32.u8(i, laneid);", "cvt.u32.u8 %i, %laneid;"),
    ("cvt.u32.u8(i, tid.x);", "cvt.u32.u8 %i, %tid.x;"),
    ("cvt.u32.u16(i, envreg0);", "cvt.u32.u16 %i, %envreg0;"),
    ("cvt.u64.u32(u, clock64);", "cvt.u64.u32 %u, %clock64;"),
    ("cvt.u64.u16(u, clock64);", "cvt.u64.u16 %u, %clock64;"),
    ("cvt.u32.s32(i, envreg0);", "cvt.u32.s32 %i, %envreg0;"),
    ("cvt.u64.u32(u, tid.x);", "cvt.u64.u32 %u, %tid.x;"),
    ("cvt.u32.u64(i, gridid);", "cvt.u32.u64 %i, %gridid;"),
    ("cvt.s32.u32(j, laneid);", "cvt.s32.u32 %j, %laneid;"),
    ("st.global.v2.b16([x], (laneid, laneid));", "st.global.v2.b16 [%x], {%laneid, %laneid};"),
    ("st.global.v2.b8([x], (tid.x, tid.x));", "st.global.v2.b8 [%x], {%tid.x, %tid.x};"),
    (
        "st.global.v4.b8([x], (laneid, laneid, laneid, laneid));",
        "st.global.v4.b8 [%x], {%laneid, %laneid, %laneid, %laneid};",
    ),
    ("st.global.v2.b32([x], (clock64, clock64));", "st.global.v2.b32 [%x], {%clock64, %clock64};"),
    ("st.global.v2.b16([x], (envreg0, envreg0));", "st.global.v2.b16 [%x], {%envreg0, %envreg0};"),
    (
        "sust.b._2d.v2.b16.trap([s, (j, j)], (laneid, laneid));",
        "sust.b.2d.v2.b16.trap [%s, {%j, %j}], {%laneid, %laneid};",
    ),
    (
        "sust.b._2d.v2.b32.trap([s, (j, j)], (clock64, clock64));",
        "sust.b.2d.v2.b32.trap [%s, {%j, %j}], {%clock64, %clock64};",
    ),
    ("mov.u32(i, gridid);", "mov.u32 %i, %gridid;"),
    ("mov.b32(b, envreg0);", "mov.b32 %b, %envreg0;"),
    ("mov.u32(i, envreg0);", "mov.u32 %i, %envreg0;"),
    ("st.global.v2.f32([x], (f, envreg0));", "st.global.v2.f32 [%x], {%f, %envreg0};"),
    ("st.global.v2.f32([x], (envreg0, envreg0));", "st.global.v2.f32 [%x], {%envreg0, %envreg0};"),
    ("st.global.v2.f32([x], (f, tid.x));", "st.global.v2.f32 [%x], {%f, %tid.x};"),
    ("st.global.v2.f32([x], (tid.x, f));", "st.global.v2.f32 [%x], {%tid.x, %f};"),
    ("st.global.v2.f32([x], (laneid, laneid));", "st.global.v2.f32 [%x], {%laneid, %laneid};"),
    ("st.global.v2.f32([x], (gridid, gridid));", "st.global.v2.f32 [%x], {%gridid, %gridid};"),
    ("st.global.v2.f32([x], (1.5, laneid));", "st.global.v2.f32 [%x], {0f3FC00000, %laneid};"),
    ("st.global.v2.f32([x], (laneid, 1));", "st.global.v2.f32 [%x], {%laneid, 0f3F800000};"),
    ("st.global.v2.f64([x], (d, gridid));", "st.global.v2.f64 [%x], {%d, %gridid};"),
    ("st.global.v2.f64([x], (clock64, d));", "st.global.v2.f64 [%x], {%clock64, %d};"),
    (
        "tex._2d.v4.f32.f32((r0, r1, r2, r3), [s, (f, tid.x)]);",
        "tex.2d.v4.f32.f32 {%r0, %r1, %r2, %r3}, [%s, {%f, %tid.x}];",
    ),
    (
        "tex._2d.v4.f32.f32((r0, r1, r2, r3), [s, (tid.x, tid.x)]);",
        "tex.2d.v4.f32.f32 {%r0, %r1, %r2, %r3}, [%s, {%tid.x, %tid.x}];",
    ),
    (
        "tld4.r._2d.v4.f32.f32((r0, r1, r2, r3), [s, (f, laneid)]);",
        "tld4.r.2d.v4.f32.f32 {%r0, %r1, %r2, %r3}, [%s, {%f, %laneid}];",
    ),
];

/// kernite writes each of [`TOGETHER`] as it is.
#[test]
fn a_vector_takes_registers_of_types_that_ptxas_takes_together() {
    for (inst, line) in TOGETHER {
        let ptx = compile("x: *mut u32, n: u32, s: u64", &format!("{TOGETHER_DECLARED} {inst}")).unwrap();
        assert!(ptx.contains(&format!("\t{line}\n")), "{inst}: {ptx}");
    }
}

/// `ptxas` takes each of [`TOGETHER`], as kernite writes it.
#[test]
fn ptxas_takes_each_vector_that_kernite_takes() {
    let Some(ptxas) = ptxas() else {
        eprintln!("no ptxas here: skipped; set PTXAS to run it");
        return;
    };
    for (n, (inst, _)) in TOGETHER.into_iter().enumerate() {
        let ptx = compile("x: *mut u32, n: u32, s: u64", &format!("{TOGETHER_DECLARED} {inst}")).unwrap();
        let run = assembled(&ptxas, &ptx, "sm_86", &format!("together_{n}"), &["--warning-as-error"]);
        assert!(run.status.success(), "{inst}: {}", String::from_utf8_lossy(&run.stderr));
    }
}

/// Where an instruction moves or converts a float and takes no special register, whatever its bits: a `mov` of an `f32`
/// or an `f64`, a `cvt` from an `f16`, an `f32` or an `f64` to a `u32`, an `s32` or a `u64`, and a `cvt` to an `f32` or
/// an `f64` from an integer or from the other float. Each is its kernel, where `R` stands for the register; the line of
/// PTX kernite would write; and the name of the instruction.
fn float_places() -> Vec<(String, String, String)> {
    let mut places = vec![];
    let mut place = |var: &str, ty: &str, name: String| {
        places.push((format!("let {var}: {ty}; {name}({var}, R);"), format!("{name} %{var}, %R;"), name));
    };
    for (float, var) in [("f32", "f"), ("f64", "d")] {
        place(var, float, format!("mov.{float}"));
    }
    for float in ["f16", "f32", "f64"] {
        for (int, var, round) in [("u32", "i", "rzi"), ("s32", "j", "rni"), ("u64", "u", "rzi")] {
            place(var, int, format!("cvt.{round}.{int}.{float}"));
        }
    }
    for (var, to, name) in [
        ("f", "f32", "cvt.rn.f32.u16"),
        ("f", "f32", "cvt.rn.f32.u32"),
        ("d", "f64", "cvt.rn.f64.u64"),
        ("f", "f32", "cvt.rn.f32.f64"),
        ("d", "f64", "cvt.f64.f32"),
    ] {
        place(var, to, name.into());
    }
    places
}

/// kernite refuses each special register where an instruction moves or converts a float, at the register, and says
/// which `mov` reads it. `ptxas` 13.5 takes none there: not a `b32` such as `envreg0` for an `f32`, and not a `u32`
/// such as `tid.x`.
#[test]
fn a_special_register_in_a_mov_or_a_cvt_of_a_float_is_refused_at_the_register() {
    for r in forms::sregs() {
        for (kernel, _, name) in float_places() {
            let body = kernel.replace('R', &r.name);
            let col = body.rfind(&r.name).unwrap() + 1 + r.name.find('.').map_or(0, |dot| dot + 1);
            let want = format!("2:{col}: `{name}` reads no special register; `mov.{}` does", r.ty.name());
            assert_eq!(compile("", &body).unwrap_err(), want, "{body}");
        }
    }
}

/// `ptxas` 13.5 refuses each special register where an instruction moves or converts a float, of the float's size or
/// not: it says that a special register is not allowed there, or, for some, that the arguments mismatch. It fails the
/// day `ptxas` takes one, for kernite to take it too.
#[test]
fn ptxas_refuses_each_special_register_in_a_mov_or_a_cvt_of_a_float() {
    let Some(ptxas) = ptxas() else {
        eprintln!("no ptxas here: skipped; set PTXAS to run it");
        return;
    };
    let lines: Vec<String> = forms::sregs()
        .iter()
        .flat_map(|r| float_places().into_iter().map(move |(_, line, _)| line.replace('R', &r.name)))
        .collect();
    for (line, said) in lines.iter().zip(said_of(&ptxas, "float_places", &lines)) {
        let said = said.unwrap_or_else(|| panic!("ptxas takes `{line}`"));
        let op = line.split(['.', ' ']).next().unwrap();
        assert!(
            said.ends_with(&format!("Special register argument not allowed for instruction '{op}'"))
                || said.ends_with(&format!("Arguments mismatch for instruction '{op}'")),
            "{line}: {said}"
        );
    }
}

/// `ptxas` 13.5 refuses registers of two types in a vector, in each of `st`, also of floats, `ld`, `ldu`, `mov`,
/// `suld.b`, `sust.b`, `sust.p`, `ldmatrix` and `mma`, and of two sizes, but checks each against its neighbour alone,
/// and takes `{%f1, %b1, %i1, %i1}`, an `f32` and a `u32` with a `b32` between, by a bug of its own, which kernite
/// refuses. It fails the day `ptxas` takes a vector it refuses here, for kernite to take it too, or refuses one it
/// takes, for the comment by kernite's rule and GUIDE.md to say no more of its bug.
#[test]
fn ptxas_checks_each_register_of_a_vector_against_its_neighbour_alone() {
    let Some(ptxas) = ptxas() else {
        eprintln!("no ptxas here: skipped; set PTXAS to run it");
        return;
    };
    let types = "Vector with elements of different types are not allowed in";
    let cases = [
        ("st.global.v2.b32 [%x], {%f1, %i1};", Some(format!("{types} st instruction"))),
        ("st.global.v2.f32 [%x], {%f1, %i1};", Some(format!("{types} st instruction"))),
        ("ld.global.v2.b32 {%f1, %i1}, [%x];", Some(format!("{types} ld instruction"))),
        ("ldu.global.v2.b32 {%f1, %i1}, [%x];", Some(format!("{types} ldu instruction"))),
        ("mov.b64 %z1, {%f1, %i1};", Some(format!("{types} mov instruction"))),
        ("suld.b.2d.v2.b32.trap {%f1, %i1}, [%x, {%i1, %i1}];", Some(format!("{types} suld.b instruction"))),
        ("sust.b.2d.v2.b32.trap [%x, {%i1, %i1}], {%f1, %i1};", Some(format!("{types} sust.b instruction"))),
        ("sust.p.2d.v2.b32.trap [%x, {%i1, %i1}], {%f1, %i1};", Some(format!("{types} sust.p instruction"))),
        ("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%f1, %i1}, [%i1];", Some(format!("{types} ldmatrix instruction"))),
        (
            "mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.xor.popc {%c1, %c1, %c1, %c1}, {%f1, %i1}, {%b1}, \
             {%c1, %c1, %c1, %c1};",
            Some(format!("{types} mma instruction")),
        ),
        ("st.global.v2.u16 [%x], {%a1, %i1};", Some("Incompatible elements of vector expression".into())),
        ("st.global.v4.b32 [%x], {%f1, %b1, %i1, %i1};", None),
        ("st.global.v4.u32 [%x], {%h1, %b1, %i1, %i1};", None),
    ];
    for (n, (inst, refused)) in cases.into_iter().enumerate() {
        let text = format!(
            ".version {}\n.target sm_86\n.address_size 64\n\n.visible .entry k(.param .u64 x)\n{{\n\
             \t.reg .u64 %x;\n\t.reg .b64 %z1;\n\t.reg .f32 %f1;\n\t.reg .f16x2 %h1;\n\t.reg .b32 %b1;\n\
             \t.reg .u32 %i1;\n\t.reg .s32 %c1;\n\t.reg .u16 %a1;\n\tld.param.u64 %x, [x];\n\tmov.b32 %f1, 0;\n\
             \tmov.b32 %h1, 0;\n\tmov.b32 %b1, 0;\n\tmov.b32 %i1, 0;\n\tmov.s32 %c1, 0;\n\tmov.b16 %a1, 0;\n\
             \t{inst}\n\tret;\n}}\n",
            dotted(NEWEST)
        );
        let run = assembled(&ptxas, &text, "sm_86", &format!("neighbour_{n}"), &[]);
        let said = String::from_utf8_lossy(&run.stderr);
        match refused {
            Some(words) => assert!(!run.status.success() && said.contains(&words), "{inst}: {said}"),
            None => assert!(run.status.success(), "{inst}: {said}"),
        }
    }
}

/// A `bf16x2` and a `tf32` where an instruction wants an integer or a float of 32 bits, and a `bf16` where one wants an
/// integer or a float of 16: in an instruction, beside an `f16x2` in a vector, as an address, in a call, as its
/// argument, its result and the address it calls at, in a `ret`, and as the argument of a helper.
const KEPT: &str = r#"
    fn bump(v: u32) { add.u32(v, v, 1); }

    pub unsafe extern "ptx-kernel" fn kept(x: *mut u32, n: u32) {
        let b: bf16x2;
        let t: tf32;
        let c: bf16;
        let h: f16x2;
        let f: f16;
        let (i, a, v): (u32, u16, f32);
        mov.b32(b, n);
        mov.b32(t, n);
        mov.b16(c, 1);
        mov.b32(h, n);
        mov.b16(f, 1);
        mov.u32(i, n);
        mov.u16(a, 1);
        mov.f32(v, 1.5);
        add.u32(i, b, i);
        add.u32(i, t, i);
        add.rn.f32(v, b, v);
        add.rn.f32(v, t, v);
        mov.u32(i, b);
        mov.u32(i, t);
        st.global.u32([x], b);
        st.global.u32([x], t);
        ld.global.u32(b, [x]);
        add.u16(a, c, a);
        add.rn.f16(f, c, f);
        mov.u16(a, c);
        st.global.u16([x], c);
        st.global.u8([x], c);
        cvt.u32.u16(i, c);
        fma.rn.bf16x2(b, t, b, b);
        st.global.v2.u32([x], (h, b));
        st.global.v2.u32([x], (t, h));
        ld.shared.u32(i, [b]);
        st.shared.u32([t + 4], i);
        bump(b);
        call(i, count, (b));
        call(b, count, (i));
        call(i, count, (t));
        call(t, count, (i));
        call(v, half, (b));
        call(t, half, (v));
        call(a, short, (c));
        call(c, short, (a));
        call(i, b, i, (count,));
        call(i, t, i);
        call(i, keep_b, (i));
        call(v, keep_t, (v));
        call(a, keep_c, (a));
        st.global.u32([x], i);
    }

    extern "ptx-func" fn count(a: u32) -> u32 { ret(a); }
    extern "ptx-func" fn half(a: f32) -> f32 { ret(a); }
    extern "ptx-func" fn short(a: u16) -> u16 { ret(a); }
    extern "ptx-func" fn keep_b(a: u32) -> u32 { let b: bf16x2; mov.b32(b, a); ret(b); }
    extern "ptx-func" fn keep_t(a: f32) -> f32 { let t: tf32; mov.b32(t, a); ret(t); }
    extern "ptx-func" fn keep_c(a: u16) -> u16 { let c: bf16; mov.b16(c, a); ret(c); }
"#;

/// PTX has no register of a `bf16`, a `bf16x2` or a `tf32`, and keeps them in a `b16` or a `b32`: kernite takes each
/// wherever it takes the `b16` or `b32`, for an integer or a float of its size, in an instruction, beside an `f16x2` in
/// a vector, as an address, as an argument of a helper and of a call, as the address a call calls at, as the result of
/// a call, and in a `ret`. Its code writes the variable as it is, with no `mov` beside it, and it is the code of the
/// same kernels with a `b16` or `b32` in place of each.
#[test]
fn a_bf16_a_bf16x2_or_a_tf32_is_taken_where_the_b16_or_b32_it_is_kept_in_is() {
    let ptx = kernite::ptx_str(KEPT).unwrap();
    for line in [
        "\tadd.u32 %i, %b, %i;\n",
        "\tadd.u32 %i, %t, %i;\n",
        "\tadd.rn.f32 %v, %b, %v;\n",
        "\tadd.rn.f32 %v, %t, %v;\n",
        "\tmov.u32 %i, %b;\n",
        "\tmov.u32 %i, %t;\n",
        "\tst.global.u32 [%x], %b;\n",
        "\tst.global.u32 [%x], %t;\n",
        "\tld.global.u32 %b, [%x];\n",
        "\tadd.u16 %a, %c, %a;\n",
        "\tadd.rn.f16 %f, %c, %f;\n",
        "\tmov.u16 %a, %c;\n",
        "\tst.global.u16 [%x], %c;\n",
        "\tst.global.u8 [%x], %c;\n",
        "\tcvt.u32.u16 %i, %c;\n",
        "\tfma.rn.bf16x2 %b, %t, %b, %b;\n",
        "\tst.global.v2.u32 [%x], {%h, %b};\n",
        "\tst.global.v2.u32 [%x], {%t, %h};\n",
        "\tld.shared.u32 %i, [%b];\n",
        "\tst.shared.u32 [%t+4], %i;\n",
        "\tadd.u32 %b, %b, 1;\n",
        "\tcall (%i), count, (%b);\n",
        "\tcall (%b), count, (%i);\n",
        "\tcall (%i), count, (%t);\n",
        "\tcall (%t), count, (%i);\n",
        "\tcall (%v), half, (%b);\n",
        "\tcall (%t), half, (%v);\n",
        "\tcall (%a), short, (%c);\n",
        "\tcall (%c), short, (%a);\n",
        "\tcall (%i), %b, (%i), $C0;\n",
        "\tcall (%i), %t, (%i), $P1;\n",
        "\tst.param.u32 [keep_b_retval], %b;\n",
        "\tst.param.f32 [keep_t_retval], %t;\n",
        "\tst.param.b16 [keep_c_retval], %c;\n",
    ] {
        assert!(ptx.contains(line), "{line}: {ptx}");
    }
    assert_eq!(ptx.matches("\tmov.").count(), KEPT.matches("mov.").count(), "a `mov` beside one: {ptx}");
    let bits = KEPT.replace(": bf16x2", ": b32").replace(": tf32", ": b32").replace(": bf16", ": b16");
    assert_eq!(ptx, kernite::ptx_str(&bits).unwrap());
}

/// `ptxas` takes what kernite writes of [`KEPT`], at the `sm_80` that `fma.rn.bf16x2` needs.
#[test]
fn ptxas_assembles_a_bf16_a_bf16x2_and_a_tf32_as_the_b16_or_b32_they_are_kept_in() {
    let Some(ptxas) = ptxas() else {
        eprintln!("no ptxas here: skipped; set PTXAS to run it");
        return;
    };
    let run = assembled(&ptxas, &kernite::ptx_str(KEPT).unwrap(), "sm_86", "kept", &["--warning-as-error"]);
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
}

/// `ptxas` 13.5 takes a `.b32`, which a `tf32` is, for an operand of `fma.rn.bf16x2`, and refuses a `.u32`, an `.f32`
/// and an `.f16x2`, as kernite does of a `u32`, an `f32` and an `f16x2`, in
/// `a_wrong_operand_is_named_by_what_the_instruction_wants`.
#[test]
fn ptxas_takes_a_b32_for_an_operand_of_fma_rn_bf16x2_and_no_integer_or_float() {
    let Some(ptxas) = ptxas() else {
        eprintln!("no ptxas here: skipped; set PTXAS to run it");
        return;
    };
    for (ty, takes) in [("b32", true), ("u32", false), ("f32", false), ("f16x2", false)] {
        let text = format!(
            ".version 7.0\n.target sm_80\n.address_size 64\n\n.visible .entry k()\n{{\n\t.reg .b32 %b;\n\t.reg .{ty} \
             %x;\n\tmov.b32 %b, 0;\n\tmov.b32 %x, 0;\n\tfma.rn.bf16x2 %b, %x, %b, %b;\n\tret;\n}}\n"
        );
        let run = assembled(&ptxas, &text, "sm_86", &format!("fma_of_{ty}"), &[]);
        assert_eq!(run.status.success(), takes, "{ty}: {}", String::from_utf8_lossy(&run.stderr));
    }
}

/// A name of each instruction PTX's ISA has for the GPUs from `sm_20` to `sm_86`, by the
/// sections of its chapter of them, is of a form kernite knows; and the first word of each form
/// is that of one of them, so that the list misses no first word kernite knows. `call` is not
/// among them: its operands are what the function it calls takes, as
/// `a_call_passes_what_the_function_takes_and_returns` checks.
#[test]
fn kernite_knows_every_instruction_ptx_has_for_sm_20_to_sm_86() {
    let isa = [
        // Integer arithmetic.
        "add.s32 sub.s32 mul.lo.s32 mad.lo.s32 mul24.lo.s32 mad24.lo.s32 sad.s32 div.s32 rem.s32 abs.s32 neg.s32 \
         min.s32 max.s32 popc.b32 clz.b32 bfind.u32 fns.b32 brev.b32 bfe.u32 bfi.b32 szext.wrap.u32 bmsk.wrap.b32 \
         dp4a.u32.u32 dp2a.lo.u32.u32 clmad.lo.u64",
        // Extended-precision integer arithmetic.
        "add.cc.u32 addc.u32 sub.cc.u32 subc.u32 mad.lo.cc.u32 madc.lo.u32",
        // Floating point.
        "testp.finite.f32 copysign.f32 add.rn.f32 sub.rn.f32 mul.rn.f32 fma.rn.f32 mad.rn.f32 div.rn.f32 abs.f32 \
         neg.f32 min.f32 max.f32 rcp.rn.f32 rcp.approx.ftz.f64 sqrt.rn.f32 rsqrt.approx.f32 rsqrt.approx.ftz.f64 \
         sin.approx.f32 cos.approx.f32 lg2.approx.f32 ex2.approx.f32 tanh.approx.f32",
        // Half precision floating point.
        "add.rn.f16 sub.rn.f16 mul.rn.f16 fma.rn.f16 neg.f16 abs.f16 min.f16 max.f16 tanh.approx.f16 ex2.approx.f16",
        // Comparison and selection, and half precision comparison.
        "set.lt.u32.s32 setp.lt.s32 selp.s32 slct.s32.s32 set.lt.u32.f16 setp.lt.f16",
        // Logic and shift.
        "and.b32 or.b32 xor.b32 not.b32 cnot.b32 lop3.b32 shf.l.clamp.b32 shl.b32 shr.u32",
        // Data movement and conversion.
        "mov.u32 shfl.sync.idx.b32 prmt.b32 ld.global.f32 ld.global.nc.f32 ldu.global.f32 st.global.f32 \
         prefetch.global.L1 prefetchu.L1 applypriority.global.L2::evict_normal discard.global.L2 \
         createpolicy.fractional.L2::evict_last.b64 isspacep.global cvta.global.u64 cvt.rn.f32.f64 \
         cvt.pack.sat.u8.s32.b32 cp.async.ca.shared.global cp.async.commit_group cp.async.wait_group cp.async.wait_all",
        // Texture, and surface.
        "tex.2d.v4.f32.f32 tld4.r.2d.v4.f32.f32 txq.width.b32 istypep.texref suld.b.2d.b32.trap sust.b.2d.b32.trap \
         sust.p.2d.b32.trap sured.b.add.1d.u32.trap suq.width.b32",
        // Control flow.
        "bra brx.idx ret exit",
        // Parallel synchronization and communication.
        "bar.sync barrier.sync bar.warp.sync membar.gl fence.sc.gpu fence.proxy.alias atom.global.add.u32 \
         red.global.add.u32 vote.sync.all.pred match.any.sync.b32 activemask.b32 redux.sync.add.u32 \
         mbarrier.init.b64 mbarrier.inval.b64 mbarrier.arrive.b64 mbarrier.arrive_drop.b64 \
         cp.async.mbarrier.arrive.b64 mbarrier.test_wait.b64 mbarrier.pending_count.b64",
        // Warp level matrix multiply and accumulate.
        "wmma.load.a.sync.aligned.row.m16n16k16.f16 wmma.store.d.sync.aligned.row.m16n16k16.f32 \
         wmma.mma.sync.aligned.row.col.m16n16k16.f32.f32 mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 \
         ldmatrix.sync.aligned.m8n8.x1.b16 movmatrix.sync.aligned.m8n8.trans.b16 \
         mma.sp.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32",
        // Stack manipulation.
        "stacksave.u64 stackrestore.u64 alloca.u64",
        // Video, and SIMD video.
        "vadd.u32.u32.u32 vsub.u32.u32.u32 vabsdiff.u32.u32.u32 vmin.u32.u32.u32 vmax.u32.u32.u32 \
         vshl.u32.u32.u32.clamp vshr.u32.u32.u32.clamp vmad.u32.u32.u32 vset.u32.u32.lt vadd2.u32.u32.u32 \
         vsub2.u32.u32.u32 vavrg2.u32.u32.u32 vabsdiff2.u32.u32.u32 vmin2.u32.u32.u32 vmax2.u32.u32.u32 \
         vset2.u32.u32.lt vadd4.u32.u32.u32 vsub4.u32.u32.u32 vavrg4.u32.u32.u32 vabsdiff4.u32.u32.u32 \
         vmin4.u32.u32.u32 vmax4.u32.u32.u32 vset4.u32.u32.lt",
        // Miscellaneous.
        "brkpt nanosleep.u32 pmevent trap",
    ];
    let names: Vec<&str> = isa.iter().flat_map(|s| s.split_whitespace()).collect();
    for name in &names {
        if let Err(e) = forms::find(name) {
            panic!("{name}: {e}");
        }
    }
    for form in forms::forms() {
        let op = Some(form.op);
        assert!(names.iter().any(|n| n.split('.').next() == op), "no name here is of `{}`", form.line);
    }
}

/// Each instruction of PTX's that [`forms::NEWER`] has is none that kernite knows, and each word
/// none that a form of its instruction has, so that the error is the GPU it is for; and each GPU is
/// newer than kernite writes for.
#[test]
fn what_is_for_newer_gpus_says_which() {
    for (gpu, items) in forms::NEWER.lines().filter_map(|l| l.split_once(": ")) {
        let number: u32 = gpu["sm_".len()..].trim_end_matches(['a', 'f']).parse().unwrap();
        assert!(number > 86, "{gpu} is one kernite writes for");
        for item in items.split_whitespace() {
            let want = match item.split_once('.') {
                Some((op, w)) => {
                    format!("PTX's `{op}` takes `.{w}` on {gpu} and newer, and kernite writes PTX for sm_30 to sm_86")
                }
                None => format!("PTX's `{item}` is for {gpu} and newer, and kernite writes PTX for sm_30 to sm_86"),
            };
            assert_eq!(forms::find(item).err(), Some(want), "{item}");
        }
    }
}

#[test]
fn a_wrong_name_is_named_by_what_ptx_has() {
    let cases = [
        (
            "let v: f32; ld.f32.global(v, [x]);",
            "2:20: PTX writes `ld.f32.global` as `ld.global.f32`, its words in that order",
        ),
        ("let p: pred; setp.u32(p, n, n);", "2:19: `setp.u32` names one comparison, as in `setp.lt.u32`"),
        (
            "let i: u32; shf.l.b32(i, n, n, n);",
            "2:19: `shf.l.b32` leaves out one of `.clamp`, `.wrap`: `shf.l.clamp.b32`",
        ),
        (
            "let i: u32; add.sat.u32(i, n, n);",
            "2:21: PTX's `add` takes no `.sat` with the rest of `add.sat.u32`: `add.u32`",
        ),
        ("let i: u8; add.u8(i, i, i);", "2:16: PTX's `add` has no `.u8` that kernite knows"),
        (
            "let (a, b, c, e): (f64, f64, f64, f64); ld.global.v4.f64((a, b, c, e), [x]);",
            "2:54: `ld.global.v4.f64` moves 256 bits at once, which PTX's `ld` does on sm_100 and newer, and kernite \
             writes PTX for sm_30 to sm_86: a vector of 128 bits at most, as `.v4.b32` or `.v2.b64`",
        ),
        (
            "let v: f32; cvt.f32.f64(v, d);",
            "2:21: `cvt.f32.f64` loses precision, so it says how it rounds: `cvt.rn.f32.f64`",
        ),
        (
            "let i: u32; let s: u16; cvt.rn.u32.u16(i, s);",
            "2:36: `cvt.rn.u32.u16` converts an integer to an integer, which does not round: `cvt.u32.u16`",
        ),
        (
            "let i: u32; cvt.u32.f32(i, 1.5);",
            "2:21: `cvt.u32.f32` converts a float to an integer, so it says how it rounds to one: `cvt.rni.u32.f32` \
             to the nearest, `cvt.rzi` toward zero",
        ),
        ("elect.sync(n);", "2:7: PTX's `elect` is for sm_90 and newer, and kernite writes PTX for sm_30 to sm_86"),
        (
            "let (a, b): (f32, f32); ld.global.v8.f32((a, b, a, b, a, b, a, b), [x]);",
            "2:38: PTX's `ld` takes `.v8` on sm_100 and newer, and kernite writes PTX for sm_30 to sm_86",
        ),
        (
            "fence.proxy.r#async();",
            "2:13: PTX's `fence` takes `.async` on sm_90 and newer, and kernite writes PTX for sm_30 to sm_86",
        ),
        (
            "call.sync(n);",
            "2:6: PTX's `call` is `call`, or `call.uni` where the threads of a warp all call the same function; not \
             `call.sync`",
        ),
    ];
    for (body, want) in cases {
        assert_eq!(compile("x: *mut f32, n: u32, d: f64", body).unwrap_err(), want, "{body}");
    }
}

/// `shfl` and `vote` without `.sync`, which PTX deprecated in 6.0, are refused wherever they are written, as
/// `ptxas` refuses them from PTX 6.4 for `sm_70` and newer and warns of them for the rest: alone, in a module of PTX
/// 3.0 for `sm_30`; beside `nanosleep.u32`, which makes it PTX 6.2 and `sm_70`; and beside `tanh.approx.f32`, which
/// makes it PTX 7.0 and `sm_75`. Each error says why, and names the form with `.sync`, whose last operand is a mask of
/// lanes.
#[test]
fn shfl_and_vote_without_sync_are_refused_whichever_the_version_and_gpu() {
    let why = "is deprecated since PTX 6.0, and `ptxas` refuses it from PTX 6.4 on `sm_70` and newer, where the \
               threads of a warp need not run together";
    let cases = [
        ("shfl.up.b32", "shfl.sync.up.b32", "let v: u32; shfl.up.b32(v, n, 1, 0);", 21),
        ("shfl.down.b32", "shfl.sync.down.b32", "let v: u32; shfl.down.b32(v, n, 1, 31);", 23),
        ("shfl.bfly.b32", "shfl.sync.bfly.b32", "let v: u32; shfl.bfly.b32(v, n, 1, 31);", 23),
        ("shfl.idx.b32", "shfl.sync.idx.b32", "let v: u32; shfl.idx.b32(v, n, 0, 31);", 22),
        ("vote.all.pred", "vote.sync.all.pred", "let p: pred; vote.all.pred(p, p);", 23),
        ("vote.any.pred", "vote.sync.any.pred", "let p: pred; vote.any.pred(p, p);", 23),
        ("vote.uni.pred", "vote.sync.uni.pred", "let p: pred; vote.uni.pred(p, p);", 23),
        ("vote.ballot.b32", "vote.sync.ballot.b32", "let v: u32; let p: pred; vote.ballot.b32(v, p);", 38),
    ];
    for (name, canon, body, column) in cases {
        for (before, line) in [("", 2), ("nanosleep.u32(n);\n", 3), ("let w: f32; tanh.approx.f32(w, f);\n", 3)] {
            let want = format!(
                "{line}:{column}: `{name}` {why}: `{canon}` takes a mask of lanes, the threads that take part, as its \
                 last operand"
            );
            assert_eq!(compile("n: u32, f: f32", &format!("{before}{body}")).unwrap_err(), want, "{before}{body}");
        }
    }
}

/// The three names GUIDE.md gives: `tex.2d` is `tex._2d`, `ld.global.L2::64B` is
/// `ld.global.L2__64B`, and `cp.async` is `cp.r#async`.
#[test]
fn an_instructions_name_is_rusts_words() {
    let ptx = compile(
        "p: *mut u32, t: u64, x: f32, y: f32",
        "let (r, g, b, a): (f32, f32, f32, f32); tex._2d.v4.f32.f32((r, g, b, a), [t, (x, y)]); let v: b32; \
         ld.global.L2__64B.b32(v, [p]); let tile: shared<[u32; 4], 16>; cp.r#async.ca.shared.global([tile], [p], 4);",
    )
    .unwrap();
    for line in [
        "\ttex.2d.v4.f32.f32 {%r, %g, %b, %a}, [%t, {%x, %y}];\n",
        "\tld.global.L2::64B.b32 %v, [%p];\n",
        "\tcp.async.ca.shared.global [tile], [%p], 4;\n",
    ] {
        assert!(ptx.contains(line), "{line}{ptx}");
    }
}

#[test]
fn a_wrong_operand_is_named_by_what_ptx_lets_it_be() {
    let cases = [
        ("let i: u32; add.u32(i, n);", "2:17: `add.u32` takes 3 operands, not 2"),
        ("add.u32(_, n, n);", "2:9: `add.u32` has no result here that it may leave unwritten, `_`"),
        ("let i: u32; div.u32(i, !n, n);", "2:24: `div.u32` takes no operand inverted"),
        ("bar.sync(16);", "2:10: `bar.sync` takes 0 to 15 here; 16 is not one of them"),
        ("bar.sync(0, 33);", "2:13: `bar.sync` takes a multiple of 32 here; 33 is not one"),
        ("bar.arrive(0);", "2:5: `bar.arrive` takes 2 operands, not 1"),
        (
            "let q: b64; createpolicy.fractional.L2__evict_last.b64(q, 1.5);",
            "2:59: `createpolicy.fractional.L2::evict_last.b64` takes a fraction here, a multiple of 1/16, more than 0 \
             and at most 1; 1.5 is not one",
        ),
        (
            "let i: u32; lop3.b32(i, n, n, n, 256);",
            "2:34: `lop3.b32` takes 8 bits here, 0 to 255; 256 is not one of them",
        ),
        (
            "let i: u32; lop3.b32(i, n, n, n, n);",
            "2:34: `lop3.b32` wants an integer known now; this is a `u32` variable",
        ),
        (
            "applypriority.global.L2__evict_normal([x], 64);",
            "2:44: `applypriority.global.L2::evict_normal` takes 128 here, the one value PTX has for it; not 64",
        ),
        (
            "let a: u64; alloca.u64(a, 64, 3);",
            "2:31: `alloca.u64` takes a power of two here, 2^23 at most; 3 is not one",
        ),
        ("mbarrier.init.b64([x], 0);", "2:24: `mbarrier.init.b64` takes 1 to 1048575 here; 0 is not one of them"),
        (
            "cp.async.ca.shared.global([n], [x], 5);",
            "2:37: `cp.async.ca.shared.global` takes 4, 8 or 16 here; 5 is not one of them",
        ),
        (
            "cp.async.cg.shared.global([n], [x], 8);",
            "2:37: `cp.async.cg.shared.global` takes 16 here, the one value PTX has for it; not 8",
        ),
        (
            "cp.async.ca.shared.global([n], [x], 8, 9);",
            "2:40: `cp.async.ca.shared.global` takes 0 to 8 here; 9 is not one of them",
        ),
        (
            "let (a, b): (f32, f32); mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32((a, b), n, n, (a, b));",
            "2:74: `mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32` wants 4 operands taken together, each a `f32` \
             variable to write, or `_`; this is 2 operands taken together",
        ),
        (
            "let i: s32; mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32((i, i), 1, n, (0, 0));",
            "2:68: `mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32` wants a `u32` variable; this is the integer 1",
        ),
        (
            "let i: s32; mma.sp.sync.aligned.m16n8k64.row.col.s32.s8.s8.s32((i, i, i, i), (n, n, n, n), (n, n, n, n), \
             (0, 0, 0, 0), n, 1);",
            "2:123: `mma.sp.sync.aligned.m16n8k64.row.col.s32.s8.s8.s32` takes 0 here, the one value PTX has for it; \
             not 1",
        ),
        (
            "let (a, b, c, d): (f32, f32, f32, f32); let g: f16x2; mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32(\
             (a, b, c, d), (g, g, g, g), (g, g, g, g), (0, 0, 0, 0), n, 2);",
            "2:167: `mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32` takes 0 or 1 here; 2 is not one of them",
        ),
        (
            "movmatrix.sync.aligned.m8n8.trans.b16(_, n);",
            "2:39: `movmatrix.sync.aligned.m8n8.trans.b16` has no result here that it may leave unwritten, `_`",
        ),
        (
            "let g: f16; add.rn.f16(g, h, 1.5);",
            "2:30: `add.rn.f16` takes a `f16` only from a variable: `mov.b16(h, 1.5f16)` puts one in it",
        ),
        ("let v: f64; mov.f64(v, 1.5);", "2:24: `mov.f64` wants an `f64`; 1.5 is an `f32`: `1.5f64`"),
        ("let i: u16; mov.u16(i, clock);", "2:24: `mov.u16` reads a `u16`; `clock` is a `u32`"),
        (
            "let i: u32; mov.u32(i, clusterid.x);",
            "2:34: `clusterid.x` is a special register of PTX's that kernite does not read: it needs sm_90, and kernite \
             writes PTX for sm_30 to sm_86",
        ),
        (
            "let d: s32; vmad.s32.s32.s32(d, -n, n, -n);",
            "2:40: `vmad.s32.s32.s32` negates `a * b`, as `-a` or `-b` does, or `c`; not both",
        ),
        (
            "let d: u32; vadd2.u32.u32.u32(d, n.b0, n, n);",
            "2:36: `vadd2.u32.u32.u32` takes two of the four halves of the two sources here, as `.h10` or `.h32`; \
             `.b0` is not one",
        ),
        (
            "let d: u32; vadd4.u32.u32.u32(d.b01, n, n, n);",
            "2:33: `vadd4.u32.u32.u32` takes the bytes of the result it writes here, from `.b3` down to `.b0`, each \
             once at most, as `.b31`; `.b01` is not one",
        ),
        (
            "let d: u32; add.u32(d, n.b0, n);",
            "2:26: `add.u32` takes a whole variable here; this is a part of one, `.b0`",
        ),
        (
            "let d: u32; vadd4.u32.u32.u32(d, n, n, 1);",
            "2:40: `vadd4.u32.u32.u32` wants a `u32` variable; this is the integer 1",
        ),
        ("let d: u32; vadd.u32.u32.u32(d, -n, n);", "2:33: `vadd.u32.u32.u32` takes no operand negated"),
        (
            "let (d, w): (u32, u64); mov.u64(w, 0); vadd.u32.u32.u32(d, w.b0, n);",
            "2:62: a video instruction takes a part of a 32-bit variable, as `.b0`; this is a `u64`",
        ),
        (
            "brx.idx(n, a); label(a);",
            "2:12: `brx.idx` wants labels taken together, as `(a, b)`, for an index to pick one; this is a label",
        ),
        (
            "brx.idx(n, (a, n)); label(a);",
            "2:12: `brx.idx` jumps to one of labels taken together; this holds a `u32` variable",
        ),
        ("brx.idx(1, (a,)); label(a);", "2:9: `brx.idx` takes 0 here, the index of its one label; not 1"),
        (
            "brx.idx(2, (a, b)); label(a); label(b);",
            "2:9: `brx.idx` takes 0 to 1 here, the index of one of its 2 labels; 2 is not one",
        ),
        (
            "let w: u64; mov.u64(w, 0); brx.idx(w, (a,)); label(a);",
            "2:36: `brx.idx` wants a `u32` variable; this is a `u64`",
        ),
        (
            "let i: u32; mov.u32(i, perctamemorysize);",
            "2:24: `perctamemorysize` is a special register of PTX's that kernite does not read: a kernel has it only \
             with the directive `.minperctamemory`, which kernite does not write",
        ),
        (
            "let i: u32; suq.width.b32(i, s);",
            "2:30: `suq.width.b32` wants a surface's handle in brackets, `[t]`; this is a `u64` variable",
        ),
        (
            "let i: u32; suld.b._2d.b32.trap(i, [s]);",
            "2:36: `suld.b.2d.b32.trap` wants a surface's handle and the coordinates in it, in brackets: `[t, (x, \
             y)]`; this is 1 operand in brackets",
        ),
        (
            "let i: u32; suld.b._1d.b32.trap(i, [s + 4]);",
            "2:36: `suld.b.1d.b32.trap` wants a surface's handle and the coordinates in it, in brackets: `[t, x]`; \
             this is an address",
        ),
        (
            "let i: u32; suld.b.a1d.b32.trap(i, [s, n]);",
            "2:36: `suld.b.a1d.b32.trap` wants 2 operands taken together, a `u32` variable, a value or a special \
             register and then a `s32` variable, a value or a special register; this is a `u32` variable",
        ),
        (
            "let (i, f): (u32, f32); suld.b.a1d.b32.trap(i, [s, (f, n)]);",
            "2:48: `suld.b.a1d.b32.trap` wants a `u32` variable; this is a `f32`",
        ),
        // ptxas 13.5 refuses an `f16x2` for the layer of an array of textures or surfaces, "Array index in the 'tex'
        // instruction has unexpected type", and for a coordinate beside an `s32`: kernite refuses both. It refuses it
        // for the sample of `tex.a2dms`, the first of its vector, too, whose type `ptxas` does not check: that looks
        // like a bug of its own.
        (
            "let (h, i): (f16x2, u32); mov.b32(h, n); suld.b.a1d.b32.trap(i, [s, (h, n)]);",
            "2:65: `suld.b.a1d.b32.trap` wants a `u32` variable; this is a `f16x2`",
        ),
        (
            "let (h, i): (f16x2, u32); mov.b32(h, n); mov.u32(i, n); sust.b.a1d.b32.trap([s, (h, n)], i);",
            "2:77: `sust.b.a1d.b32.trap` wants a `u32` variable; this is a `f16x2`",
        ),
        (
            "let (h, a, b, c, d): (f16x2, f32, f32, f32, f32); mov.b32(h, n); \
             tex.a2d.v4.f32.s32((a, b, c, d), [s, (h, n, n, n)]);",
            "2:99: `tex.a2d.v4.f32.s32` wants a `u32` variable; this is a `f16x2`",
        ),
        (
            "let (h, a, b, c, d): (f16x2, s32, s32, s32, s32); mov.b32(h, n); \
             tex.a2dms.v4.s32.s32((a, b, c, d), [s, (h, n, n, n)]);",
            "2:101: `tex.a2dms.v4.s32.s32` wants a `s32` variable; this is a `f16x2`",
        ),
        (
            "let (h, a, b, c, d): (f16x2, f32, f32, f32, f32); mov.b32(h, n); \
             tex.a2d.v4.f32.s32((a, b, c, d), [s, (n, h, n, n)]);",
            "2:99: `tex.a2d.v4.f32.s32` takes the operands of a vector of one type, bits beside any and an integer \
             beside an integer; this is a `u32` variable beside a `f16x2` variable",
        ),
        (
            "let (i, w): (u64, s64); mov.s64(w, 0); suld.b._1d.b32.trap(i, [s, w]);",
            "2:63: `suld.b.1d.b32.trap` wants a `s32` variable; this is a `s64`",
        ),
        (
            "let i: u32; suld.b._1d.b32.trap(i, [n, n]);",
            "2:36: `suld.b.1d.b32.trap` wants a `u64` variable; this is a `u32`",
        ),
        (
            "suld.b._1d.b32.trap(_, [s, n]);",
            "2:21: `suld.b.1d.b32.trap` writes its one result; `_` stands for one of a vector's, as `.v2` and `.v4` \
             write",
        ),
        (
            "let v: f32; sured.b.add._1d.u32.trap([s, 4], v);",
            "2:46: `sured.b.add.1d.u32.trap` wants a `u32` variable; this is a `f32`",
        ),
        ("sured.b.xor._1d.b32.trap([s, n], n);", "2:21: PTX's `sured` has no `.xor` that kernite knows"),
        (
            "let v: b64; suld.b._1d.v4.b64.trap((v, v, v, v), [s, n]);",
            "2:31: `suld.b.1d.v4.b64.trap` moves 256 bits at once; a vector of PTX's holds 128 at most, as \
             `.v4.b32` or `.v2.b64`",
        ),
        (
            "let (a, b): (f32, f32); tex._1d.v2.f32.f32((a, b), [s, z]);",
            "2:40: `tex.1d.v2.f32.f32` returns four of a `f32`, `.v4`; two only of an `f16x2`, `.v2`",
        ),
        (
            "let g: f16x2; tex._1d.v2.f16x2.f32((g, _), [s, z]);",
            "2:36: `tex.1d.v2.f16x2.f32` has no result here that it may leave unwritten, `_`",
        ),
        (
            "let (a, b, c, d): (f32, f32, f32, f32); tex._2d.v4.f32.f32((a, b, c, d), [s, (z, z)], (8, 0));",
            "2:87: `tex.2d.v4.f32.f32` takes -8 to 7 here; 8 is not one of them",
        ),
        (
            "let (a, b, c, d): (f32, f32, f32, f32); tex.a1d.v4.f32.f32((a, b, c, d), [s, (z, z)]);",
            "2:74: `tex.a1d.v4.f32.f32` wants a `u32` variable; this is a `f32`",
        ),
        (
            "let (a, b, c, d): (s32, s32, s32, s32); tex.level._2dms.v4.s32.s32((a, b, c, d), [s, (n, n, n, n)], n);",
            "2:64: PTX's `tex` takes no `.level` with the rest of `tex.level.2dms.v4.s32.s32`: `tex.2dms.v4.s32.s32`",
        ),
        (
            "let (a, b, c, d): (f32, f32, f32, f32); tld4.r._2d.v4.f32.s32((a, b, c, d), [s, (n, n)]);",
            "2:59: `tld4.r.2d.v4.f32.s32` is no form of `tld4` that kernite knows; the nearest is `tld4.r.2d.v4.f32.f32`",
        ),
        ("let i: u32; txq.level.width.b32(i, [s]);", "2:29: `txq.level.width.b32` takes 3 operands, not 2"),
        (
            "let i: u32; txq.force_unnormalized_coords.b32(i, [s]);",
            "2:43: PTX's `txq.force_unnormalized_coords` asks of a sampler apart from its texture, in the independent \
             mode, which kernite does not write",
        ),
        (
            "let p: pred; istypep.texref(p, [s]);",
            "2:32: `istypep.texref` wants a `u64` variable or a value; this is an address",
        ),
        (
            "let i: u32; ld.local.u32(i, [256]);",
            "2:29: `ld.local.u32` takes an address that starts at a variable, as `[p + 4]`; this is memory holding \
             a value known now",
        ),
        (
            "st.global.u32([x], tid.x);",
            "2:24: `st.global.u32` stores a special register as one of a vector's, as `.v2` and `.v4` store; \
             `mov.u32` puts `tid.x` in a variable",
        ),
        ("let f: f32; cvt.rn.f32.u32(f, tid.x);", "2:35: `cvt.rn.f32.u32` reads no special register; `mov.u32` does"),
        (
            "st.global.b16([x], laneid);",
            "2:20: `st.global.b16` stores a special register as one of a vector's, as `.v2` and `.v4` store; \
             `mov.u32` puts `laneid` in a variable",
        ),
        ("let f: f32; cvt.rn.f32.u16(f, laneid);", "2:31: `cvt.rn.f32.u16` reads no special register; `mov.u32` does"),
        (
            "let (a, b, c, d): (s32, s32, s32, s32); mma.sync.aligned.m16n8k16.row.col.s32.u8.u8.s32((a, b, c, d), \
             (tid.x, 8), n, (0, 0, 0, 0));",
            "2:111: `mma.sync.aligned.m16n8k16.row.col.s32.u8.u8.s32` takes no part of a special register beside a \
             value known now, as `tid.x` is of `tid`; this is the integer 8 beside `tid.x`",
        ),
        (
            "let (a, b, c, d): (s32, s32, s32, s32); mma.sync.aligned.m16n8k16.row.col.s32.u8.u8.s32((a, b, c, d), \
             (7, 8), n, (0, 0, 0, 0));",
            "2:103: `mma.sync.aligned.m16n8k16.row.col.s32.u8.u8.s32` takes values known now among these only beside \
             a variable",
        ),
        ("let d: u32; and.b32(d, !n, n);", "2:24: `and.b32` takes no operand inverted"),
        ("bar.sync(0, 1056);", "2:13: `bar.sync` takes 0 to 1024 here; 1056 is not one of them"),
        ("bar.arrive(0, 2048);", "2:15: `bar.arrive` takes 32 to 1024 here; 2048 is not one of them"),
        ("bar.warp.sync(0);", "2:15: `bar.warp.sync` takes 1 to 4294967295 here; 0 is not one of them"),
        (
            "let q: b64; createpolicy.fractional.L2__evict_last.b64(q, 0.1);",
            "2:59: `createpolicy.fractional.L2::evict_last.b64` takes a fraction here, a multiple of 1/16, more than \
             0 and at most 1; 0.1 is not one",
        ),
        (
            "let q: b64; createpolicy.range.L2__evict_last.b64(q, [x], 128, 64);",
            "2:59: `createpolicy.range.L2::evict_last.b64` takes a primary size here no more than the total, 64; 128 \
             is more",
        ),
        ("let i: u32; bfe.u32(i, n, 256, 8);", "2:27: `bfe.u32` takes 0 to 255 here; 256 is not one of them"),
        ("let i: u32; bfi.b32(i, n, n, 0, 300);", "2:33: `bfi.b32` takes 0 to 255 here; 300 is not one of them"),
        ("cp.async.wait_group(64);", "2:21: `cp.async.wait_group` takes 6 bits here, 0 to 63; 64 is not one of them"),
        (
            "let i: u32; mul24.lo.u32(i, n, 0x100_0001);",
            "2:32: `mul24.lo.u32` takes 0 to 16777215 here; 16777217 is not one of them",
        ),
        (
            "let i: s32; mad24.hi.s32(i, i, 8388608, i);",
            "2:32: `mad24.hi.s32` takes -8388608 to 8388607 here; 8388608 is not one of them",
        ),
        (
            "let i: u32; shfl.sync.idx.b32(i, n, 32, 0x1f, 0xffff_ffff);",
            "2:37: `shfl.sync.idx.b32` takes 0 to 31 here; 32 is not one of them",
        ),
        (
            "let i: u32; shfl.sync.idx.b32(i, n, 0, 0x201f, 0xffff_ffff);",
            "2:40: `shfl.sync.idx.b32` takes no bits here but those of 0x1f1f; 0x201f has others",
        ),
        (
            "let i: u32; shfl.sync.idx.b32(i, n, 0, -1, 0xffff_ffff);",
            "2:40: `shfl.sync.idx.b32` takes no bits here but those of 0x1f1f; 0xffffffff has others",
        ),
        (
            "let i: u32; shfl.sync.idx.b32(i, n, 0, 0x1f, 0);",
            "2:46: `shfl.sync.idx.b32` takes 1 to 4294967295 here; 0 is not one of them",
        ),
        (
            "let (i, p): (u32, pred); vote.sync.ballot.b32(i, p, 0);",
            "2:53: `vote.sync.ballot.b32` takes 1 to 4294967295 here; 0 is not one of them",
        ),
        (
            "let i: u32; prmt.b32(i, n, n, 0x1_0000);",
            "2:31: `prmt.b32` takes 0 to 65535 here; 65536 is not one of them",
        ),
        ("let i: u32; prmt.b32.f4e(i, n, n, 4);", "2:35: `prmt.b32.f4e` takes 0 to 3 here; 4 is not one of them"),
        ("let i: u32; fns.b32(i, n, 32, 1);", "2:27: `fns.b32` takes 0 to 31 here; 32 is not one of them"),
        (
            "let (i, s): (u32, s32); cvt.pack.sat.u8.s32.b32(i, s, s, 0x1_0000);",
            "2:58: `cvt.pack.sat.u8.s32.b32` takes 0 to 65535 here; 65536 is not one of them",
        ),
        (
            "let (i, s): (u32, s32); cvt.pack.sat.u2.s32.b32(i, s, s, 0x1000_0000);",
            "2:58: `cvt.pack.sat.u2.s32.b32` takes 0 to 268435455 here; 268435456 is not one of them",
        ),
        (
            "let i: u32; add.u32(i, n, 0xffff_ffff_ffff_ffff);",
            "2:27: `add.u32` wants 32 bits; 18446744073709551615 does not fit them",
        ),
    ];
    for (body, want) in cases {
        assert_eq!(compile("x: *mut f32, n: u32, h: f16, s: u64, z: f32", body).unwrap_err(), want, "{body}");
    }
}

/// The mask of lanes that `name` is refused for: thread 0 is in lane 0, and the mask has not.
fn without_lane_0(name: &str) -> String {
    format!(
        "`{name}` is run by thread 0, which is in lane 0; this mask leaves lane 0 out, and PTX leaves the \
         instruction undefined in a thread outside its mask"
    )
}

#[test]
fn a_mask_known_now_without_lane_0_is_refused_where_thread_0_runs_the_instruction() {
    // Every kernel has a thread 0, and it is in lane 0. An instruction that no guard, `ret`,
    // call or jump keeps it from is run by it, with the mask given.
    assert_eq!(
        compile("n: u32", "bar.warp.sync(2);").unwrap_err(),
        "2:10: `bar.warp.sync` is run by thread 0, which is in lane 0; this mask leaves lane 0 out, and PTX leaves \
         the instruction undefined in a thread outside its mask"
    );
    let cases = [
        // Each instruction that has a mask of lanes, at the start of the kernel.
        ("let i: u32; shfl.sync.idx.b32(i, n, 0, 0x1f, 0xffff_fffe);", "2:27", "shfl.sync.idx.b32"),
        ("let (p, q): (pred, pred); setp.eq.u32(p, n, 0); vote.sync.any.pred(q, p, 6);", "2:63", "vote.sync.any.pred"),
        ("let i: b32; r#match.any.sync.b32(i, n, 0x8000_0000);", "2:30", "match.any.sync.b32"),
        ("let i: u32; redux.sync.add.u32(i, n, 0xffff_0000);", "2:28", "redux.sync.add.u32"),
        // After an `if` that guards another instruction: thread 0 may skip that one, and runs
        // this.
        (
            "let (i, p): (u32, pred); setp.eq.u32(p, n, 0); if p { mov.u32(i, 1); } bar.warp.sync(2);",
            "2:81",
            "bar.warp.sync",
        ),
        // After a jump that comes back before it: the loop has a way out, which every thread
        // takes in the end.
        (
            "let p: pred; label(again); setp.eq.u32(p, n, 0); if p { bra(again); } bar.warp.sync(2);",
            "2:80",
            "bar.warp.sync",
        ),
        // In the body of a loop, each round of which runs it.
        (
            "let (i, p): (u32, pred); mov.u32(i, 0); label(top); bar.warp.sync(2); add.u32(i, i, 1); \
             setp.lt.u32(p, i, n); if p { bra(top); }",
            "2:62",
            "bar.warp.sync",
        ),
        // The first of two.
        ("bar.warp.sync(2); bar.warp.sync(4);", "2:10", "bar.warp.sync"),
    ];
    for (body, at, name) in cases {
        let want = format!("{at}: {}", without_lane_0(name));
        assert_eq!(compile("x: *mut f32, n: u32", body).unwrap_err(), want, "{body}");
    }
}

#[test]
fn a_mask_without_lane_0_is_taken_where_thread_0_may_not_run_the_instruction() {
    let cases = [
        // Under a guard.
        "let p: pred; setp.eq.u32(p, n, 0); if p { bar.warp.sync(2); }",
        // After a `ret` that a guard keeps from some threads.
        "let p: pred; setp.eq.u32(p, n, 0); if p { ret(); } bar.warp.sync(2);",
        "let p: pred; setp.eq.u32(p, n, 0); if p { exit(); } bar.warp.sync(2);",
        "let p: pred; setp.eq.u32(p, n, 0); if p { trap(); } bar.warp.sync(2);",
        // And with a label between, where a block of its own begins.
        "let p: pred; setp.eq.u32(p, n, 0); if p { ret(); } label(on); bar.warp.sync(2);",
        // Behind a jump that skips it.
        "let p: pred; setp.eq.u32(p, n, 0); if p { bra(done); } bar.warp.sync(2); label(done);",
        // In a loop that has a way out, but not every round.
        "let (i, p): (u32, pred); mov.u32(i, 0); label(top); setp.lt.u32(p, i, n); if p { bra(skip); } \
         bar.warp.sync(2); label(skip); add.u32(i, i, 1); setp.lt.u32(p, i, n); if p { bra(top); }",
        // Beside a loop that has no way out, which a thread may go into instead, and after one.
        "let p: pred; setp.eq.u32(p, n, 0); if p { bra(spin); } bar.warp.sync(2); ret(); label(spin); bra(spin);",
        "label(spin); bra(spin); bar.warp.sync(2);",
        // With lane 0 in the mask, however many others.
        "bar.warp.sync(1);",
        "bar.warp.sync(3);",
        "bar.warp.sync(0xffff_ffff);",
        "bar.warp.sync(-1);",
        // A mask that is not known now.
        "let m: u32; mov.u32(m, 2); bar.warp.sync(m);",
        "bar.warp.sync(n);",
    ];
    for body in cases {
        if let Err(e) = compile("x: *mut f32, n: u32", body) {
            panic!("{body}: {e}");
        }
    }
    // After a call, which may end the thread, or keep it for ever.
    let ptx = kernite::ptx_str(
        "unsafe extern \"ptx-kernel\" fn k() { call(nothing); bar.warp.sync(2); }\n\
         extern \"ptx-func\" fn nothing() {}\n",
    )
    .unwrap();
    assert!(ptx.contains("\tcall nothing;\n\tbar.warp.sync 2;\n"), "{ptx}");
    // In a function kernels call, which thread 0 runs wherever it runs a call of it.
    let ptx = kernite::ptx_str(
        "unsafe extern \"ptx-kernel\" fn k() { call(warp); }\n\
         extern \"ptx-func\" fn warp() { bar.warp.sync(2); }\n",
    )
    .unwrap();
    assert!(ptx.contains("\tbar.warp.sync 2;\n\tret;\n}\n"), "{ptx}");
}

/// Each form with a mask of lanes is marked: those of `bar.warp.sync`, `vote.sync`, `match.sync`,
/// `redux.sync` and `shfl.sync`, and no others.
#[test]
fn a_mask_of_lanes_is_marked_in_each_form_that_has_one() {
    let with = ["bar.warp.sync", "vote.sync", "match.", "redux.sync", "shfl.sync"];
    let mut marked = 0;
    for form in forms::forms() {
        let has = form.operands.iter().any(|s| s.lanes);
        assert_eq!(has, with.iter().any(|w| form.line.starts_with(w)), "{}", form.line);
        marked += usize::from(has);
    }
    assert_eq!(marked, 8);
}

#[test]
fn a_module_is_of_the_oldest_ptx_and_gpu_its_instructions_need() {
    let cases = [
        ("", ".version 3.0 .target sm_30"),
        ("let i: u32; popc.b32(i, n);", ".version 3.0 .target sm_30"),
        ("let i: u32; shfl.sync.idx.b32(i, n, 0, 31, 0xffff_ffff);", ".version 6.0 .target sm_30"),
        (
            "let (i, p): (u32, pred); setp.eq.u32(p, n, 0); vote.sync.ballot.b32(i, p, 0xffff_ffff);",
            ".version 6.0 .target sm_30",
        ),
        ("let i: u32; vadd.u32.u32.u32(i, n.b0, n.h1);", ".version 3.0 .target sm_30"),
        ("let i: u32; vset4.u32.u32.lt(i, n, n, n);", ".version 3.0 .target sm_30"),
        ("let h: f16x2;", ".version 4.2 .target sm_53"),
        ("let i: u32; dp4a.u32.u32(i, n, n, n);", ".version 5.0 .target sm_61"),
        ("brx.idx(n, (a, b)); label(a); label(b);", ".version 6.0 .target sm_30"),
        ("nanosleep.u32(n);", ".version 6.2 .target sm_70"),
        ("let i: u32; shfl.sync.idx.b32(i, n, 0, 31, 0xffff_ffff); nanosleep.u32(n);", ".version 6.2 .target sm_70"),
        ("let v: f32; tanh.approx.f32(v, f);", ".version 7.0 .target sm_75"),
        ("let d: b32; ldmatrix.sync.aligned.m8n8.x1.b16(d, [y]);", ".version 6.5 .target sm_75"),
        ("let d: b32; movmatrix.sync.aligned.m8n8.trans.b16(d, n);", ".version 7.8 .target sm_75"),
        (
            "let (a, b, c, d): (s32, s32, s32, s32); mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32((a, b, c, d), \
             (n, n), n, (0, 0, 0, 0));",
            ".version 7.0 .target sm_80",
        ),
        ("cp.async.wait_all();", ".version 7.0 .target sm_80"),
        ("mbarrier.arrive.b64(_, [y]);", ".version 7.1 .target sm_80"),
        ("let i: b32; mov.b32(i, reserved_smem_offset_begin);", ".version 7.6 .target sm_80"),
        (
            "let (a, b): ([f16x2; 8], [f16x2; 8]); let c: [f32; 8]; wmma.load.a.sync.aligned.row.m16n16k16.f16(a, \
             [y]); wmma.load.b.sync.aligned.col.m16n16k16.f16(b, [y], 16); wmma.load.c.sync.aligned.row.m16n16k16.f32(\
             c, [y]); wmma.mma.sync.aligned.row.col.m16n16k16.f32.f32(c, a, b, c); \
             wmma.store.d.sync.aligned.row.m16n16k16.f32([y], c, n);",
            ".version 6.3 .target sm_70",
        ),
        ("let a: u32; wmma.load.a.sync.aligned.row.m8n8k32.s4(a, [y]);", ".version 6.3 .target sm_75"),
        (
            "let (a, b, c, d): (s32, s32, s32, s32); mma.sp__ordered_metadata.sync.aligned.m16n8k64.row.col.s32.s8.s8.s32(\
             (a, b, c, d), (n, n, n, n), (n, n, n, n), (0, 0, 0, 0), n, 0);",
            ".version 8.5 .target sm_80",
        ),
        ("let v: f32; max.xorsign.abs.f32(v, f, f);", ".version 7.2 .target sm_86"),
        ("let v: b128; ld.global.b128(v, [x]);", ".version 8.3 .target sm_70"),
        // The ISA has it from sm_75, and ptxas takes it for sm_70.
        ("let i: u32; ld.mmio.acquire.sys.global.u32(i, [y]);", ".version 9.3 .target sm_70"),
        ("let a: u64; clmad.lo.u64(a, y, y, y);", ".version 9.3 .target sm_80"),
        ("let i: u32; suld.b._2d.b32.trap(i, [y, (n, n)]);", ".version 3.1 .target sm_30"),
        ("let i: u32; suq.memory_layout.b32(i, [y]);", ".version 4.2 .target sm_30"),
        ("sured.b.max._1d.u64.trap([y, 0], y);", ".version 8.1 .target sm_50"),
        (
            "let (a, b, c, d): (f32, f32, f32, f32); tex._2d.v4.f32.f32((a, b, c, d), [y, (f, f)]);",
            ".version 3.1 .target sm_30",
        ),
        (
            "let (a, b, c, d): (s32, s32, s32, s32); tex.a2dms.v4.s32.s32((a, b, c, d), [y, (n, n, n, n)]);",
            ".version 3.2 .target sm_30",
        ),
        (
            "let (a, b, c, d): (f32, f32, f32, f32); tex._2d.v4.f32.f32((a, b, c, d), [y, (f, f)], (1, -1));",
            ".version 4.3 .target sm_30",
        ),
        (
            "let (a, b, c, d): (f32, f32, f32, f32); tex.grad.cube.v4.f32.f32((a, b, c, d), [y, (f, f, f, f)], (f, f, \
             f, f), (f, f, f, f));",
            ".version 4.3 .target sm_30",
        ),
        (
            "let (a, b, c, d): (f32, f32, f32, f32); let p: pred; tex._2d.v4.f32.f32((a, b, c, d) | p, [y, (f, f)]);",
            ".version 7.1 .target sm_60",
        ),
        ("let i: u32; txq.num_samples.b32(i, [y]);", ".version 4.1 .target sm_30"),
        ("let p: pred; istypep.texref(p, y);", ".version 4.0 .target sm_30"),
        ("mov.u64(y, k);", ".version 3.1 .target sm_35"),
    ];
    for (body, want) in cases {
        let ptx = compile("x: *mut f32, n: u32, f: f32, y: u64", body).unwrap();
        let header: Vec<&str> = ptx.lines().filter(|l| l.starts_with(".version") || l.starts_with(".target")).collect();
        assert_eq!(header.join(" "), want, "{body}");
    }
}

#[test]
fn a_fragment_is_braced_even_when_it_is_one_register() {
    let ptx = compile(
        "y: u64, n: u32",
        "let (d, e): (b32, b32); ldmatrix.sync.aligned.m8n8.x1.b16(d, [y]); ldmatrix.sync.aligned.m8n8.x1.b16((e,), \
         [y]); let (a, b): (s32, s32); mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32((a, b), d, (e,), (0, n));",
    )
    .unwrap();
    for line in [
        "\tldmatrix.sync.aligned.m8n8.x1.b16 {%d}, [%y];\n",
        "\tldmatrix.sync.aligned.m8n8.x1.b16 {%e}, [%y];\n",
        "\tmma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%a, %b}, {%d}, {%e}, {0, %n};\n",
    ] {
        assert!(ptx.contains(line), "{line}{ptx}");
    }
}

#[test]
fn an_operand_is_written_wherever_ptxas_takes_it() {
    let ptx = compile(
        "x: *mut f32, n: u32",
        "let d: u32; selp.u32(d, n, 0, 7); st.global.v2.u32([x], (tid.x, tid.y)); let (a, b, c, e): (s32, s32, s32, \
         s32); mma.sync.aligned.m16n8k16.row.col.s32.u8.u8.s32((a, b, c, e), (n, 8), n, (0, 0, 0, 0)); \
         cp.async.ca.shared.global([n], [n], 4); brx.idx(1, (one, two)); label(one); label(two); \
         mul24.lo.s32(a, a, 0xffff_ffff);",
    )
    .unwrap();
    for line in [
        "\tselp.u32 %d, %n, 0, 1;\n",
        "\tst.global.v2.u32 [%x], {%tid.x, %tid.y};\n",
        "\tmma.sync.aligned.m16n8k16.row.col.s32.u8.u8.s32 {%a, %b, %c, %e}, {%n, 8}, {%n}, {0, 0, 0, 0};\n",
        "\tcp.async.ca.shared.global [%n], [%n], 4;\n",
        "\tbrx.idx 1, $T0;\n",
        "\tmul24.lo.s32 %a, %a, 4294967295;\n",
    ] {
        assert!(ptx.contains(line), "{line}{ptx}");
    }
}

#[test]
fn a_part_of_a_variable_is_written_after_it() {
    let ptx = compile(
        "n: u32",
        "let (d, e): (u32, s32); vadd.u32.u32.u32.sat(d.h1, n.b0, n.h1, n); vmad.s32.u32.u32.shr7(e, -n.b3, n, n); \
         vadd4.u32.u32.u32(d.b32, n.b7654, n.b3210, d);",
    )
    .unwrap();
    for line in [
        "\tvadd.u32.u32.u32.sat %d.h1, %n.b0, %n.h1, %n;\n",
        "\tvmad.s32.u32.u32.shr7 %e, -%n.b3, %n, %n;\n",
        "\tvadd4.u32.u32.u32 %d.b32, %n.b7654, %n.b3210, %d;\n",
    ] {
        assert!(ptx.contains(line), "{line}{ptx}");
    }
}

/// Kernels that jump through tables of labels, under a guard and not.
const TABLES: &str = r#"
    /// `x[0] = 1`, `2` or `3` as `n` is 0, 1 or 2, and nothing for another `n`.
    pub unsafe extern "ptx-kernel" fn pick(x: *mut f32, n: u32) {
        let p: pred;
        setp.lt.u32(p, n, 3);
        if p {
            brx.idx.uni(n, (one, two, three));
        }
        ret();
        let v: f32;
        label(one);
        mov.f32(v, 1);
        bra(done);
        label(two);
        mov.f32(v, 2);
        bra(done);
        label(three);
        mov.f32(v, 3);
        label(done);
        st.global.f32([x], v);
    }

    /// `x[0] = 1` where `n` is even, and `2` where it is odd.
    pub unsafe extern "ptx-kernel" fn parity(x: *mut f32, n: u32) {
        let i: u32;
        and.b32(i, n, 1);
        let v: f32;
        brx.idx(i, (even, odd));
        label(even);
        mov.f32(v, 1);
        bra(done);
        label(odd);
        mov.f32(v, 2);
        label(done);
        st.global.f32([x], v);
    }
"#;

#[test]
fn a_jump_through_a_table_declares_the_table_before_it() {
    let ptx = kernite::ptx_str(TABLES).unwrap();
    let (head, entries) = ptx.split_once(".visible .entry ").unwrap();
    assert!(head.contains(".version 6.0\n.target sm_30\n"), "{head}");
    let (pick, parity) = entries.split_once(".visible .entry ").unwrap();
    let line = "\t$T0: .branchtargets $L0, $L1, $L2;\n\t@%p brx.idx.uni %n, $T0;\n\tret;\n$L0:\n";
    assert!(pick.contains(line), "{line}{pick}");
    let line = "\t$T0: .branchtargets $L0, $L1;\n\tbrx.idx %i, $T0;\n$L0:\n";
    assert!(parity.contains(line), "{line}{parity}");
}

const SURFACES: &str = r#"
    /// The texel at the thread's `(x, y)` of the 2D surface `from`, four `f32`s, copied to the
    /// same place in `to`.
    pub unsafe extern "ptx-kernel" fn copy(from: u64, to: u64) {
        let (x, y): (u32, u32);
        mov.u32(x, tid.x);
        mov.u32(y, tid.y);
        shl.b32(x, x, 4);
        let (r, g, b, a): (f32, f32, f32, f32);
        suld.b._2d.v4.b32.clamp((r, g, b, a), [from, (x, y)]);
        sust.b._2d.v4.b32.clamp([to, (x, y)], (r, g, b, a));
    }

    /// One more in the bin `x[i]` of the 1D surface `bins`, `u32`s, for each thread `i` of the
    /// block, and none for a bin past the last.
    pub unsafe extern "ptx-kernel" fn histogram(x: *const u32, bins: u64) {
        let i: u32;
        mov.u32(i, tid.x);
        let at: u64;
        mad.wide.u32(at, i, 4, x);
        let (v, width): (u32, u32);
        ld.global.u32(v, [at]);
        suq.width.b32(width, [bins]);
        let p: pred;
        setp.ge.u32(p, v, width);
        if p {
            ret();
        }
        shl.b32(v, v, 2);
        sured.b.add._1d.u32.trap([bins, v], 1);
    }
"#;

const TEXTURES: &str = r#"
    /// `y[i]`, for each thread `i` of the block, the first of the four `f32`s of the 2D
    /// texture `t` at the middle of its texel `(i, 0)`, of its level of detail 0, where it
    /// takes coordinates in texels.
    pub unsafe extern "ptx-kernel" fn sample(t: u64, y: *mut f32) {
        let i: u32;
        mov.u32(i, tid.x);
        let x: f32;
        cvt.rn.f32.u32(x, i);
        add.rn.f32(x, x, 0.5);
        let r: f32;
        tex.level._2d.v4.f32.f32((r, _, _, _), [t, (x, 0.5)], 0);
        let at: u64;
        mad.wide.u32(at, i, 4, y);
        st.global.f32([at], r);
    }

    /// `y[0]` to `y[3]`, the greens of the four texels around the middle of the 2D texture
    /// `t`, where it takes coordinates in texels, and `w[0]` its width.
    pub unsafe extern "ptx-kernel" fn gather(t: u64, y: *mut f32, w: *mut u32) {
        let (width, height): (u32, u32);
        txq.width.b32(width, [t]);
        txq.height.b32(height, [t]);
        st.global.u32([w], width);
        let (x, z): (f32, f32);
        cvt.rn.f32.u32(x, width);
        cvt.rn.f32.u32(z, height);
        mul.rn.f32(x, x, 0.5);
        mul.rn.f32(z, z, 0.5);
        let (a, b, c, d): (f32, f32, f32, f32);
        tld4.g._2d.v4.f32.f32((a, b, c, d), [t, (x, z)]);
        st.global.v4.f32([y], (a, b, c, d));
    }
"#;

#[test]
fn a_texture_is_its_handle_in_brackets_with_the_coordinates_in_it() {
    let ptx = kernite::ptx_str(TEXTURES).unwrap();
    let (head, entries) = ptx.split_once(".visible .entry ").unwrap();
    assert!(head.contains(".version 3.1\n.target sm_30\n"), "{head}");
    for line in [
        "\ttex.level.2d.v4.f32.f32 {%r, _, _, _}, [%t, {%x, 0f3F000000}], 0f00000000;\n",
        "\ttxq.width.b32 %width, [%t];\n",
        "\ttld4.g.2d.v4.f32.f32 {%a, %b, %c, %d}, [%t, {%x, %z}];\n",
    ] {
        assert!(entries.contains(line), "{line}{entries}");
    }
}

#[test]
fn a_surface_is_its_handle_in_brackets_with_the_coordinates_in_it() {
    let ptx = kernite::ptx_str(SURFACES).unwrap();
    let (head, entries) = ptx.split_once(".visible .entry ").unwrap();
    assert!(head.contains(".version 3.1\n.target sm_30\n"), "{head}");
    for line in [
        "\tsuld.b.2d.v4.b32.clamp {%r, %g, %b, %a}, [%from, {%x, %y}];\n",
        "\tsust.b.2d.v4.b32.clamp [%to, {%x, %y}], {%r, %g, %b, %a};\n",
        "\tsuq.width.b32 %width, [%bins];\n",
        "\tsured.b.add.1d.u32.trap [%bins, {%v}], 1;\n",
    ] {
        assert!(entries.contains(line), "{line}{entries}");
    }
}

/// PTX's ISA has a cache operation on `sust.b`, and none on `sust.p`, which `ptxas` takes one
/// on too.
#[test]
fn sust_p_takes_a_cache_operation_as_sust_b_has() {
    let ptx = compile(
        "s: u64, x: i32, y: i32",
        "let v: b32; mov.b32(v, 7); sust.b._2d.cg.b32.trap([s, (x, y)], v); sust.p._2d.cg.b32.trap([s, (x, y)], v);",
    )
    .unwrap();
    for line in ["\tsust.b.2d.cg.b32.trap [%s, {%x, %y}], %v;\n", "\tsust.p.2d.cg.b32.trap [%s, {%x, %y}], %v;\n"] {
        assert!(ptx.contains(line), "{line}{ptx}");
    }
}

#[test]
fn a_kernel_takes_pointers_and_numbers_and_returns_nothing() {
    let e = compile("x: bool", "").unwrap_err();
    assert_eq!(
        e,
        "1:36: a GPU kernel takes pointers, integers and the floats `f16`, `bf16`, `f32` and `f64`, not `bool`"
    );
    let e = kernite::ptx_str("unsafe extern \"ptx-kernel\" fn k() -> f32 {}").unwrap_err();
    assert_eq!(e, "1:38: a GPU kernel returns nothing, not `f32`: it writes what it works out to memory");
}

/// A kernel's pointer is an address on the GPU whatever it points to: an array, a pointer, a tuple,
/// a generic type. One to what has no size, a slice, a `str` or a `dyn`, in parentheses or not, or
/// a tuple that ends in one, is two words in Rust, and an error; and a function takes no pointer
/// at all, whatever it points to.
#[test]
fn a_kernel_takes_a_pointer_to_any_type() {
    let ptx = compile(
        "shape: *const [usize; 3], out: *mut *const f32, pair: *const (u32, u32), cell: *const Cell<u8, 2>",
        "let v: u64; ld.global.u64(v, [shape + 8]); st.global.u64([out], v); st.global.u64([pair], v);\n\
         st.global.u64([cell], v);",
    )
    .unwrap();
    for (i, name) in ["shape", "out", "pair", "cell"].iter().enumerate() {
        let part = format!("\tld.param.u64 %{name}, [k_param_{i}];\n\tcvta.to.global.u64 %{name}, %{name};\n");
        assert!(ptx.contains(&part), "{part}\nin\n{ptx}");
    }
    for to in ["()", "(u32,)", "(u32, u32)"] {
        compile(&format!("p: *const {to}"), "").unwrap();
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
        assert_eq!(compile(&format!("p: *const {to}"), "").unwrap_err(), format!("1:36: {why}"), "{to}");
    }
    let func = "a function takes and returns PTX's types, an address is a `u64`";
    for to in ["[usize; 3]", "[f32]", "str", "dyn Foo", "(dyn A + Send)", "f32"] {
        let src = format!("extern \"ptx-func\" fn f(a: *const {to}) {{}}");
        let want = format!("1:27: {func}; not `*const {to}`");
        assert_eq!(kernite::ptx_str(&src).unwrap_err(), want, "function {to}");
    }
}

/// A type may hold an arrow, `->`, one operator whose `>` ends no type and closes no `<`: a pointer to
/// a function's pointer, or to a type with one among its arguments, is an address as any pointer is,
/// and the parameter after it is the next.
#[test]
fn a_kernel_takes_a_pointer_to_a_type_that_holds_an_arrow() {
    for params in [
        "call: *const fn(u32) -> u32, table: *mut Pair<fn() -> u8, 2>, \
         boxed: *const Box<dyn Fn(u8) -> u8 + Send>, n: u64",
        "call: *const fn(u32)->u32, table: *mut Pair<fn()->&u8,2>, boxed: *const Box<dyn Fn(u8)->u8+Send>,n: u64",
    ] {
        let ptx =
            compile(params, "st.global.u64([call], n); st.global.u64([table], n); st.global.u64([boxed], n);").unwrap();
        for (i, name) in ["call", "table", "boxed"].iter().enumerate() {
            let part = format!("\tld.param.u64 %{name}, [k_param_{i}];\n\tcvta.to.global.u64 %{name}, %{name};\n");
            assert!(ptx.contains(&part), "{part}\nin\n{ptx}");
        }
        assert!(ptx.contains("\tld.param.u64 %n, [k_param_3];\n"), "{ptx}");
    }
    let to = "dyn Fn(u8) -> u8";
    let why = format!(
        "a pointer to `{to}` has two words, an address and a length or a table, and a kernel's pointer has one"
    );
    assert_eq!(compile("p: *const dyn Fn(u8)->u8, n: u64", "").unwrap_err(), format!("1:36: {why}"));
    let func = "a function takes and returns PTX's types, an address is a `u64`";
    let src = "extern \"ptx-func\" fn f(a: *const fn(u32)->u32) {}";
    assert_eq!(kernite::ptx_str(src).unwrap_err(), format!("1:27: {func}; not `*const fn(u32) -> u32`"));
}

/// A type in a message is written as Rust writes it, whatever the source spaced: a parameter by
/// value, a result, a variable. A block and a macro's arguments are written as their source has them, packed
/// or spaced, as the messages of rustc write them. So is the length of an array, but where kernite takes the type
/// of a parameter or a variable apart, and works the length out.
#[test]
fn a_type_in_a_message_is_written_as_rust_writes_it() {
    let takes = "a GPU kernel takes pointers, integers and the floats `f16`, `bf16`, `f32` and `f64`, not";
    for (params, ty) in [
        ("a: [usize;3]", "[usize; 3]"),
        ("a: (u32 ,u32)", "(u32, u32)"),
        ("a: &mut[f32]", "&mut [f32]"),
        ("a: Box<dyn A+B>", "Box<dyn A + B>"),
        ("a: Box<dyn Iterator<Item=u8>>", "Box<dyn Iterator<Item = u8>>"),
        ("a: Box<dyn Iterator<Item = u8> + Send>", "Box<dyn Iterator<Item = u8> + Send>"),
        ("f: fn(u32) -> u32", "fn(u32) -> u32"),
        ("f: fn(u32)->u32, n: u32", "fn(u32) -> u32"),
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
        assert_eq!(compile(params, "").unwrap_err(), format!("1:36: {takes} `{ty}`"), "{params}");
    }
    // The rows of the review, each packed and spaced, in a block and as a macro's arguments.
    for (ty, quoted) in quotes::quoted() {
        assert_eq!(compile(&format!("a: {ty}"), "").unwrap_err(), format!("1:36: {takes} `{quoted}`"), "{ty}");
    }
    // The length of an array is worked out in the type of a parameter that is an array or a name with arguments,
    // and in those of its elements and arguments that are, `[usize; 6]`; its names are the module's. In any other
    // type, a tuple, a reference or a path, it is written as its source has it.
    for (ty, quoted) in [
        ("(u8,[usize;N*2])", "(u8, [usize; N*2])"),
        ("(u8,[usize; N * 2])", "(u8, [usize; N * 2])"),
        ("(u8,[usize ;(N+1)/2 ])", "(u8, [usize; (N+1)/2])"),
        ("(u8,[usize;!0usize&N])", "(u8, [usize; !0usize&N])"),
        ("(u8,[usize;\n    N * // twice\n    2\n])", "(u8, [usize; N * 2])"),
        ("&[usize;N*2]", "&[usize; N*2]"),
        ("&mut [usize;N * 2]", "&mut [usize; N * 2]"),
        ("a::Wrap<[usize;N*2]>", "a::Wrap<[usize; N*2]>"),
        ("Wrap<(u8,[usize;N*2])>", "Wrap<(u8, [usize; N*2])>"),
        ("[(u8,[usize;N*2]);2]", "[(u8, [usize; N*2]); 2]"),
        ("[usize;N*2]", "[usize; 6]"),
        ("Wrap<[usize;N*2]>", "Wrap<[usize; 6]>"),
        ("[[usize;N*2];2]", "[[usize; 6]; 2]"),
        ("Wrap<Wrap<[usize;N*2]>>", "Wrap<Wrap<[usize; 6]>>"),
    ] {
        let src = format!("const N: usize = 3;\nunsafe extern \"ptx-kernel\" fn k(a: {ty}) {{}}\n");
        assert_eq!(kernite::ptx_str(&src).unwrap_err(), format!("2:36: {takes} `{quoted}`"), "{ty}");
    }
    let two = "has two words, an address and a length or a table, and a kernel's pointer has one";
    assert_eq!(compile("p: *const dyn A+Send", "").unwrap_err(), format!("1:36: a pointer to `dyn A + Send` {two}"));
    for (to, quoted) in [
        ("dyn Tr<N={3}>", "dyn Tr<N = {3}>"),
        ("dyn Tr<N={ 3 }>", "dyn Tr<N = { 3 }>"),
        ("dyn m!(a+b)", "dyn m!(a+b)"),
        ("dyn Tr<Item<X>=u8>", "dyn Tr<Item<X> = u8>"),
        ("dyn Tr<Item<X> =u8>", "dyn Tr<Item<X> = u8>"),
    ] {
        let why = format!("1:36: a pointer to `{quoted}` {two}");
        assert_eq!(compile(&format!("p: *const {to}"), "").unwrap_err(), why, "{to}");
    }
    let src = "unsafe extern \"ptx-kernel\" fn k() -> (u32,u32) {}";
    let returns = "a GPU kernel returns nothing, not `(u32, u32)`: it writes what it works out to memory";
    assert_eq!(kernite::ptx_str(src).unwrap_err(), format!("1:38: {returns}"));
    for (ty, quoted) in
        [("(u8,Marker<{N*2}>)", "(u8, Marker<{N*2}>)"), ("(u8,Marker<{ N * 2 }>)", "(u8, Marker<{ N * 2 }>)")]
    {
        let src = format!("unsafe extern \"ptx-kernel\" fn k() -> {ty} {{}}");
        let returns = format!("a GPU kernel returns nothing, not `{quoted}`: it writes what it works out to memory");
        assert_eq!(kernite::ptx_str(&src).unwrap_err(), format!("1:38: {returns}"), "{ty}");
    }
    // In a result the length of an array is as the source has it; in the type of a variable, worked out.
    let src = "const N: usize = 3;\nunsafe extern \"ptx-kernel\" fn k() -> [usize;N*2] {}";
    let returns = "a GPU kernel returns nothing, not `[usize; N*2]`: it writes what it works out to memory";
    assert_eq!(kernite::ptx_str(src).unwrap_err(), format!("2:38: {returns}"));
    let vars = "PTX variables are `pred`, `b8` to `b128`, `u8` to `u64`, `s8` to `s64`, `f16`, `f16x2`, `bf16`, \
                `bf16x2`, `tf32`, `f32` and `f64`, not";
    let src = "const N: usize = 3;\nunsafe extern \"ptx-kernel\" fn k() { let v: Wrap<[usize;N*2]>; }";
    assert_eq!(kernite::ptx_str(src).unwrap_err(), format!("2:44: {vars} `Wrap<[usize; 6]>`"));
    let src = "const N: usize = 3;\nunsafe extern \"ptx-kernel\" fn k() { let v: a::Wrap<[usize;N*2]>; }";
    assert_eq!(kernite::ptx_str(src).unwrap_err(), format!("2:44: {vars} `a::Wrap<[usize; N*2]>`"));
    assert_eq!(compile("", "let v: *const[u8;4];").unwrap_err(), format!("2:8: {vars} `*const [u8; 4]`"));
    assert_eq!(compile("", "let v: fn(u32)->u32;").unwrap_err(), format!("2:8: {vars} `fn(u32) -> u32`"));
    assert_eq!(compile("", "let v: m!(a+b);").unwrap_err(), format!("2:8: {vars} `m!(a+b)`"));
    assert_eq!(compile("", "let v: m!(a + b);").unwrap_err(), format!("2:8: {vars} `m!(a + b)`"));
    assert_eq!(compile("", "let v: (u8,Marker<{N*2}>);").unwrap_err(), format!("2:8: {vars} `(u8, Marker<{{N*2}}>)`"));
    let funcs = "a function takes and returns PTX's types, as its variables are: `u32`, `f32`, `b64`; not";
    let src = "extern \"ptx-func\" fn f(a: [usize;3]) {}";
    assert_eq!(kernite::ptx_str(src).unwrap_err(), format!("1:27: {funcs} `[usize; 3]`"));
}

/// A kernel says which items of its module it names, and where: in its directives, then in its body.
#[test]
fn a_kernel_names_the_items_it_names() {
    let src = r#"
        const ONE: f32 = 1.0;
        const THREADS: u32 = 64;
        const UNREAD: u32 = 2;

        #[maxntid(THREADS)]
        pub unsafe extern "ptx-kernel" fn k(x: *mut f32) {
            let v: f32;
            mov.f32(v, ONE);
            st.global.f32([x], v);
        }
    "#;
    let module = kernite::lang::parse(src).unwrap();
    let k = kernite::ptx(&module, kernite::function(&module, "k").unwrap(), &[], "").unwrap();
    let named: Vec<_> = k.mentions.iter().map(|m| (m.name.as_str(), m.span.0 as usize)).collect();
    assert_eq!(named, [("THREADS", src.find("THREADS)]").unwrap()), ("ONE", src.find("ONE);").unwrap())]);
}

/// Functions the kernels call, and a kernel that calls them each way PTX does.
const CALLS: &str = r#"
    /// The greatest common divisor of `a` and `b`, by Euclid's steps.
    extern "ptx-func" fn gcd(a: u32, b: u32) -> u32 {
        let p: pred;
        setp.eq.u32(p, b, 0);
        if p {
            ret(a);
        }
        let (r, g): (u32, u32);
        rem.u32(r, a, b);
        call(g, gcd, (b, r));
        ret(g);
    }

    /// 1 if `n` is even, else 0, by `odd`, which comes after it.
    extern "ptx-func" fn even(n: u32) -> u32 {
        let p: pred;
        setp.eq.u32(p, n, 0);
        if p {
            ret(1);
        }
        let (m, r): (u32, u32);
        sub.u32(m, n, 1);
        call(r, odd, m);
        ret(r);
    }

    /// 1 if `n` is odd, else 0, by `even`.
    extern "ptx-func" fn odd(n: u32) -> u32 {
        let p: pred;
        setp.eq.u32(p, n, 0);
        if p {
            ret(0);
        }
        let (m, r): (u32, u32);
        sub.u32(m, n, 1);
        call(r, even, m);
        ret(r);
    }

    /// `h` as bits, and a byte it leaves be: what PTX passes as bits alone.
    extern "ptx-func" fn bits(h: f16, b: u8) -> b16 {
        ret(h);
    }

    /// Stops the kernel, and with it the launch.
    extern "ptx-func" fn halt() -> ! {
        trap();
    }

    /// `x[0] = gcd(x[0], n)`, worked out each way there is to call `gcd`; and the kernel
    /// stops where it is 0.
    pub unsafe extern "ptx-kernel" fn calls(x: *mut u32, n: u32) {
        let (a, g, e, fp): (u32, u32, u32, u64);
        ld.global.u32(a, [x]);
        call(g, gcd, (a, n));
        call.uni(_, gcd, (a, 12));
        mov.u64(fp, gcd);
        call(g, fp, (a, n));
        call(g, fp, (a, n), (gcd,));
        call(e, even, 10);
        let me: u64;
        mov.u64(me, calls);
        let (h, hb, c): (f16, b16, u8);
        mov.b16(h, 1.5f16);
        cvt.u8.u32(c, a);
        call(hb, bits, (h, c));
        call(hb, bits, (1.5f16, 7));
        st.global.u32([x], g);
        let p: pred;
        setp.eq.u32(p, g, 0);
        if p {
            call(halt);
        }
    }
"#;

#[test]
fn a_function_comes_out_as_ptx_writes_it() {
    let src = r#"
        pub unsafe extern "ptx-kernel" fn first(x: *mut u32, n: u32) {
            let (a, g): (u32, u32);
            ld.global.u32(a, [x]);
            call(g, gcd, (a, n));
            st.global.u32([x], g);
        }

        extern "ptx-func" fn gcd(a: u32, b: u32) -> u32 {
            let p: pred;
            setp.eq.u32(p, b, 0);
            if p {
                ret(a);
            }
            let (r, g): (u32, u32);
            rem.u32(r, a, b);
            call(g, gcd, (b, r));
            ret(g);
        }
    "#;
    let want = "// Written by kernite.

.version 3.0
.target sm_30
.address_size 64

.func (.param .u32 gcd_retval) gcd(
\t.param .u32 gcd_param_0,
\t.param .u32 gcd_param_1
);

.visible .entry first(
\t.param .u64 first_param_0,
\t.param .u32 first_param_1
)
{
\t.reg .u32 %n, %a, %g;
\t.reg .u64 %x;

\tld.param.u64 %x, [first_param_0];
\tcvta.to.global.u64 %x, %x;
\tld.param.u32 %n, [first_param_1];
\tld.global.u32 %a, [%x];
\tcall (%g), gcd, (%a, %n);
\tst.global.u32 [%x], %g;
\tret;
}

.func (.param .u32 gcd_retval) gcd(
\t.param .u32 gcd_param_0,
\t.param .u32 gcd_param_1
)
{
\t.reg .pred %p;
\t.reg .u32 %a, %b, %r, %g;

\tld.param.u32 %a, [gcd_param_0];
\tld.param.u32 %b, [gcd_param_1];
\tsetp.eq.u32 %p, %b, 0;
\t@%p st.param.u32 [gcd_retval], %a;
\t@%p ret;
\trem.u32 %r, %a, %b;
\tcall (%g), gcd, (%b, %r);
\tst.param.u32 [gcd_retval], %g;
\tret;
}
";
    assert_eq!(kernite::ptx_str(src).unwrap(), want);
    // Each way to call one, and what a function of no value and one that never returns are.
    let ptx = kernite::ptx_str(CALLS).unwrap();
    for text in [
        ".version 6.4\n.target sm_35\n",
        "\n.func (.param .u32 odd_retval) odd(\n\t.param .u32 odd_param_0\n);\n\n.func (.param .u32 even_retval) even(",
        ".func (.param .b16 bits_retval) bits(\n\t.param .b16 bits_param_0,\n\t.param .b8 bits_param_1\n)\n",
        "\tld.param.b16 %h, [bits_param_0];\n\tld.param.b8 %b, [bits_param_1];\n\tst.param.b16 [bits_retval], %h;\n",
        ".func halt(\n) .noreturn\n{\n\n\ttrap;\n}\n",
        "\tcall.uni (_), gcd, (%a, 12);\n",
        "\tmov.u64 %fp, gcd;\n",
        "\t$P0: .callprototype (.param .u32 _) _ (.param .u32 _, .param .u32 _);\n\tcall (%g), %fp, (%a, %n), $P0;\n",
        "\t$C1: .calltargets gcd;\n\tcall (%g), %fp, (%a, %n), $C1;\n",
        "\tcall (%e), even, (10);\n",
        "\tmov.u64 %me, calls;\n",
        "\tcall (%hb), bits, (%h, %c);\n\tcall (%hb), bits, (0x3E00, 7);\n",
        "\t@%p call halt;\n\tret;\n",
    ] {
        assert!(ptx.contains(text), "{text}\nin\n{ptx}");
    }
}

/// Kernels and functions that name each other, forward and back, kernels that share what a
/// launch gives, and statics: what a module is made of.
const PARTS: &str = r#"
    static A: global<[u32; 4]>;
    static C: constant<[u32; 2]> = [1, 2];

    pub unsafe extern "ptx-kernel" fn first(x: *mut u32, n: u32) {
        let (a, g, e): (u32, u32, u32);
        ld.global.u32(a, [x]);
        call(g, gcd, (a, n));
        call(e, even, 10);
        st.global.u32([x], g);
    }

    pub unsafe extern "ptx-kernel" fn second(x: *mut u32, n: u32) {
        let (a, g, e): (u32, u32, u32);
        ld.global.u32(a, [x]);
        call(e, even, 10);
        call(g, gcd, (a, n));
        st.global.u32([x], g);
    }

    extern "ptx-func" fn gcd(a: u32, b: u32) -> u32 {
        let p: pred;
        setp.eq.u32(p, b, 0);
        if p {
            ret(a);
        }
        let (r, g): (u32, u32);
        rem.u32(r, a, b);
        call(g, gcd, (b, r));
        ret(g);
    }

    extern "ptx-func" fn even(n: u32) -> u32 {
        let p: pred;
        setp.eq.u32(p, n, 0);
        if p {
            ret(1);
        }
        let (m, r): (u32, u32);
        sub.u32(m, n, 1);
        call(r, odd, m);
        ret(r);
    }

    extern "ptx-func" fn odd(n: u32) -> u32 {
        let p: pred;
        setp.eq.u32(p, n, 0);
        if p {
            ret(0);
        }
        let (m, r): (u32, u32);
        sub.u32(m, n, 1);
        call(r, even, m);
        ret(r);
    }

    pub unsafe extern "ptx-kernel" fn reverse(x: *mut f32) {
        let buf: shared<[f32]>;
        let (t, s): (u32, u32);
        mov.u32(t, tid.x);
        mov.u32(s, buf);
        let v: f32;
        ld.global.f32(v, [x]);
        st.shared.f32([s], v);
        st.global.f32([x], v);
    }
"#;

/// A module's text, or why it is none.
type Made = Result<String, String>;

/// For each set of `entries` and `statics` kept, by the bits of its number, `entries` first: the
/// module put together from what is kept, and the module of what is kept alone.
fn every_set(entries: &[Entry], statics: &[Static]) -> Vec<(u32, Made, Made)> {
    let assembly = kernite_ptx::module(entries, statics);
    (0u32..1 << (entries.len() + statics.len()))
        .map(|set| {
            let has = |i: usize| set >> i & 1 == 1;
            let kept = |it| match it {
                Kept::Code(i) => has(i),
                Kept::Static(j) => has(entries.len() + j),
            };
            let code: Vec<Entry> = (0..entries.len()).filter(|&i| has(i)).map(|i| entries[i].clone()).collect();
            let memory: Vec<Static> =
                (0..statics.len()).filter(|&j| has(entries.len() + j)).map(|j| statics[j].clone()).collect();
            (set, assembly.text(&kept), kernite_ptx::module(&code, &memory).whole())
        })
        .collect()
}

/// The module of `assembly` put together where the entries at `kept` are kept, or why it is none.
fn put_together(assembly: &Assembly, kept: &[usize]) -> Result<String, String> {
    assembly.text(&|it| matches!(it, Kept::Code(i) if kept.contains(&i)))
}

/// `names` compiled, each a kernel or a function of `source`.
fn compiled(source: &str, names: &[&str]) -> Vec<Entry> {
    let module = kernite::lang::parse(source).unwrap();
    let compile = |name| kernite::ptx(&module, kernite::function(&module, name).unwrap(), &[], "").unwrap().code;
    names.iter().copied().map(compile).collect()
}

/// A module put together from what is kept is the module of what is kept alone, whichever of its
/// kernels, functions and statics that is: for every set of them, each function declared before
/// the first kernel or function kept that names it, if it comes after, and the rest as it is. A
/// static whose name in PTX is a function's is refused where both are kept, and where one is, the
/// other's module has it.
#[test]
fn a_module_put_together_from_what_is_kept_is_the_module_of_what_is_kept_alone() {
    let module = kernite::lang::parse(PARTS).unwrap();
    let mut statics: Vec<Static> = kernite::ptx_statics(&module, "").into_iter().map(|s| s.unwrap().code).collect();
    statics.push(Static::new("S", "odd", "global<u32>").unwrap());
    let entries = compiled(PARTS, &["first", "second", "gcd", "even", "odd", "reverse"]);
    let modules = every_set(&entries, &statics);
    for (set, together, alone) in &modules {
        assert_eq!(together, alone, "{set:b}");
    }
    // Only the static S with `odd` kept beside it makes a module none: of the 2^9 sets, the 2^7
    // that have both.
    let made = modules.iter().filter(|(_, m, _)| m.is_ok()).count();
    assert_eq!((made, modules.len() - made), (384, 128));
    let both = |it| matches!(it, Kept::Code(4) | Kept::Static(2));
    assert_eq!(kernite_ptx::module(&entries, &statics).text(&both), Err("`S` is `odd` in PTX, as `odd` is".into()));
}

/// Kernels that need more of PTX and of the GPU, one after another, and the last two the same GPU.
const NEEDS: &str = r#"
    pub unsafe extern "ptx-kernel" fn plain(x: *mut u32) {
        st.global.u32([x], 1);
    }

    pub unsafe extern "ptx-kernel" fn half(x: *mut u32) {
        let h: f16x2;
        st.global.u32([x], 2);
    }

    pub unsafe extern "ptx-kernel" fn sleep(n: u32) {
        nanosleep.u32(n);
    }

    pub unsafe extern "ptx-kernel" fn tanh(f: f32) {
        let v: f32;
        tanh.approx.f32(v, f);
    }

    pub unsafe extern "ptx-kernel" fn wait() {
        cp.async.wait_all();
    }

    pub unsafe extern "ptx-kernel" fn arrive(y: u64) {
        mbarrier.arrive.b64(_, [y]);
    }
"#;

/// A module put together from what is kept is of the PTX and the GPU that what is kept needs, as
/// the module of it alone is: its `.version` and `.target` are the oldest that have every
/// instruction of the kernels kept, and the floor if none is.
#[test]
fn a_module_put_together_from_what_is_kept_is_of_the_ptx_and_gpu_it_needs() {
    let entries = compiled(NEEDS, &["plain", "half", "sleep", "tanh", "wait", "arrive"]);
    for (set, together, alone) in every_set(&entries, &[]) {
        assert!(together.is_ok(), "{set:b}");
        assert_eq!(together, alone, "{set:b}");
    }
    let assembly = kernite_ptx::module(&entries, &[]);
    let cases: [(&[usize], &str); 14] = [
        (&[], ".version 3.0 .target sm_30"),
        (&[0], ".version 3.0 .target sm_30"),
        (&[1], ".version 4.2 .target sm_53"),
        (&[0, 1], ".version 4.2 .target sm_53"),
        (&[2], ".version 6.2 .target sm_70"),
        (&[1, 2], ".version 6.2 .target sm_70"),
        (&[3], ".version 7.0 .target sm_75"),
        (&[2, 3], ".version 7.0 .target sm_75"),
        (&[4], ".version 7.0 .target sm_80"),
        (&[3, 4], ".version 7.0 .target sm_80"),
        (&[5], ".version 7.1 .target sm_80"),
        (&[3, 5], ".version 7.1 .target sm_80"),
        (&[0, 1, 2, 3, 4], ".version 7.0 .target sm_80"),
        (&[0, 1, 2, 3, 4, 5], ".version 7.1 .target sm_80"),
    ];
    for (kept, want) in cases {
        assert_eq!(header(&put_together(&assembly, kept).unwrap()), want, "kept {kept:?}");
    }
}

/// A function with an instruction for a newer GPU, and the one kernel that calls it, beside a
/// kernel that needs nothing: the function may be kept and its caller not.
const HELPED: &str = r#"
    pub unsafe extern "ptx-kernel" fn plain(x: *mut u32) {
        st.global.u32([x], 1);
    }

    pub unsafe extern "ptx-kernel" fn shine(x: *mut f32, f: f32) {
        let v: f32;
        call(v, fast, f);
        st.global.f32([x], v);
    }

    extern "ptx-func" fn fast(f: f32) -> f32 {
        let v: f32;
        tanh.approx.f32(v, f);
        ret(v);
    }
"#;

/// A function kept sets the module's `.version` and `.target` as a kernel kept does, though the one
/// kernel that calls it is left out: `fast` with `plain`, which needs nothing, is a module of PTX 7.0
/// and `sm_75`, which `fast` needs, and `plain` alone is of the floor. In each set of the three the
/// module put together is the module of that set alone.
#[test]
fn a_function_kept_sets_the_header_though_the_kernel_that_calls_it_is_left_out() {
    let entries = compiled(HELPED, &["plain", "shine", "fast"]);
    for (set, together, alone) in every_set(&entries, &[]) {
        assert!(together.is_ok(), "{set:b}");
        assert_eq!(together, alone, "{set:b}");
    }
    let assembly = kernite_ptx::module(&entries, &[]);
    let with_function = put_together(&assembly, &[0, 2]).unwrap();
    assert_eq!(header(&with_function), ".version 7.0 .target sm_75");
    assert!(with_function.contains(".func (.param .f32 fast_retval) fast("), "{with_function}");
    assert!(!with_function.contains(".entry shine("), "{with_function}");
    assert_eq!(header(&put_together(&assembly, &[0]).unwrap()), ".version 3.0 .target sm_30");
}

/// A kernel and a function each compiled for every instance their `#[instances(..)]` lists,
/// and a kernel, before the function, that names some.
const INSTANCES: &str = r#"
    const WIDE: u32 = 8;

    /// `x[0] <<= S`.
    #[instances(1, WIDE)]
    pub unsafe extern "ptx-kernel" fn shift<const S: u32>(x: *mut u32) {
        let a: u32;
        ld.global.u32(a, [x]);
        shl.b32(a, a, S);
        st.global.u32([x], a);
    }

    /// `x[0] = affine::<-2, false>(affine::<3, true>(x[0]))`, and the address of `shift::<8>`
    /// for the GPU to launch it.
    pub unsafe extern "ptx-kernel" fn uses(x: *mut i32) {
        let (a, b): (s32, s32);
        ld.global.s32(a, [x]);
        call(b, affine::<3, true>, a);
        call(b, affine::<{ 1 - 3 }, false>, b);
        st.global.s32([x], b);
        let k: u64;
        mov.u64(k, shift::<WIDE>);
    }

    /// `a * M`, and 1 more if `MORE`.
    #[instances((3, true), (-2, false))]
    extern "ptx-func" fn affine<const M: i32, const MORE: bool>(a: s32) -> s32 {
        let r: s32;
        mul.lo.s32(r, a, M);
        if MORE {
            add.s32(r, r, 1);
        }
        ret(r);
    }
"#;

#[test]
fn each_instance_is_a_kernel_of_its_own_named_as_cpp_names_it() {
    let ptx = kernite::ptx_str(INSTANCES).unwrap();
    for text in [
        ".visible .entry _Z5shiftILj1EE(\n\t.param .u64 _Z5shiftILj1EE_param_0\n)\n",
        "\tld.param.u64 %x, [_Z5shiftILj1EE_param_0];\n",
        "\tshl.b32 %a, %a, 1;\n",
        ".visible .entry _Z5shiftILj8EE(",
        "\tshl.b32 %a, %a, 8;\n",
        // Declared before `uses`, which names them first.
        "\n.func (.param .s32 _Z6affineILi3ELb1EE_retval) _Z6affineILi3ELb1EE(\n\t.param .s32 \
         _Z6affineILi3ELb1EE_param_0\n);\n\n.func (.param .s32 _Z6affineILin2ELb0EE_retval) \
         _Z6affineILin2ELb0EE(\n\t.param .s32 _Z6affineILin2ELb0EE_param_0\n);\n\n.visible .entry uses(",
        "\tcall (%b), _Z6affineILi3ELb1EE, (%a);\n\tcall (%b), _Z6affineILin2ELb0EE, (%b);\n",
        "\tmov.u64 %k, _Z5shiftILj8EE;\n",
        "\tmul.lo.s32 %r, %a, 3;\n\tadd.s32 %r, %r, 1;\n\tst.param.s32 [_Z6affineILi3ELb1EE_retval], %r;\n",
        "\tmul.lo.s32 %r, %a, -2;\n\tst.param.s32 [_Z6affineILin2ELb0EE_retval], %r;\n",
    ] {
        assert!(ptx.contains(text), "{text}\nin\n{ptx}");
    }
    // In a scope, as `#[kernels]` names them, each name is in it.
    let module = kernite::lang::parse(INSTANCES).unwrap();
    let scope = kernite::ptx_scope(&["kernite", "app", "gpu"]);
    let uses = kernite::ptx(&module, kernite::function(&module, "uses").unwrap(), &[], &scope).unwrap().code;
    assert_eq!(uses.program.name, "_ZN7kernite3app3gpu4usesE");
    assert_eq!(
        uses.calls,
        [
            "_ZN7kernite3app3gpu6affineILi3ELb1EEE",
            "_ZN7kernite3app3gpu6affineILin2ELb0EEE",
            "_ZN7kernite3app3gpu5shiftILj8EEE"
        ]
    );
    assert!(uses.text().contains("\tmov.u64 %k, _ZN7kernite3app3gpu5shiftILj8EEE;\n"));
}

/// NVIDIA's demangler, beside `ptxas`, which Nsight and `cuda-gdb` demangle as.
fn cu_filt() -> Option<PathBuf> {
    let filt = ptxas()?.with_file_name("cu++filt");
    Command::new(&filt).arg("_Z1fv").output().is_ok_and(|o| o.status.success()).then_some(filt)
}

#[test]
fn the_tools_that_read_ptx_show_each_name_as_rust_has_it() {
    let Some(filt) = cu_filt() else {
        eprintln!("no cu++filt beside ptxas here: skipped; set PTXAS to run it");
        return;
    };
    let module = kernite::lang::parse(INSTANCES).unwrap();
    // A name that is no name in C++ is made one.
    let scope = kernite::ptx_scope(&["my-workspace", "my_app", "gpu"]);
    let mut symbols = vec![];
    for name in ["shift", "uses", "affine"] {
        let f = kernite::function(&module, name).unwrap();
        for args in kernite::lang::instances(&module, f).unwrap() {
            symbols.push(kernite::ptx(&module, f, &args, &scope).unwrap().code.program.name);
        }
    }
    let mut want = vec![
        "my_workspace::my_app::gpu::shift<(unsigned int)1>".to_string(),
        "my_workspace::my_app::gpu::shift<(unsigned int)8>".into(),
        "my_workspace::my_app::gpu::uses".into(),
        "my_workspace::my_app::gpu::affine<(int)3, (bool)1>".into(),
        "my_workspace::my_app::gpu::affine<(int)-2, (bool)0>".into(),
    ];
    // Each type of integer, as C++ has the one of its size, and at its edges.
    for (ty, cpp) in [
        (IntTy::I8, "signed char"),
        (IntTy::I16, "short"),
        (IntTy::I32, "int"),
        (IntTy::I64, "long"),
        (IntTy::Isize, "long"),
        (IntTy::U8, "unsigned char"),
        (IntTy::U16, "unsigned short"),
        (IntTy::U32, "unsigned int"),
        (IntTy::U64, "unsigned long"),
        (IntTy::Usize, "unsigned long"),
    ] {
        let x = match ty {
            IntTy::I8 | IntTy::I16 | IntTy::I32 => -5,
            IntTy::I64 | IntTy::Isize => i128::from(i64::MIN),
            _ => i128::from(u64::MAX),
        };
        symbols.push(kernite_ptx::symbol(&kernite::ptx_scope(&["2d"]), "tile", &[Value::Int(x, Some(ty))]));
        want.push(format!("_2d::tile<({cpp}){x}>"));
    }
    symbols.push(kernite_ptx::symbol("", "tile", &[Value::Int(4, None), Value::Bool(true)]));
    want.push("tile<(int)4, (bool)1>".into());
    // A static, as a variable of its namespace.
    let statics = kernite::lang::parse(STATICS).unwrap();
    symbols.push(kernite::ptx_statics(&statics, &scope).remove(0).unwrap().code.memory.name);
    want.push("my_workspace::my_app::gpu::COUNTS".into());
    // The shared memory a launch gives, as the kernel's static.
    let shared = kernite::lang::parse(SHARED).unwrap();
    let reverse = kernite::function(&shared, "reverse").unwrap();
    symbols.push(kernite::ptx(&shared, reverse, &[], &scope).unwrap().code.memory.remove(0).name);
    want.push("my_workspace::my_app::gpu::reverse::buf".into());
    // A name beyond ASCII as Rust's legacy mangling writes it.
    symbols.push(kernite_ptx::symbol(&kernite::ptx_scope(&["app", "мо"]), "σ", &[Value::Bool(false)]));
    want.push("app::$u43c$$u43e$::$u3c3$<(bool)0>".into());
    let shown = Command::new(&filt).args(&symbols).output().unwrap();
    let shown = String::from_utf8(shown.stdout).unwrap();
    assert_eq!(shown.lines().collect::<Vec<_>>(), want, "{symbols:?}");
}

/// A kernel, its parameter, its variables and its memory named beyond ASCII, which PTX's names
/// are not, and a variable by its raw name.
const NAMES: &str = r#"
    /// `ξ[0] = 2 ξ[0]`, through shared memory of each kind.
    pub unsafe extern "ptx-kernel" fn δ(ξ: *mut f32) {
        let σ: f32;
        ld.global.f32(σ, [ξ]);
        let (мем, буф): (shared<[f32; 4]>, shared<[f32]>);
        st.shared.f32([мем], σ);
        ld.shared.f32(σ, [мем]);
        add.rn.f32(σ, σ, σ);
        st.shared.f32([буф], σ);
        let r#type: f32;
        ld.shared.f32(r#type, [буф]);
        st.global.f32([ξ], r#type);
    }
"#;

#[test]
fn a_name_beyond_ascii_is_written_as_rust_writes_one_in_a_symbol() {
    let ptx = kernite::ptx_str(NAMES).unwrap();
    for text in [
        ".extern .shared .align 4 .b8 _ZZ6$u3b4$E18$u431$$u443$$u444$[];\n",
        ".visible .entry $u3b4$(\n\t.param .u64 $u3b4$_param_0\n)\n",
        "\t.shared .align 4 .b8 $u43c$$u435$$u43c$[16];\n",
        "\tld.global.f32 %$u3c3$, [%$u3be$];\n",
        "\tst.shared.f32 [$u43c$$u435$$u43c$], %$u3c3$;\n",
        "\tst.shared.f32 [_ZZ6$u3b4$E18$u431$$u443$$u444$], %$u3c3$;\n",
        "\tld.shared.f32 %type, [_ZZ6$u3b4$E18$u431$$u443$$u444$];\n",
    ] {
        assert!(ptx.contains(text), "{text}\nin\n{ptx}");
    }
    // In a scope as well, as kernite-cuda writes a module's path.
    assert_eq!(kernite::ptx_scope(&["app", "мо"]), "3app12$u43c$$u43e$");
    let module = kernite::lang::parse(NAMES).unwrap();
    let delta = kernite::function(&module, "δ").unwrap();
    let entry = kernite::ptx(&module, delta, &[], &kernite::ptx_scope(&["kernite", "app"])).unwrap().code;
    assert_eq!(entry.program.name, "_ZN7kernite3app6$u3b4$E");
    assert!(entry.text().contains("\tst.shared.f32 [_ZZN7kernite3app6$u3b4$EE18$u431$$u443$$u444$], %$u3c3$;\n"));
}

/// A variable's register is named after it, with `_` after the name of a special register, one
/// kernite reads or one it reads none of, and then a number if another register has the name.
#[test]
fn a_variable_named_as_a_special_register_takes_a_trailing_underscore() {
    let src = r#"
        pub unsafe extern "ptx-kernel" fn k(out: *mut u32) {
            let clusterid: u32;
            let tid: u32;
            let tid_: u32;
            mov.u32(clusterid, 1);
            mov.u32(tid, 2);
            mov.u32(tid_, 3);
            add.u32(clusterid, clusterid, tid);
            add.u32(clusterid, clusterid, tid_);
            st.global.u32([out], clusterid);
        }
    "#;
    let want = "// Written by kernite.

.version 3.0
.target sm_30
.address_size 64

.visible .entry k(
\t.param .u64 k_param_0
)
{
\t.reg .u32 %clusterid_, %tid_, %tid__1;
\t.reg .u64 %out;

\tld.param.u64 %out, [k_param_0];
\tcvta.to.global.u64 %out, %out;
\tmov.u32 %clusterid_, 1;
\tmov.u32 %tid_, 2;
\tmov.u32 %tid__1, 3;
\tadd.u32 %clusterid_, %clusterid_, %tid_;
\tadd.u32 %clusterid_, %clusterid_, %tid__1;
\tst.global.u32 [%out], %clusterid_;
\tret;
}
";
    assert_eq!(kernite::ptx_str(src).unwrap(), want);
}

#[test]
fn ptxas_takes_a_function_that_never_returns_from_ptx_6_4_and_sm_30() {
    let Some(ptxas) = ptxas() else {
        eprintln!("no ptxas here: skipped; set PTXAS to run it");
        return;
    };
    let src = r#"
        unsafe extern "ptx-kernel" fn k(n: u32) {
            let p: pred;
            setp.eq.u32(p, n, 0);
            if p {
                call(halt);
            }
        }

        extern "ptx-func" fn halt() -> ! {
            exit();
        }
    "#;
    let text = kernite::ptx_str(src).unwrap();
    assert!(text.contains("\n.version 6.4\n.target sm_30\n"), "{text}");
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("never");
    std::fs::create_dir_all(&dir).unwrap();
    assert_eq!(refused(&ptxas, &text, gpu_of(&text), &dir.join("needs")).0, BTreeMap::new());
    for (v, sm) in [(63, 30), (64, 20)] {
        let older = retarget(&text, v, sm);
        assert!(!refused(&ptxas, &older, gpu_of(&older), &dir.join(format!("{v}-{sm}"))).0.is_empty(), "{v} {sm}");
    }
}

/// The path that calls `halt` ends there, and the one that calls `back` goes on: so `v` is
/// written on each path that reaches its store, and `first` returns a value on each that
/// reaches its end, only where the call is of `halt`.
#[test]
fn a_call_of_a_function_that_never_returns_ends_its_callers_path_there() {
    let src = |kernel_calls: &str, first_calls: &str| {
        format!(
            "unsafe extern \"ptx-kernel\" fn k(x: *mut u32, n: u32) {{\n\
             let (p, v): (pred, u32);\n\
             setp.eq.u32(p, n, 0);\n\
             if p {{ bra(stop); }}\n\
             mov.u32(v, n);\n\
             bra(store);\n\
             label(stop);\n\
             call({kernel_calls});\n\
             label(store);\n\
             st.global.u32([x], v);\n\
             }}\n\
             extern \"ptx-func\" fn first(n: u32) -> u32 {{\n\
             let p: pred;\n\
             setp.ne.u32(p, n, 0);\n\
             if p {{ ret(n); }}\n\
             call({first_calls});\n\
             }}\n\
             extern \"ptx-func\" fn halt() -> ! {{ trap(); }}\n\
             extern \"ptx-func\" fn back() {{}}\n"
        )
    };
    let ptx = kernite::ptx_str(&src("halt", "halt")).unwrap();
    assert!(ptx.contains("\t@%p ret;\n\tcall halt;\n}\n"), "{ptx}");
    assert_eq!(kernite::ptx_str(&src("back", "halt")).unwrap_err(), "2:9: `v` is read before anything writes it");
    assert_eq!(
        kernite::ptx_str(&src("halt", "back")).unwrap_err(),
        "12:22: control reaches the end of `first`, which returns a `u32`: `ret(v)` gives it back"
    );
}

/// The kernel `k` around `body`, before the functions it may call, compiled, or the error.
fn calling(body: &str) -> Result<String, String> {
    kernite::ptx_str(&format!(
        "unsafe extern \"ptx-kernel\" fn k(n: u32, f: f32) {{\n{body}\n}}\n\
         extern \"ptx-func\" fn first(a: u32, b: u32) -> u32 {{ ret(a); }}\n\
         extern \"ptx-func\" fn half(a: f32) -> f32 {{ ret(a); }}\n\
         extern \"ptx-func\" fn short(a: u16) {{}}\n\
         extern \"ptx-func\" fn nothing() {{}}\n\
         extern \"ptx-func\" fn never() -> ! {{ trap(); }}\n\
         extern \"ptx-func\" fn narrow(a: i32) {{}}\n\
         unsafe extern \"sysv64\" fn cpu() {{}}\n\
         fn helper() {{}}\n"
    ))
}

#[test]
fn a_call_passes_what_the_function_takes_and_returns() {
    let cases = [
        ("call(n, first, (n,));", "2:16: `first` takes 2 arguments, not 1"),
        ("call(n, first, (n, f));", "2:16: `first` takes a `u32` as argument 2; this is a `f32` variable"),
        ("call(n, first, (n, 1.5));", "2:16: `first` takes a `u32` as argument 2; this is the f32 1.5"),
        (
            "call(n, first, (n, first));",
            "2:16: `first` takes a `u32` as argument 2; this is the function `first`: `mov.u64(a, first)` puts its \
             address in `a`",
        ),
        (
            "call(first, (n, n));",
            "2:1: `first` returns a `u32`: `call(r, first, ..)` writes it to `r`, or `call(_, first, ..)` to nothing",
        ),
        ("call(f, first, (n, n));", "2:6: `first` returns a `u32`; this is a `f32` variable"),
        ("call(n, nothing);", "2:6: `nothing` returns nothing: `call(nothing, ..)` calls it"),
        ("call(n, never, ());", "2:6: `never` never returns: `call(never, ..)` calls it"),
        ("call(k);", "2:6: `k` is a kernel, which the host launches; `call` calls a function"),
        ("call(cpu);", "2:6: `cpu` is no function of PTX's: a kernel calls those `extern \"ptx-func\"`"),
        ("call(narrow, n);", "2:6: `narrow` does not compile, so no kernel calls it"),
        ("let fp: u64; mov.u64(fp, narrow);", "2:26: `narrow` does not compile, so no kernel names it"),
        (
            "call(helper);",
            "2:6: `helper` is a helper, inlined where it is called, `helper(..)`: no instruction calls it",
        ),
        (
            "first(n, n);",
            "2:1: `first` names an ABI, so it is compiled on its own, not inlined as a helper is: an instruction \
             calls it, `call(r, first, (a, b))`",
        ),
        (
            "call(f);",
            "2:6: `call` calls a function by its name, or at the address in a 32- or 64-bit integer variable; this \
             is a `f32` variable",
        ),
        (
            "call(n, n, (n, 1));",
            "2:12: a call through a variable, as the prototype its operands' types make, passes variables: this is \
             the integer 1; taking the functions it may call last, `call(r, fp, (a, b), (f, g))`, lets it be more",
        ),
        (
            "call(_, n, (n,));",
            "2:6: a call through a variable, as the prototype its operands' types make, writes variables: this is \
             `_`; taking the functions it may call last, `call(r, fp, (a, b), (f, g))`, lets it be more",
        ),
        (
            "call(first, (first,));",
            "2:6: `call` names the function it calls, `first`, or takes its address from a variable, of one of the \
             functions taken together last; not both",
        ),
        ("call(n, n, (n, n), (first, nothing));", "2:6: `nothing` returns nothing: `call(nothing, ..)` calls it"),
        (
            "call(n, first, (n, n), n);",
            "2:1: `call` is `call(f, (a, b))`, or `call(r, f, (a, b))` for a function that returns a value, `f` a \
             function of the module or a variable that holds the address of one",
        ),
        (
            "let p: pred; call(n, n, (p,));",
            "2:25: PTX passes no `pred` to a function, nor back: `selp.u32(x, 1, 0, p)` makes it a `u32`",
        ),
        // ptxas 13.5 takes an `f16x2` for the `.param` of a `u32`, an `s32` and an `f32`, and kernite does. It takes no
        // integer for an `f32`, no `f16x2` for a `.param` of 16 bits, and no `f16` for one of a `u16`.
        ("call(f, half, (n));", "2:15: `half` takes a `f32`; this is a `u32` variable"),
        ("let h: f16x2; mov.b32(h, n); call(short, (h));", "2:42: `short` takes a `u16`; this is a `f16x2` variable"),
        ("let a: f16; mov.b16(a, 1); call(short, (a));", "2:40: `short` takes a `u16`; this is a `f16` variable"),
    ];
    for (body, want) in cases {
        assert_eq!(calling(body).unwrap_err(), want, "{body}");
    }
}

#[test]
fn a_function_takes_and_returns_ptx_types() {
    let cases = [
        (
            "extern \"ptx-func\" fn f(p: pred) {}",
            "1:27: PTX passes no `pred` to a function, nor back: `selp.u32(x, 1, 0, p)` makes it a `u32`",
        ),
        (
            "extern \"ptx-func\" fn f(a: i32) {}",
            "1:27: a function takes and returns PTX's types, PTX writes it `s32`; not `i32`",
        ),
        (
            "extern \"ptx-func\" fn f(a: *const f32) {}",
            "1:27: a function takes and returns PTX's types, an address is a `u64`; not `*const f32`",
        ),
        (
            "extern \"ptx-func\" fn f() -> (u32, u32) {}",
            "1:29: a function returns one value at most, as PTX's do from Fermi on, not a tuple",
        ),
        ("extern \"ptx-func\" fn f() -> u32 { ret(); }", "1:35: `f` returns a `u32`, which `ret(v)` gives back"),
        ("extern \"ptx-func\" fn f() -> u32 { ret(1.5); }", "1:39: `f` returns a `u32`; this is the f32 1.5"),
        ("extern \"ptx-func\" fn f(a: u32) -> f32 { ret(a); }", "1:45: `f` returns a `f32`; this is a `u32` variable"),
        // ptxas 13.5 takes an `f16x2` for the `.param` of a `u32` or an `s32` that `ret` writes, as a call does, but
        // not for one of an `f32`, which a call takes.
        (
            "extern \"ptx-func\" fn f(a: f32) -> f32 { let h: f16x2; mov.b32(h, a); ret(h); }",
            "1:74: `f` returns a `f32`; this is a `f16x2` variable",
        ),
        (
            "extern \"ptx-func\" fn f() -> u32 {}",
            "1:22: control reaches the end of `f`, which returns a `u32`: `ret(v)` gives it back",
        ),
        (
            "extern \"ptx-func\" fn f() -> ! { ret(); }",
            "1:33: `f` never returns, `-> !`: it ends each way out in `exit()`, `trap()` or a call of a function that \
             never returns",
        ),
        (
            "extern \"ptx-func\" fn f() -> ! {}",
            "1:22: control reaches the end of `f`, which never returns, `-> !`: end each way out in `exit()`, \
             `trap()` or a call of a function that never returns",
        ),
    ];
    for (src, want) in cases {
        assert_eq!(kernite::ptx_str(src).unwrap_err(), want, "{src}");
    }
}

/// How an instance gives its operands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Style {
    /// A variable for each, and the first of the integers an operand may be one of.
    Vars,
    /// A value known now for each that may be one, but the first of registers that take
    /// values only beside a variable, and the last of those integers.
    Values,
    /// `_` for the first result that may be left unwritten, and `!p` for each predicate that
    /// may be read inverted.
    Sinks,
    /// Without the operands that may be left out, and one result where two may be.
    Short,
    /// Every operand, and one result where two may be.
    Single,
    /// Memory the kernel declares for an address in shared or local memory, and the module's
    /// for one in global or constant memory.
    Memory,
    /// A part of each variable that may be one, as a video instruction selects it, and each
    /// operand that may be negated, negated.
    Parts,
    /// A special register for each operand that may be one: `tid.x` of 32 bits or fewer, and
    /// `clock64` of 64.
    Specials,
    /// An address in a `u32` variable for each that may be in one: in a state space of 32-bit
    /// addresses, and in any for a form of `+addr32`.
    Narrow,
    /// The least of the values known now that each operand may be, where those are fewer than
    /// its type's: of a range, or of so many bits, or a power of two.
    Least,
    /// The most of those values.
    Most,
    /// An `f16x2` variable for the `k`th of the registers of an integer type, which `ptxas` reads as the bits of a
    /// `b32` where it takes it.
    Packed(usize),
    /// The same, but a part of it, as a video instruction selects it, where the operand may be one.
    PackedPart(usize),
    /// The same, but negated, where the operand may be.
    PackedNeg(usize),
    /// An `f16x2` variable for each register of the `j`th vector at once.
    PackedAll(usize),
}

/// What `ptxas` should make of the kernels of a group, which need the same.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Group {
    /// Kernels of any operands but an `f16x2` for an integer: `ptxas` takes them at each version and for each GPU.
    Plain,
    /// Kernels that kernite takes with an `f16x2` for an integer: `ptxas` takes them, and makes the cubin of a `b32`.
    Packed,
    /// Kernels that kernite refuses with an `f16x2` for an integer, written with the register of the type the form
    /// gives and then declared `.f16x2`: `ptxas` refuses them.
    Wrong,
    /// Kernels of [`ptxas_bug`], which kernite refuses with an `f16x2` as it does those of [`Group::Wrong`], but that
    /// `ptxas` takes, by a bug of its own: it must, for the test to say when it stops.
    Bug,
}

/// Whether `ptxas` 13.5 takes the `f16x2` at `place` of `form` by a bug of its own, which kernite refuses: the first
/// register of `tex.a2dms`, the sample, whose type `ptxas` does not check.
fn ptxas_bug(form: &Form, place: &str) -> bool {
    let handle = form.operands.iter().position(|s| s.role == Role::Handle);
    form.line.starts_with("tex{.base}.a2dms")
        && handle.is_some_and(|h| place == format!("operand {}.2 of `{}`, lane 1", h + 1, form.line))
}

/// The names of `form` the test writes, and the words of each: each word of each element with
/// each of every other element, the others as they are first; and for one of `+cvt`, every
/// name.
fn names(form: &Form) -> Vec<(String, Vec<Option<usize>>)> {
    let options: Vec<Vec<Option<usize>>> = form
        .elems
        .iter()
        .map(|e| e.optional.then_some(None).into_iter().chain((0..e.alts.len()).map(Some)).collect())
        .collect();
    let first: Vec<Option<usize>> = options.iter().map(|o| o[0]).collect();
    let mut choices = vec![first.clone()];
    if form.checks.contains(&Check::Cvt) {
        choices = vec![vec![]];
        for o in &options {
            choices = choices
                .into_iter()
                .flat_map(|c| {
                    o.iter().map(move |&x| {
                        let mut c = c.clone();
                        c.push(x);
                        c
                    })
                })
                .collect();
        }
    } else {
        for (i, oi) in options.iter().enumerate() {
            for &a in oi {
                let mut c = first.clone();
                c[i] = a;
                choices.push(c.clone());
                for (j, oj) in options.iter().enumerate().skip(i + 1) {
                    for &b in oj {
                        let mut d = c.clone();
                        d[j] = b;
                        choices.push(d);
                    }
                }
            }
        }
    }
    let mut names: Vec<(String, Vec<Option<usize>>)> = choices
        .into_iter()
        .map(|c| {
            let words = form.elems.iter().zip(&c).filter_map(|(e, k)| k.map(|k| e.alts[k].words.join(".")));
            (std::iter::once(form.op.to_string()).chain(words).collect::<Vec<_>>().join("."), c)
        })
        .collect();
    names.sort();
    names.dedup_by(|a, b| a.0 == b.0);
    names
}

/// A part of a variable an operand of `sel` may be: none that would go without saying.
fn example(sel: Sel) -> &'static str {
    match sel {
        Sel::One => "b2",
        Sel::Halves => "h21",
        Sel::Bytes => "b6420",
        Sel::HalfMask => "h1",
        Sel::ByteMask => "b31",
    }
}

/// A value known now of type `t`, for an operand that may be one: a float that is not
/// `f32` or `f64` is not, nor a `b128`. An integer is a predicate, 1 true.
fn value(t: Ty, spec: &Spec) -> Option<Given<Var>> {
    match t {
        Ty::F32 if spec.fraction => Some(Given::F32(0.5)),
        Ty::F32 => Some(Given::F32(1.5)),
        Ty::F64 => Some(Given::F64(1.5)),
        Ty::B128 => None,
        _ if t.kind() == Kind::Float => None,
        _ => Some(Given::Int(spec.step.unwrap_or(1).into())),
    }
}

/// The module's memory the instances reach: 256 bytes at a multiple of 32, as a fragment
/// `wmma` loads is, in global memory, `g`, and in constant memory, `c`.
fn statics() -> [Static; 2] {
    [("g", "global"), ("c", "constant")]
        .map(|(name, space)| Static::new(name, name, &format!("{space}<[u8; 256], 32>")).unwrap())
}

/// Kernel `k{n}`, which names the module's memory, [`statics`].
fn kernel(n: usize) -> Kernel {
    let mut kernel = Kernel::new(format!("k{n}"), format!("k{n}"));
    for s in statics() {
        kernel.module_memory(&s);
    }
    kernel
}

/// An instance being made: the kernel, and whether its operands differ from those of
/// [`Style::Vars`].
struct Making {
    kernel: Kernel,
    vars: usize,
    differs: bool,
    sunk: bool,
    /// How many registers of an integer type there are so far, and how many vectors.
    ints: usize,
    vectors: usize,
    /// Whether a register that is to be an `f16x2` is made of the type the form gives: [`instance`] declares it so.
    retype: bool,
    /// The registers that are to be an `f16x2`, by [`kernite_core::VReg::id`].
    packed: Vec<usize>,
    /// The operand being made, and the operand of an `f16x2` register: which one of which line of the table.
    place: String,
    packed_place: String,
}

impl Making {
    fn var(&mut self, t: Ty) -> Var {
        self.vars += 1;
        self.kernel.var(&format!("v{}", self.vars), t.name()).unwrap()
    }

    /// Memory of 256 bytes at a multiple of 32, as a fragment `wmma` loads is: the module's,
    /// [`statics`], for `.global` and `.const`, and else the kernel's, in local memory for
    /// `.local` and in shared memory otherwise.
    fn memory(&mut self, space: Space) -> Var {
        match space {
            Space::Global => return self.kernel.global("g").unwrap(),
            Space::Const => return self.kernel.global("c").unwrap(),
            _ => {}
        }
        self.vars += 1;
        let space = if space == Space::Local { "local" } else { "shared" };
        self.kernel.var(&format!("s{}", self.vars), &format!("{space}<[u8; 256], 32>")).unwrap()
    }

    /// A variable of type `t` for `spec`: in [`Style::Parts`], a part of one where it may be,
    /// and negated where it may be; in [`Style::Packed`], an `f16x2` for the one it says, in
    /// [`Style::PackedPart`] a part of it where it may be one, and in [`Style::PackedNeg`] negated where it may be;
    /// and an `f16x2` if `all`.
    fn reg(&mut self, spec: &Spec, t: Ty, style: Style, all: bool) -> Given<Var> {
        let packable = matches!(t.kind(), Kind::Unsigned | Kind::Signed);
        let alone = matches!(style, Style::Packed(k) | Style::PackedPart(k) | Style::PackedNeg(k) if k == self.ints);
        self.ints += usize::from(packable);
        let packed = packable && (all || alone);
        self.differs |= packed && matches!(style, Style::Packed(_) | Style::PackedAll(_));
        let mut v = self.var(if packed && !self.retype { Ty::F16x2 } else { t });
        if let (true, Var::Reg { reg, .. }) = (packed, &v) {
            self.packed.push(reg.id as usize);
            self.packed_place.clone_from(&self.place);
        }
        if let Some(sel) = spec.part
            && (style == Style::Parts || packed && matches!(style, Style::PackedPart(_)))
        {
            self.differs = true;
            v = self.kernel.field(v, example(sel)).unwrap();
        }
        if spec.neg && (style == Style::Parts || packed && matches!(style, Style::PackedNeg(_))) {
            self.differs = true;
            return Given::Neg(Box::new(Given::Var(v)));
        }
        Given::Var(v)
    }

    /// The operand `spec` of `b`, as `style` gives it.
    fn operand(&mut self, b: &Bound, spec: &Spec, style: Style) -> Given<Var> {
        let lanes = match spec.count {
            Count::One => 1,
            Count::Fixed(n) => n,
            Count::Vec => b.lanes(),
        };
        // `_` stands for a register of a vector, not for the one of a name with neither `.v2`
        // nor `.v4`.
        let sink = spec.sink && (spec.count != Count::Vec || lanes > 1);
        let all = lanes > 1 && {
            self.vectors += 1;
            style == Style::PackedAll(self.vectors - 1)
        };
        let (mut parts, place) = (vec![], self.place.clone());
        for k in 0..lanes {
            if lanes > 1 {
                self.place = format!("{place}, lane {}", k + 1);
            }
            let t = match spec.lead {
                Some(lead) if k == 0 => lead,
                _ => b.ty(spec.ty.expect("a register's operand has a type")),
            };
            let part = if spec.memory {
                Some(Given::Var(self.memory(b.space())))
            } else if style == Style::Sinks && sink && !self.sunk {
                self.sunk = true;
                Some(Given::Special("_".into()))
            } else if style == Style::Sinks && spec.not && t == Ty::Pred {
                Some(Given::Not(Box::new(Given::Var(self.var(t)))))
            } else if style == Style::Values && spec.role == Role::Src && !spec.reg_only && !(spec.one_var && k == 0) {
                value(t, spec)
            } else if style == Style::Specials
                && spec.role == Role::Src
                && spec.sreg
                && (spec.count != Count::Vec || lanes > 1)
            {
                ["tid.x", "clock64"]
                    .into_iter()
                    .find(|s| forms::sreg(s).is_some_and(|r| r.read_as(t, b.form.wide, spec.count != Count::One)))
                    .map(|s| Given::Special(s.into()))
            } else if let Some((lo, hi)) =
                spec.range.or(spec.mask.map(|m| (0, m))).filter(|_| spec.role == Role::Src && !spec.reg_only)
            {
                match style {
                    Style::Least => Some(Given::Int(lo.into())),
                    Style::Most => Some(Given::Int(hi.into())),
                    _ => None,
                }
            } else {
                None
            };
            self.differs |= part.is_some() && !spec.memory;
            parts.push(part.unwrap_or_else(|| self.reg(spec, t, style, all)));
        }
        match lanes {
            1 => parts.pop().expect("one"),
            _ => Given::Vector(parts),
        }
    }
}

/// Kernel `k{n}`, of the one instruction `name`, of form `b`, its operands as `style` gives
/// them; `None` if they would be those of [`Style::Vars`]; else kernite's refusal. With `retype`, an `f16x2` that
/// the style gives is a register of the type the form gives, declared `.f16x2` after. The kernel comes with the place
/// of its `f16x2` register: which operand of which line of the table.
fn instance(n: usize, name: &str, b: &Bound, style: Style, retype: bool) -> Option<Result<(Entry, String), String>> {
    let mut m = Making {
        kernel: kernel(n),
        vars: 0,
        differs: false,
        sunk: false,
        ints: 0,
        vectors: 0,
        retype,
        packed: vec![],
        place: String::new(),
        packed_place: String::new(),
    };
    m.kernel.callee("f", "f", Callee::Calls(Prototype { params: vec![], returns: Returns::Nothing }));
    let addr = m.kernel.var("a", "u64").unwrap();
    let narrow = (style == Style::Narrow).then(|| m.kernel.var("n", "u32").unwrap());
    let mut spaces = b.spaces().into_iter();
    let mut args = vec![];
    for (i, spec) in b.form.operands.iter().enumerate().filter(|(_, s)| s.when.is_none_or(|w| b.has(w))) {
        m.place = format!("operand {} of `{}`", i + 1, b.form.line);
        if spec.optional && style == Style::Short {
            m.differs = true;
            break;
        }
        args.push(match spec.role {
            Role::Label => Given::Label(0),
            Role::Labels => Given::Vector(vec![Given::Label(0), Given::Label(1)]),
            Role::Func => Given::Func("f".into()),
            Role::Handle => {
                let items: Vec<Given<Var>> = spec
                    .inner
                    .iter()
                    .enumerate()
                    .map(|(j, s)| {
                        m.place = format!("operand {}.{} of `{}`", i + 1, j + 1, b.form.line);
                        m.operand(b, s, style)
                    })
                    .collect();
                match items.as_slice() {
                    &[Given::Var(t)] => Given::Mem(Address { base: Some(t), index: None, disp: 0 }),
                    _ => Given::List(items),
                }
            }
            Role::Addr(_) => match spaces.next().expect("a space for each address") {
                s @ (Space::Shared | Space::Local | Space::Global | Space::Const) if style == Style::Memory => {
                    m.differs = true;
                    Given::Mem(Address { base: Some(m.memory(s)), index: None, disp: 32 })
                }
                s if !s.wide() || b.form.checks.contains(&Check::Addr32) => match narrow {
                    Some(n) => {
                        m.differs = true;
                        Given::Mem(Address { base: Some(n), index: None, disp: 0 })
                    }
                    None => Given::Mem(Address { base: Some(addr), index: None, disp: 0 }),
                },
                _ => Given::Mem(Address { base: Some(addr), index: None, disp: 0 }),
            },
            Role::Imm(Imm::Bits(bits)) => {
                m.differs |= matches!(style, Style::Least | Style::Most);
                Given::Int(match style {
                    Style::Least => 0,
                    Style::Most => (1 << bits) - 1,
                    _ => 1,
                })
            }
            Role::Imm(imm @ Imm::Of(_)) => {
                let values = imm.values();
                m.differs |= style == Style::Values && values.len() > 1;
                Given::Int(if style == Style::Values { values[values.len() - 1] } else { values[0] }.into())
            }
            Role::Imm(Imm::Pow2(p)) => {
                m.differs |= matches!(style, Style::Least | Style::Most);
                Given::Int(match style {
                    Style::Least => 1,
                    Style::Most => 1 << p,
                    _ => 16,
                })
            }
            Role::Dst | Role::Src => {
                let first = m.operand(b, spec, style);
                match &spec.second {
                    Some((_, true)) if matches!(style, Style::Short | Style::Single) => {
                        m.differs = true;
                        first
                    }
                    Some((second, _)) => Given::Pair(Box::new(first), Box::new(m.operand(b, second, style))),
                    None => first,
                }
            }
        });
    }
    if style != Style::Vars && !m.differs {
        return None;
    }
    if let Err(f) = m.kernel.inst(name, &args, None) {
        return Some(Err(format!("kernite refuses `{name}` as {style:?}: {}", f.message)));
    }
    if b.form.operands.iter().any(|s| matches!(s.role, Role::Label | Role::Labels)) {
        m.kernel.label(0);
        m.kernel.label(1);
    }
    m.kernel.end().unwrap();
    let mut entry = m.kernel.finish();
    if retype {
        for id in m.packed {
            entry.regs[id].1 = Ty::F16x2;
        }
        entry.needs = entry.needs.and(Ty::F16x2.needs());
    }
    Some(Ok((entry, m.packed_place)))
}

/// Kernel `k{n}` of the one instruction `name`, of `form` with the words `choice`, written
/// as kernite would write it if it took it: for a name it refuses, which `ptxas` should too.
fn by_hand(n: usize, name: &str, form: &'static Form, choice: Vec<Option<usize>>) -> String {
    let mut types = vec![];
    for (e, k) in form.elems.iter().zip(&choice) {
        if let (Some(slot), Some(k)) = (e.slot, k) {
            types.resize(types.len().max(slot + 1), Ty::Pred);
            types[slot] = Ty::parse(e.alts[*k].words[0]).expect("a type's word is its name");
        }
    }
    let b = Bound { form, choice, types, needs: Needs::default() };
    let mut regs = vec![];
    let operands: Vec<String> = form
        .operands
        .iter()
        .filter(|s| s.when.is_none_or(|w| b.has(w)))
        .map(|spec| match spec.role {
            Role::Label => "$L0".to_string(),
            Role::Labels => "$T0".to_string(),
            Role::Func => "f".to_string(),
            Role::Handle => {
                let inner: Vec<String> = spec.inner.iter().map(|s| registers(&b, s, &mut regs)).collect();
                format!("[{}]", inner.join(", "))
            }
            Role::Addr(_) => "[%a]".to_string(),
            Role::Imm(Imm::Bits(_)) => "1".to_string(),
            Role::Imm(imm @ Imm::Of(_)) => imm.values()[0].to_string(),
            Role::Imm(Imm::Pow2(_)) => "16".to_string(),
            Role::Dst | Role::Src => registers(&b, spec, &mut regs),
        })
        .collect();
    let decls: String =
        regs.iter().enumerate().map(|(i, t)| format!("\t.reg .{} %v{};\n", t.decl().name(), i + 1)).collect();
    let label = if form.operands.iter().any(|s| matches!(s.role, Role::Label | Role::Labels)) { "$L0:\n" } else { "" };
    let table = if form.operands.iter().any(|s| s.role == Role::Labels) { "\t$T0: .branchtargets $L0;\n" } else { "" };
    format!(
        ".visible .entry k{n}()\n{{\n\t.reg .b64 %a;\n{decls}\n{table}\t{name} {};\n{label}\tret;\n}}\n",
        operands.join(", ")
    )
}

/// Operand `spec` of `b` as registers, each added to `regs`: one; a vector, braced even of
/// one; or two written at once.
fn registers(b: &Bound, spec: &Spec, regs: &mut Vec<Ty>) -> String {
    let t = b.ty(spec.ty.expect("a register's operand has a type"));
    let lanes = match spec.count {
        Count::One => 1,
        Count::Fixed(n) => n,
        Count::Vec => b.lanes(),
    };
    let names: Vec<String> = Vec::from_fn(lanes, |k| {
        regs.push(if k == 0 { spec.lead.unwrap_or(t) } else { t });
        format!("%v{}", regs.len())
    });
    let braced = matches!(spec.count, Count::Fixed(_));
    let first = if lanes == 1 && !braced { names[0].clone() } else { format!("{{{}}}", names.join(", ")) };
    match &spec.second {
        Some((second, _)) => format!("{first}|{}", registers(b, second, regs)),
        None => first,
    }
}

/// The kernels of the instructions kernite reads from no form's operands: each special
/// register read, as it is and as the fewer bits PTX lets `mov` read of it, and the floats
/// of 16 bits known now that a `b16` takes.
fn others(n: &mut usize) -> Vec<Entry> {
    let mut out = vec![];
    let mut one = |name: &str, t: Ty, a: Given<Var>| {
        *n += 1;
        let mut kernel = Kernel::new(format!("k{n}"), format!("k{n}"));
        let v = kernel.var("v", t.name()).unwrap();
        kernel
            .inst(name, &[Given::Var(v), a], None)
            .unwrap_or_else(|f| panic!("kernite refuses `{name}`: {}", f.message));
        kernel.end().unwrap();
        out.push(kernel.finish());
    };
    for r in forms::sregs() {
        one(&format!("mov.{}", r.ty.name()), r.ty, Given::Special(r.name.clone()));
        if r.fewest < r.ty.bits() {
            one("mov.u16", Ty::U16, Given::Special(r.name.clone()));
            one("cvt.u32.u16", Ty::U32, Given::Special(r.name.clone()));
        }
    }
    one("mov.b16", Ty::F16, Given::F16(0x3E00));
    one("mov.b16", Ty::Bf16, Given::Bf16(0x3FC0));
    one("mov.b32", Ty::F16x2, Given::Vector(vec![Given::F16(0x3E00), Given::F16(0x4000)]));
    one("mov.b32", Ty::Tf32, Given::F32(1.5));
    one("mov.b64", Ty::F64, Given::F64(1.5));
    // The address of the module's memory, in the bits ptxas takes it in: 64 for global
    // memory's, which it refuses in 32.
    let all = [(Ty::U64, "g"), (Ty::B64, "g"), (Ty::S64, "g"), (Ty::U64, "c"), (Ty::U32, "c"), (Ty::S32, "c")];
    for (t, name) in all {
        *n += 1;
        let mut kernel = kernel(*n);
        let (v, m) = (kernel.var("v", t.name()).unwrap(), kernel.global(name).unwrap());
        kernel.inst(&format!("mov.{}", t.name()), &[Given::Var(v), Given::Var(m)], None).unwrap();
        kernel.end().unwrap();
        out.push(kernel.finish());
    }
    // A kernel's address, its own, for the GPU to launch it.
    for t in [Ty::U32, Ty::U64, Ty::B32, Ty::B64] {
        *n += 1;
        let name = format!("k{n}");
        let mut kernel = Kernel::new(&name, &name);
        kernel.callee(&name, &name, Callee::Kernel);
        let v = kernel.var("v", t.name()).unwrap();
        kernel.inst(&format!("mov.{}", t.name()), &[Given::Var(v), Given::Func(name.clone())], None).unwrap();
        kernel.end().unwrap();
        out.push(kernel.finish());
    }
    out
}

/// The function an instance that takes a function's address takes it of, in each module of
/// one that does.
fn callee() -> Entry {
    let mut f = Kernel::func("f", "f");
    f.end().unwrap();
    f.finish()
}

/// The module of `entries`, with the function they name and the memory they reach,
/// [`statics`].
fn module(entries: &[Entry]) -> String {
    let mut all = entries.to_vec();
    if entries.iter().any(|e| e.calls.iter().any(|c| c == "f")) {
        all.insert(0, callee());
    }
    kernite_ptx::module(&all, &statics()).whole().unwrap()
}

/// The kernels `ptxas` refuses of each module, assembling it for the GPU given with it, by
/// name, each with the first error it finds in it; `?` for one outside any kernel. And a hash of the cubin it makes of
/// the rest. The modules are written in the directory `dir` of the tests' own.
fn assemble(ptxas: &Path, dir: &str, modules: &[(String, u32)]) -> Vec<(BTreeMap<String, String>, u64)> {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(dir);
    std::fs::create_dir_all(&dir).unwrap();
    let next = AtomicUsize::new(0);
    let out: Vec<Mutex<(BTreeMap<String, String>, u64)>> = modules.iter().map(|_| Mutex::default()).collect();
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some((text, gpu)) = modules.get(i) else { break };
                    *out[i].lock().unwrap() = refused(ptxas, text, *gpu, &dir.join(i.to_string()));
                }
            });
        }
    });
    out.into_iter().map(|m| m.into_inner().unwrap()).collect()
}

/// The kernels `ptxas` refuses of the module `text`, which it reads at `path` and assembles
/// for the GPU `gpu`, and a hash of the cubin it makes of the rest. An error may
/// stop it before the kernels after, so it goes on without those it refused, until it takes
/// the rest. A warning is a refusal, as `--warning-as-error` makes it: `ptxas` warns of what a
/// version of PTX deprecates, as of `shfl` without `.sync`, and kernite takes none of it. Advice,
/// as that `mma.sp` is slower than `mma.sp::ordered_metadata`, is no refusal; nor is the warning that `ptxas` 13.5
/// cannot size the stack of a kernel with `alloca` or `stackrestore`, which size it as they run.
fn refused(ptxas: &Path, text: &str, gpu: u32, path: &Path) -> (BTreeMap<String, String>, u64) {
    let (src, bin) = (path.with_extension("ptx"), path.with_extension("cubin"));
    let (mut text, mut out) = (text.to_string(), BTreeMap::new());
    loop {
        std::fs::write(&src, &text).unwrap();
        let run = Command::new(ptxas)
            .args(["--gpu-name", &format!("sm_{gpu}"), "--warning-as-error", "--suppress-stack-size-warning"])
            .arg(&src)
            .arg("-o")
            .arg(&bin)
            .output()
            .unwrap();
        let cubin = std::fs::read(&bin).unwrap_or_default();
        let _ = std::fs::remove_file(&bin);
        if run.status.success() {
            let mut hasher = DefaultHasher::new();
            cubin.hash(&mut hasher);
            return (out, hasher.finish());
        }
        let stderr = String::from_utf8_lossy(&run.stderr);
        let mut wrong: Vec<String> = vec![];
        for l in stderr.lines() {
            let Some((n, message)) = l.split_once(", line ").and_then(|(_, r)| r.split_once(';')) else { continue };
            let Ok(n) = n.parse() else { continue };
            let (k, message) = (kernel_at(&text, n).unwrap_or("?").to_string(), message.trim());
            if message.starts_with("info") {
                continue;
            }
            if !wrong.contains(&k) {
                wrong.push(k.clone());
            }
            out.entry(k).or_insert_with(|| message.to_string());
        }
        if wrong.is_empty() || wrong.iter().any(|k| k == "?") {
            // A crash says nothing.
            let why = if stderr.trim().is_empty() { format!("ptxas: {}", run.status) } else { stderr.trim().into() };
            out.entry("?".into()).or_insert(why);
            return (out, 0);
        }
        for k in &wrong {
            text = without(&text, k);
        }
    }
}

/// For each line of `text`, the kernel it is in, by name.
fn kernel_at(text: &str, line: usize) -> Option<&str> {
    text.lines().take(line).filter_map(|l| l.strip_prefix(".visible .entry ")).last().and_then(|l| l.split('(').next())
}

/// The module `text` without its kernel `k`.
fn without(text: &str, k: &str) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let start = lines
        .iter()
        .position(|l| l.strip_prefix(".visible .entry ").and_then(|l| l.split('(').next()) == Some(k))
        .expect("the kernel is in the module");
    let end = start + lines[start..].iter().position(|l| *l == "}").expect("a kernel ends");
    lines[..start].iter().chain(&lines[end + 1..]).map(|l| format!("{l}\n")).collect()
}

/// `entries` with each `f16x2` register declared `.b32`, which is how `ptxas` reads an `f16x2` for an integer.
fn as_bits(entries: &[Entry]) -> Vec<Entry> {
    let mut out = entries.to_vec();
    for (_, t) in out.iter_mut().flat_map(|e| &mut e.regs) {
        if *t == Ty::F16x2 {
            *t = Ty::B32;
        }
    }
    out
}

/// The kernels of `entries`, found by halves, of which `ptxas` makes a cubin for the GPU `gpu` that differs from the
/// cubin of the same with each `f16x2` register declared `.b32`.
fn differ(ptxas: &Path, entries: &[Entry], gpu: u32) -> Vec<String> {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("bits");
    std::fs::create_dir_all(&dir).unwrap();
    let cubin = |entries: &[Entry]| refused(ptxas, &module(entries), gpu, &dir.join("one")).1;
    if entries.is_empty() || cubin(entries) == cubin(&as_bits(entries)) {
        return vec![];
    }
    match entries {
        [e] => vec![e.program.name.clone()],
        _ => {
            let (first, rest) = entries.split_at(entries.len() / 2);
            [differ(ptxas, first, gpu), differ(ptxas, rest, gpu)].concat()
        }
    }
}

/// The GPU `ptxas` assembles `text` for, but where another is said: the one it is for, or
/// Turing, `sm_75`, the oldest that ptxas 13.5 assembles for.
fn gpu_of(text: &str) -> u32 {
    let sm: u32 = text
        .lines()
        .find_map(|l| l.strip_prefix(".target sm_"))
        .and_then(|t| t.parse().ok())
        .expect("a module has a target");
    sm.max(75)
}

/// The module `text`, which `ptxas` should make `expect` of, for the GPU [`gpu_of`] gives.
fn at(text: String, expect: Expect) -> (String, u32, Expect) {
    let g = gpu_of(&text);
    (text, g, expect)
}

/// A module's `.version` and `.target`.
fn header(text: &str) -> String {
    text.lines().filter(|l| l.starts_with(".version ") || l.starts_with(".target ")).collect::<Vec<_>>().join(" ")
}

/// What `ptxas` should make of a module.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Expect {
    /// It assembles.
    Takes,
    /// Each of its kernels is wrong.
    Refuses,
}

/// Every version of PTX, from 1.0 to 9.5, which the `ptxas` here reads: those before 2.3 without `.address_size`,
/// which they have not. PTX has no 5.1, as 6.0 came after 5.0, though `ptxas` reads it: [`READ_AS_6_0`].
const VERSIONS: [u32; 48] = [
    10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 30, 31, 32, 40, 41, 42, 43, 50, 60, 61, 62, 63, 64, 65, 70, 71, 72, 73, 74,
    75, 76, 77, 78, 80, 81, 82, 83, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 95,
];

/// The newest version of PTX, the last of [`VERSIONS`]: a test that writes a module of it takes the version from here,
/// and writes no number of its own.
const NEWEST: u32 = VERSIONS[VERSIONS.len() - 1];

/// The version 5.1, which `ptxas` 13.5 reads as 6.0, though PTX has none: in a module of it `ptxas` takes `shfl.sync`,
/// `vote.sync`, `bar.warp.sync` and `.target sm_70`, which it refuses at 5.0. The ISA says nothing of a module of 5.1,
/// so kernite writes none.
const READ_AS_6_0: u32 = 51;

/// The version `v` as PTX writes it: 95 is 9.5.
fn dotted(v: u32) -> String {
    format!("{}.{}", v / 10, v % 10)
}

/// The newest of Fermi's GPUs, which came with PTX 2.0. kernite writes for none of them, as the
/// driver starts at most 65535 blocks along x of a module for one; ptxas takes a module for one.
const FERMI: u32 = 21;

/// The newest GPU a module of PTX `v` may be for: before 2.0, one of Tesla's, `sm_11` or `sm_13`,
/// and before 3.0, Fermi's, [`FERMI`], which kernite writes for none of.
fn newest(v: u32) -> u32 {
    match v {
        ..12 => 11,
        12..20 => 13,
        20..30 => FERMI,
        _ => kernite_ptx::GPUS
            .iter()
            .filter(|&&(_, came)| came <= v)
            .map(|&(g, _)| g)
            .max()
            .expect("every version from 3.0 has Kepler"),
    }
}

/// The version of PTX before `v`.
fn before(v: u32) -> Option<u32> {
    let i = VERSIONS.iter().position(|&x| x == v).expect("a version ptxas reads");
    i.checked_sub(1).map(|i| VERSIONS[i])
}

/// The version of PTX that came with the GPU `sm`: one kernite writes for, or [`FERMI`].
fn came(sm: u32) -> u32 {
    match sm {
        FERMI => 20,
        _ => kernite_ptx::GPUS.iter().find(|&&(g, _)| g == sm).map(|&(_, v)| v).expect("a GPU kernite writes for"),
    }
}

/// The module `text` as one of PTX `v` for the GPU `sm`.
fn retarget(text: &str, v: u32, sm: u32) -> String {
    text.lines()
        .filter(|l| v >= 23 || *l != ".address_size 64")
        .map(|l| match l {
            _ if l.starts_with(".version ") => format!(".version {}\n", dotted(v)),
            _ if l.starts_with(".target ") => format!(".target sm_{sm}\n"),
            _ => format!("{l}\n"),
        })
        .collect()
}

/// A module of one kernel that returns, of which [`retarget`] makes one of each version of PTX.
const ONE_KERNEL: &str = ".version 3.0\n.target sm_30\n.address_size 64\n\n.visible .entry k()\n{\n\tret;\n}\n";

/// `ptxas` reads the versions of PTX that [`VERSIONS`] has, and [`READ_AS_6_0`], of those from 1.0 to the end of the
/// major version after the newest, 10.9 now, and no other: of a module of one kernel that returns, for the newest GPU a
/// module of it may be for, it says "Unsupported .version" of the rest. It fails the day `ptxas` reads a version that
/// kernite does not know, for [`VERSIONS`] to have it and the tests to hand each form to `ptxas` at it, and the day it
/// stops reading one that kernite does.
#[test]
fn ptxas_reads_each_version_of_ptx_that_versions_has_and_5_1_and_no_other() {
    let Some(ptxas) = ptxas() else {
        eprintln!("no ptxas here: skipped; set PTXAS to run it");
        return;
    };
    let every: Vec<u32> = (10..(NEWEST / 10 + 2) * 10).collect();
    let modules: Vec<(String, u32)> = every
        .iter()
        .map(|&v| retarget(ONE_KERNEL, v, newest(v)))
        .map(|text| {
            let gpu = gpu_of(&text);
            (text, gpu)
        })
        .collect();
    let mut failures = vec![];
    for (&v, (refused, _)) in every.iter().zip(assemble(&ptxas, "versions", &modules)) {
        let (known, said) = (VERSIONS.contains(&v) || v == READ_AS_6_0, refused.values().next());
        let (version, unread) = (dotted(v), said.is_some_and(|m| m.contains("Unsupported .version")));
        match (known, unread, said) {
            (true, true, Some(m)) => {
                failures.push(format!("ptxas does not read PTX {version}, which kernite knows: {m}"))
            }
            (true, false, Some(m)) => failures.push(format!("ptxas refuses a module of PTX {version}: {m}")),
            (false, false, _) => failures.push(format!("ptxas reads PTX {version}, which VERSIONS lacks")),
            _ => {}
        }
    }
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.join("\n"));
}

/// `ptxas` 13.5 reads PTX 5.1 as 6.0, though PTX has no 5.1: it takes `shfl.sync`, `vote.sync`, `bar.warp.sync` and
/// `.target sm_70` at 5.1 and at 6.0, which it refuses at 5.0, and refuses at 5.1 what came after 6.0, `.target sm_72`,
/// which it takes at 6.1. It fails the day `ptxas` reads 5.1 as another version, for [`READ_AS_6_0`] and GUIDE.md to
/// say what it is.
#[test]
fn ptxas_reads_ptx_5_1_as_6_0() {
    let Some(ptxas) = ptxas() else {
        eprintln!("no ptxas here: skipped; set PTXAS to run it");
        return;
    };
    // Each line of a kernel, the GPU its module is for, and the first version of PTX that has both.
    let lines = [
        ("shfl.sync.idx.b32 %r1, %r2, 0, 31, 0xffffffff;", 30, 60),
        ("vote.sync.any.pred %p1, %p2, 0xffffffff;", 30, 60),
        ("bar.warp.sync 0xffffffff;", 30, 60),
        ("", 70, 60),
        ("", 72, 61),
    ];
    let tries: Vec<_> =
        lines.into_iter().flat_map(|(line, sm, came)| [50, READ_AS_6_0, 60, 61].map(|v| (line, sm, came, v))).collect();
    let modules: Vec<(String, u32)> = tries
        .iter()
        .map(|&(line, sm, _, v)| {
            let text = format!(
                ".version 6.0\n.target sm_{sm}\n.address_size 64\n\n.visible .entry k()\n{{\n\t.reg .b32 %r<3>;\n\
                 \t.reg .pred %p<3>;\n\t{line}\n\tret;\n}}\n"
            );
            let text = retarget(&text, v, sm);
            let gpu = gpu_of(&text);
            (text, gpu)
        })
        .collect();
    let mut failures = vec![];
    for (&(line, sm, came, v), (refused, _)) in tries.iter().zip(assemble(&ptxas, "five_one", &modules)) {
        // 5.1 is read as 6.0: ptxas takes a line from the version that brought it.
        let takes = (if v == READ_AS_6_0 { 60 } else { v }) >= came;
        let (version, with) = (dotted(v), if line.is_empty() { String::new() } else { format!(", `{line}`") });
        match (takes, refused.values().next()) {
            (true, Some(m)) => failures.push(format!("ptxas refuses PTX {version} for sm_{sm}{with}: {m}")),
            (false, None) => failures.push(format!("ptxas takes PTX {version} for sm_{sm}{with}")),
            _ => {}
        }
    }
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.join("\n"));
}

/// `ptxas` takes each form kernite knows, written each way kernite has, at the version and for the GPU kernite says it
/// needs, and not at one before; and it refuses each name kernite refuses. An `f16x2` goes in each register of an
/// integer type, alone, as a part of it, negated, and in each register of a vector at once: where kernite takes it,
/// `ptxas` takes it and makes the cubin of a `b32`; where kernite refuses it, `ptxas` refuses the kernel with that
/// register declared `.f16x2`, but for the places [`ptxas_bug`] says.
#[test]
fn ptxas_takes_every_form_kernite_knows_where_kernite_says() {
    let Some(ptxas) = ptxas() else {
        eprintln!("no ptxas here: skipped; set PTXAS to run it");
        return;
    };
    let (mut n, mut failures) = (0, vec![]);
    let mut what: BTreeMap<String, String> = BTreeMap::new();
    // What kernite says of each kernel it refuses, with an `f16x2` or by name, for `ptxas` to refuse too.
    let mut why: BTreeMap<String, String> = BTreeMap::new();
    // A group is the kernels that need the same, and that `ptxas` should make the same of.
    let mut groups: BTreeMap<(Group, u32, u32), Vec<Entry>> = BTreeMap::new();
    let mut add = |(e, place): (Entry, String), what: &mut BTreeMap<String, String>, group: Group| {
        let line = e.program.body.iter().find_map(|i| match i {
            kernite_core::Item::Inst(inst) => Some(inst.op.0.clone()),
            _ => None,
        });
        let mut line = line.unwrap_or_default();
        if group != Group::Plain {
            let halves: Vec<String> =
                e.regs.iter().filter(|(_, t)| *t == Ty::F16x2).map(|(reg, _)| format!("%{reg}")).collect();
            line += &format!(" with {} an `f16x2`, {place}", halves.join(", "));
        }
        if group == Group::Bug {
            line += ", a bug of ptxas that `ptxas_bug` says";
        }
        what.insert(e.program.name.clone(), line);
        groups.entry((group, e.needs.ptx, e.needs.sm)).or_default().push(e);
    };
    let mut refused = vec![];
    for form in forms::forms() {
        for (name, choice) in names(form) {
            let bound = match forms::find(&name) {
                Ok(bound) => bound,
                Err(m) => {
                    refused.push((form, name, choice, m));
                    continue;
                }
            };
            let Some(b) = bound.iter().find(|b| std::ptr::eq(b.form, form)) else { continue };
            for style in [
                Style::Vars,
                Style::Values,
                Style::Sinks,
                Style::Short,
                Style::Single,
                Style::Memory,
                Style::Parts,
                Style::Specials,
                Style::Narrow,
                Style::Least,
                Style::Most,
            ] {
                n += 1;
                match instance(n, &name, b, style, false) {
                    None => {}
                    Some(Ok(e)) => add(e, &mut what, Group::Plain),
                    Some(Err(m)) => failures.push(m),
                }
            }
            // An `f16x2` for each register of an integer type, and for each vector, until there is no register
            // left for one.
            for k in 0.. {
                let mut left = false;
                for style in [Style::Packed(k), Style::PackedPart(k), Style::PackedNeg(k), Style::PackedAll(k)] {
                    n += 1;
                    match instance(n, &name, b, style, false) {
                        None => continue,
                        Some(Ok(e)) => add(e, &mut what, Group::Packed),
                        // kernite refuses it, so `ptxas` should: the kernel with the register declared `.f16x2`.
                        Some(Err(m)) => match instance(n, &name, b, style, true).expect("the style gives it") {
                            Ok(e) => {
                                why.insert(e.0.program.name.clone(), m);
                                let group = if ptxas_bug(b.form, &e.1) { Group::Bug } else { Group::Wrong };
                                add(e, &mut what, group);
                            }
                            Err(m) => failures.push(m),
                        },
                    }
                    left = true;
                }
                if !left {
                    break;
                }
            }
        }
    }
    for e in others(&mut n) {
        add((e, String::new()), &mut what, Group::Plain);
    }
    for ((_, ptx, ..), entries) in groups.iter().filter(|((_, ptx, ..), _)| !VERSIONS.contains(ptx)) {
        let (count, some): (usize, Vec<&str>) =
            (entries.len(), entries.iter().take(5).map(|e| what[&e.program.name].as_str()).collect());
        failures.push(format!("{count} kernels need PTX {}, which VERSIONS lacks: {}", dotted(*ptx), some.join("; ")));
    }
    // Each group as kernite writes it, for the GPU it is for and for each newer it may run
    // on, as the driver assembles it for the GPU it runs on; at the version it needs, for the
    // newest GPU of that version; and at the newest version, for each GPU from its own to the
    // newest, as ptxas takes some older that it refuses newer. And one step older in each way
    // it needs more than PTX 1.0 and Fermi, and for the GPU it is too new for: then each
    // kernel is wrong. A group of kernels with an `f16x2` goes once, as kernite writes it:
    // what it asks of ptxas is how it reads an `f16x2`, which no version or GPU changes, and
    // the other groups try each form at each version and GPU. Where kernite takes the
    // `f16x2`, the group goes with its twin, with each `f16x2` a `b32`, which `ptxas` makes
    // the same cubin of; where it refuses, `ptxas` does too, of a hundred at a time, but for
    // the refusals of `ptxas_bug`, which it takes.
    let mut modules: Vec<(String, u32, Expect)> = vec![];
    let mut twins: Vec<(usize, &[Entry])> = vec![];
    for (&(group, ptx, sm), entries) in groups.iter().filter(|((_, ptx, ..), _)| VERSIONS.contains(ptx)) {
        let gpu = newest(ptx);
        if gpu < sm {
            let v = dotted(ptx);
            failures.push(format!("{} kernels need PTX {v} and sm_{sm}, which no module of {v} is for", entries.len()));
        }
        if matches!(group, Group::Wrong | Group::Bug) {
            let expect = if group == Group::Bug { Expect::Takes } else { Expect::Refuses };
            modules.extend(entries.chunks(100).map(|some| at(module(some), expect)));
            continue;
        }
        let text = module(entries);
        if group == Group::Packed {
            twins.push((modules.len(), entries));
            modules.push(at(text, Expect::Takes));
            modules.push(at(module(&as_bits(entries)), Expect::Takes));
            continue;
        }
        let exact = retarget(&text, ptx, gpu);
        for &(g, _) in kernite_ptx::GPUS.iter().filter(|&&(g, _)| g >= sm) {
            let latest = retarget(&text, NEWEST, g);
            if latest != text && latest != exact {
                modules.push(at(latest, Expect::Takes));
            }
        }
        if exact != text {
            modules.push(at(exact, Expect::Takes));
        }
        for &(g, _) in kernite_ptx::GPUS.iter().filter(|&&(g, _)| g > gpu_of(&text)) {
            modules.push((text.clone(), g, Expect::Takes));
        }
        modules.push(at(text.clone(), Expect::Takes));
        if let Some(v) = before(ptx) {
            modules.push(at(retarget(&text, v, newest(v)), Expect::Refuses));
        }
        if sm > 0 {
            let g = kernite_ptx::GPUS.iter().map(|&(g, _)| g).rfind(|&g| g < sm).unwrap_or(FERMI);
            modules.push(at(retarget(&text, ptx.max(came(g)), g), Expect::Refuses));
        }
    }
    // And the names kernite refuses, for the newest GPU and at the newest version kernite knows,
    // where nothing but the name is wrong.
    let refusals = refused.len();
    let mut hand = vec![];
    for (form, name, choice, m) in refused {
        n += 1;
        let text = by_hand(n, &name, form, choice);
        let line =
            text.lines().find(|l| l.trim_start().starts_with(name.as_str())).expect("its line").trim().to_string();
        what.insert(format!("k{n}"), line);
        why.insert(format!("k{n}"), m);
        hand.push(text);
    }
    let f = callee().text();
    let (version, gpu) = (dotted(NEWEST), newest(NEWEST));
    for some in hand.chunks(100) {
        let text = format!(
            "// Written by hand.\n\n.version {version}\n.target sm_{gpu}\n.address_size 64\n\n{f}\n{}",
            some.join("\n")
        );
        modules.push(at(text, Expect::Refuses));
    }
    let texts: Vec<(String, u32)> = modules.iter().map(|(t, g, _)| (t.clone(), *g)).collect();
    let results = assemble(&ptxas, "forms", &texts);
    for ((text, g, expect), (refused, _)) in modules.iter().zip(&results) {
        let header = match *g == gpu_of(text) {
            true => header(text),
            false => format!("{} for sm_{g}", header(text)),
        };
        if let Some(m) = refused.get("?") {
            failures.push(format!("{header}: {m}"));
            continue;
        }
        let what = |k: &str| what.get(k).map_or("?", String::as_str).to_string();
        match expect {
            Expect::Takes => {
                for (k, message) in refused {
                    failures.push(format!("{header}: `{}`: {message}", what(k)));
                }
            }
            Expect::Refuses => {
                for k in
                    text.lines().filter_map(|l| l.strip_prefix(".visible .entry ")).filter_map(|l| l.split('(').next())
                {
                    if !refused.contains_key(k) {
                        let mut taken = format!("{header}: takes `{}` too", what(k));
                        if let Some(m) = why.get(k) {
                            taken += &format!(", which kernite refuses: {m}");
                        }
                        failures.push(taken);
                    }
                }
            }
        }
    }
    for (i, entries) in twins {
        let ((taken, cubin), (twin_taken, twin)) = (&results[i], &results[i + 1]);
        if taken.is_empty() && twin_taken.is_empty() && cubin != twin {
            for k in differ(&ptxas, entries, modules[i].1) {
                let line = what.get(&k).map_or("?", String::as_str);
                failures
                    .push(format!("{}: ptxas makes another cubin of `{line}` than of a `b32`", header(&modules[i].0)));
            }
        }
    }
    eprintln!("{n} instances in {} modules; {refusals} names kernite refuses", modules.len());
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.join("\n"));
}

/// A warning of `ptxas` is a refusal of the kernel it warns of, for
/// [`ptxas_takes_every_form_kernite_knows_where_kernite_says`], as an error is. `ptxas` 13.5 refuses `shfl` and `vote`
/// without `.sync` in a module of PTX 6.4 or newer for `sm_70` or newer, and takes them in any other, with a warning:
/// that they may give unpredictable results on `sm_70` and newer, and from PTX 6.0 that PTX deprecated them. So
/// `refused` has the kernel of each of them wrong at every version of PTX from 3.0, where `shfl` came, and for every
/// GPU the version has, with what `ptxas` says of the form first; but for `vote.ballot.b32` in PTX 6.0 and 6.1, which
/// `ptxas` takes without a word, though it warns of it in every other module it takes.
#[test]
fn ptxas_warns_of_shfl_and_vote_without_sync_or_refuses_them_at_every_version_and_gpu() {
    let Some(ptxas) = ptxas() else {
        eprintln!("no ptxas here: skipped; set PTXAS to run it");
        return;
    };
    let wrong = [
        ("shfl", "shfl", "shfl.idx.b32 %r1, %r2, 0, 31;"),
        ("any", "vote", "vote.any.pred %p0, %p1;"),
        ("ballot", "vote", "vote.ballot.b32 %r1, %p1;"),
    ];
    let regs = ".reg .b32 %r<4>;\n\t.reg .pred %p<2>;";
    let mut text = ".version 3.0\n.target sm_30\n.address_size 64\n\n".to_string();
    for (name, _, line) in wrong {
        text += &format!(".visible .entry {name}()\n{{\n\t{regs}\n\t{line}\n\tret;\n}}\n\n");
    }
    let at: Vec<(u32, u32)> = (VERSIONS.into_iter().filter(|&v| v >= 30))
        .flat_map(|v| kernite_ptx::GPUS.into_iter().filter(move |&(_, came)| came <= v).map(move |(g, _)| (v, g)))
        .collect();
    let modules: Vec<(String, u32)> =
        at.iter().map(|&(v, g)| retarget(&text, v, g)).map(|m| (m.clone(), gpu_of(&m))).collect();
    for ((v, g), (refusals, _)) in at.into_iter().zip(assemble(&ptxas, "deprecated", &modules)) {
        for (name, op, line) in wrong {
            let what = match (name, v, g) {
                ("ballot", 60 | 61, _) => None,
                (_, 64.., 70..) => Some("is not supported on .target sm_70 and higher from PTX ISA version 6.4"),
                _ => Some("may produce unpredictable results on sm_70 and later architectures"),
            };
            let want = what.map(|what| format!("error   : Instruction '{op}' without '.sync' {what}"));
            assert_eq!(refusals.get(name), want.as_ref(), "PTX {v} for sm_{g}: `{line}`");
        }
    }
}

/// Kernels that keep what they work on in shared memory: as much as the kernel declares, and
/// as much as the launch gives.
const SHARED: &str = r#"
    /// `y[b]`, the sum of the 256 elements of `x` from `256 b`, for blocks of 256 threads.
    pub unsafe extern "ptx-kernel" fn block_sum(x: *const f32, y: *mut f32) {
        let tile: shared<[f32; 256], 16>;
        let (t, b, i, mine): (u32, u32, u32, u32);
        mov.u32(t, tid.x);
        mov.u32(b, ctaid.x);
        mad.lo.u32(i, b, 256, t);
        let at: u64;
        mad.wide.u32(at, i, 4, x);
        let (v, w): (f32, f32);
        ld.global.f32(v, [at]);
        mov.u32(mine, tile);
        mad.lo.u32(mine, t, 4, mine);
        st.shared.f32([mine], v);
        let p: pred;
        for half in [128, 64, 32, 16, 8, 4, 2, 1] {
            bar.sync(0);
            setp.lt.u32(p, t, half);
            if p {
                ld.shared.f32(w, [mine + 4 * half]);
                add.rn.f32(v, v, w);
                st.shared.f32([mine], v);
            }
        }
        setp.eq.u32(p, t, 0);
        if p {
            ld.shared.f32(v, [tile]);
            mad.wide.u32(at, b, 4, y);
            st.global.f32([at], v);
        }
    }

    /// The `n` elements of `x` from `n b` reversed, for blocks of `n` threads, through the
    /// shared memory the launch gives: `4 n` bytes.
    pub unsafe extern "ptx-kernel" fn reverse(x: *mut f32) {
        let buf: shared<[f32]>;
        let (t, n, i, j, s): (u32, u32, u32, u32, u32);
        mov.u32(t, tid.x);
        mov.u32(n, ntid.x);
        mov.u32(i, ctaid.x);
        mad.lo.u32(i, i, n, t);
        let at: u64;
        mad.wide.u32(at, i, 4, x);
        let v: f32;
        ld.global.f32(v, [at]);
        mov.u32(s, buf);
        mad.lo.u32(j, t, 4, s);
        st.shared.f32([j], v);
        bar.sync(0);
        sub.u32(j, n, t);
        sub.u32(j, j, 1);
        mad.lo.u32(j, j, 4, s);
        ld.shared.f32(v, [j]);
        st.global.f32([at], v);
    }
"#;

#[test]
fn memory_is_declared_where_ptx_has_it() {
    let ptx = kernite::ptx_str(SHARED).unwrap();
    let (head, entries) = ptx.split_once(".visible .entry ").unwrap();
    let want = "// Written by kernite.

.version 3.0
.target sm_30
.address_size 64

.extern .shared .align 4 .b8 _ZZ7reverseE3buf[];

";
    assert_eq!(head, want);
    let block_sum = entries.split(".visible .entry ").next().unwrap();
    for line in [
        "\t.reg .u64 %x, %y, %at;\n\t.shared .align 16 .b8 tile[1024];\n\n",
        "\tmov.u32 %mine, tile;\n",
        "\t@%p ld.shared.f32 %w, [%mine+512];\n\t@%p add.rn.f32 %v, %v, %w;\n",
        "\t@%p ld.shared.f32 %v, [tile];\n",
    ] {
        assert!(block_sum.contains(line), "{line}");
    }
    assert!(entries.contains("\tmov.u32 %s, _ZZ7reverseE3buf;\n"));
}

#[test]
fn memory_is_reached_in_its_state_space_and_within_it() {
    let t = "let t: shared<[f32; 256], 16>; let f: f32;";
    let cases = [
        ("let t: shared<[f32; 0]>;".to_string(), "2:8: `shared<[f32; 0]>` holds nothing"),
        ("let t: shared<[f32; 4], 3>;".into(), "2:8: what an address is a multiple of is a power of two, not 3"),
        ("let t: shared<[f64; 4], 4>;".into(), "2:8: an array of `f64` is aligned to 8 at least, not 4"),
        (
            "let t: shared<[pred; 4]>;".into(),
            "2:8: memory holds no `pred`, which is a register's alone: a `u8` holds a bool",
        ),
        (
            "let t: shared<f32, 16, 2>;".into(),
            "2:8: `shared<..>` is an array of PTX's types and what its address is a multiple of, `shared<[f32; \
             256], 16>`, or one, `shared<f32>`; not `shared<f32, 16, 2>`",
        ),
        (
            "let t: local<[f32]>;".into(),
            "2:8: `local<[f32]>` has no size, which only shared memory may leave to the launch: `local<[f32; 64]>`",
        ),
        (
            "let a: shared<[f32; 8192]>; let b: shared<[f32; 4097]>;".into(),
            "2:36: this makes the kernel's shared memory 49156 bytes; it declares 48 KB at most, and has more as \
             `shared<[f32]>`, which the launch gives",
        ),
        (
            "let a: shared<[u8; 49153]>;".into(),
            "2:8: this makes the kernel's shared memory 49153 bytes; it declares 48 KB at most, and has more as \
             `shared<[u8]>`, which the launch gives",
        ),
        (
            "let a: shared<[f32]>; let b: shared<[u32]>;".into(),
            "2:30: `b` would start where `a` does, after the memory with a size: a kernel has one `shared<[T]>`, \
             which the launch gives",
        ),
        (
            "let l: local<[u32; 600000]>;".into(),
            "2:8: this makes a thread's local memory 2400000 bytes, and a thread has 512 KB",
        ),
        (format!("{t} ld.global.f32(f, [t]);"), "2:61: `ld.global.f32` reads global memory; `t` is shared memory"),
        (
            format!("{t} ld.f32(f, [t]);"),
            "2:54: `ld.f32` reads at a generic address; `t` is shared memory, which an instruction reaches by its \
             name only as `.shared`",
        ),
        (
            "let i: u32; let l: local<[u32; 4]>; atom.add.u32(i, [l], 1);".into(),
            "2:53: `atom.add.u32` reads and writes at a generic address; `l` is local memory, which an instruction \
             reaches by its name only as `.local`",
        ),
        (
            format!("{t} ld.shared.f32(f, [t + 1024]);"),
            "2:61: `ld.shared.f32` reads 4 bytes at `[t + 1024]`, past the end of `t`, 1024 bytes",
        ),
        (
            format!("{t} st.shared.f32([t + 1024], f);"),
            "2:58: `st.shared.f32` writes 4 bytes at `[t + 1024]`, past the end of `t`, 1024 bytes",
        ),
        (format!("{t} ld.shared.f32(f, [t - 4]);"), "2:61: `ld.shared.f32` reads at `[t - 4]`, before `t` starts"),
        (
            "let d: shared<[f32]>; let f: f32; ld.shared.f32(f, [d - 4]);".into(),
            "2:52: `ld.shared.f32` reads at `[d - 4]`, before `d` starts",
        ),
        (
            format!("{t} let (a, b, c, d): (f32, f32, f32, f32); ld.shared.v4.f32((a, b, c, d), [t + 8]);"),
            "2:115: `ld.shared.v4.f32` reads 16 bytes at once, at an address that is a multiple of 16; `[t + 8]` is \
             not one",
        ),
        (
            format!("{t} let i: u32; atom.shared.add.u32(i, [t + 2], 1);"),
            "2:79: `atom.shared.add.u32` reads and writes 4 bytes at once, at an address that is a multiple of 4; \
             `[t + 2]` is not one",
        ),
        (
            format!("{t} cp.async.ca.shared.global([t + 4], [x], 16);"),
            "2:70: `cp.async.ca.shared.global` writes 16 bytes at once, at an address that is a multiple of 16; \
             `[t + 4]` is not one",
        ),
        (
            format!("{t} cp.async.ca.shared.global([t + 1016], [x], 16);"),
            "2:70: `cp.async.ca.shared.global` writes 16 bytes at `[t + 1016]`, past the end of `t`, 1024 bytes",
        ),
        (
            format!("{t} cp.async.ca.shared.global([n], [t], 4);"),
            "2:75: `cp.async.ca.shared.global` reads global memory; `t` is shared memory",
        ),
        (
            format!("{t} let s: u64; mbarrier.arrive.shared.b64(s, [t + 4]);"),
            "2:86: `mbarrier.arrive.shared.b64` reads and writes 8 bytes at once, at an address that is a multiple \
             of 8; `[t + 4]` is not one",
        ),
        (
            format!("{t} let s: u64; mbarrier.arrive.b64(s, [t]);"),
            "2:79: `mbarrier.arrive.b64` reads and writes at a generic address; `t` is shared memory, which an \
             instruction reaches by its name only as `.shared`",
        ),
        (
            format!("{t} let d: b32; ldmatrix.sync.aligned.m8n8.x1.shared.b16(d, [t + 8]);"),
            "2:100: `ldmatrix.sync.aligned.m8n8.x1.shared.b16` reads 16 bytes at once, at an address that is a \
             multiple of 16; `[t + 8]` is not one",
        ),
        (
            format!("{t} let d: b32; ldmatrix.sync.aligned.m8n8.x1.shared.b16(d, [t + 1016]);"),
            "2:100: `ldmatrix.sync.aligned.m8n8.x1.shared.b16` reads 16 bytes at `[t + 1016]`, past the end of `t`, \
             1024 bytes",
        ),
        (
            format!("{t} let c: [f32; 8]; wmma.load.c.sync.aligned.row.m16n16k16.shared.f32(c, [t]);"),
            "2:114: `wmma.load.c.sync.aligned.row.m16n16k16.shared.f32` reads 32 bytes at once, at an address that \
             is a multiple of 32; `t` is at a multiple of 16: `shared<[f32; 256], 32>`",
        ),
        (
            "let u: shared<[f32; 256], 32>; let c: [f32; 8]; wmma.store.d.sync.aligned.row.m16n16k16.shared.f32([u + \
             1024], c);"
                .into(),
            "2:100: `wmma.store.d.sync.aligned.row.m16n16k16.shared.f32` writes 32 bytes at `[u + 1024]`, past the \
             end of `u`, 1024 bytes",
        ),
        (
            "let u: shared<[f32; 64]>; let (a, b): (f32, f32); ld.shared.v2.f32((a, b), [u]);".into(),
            "2:76: `ld.shared.v2.f32` reads 8 bytes at once, at an address that is a multiple of 8; `u` is at a \
             multiple of 4: `shared<[f32; 64], 8>`",
        ),
        (
            format!("{t} mov.f32(f, t);"),
            "2:55: `mov.f32` takes no memory as an operand: `mov.u32(a, t)` puts the address of `t` in `a`, and \
             `[t]` is what it holds",
        ),
        (
            "let l: local<[u32; 16]>; let g: u64; cvta.shared.u64(g, l);".into(),
            "2:57: `cvta.shared.u64` takes the address of shared memory; `l` is local memory",
        ),
        (
            format!("{t} if t {{ mov.f32(f, 1.0); }}"),
            "2:55: an instruction is guarded by a `pred`; this is `t`, shared memory",
        ),
        (format!("{t} mov.f32(f, t.x);"), "2:57: `t` is memory, `shared<[f32; 256], 16>`; it is no `x`"),
    ];
    for (body, want) in cases {
        assert_eq!(compile("x: *mut f32, n: u32", &body).unwrap_err(), want, "{body}");
    }
    // A kernel declares 48 KB of shared memory, to the byte, and no more.
    let ptx = compile("x: *mut f32, n: u32", "let a: shared<[u8; 49152]>; mov.u32(n, a);").unwrap();
    assert!(ptx.contains("\t.shared .align 1 .b8 a[49152];\n"));
    // Memory takes the name it is given, but for one PTX has or the kernel's other names.
    let ptx = compile(
        "x: *mut f32, n: u32",
        "let WARP_SZ: shared<u32>; let k: shared<u32>; let k_param_0: shared<u32>; mov.u32(n, WARP_SZ); \
         mov.u32(n, k); mov.u32(n, k_param_0);",
    )
    .unwrap();
    for line in ["\tmov.u32 %n, WARP_SZ_;\n", "\tmov.u32 %n, k_1;\n", "\tmov.u32 %n, k_param_0_1;\n"] {
        assert!(ptx.contains(line), "{line}");
    }
}

/// Memory of the module's, which its kernels name: a histogram they keep from launch to
/// launch, and what they read.
const STATICS: &str = r#"
    const N: usize = 16;

    /// How many of each value `count` has seen.
    pub static COUNTS: global<[u32; N]>;
    /// What `scale` multiplies by.
    static TABLE: constant<[f32; 4], 16> = [1.0, 2.0, 0.5, -1.0];
    static SEEN: global<u64> = 7;
    static HALVES: constant<[f16; 2]> = [1.5, 0.0];
    static SIGNED: global<[s8; 3]> = [-1, 2, 0];

    /// Counts `x[i]`, each below 16, for each `i < n`.
    pub unsafe extern "ptx-kernel" fn count(x: *const u32, n: u32) {
        let (i, v): (u32, u32);
        mov.u32(i, tid.x);
        let past: pred;
        setp.ge.u32(past, i, n);
        if past {
            ret();
        }
        let at: u64;
        mad.wide.u32(at, i, 4, x);
        ld.global.u32(v, [at]);
        mov.u64(at, COUNTS);
        mad.wide.u32(at, v, 4, at);
        red.global.add.u32([at], 1);
        atom.global.add.u32(v, [COUNTS + 8], 1);
        let s: u64;
        ld.global.u64(s, [SEEN]);
        cvta.global.u64(at, COUNTS);
        let b: s8;
        ld.global.s8(b, [SIGNED + 2]);
    }

    /// `x[0] *= TABLE[1]`.
    pub unsafe extern "ptx-kernel" fn scale(x: *mut f32) {
        let (v, w): (f32, f32);
        ld.global.f32(v, [x]);
        ld.const.f32(w, [TABLE + 4]);
        mul.rn.f32(v, v, w);
        st.global.f32([x], v);
        let a: u32;
        mov.u32(a, TABLE);
        let g: u64;
        cvta.const.u64(g, TABLE);
        let h: f16;
        ld.const.b16(h, [HALVES + 2]);
    }
"#;

#[test]
fn a_static_is_declared_once_for_each_kernel_to_name() {
    let ptx = kernite::ptx_str(STATICS).unwrap();
    let (head, entries) = ptx.split_once(".visible .entry ").unwrap();
    let want = "// Written by kernite.

.version 3.1
.target sm_30
.address_size 64

.global .align 4 .b8 COUNTS[64];
.const .align 16 .b8 TABLE[16] = {0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 0, 63, 0, 0, 128, 191};
.global .align 8 .b8 SEEN[8] = {7};
.const .align 2 .b8 HALVES[4] = {0, 62};
.global .align 1 .b8 SIGNED[3] = {255, 2};

";
    assert_eq!(head, want);
    for line in [
        "\tmov.u64 %at, COUNTS;\n",
        "\tatom.global.add.u32 %v, [COUNTS+8], 1;\n",
        "\tld.global.u64 %s, [SEEN];\n",
        "\tcvta.global.u64 %at, COUNTS;\n",
        "\tld.const.f32 %w, [TABLE+4];\n",
        "\tmov.u32 %a, TABLE;\n",
        "\tcvta.const.u64 %g, TABLE;\n",
        "\tld.const.b16 %h, [HALVES+2];\n",
    ] {
        assert!(entries.contains(line), "{line}");
    }
    // In a scope, each is named as C++ names a variable of its namespace, as a kernel is.
    let module = kernite::lang::parse(STATICS).unwrap();
    let scope = kernite::ptx_scope(&["kernite", "my_app", "gpu"]);
    let statics: Vec<_> = kernite::ptx_statics(&module, &scope).into_iter().map(Result::unwrap).collect();
    assert_eq!(statics[0].code.memory.name, "_ZN7kernite6my_app3gpu6COUNTSE");
    assert_eq!(statics[0].mentions.iter().map(|m| m.name.as_str()).collect::<Vec<_>>(), ["N"]);
    assert!(statics[1..].iter().all(|s| s.mentions.is_empty()));
    let count = kernite::ptx(&module, kernite::function(&module, "count").unwrap(), &[], &scope).unwrap();
    assert!(count.code.text().contains("\tmov.u64 %at, _ZN7kernite6my_app3gpu6COUNTSE;\n"));
}

#[test]
fn a_static_is_memory_of_a_type_the_host_reads_holding_what_it_says() {
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
            "static A: shared<[u32; 4]>;",
            "1:11: a static of the module is memory: in global memory, `global<[u32; 16]>`, or in constant memory, \
             which its kernels only read, `constant<[f32; 4]>`; not `shared<[u32; 4]>`",
        ),
        (
            "static A: u32;",
            "1:11: a static of the module is memory: in global memory, `global<[u32; 16]>`, or in constant memory, \
             which its kernels only read, `constant<[f32; 4]>`; not `u32`",
        ),
        (
            "static A: global<[u32]>;",
            "1:11: `global<[u32]>` has no size, which only shared memory may leave to the launch: `global<[u32; 64]>`",
        ),
        (
            "static A: global<[pred; 4]>;",
            "1:11: memory holds no `pred`, which is a register's alone: a `u8` holds a bool",
        ),
        (
            "static A: global<[f16x2; 4]>;",
            "1:11: a static holds numbers the host reads as Rust's: `b8` to `b64`, `u8` to `u64`, `s8` to `s64`, \
             `f16`, `bf16`, `f32` and `f64`; not `f16x2`",
        ),
        ("static A: global<[u32; 0]>;", "1:11: `global<[u32; 0]>` holds nothing"),
        ("static A: global<[u32; 4], 3>;", "1:11: what an address is a multiple of is a power of two, not 3"),
        ("static A: global<[u32; M]>;", "1:24: `M` is not defined"),
        ("static A: global<[u32; 2]> = [1, 2, 3];", "1:30: `global<[u32; 2]>` holds 2 elements; this has 3"),
        ("static A: global<[u32; 2]> = 1;", "1:30: `global<[u32; 2]>` holds an array of `u32`; this is an integer"),
        ("static A: global<[u32; 2]> = [1.5, 2];", "1:30: element 0 is an f32, not a `u32`"),
        ("static A: global<[u32; 2]> = [1, -1];", "1:34: a u32 cannot be negated"),
        ("static A: global<u32> = [1];", "1:25: this is an array, not a `u32`"),
        ("static A: global<u32> = true;", "1:25: this is a bool, not a `u32`"),
        ("static A: global<u32> = 1u8;", "1:25: this is a `u8`, not a `u32`"),
        ("static A: global<s32> = 1i8;", "1:25: this is an `i8`, not an `s32`"),
        ("static A: global<s32> = true;", "1:25: this is a bool, not an `s32`"),
        ("static A: global<f32> = 1;", "1:25: this is an integer, and an `f32` a float: `1.0`"),
        ("static A: global<u8> = 256;", "1:24: 256 is out of range for u8"),
        ("static A: global<s8> = 128;", "1:24: this is 128, out of range for `s8`"),
        ("static A: global<[b8; 2]> = [255, -129];", "1:29: element 1 is -129, out of range for `b8`"),
        ("static A: global<f16> = 65520.0;", "1:25: 65520.0 is out of range for f16"),
        ("static A: global<u32> = M;", "1:25: `M` is not defined"),
        (
            "static A: constant<[u8; 65536]>; static B: constant<u8>;",
            "1:44: this makes the module's constant memory 65537 bytes, and a module has 64 KB",
        ),
        (
            "static A: constant<[u8; 65535]>; static B: constant<[u32; 1]>;",
            "1:44: this makes the module's constant memory 65540 bytes, and a module has 64 KB",
        ),
        (
            "static A: global<[u8; 4294967295]>; static B: global<u8>;",
            "1:47: this makes the module's global memory as much as 4294967296 bytes, each static aligned, and \
             `ptxas` writes its size in 32 bits: a module has less than 4 GB",
        ),
        (
            "static A: global<[u8; 18446744073709551615]>;",
            "1:11: this makes the module's global memory as much as 18446744073709551615 bytes, each static aligned, \
             and `ptxas` writes its size in 32 bits: a module has less than 4 GB",
        ),
        // The value is checked before the room left.
        (
            "static A: constant<[u8; 65536]>; static B: constant<[u8; 2]> = [1, 2, 3];",
            "1:64: `constant<[u8; 2]>` holds 2 elements; this has 3",
        ),
        ("static k_param_0: global<u32>;", "`k_param_0` is `k_param_0` in PTX, as a parameter of `k` is"),
        ("static WARP_SZ: global<u32>;", "`WARP_SZ` is `WARP_SZ` in PTX, as PTX's own `WARP_SZ` is"),
        // A kernel that names one that does not compile says so, and why, where it is declared.
        (
            "static A: global<[u32; 2]> = [1]; unsafe extern \"ptx-kernel\" fn j() { ld.global.u32(v, [A]); }",
            "1:30: `global<[u32; 2]>` holds 2 elements; this has 1",
        ),
    ];
    let k = "unsafe extern \"ptx-kernel\" fn k(x: *mut u32) {}";
    for (items, want) in cases {
        assert_eq!(kernite::ptx_str(&format!("{items}\n{k}")).unwrap_err(), want, "{items}");
    }
    // Every byte of constant memory, and a `b` type's integers of either sign.
    let ptx = kernite::ptx_str("static A: constant<[u8; 65535]>; static B: constant<u8>;").unwrap();
    assert!(ptx.contains(".const .align 1 .b8 A[65535];\n.const .align 1 .b8 B[1];\n"));
    let ptx = kernite::ptx_str("static A: global<[b8; 2]> = [255, -128];").unwrap();
    assert!(ptx.contains(".global .align 1 .b8 A[2] = {255, 128};\n"));
    // A static that fails takes no room: `A` fails at its value, and `B` lays out. `ptx_str` stops
    // at `A`, so only `ptx_statics` shows it.
    let src = "static A: constant<[u8; 65536]> = 1; static B: constant<u8>;";
    let made = kernite::ptx_statics(&kernite::lang::parse(src).unwrap(), "");
    let want = "1:35: `constant<[u8; 65536]>` holds an array of `u8`; this is an integer";
    assert_eq!(made[0].as_ref().unwrap_err().at(src), want);
    assert_eq!(made[1].as_ref().err(), None);
    // Nor does one that has no room: `B` is refused beside `A`, and `C` fits beside `A` alone.
    let src = "static A: constant<[u8; 65535]>; static B: constant<[u32; 1]>; static C: constant<u8>;";
    let made = kernite::ptx_statics(&kernite::lang::parse(src).unwrap(), "");
    let want = "1:44: this makes the module's constant memory 65540 bytes, and a module has 64 KB";
    assert_eq!(made[1].as_ref().unwrap_err().at(src), want);
    assert_eq!(made[2].as_ref().err(), None);
}

/// The room of the types of the statics a module keeps, `ptx_room`, which the macro asks once Rust has
/// left out those `cfg` does, is that of `ptx_statics` for a module that keeps them all: each type that
/// does not fit beside those before it that do is refused in the same words, and takes none; the
/// constant memory and the global are counted apart.
#[test]
fn the_room_of_the_types_a_module_keeps_is_counted_in_order() {
    let constant = |all| format!("this makes the module's constant memory {all} bytes, and a module has 64 KB");
    let global = "this makes the module's global memory as much as 4294967296 bytes, each static aligned, and \
                  `ptxas` writes its size in 32 bits: a module has less than 4 GB";
    let cases = [
        ("constant<[u8; 65535]>", None),
        ("constant<[u32; 1]>", Some(constant(65540))),
        // Beside the first alone, it fills the 64 KB; the next has none left.
        ("constant<u8>", None),
        ("constant<u8>", Some(constant(65537))),
        ("global<[u8; 4294967295]>", None),
        ("global<u8>", Some(global.to_string())),
    ];
    let types: Vec<&str> = cases.iter().map(|(ty, _)| *ty).collect();
    let want: Vec<Option<String>> = cases.into_iter().map(|(_, why)| why).collect();
    assert_eq!(kernite::ptx_room(&types), want);
    assert_eq!(kernite::ptx_room(&[]), []);
}

#[test]
fn a_static_is_reached_in_its_state_space_and_within_it() {
    let (a, c) = ("static A: global<[u32; 4]>;", "static C: constant<[u32; 4]>;");
    let cases = [
        (
            a,
            "ld.global.u32(v, [A + 16]);",
            "3:18: `ld.global.u32` reads 4 bytes at `[A + 16]`, past the end of `A`, 16 bytes",
        ),
        (a, "ld.global.u32(v, [A - 4]);", "3:18: `ld.global.u32` reads at `[A - 4]`, before `A` starts"),
        (
            a,
            "ld.global.u32(v, [A + 2]);",
            "3:18: `ld.global.u32` reads 4 bytes at once, at an address that is a multiple of 4; `[A + 2]` is not \
             one",
        ),
        (a, "ld.shared.u32(v, [A]);", "3:18: `ld.shared.u32` reads shared memory; `A` is global memory"),
        (
            a,
            "ld.u32(v, [A]);",
            "3:11: `ld.u32` reads at a generic address; `A` is global memory, which an instruction reaches by its \
             name only as `.global`",
        ),
        (
            c,
            "atom.global.add.u32(v, [C], 1);",
            "3:24: `atom.global.add.u32` reads and writes global memory; `C` is constant memory",
        ),
        (c, "st.const.u32([C], v);", "3:10: PTX's `st` has no `.const` that kernite knows"),
        // ptxas refuses a global address in 32 bits, and takes a constant one.
        (a, "mov.u32(v, A);", "3:12: `mov.u32` puts a global address in 32 bits; it is 64, `mov.u64(a, A)`"),
        (a, "mov.b32(v, A);", "3:12: `mov.b32` puts a global address in 32 bits; it is 64, `mov.u64(a, A)`"),
        (
            a,
            "st.global.u32([A], A);",
            "3:20: `st.global.u32` takes no memory as an operand: `mov.u64(a, A)` puts the address of `A` in `a`, \
             and `[A]` is what it holds",
        ),
        (
            a,
            "let b: u64; cvta.shared.u64(b, A);",
            "3:32: `cvta.shared.u64` takes the address of shared memory; `A` is global memory",
        ),
        (
            c,
            "let b: u64; cvta.global.u64(b, C);",
            "3:32: `cvta.global.u64` takes the address of global memory; `C` is constant memory",
        ),
        (a, "mov.u32(v, A.x);", "3:14: `A` is memory, `global<[u32; 4]>`; it is no `x`"),
        (a, "if A { ret(); }", "3:8: an instruction is guarded by a `pred`; this is `A`, global memory"),
        (
            a,
            "let a: global<[u32; 4]>;",
            "3:8: global and constant memory are the module's, which a static declares, as `static a: global<[u32; \
             4]>;` beside the kernels: a kernel declares shared memory, `shared<..>`, and local, `local<..>`",
        ),
        (
            a,
            "let a: constant<u32>;",
            "3:8: global and constant memory are the module's, which a static declares, as `static a: \
             constant<u32>;` beside the kernels: a kernel declares shared memory, `shared<..>`, and local, \
             `local<..>`",
        ),
    ];
    for (items, body, want) in cases {
        let src = format!("{items}\nunsafe extern \"ptx-kernel\" fn k(x: *mut u32) {{ let v: u32;\n{body}\n}}\n");
        assert_eq!(kernite::ptx_str(&src).unwrap_err(), want, "{body}");
    }
    // A variable of the kernel's own takes the name it is given, before the module's.
    let ptx = kernite::ptx_str(&format!("{a}\nunsafe extern \"ptx-kernel\" fn k() {{ let A: u32; mov.u32(A, 1); }}\n"));
    assert!(ptx.unwrap().contains("\tmov.u32 %A, 1;\n"));
}

/// A kernel whose blocks wait for each other, in its body and in a helper's, with variables
/// of the names the barrier's registers have, which stay its own.
const COOPERATIVE: &str = r#"
    /// `x[i] = x[n - 1 - i]` for each `i < n`, a thread each, of the `x` before: every thread
    /// reads its pair, the grid waits, then each writes. Then the grid waits again, in a
    /// helper, before the kernel ends.
    #[cooperative]
    pub unsafe extern "ptx-kernel" fn reverse(x: *mut u32, n: u32) {
        let (i, t): (u32, u32);
        mov.u32(t, ntid.x);
        mov.u32(i, ctaid.x);
        mul.lo.u32(i, i, t);
        mov.u32(t, tid.x);
        add.u32(i, i, t);
        let others: pred;
        setp.ge.u32(others, i, n);
        let at: u64;
        if !others {
            sub.u32(t, n, i);
            sub.u32(t, t, 1);
            mad.wide.u32(at, t, 4, x);
            ld.global.u32(t, [at]);
        }
        grid.sync();
        if !others {
            mad.wide.u32(at, i, 4, x);
            st.global.u32([at], t);
        }
        wait();
    }

    fn wait() {
        grid.sync();
    }
"#;

#[test]
fn a_cooperative_kernel_waits_for_its_grid_as_cuda_does() {
    let ptx = kernite::ptx_str(COOPERATIVE).unwrap();
    // A cooperative launch is from Pascal's GPUs on, and PTX 5.0.
    let (head, reverse) = ptx.split_once(".visible .entry ").unwrap();
    assert_eq!(head, "// Written by kernite.\n\n.version 5.0\n.target sm_60\n.address_size 64\n\n");
    // Each block's first thread adds to the count of the driver's barrier for the grid, the
    // first block's so that it turns over its top bit once all have, and waits for that.
    let sync = "\t{
\t.reg .pred %others, %wait;
\t.reg .b32 %t, %u, %add, %old;
\t.reg .b64 %barrier;
\tbar.sync 0;
\tmov.u32 %t, %tid.x;
\tmov.u32 %u, %tid.y;
\tor.b32 %t, %t, %u;
\tmov.u32 %u, %tid.z;
\tor.b32 %t, %t, %u;
\tsetp.ne.u32 %others, %t, 0;
\t@%others bra $Lgrid_done;
\tmov.u32 %t, %envreg1;
\tmov.u32 %u, %envreg2;
\tmov.b64 %barrier, {%u, %t};
\tsetp.eq.u64 %wait, %barrier, 0;
\t@%wait trap;
\tadd.u64 %barrier, %barrier, 4;
\tmov.u32 %t, %nctaid.x;
\tmov.u32 %u, %nctaid.y;
\tmul.lo.u32 %t, %t, %u;
\tmov.u32 %u, %nctaid.z;
\tmul.lo.u32 %t, %t, %u;
\tsub.u32 %add, 2147483649, %t;
\tmov.u32 %t, %ctaid.x;
\tmov.u32 %u, %ctaid.y;
\tor.b32 %t, %t, %u;
\tmov.u32 %u, %ctaid.z;
\tor.b32 %t, %t, %u;
\tsetp.eq.u32 %wait, %t, 0;
\tselp.b32 %add, %add, 1, %wait;
\tmembar.gl;
\tatom.add.u32 %old, [%barrier], %add;
$Lgrid_wait:
\tld.volatile.u32 %t, [%barrier];
\txor.b32 %t, %t, %old;
\tsetp.ge.s32 %wait, %t, 0;
\t@%wait bra $Lgrid_wait;
\tmembar.gl;
$Lgrid_done:
\tbar.sync 0;
\t}
";
    let want = format!(
        "reverse(
\t.param .u64 reverse_param_0,
\t.param .u32 reverse_param_1
)
{{
\t.reg .pred %others;
\t.reg .u32 %n, %i, %t;
\t.reg .u64 %x, %at;

\tld.param.u64 %x, [reverse_param_0];
\tcvta.to.global.u64 %x, %x;
\tld.param.u32 %n, [reverse_param_1];
\tmov.u32 %t, %ntid.x;
\tmov.u32 %i, %ctaid.x;
\tmul.lo.u32 %i, %i, %t;
\tmov.u32 %t, %tid.x;
\tadd.u32 %i, %i, %t;
\tsetp.ge.u32 %others, %i, %n;
\t@!%others sub.u32 %t, %n, %i;
\t@!%others sub.u32 %t, %t, 1;
\t@!%others mad.wide.u32 %at, %t, 4, %x;
\t@!%others ld.global.u32 %t, [%at];
{sync}\t@!%others mad.wide.u32 %at, %i, 4, %x;
\t@!%others st.global.u32 [%at], %t;
{sync}\tret;
}}
"
    );
    assert_eq!(reverse, want);
}

#[test]
fn grid_sync_is_where_every_thread_of_the_grid_comes_to_it() {
    let cases = [
        (
            "",
            "grid.sync();",
            "3:6: `grid.sync()` waits for every block of the grid, which runs all at once only if `k` is \
             `#[cooperative]`",
        ),
        // A helper's waits for the grid of the kernel it is inlined in.
        (
            "",
            "h();",
            "9:10: `grid.sync()` waits for every block of the grid, which runs all at once only if `k` is \
             `#[cooperative]`",
        ),
        ("#[cooperative]", "grid.sync(n);", "3:6: `grid.sync` takes no operands, not 1"),
        ("#[cooperative]", "grid.wait();", "3:6: kernite's barrier for the grid is `grid.sync()`; not `grid.wait`"),
        (
            "#[cooperative]",
            "grid.sync.all();",
            "3:11: kernite's barrier for the grid is `grid.sync()`; not `grid.sync.all`",
        ),
        (
            "#[cooperative]",
            "let p: pred; setp.eq.u32(p, n, 0); if !p { grid.sync(); }",
            "3:49: `grid.sync()` is for every thread of the grid, under no `if`: kernite writes it with no guard, and a \
             kernel that wants it under a condition the same in every thread jumps around it",
        ),
        // A function the kernels call may be called by one whose grid is not all running.
        (
            "#[cooperative]",
            "call(f);",
            "6:10: `grid.sync()` goes in a `#[cooperative]` kernel, or in a helper inlined there: a function the \
             kernels call may be called by a kernel that is not",
        ),
    ];
    for (attr, body, want) in cases {
        let src = format!(
            "{attr}\nunsafe extern \"ptx-kernel\" fn k(x: *mut u32, n: u32) {{\n{body}\n}}\n\
             extern \"ptx-func\" fn f() {{\n    grid.sync();\n}}\nfn h() {{\n    grid.sync();\n}}\n"
        );
        assert_eq!(kernite::ptx_str(&src).unwrap_err(), want, "{body}");
    }
}

#[test]
fn cooperative_is_said_of_a_kernel_alone() {
    let not = "`#[cooperative]` goes on a kernel whose blocks or threads wait for each other, `extern \"ptx-kernel\"` \
               or `extern \"sysv64\"`";
    let nothing = "1:3: `#[cooperative]` takes nothing: it says that a kernel's blocks or threads wait for each other";
    let cases = [
        (
            "#[cooperative] extern \"ptx-func\" fn f() {}",
            format!(
                "1:3: {not}; `f` is a function the kernels call, which runs as the kernel that calls it is launched"
            ),
        ),
        ("#[cooperative] fn h() {}", format!("1:3: {not}; `h` is a helper, inlined where it is called")),
        ("#[cooperative] static A: global<u32>;", format!("1:3: {not}")),
        ("#[cooperative] const A: u32 = 1;", format!("1:3: {not}")),
        (
            "#[cooperative] #[cooperative] unsafe extern \"ptx-kernel\" fn j() {}",
            "1:18: a kernel has one `#[cooperative]`".into(),
        ),
        ("#[cooperative(2)] unsafe extern \"ptx-kernel\" fn j() {}", nothing.into()),
        ("#[cooperative = 2] unsafe extern \"ptx-kernel\" fn j() {}", nothing.into()),
    ];
    for (items, want) in cases {
        let src = format!("{items}\nunsafe extern \"ptx-kernel\" fn k(x: *mut u32) {{}}");
        assert_eq!(kernite::ptx_str(&src).unwrap_err(), want, "{items}");
    }
    // Beside `#[instances(..)]`, either first, it makes each instance cooperative.
    for attrs in ["#[cooperative] #[instances(1, 2)]", "#[instances(1, 2)] #[cooperative]"] {
        let src = format!("{attrs}\nunsafe extern \"ptx-kernel\" fn k<const N: u32>() {{ grid.sync(); }}");
        assert_eq!(kernite::ptx_str(&src).unwrap().matches("\tatom.add.u32 %old, [%barrier], %add;\n").count(), 2);
    }
}

/// A kernel for each form of each directive: `.maxntid` and `.reqntid` of one, two and three
/// sizes, `.minnctapersm` beside either, `.maxnreg`, and the three beside one another in an order
/// that is not the ISA's. A size is a number known now, as `ROW` is.
const TUNED: &str = r#"
    const ROW: u32 = 16;

    /// Every thread writes its place in its block.
    fn mark(x: u64) {
        let t: u32;
        mov.u32(t, tid.x);
        st.global.u32([x], t);
    }

    pub unsafe extern "ptx-kernel" fn none(x: *mut u32) {
        mark(x);
    }

    #[maxntid(256)]
    pub unsafe extern "ptx-kernel" fn max_x(x: *mut u32) {
        mark(x);
    }

    #[maxntid(ROW, ROW)]
    pub unsafe extern "ptx-kernel" fn max_xy(x: *mut u32) {
        mark(x);
    }

    #[maxntid(8, 8, 4)]
    pub unsafe extern "ptx-kernel" fn max_xyz(x: *mut u32) {
        mark(x);
    }

    #[reqntid(256)]
    pub unsafe extern "ptx-kernel" fn req_x(x: *mut u32) {
        mark(x);
    }

    #[reqntid(ROW, ROW)]
    pub unsafe extern "ptx-kernel" fn req_xy(x: *mut u32) {
        mark(x);
    }

    #[reqntid(8, 8, 4)]
    pub unsafe extern "ptx-kernel" fn req_xyz(x: *mut u32) {
        mark(x);
    }

    #[maxntid(256)]
    #[minnctapersm(2)]
    pub unsafe extern "ptx-kernel" fn max_blocks(x: *mut u32) {
        mark(x);
    }

    #[reqntid(ROW * 8, 2)]
    #[minnctapersm(4)]
    pub unsafe extern "ptx-kernel" fn req_blocks(x: *mut u32) {
        mark(x);
    }

    #[maxnreg(32)]
    pub unsafe extern "ptx-kernel" fn regs(x: *mut u32) {
        mark(x);
    }

    #[maxnreg(48)]
    #[minnctapersm(4)]
    #[reqntid(8, 8, 4)]
    pub unsafe extern "ptx-kernel" fn all(x: *mut u32) {
        mark(x);
    }
"#;

/// The directives of the `.entry` `name` of `ptx`: its lines between its parameters and its body.
fn directives_of(ptx: &str, name: &str) -> Vec<String> {
    let entry = ptx.split_once(&format!(".visible .entry {name}(")).expect("the module has the kernel").1;
    let head = entry.split_once("\n{\n").expect("a kernel has a body").0;
    head.lines().skip_while(|l| *l != ")").skip(1).map(String::from).collect()
}

#[test]
fn a_kernel_says_its_directives_between_its_parameters_and_its_body() {
    let ptx = kernite::ptx_str(TUNED).unwrap();
    // `.maxntid` and `.maxnreg` came with PTX 1.3, `.minnctapersm` with 2.0 and `.reqntid` with 2.1, and
    // every module kernite writes is of `.version 3.0` at least, `Needs::FLOOR`, so a directive asks nothing
    // of the module.
    assert!(ptx.starts_with("// Written by kernite.\n\n.version 3.0\n.target sm_30\n.address_size 64\n\n"), "{ptx}");
    let all: [(&str, &[&str]); 11] = [
        ("none", &[]),
        ("max_x", &[".maxntid 256"]),
        ("max_xy", &[".maxntid 16, 16"]),
        ("max_xyz", &[".maxntid 8, 8, 4"]),
        ("req_x", &[".reqntid 256"]),
        ("req_xy", &[".reqntid 16, 16"]),
        ("req_xyz", &[".reqntid 8, 8, 4"]),
        ("max_blocks", &[".maxntid 256", ".minnctapersm 2"]),
        ("req_blocks", &[".reqntid 128, 2", ".minnctapersm 4"]),
        ("regs", &[".maxnreg 32"]),
        // In the order they are written.
        ("all", &[".maxnreg 48", ".minnctapersm 4", ".reqntid 8, 8, 4"]),
    ];
    for (name, lines) in all {
        assert_eq!(directives_of(&ptx, name), lines, "{name}");
    }
    // As `nvcc` prints them: after the `)` of the parameters, one to a line, before the `{`.
    let want =
        ".visible .entry max_blocks(\n\t.param .u64 max_blocks_param_0\n)\n.maxntid 256\n.minnctapersm 2\n{\n\t.reg ";
    assert!(ptx.contains(want), "{ptx}");
}

/// A module of a kernel for each of `attrs`, `k0`, `k1` and so on, that does nothing.
fn tuned(attrs: &[&str]) -> String {
    let kernel = |(i, attr): (usize, &&str)| format!("{attr}\npub unsafe extern \"ptx-kernel\" fn k{i}() {{}}\n");
    attrs.iter().enumerate().map(kernel).collect()
}

/// Each directive at the edge of what kernite takes, as a kernel of its own.
const EDGES: &[&str] = &[
    "#[maxntid(1)]",
    "#[maxntid(1024)]",
    "#[maxntid(1024, 1, 1)]",
    "#[maxntid(1, 1024)]",
    "#[maxntid(1, 1, 64)]",
    "#[maxntid(32, 32)]",
    "#[maxntid(4, 4, 64)]",
    "#[reqntid(1)]",
    "#[reqntid(1024)]",
    "#[reqntid(1024, 1, 1)]",
    "#[reqntid(1, 1024)]",
    "#[reqntid(1, 1, 64)]",
    "#[reqntid(32, 32)]",
    "#[reqntid(4, 4, 64)]",
    "#[maxntid(1024)] #[minnctapersm(1)]",
    "#[maxntid(64)] #[minnctapersm(16)]",
    "#[reqntid(1024)] #[minnctapersm(1)]",
    "#[reqntid(8, 8)] #[minnctapersm(16)]",
    // `ptxas` counts the warps of the blocks, a warp being 32 threads and the last of a block maybe fewer:
    // 10 blocks of 96 threads make 30 warps, 16 of 33 make 32, and 8 of 99 make 32, which Turing's SM holds.
    "#[maxntid(96)] #[minnctapersm(10)]",
    "#[maxntid(33)] #[minnctapersm(16)]",
    "#[reqntid(33, 3)] #[minnctapersm(8)]",
    "#[maxnreg(24)]",
    "#[maxnreg(255)]",
    "#[maxntid(1024)] #[maxnreg(64)]",
    "#[reqntid(512)] #[maxnreg(128)]",
    "#[maxntid(16, 16)] #[maxnreg(255)]",
    "#[reqntid(32, 32)] #[minnctapersm(1)] #[maxnreg(64)]",
    // A launch counts the registers of a thread by eights, and the warps of a block by fours: 255 registers
    // for 8 warps are 256 * 32 * 8 = 65536, 168 for 9 warps, counted as 12, 168 * 32 * 12 = 64512, 64 for 32
    // warps, 64 * 32 * 32 = 65536, and 255 for 2 warps, counted as 4, 256 * 32 * 4 = 32768.
    "#[maxntid(256)] #[maxnreg(255)]",
    "#[reqntid(256)] #[maxnreg(255)]",
    "#[maxntid(288)] #[maxnreg(168)]",
    "#[maxntid(384)] #[maxnreg(168)]",
    "#[maxntid(1000)] #[maxnreg(64)]",
    "#[reqntid(33)] #[maxnreg(255)]",
];

/// What `ptxas` makes of the module `text` for the GPU `gpu`, with `flags`, named `name` in the
/// directory of this test.
fn assembled(ptxas: &Path, text: &str, gpu: &str, name: &str, flags: &[&str]) -> std::process::Output {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    let (src, out) = (dir.join(format!("{name}.ptx")), dir.join(format!("{name}.cubin")));
    std::fs::write(&src, text).unwrap();
    Command::new(ptxas).args(["--gpu-name", gpu]).args(flags).arg(&src).arg("-o").arg(&out).output().unwrap()
}

#[test]
fn ptxas_takes_each_directive_kernite_writes_without_a_word() {
    let Some(ptxas) = ptxas() else {
        eprintln!("no ptxas here: skipped; set PTXAS to run it");
        return;
    };
    // Turing's SM holds the fewest blocks and warps of the GPUs kernite writes for, and a
    // GeForce RTX 3050 is Ampere.
    for (name, text) in
        [("tuned", kernite::ptx_str(TUNED).unwrap()), ("edges", kernite::ptx_str(&tuned(EDGES)).unwrap())]
    {
        for gpu in ["sm_75", "sm_86"] {
            let run = assembled(&ptxas, &text, gpu, &format!("{name}_{gpu}"), &["--warning-as-error"]);
            assert!(run.status.success(), "{name} for {gpu}: {}", String::from_utf8_lossy(&run.stderr));
        }
    }
}

/// A kernel that loads the 64 floats `v[k]` at `x[tid.x]`, 1024 apart, and writes to `y[tid.x]` the sum of
/// their products in pairs, `v[k] * v[63 - k]`. It loads them all before it uses the first: without a
/// limit, `ptxas` gives it more than 32 registers.
fn heavy(attrs: &str) -> String {
    let body = r#"
    pub unsafe extern "ptx-kernel" fn heavy(x: *const f32, y: *mut f32) {
        let (i, at): (u32, u64);
        mov.u32(i, tid.x);
        mad.wide.u32(at, i, 4, x);
        let v: [f32; 64];
        for k in 0..64 {
            ld.global.f32(v[k], [at + 4096 * k]);
        }
        let s: f32;
        mov.f32(s, 0);
        for k in 0..64 {
            fma.rn.f32(s, v[k], v[63 - k], s);
        }
        mad.wide.u32(at, i, 4, y);
        st.global.f32([at], s);
    }
    "#;
    kernite::ptx_str(&format!("{attrs}{body}")).unwrap()
}

/// The registers `ptxas -v` says the kernel of the module `text` uses, for the RTX 3050's `sm_86`.
fn used_registers(ptxas: &Path, text: &str, name: &str) -> u32 {
    let run = assembled(ptxas, text, "sm_86", name, &["-v", "--warning-as-error"]);
    let said = String::from_utf8_lossy(&run.stderr).into_owned();
    assert!(run.status.success(), "{name}: {said}");
    let after = said.split_once("Used ").unwrap_or_else(|| panic!("{name}: {said}")).1;
    after.split_once(" registers").unwrap().0.parse().unwrap()
}

#[test]
fn a_kernel_with_maxnreg_uses_no_more_registers_than_it_says() {
    let Some(ptxas) = ptxas() else {
        eprintln!("no ptxas here: skipped; set PTXAS to run it");
        return;
    };
    let used = |attrs: &str, name: &str| used_registers(&ptxas, &heavy(attrs), name);
    let free = used("", "free");
    assert!(free > 32, "the kernel needs {free} registers, and the test means nothing if that is 32 or fewer");
    for most in [24, 32, 48] {
        let held = used(&format!("#[maxnreg({most})]"), &format!("held_{most}"));
        assert!(held <= most, "`#[maxnreg({most})]` and {held} registers, not {free}");
    }
}

#[test]
fn ptxas_lowers_maxnreg_for_the_blocks_an_sm_holds_but_not_for_the_threads_of_a_block() {
    let Some(ptxas) = ptxas() else {
        eprintln!("no ptxas here: skipped; set PTXAS to run it");
        return;
    };
    let used = |attrs: &str, name: &str| used_registers(&ptxas, &heavy(attrs), name);
    let free = used("", "free_again");
    assert!(free > 64, "the kernel needs {free} registers, and the test means nothing if that is 64 or fewer");
    // 1024 threads have 64 registers each, and `.maxntid 1024` alone gives the kernel no more.
    assert_eq!(used("#[maxntid(1024)]", "threads"), 64);
    // Four blocks of 256 threads are 1024 threads an SM, and `.minnctapersm 4` gives it 64 as well,
    // though `.maxnreg` says 128.
    assert_eq!(used("#[maxntid(256)] #[minnctapersm(4)] #[maxnreg(128)]", "blocks"), 64);
    // `.maxnreg 128` beside `.maxntid 1024` is kept, and kernite refuses it: no block of 1024
    // threads has 128 registers each. The module is by hand, for kernite does not write it.
    let text = heavy("#[maxntid(1024)]").replace(".maxntid 1024\n", ".maxntid 1024\n.maxnreg 128\n");
    assert!(text.contains(".maxnreg 128"), "{text}");
    assert_eq!(used_registers(&ptxas, &text, "kept"), free);
}

#[test]
fn ptxas_keeps_the_last_of_a_directive_said_twice_and_says_nothing_of_the_first() {
    let Some(ptxas) = ptxas() else {
        eprintln!("no ptxas here: skipped; set PTXAS to run it");
        return;
    };
    let once = |most: u32| used_registers(&ptxas, &heavy(&format!("#[maxnreg({most})]")), &format!("once_{most}"));
    // The modules are by hand, for kernite refuses a directive twice, at the second.
    let twice = |first: u32, last: u32| {
        let text = heavy(&format!("#[maxnreg({last})]"));
        let text = text.replace(&format!(".maxnreg {last}\n"), &format!(".maxnreg {first}\n.maxnreg {last}\n"));
        assert!(text.contains(&format!(".maxnreg {first}\n.maxnreg {last}\n")), "{text}");
        used_registers(&ptxas, &text, &format!("twice_{first}_{last}"))
    };
    let (low, high) = (once(32), once(48));
    assert!(low <= 32 && high > 32, "`#[maxnreg(32)]` and {low} registers, `#[maxnreg(48)]` and {high}");
    assert_eq!(twice(48, 32), low);
    assert_eq!(twice(32, 48), high);
}

#[test]
fn what_ptx_forbids_and_ptxas_would_cut_or_ignore_is_refused_at_the_directive() {
    let size = |name: &str| format!("`#[{name}(..)]` gives one size, two or three: the threads along x, then y and z");
    let few = "threads are too few: a block has 1 at least along each axis";
    let one = |name: &str, what: &str| format!("`#[{name}(..)]` gives one number: {what}");
    let blocks = "the blocks an SM should hold at once, at least";
    let registers = "the most registers a thread may use";
    let both = "a kernel has `#[maxntid(..)]` or `#[reqntid(..)]`, not both: its blocks have at most these threads, or \
                exactly these";
    // `n` blocks of `threads` threads make `all` warps, `w` to a block.
    let warps = |n: u32, threads: u32, all: u32, w: u32| {
        format!(
            "{n} blocks of {threads} threads make {all} warps, {w} to a block: an SM holds 32 on some GPUs, and \
             `ptxas` counts the warps of the blocks, not their threads, and ignores the number on a GPU whose SM \
             holds fewer"
        )
    };
    // `n` registers for each of `threads` threads make `all` in a launch: `r` registers by eights, 32 threads to a
    // warp, and `w` warps by fours.
    let launch = |n: u32, threads: u32, all: u32, r: u32, w: u32| {
        format!(
            "{n} registers for each of {threads} threads make {all} in a launch, which counts a thread's registers by \
             eights and a block's warps by fours, {r} * 32 * {w}: a block has 65536 at most, and `ptxas` does not \
             lower `#[maxnreg(..)]` to fit the threads of a block, so a kernel that used them all could not launch"
        )
    };
    let cases = [
        // A size of 0, which `ptxas` refuses, at the size.
        ("#[maxntid(0)]", "1:11", format!("0 {few}")),
        ("#[reqntid(8, 0)]", "1:14", format!("0 {few}")),
        ("#[maxntid(-1)]", "1:11", format!("-1 {few}")),
        // A size past what a block has along an axis, and the threads of a block in all: `ptxas`
        // ignores `.maxntid 2048`, takes `.reqntid 32, 33`, and dies of `SIGFPE` on 4294967295.
        ("#[maxntid(2048)]", "1:11", "2048 threads are too many: a block has 1024 at most along x".into()),
        ("#[maxntid(1, 2048)]", "1:14", "2048 threads are too many: a block has 1024 at most along y".into()),
        ("#[reqntid(1, 1, 65)]", "1:17", "65 threads are too many: a block has 64 at most along z".into()),
        ("#[maxntid(4294967295)]", "1:11", "4294967295 threads are too many: a block has 1024 at most along x".into()),
        ("#[reqntid(32, 33)]", "1:3", "these sizes make 1056 threads: a block has 1024 at most in all".into()),
        ("#[maxntid(1, 2, 3, 4)]", "1:20", size("maxntid")),
        ("#[reqntid()]", "1:3", size("reqntid")),
        // `.maxntid` with `.reqntid`: `ptxas` refuses a module that has both. The second is at fault.
        ("#[maxntid(256)] #[reqntid(256)]", "1:19", both.into()),
        ("#[reqntid(8, 8)] #[maxntid(8, 8)]", "1:20", both.into()),
        // `.minnctapersm` alone, which `ptxas` ignores, and more blocks than the GPU's SM holds.
        (
            "#[minnctapersm(2)]",
            "1:3",
            "`#[minnctapersm(..)]` needs `#[maxntid(..)]` or `#[reqntid(..)]`: `ptxas` ignores it where it does not \
             know the threads of a block"
                .into(),
        ),
        ("#[maxntid(256)] #[minnctapersm(0)]", "1:32", "0 blocks are too few: an SM holds 1 at least".into()),
        (
            "#[maxntid(256)] #[minnctapersm(17)]",
            "1:32",
            "17 blocks are too many: an SM holds 16 on some GPUs, and `ptxas` ignores the number on a GPU whose SM \
             holds fewer"
                .into(),
        ),
        ("#[maxntid(1024)] #[minnctapersm(2)]", "1:33", warps(2, 1024, 64, 32)),
        ("#[reqntid(512)] #[minnctapersm(3)]", "1:32", warps(3, 512, 48, 16)),
        // The warps of the blocks, not their threads: 970 threads are fewer than Turing's SM holds, but 40
        // warps are more than the 32 it holds, and `ptxas` ignores the number there. At the number.
        ("#[maxntid(97)] #[minnctapersm(10)]", "1:31", warps(10, 97, 40, 4)),
        ("#[maxntid(330)] #[minnctapersm(3)]", "1:32", warps(3, 330, 33, 11)),
        ("#[reqntid(33, 3)] #[minnctapersm(9)]", "1:34", warps(9, 99, 36, 4)),
        ("#[maxntid(256)] #[minnctapersm()]", "1:19", one("minnctapersm", blocks)),
        ("#[maxntid(256)] #[minnctapersm(1, 2)]", "1:35", one("minnctapersm", blocks)),
        // `.maxnreg` below 24, which `ptxas` raises to 24, and past 255, which it ignores.
        (
            "#[maxnreg(0)]",
            "1:11",
            "0 registers are too few: `ptxas` gives a thread 24 at least, and raises a smaller number to it".into(),
        ),
        (
            "#[maxnreg(16)]",
            "1:11",
            "16 registers are too few: `ptxas` gives a thread 24 at least, and raises a smaller number to it".into(),
        ),
        (
            "#[maxnreg(300)]",
            "1:11",
            "300 registers are too many: a thread has 255 at most, and `ptxas` ignores a bigger number".into(),
        ),
        ("#[maxnreg()]", "1:3", one("maxnreg", registers)),
        ("#[maxnreg(32, 64)]", "1:15", one("maxnreg", registers)),
        // The registers of a block's threads, which `ptxas` does not lower `.maxnreg` for.
        ("#[maxntid(1024)] #[maxnreg(128)]", "1:28", launch(128, 1024, 131072, 128, 32)),
        ("#[maxnreg(255)] #[reqntid(1024)]", "1:11", launch(255, 1024, 262144, 256, 32)),
        ("#[reqntid(32, 32)] #[maxnreg(65)]", "1:30", launch(65, 1024, 73728, 72, 32)),
        ("#[maxntid(8, 8, 8)] #[maxnreg(255)]", "1:31", launch(255, 512, 131072, 256, 16)),
        // A launch counts by eights and by fours, so these cannot launch: one thread past 8 warps is 9, counted as
        // 12, and the registers of a thread, 170 and 65, as 176 and 72. The edges are in `EDGES`.
        ("#[maxntid(257)] #[maxnreg(255)]", "1:27", launch(255, 257, 98304, 256, 12)),
        ("#[reqntid(257)] #[maxnreg(255)]", "1:27", launch(255, 257, 98304, 256, 12)),
        ("#[maxntid(288)] #[maxnreg(176)]", "1:27", launch(176, 288, 67584, 176, 12)),
        ("#[maxntid(384)] #[maxnreg(170)]", "1:27", launch(170, 384, 67584, 176, 12)),
        ("#[maxntid(1000)] #[maxnreg(65)]", "1:28", launch(65, 1000, 73728, 72, 32)),
        // The attribute is not one: it names no number, or twice, or a number that is not whole.
        (
            "#[maxntid]",
            "1:3",
            "`#[maxntid(..)]` gives the most threads a block has, along x, then y and z, as in `#[maxntid(16, 16)]`"
                .into(),
        ),
        (
            "#[reqntid = 256]",
            "1:3",
            "`#[reqntid(..)]` gives the threads a block has, along x, then y and z, as in `#[reqntid(8, 8, 4)]`".into(),
        ),
        (
            "#[minnctapersm]",
            "1:3",
            "`#[minnctapersm(..)]` gives the blocks an SM should hold at once, at least, as in `#[minnctapersm(2)]`"
                .into(),
        ),
        (
            "#[maxnreg]",
            "1:3",
            "`#[maxnreg(..)]` gives the most registers a thread may use, as in `#[maxnreg(32)]`".into(),
        ),
        ("#[maxntid(1)] #[maxntid(1)]", "1:17", "a kernel has one `#[maxntid(..)]`".into()),
        ("#[maxnreg(32)] #[maxnreg(32)]", "1:18", "a kernel has one `#[maxnreg(..)]`".into()),
        ("#[maxntid(1.5)]", "1:11", "`#[maxntid(..)]` gives whole numbers; this is an f32".into()),
    ];
    for (attrs, at, want) in cases {
        let src = format!("{attrs}\npub unsafe extern \"ptx-kernel\" fn k() {{}}\n");
        assert_eq!(kernite::ptx_str(&src).unwrap_err(), format!("{at}: {want}"), "{attrs}");
    }
    // The other side of each edge is taken.
    kernite::ptx_str(&tuned(EDGES)).unwrap();
}

#[test]
fn a_directive_is_said_of_a_gpu_kernel_alone() {
    let not = |name: &str| format!("`#[{name}(..)]` goes on a GPU kernel, `extern \"ptx-kernel\"`");
    let cooperative = "`#[cooperative]` goes on a kernel whose blocks or threads wait for each other, \
                       `extern \"ptx-kernel\"` or `extern \"sysv64\"`";
    let cases = [
        (
            "#[maxntid(256)] extern \"ptx-func\" fn f() {}",
            format!(
                "1:3: {}; `f` is a function the kernels call, which runs as the kernel that calls it is launched",
                not("maxntid")
            ),
        ),
        // Nothing calls `h`, and a helper that is not called is wrong all the same.
        ("#[maxnreg(32)] fn h() {}", format!("1:3: {}; `h` is a helper, inlined where it is called", not("maxnreg"))),
        ("#[reqntid(8, 8)] static A: global<u32>;", format!("1:3: {}", not("reqntid"))),
        ("#[minnctapersm(2)] const A: u32 = 1;", format!("1:3: {}", not("minnctapersm"))),
        (
            "#[maxntid(256)] unsafe extern \"sysv64\" fn s() {}",
            format!("1:3: {}; `s` is a kernel for x86-64, which the process runs", not("maxntid")),
        ),
        // The first of its directives is the one said, and `#[cooperative]` is said before them.
        (
            "#[maxnreg(32)] #[maxntid(8)] fn h() {}",
            format!("1:3: {}; `h` is a helper, inlined where it is called", not("maxnreg")),
        ),
        (
            "#[maxnreg(32)] #[cooperative] fn h() {}",
            format!("1:18: {cooperative}; `h` is a helper, inlined where it is called"),
        ),
    ];
    for (items, want) in cases {
        let src = format!("{items}\npub unsafe extern \"ptx-kernel\" fn k() {{}}\n");
        assert_eq!(kernite::ptx_str(&src).unwrap_err(), want, "{items}");
    }
    // Where the function is called, it is said once, and for the function, not for the kernel.
    let src = "#[maxnreg(32)]\nfn h() {\n    bar.sync(0);\n}\npub unsafe extern \"ptx-kernel\" fn j() {\n    h();\n}\n\
               pub unsafe extern \"ptx-kernel\" fn k() {\n    h();\n}\n";
    let want = format!("1:3: {}; `h` is a helper, inlined where it is called", not("maxnreg"));
    assert_eq!(kernite::ptx_str(src).unwrap_err(), want);
}

/// `.maxnctapersm` is gone from PTX 2.1 on, and `ptxas` refuses it: the name is what is wrong with
/// `#[maxnctapersm(..)]`, in whatever form it is written and on whatever it is. The error says so, at the
/// name, and not that the directive goes on a GPU kernel, where it would be refused too.
#[test]
fn maxnctapersm_is_refused_by_its_name_in_every_form_and_every_place() {
    let want = "1:3: PTX has no `.maxnctapersm` from 2.1 on; `#[minnctapersm(..)]` gives the blocks an SM should hold \
                at once, at least";
    for items in [
        // On a GPU kernel, with the blocks it asks for, with none, and as `= 2`.
        "#[maxnctapersm(2)]",
        "#[maxnctapersm]",
        "#[maxnctapersm = 2]",
        // On what is no GPU kernel: a helper, a function the kernels call, a static and a const.
        "#[maxnctapersm(2)] fn h() {}",
        "#[maxnctapersm(2)] extern \"ptx-func\" fn f() {}",
        "#[maxnctapersm(2)] static A: global<u32>;",
        "#[maxnctapersm(2)] const A: u32 = 1;",
    ] {
        let src = format!("{items}\npub unsafe extern \"ptx-kernel\" fn k() {{}}\n");
        assert_eq!(kernite::ptx_str(&src).unwrap_err(), want, "{items}");
    }
}

/// A kernel of a tile `N` along y and 8 along x, compiled for 16 and for 32.
const GENERIC: &str = r#"
    const ROW: u32 = 8;

    #[instances(16, 32)]
    #[maxntid(ROW, N)]
    #[minnctapersm(N / ROW)]
    pub unsafe extern "ptx-kernel" fn tile<const N: u32>() {}
"#;

#[test]
fn a_directive_works_out_the_consts_and_the_instance_of_its_kernel() {
    let ptx = kernite::ptx_str(GENERIC).unwrap();
    assert_eq!(directives_of(&ptx, "_Z4tileILj16EE"), [".maxntid 8, 16", ".minnctapersm 2"]);
    assert_eq!(directives_of(&ptx, "_Z4tileILj32EE"), [".maxntid 8, 32", ".minnctapersm 4"]);
    // What is wrong with an instance is said of that instance, at the value.
    for (instances, want) in [
        ("16, 2048", "2:11: in `tile::<2048>`: 2048 threads are too many: a block has 1024 at most along x"),
        ("0, 16", "2:11: in `tile::<0>`: 0 threads are too few: a block has 1 at least along each axis"),
    ] {
        let src = format!(
            "#[instances({instances})]\n#[maxntid(N)]\npub unsafe extern \"ptx-kernel\" fn tile<const N: u32>() {{}}\n"
        );
        assert_eq!(kernite::ptx_str(&src).unwrap_err(), want, "{instances}");
    }
}

/// A kernel `k` written by hand, with `lines` for its directives.
fn written_with(lines: &str) -> String {
    format!(".version 3.0\n.target sm_30\n.address_size 64\n\n.visible .entry k()\n{lines}\n{{\n\tret;\n}}\n")
}

#[test]
fn ptxas_refuses_cuts_or_ignores_what_kernite_refuses_of_the_directives() {
    let Some(ptxas) = ptxas() else {
        eprintln!("no ptxas here: skipped; set PTXAS to run it");
        return;
    };
    // What `ptxas` says of a kernel with `lines`, for the GPU `gpu`.
    let said = |gpu: &str, lines: &str, name: usize| -> (bool, String) {
        let run = assembled(&ptxas, &written_with(lines), gpu, &format!("written_with_{name}"), &[]);
        (run.status.success(), String::from_utf8_lossy(&run.stderr).into_owned())
    };
    let cases = [
        // Refused, which kernite refuses at the directive in its words.
        (".maxntid 0", false, "Positive non-zero value expected for maxntid"),
        (".reqntid 256, 0", false, "Positive non-zero value expected for reqntid"),
        (".maxntid 32\n.minnctapersm 0", false, "Positive non-zero value expected for minnctapersm"),
        (".maxnreg 0", false, "Positive non-zero value expected for maxnreg"),
        (".maxntid 256\n.reqntid 256", false, "Conflicting directives: .maxntid and .reqntid cannot both be specified"),
        (".maxnctapersm 2", false, "'.maxnctapersm directive' not supported as of PTX version 2.1"),
        // Taken with a warning that it changes or ignores the number.
        (
            ".maxntid 2048",
            true,
            "Value of threads per SM for entry k is out of range. .minnctapersm and .maxntid will be ignored",
        ),
        (".maxnreg 300", true, "Too many maxnreg specified for entry k, will be ignored"),
        (".maxnreg 16", true, "adjusting per thread register count of 16 to lower bound of 24"),
        (".minnctapersm 2", true, ".minnctapersm is ignored when neither .maxntid nor .reqntid is specified"),
        (
            ".maxntid 32\n.minnctapersm 17",
            true,
            "Value of minnctapersm for entry k is out of range. minnctapersm will be ignored",
        ),
        (
            ".maxntid 1024\n.minnctapersm 2",
            true,
            "Value of threads per SM for entry k is out of range. .minnctapersm will be ignored",
        ),
        // Taken without a word, though no launch has these threads: a block has 1024 at most, and
        // 64 along z.
        (".reqntid 32, 33", true, ""),
        (".reqntid 1, 1, 65", true, ""),
    ];
    // For the RTX 3050's `sm_86`.
    let before = cases.len();
    for (i, (lines, takes, word)) in cases.into_iter().enumerate() {
        let (took, stderr) = said("sm_86", lines, i);
        assert_eq!(took, takes, "{lines}: {stderr}");
        assert!(stderr.contains(word) && (!word.is_empty() || stderr.is_empty()), "{lines}: {stderr}");
    }
    // An SM holds the blocks and the warps its GPU has, and Turing's, `sm_75`, holds fewer than Ampere's
    // `sm_86`: 32 warps against 48. `ptxas` counts the warps of the blocks, 32 threads each, and not their
    // threads: 10 blocks of 96 threads are 30 warps and are taken, and 10 of 97 are 40 warps, though they are 970
    // threads and the SM holds 1024, so 3 blocks of 330, 33 warps, are ignored as well.
    let ignored = "Value of threads per SM for entry k is out of range. .minnctapersm will be ignored";
    let turing = [
        (".maxntid 96\n.minnctapersm 10", true),
        (".maxntid 97\n.minnctapersm 10", false),
        (".reqntid 100\n.minnctapersm 10", false),
        (".maxntid 33\n.minnctapersm 16", true),
        (".maxntid 330\n.minnctapersm 3", false),
        (".maxntid 320\n.minnctapersm 3", true),
        (".maxntid 400\n.minnctapersm 3", false),
        (".maxntid 400\n.minnctapersm 2", true),
        (".maxntid 33, 3\n.minnctapersm 8", true),
        (".maxntid 33, 3\n.minnctapersm 9", false),
        (".reqntid 7, 7, 2\n.minnctapersm 9", false),
        (".maxntid 1024\n.minnctapersm 1", true),
        (".maxntid 513\n.minnctapersm 2", false),
    ];
    for (i, (lines, takes)) in turing.into_iter().enumerate() {
        let (took, stderr) = said("sm_75", lines, before + 2 * i);
        let word = if takes { "" } else { ignored };
        assert!(took && stderr.contains(word) && (!takes || stderr.is_empty()), "{lines}: {stderr}");
        // Ampere's holds 48 warps, more than any of these makes.
        assert_eq!(said("sm_86", lines, before + 2 * i + 1), (true, String::new()), "{lines}");
    }
}
