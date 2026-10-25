//! Kernels in the language, compiled to PTX: what they come to, that `ptxas` takes them, and
//! what the errors say; and every form of every instruction kgen knows, which `ptxas` takes
//! at the version and for the GPU kgen says it needs, and not at one older, and the names of
//! them kgen refuses, which `ptxas` refuses too.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use kgen::lang::{Mem as Address, Operand as Given, Target};
use kgen_ptx::forms::{self, Bound, Check, Count, Form, Imm, Role, Sel, Space, Spec};
use kgen_ptx::lang::{Kernel, Var};
use kgen_ptx::{Entry, Kind, Needs, Ty};

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
    let ptx = kgen::ptx_str(KERNELS).unwrap();
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
fn ptxas_takes_what_kgen_writes() {
    let Some(ptxas) = ptxas() else {
        eprintln!("no ptxas here: skipped; set PTXAS to run it");
        return;
    };
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    for (name, kernels) in [("kernels", KERNELS), ("shared", SHARED), ("tables", TABLES), ("surfaces", SURFACES)] {
        let (src, out) = (dir.join(format!("{name}.ptx")), dir.join(format!("{name}.cubin")));
        std::fs::write(&src, kgen::ptx_str(kernels).unwrap()).unwrap();
        let run = Command::new(&ptxas)
            .args(["--gpu-name", "sm_75", "--warning-as-error"])
            .arg(&src)
            .arg("-o")
            .arg(&out)
            .output()
            .unwrap();
        assert!(run.status.success(), "{name}: {}", String::from_utf8_lossy(&run.stderr));
    }
}

/// The kernel `k` around `body`, compiled, or the error.
fn compile(params: &str, body: &str) -> Result<String, String> {
    kgen::ptx_str(&format!("unsafe extern \"ptx-kernel\" fn k({params}) {{\n{body}\n}}\n"))
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
        ("let v: f32; ld.glob.f32(v, [x]);", "2:21: PTX's `ld` has no `.glob` that kgen knows"),
        ("let i: u32; add.u32(i, tid.x, 1);", "2:28: `add.u32` reads no special register; `mov.u32` does"),
        ("let i: u32; mov.u32(i, tod.x);", "2:28: `tod.x` is not defined"),
        ("let i: u32; if n { mov.u32(i, 1); }", "2:24: an instruction is guarded by a `pred`; this is a `u32`"),
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
        ("vfrob.f32(x);", "2:7: PTX has no instruction `vfrob` that kgen knows"),
    ];
    for (body, want) in cases {
        assert_eq!(compile("x: *mut f32, n: u32", body).unwrap_err(), want, "{body}");
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
        ("let i: u8; add.u8(i, i, i);", "2:16: PTX's `add` has no `.u8` that kgen knows"),
        (
            "let (a, b, c, e): (f64, f64, f64, f64); ld.global.v4.f64((a, b, c, e), [x]);",
            "2:54: `ld.global.v4.f64` moves 256 bits at once; a vector of PTX's holds 128 at most, as `.v4.b32` or \
             `.v2.b64`",
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
        ("elect.sync(n);", "2:7: PTX's `elect` is for sm_90 and newer, and kgen writes PTX for sm_20 to sm_86"),
    ];
    for (body, want) in cases {
        assert_eq!(compile("x: *mut f32, n: u32, d: f64", body).unwrap_err(), want, "{body}");
    }
}

#[test]
fn a_wrong_operand_is_named_by_what_ptx_lets_it_be() {
    let cases = [
        ("let i: u32; add.u32(i, n);", "2:17: `add.u32` takes 3 operands, not 2"),
        ("add.u32(_, n, n);", "2:9: `add.u32` has no result here that it may leave unwritten, `_`"),
        ("let i: u32; div.u32(i, !n, n);", "2:24: `div.u32` takes no operand inverted"),
        ("bar.sync(16);", "2:10: `bar.sync` takes 0 to 15 here; 16 is not one of them"),
        ("bar.sync(0, 33);", "2:13: `bar.sync` takes a multiple of 32 here, and more than 0; 33 is not one"),
        ("bar.arrive(0);", "2:5: `bar.arrive` takes 2 operands, not 1"),
        (
            "let q: b64; createpolicy.fractional.L2__evict_last.b64(q, 1.5);",
            "2:59: `createpolicy.fractional.L2::evict_last.b64` takes a fraction here, more than 0 and at most 1; 1.5 \
             is not one",
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
            "2:34: `clusterid.x` is a special register of PTX's that kgen does not read: it needs sm_90, and kgen \
             writes PTX for sm_20 to sm_86",
        ),
        (
            "let d: s32; vmad.s32.s32.s32(d, -n, n, -n);",
            "2:40: `vmad.s32.s32.s32` negates `a * b`, as `-a` or `-b` does, or `c`; not both",
        ),
        (
            "let d: u32; vadd.u32.u32.u32(d, n, n, n);",
            "2:30: `vadd.u32.u32.u32` writes a byte or a half of a variable here, `.b0` to `.b3`, `.h0` or `.h1`, \
             and the rest from its last operand; this is a whole `u32` variable",
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
            "2:12: `brx.idx` wants labels taken together, as `(a, b)`, for a variable to pick one; this is a label",
        ),
        (
            "brx.idx(n, (a, n)); label(a);",
            "2:12: `brx.idx` jumps to one of labels taken together; this holds a `u32` variable",
        ),
        ("brx.idx(1, (a,)); label(a);", "2:9: `brx.idx` wants a `u32` variable; this is the integer 1"),
        (
            "let w: u64; mov.u64(w, 0); brx.idx(w, (a,)); label(a);",
            "2:36: `brx.idx` wants a `u32` variable; this is a `u64`",
        ),
        (
            "let i: u32; mov.u32(i, perctamemorysize);",
            "2:24: `perctamemorysize` is a special register of PTX's that kgen does not read: a kernel has it only \
             with the directive `.minperctamemory`, which kgen does not write",
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
            "2:36: `suld.b.a1d.b32.trap` wants 2 operands taken together, a `u32` variable or a value and then a \
             `s32` variable or a value; this is a `u32` variable",
        ),
        (
            "let (i, f): (u32, f32); suld.b.a1d.b32.trap(i, [s, (f, n)]);",
            "2:48: `suld.b.a1d.b32.trap` wants a `u32` variable; this is a `f32`",
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
        ("sured.b.xor._1d.b32.trap([s, n], n);", "2:21: PTX's `sured` has no `.xor` that kgen knows"),
    ];
    for (body, want) in cases {
        assert_eq!(compile("x: *mut f32, n: u32, h: f16, s: u64", body).unwrap_err(), want, "{body}");
    }
}

#[test]
fn a_module_is_of_the_oldest_ptx_and_gpu_its_instructions_need() {
    let cases = [
        ("", ".version 2.3 .target sm_20"),
        ("let i: u32; popc.b32(i, n);", ".version 2.3 .target sm_20"),
        ("let i: u32; shfl.idx.b32(i, n, 0, 31);", ".version 3.0 .target sm_30"),
        ("let i: u32; vadd.u32.u32.u32(i, n.b0, n.h1);", ".version 2.3 .target sm_20"),
        ("let i: u32; vset4.u32.u32.lt(i, n, n, n);", ".version 3.0 .target sm_30"),
        ("let h: f16x2;", ".version 4.2 .target sm_53"),
        ("let i: u32; dp4a.u32.u32(i, n, n, n);", ".version 5.0 .target sm_61"),
        ("brx.idx(n, (a, b)); label(a); label(b);", ".version 6.0 .target sm_30"),
        ("nanosleep.u32(n);", ".version 6.2 .target sm_70"),
        ("let i: u32; shfl.idx.b32(i, n, 0, 31); nanosleep.u32(n);", ".version 6.2 .target sm_70"),
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
        ("let a: u64; clmad.lo.u64(a, y, y, y);", ".version 9.3 .target sm_80"),
        ("let i: u32; suld.b._2d.b32.trap(i, [y, (n, n)]);", ".version 3.1 .target sm_20"),
        ("let i: u32; suq.memory_layout.b32(i, [y]);", ".version 4.2 .target sm_20"),
        ("sured.b.max._1d.u64.trap([y, 0], y);", ".version 8.1 .target sm_50"),
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
    let ptx = kgen::ptx_str(TABLES).unwrap();
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

#[test]
fn a_surface_is_its_handle_in_brackets_with_the_coordinates_in_it() {
    let ptx = kgen::ptx_str(SURFACES).unwrap();
    let (head, entries) = ptx.split_once(".visible .entry ").unwrap();
    assert!(head.contains(".version 3.1\n.target sm_20\n"), "{head}");
    for line in [
        "\tsuld.b.2d.v4.b32.clamp {%r, %g, %b, %a}, [%from, {%x, %y}];\n",
        "\tsust.b.2d.v4.b32.clamp [%to, {%x, %y}], {%r, %g, %b, %a};\n",
        "\tsuq.width.b32 %width, [%bins];\n",
        "\tsured.b.add.1d.u32.trap [%bins, {%v}], 1;\n",
    ] {
        assert!(entries.contains(line), "{line}{entries}");
    }
}

#[test]
fn a_kernel_or_module_is_for_gpus_that_have_all_it_uses() {
    let cases = [
        (
            "let i: u32; let v: f32; shfl.idx.b32(i, n, 0, 31); tanh.approx.f32(v, f);",
            "2:64: `shfl.idx.b32` before it is for GPUs older than sm_70 from PTX 6.4 on, and `tanh.approx.f32` needs \
             sm_75 and PTX 7.0: no module has both",
        ),
        (
            "let i: u32; let v: f32; tanh.approx.f32(v, f); shfl.idx.b32(i, n, 0, 31);",
            "2:57: `shfl.idx.b32` is for GPUs older than sm_70 from PTX 6.4 on, and `tanh.approx.f32` before it needs \
             sm_75 and PTX 7.0: no module has both",
        ),
        (
            "let i: u32; shfl.idx.b32(i, n, 0, 31); redux.sync.add.u32(i, n, n);",
            "2:55: `shfl.idx.b32` before it is for GPUs older than sm_70 from PTX 6.4 on, and `redux.sync.add.u32` \
             needs sm_80, which PTX has from 7.0 on: no module has both",
        ),
        (
            "let i: u32; let g: f16; shfl.idx.b32(i, n, 0, 31); nanosleep.u32(n); abs.f16(g, h);",
            "2:74: `shfl.idx.b32` before it is for GPUs older than sm_70 from PTX 6.4 on; `nanosleep.u32` before it \
             needs sm_70, and `abs.f16` PTX 6.5: no module has them all",
        ),
    ];
    for (body, want) in cases {
        assert_eq!(compile("n: u32, f: f32, h: f16", body).unwrap_err(), want, "{body}");
    }
    let src = r#"
        unsafe extern "ptx-kernel" fn old(n: u32) {
            let i: u32;
            shfl.idx.b32(i, n, 0, 31);
        }
        unsafe extern "ptx-kernel" fn new(f: f32) {
            let v: f32;
            tanh.approx.f32(v, f);
        }
    "#;
    assert_eq!(
        kgen::ptx_str(src).unwrap_err(),
        "`shfl.idx.b32` in `old` is for GPUs older than sm_70 from PTX 6.4 on, and `tanh.approx.f32` in `new` needs \
         sm_75 and PTX 7.0: no module has both"
    );
}

#[test]
fn a_kernel_takes_pointers_and_numbers_and_returns_nothing() {
    let e = compile("x: bool", "").unwrap_err();
    assert_eq!(
        e,
        "1:36: a GPU kernel takes pointers, integers and the floats `f16`, `bf16`, `f32` and `f64`, not `bool`"
    );
    let e = kgen::ptx_str("unsafe extern \"ptx-kernel\" fn k() -> f32 {}").unwrap_err();
    assert_eq!(e, "1:38: a GPU kernel returns nothing, not `f32`: it writes what it works out to memory");
}

#[test]
fn a_kernel_names_the_constants_it_reads() {
    let src = r#"
        const ONE: f32 = 1.0;
        const UNREAD: u32 = 2;

        pub unsafe extern "ptx-kernel" fn k(x: *mut f32) {
            let v: f32;
            mov.f32(v, ONE);
            st.global.f32([x], v);
        }
    "#;
    let module = kgen::lang::parse(src).unwrap();
    let k = kgen::ptx(&module, kgen::function(&module, "k").unwrap()).unwrap();
    assert_eq!(k.consts, ["ONE"]);
}

/// How an instance gives its operands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Style {
    /// A variable for each, and the first of the integers an operand may be one of.
    Vars,
    /// A value known now for each that may be one, and the last of those integers.
    Values,
    /// `_` for the first result that may be left unwritten, and `!p` for each predicate that
    /// may be read inverted.
    Sinks,
    /// Without the operands that may be left out, and one result where two may be.
    Short,
    /// Every operand, and one result where two may be.
    Single,
    /// Memory the kernel declares for an address in shared or local memory.
    Memory,
    /// A part of each variable that may be one, as a video instruction selects it, and each
    /// operand that may be negated, negated.
    Parts,
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
/// `f32` or `f64` is not, nor a predicate.
fn value(t: Ty, spec: &Spec) -> Option<Given<Var>> {
    match t {
        Ty::F32 if spec.fraction => Some(Given::F32(0.5)),
        Ty::F32 => Some(Given::F32(1.5)),
        Ty::F64 => Some(Given::F64(1.5)),
        Ty::Pred | Ty::B128 => None,
        _ if t.kind() == Kind::Float => None,
        _ => Some(Given::Int(spec.step.unwrap_or(1))),
    }
}

/// An instance being made: the kernel, and whether its operands differ from those of
/// [`Style::Vars`].
struct Making {
    kernel: Kernel,
    vars: usize,
    differs: bool,
    sunk: bool,
}

impl Making {
    fn var(&mut self, t: Ty) -> Var {
        self.vars += 1;
        self.kernel.var(&format!("v{}", self.vars), t.name()).unwrap()
    }

    /// Memory of 256 bytes at a multiple of 32, as a fragment `wmma` loads is, in local memory
    /// for `.local` and in shared memory otherwise.
    fn memory(&mut self, space: Space) -> Var {
        self.vars += 1;
        let space = if space == Space::Local { "local" } else { "shared" };
        self.kernel.var(&format!("s{}", self.vars), &format!("{space}<[u8; 256], 32>")).unwrap()
    }

    /// A variable of type `t` for `spec`: a part of one where it must be, and in
    /// [`Style::Parts`] where it may be, and negated there where it may be.
    fn reg(&mut self, spec: &Spec, t: Ty, style: Style) -> Given<Var> {
        let mut v = self.var(t);
        if let Some((sel, must)) = spec.part
            && (must || style == Style::Parts)
        {
            self.differs |= !must;
            v = self.kernel.field(v, example(sel)).unwrap();
        }
        if spec.neg && style == Style::Parts {
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
        let mut parts = vec![];
        for k in 0..lanes {
            let t = match spec.lead {
                Some(lead) if k == 0 => lead,
                _ => b.ty(spec.ty.expect("a register's operand has a type")),
            };
            let part = if spec.memory {
                Some(Given::Var(self.memory(b.space())))
            } else if style == Style::Sinks && sink && !self.sunk {
                self.sunk = true;
                Some(Given::Special("_".into()))
            } else if style == Style::Sinks && spec.not {
                Some(Given::Not(Box::new(Given::Var(self.var(t)))))
            } else if style == Style::Values && spec.role == Role::Src && !spec.reg_only {
                value(t, spec)
            } else {
                None
            };
            self.differs |= part.is_some() && !spec.memory;
            parts.push(part.unwrap_or_else(|| self.reg(spec, t, style)));
        }
        match lanes {
            1 => parts.pop().expect("one"),
            _ => Given::Vector(parts),
        }
    }
}

/// Kernel `k{n}`, of the one instruction `name`, of form `b`, its operands as `style` gives
/// them; `None` if they would be those of [`Style::Vars`], or kgen's refusal.
fn instance(n: usize, name: &str, b: &Bound, style: Style) -> Option<Result<Entry, String>> {
    let mut m = Making { kernel: Kernel::new(format!("k{n}")), vars: 0, differs: false, sunk: false };
    let addr = m.kernel.var("a", "u64").unwrap();
    let mut spaces = b.spaces().into_iter();
    let mut args = vec![];
    for spec in b.form.operands.iter().filter(|s| s.when.is_none_or(|w| b.has(w))) {
        if spec.optional && style == Style::Short {
            m.differs = true;
            break;
        }
        args.push(match spec.role {
            Role::Label => Given::Label(0),
            Role::Labels => Given::Vector(vec![Given::Label(0)]),
            Role::Handle => {
                let items: Vec<Given<Var>> = spec.inner.iter().map(|s| m.operand(b, s, style)).collect();
                match items.as_slice() {
                    &[Given::Var(t)] => Given::Mem(Address { base: Some(t), index: None, disp: 0 }),
                    _ => Given::List(items),
                }
            }
            Role::Addr(_) => match spaces.next().expect("a space for each address") {
                s @ (Space::Shared | Space::Local) if style == Style::Memory => {
                    m.differs = true;
                    Given::Mem(Address { base: Some(m.memory(s)), index: None, disp: 32 })
                }
                _ => Given::Mem(Address { base: Some(addr), index: None, disp: 0 }),
            },
            Role::Imm(Imm::Bits(_)) => Given::Int(1),
            Role::Imm(imm @ Imm::Of(_)) => {
                let values = imm.values();
                m.differs |= style == Style::Values && values.len() > 1;
                Given::Int(if style == Style::Values { values[values.len() - 1] } else { values[0] })
            }
            Role::Imm(Imm::Pow2(_)) => Given::Int(16),
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
        return Some(Err(format!("kgen refuses `{name}` as {style:?}: {}", f.message)));
    }
    if b.form.operands.iter().any(|s| matches!(s.role, Role::Label | Role::Labels)) {
        m.kernel.label(0);
    }
    m.kernel.end().unwrap();
    Some(Ok(m.kernel.finish()))
}

/// Kernel `k{n}` of the one instruction `name`, of `form` with the words `choice`, written
/// as kgen would write it if it took it: for a name it refuses, which `ptxas` should too.
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

/// Operand `spec` of `b` as registers, each added to `regs`: one, or a part of one where it
/// must be; a vector, braced even of one; or two written at once.
fn registers(b: &Bound, spec: &Spec, regs: &mut Vec<Ty>) -> String {
    let t = b.ty(spec.ty.expect("a register's operand has a type"));
    let lanes = match spec.count {
        Count::One => 1,
        Count::Fixed(n) => n,
        Count::Vec => b.lanes(),
    };
    let names: Vec<String> = (0..lanes)
        .map(|k| {
            regs.push(if k == 0 { spec.lead.unwrap_or(t) } else { t });
            format!("%v{}", regs.len())
        })
        .collect();
    let braced = matches!(spec.count, Count::Fixed(_));
    let first = match spec.part {
        Some((sel, true)) => format!("{}.{}", names[0], example(sel)),
        _ if lanes == 1 && !braced => names[0].clone(),
        _ => format!("{{{}}}", names.join(", ")),
    };
    match &spec.second {
        Some((second, _)) => format!("{first}|{}", registers(b, second, regs)),
        None => first,
    }
}

/// The kernels of the instructions kgen reads from no form's operands: each special
/// register read, as it is and as the fewer bits PTX lets `mov` read of it, and the floats
/// of 16 bits known now that a `b16` takes.
fn others(n: &mut usize) -> Vec<Entry> {
    let mut out = vec![];
    let mut one = |name: &str, t: Ty, a: Given<Var>| {
        *n += 1;
        let mut kernel = Kernel::new(format!("k{n}"));
        let v = kernel.var("v", t.name()).unwrap();
        kernel.inst(name, &[Given::Var(v), a], None).unwrap_or_else(|f| panic!("kgen refuses `{name}`: {}", f.message));
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
    out
}

/// The kernels `ptxas` refuses of each module, by name, each with the first error it finds
/// in it; `?` for one outside any kernel.
fn assemble(ptxas: &Path, modules: &[String]) -> Vec<BTreeMap<String, String>> {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("forms");
    std::fs::create_dir_all(&dir).unwrap();
    let next = AtomicUsize::new(0);
    let out: Vec<Mutex<BTreeMap<String, String>>> = modules.iter().map(|_| Mutex::default()).collect();
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(text) = modules.get(i) else { break };
                    *out[i].lock().unwrap() = refused(ptxas, text, &dir.join(i.to_string()));
                }
            });
        }
    });
    out.into_iter().map(|m| m.into_inner().unwrap()).collect()
}

/// The kernels `ptxas` refuses of the module `text`, which it reads at `path`. A fatal error
/// stops it in the kernel it is in, so it goes on without that one; advice, as that `mma.sp`
/// is slower than `mma.sp::ordered_metadata`, is no refusal.
fn refused(ptxas: &Path, text: &str, path: &Path) -> BTreeMap<String, String> {
    let (src, bin) = (path.with_extension("ptx"), path.with_extension("cubin"));
    let sm: u32 = text
        .lines()
        .find_map(|l| l.strip_prefix(".target sm_"))
        .and_then(|t| t.parse().ok())
        .expect("a module has a target");
    let (mut text, mut out) = (text.to_string(), BTreeMap::new());
    loop {
        std::fs::write(&src, &text).unwrap();
        let run = Command::new(ptxas)
            .args(["--gpu-name", &format!("sm_{}", sm.max(75))])
            .arg(&src)
            .arg("-o")
            .arg(&bin)
            .output()
            .unwrap();
        let _ = std::fs::remove_file(&bin);
        if run.status.success() {
            return out;
        }
        let stderr = String::from_utf8_lossy(&run.stderr);
        let mut fatal = None;
        for l in stderr.lines() {
            let Some((n, message)) = l.split_once(", line ").and_then(|(_, r)| r.split_once(';')) else { continue };
            let Ok(n) = n.parse() else { continue };
            let (k, message) = (kernel_at(&text, n).unwrap_or("?").to_string(), message.trim());
            if message.starts_with("info") {
                continue;
            }
            if message.starts_with("fatal") {
                fatal = Some(k.clone());
            }
            out.entry(k).or_insert_with(|| message.to_string());
        }
        match fatal {
            Some(k) if k != "?" => text = without(&text, &k),
            _ => {
                if out.is_empty() {
                    // A crash says nothing.
                    let why =
                        if stderr.trim().is_empty() { format!("ptxas: {}", run.status) } else { stderr.trim().into() };
                    out.insert("?".into(), why);
                }
                return out;
            }
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

/// Every version of PTX the `ptxas` here reads, from 1.0 to 9.4: those before 2.3 without
/// `.address_size`, which they have not.
const VERSIONS: [u32; 47] = [
    10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 30, 31, 32, 40, 41, 42, 43, 50, 60, 61, 62, 63, 64, 65, 70, 71, 72, 73, 74,
    75, 76, 77, 78, 80, 81, 82, 83, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94,
];

/// The newest GPU a module of PTX `v` with what `n` needs may be for: before 2.0, one of
/// Tesla's, `sm_11` or `sm_13`, which kgen writes for neither of.
fn newest(v: u32, n: Needs) -> u32 {
    match v {
        ..12 => 11,
        12..20 => 13,
        _ => kgen_ptx::GPUS
            .iter()
            .filter(|&&(g, came)| came <= v && (n.below == 0 || g < n.below || v < n.since))
            .map(|&(g, _)| g)
            .max()
            .expect("every version from 2.0 has Fermi"),
    }
}

/// The version of PTX before `v`.
fn before(v: u32) -> Option<u32> {
    let i = VERSIONS.iter().position(|&x| x == v).expect("a version ptxas reads");
    i.checked_sub(1).map(|i| VERSIONS[i])
}

/// The version of PTX that came with the GPU `sm`.
fn came(sm: u32) -> u32 {
    kgen_ptx::GPUS.iter().find(|&&(g, _)| g == sm).map(|&(_, v)| v).expect("a GPU kgen writes for")
}

/// The module `text` as one of PTX `v` for the GPU `sm`.
fn retarget(text: &str, v: u32, sm: u32) -> String {
    text.lines()
        .filter(|l| v >= 23 || *l != ".address_size 64")
        .map(|l| match l {
            _ if l.starts_with(".version ") => format!(".version {}.{}\n", v / 10, v % 10),
            _ if l.starts_with(".target ") => format!(".target sm_{sm}\n"),
            _ => format!("{l}\n"),
        })
        .collect()
}

#[test]
fn ptxas_takes_every_form_kgen_knows_where_kgen_says() {
    let Some(ptxas) = ptxas() else {
        eprintln!("no ptxas here: skipped; set PTXAS to run it");
        return;
    };
    let (mut n, mut failures) = (0, vec![]);
    let mut what: BTreeMap<String, String> = BTreeMap::new();
    let mut groups: BTreeMap<(u32, u32, u32, u32), Vec<Entry>> = BTreeMap::new();
    let mut add = |e: Entry, what: &mut BTreeMap<String, String>| {
        let line = e.program.body.iter().find_map(|i| match i {
            kgen_core::Item::Inst(inst) => Some(inst.op.0.clone()),
            _ => None,
        });
        what.insert(e.program.name.clone(), line.unwrap_or_default());
        let n = e.needs();
        groups.entry((n.ptx, n.sm, n.below, n.since)).or_default().push(e);
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
            for style in
                [Style::Vars, Style::Values, Style::Sinks, Style::Short, Style::Single, Style::Memory, Style::Parts]
            {
                n += 1;
                match instance(n, &name, b, style) {
                    None => {}
                    Some(Ok(e)) => add(e, &mut what),
                    Some(Err(m)) => failures.push(m),
                }
            }
        }
    }
    for e in others(&mut n) {
        add(e, &mut what);
    }
    for ((ptx, ..), entries) in groups.iter().filter(|((ptx, ..), _)| !VERSIONS.contains(ptx)) {
        let (count, some): (usize, Vec<&str>) =
            (entries.len(), entries.iter().take(5).map(|e| what[&e.program.name].as_str()).collect());
        failures.push(format!("{count} kernels need PTX {ptx}, which ptxas reads none of: {}", some.join("; ")));
    }
    // Each group as kgen writes it, and at the version it needs, for the newest GPU of that
    // version; and one step older in each way it needs more than PTX 1.0 and Fermi, and for
    // the GPU it is too new for: then each kernel is wrong.
    let mut modules: Vec<(String, Expect)> = vec![];
    for (&(ptx, sm, below, since), entries) in groups.iter().filter(|((ptx, ..), _)| VERSIONS.contains(ptx)) {
        let n = Needs { ptx, sm, below, since };
        let text = kgen_ptx::module(entries).unwrap();
        let gpu = newest(ptx, n);
        if gpu < sm {
            let v = format!("{}.{}", ptx / 10, ptx % 10);
            failures.push(format!("{} kernels need PTX {v} and sm_{sm}, which no module of {v} is for", entries.len()));
        }
        let exact = retarget(&text, ptx, gpu);
        if exact != text {
            modules.push((exact, Expect::Takes));
        }
        modules.push((text.clone(), Expect::Takes));
        if let Some(v) = before(ptx) {
            modules.push((retarget(&text, v, newest(v, n)), Expect::Refuses));
        }
        if sm > Needs::FLOOR.sm {
            let i = kgen_ptx::GPUS.iter().position(|&(g, _)| g == sm).expect("a GPU kgen writes for");
            let g = kgen_ptx::GPUS[i - 1].0;
            modules.push((retarget(&text, ptx.max(came(g)), g), Expect::Refuses));
        }
        if below != 0 {
            let v = ptx.max(came(below));
            if let Some(last) = before(since).filter(|&last| last >= v) {
                modules.push((retarget(&text, last, below), Expect::Takes));
            }
            modules.push((retarget(&text, v.max(since), below), Expect::Refuses));
        }
    }
    // And the names kgen refuses, for the newest GPU and at the newest version kgen knows,
    // where nothing but the name is wrong.
    let mut why: BTreeMap<String, String> = BTreeMap::new();
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
    for some in hand.chunks(100) {
        let text =
            format!("// Written by hand.\n\n.version 9.4\n.target sm_86\n.address_size 64\n\n{}", some.join("\n"));
        modules.push((text, Expect::Refuses));
    }
    let texts: Vec<String> = modules.iter().map(|(t, _)| t.clone()).collect();
    let results = assemble(&ptxas, &texts);
    for ((text, expect), refused) in modules.iter().zip(&results) {
        let header = header(text);
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
                            taken += &format!(", which kgen refuses: {m}");
                        }
                        failures.push(taken);
                    }
                }
            }
        }
    }
    eprintln!("{n} instances in {} modules; {refusals} names kgen refuses", modules.len());
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.join("\n"));
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
    let ptx = kgen::ptx_str(SHARED).unwrap();
    let (head, entries) = ptx.split_once(".visible .entry ").unwrap();
    let want = "// Written by kgen.

.version 2.3
.target sm_20
.address_size 64

.extern .shared .align 4 .b8 reverse$buf[];

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
    assert!(entries.contains("\tmov.u32 %s, reverse$buf;\n"));
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
