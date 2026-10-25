//! Kernels in the language, compiled to PTX: what they come to, that `ptxas` takes them, and
//! what the errors say.

use std::path::PathBuf;
use std::process::Command;

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
    let (src, out) = (dir.join("kernels.ptx"), dir.join("kernels.cubin"));
    std::fs::write(&src, kgen::ptx_str(KERNELS).unwrap()).unwrap();
    let run = Command::new(&ptxas)
        .args(["--gpu-name", "sm_75", "--warning-as-error"])
        .arg(&src)
        .arg("-o")
        .arg(&out)
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
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
        ("let v: f32; ld.f32(v, [x]);", "2:16: `ld.f32` says where it reads or writes: `ld.global`"),
        ("let v: f32; ld.glob.f32(v, [x]);", "2:21: PTX's `ld` has no `.glob` that kgen knows"),
        ("let i: u32; add.u32(i, tid.x, 1);", "2:28: `add.u32` reads no special register; `mov.u32` does"),
        ("let i: u32; mov.u32(i, tod.x);", "2:28: `tod.x` is not defined"),
        ("let i: u32; if n { mov.u32(i, 1); }", "2:24: an instruction is guarded by a `pred`; this is a `u32`"),
        ("let v: f32; add.rn.f32(v, v, 1.5);", "2:5: `v` is read before anything writes it"),
        ("let v: f64;", "2:8: PTX variables are `pred`, `b32`, `u32`, `s32`, `f32`, `b64`, `u64` and `s64`, not `f64`"),
        ("vfrob.f32(x);", "2:7: PTX has no instruction `vfrob` that kgen knows"),
    ];
    for (body, want) in cases {
        assert_eq!(compile("x: *mut f32, n: u32", body).unwrap_err(), want, "{body}");
    }
}

#[test]
fn a_kernel_takes_pointers_and_numbers_and_returns_nothing() {
    let e = compile("x: bool", "").unwrap_err();
    assert_eq!(e, "1:36: a GPU kernel takes pointers, integers of 32 and 64 bits and `f32`, not `bool`");
    let e = kgen::ptx_str("unsafe extern \"ptx-kernel\" fn k() -> f32 {}").unwrap_err();
    assert_eq!(e, "1:38: a GPU kernel returns nothing, not `f32`: it writes what it works out to memory");
}
