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
use kgen_ptx::forms::{self, Bound, Check, Count, Form, Imm, Role, Spec};
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
        ("let v: f32; ld.f32(v, [n]);", "2:23: `ld.f32` takes an address in a 64-bit variable; this is a `u32`"),
        ("let v: f32; ld.glob.f32(v, [x]);", "2:21: PTX's `ld` has no `.glob` that kgen knows"),
        ("let i: u32; add.u32(i, tid.x, 1);", "2:28: `add.u32` reads no special register; `mov.u32` does"),
        ("let i: u32; mov.u32(i, tod.x);", "2:28: `tod.x` is not defined"),
        ("let i: u32; if n { mov.u32(i, 1); }", "2:24: an instruction is guarded by a `pred`; this is a `u32`"),
        ("let v: f32; add.rn.f32(v, v, 1.5);", "2:5: `v` is read before anything writes it"),
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
        ("elect.sync(n);", "2:7: PTX's `elect` is for sm_90 and newer, and kgen writes PTX for sm_61 to sm_86"),
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
        (
            "let g: f16; add.rn.f16(g, h, 1.5);",
            "2:30: `add.rn.f16` takes a `f16` only from a variable: `mov.b16(h, 1.5f16)` puts one in it",
        ),
        ("let v: f64; mov.f64(v, 1.5);", "2:24: `mov.f64` wants an `f64`; 1.5 is an `f32`: `1.5f64`"),
        ("let i: u16; mov.u16(i, clock);", "2:24: `mov.u16` reads a `u16`; `clock` is a `u32`"),
        (
            "let i: u32; mov.u32(i, clusterid.x);",
            "2:34: `clusterid.x` is a special register of PTX's that kgen does not read: it needs sm_90, and kgen \
             writes PTX for sm_61 to sm_86",
        ),
        (
            "let i: u32; mov.u32(i, perctamemorysize);",
            "2:24: `perctamemorysize` is a special register of PTX's that kgen does not read: a kernel has it only \
             with the directive `.minperctamemory`, which kgen does not write",
        ),
    ];
    for (body, want) in cases {
        assert_eq!(compile("x: *mut f32, n: u32, h: f16", body).unwrap_err(), want, "{body}");
    }
}

#[test]
fn a_module_is_of_the_oldest_ptx_and_gpu_its_instructions_need() {
    let cases = [
        ("", ".version 7.5 .target sm_61"),
        ("let i: u32; shfl.idx.b32(i, n, 0, 31);", ".version 7.5 .target sm_61"),
        ("nanosleep.u32(n);", ".version 7.5 .target sm_70"),
        ("let v: b128; ld.global.b128(v, [x]);", ".version 8.3 .target sm_70"),
        ("let v: f32; tanh.approx.f32(v, f);", ".version 7.5 .target sm_75"),
        ("let i: b32; mov.b32(i, reserved_smem_offset_begin);", ".version 7.6 .target sm_80"),
        ("let a: u64; clmad.lo.u64(a, y, y, y);", ".version 9.3 .target sm_80"),
        ("let v: f32; max.xorsign.abs.f32(v, f, f);", ".version 7.5 .target sm_86"),
    ];
    for (body, want) in cases {
        let ptx = compile("x: *mut f32, n: u32, f: f32, y: u64", body).unwrap();
        let header: Vec<&str> = ptx.lines().filter(|l| l.starts_with(".version") || l.starts_with(".target")).collect();
        assert_eq!(header.join(" "), want, "{body}");
    }
}

#[test]
fn a_kernel_or_module_is_for_gpus_that_have_all_it_uses() {
    let e = compile("n: u32", "let i: u32; shfl.idx.b32(i, n, 0, 31); nanosleep.u32(n);").unwrap_err();
    assert_eq!(
        e,
        "2:50: `nanosleep.u32` needs sm_70 or newer, and `shfl.idx.b32` before it is for GPUs older than sm_70: no \
         GPU has both"
    );
    let src = r#"
        unsafe extern "ptx-kernel" fn old(n: u32) {
            let i: u32;
            shfl.idx.b32(i, n, 0, 31);
        }
        unsafe extern "ptx-kernel" fn new(n: u32) {
            nanosleep.u32(n);
        }
    "#;
    assert_eq!(
        kgen::ptx_str(src).unwrap_err(),
        "`shfl.idx.b32` in `old` is for GPUs older than sm_70, and `nanosleep.u32` in `new` needs sm_70 or newer: \
         one module cannot have both"
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
    /// A variable for each.
    Vars,
    /// A value known now for each that may be one.
    Values,
    /// `_` for the first result that may be left unwritten, and `!p` for each predicate that
    /// may be read inverted.
    Sinks,
    /// Without the operands that may be left out, and one result where two may be.
    Short,
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

    /// The operand `spec` of `b`, as `style` gives it.
    fn operand(&mut self, b: &Bound, spec: &Spec, style: Style) -> Given<Var> {
        let lanes = match spec.count {
            Count::One => 1,
            Count::Fixed(n) => n,
            Count::Vec => b.lanes(),
        };
        let t = b.ty(spec.ty.expect("a register's operand has a type"));
        let mut parts = vec![];
        for _ in 0..lanes {
            let part = if style == Style::Sinks && spec.sink && !self.sunk {
                self.sunk = true;
                Some(Given::Special("_".into()))
            } else if style == Style::Sinks && spec.not {
                Some(Given::Not(Box::new(Given::Var(self.var(t)))))
            } else if style == Style::Values && spec.role == Role::Src && !spec.reg_only {
                value(t, spec)
            } else {
                None
            };
            self.differs |= part.is_some();
            parts.push(part.unwrap_or_else(|| Given::Var(self.var(t))));
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
    let mut args = vec![];
    for spec in b.form.operands.iter().filter(|s| s.when.is_none_or(|w| b.has(w))) {
        if spec.optional && style == Style::Short {
            m.differs = true;
            break;
        }
        args.push(match spec.role {
            Role::Label => Given::Label(0),
            Role::Addr(_) => Given::Mem(Address { base: Some(addr), index: None, disp: 0 }),
            Role::Imm(Imm::Bits(_)) => Given::Int(1),
            Role::Imm(Imm::Exactly(x)) => Given::Int(x),
            Role::Imm(Imm::Pow2(_)) => Given::Int(16),
            Role::Dst | Role::Src => {
                let first = m.operand(b, spec, style);
                match &spec.second {
                    Some((_, true)) if style == Style::Short => {
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
    if args.iter().any(|a| matches!(a, Given::Label(_))) {
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
            Role::Addr(_) => "[%a]".to_string(),
            Role::Imm(Imm::Bits(_)) => "1".to_string(),
            Role::Imm(Imm::Exactly(x)) => x.to_string(),
            Role::Imm(Imm::Pow2(_)) => "16".to_string(),
            Role::Dst | Role::Src => registers(&b, spec, &mut regs),
        })
        .collect();
    let decls: String =
        regs.iter().enumerate().map(|(i, t)| format!("\t.reg .{} %v{};\n", t.decl().name(), i + 1)).collect();
    let label = if form.operands.iter().any(|s| s.role == Role::Label) { "$L0:\n" } else { "" };
    format!(
        ".visible .entry k{n}()\n{{\n\t.reg .b64 %a;\n{decls}\n\t{name} {};\n{label}\tret;\n}}\n",
        operands.join(", ")
    )
}

/// Operand `spec` of `b` as registers, each added to `regs`: one, a vector, or two written
/// at once.
fn registers(b: &Bound, spec: &Spec, regs: &mut Vec<Ty>) -> String {
    let t = b.ty(spec.ty.expect("a register's operand has a type"));
    let lanes = match spec.count {
        Count::One => 1,
        Count::Fixed(n) => n,
        Count::Vec => b.lanes(),
    };
    let names: Vec<String> = (0..lanes)
        .map(|_| {
            regs.push(t);
            format!("%v{}", regs.len())
        })
        .collect();
    let first = if lanes == 1 { names[0].clone() } else { format!("{{{}}}", names.join(", ")) };
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

/// The errors `ptxas` finds in a module: each line's, by its number.
type Errors = Vec<(usize, String)>;

/// What `ptxas` makes of each module: `None` if it takes it, or the errors it finds.
fn assemble(ptxas: &Path, modules: &[String]) -> Vec<Option<Errors>> {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("forms");
    std::fs::create_dir_all(&dir).unwrap();
    let next = AtomicUsize::new(0);
    let out: Vec<Mutex<Option<Errors>>> = modules.iter().map(|_| Mutex::new(None)).collect();
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(text) = modules.get(i) else { break };
                    let (src, bin) = (dir.join(format!("{i}.ptx")), dir.join(format!("{i}.cubin")));
                    std::fs::write(&src, text).unwrap();
                    let sm: u32 = text
                        .lines()
                        .find_map(|l| l.strip_prefix(".target sm_"))
                        .and_then(|t| t.parse().ok())
                        .expect("a module has a target");
                    let run = Command::new(ptxas)
                        .args(["--gpu-name", &format!("sm_{}", sm.max(75))])
                        .arg(&src)
                        .arg("-o")
                        .arg(&bin)
                        .output()
                        .unwrap();
                    if !run.status.success() {
                        let stderr = String::from_utf8_lossy(&run.stderr);
                        let mut errors: Errors = stderr
                            .lines()
                            .filter(|l| l.contains("error"))
                            .filter_map(|l| {
                                let (_, rest) = l.split_once(", line ")?;
                                let (n, message) = rest.split_once(';')?;
                                Some((n.parse().ok()?, message.trim().to_string()))
                            })
                            .collect();
                        if errors.is_empty() {
                            errors.push((0, stderr.trim().to_string()));
                        }
                        *out[i].lock().unwrap() = Some(errors);
                    }
                    let _ = std::fs::remove_file(&bin);
                }
            });
        }
    });
    out.into_iter().map(|m| m.into_inner().unwrap()).collect()
}

/// For each line of `text`, the kernel it is in, by name.
fn kernel_at(text: &str, line: usize) -> Option<&str> {
    text.lines().take(line).filter_map(|l| l.strip_prefix(".visible .entry ")).last().and_then(|l| l.split('(').next())
}

/// What `entries` need, as a module's header writes it: each at least [`Needs::FLOOR`].
fn key(n: Needs) -> (u32, u32, u32) {
    (n.ptx.max(Needs::FLOOR.ptx), n.sm.max(Needs::FLOOR.sm), n.below)
}

/// What `ptxas` should make of a module.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Expect {
    /// It assembles.
    Takes,
    /// Each of its kernels is wrong.
    Refuses,
    /// Each is wrong, if `ptxas` checks it for the module's `.target` and not for sm_75.
    Unsure,
}

/// The versions of PTX from kgen's oldest to the newest `ptxas` here reads, and the GPUs
/// from Pascal to Ampere.
const VERSIONS: [u32; 18] = [75, 76, 77, 78, 80, 81, 82, 83, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94];
const GPUS: [u32; 7] = [61, 62, 70, 72, 75, 80, 86];

#[test]
fn ptxas_takes_every_form_kgen_knows_where_kgen_says() {
    let Some(ptxas) = ptxas() else {
        eprintln!("no ptxas here: skipped; set PTXAS to run it");
        return;
    };
    let (mut n, mut failures) = (0, vec![]);
    let mut what: BTreeMap<String, String> = BTreeMap::new();
    let mut groups: BTreeMap<(u32, u32, u32), Vec<Entry>> = BTreeMap::new();
    let mut add = |e: Entry, what: &mut BTreeMap<String, String>| {
        let line = e.program.body.iter().find_map(|i| match i {
            kgen_core::Item::Inst(inst) => Some(inst.op.0.clone()),
            _ => None,
        });
        what.insert(e.program.name.clone(), line.unwrap_or_default());
        groups.entry(key(e.needs())).or_default().push(e);
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
            for style in [Style::Vars, Style::Values, Style::Sinks, Style::Short] {
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
    // Each group at what it needs, and one step older in each way it needs more than the
    // floor: then each kernel is wrong. ptxas builds for no GPU older than sm_75, so one
    // that takes a kernel for an older `.target` may only be checking it for sm_75.
    let mut modules: Vec<(String, Expect)> = vec![];
    for (&(ptx, sm, below), entries) in &groups {
        let text = kgen_ptx::module(entries).unwrap();
        modules.push((text.clone(), Expect::Takes));
        let older = |from: &str, to: String| text.replacen(from, &to, 1);
        if ptx > Needs::FLOOR.ptx {
            let v = VERSIONS[VERSIONS.iter().position(|&v| v == ptx).unwrap() - 1];
            let (from, to) =
                (format!(".version {}.{}\n", ptx / 10, ptx % 10), format!(".version {}.{}\n", v / 10, v % 10));
            modules.push((older(&from, to), Expect::Refuses));
        }
        if sm > Needs::FLOOR.sm {
            let g = GPUS[GPUS.iter().position(|&g| g == sm).unwrap() - 1];
            let expect = if g < 75 { Expect::Unsure } else { Expect::Refuses };
            modules.push((older(&format!(".target sm_{sm}\n"), format!(".target sm_{g}\n")), expect));
        }
        if below != 0 {
            modules.push((older(&format!(".target sm_{sm}\n"), format!(".target sm_{below}\n")), Expect::Refuses));
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
    let mut unsure = vec![];
    for ((text, expect), errors) in modules.iter().zip(&results) {
        let header: String = text.lines().skip(2).take(2).collect::<Vec<_>>().join(" ");
        match (expect, errors) {
            (Expect::Takes, None) => {}
            (Expect::Takes, Some(errors)) => {
                for (line, message) in errors {
                    let k = kernel_at(text, *line).unwrap_or("?");
                    failures.push(format!("{header}: `{}`: {message}", what.get(k).map_or("?", String::as_str)));
                }
            }
            (Expect::Refuses | Expect::Unsure, errors) => {
                let wrong: Vec<&str> = errors.iter().flatten().filter_map(|(line, _)| kernel_at(text, *line)).collect();
                for k in
                    text.lines().filter_map(|l| l.strip_prefix(".visible .entry ")).filter_map(|l| l.split('(').next())
                {
                    if !wrong.contains(&k) {
                        let mut taken = format!("{header}: takes `{}` too", what.get(k).map_or("?", String::as_str));
                        if let Some(m) = why.get(k) {
                            taken += &format!(", which kgen refuses: {m}");
                        }
                        match expect {
                            Expect::Unsure => unsure.push(taken),
                            _ => failures.push(taken),
                        }
                    }
                }
            }
        }
    }
    eprintln!("{n} instances in {} modules; {refusals} names kgen refuses", modules.len());
    eprintln!("{} kernels ptxas may only have checked for sm_75:\n{}", unsure.len(), unsure.join("\n"));
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.join("\n"));
}
