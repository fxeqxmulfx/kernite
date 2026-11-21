//! `X86` through the ports kernite-build calls, driven by hand as a use case drives them: each
//! static laid out and each callee seen, then the kernel told of them, written and finished.

use kernite_build::{Backend, Bytes, Called, Compiled, HostStatic, Naked, Native, StaticError, Unfinished, function};
use kernite_core::{Class, PReg};
use kernite_lang::{ItemKind, Module, parse};
use kernite_x86::X86;
use kernite_x86::intel::Listing;
use kernite_x86::lang::Callee;

type Result<T = (), E = Box<dyn std::error::Error + Send + Sync>> = std::result::Result<T, E>;

/// A kernel that names memory of its module and calls a function of it comes out of the port as
/// Rust takes it: its code as `naked_asm!` reads it, its listing, the CPU features it needs, the
/// constants it reads, and the function and the static it names. The static comes out as Rust
/// lays it out, holding what it says.
#[test]
fn x86_compiles_a_kernel_through_the_port() -> Result {
    let src = "static COUNTS: global<[u32; 2]> = [1, 2];\n\
               unsafe extern \"sysv64\" fn half(x: f32) -> f32 { ret(x); }\n\
               unsafe extern \"sysv64\" fn k(out: *mut f32) {\n\
               let v: ymm;\n\
               call(v.xmm, half, 2.5);\n\
               vmovss([out], v.xmm);\n\
               add(dword[COUNTS + 4], 1);\n\
               }";
    let module = parse(src)?;
    let ItemKind::Static(counts) = &module.items[0].kind else { return Err("a static first".into()) };
    let (half, k) =
        (function(&module, "half").ok_or("function gave None")?, function(&module, "k").ok_or("function gave None")?);
    let statics = [X86.lay_out("COUNTS", &kernite_lang::memory(&module, counts)?).map_err(|e| format!("{e:?}"))?];
    let callee = X86.callee(half).ok_or("callee gave None")?;
    let callees = [Called { name: "half".into(), function: "half".into(), args: vec![], callee }];
    let mut kernel = X86.kernel(k, "k", &[], false, &statics, &callees);
    kernite_lang::kernel(&module, k, &[], &mut kernel)?;
    let code = X86.finish(kernel, &[]).map_err(|e| format!("{e:?}"))?;
    let text = "# k(gpr) -> ()
# needs: avx
# frame: 0 bytes aligned to 0, saves [rbx]
# out=rbx,rdi v=ymm0 x=ymm0
    push rbx
    mov rbx, rdi
    vmovss xmm0, dword ptr [rip + {c0}]
    call {f0}
    vmovss dword ptr [rbx], xmm0
    add dword ptr [rip + {COUNTS} + 4], 1
    pop rbx
    ret
# f0: half
# c0, aligned to 4:
#   00 00 20 40
";
    let lines = [
        "push rbx",
        "mov rbx, rdi",
        "vmovss xmm0, dword ptr [rip + {c0}]",
        "call {f0}",
        "vmovss dword ptr [rbx], xmm0",
        "add dword ptr [rip + {COUNTS} + 4], 1",
        "pop rbx",
        "ret",
    ];
    let want = Naked {
        code: lines.map(String::from).to_vec(),
        text: text.into(),
        needs: vec!["avx".into()],
        consts: vec![Bytes { bytes: 2.5f32.to_le_bytes().to_vec(), align: 4 }],
        funcs: vec!["half".into()],
        reach: vec![],
        statics: vec!["COUNTS".into()],
    };
    assert_eq!(X86.naked(&code), want);
    let want = HostStatic { elem: "u32".into(), size: 4, len: 2, align: 4, init: vec![1, 0, 0, 0, 2, 0, 0, 0] };
    assert_eq!(X86.host_static(&statics[0]), want);
    Ok(())
}

/// A call at an address names no function in the code, so `funcs` has none: the functions taken
/// together last come out of the port in a list of their own, each once, in the order the source
/// gives them.
#[test]
fn a_call_at_an_address_hands_on_the_functions_it_may_reach() -> Result {
    let src = "unsafe extern \"sysv64\" fn half(x: f32) -> f32 { ret(x); }\n\
               unsafe extern \"sysv64\" fn twice(x: f32) -> f32 { ret(x); }\n\
               unsafe extern \"sysv64\" fn third(x: f32) -> f32 { ret(x); }\n\
               unsafe extern \"sysv64\" fn k(fp: u64) {\n\
               let v: ymm;\n\
               call(v.xmm, fp, 2.5, (half, twice));\n\
               call(v.xmm, fp, 1.5, (third, half));\n\
               }";
    let module = parse(src)?;
    let mut callees = vec![];
    for name in ["half", "twice", "third"] {
        let callee = X86.callee(function(&module, name).ok_or("function gave None")?).ok_or("callee gave None")?;
        callees.push(Called { name: name.into(), function: name.into(), args: vec![], callee });
    }
    let k = function(&module, "k").ok_or("function gave None")?;
    let mut kernel = X86.kernel(k, "k", &[], false, &[], &callees);
    kernite_lang::kernel(&module, k, &[], &mut kernel)?;
    let naked = X86.naked(&X86.finish(kernel, &[]).map_err(|e| format!("{e:?}"))?);
    assert!(naked.funcs.is_empty());
    assert_eq!(naked.reach, ["half", "twice", "third"]);
    Ok(())
}

/// Function `name` of `module`, any function of which it may call, finished with `done`, the code of the functions it
/// calls, as a use case that compiles them first gives it.
fn finished(module: &Module, name: &str, done: &[(String, Listing)]) -> Result<Listing> {
    let callees: Vec<_> = module
        .items
        .iter()
        .filter_map(|item| match &item.kind {
            ItemKind::Fn(f) => {
                Some(Called { name: f.name.clone(), function: f.name.clone(), args: vec![], callee: X86.callee(f)? })
            }
            _ => None,
        })
        .collect();
    let f = function(module, name).ok_or("function gave None")?;
    let mut kernel = X86.kernel(f, name, &[], false, &[], &callees);
    kernite_lang::kernel(module, f, &[], &mut kernel)?;
    let done: Vec<_> =
        done.iter().map(|(name, code)| (name.clone(), Ok(Compiled { code: code.clone(), mentions: vec![] }))).collect();
    Ok(X86.finish(kernel, &done).map_err(|e| format!("{e:?}"))?)
}

/// A listing says which registers a call of it changes, of those SysV leaves to a callee: those its code writes, less
/// those it saves, and none of those the functions it calls write, which their listings say. A kernel is finished with
/// the listings of the functions it calls, and a call changes what each of them changes, and each function they call in
/// turn: `jumps` writes none and `writes_rax` writes rax, so `x` is kept in rcx. Where a listing is missing, as it is
/// for a function that does not compile, a call changes every register a callee may, as one at an address with no
/// functions given does, which may reach any function: the listing of a kernel that makes one says it changes them all.
#[test]
fn a_call_changes_what_the_listings_of_its_callees_say() -> Result {
    let src = "unsafe extern \"sysv64\" fn writes_rax() { let t: gpr; mov(t, 1); add(t, t); }\n\
               unsafe extern \"sysv64\" fn jumps() { call(writes_rax); }\n\
               unsafe extern \"sysv64\" fn calls() { call(writes_rax); pause(); }\n\
               unsafe extern \"sysv64\" fn anywhere(fp: u64) { call(fp); pause(); }\n\
               unsafe extern \"sysv64\" fn ten() {\n\
               let (a, b, c, d, e, f, g, h, i, j): (gpr, gpr, gpr, gpr, gpr, gpr, gpr, gpr, gpr, gpr);\n\
               mov(a, 1); mov(b, 1); mov(c, 1); mov(d, 1); mov(e, 1);\n\
               mov(f, 1); mov(g, 1); mov(h, 1); mov(i, 1); mov(j, 1);\n\
               add(a, b); add(a, c); add(a, d); add(a, e); add(a, f); add(a, g); add(a, h); add(a, i); add(a, j);\n\
               }\n\
               unsafe extern \"sysv64\" fn k() { let x: gpr; mov(x, 5); call(jumps); add(x, 1); }";
    let module = parse(src)?;
    let gpr = |hw| PReg { class: Class::Gpr, hw };
    let caller_saved = [0, 1, 2, 6, 7, 8, 9, 10, 11].map(gpr);
    let mut done = vec![];
    for (name, changes) in
        [("writes_rax", vec![gpr(0)]), ("jumps", vec![]), ("calls", vec![]), ("ten", caller_saved.to_vec())]
    {
        let listing = finished(&module, name, &done)?;
        assert_eq!(listing.changes, changes, "{name}");
        done.push((name.to_string(), listing));
    }
    assert!(done[3].1.code().any(|line| line == "push rbx"), "`ten` saves rbx, and a call of it leaves it as it was");
    let vectors = (0..16).map(|hw| PReg { class: Class::Vec, hw });
    let every: Vec<_> = caller_saved.into_iter().chain(vectors).collect();
    assert_eq!(finished(&module, "anywhere", &[])?.changes, every);
    let code = |done: &[(String, Listing)]| -> Result<_> {
        Ok(finished(&module, "k", done)?.code().collect::<Vec<_>>().join("\n"))
    };
    assert_eq!(code(&done)?, "sub rsp, 8\nmov rcx, 5\ncall {f0}\nadd rcx, 1\nadd rsp, 8\nret");
    let saving = "push rbx\nmov rbx, 5\ncall {f0}\nadd rbx, 1\npop rbx\nret";
    assert_eq!(code(&done[1..2])?, saving, "without the listing of `writes_rax`");
    assert_eq!(code(&[])?, saving, "without any");
    Ok(())
}

/// A listing says which vector registers its code writes, as it says the others, and whether it runs `vzeroupper`,
/// which clears the upper half of each vector register and leaves the lower half. A call changes the vector registers
/// that each function it may reach writes, and clears the upper half of each if one of them runs `vzeroupper`: a vector
/// alive across it stays in the frame if the kernel reads all of it after, and in a register if it reads the lower half
/// alone. Where a listing is missing, as it is for a function that does not compile, or for one that a function it
/// calls calls, the call may change every register a callee may, so the vector stays in the frame whatever the kernel
/// reads of it.
#[test]
fn a_call_clears_the_upper_halves_where_the_listings_of_its_callees_say() -> Result {
    let callees = "unsafe extern \"sysv64\" fn writes_ymm0() { let v: ymm; zero(v); }\n\
                   unsafe extern \"sysv64\" fn runs_vzeroupper() { let v: ymm; zero(v); vaddps(v, v, v); }\n\
                   unsafe extern \"sysv64\" fn calls_vzeroupper() { call(runs_vzeroupper); pause(); }\n";
    let ymm0 = PReg { class: Class::Vec, hw: 0 };
    let module = parse(callees)?;
    let mut done = vec![];
    for (name, changes, clears) in
        [("writes_ymm0", vec![ymm0], false), ("runs_vzeroupper", vec![ymm0], true), ("calls_vzeroupper", vec![], false)]
    {
        let listing = finished(&module, name, &done)?;
        assert_eq!((&listing.changes, listing.clears_upper), (&changes, clears), "{name}");
        done.push((name.to_string(), listing));
    }
    // Whether a kernel that calls `callee` keeps `v` in the frame, when it reads `read` of it after.
    let spills = |callee: &str, read: &str, done: &[(String, Listing)]| -> Result<_> {
        let k = format!(
            "unsafe extern \"sysv64\" fn k(p: *mut f32) {{ let v: ymm; vmovups(v, [p]); call({callee}); \
             vmovups([p], {read}); }}"
        );
        let module = parse(&format!("{callees}{k}"))?;
        Ok(finished(&module, "k", done)?.frame.slots.iter().any(|slot| slot.spill))
    };
    assert!(!spills("writes_ymm0", "v", &done)?);
    assert!(spills("runs_vzeroupper", "v", &done)?);
    assert!(!spills("runs_vzeroupper", "v.xmm", &done)?);
    assert!(spills("calls_vzeroupper", "v", &done)?);
    assert!(!spills("calls_vzeroupper", "v.xmm", &done)?);
    assert!(spills("writes_ymm0", "v.xmm", &[])?, "without the listing of `writes_ymm0`");
    assert!(spills("calls_vzeroupper", "v.xmm", &done[2..])?, "without that of `runs_vzeroupper`, which it calls");
    Ok(())
}

/// A listing names a constant `{c0}` and a function `{f0}`, and a static by its name, so no
/// static is named as a constant or a function is: the port refuses it at its name. It checks
/// the name before the type: `c0`, whose type holds nothing, is refused at its name.
#[test]
fn a_static_named_as_a_listing_names_a_constant_is_refused() -> Result {
    let module = parse("static c0: global<[u32; 0]>;\nstatic f12: global<u32>;")?;
    let cases = [
        "a kernel's listing names a constant `{c0}` and a function `{f0}`, and a static by its name, so one is not \
         named `c0`",
        "a kernel's listing names a constant `{c0}` and a function `{f0}`, and a static by its name, so one is not \
         named `f12`",
    ];
    for (item, want) in module.items.iter().zip(cases) {
        let ItemKind::Static(s) = &item.kind else { return Err("statics alone".into()) };
        let memory = kernite_lang::memory(&module, s)?;
        assert_eq!(X86.lay_out(&s.name, &memory), Err(StaticError::Name(want.into())));
    }
    Ok(())
}

/// A kernel calls no function that is `#[cooperative]`, whose threads run on a grid its
/// launcher starts. The port sees one as such, and the kernel that calls it is refused where it
/// names it.
#[test]
fn a_call_to_a_cooperative_function_is_refused() -> Result {
    let src = "#[cooperative]\n\
               unsafe extern \"sysv64\" fn together(x: *mut u64) {}\n\
               unsafe extern \"sysv64\" fn k(x: *mut u64) {\n\
               call(together, x);\n\
               }";
    let module = parse(src)?;
    let (together, k) = (
        function(&module, "together").ok_or("function gave None")?,
        function(&module, "k").ok_or("function gave None")?,
    );
    assert_eq!(X86.callee(together), Some(Callee::Cooperative));
    let callees =
        [Called { name: "together".into(), function: "together".into(), args: vec![], callee: Callee::Cooperative }];
    let mut kernel = X86.kernel(k, "k", &[], false, &[], &callees);
    let e = kernite_lang::kernel(&module, k, &[], &mut kernel).err().ok_or("kernel gave Ok")?;
    let want = "4:6: `together` is `#[cooperative]`: its threads run on a grid, which its launcher starts, so no kernel \
                calls it";
    assert_eq!(e.at(src), want);
    Ok(())
}

/// The port sees a function `extern "sysv64"` that takes what SysV does not pass whole in a
/// register as broken: an array, or a `u32`, whose high bits SysV leaves as they were. A
/// function of another ABI, `extern "ptx-func"`, or a helper, which names no ABI, is one its
/// kernels cannot name.
#[test]
fn the_port_sees_each_function_by_its_abi_and_its_types() -> Result {
    let src = "unsafe extern \"sysv64\" fn pair(a: [u64; 2]) {}\n\
               unsafe extern \"sysv64\" fn narrow(x: u32) {}\n\
               pub unsafe extern \"ptx-func\" fn far() {}\n\
               fn h() {}";
    let module = parse(src)?;
    let callee = |name| -> Result<_> { Ok(X86.callee(function(&module, name).ok_or("function gave None")?)) };
    assert_eq!(callee("pair")?, Some(Callee::Broken));
    assert_eq!(callee("narrow")?, Some(Callee::Broken));
    assert_eq!(callee("far")?, None);
    assert_eq!(callee("h")?, None);
    Ok(())
}

/// A frame of more than 2^31 bytes, past what an address reaches, is the kernel's to blame and
/// not a variable's: the port says why, and names no variable. Eighteen vectors alive at once
/// spill two, after stack memory of 2^31 - 63 bytes.
#[test]
fn a_frame_past_what_an_address_reaches_blames_no_variable() -> Result {
    let loads: String = (0..18).map(|i| format!("let v{i}: ymm;\nvmovups(v{i}, [x + {}]);\n", 32 * i)).collect();
    let sums: String = (1..18).map(|i| format!("vaddps(v0, v0, v{i});\n")).collect();
    let src = format!(
        "unsafe extern \"sysv64\" fn k(x: *mut f32) {{\nlet big: local<[u8; 2147483585], 32>;\n{loads}{sums}\
         vmovups([big], v0);\n}}"
    );
    let module = parse(&src)?;
    let k = function(&module, "k").ok_or("function gave None")?;
    let mut kernel = X86.kernel(k, "k", &[], false, &[], &[]);
    kernite_lang::kernel(&module, k, &[], &mut kernel)?;
    let message = "the kernel's frame, its stack memory and the variables that spilled, reaches 2147483680 bytes past \
                   the stack pointer: more than an address does";
    assert_eq!(X86.finish(kernel, &[]), Err(Unfinished { at: None, message: message.into() }));
    Ok(())
}

/// A listing says whether a call of it may run SSE code, which pays for an upper half set: its code calls a function it
/// does not know, at an address with no functions given, or calls one that may, by name or at an address with the
/// functions given. A call that passes all of a ymm or takes all of one back does not count: its callee runs AVX
/// instructions. A call of the function itself runs what its other calls do. A kernel is finished with the listings of
/// the functions it calls, and has `vzeroupper` before a call of one that may where it may have set an upper half, as
/// it does before a call at an address with no functions given. Where a listing is missing, as it is for a function
/// that does not compile, or for one that a function it calls calls, a call of it may.
#[test]
fn a_listing_says_whether_a_call_of_it_may_run_sse_code() -> Result {
    let src = "unsafe extern \"sysv64\" fn plain() { let t: gpr; mov(t, 1); add(t, t); }\n\
               unsafe extern \"sysv64\" fn plain_one(n: u64) { let t: gpr; mov(t, n); add(t, t); }\n\
               unsafe extern \"sysv64\" fn unknown(fp: u64) { call(fp, ()); }\n\
               unsafe extern \"sysv64\" fn via(fp: u64) { call(unknown, (fp)); }\n\
               unsafe extern \"sysv64\" fn given() { let q: gpr; lea(q, [plain]); call(q, (), (plain, plain)); }\n\
               unsafe extern \"sysv64\" fn given_via(fp: u64) {\n\
               let q: gpr; lea(q, [plain_one]); call(q, (fp), (plain_one, via)); }\n\
               unsafe extern \"sysv64\" fn ymm_to_unknown(fp: u64) { let v: ymm; zero(v); call(fp, (v)); }\n\
               unsafe extern \"sysv64\" fn itself(n: u64) { call(itself, (n)); }\n\
               unsafe extern \"sysv64\" fn itself_unknown(fp: u64) { call(itself_unknown, (fp)); call(fp, ()); }\n\
               unsafe extern \"sysv64\" fn k(p: *mut f32) {\n\
               let v: ymm; vmovups(v, [p]); vaddps(v, v, v); vmovups([p], v); call(plain); pause(); }";
    let module = parse(src)?;
    let mut done = vec![];
    for (name, may) in [
        ("plain", false),
        ("plain_one", false),
        ("unknown", true),
        ("via", true),
        ("given", false),
        ("given_via", true),
        ("ymm_to_unknown", false),
        ("itself", false),
        ("itself_unknown", true),
    ] {
        let listing = finished(&module, name, &done)?;
        assert_eq!(listing.reaches_sse, may, "{name}");
        done.push((name.to_string(), listing));
    }
    let code = |done: &[(String, Listing)]| -> Result<_> {
        Ok(finished(&module, "k", done)?.code().collect::<Vec<_>>().join("\n"))
    };
    let work = "sub rsp, 8\nvmovups ymm0, ymmword ptr [rdi]\nvaddps ymm0, ymm0, ymm0\nvmovups ymmword ptr [rdi], ymm0";
    assert_eq!(code(&done)?, format!("{work}\ncall {{f0}}\npause\nadd rsp, 8\nvzeroupper\nret"));
    assert_eq!(code(&[])?, format!("{work}\nvzeroupper\ncall {{f0}}\npause\nadd rsp, 8\nret"), "without the listing");
    Ok(())
}
