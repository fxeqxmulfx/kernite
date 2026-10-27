//! `X86` through the ports kgen-build calls, driven by hand as a use case drives them: each
//! static laid out and each callee seen, then the kernel told of them, written and finished.

use kgen_build::{Backend, Bytes, Called, HostStatic, Naked, Native, StaticError, Unfinished, function};
use kgen_lang::{ItemKind, parse};
use kgen_x86::X86;
use kgen_x86::lang::Callee;

/// A kernel that names memory of its module and calls a function of it comes out of the port as
/// Rust takes it: its code as `naked_asm!` reads it, its listing, the CPU features it needs, the
/// constants it reads, and the function and the static it names. The static comes out as Rust
/// lays it out, holding what it says.
#[test]
fn x86_compiles_a_kernel_through_the_port() {
    let src = "static COUNTS: global<[u32; 2]> = [1, 2];\n\
               unsafe extern \"sysv64\" fn half(x: f32) -> f32 { ret(x); }\n\
               unsafe extern \"sysv64\" fn k(out: *mut f32) {\n\
               let v: ymm;\n\
               call(v.xmm, half, 2.5);\n\
               vmovss([out], v.xmm);\n\
               add(dword[COUNTS + 4], 1);\n\
               }";
    let module = parse(src).unwrap();
    let ItemKind::Static(counts) = &module.items[0].kind else { panic!("a static first") };
    let (half, k) = (function(&module, "half").unwrap(), function(&module, "k").unwrap());
    let statics = [X86.lay_out("COUNTS", &kgen_lang::memory(&module, counts).unwrap(), &[]).unwrap()];
    let callee = X86.callee(half).unwrap();
    let callees = [Called { name: "half".into(), function: "half".into(), args: vec![], callee }];
    let mut kernel = X86.kernel(k, "k", &[], false, &statics, &callees);
    kgen_lang::kernel(&module, k, &[], &mut kernel).unwrap();
    let code = X86.finish(kernel).unwrap();
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
    vzeroupper
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
        "vzeroupper",
        "ret",
    ];
    let want = Naked {
        code: lines.map(String::from).to_vec(),
        text: text.into(),
        needs: vec!["avx".into()],
        consts: vec![Bytes { bytes: 2.5f32.to_le_bytes().to_vec(), align: 4 }],
        funcs: vec!["half".into()],
        statics: vec!["COUNTS".into()],
    };
    assert_eq!(X86.naked(&code), want);
    let want = HostStatic { elem: "u32".into(), size: 4, len: 2, align: 4, init: vec![1, 0, 0, 0, 2, 0, 0, 0] };
    assert_eq!(X86.host_static(&statics[0]), want);
}

/// A listing names a constant `{c0}` and a function `{f0}`, and a static by its name, so no
/// static is named as a constant or a function is: the port refuses it at its name. It checks
/// the name before the type: `c0`, whose type holds nothing, is refused at its name.
#[test]
fn a_static_named_as_a_listing_names_a_constant_is_refused() {
    let module = parse("static c0: global<[u32; 0]>;\nstatic f12: global<u32>;").unwrap();
    let cases = [
        "a kernel's listing names a constant `{c0}` and a function `{f0}`, and a static by its name, so one is not \
         named `c0`",
        "a kernel's listing names a constant `{c0}` and a function `{f0}`, and a static by its name, so one is not \
         named `f12`",
    ];
    for (item, want) in module.items.iter().zip(cases) {
        let ItemKind::Static(s) = &item.kind else { panic!("statics alone") };
        let memory = kgen_lang::memory(&module, s).unwrap();
        assert_eq!(X86.lay_out(&s.name, &memory, &[]), Err(StaticError::Name(want.into())));
    }
}

/// A kernel calls no function that is `#[cooperative]`, whose threads run on a grid its
/// launcher starts. The port sees one as such, and the kernel that calls it is refused where it
/// names it.
#[test]
fn a_call_to_a_cooperative_function_is_refused() {
    let src = "#[cooperative]\n\
               unsafe extern \"sysv64\" fn together(x: *mut u64) {}\n\
               unsafe extern \"sysv64\" fn k(x: *mut u64) {\n\
               call(together, x);\n\
               }";
    let module = parse(src).unwrap();
    let (together, k) = (function(&module, "together").unwrap(), function(&module, "k").unwrap());
    assert_eq!(X86.callee(together), Some(Callee::Cooperative));
    let callees =
        [Called { name: "together".into(), function: "together".into(), args: vec![], callee: Callee::Cooperative }];
    let mut kernel = X86.kernel(k, "k", &[], false, &[], &callees);
    let e = kgen_lang::kernel(&module, k, &[], &mut kernel).unwrap_err();
    let want = "4:6: `together` is `#[cooperative]`: its threads run on a grid, which its launcher starts, so no kernel \
                calls it";
    assert_eq!(e.at(src), want);
}

/// The port sees a function `extern "sysv64"` that takes what SysV does not pass whole in a
/// register as broken: an array, or a `u32`, whose high bits SysV leaves as they were. A
/// function of another ABI, `extern "ptx-func"`, or a helper, which names no ABI, is one its
/// kernels cannot name.
#[test]
fn the_port_sees_each_function_by_its_abi_and_its_types() {
    let src = "unsafe extern \"sysv64\" fn pair(a: [u64; 2]) {}\n\
               unsafe extern \"sysv64\" fn narrow(x: u32) {}\n\
               pub unsafe extern \"ptx-func\" fn far() {}\n\
               fn h() {}";
    let module = parse(src).unwrap();
    let callee = |name| X86.callee(function(&module, name).unwrap());
    assert_eq!(callee("pair"), Some(Callee::Broken));
    assert_eq!(callee("narrow"), Some(Callee::Broken));
    assert_eq!(callee("far"), None);
    assert_eq!(callee("h"), None);
}

/// A frame of more than 2^31 bytes, past what an address reaches, is the kernel's to blame and
/// not a variable's: the port says why, and names no variable. Seventeen vectors alive at once
/// spill two, after stack memory of 2^31 - 63 bytes.
#[test]
fn a_frame_past_what_an_address_reaches_blames_no_variable() {
    let loads: String = (0..17).map(|i| format!("let v{i}: ymm;\nvmovups(v{i}, [x + {}]);\n", 32 * i)).collect();
    let sums: String = (1..17).map(|i| format!("vaddps(v0, v0, v{i});\n")).collect();
    let src = format!(
        "unsafe extern \"sysv64\" fn k(x: *mut f32) {{\nlet big: local<[u8; 2147483585], 32>;\n{loads}{sums}\
         vmovups([big], v0);\n}}"
    );
    let module = parse(&src).unwrap();
    let k = function(&module, "k").unwrap();
    let mut kernel = X86.kernel(k, "k", &[], false, &[], &[]);
    kgen_lang::kernel(&module, k, &[], &mut kernel).unwrap();
    let message = "the kernel's frame, its stack memory and the variables that spilled, is 2147483680 bytes: more \
                   than an address reaches past the stack pointer";
    assert_eq!(X86.finish(kernel), Err(Unfinished { var: None, message: message.into() }));
}
