//! `Ptx` through the ports kernite-build calls, driven by hand as a use case drives them: each
//! static laid out after those before it and each callee seen, then each kernel told of them,
//! written and finished, and the module of them all.

use kernite_build::{Backend, Called, Device, DeviceStatic, StaticError, Unfinished};
use kernite_lang::{Fault, ItemKind, Mem, Module, Operand, Span, Target};
use kernite_ptx::lang::{Callee, Kernel};
use kernite_ptx::{Prototype, Ptx, Returns, Static, Ty};

type Result<T = (), E = Box<dyn std::error::Error + Send + Sync>> = std::result::Result<T, E>;

/// Static `name` of `module`, laid out by `ptx` beside the statics `before` it.
fn lay_out(ptx: &Ptx, module: &Module, name: &str, before: &[Static]) -> Result<Result<Static, StaticError>> {
    let s = module.items.iter().find_map(|item| match &item.kind {
        ItemKind::Static(s) if s.name == name => Some(s),
        _ => None,
    });
    Ok(ptx.lay_out(name, &kernite_lang::memory(module, s.ok_or("find_map gave None")?)?, before))
}

/// The kernel `ptx` makes of `name`, a function of `module` or an instance of one, told of
/// `statics` and `callees`, with its body written, for `finish`.
fn written(ptx: &Ptx, module: &Module, name: &str, statics: &[Static], callees: &[Called<Callee>]) -> Result<Kernel> {
    let (f, args, _) = kernite_build::instance(module, name)?.ok_or("unwrap gave None")?;
    let mut kernel = ptx.kernel(f, name, &args, kernite_build::cooperative(f)?, statics, callees);
    kernite_lang::kernel(module, f, &args, &mut kernel)?;
    Ok(kernel)
}

/// A `#[cooperative]` kernel that names memory of its module comes out of the port as the
/// driver loads it: one module of the static, holding what it says, and the kernel, for `sm_60`
/// as a cooperative launch needs. Each is named by its symbol in the scope, taken as it is
/// given, here the namespaces `kernite::gpu`. The host finds the static by its symbol and reads its
/// `b32`s as `u32`s.
#[test]
fn ptx_compiles_a_kernel_through_the_port() -> Result {
    let src = r#"
        static COUNTS: global<[b32; 4]> = [1, 2, 3, 4];

        #[cooperative]
        pub unsafe extern "ptx-kernel" fn count(x: u32) {
            red.global.add.u32([COUNTS + 4], x);
        }
    "#;
    let module = kernite_lang::parse(src)?;
    let ptx = Ptx::new("7kernite3gpu");
    let counts = lay_out(&ptx, &module, "COUNTS", &[])?.map_err(|e| format!("{e:?}"))?;
    let count = ptx
        .finish(written(&ptx, &module, "count", std::slice::from_ref(&counts), &[])?, &[])
        .map_err(|e| format!("{e:?}"))?;
    assert_eq!(ptx.symbol(&count), "_ZN7kernite3gpu5countE");
    let host = DeviceStatic {
        symbol: "_ZN7kernite3gpu6COUNTSE".into(),
        ty: "global<[b32; 4]>".into(),
        host: "u32".into(),
        outside: None,
    };
    assert_eq!(ptx.device_static(&counts), host);
    let want = "// Written by kernite.

.version 5.0
.target sm_60
.address_size 64

.global .align 4 .b8 _ZN7kernite3gpu6COUNTSE[16] = {1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4};

// A warp of 32 threads moves, in sectors of 32 bytes:
//   red.global.add.u32 [_ZN7kernite3gpu6COUNTSE+4], %x: not counted, as it reaches memory that the module declares, \
which kernite does not follow yet
.visible .entry _ZN7kernite3gpu5countE(
\t.param .u32 _ZN7kernite3gpu5countE_param_0
)
{
\t.reg .u32 %x;

\tld.param.u32 %x, [_ZN7kernite3gpu5countE_param_0];
\tred.global.add.u32 [_ZN7kernite3gpu6COUNTSE+4], %x;
\tret;
}
";
    assert_eq!(ptx.module(&[count], &[counts]).whole()?, want);
    Ok(())
}

/// A module has 64 KB of constant memory, so a static that fits alone has no room after statics
/// that leave it too little: the port says so of the statics it laid out, in order, and a static
/// that has none takes none, so `D` fits where `B` did not. What a static holds is checked when it
/// is laid out, before the room left: one wrong in both is refused at its value.
#[test]
fn a_static_that_does_not_fit_beside_those_before_it_has_no_room() -> Result {
    let src = "
        static A: constant<[u8; 65535]>;
        static B: constant<[u32; 1]>;
        static C: constant<[u32; 2]> = [1, 2, 3];
        static D: constant<[u8; 1]>;
    ";
    let module = kernite_lang::parse(src)?;
    let ptx = Ptx::new("");
    let three = StaticError::Value("`constant<[u32; 2]>` holds 2 elements; this has 3".into());
    assert_eq!(lay_out(&ptx, &module, "C", &[])?, Err(three));
    let mut laid = vec![];
    for name in ["A", "B", "D"] {
        laid.push(lay_out(&ptx, &module, name, &[])?.map_err(|e| format!("{e:?}"))?);
    }
    let full = "this makes the module's constant memory 65540 bytes, and a module has 64 KB";
    assert_eq!(ptx.room(&laid), [None, Some(full.into()), None]);
    Ok(())
}

/// A static that makes a module's global memory 4 GB beside those before it that were laid out is outside the module:
/// the port says so by its `device_static`, whose type is the constant memory that holds its address, `constant<u64>`,
/// and whose `outside` is the bytes the driver allocates; and the slot takes 8 bytes of the 64 KB, which `room` counts.
/// The driver allocates zeros, so a value is refused, at the static's alignment, whatever it is.
#[test]
fn a_static_the_module_has_no_room_for_is_outside_it() -> Result {
    let src = "
        static A: global<[u8; 4294967295]>;
        static B: global<[u32; 4]>;
        static C: global<u8> = 1;
        static D: global<[u8; 4], 512>;
        static E: constant<[u8; 65528]>;
    ";
    let module = kernite_lang::parse(src)?;
    let ptx = Ptx::new("");
    let a = lay_out(&ptx, &module, "A", &[])?.map_err(|e| format!("{e:?}"))?;
    let b = lay_out(&ptx, &module, "B", std::slice::from_ref(&a))?.map_err(|e| format!("{e:?}"))?;
    let inside =
        DeviceStatic { symbol: "A".into(), ty: "global<[u8; 4294967295]>".into(), host: "u8".into(), outside: None };
    assert_eq!(ptx.device_static(&a), inside);
    let outside =
        DeviceStatic { symbol: "B".into(), ty: "constant<u64>".into(), host: "u32".into(), outside: Some((16, 4)) };
    assert_eq!(ptx.device_static(&b), outside);
    let laid = [a.clone(), b.clone()];
    let value = "the module has no room for `global<u8>` in its global memory, so the driver allocates it, and it \
                 holds zeros: the host writes what it should hold";
    assert_eq!(lay_out(&ptx, &module, "C", &laid)?, Err(StaticError::Value(value.into())));
    let d = lay_out(&ptx, &module, "D", &laid)?.map_err(|e| format!("{e:?}"))?;
    assert_eq!(ptx.device_static(&d).outside, Some((4, 512)));
    // With none before it, `B` is in the module.
    let b_alone = lay_out(&ptx, &module, "B", &[])?.map_err(|e| format!("{e:?}"))?;
    assert_eq!(ptx.device_static(&b_alone).outside, None);
    // A slot is 8 bytes of constant memory at a multiple of 8: it fits after 65528 bytes, and not after 65529.
    let e = lay_out(&ptx, &module, "E", &[])?.map_err(|e| format!("{e:?}"))?;
    assert_eq!(ptx.room(&[e, b.clone()]), [None, None]);
    let e = Static::new("E", "E", "constant<[u8; 65529]>")?;
    let full = "this makes the module's constant memory 65544 bytes, and a module has 64 KB";
    assert_eq!(ptx.room(&[e, b]), [None, Some(full.into())]);
    Ok(())
}

/// A function `extern "ptx-func"` is a `.func` of the module, which a kernel calls by its
/// symbol, as its prototype says. An instance, `times::<2>` or `k::<1>`, is named in PTX as C++
/// names a template's instance, and in messages as Rust writes it. The port sees a kernel as one
/// a kernel takes the address of, for the GPU to launch it. It sees a function that takes a
/// `pred` or an array, which PTX does not pass, as broken, and a function of the process as none.
#[test]
fn a_function_the_kernels_call_is_a_func() -> Result {
    let src = r#"
        #[instances(2)]
        pub unsafe extern "ptx-func" fn times<const N: u32>(x: u32) -> u32 {
            let y: u32;
            mul.lo.u32(y, x, N);
            ret(y);
        }

        #[instances(1)]
        pub unsafe extern "ptx-kernel" fn k<const S: u32>(p: *mut u32) {
            let x: u32;
            ld.global.u32(x, [p]);
            call(x, times::<2>, x);
            shl.b32(x, x, S);
            st.global.u32([p], x);
        }

        pub unsafe extern "ptx-func" fn flag(p: pred) {}

        pub unsafe extern "ptx-func" fn pair(a: [u32; 2]) {}

        unsafe extern "sysv64" fn nothing() {}
    "#;
    let module = kernite_lang::parse(src)?;
    let ptx = Ptx::new("7kernite3gpu");
    // A function is seen as one of its instances is, by the types of its parameters worked out.
    let callee = |name| -> Result<_> {
        let f = kernite_build::function(&module, name).ok_or("function gave None")?;
        let args = kernite_lang::instances(&module, f)?.into_iter().next().unwrap_or_default();
        Ok(ptx.callee(f, &kernite_lang::param_types(&module, f, &args)?))
    };
    let calls = Callee::Calls(Prototype { params: vec![Ty::U32], returns: Returns::Value(Ty::U32) });
    assert_eq!(callee("times")?, Some(calls.clone()));
    assert_eq!(callee("flag")?, Some(Callee::Broken));
    assert_eq!(callee("pair")?, Some(Callee::Broken));
    assert_eq!(callee("k")?, Some(Callee::Kernel));
    assert_eq!(callee("nothing")?, None);
    let (_, args, _) = kernite_build::instance(&module, "times::<2>")?.ok_or("unwrap gave None")?;
    let called = Called { name: "times::<2>".into(), function: "times".into(), args, callee: calls };
    let times = ptx.finish(written(&ptx, &module, "times::<2>", &[], &[])?, &[]).map_err(|e| format!("{e:?}"))?;
    let k = ptx.finish(written(&ptx, &module, "k::<1>", &[], &[called])?, &[]).map_err(|e| format!("{e:?}"))?;
    assert_eq!([&times.name, &k.name], ["times::<2>", "k::<1>"]);
    let want = "// Written by kernite.

.version 3.0
.target sm_30
.address_size 64

.func (.param .u32 _ZN7kernite3gpu5timesILj2EEE_retval) _ZN7kernite3gpu5timesILj2EEE(
\t.param .u32 _ZN7kernite3gpu5timesILj2EEE_param_0
)
{
\t.reg .u32 %x, %y;

\tld.param.u32 %x, [_ZN7kernite3gpu5timesILj2EEE_param_0];
\tmul.lo.u32 %y, %x, 2;
\tst.param.u32 [_ZN7kernite3gpu5timesILj2EEE_retval], %y;
\tret;
}

// A warp of 32 threads moves, in sectors of 32 bytes:
//   ld.global.u32 %x, [%p]: 1 sector for 4 bytes, an eighth used
//   st.global.u32 [%p], %x: 1 sector for 4 bytes, an eighth used
//   p: each byte is read at most once by every thread
.visible .entry _ZN7kernite3gpu1kILj1EEE(
\t.param .u64 _ZN7kernite3gpu1kILj1EEE_param_0
)
{
\t.reg .u32 %x;
\t.reg .u64 %p;

\tld.param.u64 %p, [_ZN7kernite3gpu1kILj1EEE_param_0];
\tcvta.to.global.u64 %p, %p;
\tld.global.u32 %x, [%p];
\tcall (%x), _ZN7kernite3gpu5timesILj2EEE, (%x);
\tshl.b32 %x, %x, 1;
\tst.global.u32 [%p], %x;
\tret;
}
";
    assert_eq!(ptx.module(&[times, k], &[]).whole()?, want);
    Ok(())
}

/// Each refusal says what to blame, for kernite-build to place it: the part of a static that is
/// wrong, or the variable a kernel reads before anything writes it.
#[test]
fn each_refusal_says_what_to_blame() -> Result {
    let src = r#"
        static EMPTY: global<[u32; 0]>;
        static BIG: global<s8> = 128;

        pub unsafe extern "ptx-kernel" fn unwritten(p: *mut u32) {
            let x: u32;
            st.global.u32([p], x);
        }
    "#;
    let module = kernite_lang::parse(src)?;
    let ptx = Ptx::new("");
    let empty = StaticError::Type("`global<[u32; 0]>` holds nothing".into());
    assert_eq!(lay_out(&ptx, &module, "EMPTY", &[])?, Err(empty));
    let big = StaticError::Value("this is 128, out of range for `s8`".into());
    assert_eq!(lay_out(&ptx, &module, "BIG", &[])?, Err(big));
    // Where `x` is declared: a byte of the source, as a string's spans are.
    let at = Some(Span(src.find("x: u32").ok_or("find gave None")? as u32));
    let unfinished = Unfinished { at, message: "`x` is read before anything writes it".into() };
    assert_eq!(ptx.finish(written(&ptx, &module, "unwritten", &[], &[])?, &[]), Err(unfinished));
    Ok(())
}

/// A mask of lanes that leaves out lane 0, on an instruction that thread 0 runs, is refused when
/// the kernel is finished, and says where the instruction is, for kernite-build to place it there;
/// under a guard it is taken.
#[test]
fn a_mask_without_lane_0_is_refused_at_its_instruction() -> Result {
    let src = r#"
        pub unsafe extern "ptx-kernel" fn masked(n: u32) {
            bar.warp.sync(2);
        }

        pub unsafe extern "ptx-kernel" fn guarded(n: u32) {
            let p: pred;
            setp.eq.u32(p, n, 0);
            if p {
                bar.warp.sync(2);
            }
        }
    "#;
    let module = kernite_lang::parse(src)?;
    let ptx = Ptx::new("");
    // At the last word of the instruction's name, as an error of the instruction's is.
    let at = Some(Span(src.find("sync(2)").ok_or("find gave None")? as u32));
    let message = "`bar.warp.sync` is run by thread 0, which is in lane 0; this mask leaves lane 0 out, and PTX \
                   leaves the instruction undefined in a thread outside its mask";
    let unfinished = Unfinished { at, message: message.into() };
    assert_eq!(ptx.finish(written(&ptx, &module, "masked", &[], &[])?, &[]), Err(unfinished));
    assert!(ptx.finish(written(&ptx, &module, "guarded", &[], &[])?, &[]).is_ok());
    Ok(())
}

/// The instruction `st.global.u32` over an `f32` variable, which the target refuses.
fn store_an_f32_as_a_u32() -> Result<(), Fault> {
    let mut kernel = Kernel::new("k", "k");
    let out = kernel.param("out", "*mut f32").map_err(Fault::new)?;
    let x = Operand::Var(kernel.var("x", "f32").map_err(Fault::new)?);
    kernel.inst("st.global.u32", &[Operand::Mem(Mem { base: Some(out), index: None, disp: 0 }), x], None)
}

/// A refusal is an error: `?` takes a `Fault` into a `Box<dyn Error>`, and keeps it whole.
#[test]
fn question_mark_takes_a_fault_into_a_boxed_error() -> Result {
    fn taken() -> Result<(), Box<dyn std::error::Error>> {
        let mut kernel = Kernel::new("k", "k");
        kernel.inst("ret", &[], None)?;
        Ok(())
    }
    fn refused() -> Result<(), Box<dyn std::error::Error>> {
        store_an_f32_as_a_u32()?;
        Ok(())
    }
    taken().map_err(|e| e.to_string())?;
    let fault = store_an_f32_as_a_u32().err().ok_or("store_an_f32_as_a_u32 gave Ok")?;
    assert_eq!(refused().err().ok_or("refused gave Ok")?.downcast_ref::<Fault>(), Some(&fault));
    Ok(())
}

/// A refusal shown with `{}` says what is wrong, as its `message` does.
#[test]
fn a_fault_shown_says_what_is_wrong() -> Result {
    let fault = store_an_f32_as_a_u32().err().ok_or("store_an_f32_as_a_u32 gave Ok")?;
    assert_eq!(format!("{fault}"), "`st.global.u32` wants a `u32` variable; this is an `f32`");
    assert_eq!(format!("{fault}"), fault.message);
    Ok(())
}
