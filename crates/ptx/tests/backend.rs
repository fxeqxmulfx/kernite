//! `Ptx` through the ports kgen-build calls, driven by hand as a use case drives them: each
//! static laid out after those before it and each callee seen, then each kernel told of them,
//! written and finished, and the module of them all.

use kgen_build::{Backend, Called, Device, DeviceStatic, StaticError, Unfinished};
use kgen_lang::{ItemKind, Module};
use kgen_ptx::lang::{Callee, Kernel};
use kgen_ptx::{Prototype, Ptx, Returns, Static, Ty};

/// Static `name` of `module`, laid out by `ptx` after `before`.
fn lay_out(ptx: &Ptx, module: &Module, name: &str, before: &[Static]) -> Result<Static, StaticError> {
    let s = module.items.iter().find_map(|item| match &item.kind {
        ItemKind::Static(s) if s.name == name => Some(s),
        _ => None,
    });
    ptx.lay_out(name, &kgen_lang::memory(module, s.unwrap()).unwrap(), before)
}

/// The kernel `ptx` makes of `name`, a function of `module` or an instance of one, told of
/// `statics` and `callees`, with its body written, for `finish`.
fn written(ptx: &Ptx, module: &Module, name: &str, statics: &[Static], callees: &[Called<Callee>]) -> Kernel {
    let (f, args) = kgen_build::instance(module, name).unwrap().unwrap();
    let mut kernel = ptx.kernel(f, name, &args, kgen_build::cooperative(f).unwrap(), statics, callees);
    kgen_lang::kernel(module, f, &args, &mut kernel).unwrap();
    kernel
}

/// A `#[cooperative]` kernel that names memory of its module comes out of the port as the
/// driver loads it: one module of the static, holding what it says, and the kernel, for `sm_60`
/// as a cooperative launch needs. Each is named by its symbol in the scope, taken as it is
/// given, here the namespaces `kgen::gpu`. The host finds the static by its symbol and reads its
/// `b32`s as `u32`s.
#[test]
fn ptx_compiles_a_kernel_through_the_port() {
    let src = r#"
        static COUNTS: global<[b32; 4]> = [1, 2, 3, 4];

        #[cooperative]
        pub unsafe extern "ptx-kernel" fn count(x: u32) {
            red.global.add.u32([COUNTS + 4], x);
        }
    "#;
    let module = kgen_lang::parse(src).unwrap();
    let ptx = Ptx::new("4kgen3gpu");
    let counts = lay_out(&ptx, &module, "COUNTS", &[]).unwrap();
    let count = ptx.finish(written(&ptx, &module, "count", std::slice::from_ref(&counts), &[])).unwrap();
    assert_eq!(ptx.symbol(&count), "_ZN4kgen3gpu5countE");
    let host = DeviceStatic { symbol: "_ZN4kgen3gpu6COUNTSE".into(), host: "u32".into() };
    assert_eq!(ptx.device_static(&counts), host);
    let want = "// Written by kgen.

.version 5.0
.target sm_60
.address_size 64

.global .align 4 .b8 _ZN4kgen3gpu6COUNTSE[16] = {1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4};

.visible .entry _ZN4kgen3gpu5countE(
\t.param .u32 _ZN4kgen3gpu5countE_param_0
)
{
\t.reg .u32 %x;

\tld.param.u32 %x, [_ZN4kgen3gpu5countE_param_0];
\tred.global.add.u32 [_ZN4kgen3gpu6COUNTSE+4], %x;
\tret;
}
";
    assert_eq!(ptx.module(&[count], &[counts]).unwrap(), want);
}

/// A module has 64 KB of constant memory, so a static that fits alone is refused at its type
/// after statics that leave it too little: the port is told of those laid out before it. What a
/// static holds is checked before the room left: one wrong in both is refused at its value.
#[test]
fn a_static_that_does_not_fit_beside_those_before_it_is_refused() {
    let src = "
        static A: constant<[u8; 65535]>;
        static B: constant<[u32; 1]>;
        static C: constant<[u32; 2]> = [1, 2, 3];
    ";
    let module = kgen_lang::parse(src).unwrap();
    let ptx = Ptx::new("");
    let a = lay_out(&ptx, &module, "A", &[]).unwrap();
    lay_out(&ptx, &module, "B", &[]).unwrap();
    let full = "this makes the module's constant memory 65540 bytes, and a module has 64 KB";
    assert_eq!(lay_out(&ptx, &module, "B", std::slice::from_ref(&a)), Err(StaticError::Type(full.into())));
    let three = StaticError::Value("`constant<[u32; 2]>` holds 2 elements; this has 3".into());
    assert_eq!(lay_out(&ptx, &module, "C", &[a]), Err(three));
}

/// A function `extern "ptx-func"` is a `.func` of the module, which a kernel calls by its
/// symbol, as its prototype says. An instance, `times::<2>` or `k::<1>`, is named in PTX as C++
/// names a template's instance, and in messages as Rust writes it. The port sees a kernel as one
/// a kernel takes the address of, for the GPU to launch it. It sees a function that takes a
/// `pred` or an array, which PTX does not pass, as broken, and a function of the process as none.
#[test]
fn a_function_the_kernels_call_is_a_func() {
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
    let module = kgen_lang::parse(src).unwrap();
    let ptx = Ptx::new("4kgen3gpu");
    let callee = |name| ptx.callee(kgen_build::function(&module, name).unwrap());
    let calls = Callee::Calls(Prototype { params: vec![Ty::U32], returns: Returns::Value(Ty::U32) });
    assert_eq!(callee("times"), Some(calls.clone()));
    assert_eq!(callee("flag"), Some(Callee::Broken));
    assert_eq!(callee("pair"), Some(Callee::Broken));
    assert_eq!(callee("k"), Some(Callee::Kernel));
    assert_eq!(callee("nothing"), None);
    let (_, args) = kgen_build::instance(&module, "times::<2>").unwrap().unwrap();
    let called = Called { name: "times::<2>".into(), function: "times".into(), args, callee: calls };
    let times = ptx.finish(written(&ptx, &module, "times::<2>", &[], &[])).unwrap();
    let k = ptx.finish(written(&ptx, &module, "k::<1>", &[], &[called])).unwrap();
    assert_eq!([&times.name, &k.name], ["times::<2>", "k::<1>"]);
    let want = "// Written by kgen.

.version 2.3
.target sm_20
.address_size 64

.func (.param .u32 _ZN4kgen3gpu5timesILj2EEE_retval) _ZN4kgen3gpu5timesILj2EEE(
\t.param .u32 _ZN4kgen3gpu5timesILj2EEE_param_0
)
{
\t.reg .u32 %x, %y;

\tld.param.u32 %x, [_ZN4kgen3gpu5timesILj2EEE_param_0];
\tmul.lo.u32 %y, %x, 2;
\tst.param.u32 [_ZN4kgen3gpu5timesILj2EEE_retval], %y;
\tret;
}

.visible .entry _ZN4kgen3gpu1kILj1EEE(
\t.param .u64 _ZN4kgen3gpu1kILj1EEE_param_0
)
{
\t.reg .u32 %x;
\t.reg .u64 %p;

\tld.param.u64 %p, [_ZN4kgen3gpu1kILj1EEE_param_0];
\tcvta.to.global.u64 %p, %p;
\tld.global.u32 %x, [%p];
\tcall (%x), _ZN4kgen3gpu5timesILj2EEE, (%x);
\tshl.b32 %x, %x, 1;
\tst.global.u32 [%p], %x;
\tret;
}
";
    assert_eq!(ptx.module(&[times, k], &[]).unwrap(), want);
}

/// Each refusal says what to blame, for kgen-build to place it: the part of a static that is
/// wrong, or the variable a kernel reads before anything writes it.
#[test]
fn each_refusal_says_what_to_blame() {
    let src = r#"
        static EMPTY: global<[u32; 0]>;
        static BIG: global<s8> = 128;

        pub unsafe extern "ptx-kernel" fn unwritten(p: *mut u32) {
            let x: u32;
            st.global.u32([p], x);
        }
    "#;
    let module = kgen_lang::parse(src).unwrap();
    let ptx = Ptx::new("");
    let empty = StaticError::Type("`global<[u32; 0]>` holds nothing".into());
    assert_eq!(lay_out(&ptx, &module, "EMPTY", &[]), Err(empty));
    let big = StaticError::Value("this is 128, out of range for `s8`".into());
    assert_eq!(lay_out(&ptx, &module, "BIG", &[]), Err(big));
    let unfinished = Unfinished { var: Some("x".into()), message: "`x` is read before anything writes it".into() };
    assert_eq!(ptx.finish(written(&ptx, &module, "unwritten", &[], &[])), Err(unfinished));
}
