//! What kgen-build decides of a module's items, on modules the language parses. Each kernel in
//! them compiles, for its machine.

use kgen_lang::{IntTy, Value, parse};

use crate::{Owner, cooperative, function, instance, owner};

/// The statics of a module whose kernels are x86's are the process's; those of a module of PTX
/// kernels, or of none, the GPU's. A module of both kinds of kernel and a static is two modules,
/// which its first static says.
#[test]
fn a_module_of_x86_kernels_owns_its_statics() {
    let x86 = "unsafe extern \"sysv64\" fn k(x: u64) -> u64 { ret(x); }";
    let gpu = "unsafe extern \"ptx-kernel\" fn g(x: *mut u32) { st.global.u32([x], 1); }";
    let (s, t) = ("static S: global<u32>;", "static T: constant<u32> = 1;");
    let whose = |items: &[&str]| {
        let src = items.join("\n");
        owner(&parse(&src).unwrap()).map_err(|e| e.at(&src))
    };
    assert_eq!(whose(&[x86, s, t]), Ok(Owner::Process));
    assert_eq!(whose(&[gpu, s]), Ok(Owner::Gpu));
    assert_eq!(whose(&[s]), Ok(Owner::Gpu));
    // Both kinds, and no static for the x86 kernels to miss.
    assert_eq!(whose(&[x86, gpu]), Ok(Owner::Gpu));
    let two = "a static is memory of the module's kernels: the GPU's, for PTX kernels, or the process's, for x86 \
               ones; and this module has both, so it is two modules, one of each kind of kernel and the statics they \
               name";
    assert_eq!(whose(&[x86, t, gpu, s]), Err(format!("2:8: {two}")));
}

/// `#[cooperative]` goes on a kernel alone, of either kind: a helper, or a function the kernels
/// call, that says it is refused where it says so. A function of an ABI kgen knows nothing of is
/// no kernel, whatever it says, and one that says nothing is not cooperative.
#[test]
fn a_cooperative_helper_is_refused() {
    let src = "#[cooperative] unsafe extern \"ptx-kernel\" fn k() { grid.sync(); }\n\
               #[cooperative] unsafe extern \"sysv64\" fn x() { grid.sync(); ret(); }\n\
               #[cooperative] fn h() {}\n\
               #[cooperative] extern \"ptx-func\" fn f() {}\n\
               #[cooperative] extern \"C\" fn c() {}\n\
               unsafe extern \"ptx-kernel\" fn g(x: *mut u32) { st.global.u32([x], 1); }";
    let module = parse(src).unwrap();
    let said = |name| cooperative(function(&module, name).unwrap()).map_err(|e| e.at(src));
    let not = "`#[cooperative]` goes on a kernel whose blocks or threads wait for each other, `extern \"ptx-kernel\"` \
               or `extern \"sysv64\"`";
    assert_eq!(said("k"), Ok(true));
    assert_eq!(said("x"), Ok(true));
    assert_eq!(said("h"), Err(format!("3:3: {not}; `h` is a helper, inlined where it is called")));
    let called = "a function the kernels call, which runs as the kernel that calls it is launched";
    assert_eq!(said("f"), Err(format!("4:3: {not}; `f` is {called}")));
    assert_eq!(said("c"), Ok(false));
    assert_eq!(said("g"), Ok(false));
}

/// A name names a function that takes no compile-time arguments, or an instance of one, as
/// `tile::<4>` names `tile` for 4. A name of no function, or of no instance its function has,
/// names none.
#[test]
fn an_instance_is_found_by_its_name() {
    let tile =
        "#[instances(2, 4)] unsafe extern \"sysv64\" fn tile<const N: u64>(x: u64) -> u64 { add(x, N); ret(x); }";
    let src = format!("{tile}\nunsafe extern \"sysv64\" fn k(x: u64) -> u64 {{ ret(x); }}");
    let module = parse(&src).unwrap();
    let found = |name| instance(&module, name).unwrap().map(|(f, args)| (f.name.as_str(), args));
    assert_eq!(found("tile::<4>"), Some(("tile", vec![Value::Int(4, Some(IntTy::U64))])));
    assert_eq!(found("k"), Some(("k", vec![])));
    for none in ["tile::<3>", "tile", "k::<4>", "j"] {
        assert_eq!(found(none), None, "{none}");
    }
}
