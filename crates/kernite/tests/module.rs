//! A module through the façades: `x86_str` compiles a kernel the process runs and `ptx_str` the
//! functions the GPU runs, and once that has gone well each refuses what is wrong with the rest
//! of the module, as the macro does.

/// Kernel `name` of `src`, for x86: the first line of its listing.
fn x86(src: &str, name: &str) -> String {
    let listing = kernite::x86_str(src, name).unwrap_or_else(|e| panic!("{e}")).text();
    listing.lines().next().unwrap_or_default().to_string()
}

/// Whether `src` compiles for PTX, each of its kernels as an `.entry`.
fn compiles_for_ptx(src: &str) -> bool {
    kernite::ptx_str(src).unwrap_or_else(|e| panic!("{e}")).contains(".visible .entry ")
}

/// What kernite says of a function of ABI `C`, which it compiles nothing for, at the ABI.
const ABI_C: &str = "kernite compiles `extern \"sysv64\"` kernels, for x86-64, and `extern \"ptx-kernel\"` ones, for \
                     NVIDIA GPUs, with the `extern \"ptx-func\"` functions they call; `extern \"C\"` is none of these";

#[test]
fn x86_str_compiles_no_gpu_kernel() {
    let src = "pub unsafe extern \"sysv64\" fn k() {}\n\
               pub unsafe extern \"ptx-kernel\" fn store(x: *mut u32) { st.global.u32([x], 1); }\n\
               pub extern \"ptx-func\" fn func() {}";
    assert_eq!(x86(src, "k"), "# k() -> ()");
    // The GPU's functions are none of the process's: the error says the GPU runs them, and `ptx_str` compiles them.
    for name in ["store", "func"] {
        let said = format!("`{name}` runs on the GPU: `ptx_str` compiles it");
        assert_eq!(kernite::x86_str(src, name).unwrap_err(), said, "{name}");
    }
    assert!(compiles_for_ptx(src));
}

/// A name of a function the process runs that is none of its instances says that the function has none of that
/// name, as the function takes no compile-time arguments, or lists no such instance, or the name is of the function
/// alone where it takes some; and the GPU's function says the GPU runs it, whichever instance the name is of. A
/// name that no function has, or a helper's, is no kernel.
#[test]
fn x86_str_says_why_a_name_finds_no_kernel() {
    let src = "pub unsafe extern \"sysv64\" fn sum4() {}\n\
               #[instances(2, 4)] pub unsafe extern \"sysv64\" fn tile<const N: u64>() {}\n\
               #[instances(2, 4)] pub unsafe extern \"ptx-kernel\" fn gpu_tile<const N: u32>() {}\n\
               fn h() {}";
    let why = |name| kernite::x86_str(src, name).unwrap_err();
    assert_eq!(why("sum4::<2>"), "`sum4` has no instance `sum4::<2>`");
    assert_eq!(why("tile::<3>"), "`tile` has no instance `tile::<3>`");
    assert_eq!(why("tile"), "`tile` has no instance `tile`");
    for name in ["gpu_tile", "gpu_tile::<2>", "gpu_tile::<3>"] {
        assert_eq!(why(name), "`gpu_tile` runs on the GPU: `ptx_str` compiles it", "{name}");
    }
    for name in ["nope", "nope::<2>", "h", "h::<2>"] {
        assert_eq!(why(name), format!("no kernel `{name}`"), "{name}");
    }
    // The names that name a kernel compile.
    for name in ["sum4", "tile::<2>", "tile::<4>"] {
        assert!(kernite::x86_str(src, name).is_ok(), "{name}");
    }
}

#[test]
fn ptx_str_compiles_no_function_of_an_abi_kernite_does_not_know() {
    // PTX would refuse `add(x, 1)`, but it is not asked: the ABI is the error.
    let src = "pub unsafe extern \"ptx-kernel\" fn k() {}\nextern \"C\" fn c(x: u64) { add(x, 1); }\n";
    assert_eq!(kernite::ptx_str(src).unwrap_err(), format!("2:8: {ABI_C}"));
}

/// A function of an ABI kernite knows nothing of is none of the process's, and where the name given is its own, or an
/// instance's of it, the error is the one `ptx_str` gives of the module, at the ABI: not that no kernel has the name,
/// nor what is wrong with the function's `#[instances(..)]`, which kernite does not check. The name of a helper, and
/// one that no function has, find no kernel, whatever else the module has.
#[test]
fn x86_str_refuses_the_name_of_a_function_of_an_abi_kernite_does_not_know() {
    // `c` takes no compile-time arguments, so the second's `#[instances(..)]` is wrong.
    for (src, at) in [("extern \"C\" fn c() {}", "1:8"), ("#[instances(2)] extern \"C\" fn c() {}", "1:24")] {
        let at_the_abi = kernite::ptx_str(src).unwrap_err();
        assert_eq!(at_the_abi, format!("{at}: {ABI_C}"));
        for name in ["c", "c::<2>"] {
            assert_eq!(kernite::x86_str(src, name).unwrap_err(), at_the_abi, "{src}: {name}");
        }
    }
    let src = "fn h() {}\nunsafe extern \"sysv64\" fn k() {}\nextern \"C\" fn c() {}";
    for name in ["h", "nope"] {
        assert_eq!(kernite::x86_str(src, name).unwrap_err(), format!("no kernel `{name}`"), "{name}");
    }
}

#[test]
fn the_facades_refuse_a_kernel_that_is_not_unsafe() {
    let not = "a kernel trusts the pointers it is given, so it is an `unsafe fn`";
    let x = kernite::x86_str("extern \"sysv64\" fn k() {}", "k");
    assert_eq!(x.unwrap_err(), format!("1:20: {not}"));
    let p = kernite::ptx_str("extern \"ptx-kernel\" fn k() {}");
    assert_eq!(p.unwrap_err(), format!("1:24: {not}"));
    // A function a kernel calls promises Rust nothing, so it need not be one.
    assert!(compiles_for_ptx("pub unsafe extern \"ptx-kernel\" fn k() {}\nextern \"ptx-func\" fn f() {}\n"));
    // What does not compile is refused for that, as it was: the rest of the module is read after.
    let x = kernite::x86_str("extern \"sysv64\" fn k() { vaddps(1, 2, 3); }", "k");
    assert_eq!(x.unwrap_err(), "1:33: `vaddps` wants a ymm or an xmm; this is the integer 1");
    let p = kernite::ptx_str("extern \"ptx-kernel\" fn k() { add(1, 2, 3); }");
    assert_eq!(
        p.unwrap_err(),
        "1:30: `add` leaves out one of `.u16`, `.u32`, `.u64`, `.s16`, `.s32`, `.s64`: `add.u16`"
    );
}

#[test]
fn the_facades_refuse_a_module_of_both_kinds_with_a_static() {
    let kinds = "pub unsafe extern \"sysv64\" fn x() {}\npub unsafe extern \"ptx-kernel\" fn k() {}\n";
    // Without a static it is a module of two kinds of kernel, and each compiles.
    assert_eq!(x86(kinds, "x"), "# x() -> ()");
    assert!(compiles_for_ptx(kinds));
    // With one, whose memory it is cannot be said: it is two modules.
    let two = "a static is memory of the module's kernels: the GPU's, for PTX kernels, or the process's, for x86 \
               ones; and this module has both, so it is two modules, one of each kind of kernel and the statics \
               they name";
    let src = format!("{kinds}static A: global<u32>;\n");
    assert_eq!(kernite::x86_str(&src, "x").unwrap_err(), format!("3:8: {two}"));
    assert_eq!(kernite::ptx_str(&src).unwrap_err(), format!("3:8: {two}"));
}

#[test]
fn ptx_str_says_a_helper_does_not_parse() {
    // Nothing calls `h`, and a helper is never written out as Rust, so only kernite reads it.
    let src = "pub unsafe extern \"ptx-kernel\" fn k() {}\nfn h() { let x = ; }\n";
    assert_eq!(kernite::ptx_str(src).unwrap_err(), "2:18: expected an expression, found `;`");
}

#[test]
fn the_facades_read_the_attributes_of_every_function_of_the_module() {
    // `x86_str` reads the GPU's functions and the helpers too, and `ptx_str` the process's.
    let cooperative = "`#[cooperative]` goes on a kernel whose blocks or threads wait for each other, \
                       `extern \"ptx-kernel\"` or `extern \"sysv64\"`; `f` is a function the kernels call, which runs \
                       as the kernel that calls it is launched";
    let src = "pub unsafe extern \"sysv64\" fn k() {}\n#[cooperative] extern \"ptx-func\" fn f() {}\n";
    assert_eq!(kernite::x86_str(src, "k").unwrap_err(), format!("2:3: {cooperative}"));
    let src = "pub unsafe extern \"sysv64\" fn k() {}\n#[cooperative] fn h() {}\n";
    let helper = "`#[cooperative]` goes on a kernel whose blocks or threads wait for each other, \
                  `extern \"ptx-kernel\"` or `extern \"sysv64\"`; `h` is a helper, inlined where it is called";
    assert_eq!(kernite::x86_str(src, "k").unwrap_err(), format!("2:3: {helper}"));
    let kernel = "pub unsafe extern \"ptx-kernel\" fn k() {}\n";
    let once = "`j` takes no compile-time arguments, so it is compiled once, with no `#[instances(..)]`";
    let src = format!("{kernel}#[instances(1)] pub unsafe extern \"sysv64\" fn j() {{}}\n");
    assert_eq!(kernite::ptx_str(&src).unwrap_err(), format!("2:3: {once}"));
    // A kernel that is not `unsafe` is said first, before its attributes.
    let src = format!("{kernel}#[instances(1)] pub extern \"sysv64\" fn j() {{}}\n");
    let not = "a kernel trusts the pointers it is given, so it is an `unsafe fn`";
    assert_eq!(kernite::ptx_str(&src).unwrap_err(), format!("2:40: {not}"));
}

#[test]
fn the_facades_read_the_directives_of_every_function_of_the_module() {
    // A directive is a GPU kernel's: `x86_str` reads those of the GPU's other functions and of the
    // helpers, and `ptx_str` those of the process's kernels.
    let goes = "goes on a GPU kernel, `extern \"ptx-kernel\"`";
    let x86 = "pub unsafe extern \"sysv64\" fn k() {}\n";
    let func = "`f` is a function the kernels call, which runs as the kernel that calls it is launched";
    let helper = "`h` is a helper, inlined where it is called";
    for (item, what) in [("#[maxntid(256)] extern \"ptx-func\" fn f() {}", func), ("#[maxntid(256)] fn h() {}", helper)]
    {
        let src = format!("{x86}{item}\n");
        let want = format!("2:3: `#[maxntid(..)]` {goes}; {what}");
        assert_eq!(kernite::x86_str(&src, "k").unwrap_err(), want, "{item}");
    }
    let ptx = "pub unsafe extern \"ptx-kernel\" fn k() {}\n";
    let src = format!("{ptx}#[reqntid(8)] pub unsafe extern \"sysv64\" fn j() {{}}\n");
    let want = format!("2:3: `#[reqntid(..)]` {goes}; `j` is a kernel for x86-64, which the process runs");
    assert_eq!(kernite::ptx_str(&src).unwrap_err(), want);
    // A kernel that is not `unsafe` is said first, before its directives.
    let src = format!("{ptx}#[maxntid(1)] pub extern \"sysv64\" fn j() {{}}\n");
    let not = "a kernel trusts the pointers it is given, so it is an `unsafe fn`";
    assert_eq!(kernite::ptx_str(&src).unwrap_err(), format!("2:38: {not}"));
}
