//! A module through the façades: `x86_str` compiles a kernel the process runs and `ptx_str` the
//! functions the GPU runs, and once that has gone well each refuses what is wrong with the rest
//! of the module, as the macro does.

type Result<T = (), E = Box<dyn std::error::Error + Send + Sync>> = std::result::Result<T, E>;

/// Kernel `name` of `src`, for x86: the first line of its listing.
fn x86(src: &str, name: &str) -> Result<String> {
    let listing = kernite::x86_str(src, name).map_err(|e| e.to_string())?.to_string();
    Ok(listing.lines().next().unwrap_or_default().to_string())
}

/// Whether `src` compiles for PTX, each of its kernels as an `.entry`.
fn compiles_for_ptx(src: &str) -> Result<bool> {
    Ok(kernite::ptx_str(src).map_err(|e| e.to_string())?.contains(".visible .entry "))
}

/// What kernite says of a function of ABI `C`, which it compiles nothing for, at the ABI.
const ABI_C: &str = "kernite compiles `extern \"sysv64\"` kernels, for x86-64, and `extern \"ptx-kernel\"` ones, for \
                     NVIDIA GPUs, with the `extern \"ptx-func\"` functions they call; `extern \"C\"` is none of these";

#[test]
fn x86_str_compiles_no_gpu_kernel() -> Result {
    let src = "pub unsafe extern \"sysv64\" fn k() {}\n\
               pub unsafe extern \"ptx-kernel\" fn store(x: *mut u32) { st.global.u32([x], 1); }\n\
               pub extern \"ptx-func\" fn func() {}";
    assert_eq!(x86(src, "k")?, "# k() -> ()");
    // The GPU's functions are none of the process's: the error says the GPU runs them, and `ptx_str` compiles them.
    for name in ["store", "func"] {
        let said = format!("`{name}` runs on the GPU: `ptx_str` compiles it");
        assert_eq!(kernite::x86_str(src, name).err().ok_or("x86_str gave Ok")?, said, "{name}");
    }
    assert!(compiles_for_ptx(src)?);
    Ok(())
}

/// A name of a function the process runs that is none of its instances says that the function has none of that
/// name, as the function takes no compile-time arguments, or lists no such instance, or the name is of the function
/// alone where it takes some; and the GPU's function says the GPU runs it, whichever instance the name is of. A
/// name that no function has, or a helper's, is no kernel.
#[test]
fn x86_str_says_why_a_name_finds_no_kernel() -> Result {
    let src = "pub unsafe extern \"sysv64\" fn sum4() {}\n\
               #[instances(2, 4)] pub unsafe extern \"sysv64\" fn tile<const N: u64>() {}\n\
               #[instances(2, 4)] pub unsafe extern \"ptx-kernel\" fn gpu_tile<const N: u32>() {}\n\
               fn h() {}";
    let why = |name: &str| -> Result<String> { Ok(kernite::x86_str(src, name).err().ok_or("x86_str gave Ok")?) };
    assert_eq!(why("sum4::<2>")?, "`sum4` has no instance `sum4::<2>`");
    assert_eq!(why("tile::<3>")?, "`tile` has no instance `tile::<3>`");
    assert_eq!(why("tile")?, "`tile` has no instance `tile`");
    for name in ["gpu_tile", "gpu_tile::<2>", "gpu_tile::<3>"] {
        assert_eq!(why(name)?, "`gpu_tile` runs on the GPU: `ptx_str` compiles it", "{name}");
    }
    for name in ["nope", "nope::<2>", "h", "h::<2>"] {
        assert_eq!(why(name)?, format!("no kernel `{name}`"), "{name}");
    }
    // The names that name a kernel compile.
    for name in ["sum4", "tile::<2>", "tile::<4>"] {
        assert!(kernite::x86_str(src, name).is_ok(), "{name}");
    }
    Ok(())
}

#[test]
fn ptx_str_compiles_no_function_of_an_abi_kernite_does_not_know() -> Result {
    // PTX would refuse `add(x, 1)`, but it is not asked: the ABI is the error.
    let src = "pub unsafe extern \"ptx-kernel\" fn k() {}\nextern \"C\" fn c(x: u64) { add(x, 1); }\n";
    assert_eq!(kernite::ptx_str(src).err().ok_or("ptx_str gave Ok")?, format!("2:8: {ABI_C}"));
    Ok(())
}

/// A function of an ABI kernite knows nothing of is none of the process's, and where the name given is its own, or an
/// instance's of it, the error is the one `ptx_str` gives of the module, at the ABI: not that no kernel has the name,
/// nor what is wrong with the function's `#[instances(..)]`, which kernite does not check. The name of a helper, and
/// one that no function has, find no kernel, whatever else the module has.
#[test]
fn x86_str_refuses_the_name_of_a_function_of_an_abi_kernite_does_not_know() -> Result {
    // `c` takes no compile-time arguments, so the second's `#[instances(..)]` is wrong.
    for (src, at) in [("extern \"C\" fn c() {}", "1:8"), ("#[instances(2)] extern \"C\" fn c() {}", "1:24")] {
        let at_the_abi = kernite::ptx_str(src).err().ok_or("ptx_str gave Ok")?;
        assert_eq!(at_the_abi, format!("{at}: {ABI_C}"));
        for name in ["c", "c::<2>"] {
            assert_eq!(kernite::x86_str(src, name).err().ok_or("x86_str gave Ok")?, at_the_abi, "{src}: {name}");
        }
    }
    let src = "fn h() {}\nunsafe extern \"sysv64\" fn k() {}\nextern \"C\" fn c() {}";
    for name in ["h", "nope"] {
        let said = kernite::x86_str(src, name).err().ok_or("x86_str gave Ok")?;
        assert_eq!(said, format!("no kernel `{name}`"), "{name}");
    }
    Ok(())
}

/// The message of a kernel on the threads of a grid that is not an `unsafe fn`.
const GRID: &str = "the threads of a grid run at once, and kernite cannot prove that they reach memory apart, so a \
                    `#[cooperative]` kernel is an `unsafe fn`";

/// The message of a kernel on the blocks of a grid that is not an `unsafe fn`.
const BLOCKS: &str = "the blocks of a grid run at once and wait for each other, and kernite cannot prove that they \
                      reach memory apart, so a `#[cooperative]` kernel is an `unsafe fn`";

#[test]
fn the_facades_refuse_a_cooperative_kernel_that_is_not_unsafe() -> Result {
    // A kernel the process runs on one thread, or the GPU on threads that reach memory apart, is no `unsafe fn` where
    // kernite proves it safe, as its machine says.
    assert_eq!(x86("extern \"sysv64\" fn k() {}", "k")?, "# k() -> ()");
    assert!(compiles_for_ptx("pub extern \"ptx-kernel\" fn k() {}\n")?);
    let x = kernite::x86_str("#[cooperative] extern \"sysv64\" fn k() {}", "k");
    assert_eq!(x.err().ok_or("x gave Ok")?, format!("1:3: {GRID}"));
    let p = kernite::ptx_str("#[cooperative] extern \"ptx-kernel\" fn k() {}");
    assert_eq!(p.err().ok_or("p gave Ok")?, format!("1:3: {BLOCKS}"));
    // A function a kernel calls promises Rust nothing, so it need not be one.
    assert!(compiles_for_ptx("pub unsafe extern \"ptx-kernel\" fn k() {}\nextern \"ptx-func\" fn f() {}\n")?);
    // What does not compile is refused for that, as it was: the rest of the module is read after.
    let x = kernite::x86_str("extern \"sysv64\" fn k() { vaddps(1, 2, 3); }", "k");
    assert_eq!(x.err().ok_or("x gave Ok")?, "1:33: `vaddps` wants a ymm or an xmm; this is the integer 1");
    let p = kernite::ptx_str("extern \"ptx-kernel\" fn k() { add(1, 2, 3); }");
    assert_eq!(
        p.err().ok_or("p gave Ok")?,
        "1:30: `add` leaves out one of `.u16`, `.u32`, `.u64`, `.s16`, `.s32`, `.s64`: `add.u16`"
    );
    Ok(())
}

#[test]
fn the_facades_refuse_a_module_of_both_kinds_with_a_static() -> Result {
    let kinds = "pub unsafe extern \"sysv64\" fn x() {}\npub unsafe extern \"ptx-kernel\" fn k() {}\n";
    // Without a static it is a module of two kinds of kernel, and each compiles.
    assert_eq!(x86(kinds, "x")?, "# x() -> ()");
    assert!(compiles_for_ptx(kinds)?);
    // With one, whose memory it is cannot be said: it is two modules.
    let two = "a static is memory of the module's kernels: the GPU's, for PTX kernels, or the process's, for x86 \
               ones; and this module has both, so it is two modules, one of each kind of kernel and the statics \
               they name";
    let src = format!("{kinds}static A: global<u32>;\n");
    assert_eq!(kernite::x86_str(&src, "x").err().ok_or("x86_str gave Ok")?, format!("3:8: {two}"));
    assert_eq!(kernite::ptx_str(&src).err().ok_or("ptx_str gave Ok")?, format!("3:8: {two}"));
    Ok(())
}

#[test]
fn ptx_str_says_a_helper_does_not_parse() -> Result {
    // Nothing calls `h`, and a helper is never written out as Rust, so only kernite reads it.
    let src = "pub unsafe extern \"ptx-kernel\" fn k() {}\nfn h() { let x = ; }\n";
    assert_eq!(kernite::ptx_str(src).err().ok_or("ptx_str gave Ok")?, "2:18: expected an expression, found `;`");
    Ok(())
}

#[test]
fn the_facades_read_the_attributes_of_every_function_of_the_module() -> Result {
    // `x86_str` reads the GPU's functions and the helpers too, and `ptx_str` the process's.
    let cooperative = "`#[cooperative]` goes on a kernel whose blocks or threads wait for each other, \
                       `extern \"ptx-kernel\"` or `extern \"sysv64\"`; `f` is a function the kernels call, which runs \
                       as the kernel that calls it is launched";
    let src = "pub unsafe extern \"sysv64\" fn k() {}\n#[cooperative] extern \"ptx-func\" fn f() {}\n";
    assert_eq!(kernite::x86_str(src, "k").err().ok_or("x86_str gave Ok")?, format!("2:3: {cooperative}"));
    let src = "pub unsafe extern \"sysv64\" fn k() {}\n#[cooperative] fn h() {}\n";
    let helper = "`#[cooperative]` goes on a kernel whose blocks or threads wait for each other, \
                  `extern \"ptx-kernel\"` or `extern \"sysv64\"`; `h` is a helper, inlined where it is called";
    assert_eq!(kernite::x86_str(src, "k").err().ok_or("x86_str gave Ok")?, format!("2:3: {helper}"));
    let kernel = "pub unsafe extern \"ptx-kernel\" fn k() {}\n";
    let once = "`j` takes no compile-time arguments, so it is compiled once, with no `#[instances(..)]`";
    let src = format!("{kernel}#[instances(1)] pub unsafe extern \"sysv64\" fn j() {{}}\n");
    assert_eq!(kernite::ptx_str(&src).err().ok_or("ptx_str gave Ok")?, format!("2:3: {once}"));
    // A cooperative kernel that is not `unsafe` is said first, before its attributes.
    let src =
        "pub unsafe extern \"sysv64\" fn k() {}\n#[instances(1)] #[cooperative] pub extern \"ptx-kernel\" fn j() {}\n";
    assert_eq!(kernite::x86_str(src, "k").err().ok_or("x86_str gave Ok")?, format!("2:19: {BLOCKS}"));
    let src = format!("{kernel}#[instances(1)] #[cooperative] pub extern \"sysv64\" fn j() {{}}\n");
    assert_eq!(kernite::ptx_str(&src).err().ok_or("ptx_str gave Ok")?, format!("2:19: {GRID}"));
    Ok(())
}

#[test]
fn the_facades_read_the_directives_of_every_function_of_the_module() -> Result {
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
        assert_eq!(kernite::x86_str(&src, "k").err().ok_or("x86_str gave Ok")?, want, "{item}");
    }
    let ptx = "pub unsafe extern \"ptx-kernel\" fn k() {}\n";
    let src = format!("{ptx}#[reqntid(8)] pub unsafe extern \"sysv64\" fn j() {{}}\n");
    let want = format!("2:3: `#[reqntid(..)]` {goes}; `j` is a kernel for x86-64, which the process runs");
    assert_eq!(kernite::ptx_str(&src).err().ok_or("ptx_str gave Ok")?, want);
    // A kernel that is not `unsafe` is said first, before its directives.
    let src = format!("{ptx}#[maxntid(1)] #[cooperative] pub extern \"sysv64\" fn j() {{}}\n");
    assert_eq!(kernite::ptx_str(&src).err().ok_or("ptx_str gave Ok")?, format!("2:17: {GRID}"));
    Ok(())
}
