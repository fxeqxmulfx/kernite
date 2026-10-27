//! What kgen-build decides of a module's items, and what its use cases do, on modules the
//! language parses: the use cases on [`Echo`], a fake machine. Each kernel in them compiles, for
//! its machine, but those written to fail.

use kgen_lang::{Fault, Fn, Guard, IntTy, Memory, Operand, Target, Value, parse};

use crate::{
    Abi, Backend, Called, Compiled, Device, DeviceStatic, Owner, StaticError, Unfinished, abi, all, compile,
    cooperative, function, instance, one, owner, prepare, statics,
};

/// An ABI is one kgen compiles for, by the name Rust writes it as, or none. Of them, the process
/// runs `extern "sysv64"` alone.
#[test]
fn an_abi_is_one_kgen_compiles_for() {
    let named = [("sysv64", Abi::Sysv64), ("ptx-kernel", Abi::PtxKernel), ("ptx-func", Abi::PtxFunc)];
    for (name, want) in named {
        assert_eq!(Abi::of(name), Some(want), "{name}");
    }
    for name in ["C", "Rust", "win64", "system", "ptx"] {
        assert_eq!(Abi::of(name), None, "{name}");
    }
    let native: Vec<Abi> = named.iter().map(|&(_, abi)| abi).filter(|abi| abi.native()).collect();
    assert_eq!(native, [Abi::Sysv64]);
    let src = "unsafe extern \"sysv64\" fn k(x: u64) -> u64 { ret(x); }\n\
               unsafe extern \"ptx-kernel\" fn g(x: *mut u32) { st.global.u32([x], 1); }\n\
               extern \"C\" fn c() {}\n\
               fn h() {}";
    let module = parse(src).unwrap();
    let of = |name| abi(function(&module, name).unwrap());
    assert_eq!([of("k"), of("g"), of("c"), of("h")], [Some(Abi::Sysv64), Some(Abi::PtxKernel), None, None]);
}

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

/// A machine whose code is the text of what it was told, a line for each thing, in order: a
/// fake, which does the work with simpler rules than a machine's, and asserts nothing.
struct Echo;

/// A static as Echo lays it out: its name, what its `global<..>` holds, and the names of the
/// statics laid out before it.
#[derive(Clone, Debug, PartialEq)]
struct Laid {
    name: String,
    held: String,
    before: Vec<String>,
}

/// What Echo lays out as `name`, holding `held`, after `before`.
fn laid(name: &str, held: &str, before: &[&str]) -> Laid {
    Laid { name: name.into(), held: held.into(), before: before.iter().map(|&b| b.into()).collect() }
}

/// A kernel as Echo writes it, and its code: its name, a line for each thing it was told, and
/// its variables, by name. A variable is its place among them.
struct Told {
    name: String,
    lines: Vec<String>,
    vars: Vec<String>,
}

impl Backend for Echo {
    type Kernel = Told;
    type Static = Laid;
    type Callee = Abi;
    type Code = Told;

    /// Refuses a name that starts with `taken`, a type that is not `global<..>`, and a value
    /// of 13.
    fn lay_out(&self, name: &str, memory: &Memory, before: &[Laid]) -> Result<Laid, StaticError> {
        if name.starts_with("taken") {
            return Err(StaticError::Name(format!("`{name}` is taken")));
        }
        let held = memory.ty.strip_prefix("global<").and_then(|ty| ty.strip_suffix('>'));
        let Some(held) = held else {
            return Err(StaticError::Type(format!("Echo lays out `global<..>` alone, not `{}`", memory.ty)));
        };
        if let Some(Value::Int(13, _)) = memory.value {
            return Err(StaticError::Value("Echo holds no 13".into()));
        }
        Ok(Laid { name: name.into(), held: held.into(), before: before.iter().map(|s| s.name.clone()).collect() })
    }

    /// Its kernels call each function of an ABI kgen compiles for, as that ABI.
    fn callee(&self, g: &Fn) -> Option<Abi> {
        abi(g)
    }

    fn kernel(
        &self,
        f: &Fn,
        name: &str,
        args: &[Value],
        cooperative: bool,
        statics: &[Laid],
        callees: &[Called<Abi>],
    ) -> Told {
        let listed = |args: &[Value]| args.iter().map(Value::to_string).collect::<Vec<_>>().join(", ");
        let mut lines = vec![format!("new {name}: {} [{}]", f.name, listed(args))];
        if cooperative {
            lines.push("cooperative".into());
        }
        lines.extend(statics.iter().map(|s| format!("memory {}", s.name)));
        for c in callees {
            lines.push(format!("callee {}: {} [{}] {:?}", c.name, c.function, listed(&c.args), c.callee));
        }
        Told { name: name.into(), lines, vars: vec![] }
    }

    /// Fails for a variable named `fail_v`, which it blames, or one named `fail`, blaming none.
    fn finish(&self, kernel: Told) -> Result<Told, Unfinished> {
        for var in &kernel.vars {
            let blamed = match var.as_str() {
                "fail_v" => Some(var.clone()),
                "fail" => None,
                _ => continue,
            };
            return Err(Unfinished { var: blamed, message: format!("`{var}` fails") });
        }
        Ok(kernel)
    }
}

impl Device for Echo {
    /// The statics, a line each, then the lines of each code. No code is no module.
    fn module(&self, code: &[Told], statics: &[Laid]) -> Result<String, String> {
        if code.is_empty() {
            return Err("nothing to load".into());
        }
        let statics = statics.iter().map(|s| format!("static {}", s.name));
        let code = code.iter().flat_map(|c| c.lines.iter().cloned());
        Ok(statics.chain(code).collect::<Vec<_>>().join("\n"))
    }

    fn symbol(&self, code: &Told) -> String {
        code.name.clone()
    }

    fn device_static(&self, s: &Laid) -> DeviceStatic {
        DeviceStatic { symbol: s.name.clone(), host: s.held.clone() }
    }
}

impl Told {
    /// Variable `name`, declared by `line`.
    fn declare(&mut self, line: String, name: &str) -> usize {
        self.lines.push(line);
        self.vars.push(name.into());
        self.vars.len() - 1
    }
}

/// Echo's kernel writes each instruction by its name and its operands, and a view of a variable
/// is the variable.
impl Target for Told {
    type Var = usize;

    fn param(&mut self, name: &str, ty: &str) -> Result<usize, String> {
        Ok(self.declare(format!("param {name} {ty}"), name))
    }

    fn result(&mut self, ty: &str) -> Result<(), String> {
        self.lines.push(format!("result {ty}"));
        Ok(())
    }

    fn var(&mut self, name: &str, ty: &str) -> Result<usize, String> {
        Ok(self.declare(format!("var {name} {ty}"), name))
    }

    fn view(&mut self, var: usize, _: &str) -> Result<usize, String> {
        Ok(var)
    }

    fn label(&mut self, label: u32) {
        self.lines.push(format!("label {label}"));
    }

    fn note(&mut self, text: &str) {
        self.lines.push(format!("note {text}"));
    }

    fn inst(&mut self, name: &str, args: &[Operand<usize>], _: Option<Guard<usize>>) -> Result<(), Fault> {
        let args: Vec<String> = args
            .iter()
            .map(|a| match a {
                Operand::Var(v) => self.vars[*v].clone(),
                Operand::Int(n) => n.to_string(),
                a => format!("{a:?}"),
            })
            .collect();
        self.lines.push(format!("inst {name} {}", args.join(" ")));
        Ok(())
    }

    fn end(&mut self) -> Result<(), String> {
        self.lines.push("end".into());
        Ok(())
    }
}

/// Each static is laid out after those before it that were laid out: Echo refuses the second,
/// so the third is laid out beside the first alone, and the fourth beside the first and the
/// third. Each has the constants it read.
#[test]
fn a_static_is_laid_out_beside_those_before_it() {
    let src = "const N: usize = 4;\n\
               static A: global<[u32; N]>;\n\
               static B: global<u32> = 13;\n\
               static C: global<u32>;\n\
               static D: global<u32>;";
    let laid_out: Vec<_> = statics(&parse(src).unwrap(), &Echo).into_iter().map(|s| s.map_err(|e| e.at(src))).collect();
    assert_eq!(
        laid_out,
        [
            Ok(Compiled { code: laid("A", "[u32; 4]", &[]), consts: vec!["N".into()] }),
            Err("3:25: Echo holds no 13".into()),
            Ok(Compiled { code: laid("C", "u32", &["A"]), consts: vec![] }),
            Ok(Compiled { code: laid("D", "u32", &["A", "C"]), consts: vec![] }),
        ]
    );
}

/// What Echo refuses of a static is placed where it can be mended: at its name, its type or
/// what it holds, as Echo says. What the language refuses is placed where the language says.
#[test]
fn a_static_that_does_not_lay_out_is_placed_where_it_can_be_mended() {
    let src = "static taken_x: global<u32>;\n\
               static Y: constant<u32>;\n\
               static Z: global<u32> = 13;\n\
               static W: global<[u32; M]>;";
    let refused: Vec<_> =
        statics(&parse(src).unwrap(), &Echo).into_iter().map(|s| s.map(|_| ()).map_err(|e| e.at(src))).collect();
    assert_eq!(
        refused,
        [
            Err("1:8: `taken_x` is taken".to_string()),
            Err("2:11: Echo lays out `global<..>` alone, not `constant<u32>`".into()),
            Err("3:25: Echo holds no 13".into()),
            Err("4:24: `M` is not defined".into()),
        ]
    );
}

/// A kernel is told whether it is cooperative, each static laid out, and each instance of each
/// function the machine lets it call, each in the module's order, before its body is written. A
/// function whose instances are wrong has none. It has the constants it read.
#[test]
fn a_kernel_hears_of_its_statics_and_callees_in_order() {
    let src = "const ONE: u64 = 1;\n\
               static A: global<u32>;\n\
               #[instances(2, 4)] extern \"ptx-func\" fn tile<const N: u32>() {}\n\
               static B: global<u32> = 13;\n\
               fn h() {}\n\
               #[instances(1)] extern \"ptx-func\" fn once() {}\n\
               unsafe extern \"ptx-kernel\" fn g() {}\n\
               extern \"C\" fn c() {}\n\
               static C: global<[u32; 4]>;\n\
               #[cooperative] unsafe extern \"sysv64\" fn k(x: u64) { add(x, ONE); }";
    let module = parse(src).unwrap();
    let prepared = prepare(&module, &Echo);
    assert_eq!(prepared.made, [laid("A", "u32", &[]), laid("C", "[u32; 4]", &["A"])]);
    let k = compile(&module, &prepared, &Echo, function(&module, "k").unwrap(), &[]);
    let told = k.map(|k| Compiled { code: k.code.lines, consts: k.consts });
    let heard = [
        "new k: k []",
        "cooperative",
        "memory A",
        "memory C",
        "callee tile::<2>: tile [2] PtxFunc",
        "callee tile::<4>: tile [4] PtxFunc",
        "callee g: g [] PtxKernel",
        "callee k: k [] Sysv64",
        "param x u64",
        "inst add x 1",
        "end",
    ];
    assert_eq!(told, Ok(Compiled { code: heard.map(String::from).to_vec(), consts: vec!["ONE".into()] }));
}

/// What Echo blames on a variable is placed where the variable is first declared, and what it
/// blames on none, at the function.
#[test]
fn an_unfinished_kernel_is_blamed_where_its_variable_is_declared() {
    let src = "unsafe extern \"ptx-kernel\" fn k(x: u32) {\n\
               let fail_v: u32; let fail_v: u32;\n\
               }\n\
               unsafe extern \"ptx-kernel\" fn j() { let fail: u32; }";
    let module = parse(src).unwrap();
    let prepared = prepare(&module, &Echo);
    let compiled = |name| {
        let f = function(&module, name).unwrap();
        compile(&module, &prepared, &Echo, f, &[]).map(|_| ()).map_err(|e| e.at(src))
    };
    assert_eq!(compiled("k"), Err("2:5: `fail_v` fails".into()));
    assert_eq!(compiled("j"), Err("4:31: `fail` fails".into()));
}

/// What goes wrong in an instance's body, or in its code, says which instance it is. What goes
/// wrong in a function of no compile-time arguments does not, and neither does what is wrong
/// with the instance itself or with the function's `#[cooperative]`. What is wrong with the
/// instance is said before what is wrong with its `#[cooperative]`.
#[test]
fn an_instance_says_which_when_it_fails() {
    let src = "#[instances(2, 4)] unsafe extern \"ptx-kernel\" fn tile<const N: u32>() {\n\
               if N == 4 { nop(true); }\n\
               if N == 2 { let fail_v: u32; }\n\
               }\n\
               unsafe extern \"ptx-kernel\" fn k() { nop(true); }\n\
               #[cooperative] #[instances(1)] extern \"ptx-func\" fn f<const N: u32>() {}";
    let module = parse(src).unwrap();
    let prepared = prepare(&module, &Echo);
    let compiled = |f, args: &[Value]| compile(&module, &prepared, &Echo, f, args).map(|_| ()).map_err(|e| e.at(src));
    let named = |name| instance(&module, name).unwrap().unwrap();
    let (tile, four) = named("tile::<4>");
    assert_eq!(compiled(tile, &four), Err("2:17: in `tile::<4>`: a bool is no operand".into()));
    let (_, two) = named("tile::<2>");
    assert_eq!(compiled(tile, &two), Err("3:17: in `tile::<2>`: `fail_v` fails".into()));
    let (k, none) = named("k");
    assert_eq!(compiled(k, &four), Err("5:31: `k` takes 0 compile-time arguments; this gives 1".into()));
    assert_eq!(compiled(k, &none), Err("5:41: a bool is no operand".into()));
    let (f, one) = named("f::<1>");
    assert_eq!(compiled(f, &[]), Err("6:53: `f` takes 1 compile-time arguments; this gives 0".into()));
    let not = "`#[cooperative]` goes on a kernel whose blocks or threads wait for each other, `extern \"ptx-kernel\"` \
               or `extern \"sysv64\"`; `f` is a function the kernels call, which runs as the kernel that calls it is \
               launched";
    assert_eq!(compiled(f, &one), Err(format!("6:3: {not}")));
}

/// `one` compiles what a name names: a function that takes no compile-time arguments, or an
/// instance of one that does. A name of no function, of no instance or of a helper names none.
/// A name of a function that does not compile, or whose `#[instances(..)]` is wrong, is the
/// error. So is the first static that is not laid out, whatever the name. What it compiles is
/// told its function and its compile-time arguments, and the module's statics and callees.
#[test]
fn one_names_no_kernel_it_does_not_find() {
    let src = "#[instances(2, 4)] unsafe extern \"ptx-kernel\" fn tile<const N: u32>() {}\n\
               fn h() {}\n\
               unsafe extern \"sysv64\" fn k() {}\n\
               unsafe extern \"sysv64\" fn bad() { nop(true); }\n\
               #[instances(1)] unsafe extern \"sysv64\" fn once() {}\n\
               static A: global<u32>;";
    let module = parse(src).unwrap();
    let found = |name| one(&module, &Echo, name).map(|k| k.map(|k| k.code.lines.join("\n"))).map_err(|e| e.at(src));
    let heard = "memory A\ncallee tile::<2>: tile [2] PtxKernel\ncallee tile::<4>: tile [4] PtxKernel\n\
                 callee k: k [] Sysv64\ncallee bad: bad [] Sysv64\nend";
    assert_eq!(found("tile::<4>"), Ok(Some(format!("new tile::<4>: tile [4]\n{heard}"))));
    assert_eq!(found("k"), Ok(Some(format!("new k: k []\n{heard}"))));
    for none in ["j", "tile::<3>", "tile", "h"] {
        assert_eq!(found(none), Ok(None), "{none}");
    }
    assert_eq!(found("bad"), Err("4:39: a bool is no operand".into()));
    let once = "`once` takes no compile-time arguments, so it is compiled once, with no `#[instances(..)]`";
    assert_eq!(found("once"), Err(format!("5:3: {once}")));
    let refusing = [
        (format!("static B: global<u32> = 13;\nstatic T: constant<u32>;\n{src}"), "1:25"),
        (format!("{src}\nstatic B: global<u32> = 13;"), "7:25"),
    ];
    for (src, at) in refusing {
        let module = parse(&src).unwrap();
        for name in ["k", "j", "bad", "once"] {
            let refused = one(&module, &Echo, name).map(|_| ()).map_err(|e| e.at(&src));
            assert_eq!(refused, Err(format!("{at}: Echo holds no 13")), "{name}");
        }
    }
}

/// `all` compiles each instance of each function the process does not run, in the module's
/// order, into one module with the statics: a PTX kernel or function, and one of an ABI kgen
/// knows nothing of, but no `extern "sysv64"` one. Each is told its function and its
/// compile-time arguments. A helper has no instance, but its `#[cooperative]` is the error,
/// before its `#[instances(..)]` and a later kernel's, and its `#[instances(..)]` is one. So is
/// a kernel that does not compile. The first static that is not laid out is the error before
/// them all, and a module the device refuses is refused as a whole.
#[test]
fn all_compiles_each_instance_in_order() {
    let src = "static A: global<u32>;\n\
               unsafe extern \"sysv64\" fn x() {}\n\
               #[instances(4, 2)] extern \"ptx-func\" fn tile<const N: u32>() {}\n\
               fn h() {}\n\
               extern \"C\" fn c() {}\n\
               unsafe extern \"ptx-kernel\" fn k() {}";
    let heard = "memory A\ncallee x: x [] Sysv64\ncallee tile::<4>: tile [4] PtxFunc\ncallee tile::<2>: tile [2] PtxFunc\n\
                 callee k: k [] PtxKernel\nend";
    let codes =
        ["tile::<4>: tile [4]", "tile::<2>: tile [2]", "c: c []", "k: k []"].map(|line| format!("new {line}\n{heard}"));
    assert_eq!(all(&parse(src).unwrap(), &Echo), Ok(Ok(format!("static A\n{}", codes.join("\n")))));
    let x86 = "unsafe extern \"sysv64\" fn x() {}";
    assert_eq!(all(&parse(x86).unwrap(), &Echo), Ok(Err("nothing to load".into())));
    let not = "`#[cooperative]` goes on a kernel whose blocks or threads wait for each other, `extern \"ptx-kernel\"` \
               or `extern \"sysv64\"`; `h` is a helper, inlined where it is called";
    let helper = "a helper is compiled where it is called, for what `::<..>` gives it there: `#[instances(..)]` lists \
                  what a function that names an ABI is compiled for";
    let failing = [
        (
            "#[cooperative] fn h() {}\nstatic B: global<u32> = 13;\nstatic T: constant<u32>;",
            "2:25: Echo holds no 13".into(),
        ),
        ("static A: global<u32>;\nstatic B: global<u32> = 13;", "2:25: Echo holds no 13".into()),
        (
            "#[cooperative] #[instances(1)] fn h() {}\nunsafe extern \"ptx-kernel\" fn k() { nop(true); }",
            format!("1:3: {not}"),
        ),
        ("#[instances(1)] fn h() {}", format!("1:3: {helper}")),
        ("unsafe extern \"ptx-kernel\" fn k() { nop(true); }", "1:41: a bool is no operand".into()),
    ];
    for (src, want) in failing {
        assert_eq!(all(&parse(src).unwrap(), &Echo).map_err(|e| e.at(src)), Err(want));
    }
}
