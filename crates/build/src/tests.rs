//! What kernite-build decides of a module's items, and what its use cases do, on modules the
//! language parses: the use cases on [`Echo`], a fake machine. Each kernel in them compiles, for
//! its machine, but those written to fail.

use kernite_lang::{Error, Fault, Fn, Guard, IntTy, Memory, Operand, Span, Target, Value, parse};

use crate::{
    Abi, Backend, Call, Called, Compiled, ConstItem, Device, DeviceStatic, Given, HostStatic, ItemOut, Launcher,
    ModuleOut, Naked, Native, NativeOut, Owner, StaticError, Unfinished, abi, all, compile, consts, cooperative,
    function, instance, module, one, owner, prepare, statics,
};

/// An ABI is one kernite compiles for, by the name Rust writes it as, or none. Of them, the process
/// runs `extern "sysv64"` alone.
#[test]
fn an_abi_is_one_kernite_compiles_for() {
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
/// call, that says it is refused where it says so. A function of an ABI kernite knows nothing of is
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
/// `tile::<4>` names `tile` for 4, the second it lists. A name of no function, or of no instance
/// its function has, names none.
#[test]
fn an_instance_is_found_by_its_name() {
    let tile =
        "#[instances(2, 4)] unsafe extern \"sysv64\" fn tile<const N: u64>(x: u64) -> u64 { add(x, N); ret(x); }";
    let src = format!("{tile}\nunsafe extern \"sysv64\" fn k(x: u64) -> u64 {{ ret(x); }}");
    let module = parse(&src).unwrap();
    let found = |name| instance(&module, name).unwrap().map(|(f, args, at)| (f.name.as_str(), args, at));
    assert_eq!(found("tile::<2>"), Some(("tile", vec![Value::Int(2, Some(IntTy::U64))], 0)));
    assert_eq!(found("tile::<4>"), Some(("tile", vec![Value::Int(4, Some(IntTy::U64))], 1)));
    assert_eq!(found("k"), Some(("k", vec![], 0)));
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
/// its variables, by name, with where each is declared, as `declared_at` tells it. A variable
/// is its place among them. Beside them, the statics it was told of, and the functions its body
/// names, those it may reach at an address, and the statics it names, each once, in the order
/// first named.
struct Told {
    name: String,
    lines: Vec<String>,
    vars: Vec<String>,
    declared: Vec<(usize, Span)>,
    memory: Vec<String>,
    funcs: Vec<String>,
    reach: Vec<String>,
    statics: Vec<String>,
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

    /// Its kernels call each function of an ABI kernite compiles for, as that ABI.
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
        let memory = statics.iter().map(|s| s.name.clone()).collect();
        Told {
            name: name.into(),
            lines,
            vars: vec![],
            declared: vec![],
            memory,
            funcs: vec![],
            reach: vec![],
            statics: vec![],
        }
    }

    /// Fails for a variable named `fail_v`, which it blames, or one named `fail`, blaming none.
    fn finish(&self, kernel: Told) -> Result<Told, Unfinished> {
        for (i, var) in kernel.vars.iter().enumerate() {
            let blamed = match var.as_str() {
                "fail_v" => Some(i),
                "fail" => None,
                _ => continue,
            };
            return Err(Unfinished::blaming(blamed, &kernel.declared, format!("`{var}` fails")));
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

impl Native for Echo {
    /// The lines, and the listing they make, the functions the body names and may reach at an
    /// address, and the statics it names. It needs no feature, and reads no constant from
    /// memory.
    fn naked(&self, code: &Told) -> Naked {
        Naked {
            code: code.lines.clone(),
            text: code.lines.join("\n"),
            needs: vec![],
            consts: vec![],
            funcs: code.funcs.clone(),
            reach: code.reach.clone(),
            statics: code.statics.clone(),
        }
    }

    /// What the static holds, as one element of no bytes: Echo makes no memory.
    fn host_static(&self, s: &Laid) -> HostStatic {
        HostStatic { elem: s.held.clone(), size: 0, len: 1, align: 1, init: vec![] }
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
/// is the variable. It takes each static it was told of as a variable of its name.
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

    fn global(&mut self, name: &str) -> Result<usize, String> {
        if !self.memory.iter().any(|m| m == name) {
            return Err(format!("`{name}` does not compile, so no kernel names it"));
        }
        self.statics.push(name.into());
        Ok(self.declare(format!("global {name}"), name))
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
        for a in args {
            if let Operand::Func(f) | Operand::Code(f) = a
                && !self.funcs.contains(f)
            {
                self.funcs.push(f.clone());
            }
        }
        // The functions taken together last: a call at an address may reach each.
        if let Some(Operand::Vector(fs)) = args.last() {
            for f in fs {
                if let Operand::Func(f) = f
                    && !self.reach.contains(f)
                {
                    self.reach.push(f.clone());
                }
            }
        }
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

    fn declared_at(&mut self, var: usize, at: Span) {
        self.declared.push((var, at));
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

/// What Echo blames on a variable is placed where that variable is declared, and what it blames
/// on none, at the function.
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
    let (tile, four, _) = named("tile::<4>");
    assert_eq!(compiled(tile, &four), Err("2:17: in `tile::<4>`: a bool is no operand".into()));
    let (_, two, _) = named("tile::<2>");
    assert_eq!(compiled(tile, &two), Err("3:17: in `tile::<2>`: `fail_v` fails".into()));
    let (k, none, _) = named("k");
    assert_eq!(compiled(k, &four), Err("5:31: `k` takes 0 compile-time arguments; this gives 1".into()));
    assert_eq!(compiled(k, &none), Err("5:41: a bool is no operand".into()));
    let (f, one, _) = named("f::<1>");
    assert_eq!(compiled(f, &[]), Err("6:53: `f` takes 1 compile-time arguments; this gives 0".into()));
    let not = "`#[cooperative]` goes on a kernel whose blocks or threads wait for each other, `extern \"ptx-kernel\"` \
               or `extern \"sysv64\"`; `f` is a function the kernels call, which runs as the kernel that calls it is \
               launched";
    assert_eq!(compiled(f, &one), Err(format!("6:3: {not}")));
}

/// `one` compiles what a name names: a function the process runs that takes no compile-time
/// arguments, or an instance of one that does. A name of no function, of no instance, of a
/// helper or of a function the GPU runs names none. A name of a function that does not compile,
/// or whose `#[instances(..)]` is wrong, is the error. So is the first static that is not laid
/// out, whatever the name. What it compiles is told its function and its compile-time arguments,
/// and the module's statics and callees. Once it has compiled, what is wrong with the rest of
/// the module is the error as well.
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
    for none in ["j", "tile::<3>", "tile", "tile::<4>", "h"] {
        assert_eq!(found(none), Ok(None), "{none}");
    }
    assert_eq!(found("bad"), Err("4:39: a bool is no operand".into()));
    let once = "`once` takes no compile-time arguments, so it is compiled once, with no `#[instances(..)]`";
    assert_eq!(found("once"), Err(format!("5:3: {once}")));
    // `k` compiles, but with `tile` and the static the module is two modules.
    assert!(found("k").unwrap_err().starts_with("6:8: a static is memory of the module's kernels"));
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
    // Where the rest of the module is right, what a name names is compiled, and told the module.
    let src = "#[instances(2, 4)] unsafe extern \"sysv64\" fn tile<const N: u32>() {}\n\
               fn h() {}\n\
               unsafe extern \"sysv64\" fn k() {}\n\
               static A: global<u32>;";
    let module = parse(src).unwrap();
    let found = |name| one(&module, &Echo, name).map(|k| k.map(|k| k.code.lines.join("\n"))).map_err(|e| e.at(src));
    let heard = "memory A\ncallee tile::<2>: tile [2] Sysv64\ncallee tile::<4>: tile [4] Sysv64\n\
                 callee k: k [] Sysv64\nend";
    assert_eq!(found("tile::<4>"), Ok(Some(format!("new tile::<4>: tile [4]\n{heard}"))));
    assert_eq!(found("k"), Ok(Some(format!("new k: k []\n{heard}"))));
}

/// `all` compiles each instance of each function the GPU runs, in the module's order, into one
/// module with the statics: a PTX kernel or function, but no `extern "sysv64"` one and none of
/// an ABI kernite knows nothing of. Each is told its function and its compile-time arguments. A
/// helper has no instance, but its `#[cooperative]` is the error, before its `#[instances(..)]`
/// and a later kernel's, and its `#[instances(..)]` is one. So is a kernel that does not
/// compile. The first static that is not laid out is the error before them all, and a module the
/// device refuses is refused as a whole. Once the device has made its module, a module of both
/// kinds of kernel and a static is the error, and so is a function of an ABI kernite knows
/// nothing of.
#[test]
fn all_compiles_each_instance_in_order() {
    let src = "static A: global<u32>;\n\
               #[instances(4, 2)] extern \"ptx-func\" fn tile<const N: u32>() {}\n\
               fn h() {}\n\
               unsafe extern \"ptx-kernel\" fn k() {}";
    let heard = "memory A\ncallee tile::<4>: tile [4] PtxFunc\ncallee tile::<2>: tile [2] PtxFunc\n\
                 callee k: k [] PtxKernel\nend";
    let codes = ["tile::<4>: tile [4]", "tile::<2>: tile [2]", "k: k []"].map(|line| format!("new {line}\n{heard}"));
    assert_eq!(all(&parse(src).unwrap(), &Echo), Ok(Ok(format!("static A\n{}", codes.join("\n")))));
    let x86 = "unsafe extern \"sysv64\" fn x() {}";
    assert_eq!(all(&parse(x86).unwrap(), &Echo), Ok(Err("nothing to load".into())));
    let not = "`#[cooperative]` goes on a kernel whose blocks or threads wait for each other, `extern \"ptx-kernel\"` \
               or `extern \"sysv64\"`; `h` is a helper, inlined where it is called";
    let helper = "a helper is compiled where it is called, for what `::<..>` gives it there: `#[instances(..)]` lists \
                  what a function that names an ABI is compiled for";
    let two = "a static is memory of the module's kernels: the GPU's, for PTX kernels, or the process's, for x86 \
               ones; and this module has both, so it is two modules, one of each kind of kernel and the statics \
               they name";
    let abi = "kernite compiles `extern \"sysv64\"` kernels, for x86-64, and `extern \"ptx-kernel\"` ones, for NVIDIA \
               GPUs, with the `extern \"ptx-func\"` functions they call; `extern \"C\"` is none of these";
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
        (
            "static A: global<u32>;\nunsafe extern \"sysv64\" fn x() {}\nunsafe extern \"ptx-kernel\" fn k() {}",
            format!("1:8: {two}"),
        ),
        ("unsafe extern \"ptx-kernel\" fn k() {}\nextern \"C\" fn c() {}", format!("2:8: {abi}")),
    ];
    for (src, want) in failing {
        assert_eq!(all(&parse(src).unwrap(), &Echo).map_err(|e| e.at(src)), Err(want));
    }
}

/// The module in `src`, given no consts, its kernels compiled by Echo, the process's and the
/// GPU's alike.
fn echoed(src: &str) -> ModuleOut {
    module(parse(src).unwrap(), None, &Echo, &Echo)
}

/// Each item of a module comes out as what it is, in the module's order: Rust passes through, a
/// static is laid out or says why not, and a function is a helper, of an ABI kernite does not
/// know, the process's or the GPU's. A kernel the process runs whose `#[instances(..)]` is wrong
/// compiles nothing, and one whose instances do not all compile says why of each that does not,
/// in order; neither is written. The listings of the kernels the process runs, and the code of
/// the functions the GPU runs, are in the module's order as well.
#[test]
fn a_module_s_items_come_out_in_order() {
    let src = "const ONE: u64 = 1;\n\
               const TWO: u64 = 2;\n\
               static A: global<[u32; 4]>;\n\
               static U: global<u32> = 13;\n\
               fn h() {}\n\
               extern \"C\" fn c() {}\n\
               unsafe extern \"sysv64\" fn k(x: u64) { add(x, ONE); add(x, TWO); }\n\
               #[instances(2, 4)] unsafe extern \"sysv64\" fn tile<const N: u64>(x: u64) { lea(x, [A]); }\n\
               #[instances(1)] unsafe extern \"sysv64\" fn once(x: u64) {}\n\
               #[instances(1, 2, 3)] unsafe extern \"sysv64\" fn half<const N: u64>(x: u64) {\n\
               if N >= 2 { lea(x, [U]); }\n\
               }\n\
               struct S;";
    let out = echoed(src);
    let [
        ItemOut::Rust,
        ItemOut::Rust,
        ItemOut::HostStatic(Ok(a)),
        ItemOut::HostStatic(Err(u)),
        ItemOut::Helper(h),
        ItemOut::Unknown(c),
        ItemOut::Native(k),
        ItemOut::Native(tile),
        ItemOut::Native(once),
        ItemOut::Native(half),
        ItemOut::Rust,
    ] = &out.items[..]
    else {
        panic!("{:?}", out.items);
    };
    assert_eq!(a.code, HostStatic { elem: "[u32; 4]".into(), size: 0, len: 1, align: 1, init: vec![] });
    assert_eq!(u.at(src), "4:25: Echo holds no 13");
    assert!(h.is_empty() && c.at(src).starts_with("6:8: "));
    let args: Vec<_> = tile.instances.iter().map(|i| i.args.clone()).collect();
    assert_eq!(args, [[Value::Int(2, Some(IntTy::U64))], [Value::Int(4, Some(IntTy::U64))]]);
    assert!(k.written && tile.written);
    let listing = |name: &str, kernel: &NativeOut, i: usize| (name.to_string(), kernel.instances[i].naked.text.clone());
    assert_eq!(out.listings, [listing("k", k, 0), listing("tile::<2>", tile, 0), listing("tile::<4>", tile, 1)]);
    let once_said = "`once` takes no compile-time arguments, so it is compiled once, with no `#[instances(..)]`";
    assert_eq!(once.said.as_ref().map(|e| e.at(src)), Some(format!("9:3: {once_said}")));
    assert_eq!((once.count, once.instances.len(), once.errors.len(), once.written), (0, 0, 0, false));
    let half_said: Vec<_> = half.errors.iter().map(|e| e.at(src)).collect();
    assert_eq!(
        half_said,
        [
            "11:21: in `half::<2>`: `U` does not compile, so no kernel names it",
            "11:21: in `half::<3>`: `U` does not compile, so no kernel names it",
        ]
    );
    assert_eq!((half.said.is_none(), half.count, half.instances.len(), half.written), (true, 3, 1, false));
    assert!(out.given.is_empty() && out.owner.is_none() && out.device.is_none());
    let src = "static T: global<u32> = 13;\n\
               static B: global<u32>;\n\
               unsafe extern \"ptx-kernel\" fn g() {}\n\
               #[instances(1)] extern \"ptx-func\" fn f<const K: u32>() {}";
    let out = echoed(src);
    let [ItemOut::DeviceStatic(Err(t)), ItemOut::DeviceStatic(Ok(b)), ItemOut::Device(_), ItemOut::Device(_)] =
        &out.items[..]
    else {
        panic!("{:?}", out.items);
    };
    assert_eq!(t.at(src), "1:25: Echo holds no 13");
    assert_eq!(b.code, DeviceStatic { symbol: "B".into(), host: "u32".into() });
    let device = out.device.unwrap();
    let made: Vec<_> = device.text.lines().filter(|line| !line.starts_with("callee ")).collect();
    assert_eq!(made, ["static B", "new g: g []", "memory B", "end", "new f::<1>: f [1]", "memory B", "end"]);
}

/// Each item has the consts it read, by name, each once, in the module's order, and no other's. A
/// kernel's are those of its body, of a helper it calls, and of the `#[instances(..)]` it lists, in
/// whichever of its instances read them, itself or through other consts: instances that read one
/// const give it once, and the consts of them all come in the module's order, not in the order the
/// instances read them. A function it calls by name has its own, and so has an `extern "ptx-func"`
/// function, which reads those of a helper it calls. A static's are those of its type and of what
/// it holds. A helper has none, for the kernel or function that calls it reads what it reads.
#[test]
fn an_item_has_the_consts_it_read() {
    let src = "const A: u64 = 1;\n\
               const B: u64 = A + 1;\n\
               const C: u64 = 3;\n\
               const D: u64 = 4;\n\
               const E: usize = 2;\n\
               const F: u32 = 6;\n\
               const G: u64 = 7;\n\
               const L: u64 = 2;\n\
               static S: global<[u32; E]>;\n\
               static T: global<u32> = F;\n\
               fn help(x: gpr) { add(x, C); }\n\
               unsafe extern \"sysv64\" fn direct(x: u64) { add(x, B); }\n\
               unsafe extern \"sysv64\" fn through(x: u64) { help(x); }\n\
               #[instances(1, 2)] unsafe extern \"sysv64\" fn each<const N: u64>(x: u64) {\n\
               if N == 2 { add(x, D); }\n\
               }\n\
               #[instances(1, 2)] unsafe extern \"sysv64\" fn both<const N: u64>(x: u64) {\n\
               add(x, D);\n\
               if N == 2 { add(x, C); }\n\
               }\n\
               #[instances(L)] unsafe extern \"sysv64\" fn listed<const N: u64>(x: u64) { add(x, N); }\n\
               unsafe extern \"sysv64\" fn none(x: u64) { add(x, 1); }";
    let read = |src: &str| echoed(src).items.iter().map(|item| item.consts().join(" ")).collect::<Vec<_>>();
    assert_eq!(read(src), ["", "", "", "", "", "", "", "", "E", "F", "", "A B", "C", "D", "C D", "L", ""]);
    let src = "const A: u32 = 1;\n\
               const B: u32 = 2;\n\
               const C: usize = 4;\n\
               const D: u32 = 8;\n\
               static S: global<[u32; C]>;\n\
               fn aid(x: u32) { add(x, D); }\n\
               extern \"ptx-func\" fn f(x: u32) { add(x, B); aid(x); }\n\
               unsafe extern \"ptx-kernel\" fn g(x: u32) { add(x, A); call(f, x); }\n\
               #[instances(1, 2)] unsafe extern \"ptx-kernel\" fn h<const N: u32>(x: u32) {\n\
               add(x, B);\n\
               if N == 2 { add(x, A); }\n\
               }";
    assert_eq!(read(src), ["", "", "", "", "C", "", "B D", "A", "A B"]);
}

/// A const the module is given that an item of its own is named as too is left out, and the item
/// of its own says so, where it is named, in the order of its own items. What no name belongs to,
/// among the consts given, stays.
#[test]
fn a_const_given_twice_is_refused_at_the_module_s_own() {
    let src = "struct G;\n\
               const A: u64 = 1;\n\
               const W: u64 = 2;\n\
               const k: u64 = 5;\n\
               const STEP: u64 = 3;\n\
               const STEP: u64 = 4;\n\
               static A: global<u32>;\n\
               unsafe extern \"sysv64\" fn k(x: u64) { add(x, STEP); add(x, W); }";
    // The first five items are the consts given: the macro tells the parser how many tokens they
    // are, and the parser leaves an item of the module's own named as one of them to `module`.
    let before = kernite_lang::lex(&src[..src.find("const STEP: u64 = 4;").unwrap()]).unwrap().len();
    let parsed = kernite_lang::module(&kernite_lang::lex(src).unwrap(), before).unwrap();
    let kept = [0, 2, 5, 6, 7].map(|i| parsed.items[i].tokens.clone());
    let out = module(parsed, Some(Given { before, from: "crate::sizes".into() }), &Echo, &Echo);
    let said: Vec<_> = out.given.iter().map(|e| e.at(src)).collect();
    assert_eq!(
        said,
        [
            "6:7: `STEP` names one of the consts the module takes from `crate::sizes` as well",
            "7:8: `A` names one of the consts the module takes from `crate::sizes` as well",
            "8:27: `k` names one of the consts the module takes from `crate::sizes` as well",
        ]
    );
    let tokens: Vec<_> = out.module.items.iter().map(|item| item.tokens.clone()).collect();
    assert_eq!(tokens, kept);
    let [ItemOut::Rust, ItemOut::Given, ItemOut::Rust, ItemOut::HostStatic(Ok(_)), ItemOut::Native(_)] = &out.items[..]
    else {
        panic!("{:?}", out.items);
    };
}

/// A kernel that is not an `unsafe fn` is refused where it is named, though it is compiled: one
/// the process runs is not written, and one the GPU runs is never launched, though its code is in
/// the GPU's module. A function the kernels call promises Rust nothing, so it need not be one.
#[test]
fn a_kernel_that_is_not_unsafe_is_refused() {
    let not = "a kernel trusts the pointers it is given, so it is an `unsafe fn`";
    let src = "extern \"sysv64\" fn k(x: u64) { add(x, 1); }";
    let out = echoed(src);
    let [ItemOut::Native(k)] = &out.items[..] else { panic!("{:?}", out.items) };
    assert_eq!(k.not_unsafe.as_ref().map(|e| e.at(src)), Some(format!("1:20: {not}")));
    assert_eq!((k.instances.len(), k.count, k.written), (1, 1, false));
    assert!(out.listings.is_empty());
    let src = "extern \"ptx-kernel\" fn g() {}\nextern \"ptx-func\" fn f() {}";
    let out = echoed(src);
    let [ItemOut::Device(g), ItemOut::Device(f)] = &out.items[..] else { panic!("{:?}", out.items) };
    assert_eq!(g.not_unsafe.as_ref().map(|e| e.at(src)), Some(format!("1:24: {not}")));
    assert_eq!(g.launcher, Some(Launcher::Never));
    assert!(f.not_unsafe.is_none());
    let device = out.device.unwrap();
    let made: Vec<_> = device.text.lines().filter(|line| line.starts_with("new ")).collect();
    assert_eq!(made, ["new g: g []", "new f: f []"]);
    assert!(device.symbols.is_empty());
}

/// A function of an ABI kernite compiles nothing for is refused where its ABI is named, and is
/// none of the GPU's.
#[test]
fn a_function_of_an_abi_kernite_does_not_know_is_refused() {
    let src = "unsafe extern \"C\" fn f() {}";
    let out = echoed(src);
    let [ItemOut::Unknown(e)] = &out.items[..] else { panic!("{:?}", out.items) };
    let none = "1:15: kernite compiles `extern \"sysv64\"` kernels, for x86-64, and `extern \"ptx-kernel\"` ones, for \
                NVIDIA GPUs, with the `extern \"ptx-func\"` functions they call; `extern \"C\"` is none of these";
    assert_eq!(e.at(src), none);
    assert!(out.device.is_none());
}

/// A helper is checked where it is called, but what is wrong with it whether it is called or not
/// is said where it is: a body that does not parse, an `#[instances(..)]`, then a
/// `#[cooperative]`, in that order.
#[test]
fn a_helper_says_what_is_wrong_with_it_whether_called_or_not() {
    let src = "#[cooperative] #[instances(1)] fn h() { let = 1; }\n\
               #[cooperative] fn g(x: gpr) { add(x, 1); }\n\
               unsafe extern \"sysv64\" fn k(x: u64) { g(x); }";
    let out = echoed(src);
    let [ItemOut::Helper(h), ItemOut::Helper(g), ItemOut::Native(k)] = &out.items[..] else {
        panic!("{:?}", out.items);
    };
    let said = |errors: &[Error]| errors.iter().map(|e| e.at(src)).collect::<Vec<_>>();
    let not = |name| {
        format!(
            "`#[cooperative]` goes on a kernel whose blocks or threads wait for each other, `extern \"ptx-kernel\"` \
             or `extern \"sysv64\"`; `{name}` is a helper, inlined where it is called"
        )
    };
    let helper = "a helper is compiled where it is called, for what `::<..>` gives it there: `#[instances(..)]` lists \
                  what a function that names an ABI is compiled for";
    assert_eq!(
        said(h),
        ["1:45: expected a name, found `=`".to_string(), format!("1:18: {helper}"), format!("1:3: {}", not("h"))]
    );
    assert_eq!(said(g), [format!("2:3: {}", not("g"))]);
    assert!(k.written);
}

/// A kernel names the statics the code of its instances names, those the functions it calls name,
/// and on, in the module's order: though it calls a function declared after it, which names them
/// in another order. Of a function's instances, it names those of the one it calls alone. Two
/// kernels that call each other name the same, and a kernel is not written to name them.
#[test]
fn a_kernel_names_the_statics_its_callees_reach() {
    let src = "static A: global<u32>;\n\
               static B: global<u32>;\n\
               unsafe extern \"sysv64\" fn outer(x: u64) { call(inner, x); }\n\
               unsafe extern \"sysv64\" fn inner(x: u64) { lea(x, [B]); lea(x, [A]); }\n\
               unsafe extern \"sysv64\" fn top(x: u64) { call(outer, x); }\n\
               #[instances(2, 4)] unsafe extern \"sysv64\" fn tile<const N: u64>(x: u64) {\n\
               if N == 4 { lea(x, [B]); }\n\
               }\n\
               unsafe extern \"sysv64\" fn two(x: u64) { call(tile::<2>, x); }\n\
               unsafe extern \"sysv64\" fn four(x: u64) { call(tile::<4>, x); }\n\
               unsafe extern \"sysv64\" fn ping(x: u64) { call(pong, x); }\n\
               unsafe extern \"sysv64\" fn pong(x: u64) { call(ping, x); lea(x, [A]); }\n\
               extern \"sysv64\" fn plain(x: u64) { lea(x, [A]); }";
    let out = echoed(src);
    let [
        _,
        _,
        ItemOut::Native(outer),
        ItemOut::Native(inner),
        ItemOut::Native(top),
        ItemOut::Native(tile),
        ItemOut::Native(two),
        ItemOut::Native(four),
        ItemOut::Native(ping),
        ItemOut::Native(pong),
        ItemOut::Native(plain),
    ] = &out.items[..]
    else {
        panic!("{:?}", out.items);
    };
    assert_eq!(inner.instances[0].naked.statics, ["B", "A"]);
    assert_eq!(outer.touched, ["A", "B"]);
    assert_eq!(inner.touched, ["A", "B"]);
    assert_eq!(top.touched, ["A", "B"]);
    assert_eq!(tile.touched, ["B"]);
    assert!(two.touched.is_empty());
    assert_eq!(four.touched, ["B"]);
    assert_eq!(ping.touched, ["A"]);
    assert_eq!(pong.touched, ["A"]);
    assert!(plain.touched == ["A"] && !plain.written);
}

/// A kernel names the statics of each function it may call at an address, those taken together
/// last, though its code names none of them: `idle` names none and `run` one, the last of the two.
/// A kernel that calls it by name names them as well.
#[test]
fn a_kernel_names_the_statics_of_the_functions_it_may_call_at_an_address() {
    let src = "static A: global<u32>;\n\
               static B: global<u32>;\n\
               unsafe extern \"sysv64\" fn idle() {}\n\
               unsafe extern \"sysv64\" fn run() { let p: gpr; lea(p, [A]); }\n\
               unsafe extern \"sysv64\" fn far() { let p: gpr; lea(p, [B]); }\n\
               unsafe extern \"sysv64\" fn either(fp: u64) { call(fp, (idle, run)); }\n\
               unsafe extern \"sysv64\" fn top(fp: u64) { call(either, fp); }";
    let out = echoed(src);
    let [_, _, _, ItemOut::Native(run), _, ItemOut::Native(either), ItemOut::Native(top)] = &out.items[..] else {
        panic!("{:?}", out.items);
    };
    let naked = &either.instances[0].naked;
    assert!(naked.funcs.is_empty());
    assert_eq!(naked.reach, ["idle", "run"]);
    assert_eq!(run.touched, ["A"]);
    assert_eq!(either.touched, ["A"]);
    assert_eq!(top.touched, ["A"]);
}

/// In a module of both kinds of kernel and a static, the statics are the GPU's, as the first of
/// them says, and a kernel the process runs that names one, in any of its instances, does not
/// reach it, so it is not written. One that names none is written, and so is one of a module of
/// both kinds and no static, whose statics are the GPU's.
#[test]
fn an_x86_kernel_of_a_module_of_both_kinds_is_written_unless_it_names_a_static() {
    let src = "static S: global<u32>;\n\
               #[instances(1, 2, 3)] unsafe extern \"sysv64\" fn a<const N: u64>(x: u64) {\n\
               if N == 2 { lea(x, [S]); }\n\
               }\n\
               unsafe extern \"sysv64\" fn b(x: u64) { add(x, 1); }\n\
               unsafe extern \"ptx-kernel\" fn g() {}";
    let out = echoed(src);
    assert!(out.owner.is_some_and(|e| e.at(src).starts_with("1:8: a static is memory of the module's kernels")));
    let [ItemOut::DeviceStatic(Ok(_)), ItemOut::Native(a), ItemOut::Native(b), ItemOut::Device(_)] = &out.items[..]
    else {
        panic!("{:?}", out.items);
    };
    assert!(a.not_unsafe.is_none() && a.errors.is_empty() && a.touched.is_empty());
    assert_eq!(a.instances[1].naked.statics, ["S"]);
    assert_eq!((a.written, b.written), (false, true));
    let src = "unsafe extern \"sysv64\" fn c(x: u64) { add(x, 1); }\nunsafe extern \"ptx-kernel\" fn g() {}";
    let out = echoed(src);
    let [ItemOut::Native(c), ItemOut::Device(_)] = &out.items[..] else { panic!("{:?}", out.items) };
    assert!(out.owner.is_none() && c.written);
}

/// Rust launches each instance of each kernel the GPU runs by the index of its symbol, in the
/// module's order. A kernel one of whose instances does not compile is never launched, and takes
/// no index, though the code of those that compile is in the GPU's module; it says why of each
/// that does not, in order. Nor is a kernel whose attributes are wrong, of which nothing is
/// compiled; each says what is wrong with it. A function the kernels call is not launched.
#[test]
fn the_gpu_s_kernels_are_launched_by_index() {
    let src = "unsafe extern \"ptx-kernel\" fn k() {}\n\
               #[instances(2, 4)] unsafe extern \"ptx-kernel\" fn tile<const N: u32>() {}\n\
               #[instances(1)] extern \"ptx-func\" fn f<const N: u32>() {}\n\
               #[instances(1, 2, 3)] unsafe extern \"ptx-kernel\" fn half<const N: u32>() {\n\
               if N >= 2 { let fail_v: u32; }\n\
               }\n\
               unsafe extern \"ptx-kernel\" fn last() {}\n\
               #[instances(1)] unsafe extern \"ptx-kernel\" fn once() {}";
    let out = echoed(src);
    let functions: Vec<_> = out
        .items
        .iter()
        .map(|item| match item {
            ItemOut::Device(d) => d,
            item => panic!("{item:?}"),
        })
        .collect();
    let launchers: Vec<_> = functions.iter().map(|d| d.launcher.as_ref()).collect();
    let n = |n| vec![Value::Int(n, Some(IntTy::U32))];
    let tile = Launcher::Of(vec![(n(2), 1), (n(4), 2)]);
    let want = [
        Some(&Launcher::Of(vec![(vec![], 0)])),
        Some(&tile),
        None,
        Some(&Launcher::Never),
        Some(&Launcher::Of(vec![(vec![], 3)])),
        Some(&Launcher::Never),
    ];
    assert_eq!(launchers, want);
    let [_, _, _, half, _, once] = &functions[..] else { panic!("{:?}", out.items) };
    let said = "`once` takes no compile-time arguments, so it is compiled once, with no `#[instances(..)]`";
    assert_eq!(once.said.as_ref().map(|e| e.at(src)), Some(format!("8:3: {said}")));
    let errors: Vec<_> = half.errors.iter().map(|e| e.at(src)).collect();
    assert_eq!(errors, ["5:17: in `half::<2>`: `fail_v` fails", "5:17: in `half::<3>`: `fail_v` fails"]);
    let device = out.device.unwrap();
    assert_eq!(device.symbols, ["k", "tile::<2>", "tile::<4>", "last"]);
    let made: Vec<_> = device.text.lines().filter(|line| line.starts_with("new ")).collect();
    let all = [
        "k: k []",
        "tile::<2>: tile [2]",
        "tile::<4>: tile [4]",
        "f::<1>: f [1]",
        "half::<1>: half [1]",
        "last: last []",
    ];
    assert_eq!(made, all.map(|line| format!("new {line}")));
}

/// Why the GPU's module is none is said at the module's first item that is the GPU's: a function
/// it runs, whatever is wrong with it, or a static of its memory. A helper, and a function of an
/// ABI kernite does not know, are not the GPU's.
#[test]
fn the_module_s_error_goes_at_its_first_gpu_item() {
    let src = "fn h() {}\n\
               extern \"C\" fn c() {}\n\
               #[instances(1)] extern \"ptx-kernel\" fn k() {}\n\
               unsafe extern \"ptx-kernel\" fn j() { let fail_v: u32; }";
    let device = echoed(src).device.unwrap();
    assert_eq!(device.error.map(|e| e.at(src)), Some("3:40: nothing to load".into()));
    assert!(device.text.is_empty());
    let src = "extern \"ptx-func\" fn f() { let fail_v: u32; }\nstatic A: global<u32>;\nstatic B: global<u32>;";
    let device = echoed(src).device.unwrap();
    assert_eq!(device.error.map(|e| e.at(src)), Some("1:22: nothing to load".into()));
    let src = "static A: global<u32>;";
    let out = echoed(src);
    let [ItemOut::DeviceStatic(Ok(a))] = &out.items[..] else { panic!("{:?}", out.items) };
    assert_eq!(a.code, DeviceStatic { symbol: "A".into(), host: "u32".into() });
    assert_eq!(out.device.unwrap().error.map(|e| e.at(src)), Some("1:8: nothing to load".into()));
}

/// A function a kernel's code calls, or whose code it takes, as `[tile::<4>]`, is named once
/// however often the code names it: by its function and, for one that takes compile-time
/// arguments, by the index of the instance among those its `#[instances(..)]` lists, not by its
/// place among their values sorted, nor by its value. A kernel Rust does not get the code of names
/// none.
#[test]
fn a_call_reaches_the_instance_its_code_names() {
    let src = "unsafe extern \"sysv64\" fn step(x: u64) { add(x, 1); }\n\
               #[instances(4, 2)] unsafe extern \"sysv64\" fn tile<const N: u64>(x: u64) { add(x, N); }\n\
               unsafe extern \"sysv64\" fn twice(x: u64) { call(tile::<2>, x); call(step, x); }\n\
               unsafe extern \"sysv64\" fn address(x: u64) { lea(x, [tile::<4>]); lea(x, [tile::<4>]); }\n\
               extern \"sysv64\" fn plain(x: u64) { call(step, x); }";
    let out = echoed(src);
    let [_, _, ItemOut::Native(twice), ItemOut::Native(address), ItemOut::Native(plain)] = &out.items[..] else {
        panic!("{:?}", out.items);
    };
    let call = |function: &str, instance| Call { function: function.into(), instance };
    assert_eq!(twice.instances[0].calls, [call("tile", Some(1)), call("step", None)]);
    assert_eq!(address.instances[0].calls, [call("tile", Some(0))]);
    assert!(plain.instances[0].calls.is_empty() && !plain.written);
}

/// A function that takes no compile-time arguments and lists instances is wrong at its own item,
/// which says so. A kernel that calls it still names it, whole, as it names any function of none.
#[test]
fn a_call_to_a_function_with_instances_it_cannot_have_names_it_whole() {
    let src = "#[instances(2)] unsafe extern \"sysv64\" fn g(x: u64) { add(x, 1); }\n\
               unsafe extern \"sysv64\" fn k(x: u64) { call(g, x); }";
    let out = echoed(src);
    let [ItemOut::Native(g), ItemOut::Native(k)] = &out.items[..] else { panic!("{:?}", out.items) };
    let wrong = "1:3: `g` takes no compile-time arguments, so it is compiled once, with no `#[instances(..)]`";
    assert_eq!(g.said.as_ref().map(|e| e.at(src)), Some(wrong.into()));
    assert_eq!(k.instances[0].calls, [Call { function: "g".into(), instance: None }]);
}

/// Each item of a module of consts is a const known now, a const that is not, with why, or no
/// const; a const is worked out on its own, `pub` or not, whichever of the consts it reads comes
/// first.
#[test]
fn a_module_of_consts_says_what_each_of_its_items_is() {
    let src = "#![allow(dead_code)]\n\
               pub const A: u64 = 3;\n\
               const B: u64 = A * 2;\n\
               pub const C: u64 = double(3);\n\
               pub const D: u64 = E + 1;\n\
               const E: u64 = 4;\n\
               pub const fn double(x: u64) -> u64 { 2 * x }";
    let said: Vec<String> = consts(&parse(src).unwrap())
        .iter()
        .map(|item| match item {
            ConstItem::Given => "given".to_string(),
            ConstItem::Wrong(e) => e.at(src),
            ConstItem::NotAConst => "not a const".to_string(),
        })
        .collect();
    let call = "4:20: a call is a statement: an instruction or a helper, not a value";
    assert_eq!(said, ["not a const", "given", "given", call, "given", "given", "not a const"]);
}
