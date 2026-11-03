//! What the evaluator asks of its target, in order. [`Trace`] is a fake target that writes a line
//! for each call, or keeps what it is told, so these tests pin the calls, and what each is given,
//! with no machine.

use kernite_lang::{Error, Fault, Guard, ItemKind, Mem, Operand, Run, Span, Target};

/// A kernel with a pointer, a `ymm`, a loop of `v.xmm` fields, which the evaluator unrolls, a
/// label and a helper, whose parameter is a view of what it is given, and which declares a
/// variable of its own. [`Trace`] is a machine of its own, which takes every type and instruction.
/// Each variable is written before it is read: `add` writes `t`.
const KERNEL: &str = r#"
pub unsafe extern "sysv64" fn k(xs: *const f32, n: u64) -> u64 {
    let v: ymm;
    for i in 0..2 {
        vmovups(v.xmm, [xs + i * 16]);
    }
    label(top);
    twice(n);
    jnz(top);
    ret(n);
}

fn twice(x: u64) {
    note("twice");
    let t: u64;
    add(t, x, x);
}
"#;

/// A kernel that declares its variables in an array.
const ARRAY: &str = r#"
pub unsafe extern "sysv64" fn k() {
    let a: [u32; 3];
}
"#;

/// A module of two statics, and a kernel that names them after an `add`: `B`, then `A`, then `B`
/// again, in an address.
const STATICS: &str = r#"
static A: global<u64>;
static B: global<u64>;

pub unsafe extern "sysv64" fn k(n: u64) {
    add(n, n, n);
    mov(n, B);
    mov(n, A);
    mov(n, [B + 8]);
}
"#;

/// A target that writes a line for each call but `declared_at` and `inst_at`, whose places it
/// keeps, makes a variable for each `param`, `var`, `global`, `view` and `field`, and refuses one
/// instruction, by its name, with a given fault.
#[derive(Default)]
struct Trace {
    /// A line for each call, in the order of the calls.
    lines: Vec<String>,
    /// How many variables it has made: the next is `v` and this number.
    vars: usize,
    /// The instruction it refuses, and why.
    refuse: Option<(&'static str, Fault)>,
    /// Each variable it is told is declared, with where, in the order it is told.
    declared: Vec<(usize, Span)>,
    /// The place it was last told of an instruction, until `inst` takes it.
    at: Option<Span>,
    /// Each instruction it is given, with the place it was told of before: `None` if it was
    /// told of none.
    placed: Vec<(String, Option<Span>)>,
}

impl Trace {
    /// Makes the next variable, and writes `what` with its name: `var v: ymm = v2`.
    fn make(&mut self, what: String) -> usize {
        self.lines.push(format!("{what} = v{}", self.vars));
        self.vars += 1;
        self.vars - 1
    }
}

impl Target for Trace {
    type Var = usize;

    fn param(&mut self, name: &str, ty: &str) -> Result<usize, String> {
        Ok(self.make(format!("param {name}: {ty}")))
    }

    fn result(&mut self, ty: &str) -> Result<(), String> {
        self.lines.push(format!("result {ty}"));
        Ok(())
    }

    fn var(&mut self, name: &str, ty: &str) -> Result<usize, String> {
        Ok(self.make(format!("var {name}: {ty}")))
    }

    fn global(&mut self, name: &str) -> Result<usize, String> {
        Ok(self.make(format!("global {name}")))
    }

    fn view(&mut self, var: usize, view: &str) -> Result<usize, String> {
        Ok(self.make(format!("view v{var} as {view}")))
    }

    fn field(&mut self, var: usize, field: &str) -> Result<usize, String> {
        Ok(self.make(format!("field v{var}.{field}")))
    }

    fn label(&mut self, label: u32) {
        self.lines.push(format!("label {label}"));
    }

    fn note(&mut self, text: &str) {
        self.lines.push(format!("note {text}"));
    }

    fn inst(&mut self, name: &str, args: &[Operand<usize>], guard: Option<Guard<usize>>) -> Result<(), Fault> {
        self.placed.push((name.to_string(), self.at.take()));
        let mut line = match guard {
            Some(g) => format!("if {}v{}: {name}", if g.negated { "!" } else { "" }, g.pred),
            None => name.to_string(),
        };
        if !args.is_empty() {
            line += &format!(" {}", args.iter().map(operand).collect::<Vec<_>>().join(", "));
        }
        self.lines.push(line);
        match &self.refuse {
            Some((refused, fault)) if *refused == name => Err(fault.clone()),
            _ => Ok(()),
        }
    }

    fn end(&mut self) -> Result<(), String> {
        self.lines.push("end".into());
        Ok(())
    }

    fn declared_at(&mut self, var: usize, at: Span) {
        self.declared.push((var, at));
    }

    fn inst_at(&mut self, at: Span) {
        self.at = Some(at);
    }
}

/// An operand as the trace writes it: `v3`, `[v0 + 16]`, `label 0`, and any other as Rust's
/// `Debug` writes it.
fn operand(o: &Operand<usize>) -> String {
    match o {
        Operand::Var(v) => format!("v{v}"),
        Operand::Label(l) => format!("label {l}"),
        Operand::Mem(Mem { base, index, disp }) => {
            let mut sum: Vec<String> = base.iter().map(|b| format!("v{b}")).collect();
            sum.extend(index.iter().map(|(i, k)| format!("v{i} * {k}")));
            if *disp != 0 || sum.is_empty() {
                sum.push(disp.to_string());
            }
            format!("[{}]", sum.join(" + "))
        }
        other => format!("{other:?}"),
    }
}

/// Runs the kernel `k` of `src` on `trace`; or the error, at its line and column.
fn run(src: &str, trace: &mut Trace) -> Result<Run, String> {
    let module = kernite_lang::parse(src).map_err(|e| e.at(src))?;
    let k = module.items.iter().find_map(|item| match &item.kind {
        ItemKind::Fn(f) if f.name == "k" => Some(f),
        _ => None,
    });
    kernite_lang::kernel(&module, k.expect("a kernel `k`"), &[], trace).map_err(|e| e.at(src))
}

#[test]
fn the_evaluator_asks_its_target_in_source_order() {
    let mut trace = Trace::default();
    run(KERNEL, &mut trace).unwrap();
    let want = [
        "param xs: *const f32 = v0",
        "param n: u64 = v1",
        "result u64",
        "var v: ymm = v2",
        "field v2.xmm = v3",
        "vmovups v3, [v0]",
        "field v2.xmm = v4",
        "vmovups v4, [v0 + 16]",
        "label 0",
        "view v1 as u64 = v5",
        "note twice",
        "var t: u64 = v6",
        "add v6, v5, v5",
        "jnz label 0",
        "ret v1",
        "end",
    ];
    assert_eq!(trace.lines, want);
}

#[test]
fn a_static_is_made_once_the_first_time_the_kernel_names_it() {
    let mut trace = Trace::default();
    run(STATICS, &mut trace).unwrap();
    let want = [
        "param n: u64 = v0",
        "add v0, v0, v0",
        "global B = v1",
        "mov v0, v1",
        "global A = v2",
        "mov v0, v2",
        "mov v0, [v1 + 8]",
        "end",
    ];
    assert_eq!(trace.lines, want);
}

#[test]
fn each_declaration_is_one_param_or_var() {
    // The target is told where each declaration is, by the variable it made for it: `2:33: v0`.
    // A helper's parameter is a view of what it is given, so not a declaration. An array is a
    // declaration for each of its variables.
    let told = |src: &str| -> Vec<String> {
        let mut trace = Trace::default();
        run(src, &mut trace).unwrap();
        trace.declared.into_iter().map(|(v, span)| Error { span, message: format!("v{v}") }.at(src)).collect()
    };
    assert_eq!(told(KERNEL), ["2:33: v0", "2:49: v1", "3:9: v2", "15:9: v6"]);
    assert_eq!(told(ARRAY), ["3:9: v0", "3:9: v1", "3:9: v2"]);
}

#[test]
fn the_target_is_told_where_each_instruction_is_before_it_is_given_it() {
    // An instruction is at its name. One in a loop is at the same place each time it runs, and one
    // of a helper is in the helper.
    let told = |src: &str| -> Vec<String> {
        let mut trace = Trace::default();
        run(src, &mut trace).unwrap();
        let place = |(name, at): (String, Option<Span>)| match at {
            Some(span) => Error { span, message: name }.at(src),
            None => format!("{name} is not placed"),
        };
        trace.placed.into_iter().map(place).collect()
    };
    assert_eq!(told(KERNEL), ["5:9: vmovups", "5:9: vmovups", "16:5: add", "9:5: jnz", "10:5: ret"]);
}

#[test]
fn a_fault_points_at_the_operand_it_names() {
    let refused = |name: &'static str, fault: Fault| {
        let mut trace = Trace { refuse: Some((name, fault)), ..Trace::default() };
        run(KERNEL, &mut trace).unwrap_err()
    };
    assert_eq!(refused("add", Fault::at(1, "no")), "16:12: no");
    assert_eq!(refused("vmovups", Fault::at(0, "no")), "5:19: no");
}
