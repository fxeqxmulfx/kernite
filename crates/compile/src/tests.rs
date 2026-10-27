//! The use cases on a toy machine: a few general registers, some callee-saved, and opcodes
//! that are only names, which [`run`] runs.

use std::collections::BTreeMap;

use kgen_core::{
    Access, Arg, Base, Class, Flow, Imm, Inst, Item, Label, Mem, Operand, PReg, Program, RegInfo, Register, Signature,
    Slot, SlotId, VReg, Width,
};

use crate::{Allocated, Error, Machine, Spill, compile};

struct Toy {
    gpr: Vec<PReg>,
    saved: Vec<PReg>,
}

impl Machine for Toy {
    fn allocatable(&self, class: Class) -> &[PReg] {
        match class {
            Class::Gpr => &self.gpr,
            Class::Vec | Class::Pred | Class::Float => &[],
        }
    }

    fn callee_saved(&self) -> &[PReg] {
        &self.saved
    }
}

fn slot_arg<R>(slot: SlotId, access: Access) -> Arg<R> {
    Arg { operand: Operand::Mem(Mem { base: Base::Slot(slot), index: None, disp: 0, width: Width::W64 }), access }
}

fn reg_arg<R>(r: R, access: Access) -> Arg<R> {
    Arg { operand: Operand::Reg(r, Width::W64), access }
}

/// An instruction of the toy's that goes on to the next and changes nothing but its operands.
fn plain<R>(op: &'static str, args: Vec<Arg<R>>) -> Inst<&'static str, R> {
    Inst { op, args, implicit: vec![], flow: Flow::Next, clobbers: vec![] }
}

impl Spill<&'static str> for Toy {
    fn width(&self, _: Class) -> Width {
        Width::W64
    }

    fn store(&self, slot: SlotId, var: VReg) -> Inst<&'static str, VReg> {
        plain("store", vec![slot_arg(slot, Access::Write), reg_arg(var, Access::Read)])
    }

    fn load(&self, var: VReg, slot: SlotId) -> Inst<&'static str, VReg> {
        plain("load", vec![reg_arg(var, Access::Write), slot_arg(slot, Access::Read)])
    }

    fn copy(&self, to: VReg, from: VReg) -> Inst<&'static str, VReg> {
        plain("copy", vec![reg_arg(to, Access::Write), reg_arg(from, Access::Read)])
    }
}

/// The FNV hash of `values`, which `op` and `call` write.
fn hash(values: impl IntoIterator<Item = u64>) -> u64 {
    values.into_iter().fold(0xcbf2_9ce4_8422_2325u64, |h, x| (h ^ x).wrapping_mul(0x100_0000_01b3))
}

/// What `body` returns, run from registers that hold `params` and `slots` slots of memory
/// that hold zero: `op` writes a hash of its immediate and the registers it reads, `add` adds
/// the second to the first, `dec` takes one from its register and jumps unless that leaves
/// zero, `load` and `store` move a register's value to a slot and back, `copy` moves one to
/// another, `call` writes a hash of the arguments it reads to its result, if it has one, and
/// `ret` returns what it reads. Reading a register nothing wrote, or running past ten thousand
/// instructions, panics.
///
/// A call writes junk to each of the machine's registers it changes, its `clobbers`, that
/// `reg` says is a register of `R`'s, as a callee may; the result goes in after. So a value
/// kept in one across a call is lost, and the run returns something else.
fn run<R: Register + Ord>(
    body: &[Item<&'static str, R>],
    params: &[(R, u64)],
    slots: usize,
    reg: fn(PReg) -> Option<R>,
) -> u64 {
    let mut regs: BTreeMap<R, u64> = params.iter().copied().collect();
    let mut mem = vec![0u64; slots];
    let place = |l: Label| body.iter().position(|item| *item == Item::Label(l)).expect("placed");
    let mut at = 0;
    let mut calls = 0;
    for _ in 0..10_000 {
        let Item::Inst(inst) = &body[at] else {
            at += 1;
            continue;
        };
        let value = |a: &Arg<R>, regs: &BTreeMap<R, u64>| match a.operand {
            Operand::Reg(r, _) => *regs.get(&r).unwrap_or_else(|| panic!("{r:?} is read before anything writes it")),
            Operand::Mem(Mem { base: Base::Slot(s), .. }) => mem[s.0 as usize],
            Operand::Imm(imm) => imm.value as u64,
            _ => unreachable!("the toy has no {:?}", a.operand),
        };
        let reg_of = |a: &Arg<R>| match a.operand {
            Operand::Reg(r, _) => r,
            _ => unreachable!("{:?} is no register", a.operand),
        };
        let mut jump = false;
        match inst.op {
            "op" => {
                let read = inst.args[1..].iter().filter(|a| a.access == Access::Read);
                let h = hash(read.map(|a| value(a, &regs)));
                regs.insert(reg_of(&inst.args[0]), h);
            }
            "add" => {
                let sum = value(&inst.args[0], &regs).wrapping_add(value(&inst.args[1], &regs));
                regs.insert(reg_of(&inst.args[0]), sum);
            }
            "dec" => {
                let left = value(&inst.args[0], &regs) - 1;
                regs.insert(reg_of(&inst.args[0]), left);
                jump = left != 0;
            }
            "load" | "copy" => {
                let v = value(&inst.args[1], &regs);
                regs.insert(reg_of(&inst.args[0]), v);
            }
            "store" => {
                let Operand::Mem(Mem { base: Base::Slot(s), .. }) = inst.args[0].operand else { unreachable!() };
                mem[s.0 as usize] = value(&inst.args[1], &regs);
            }
            "call" => {
                let h = hash(inst.implicit.iter().filter(|a| a.access == Access::Read).map(|a| value(a, &regs)));
                calls += 1;
                for (k, &c) in inst.clobbers.iter().enumerate() {
                    if let Some(c) = reg(c) {
                        regs.insert(c, 0xdead_0000 + calls * 64 + k as u64);
                    }
                }
                if let Some(result) = inst.implicit.iter().find(|a| a.access == Access::Write) {
                    regs.insert(reg_of(result), h);
                }
            }
            "ret" => return value(inst.operands().next().expect("a result"), &regs),
            op => unreachable!("the toy does not run `{op}`"),
        }
        at = match inst.flow {
            Flow::Next => at + 1,
            Flow::Branch(l) if jump => place(l),
            Flow::Branch(_) => at + 1,
            _ => unreachable!("the toy runs no {:?}", inst.flow),
        };
    }
    panic!("it runs on past ten thousand instructions");
}

/// What `p` returns for the parameters `args`, and what it returns compiled for `m`, which
/// must be the same.
fn same(p: &Program<&'static str>, m: &Toy, args: &[u64]) -> Allocated<&'static str> {
    let c = compile(p, m).unwrap();
    let before: Vec<(VReg, u64)> = p.params.iter().map(|&(v, _)| v).zip(args.iter().copied()).collect();
    let after: Vec<(PReg, u64)> = p.params.iter().map(|&(_, r)| r).zip(args.iter().copied()).collect();
    assert_eq!(
        run(&c.body, &after, c.frame.slots.len(), Some),
        run(&p.body, &before, p.slots.len(), |_| None),
        "for {args:?}"
    );
    c
}

/// The names of the variables `c` spilled, in the order they did.
fn spilled(c: &Allocated<&'static str>) -> Vec<String> {
    c.frame.slots.iter().filter(|s| s.spill).map(|s| s.name.clone()).collect()
}

/// The opcodes of `c`, in order.
fn ops(c: &Allocated<&'static str>) -> Vec<&'static str> {
    c.body.iter().filter_map(|i| if let Item::Inst(i) = i { Some(i.op) } else { None }).collect()
}

fn r(hw: u8) -> PReg {
    PReg { class: Class::Gpr, hw }
}

/// `n` registers, the ones in `saved` callee-saved.
fn toy(n: u8, saved: &[u8]) -> Toy {
    Toy { gpr: (0..n).map(r).collect(), saved: saved.iter().map(|&h| r(h)).collect() }
}

struct Builder(Program<&'static str>, i64);

impl Builder {
    fn new() -> Self {
        Builder(
            Program {
                name: "toy".into(),
                signature: Signature { params: vec![], result: None },
                regs: vec![],
                params: vec![],
                results: vec![],
                pinned: vec![],
                body: vec![],
                slots: vec![],
                consts: vec![],
                funcs: vec![],
                statics: vec![],
            },
            0,
        )
    }

    fn var(&mut self, name: &str) -> VReg {
        self.0.regs.push(RegInfo { class: Class::Gpr, name: name.into() });
        VReg { id: self.0.regs.len() as u32 - 1, class: Class::Gpr }
    }

    fn param(&mut self, name: &str, hw: u8) -> VReg {
        let v = self.var(name);
        self.0.params.push((v, r(hw)));
        self.0.signature.params.push(Class::Gpr);
        v
    }

    fn inst(&mut self, op: &'static str, args: &[(VReg, Access)], flow: Flow) {
        let args = args.iter().map(|&(v, access)| reg_arg(v, access)).collect();
        self.0.body.push(Item::Inst(Inst { op, args, implicit: vec![], flow, clobbers: vec![] }));
    }

    /// `dst = op(srcs)`, with an immediate of its own, so that no two compute the same.
    fn def(&mut self, dst: VReg, srcs: &[VReg]) {
        self.1 += 1;
        let mut args = vec![reg_arg(dst, Access::Write)];
        args.push(Arg { operand: Operand::Imm(Imm { value: self.1, hex: false }), access: Access::Read });
        args.extend(srcs.iter().map(|&s| reg_arg(s, Access::Read)));
        self.0.body.push(Item::Inst(plain("op", args)));
    }

    /// A call that changes the registers `clobbers`, as a machine's code makes one: each of
    /// `args` copied to a variable of its own, pinned in the register given with it, and the
    /// result, if `dst` says where it goes, given back in a variable pinned in the register
    /// given with it and copied to `dst`.
    fn call(&mut self, dst: Option<(VReg, u8)>, args: &[(VReg, u8)], clobbers: &[u8]) {
        let mut implicit = vec![];
        for &(a, hw) in args {
            let name = self.0.regs[a.id as usize].name.clone();
            let pinned = self.var(&name);
            self.0.pinned.push((pinned, r(hw)));
            self.0.body.push(Item::Inst(Toy::copy(&toy(0, &[]), pinned, a)));
            implicit.push(reg_arg(pinned, Access::Read));
        }
        let back = dst.map(|(d, hw)| {
            let name = self.0.regs[d.id as usize].name.clone();
            let pinned = self.var(&name);
            self.0.pinned.push((pinned, r(hw)));
            implicit.push(reg_arg(pinned, Access::Write));
            (d, pinned)
        });
        let clobbers = clobbers.iter().map(|&h| r(h)).collect();
        self.0.body.push(Item::Inst(Inst { op: "call", args: vec![], implicit, flow: Flow::Next, clobbers }));
        if let Some((d, pinned)) = back {
            self.0.body.push(Item::Inst(Toy::copy(&toy(0, &[]), d, pinned)));
        }
    }

    fn label(&mut self, l: u32) {
        self.0.body.push(Item::Label(Label(l)));
    }

    fn ret(&mut self, v: VReg, hw: u8) {
        self.0.results.push((v, r(hw)));
        self.0.signature.result = Some(Class::Gpr);
        self.inst("ret", &[(v, Access::Read)], Flow::Return);
    }
}

/// The register of the first variable named `name`: the program's own, before any that
/// splitting or spilling made of it.
fn reg_of(c: &crate::Allocated<&'static str>, name: &str) -> PReg {
    c.assignment.iter().find(|(info, _)| info.name == name).and_then(|(_, r)| *r).unwrap()
}

/// The registers of every variable named `name`, in order.
fn regs_of(c: &crate::Allocated<&'static str>, name: &str) -> Vec<PReg> {
    c.assignment.iter().filter(|(info, _)| info.name == name).filter_map(|(_, r)| *r).collect()
}

#[test]
fn reuses_a_register_once_its_value_is_dead() {
    let mut b = Builder::new();
    let (x, y, z) = (b.var("x"), b.var("y"), b.var("z"));
    b.def(x, &[]);
    b.def(y, &[x]);
    b.def(z, &[y]);
    b.ret(z, 0);
    let c = compile(&b.0, &toy(1, &[])).unwrap();
    assert_eq!([reg_of(&c, "x"), reg_of(&c, "y"), reg_of(&c, "z")], [r(0); 3]);
}

#[test]
fn an_instruction_that_reads_two_values_needs_two_registers_whatever_spills() {
    let mut b = Builder::new();
    let (x, y, z) = (b.var("x"), b.var("y"), b.var("z"));
    b.0.body.push(Item::Note("sum".into()));
    b.def(x, &[]);
    b.def(y, &[]);
    b.def(z, &[x, y]);
    b.ret(z, 0);
    let err = compile(&b.0, &toy(1, &[])).unwrap_err();
    assert_eq!(
        err,
        Error::OutOfRegisters {
            class: Class::Gpr,
            var: "y".into(),
            near: Some("sum".into()),
            taken: vec![("x".into(), r(0))],
            changed: vec![],
            have: 1,
        }
    );
    let c = compile(&b.0, &toy(2, &[])).unwrap();
    assert_ne!(reg_of(&c, "x"), reg_of(&c, "y"));
}

#[test]
fn a_value_live_around_a_loop_keeps_its_register_inside() {
    let mut b = Builder::new();
    let n = b.param("n", 1);
    let (acc, t) = (b.var("acc"), b.var("t"));
    b.def(acc, &[]);
    b.label(2);
    b.def(t, &[n]);
    b.inst("add", &[(acc, Access::ReadWrite), (t, Access::Read)], Flow::Next);
    b.inst("dec", &[(n, Access::ReadWrite)], Flow::Branch(Label(2)));
    b.ret(acc, 0);
    let c = compile(&b.0, &toy(3, &[])).unwrap();
    let (acc, t, n) = (reg_of(&c, "acc"), reg_of(&c, "t"), reg_of(&c, "n"));
    assert!(acc != t && t != n && acc != n, "{acc:?} {t:?} {n:?}");
    // Without the back edge `acc` would be dead in the loop's first half; with it, three
    // values are live at once. In two registers `acc` spills, then `t`, and the add still needs
    // both beside `n`, which may not spill: the instruction that writes it jumps, and leaves no
    // place for its store.
    let short = Error::OutOfRegisters {
        class: Class::Gpr,
        var: "t".into(),
        near: None,
        taken: vec![("n".into(), r(1)), ("acc".into(), r(0))],
        changed: vec![],
        have: 2,
    };
    assert_eq!(compile(&b.0, &toy(2, &[])).unwrap_err(), short);
}

#[test]
fn a_parameter_and_a_result_share_a_register_when_they_do_not_overlap() {
    let mut b = Builder::new();
    let p = b.param("p", 0);
    let q = b.var("q");
    b.def(q, &[p]);
    b.ret(q, 0);
    let c = same(&b.0, &toy(1, &[]), &[5]);
    assert_eq!((reg_of(&c, "q"), ops(&c)), (r(0), vec!["op", "ret"]));

    // When they overlap, the result moves to its register just before the return.
    let mut b = Builder::new();
    let p = b.param("p", 0);
    let (q, u) = (b.var("q"), b.var("u"));
    b.def(q, &[p]);
    b.def(u, &[p]);
    b.ret(q, 0);
    let c = same(&b.0, &toy(2, &[]), &[5]);
    assert_eq!(regs_of(&c, "q"), [r(1), r(0)]);
    assert_eq!(ops(&c), ["op", "op", "copy", "ret"]);
}

#[test]
fn a_read_before_any_write_is_an_error_but_an_idiom_is_not() {
    let mut b = Builder::new();
    let x = b.var("x");
    b.inst("inc", &[(x, Access::ReadWrite)], Flow::Next);
    b.ret(x, 0);
    assert_eq!(compile(&b.0, &toy(1, &[])).unwrap_err(), Error::ReadBeforeWrite { var: "x".into() });

    let mut b = Builder::new();
    let x = b.var("x");
    b.inst("xor", &[(x, Access::Write), (x, Access::Ignored), (x, Access::Ignored)], Flow::Next);
    b.ret(x, 0);
    assert!(compile(&b.0, &toy(1, &[])).is_ok());
}

#[test]
fn a_value_one_path_reads_is_live_across_the_branch() {
    let mut b = Builder::new();
    let n = b.param("n", 0);
    let (x, y) = (b.var("x"), b.var("y"));
    b.def(x, &[]);
    b.inst("test", &[(n, Access::Read)], Flow::Branch(Label(3)));
    b.def(y, &[]);
    b.ret(y, 1);
    b.label(3);
    b.ret(x, 1);
    let c = compile(&b.0, &toy(3, &[])).unwrap();
    assert_ne!(reg_of(&c, "x"), reg_of(&c, "n"));
    assert_eq!(reg_of(&c, "y"), r(1));
}

#[test]
fn a_jump_through_a_table_goes_to_any_of_its_labels() {
    // To label 2 as well as to label 1, past the write of `x`.
    let mut b = Builder::new();
    let n = b.param("n", 0);
    let x = b.var("x");
    b.inst("jmp", &[(n, Access::Read)], Flow::Table { to: vec![Label(1), Label(2)], or_next: false });
    b.label(1);
    b.def(x, &[]);
    b.label(2);
    b.ret(x, 1);
    assert_eq!(compile(&b.0, &toy(2, &[])).unwrap_err(), Error::ReadBeforeWrite { var: "x".into() });

    // And to the next item, where it may.
    let mut b = Builder::new();
    let n = b.param("n", 0);
    let x = b.var("x");
    b.inst("jmp", &[(n, Access::Read)], Flow::Table { to: vec![Label(1)], or_next: true });
    b.inst("use", &[(x, Access::Read)], Flow::Next);
    b.label(1);
    b.def(x, &[]);
    b.ret(x, 1);
    assert_eq!(compile(&b.0, &toy(2, &[])).unwrap_err(), Error::ReadBeforeWrite { var: "x".into() });
    let Item::Inst(jmp) = &mut b.0.body[0] else { unreachable!("the jump is first") };
    jmp.flow = Flow::Table { to: vec![Label(1)], or_next: false };
    assert!(compile(&b.0, &toy(2, &[])).is_ok());
}

#[test]
fn callee_saved_registers_are_saved_only_when_used() {
    let mut b = Builder::new();
    let (x, y, z) = (b.var("x"), b.var("y"), b.var("z"));
    b.def(x, &[]);
    b.def(y, &[]);
    b.def(z, &[x, y]);
    b.ret(z, 0);
    assert_eq!(compile(&b.0, &toy(3, &[2])).unwrap().frame.saved, []);

    let mut b = Builder::new();
    let (x, y, z, w) = (b.var("x"), b.var("y"), b.var("z"), b.var("w"));
    b.def(x, &[]);
    b.def(y, &[]);
    b.def(z, &[]);
    b.def(w, &[x, y, z]);
    b.ret(w, 0);
    assert_eq!(compile(&b.0, &toy(3, &[2])).unwrap().frame.saved, [r(2)]);
}

#[test]
fn slots_are_laid_out_at_their_alignment() {
    let mut b = Builder::new();
    let x = b.var("x");
    b.def(x, &[]);
    b.ret(x, 0);
    let slot = |name: &str, size, align| Slot { name: name.into(), size, align, spill: false };
    b.0.slots = vec![slot("a", 4, 4), slot("b", 32, 32), slot("c", 8, 8)];
    let frame = compile(&b.0, &toy(1, &[])).unwrap().frame;
    assert_eq!((frame.offsets, frame.size, frame.align), (vec![0, 32, 64], 96, 32));
    assert_eq!(frame.slots, b.0.slots);
}

#[test]
fn control_must_stay_inside_the_body() {
    let mut b = Builder::new();
    let x = b.var("x");
    b.def(x, &[]);
    assert_eq!(compile(&b.0, &toy(1, &[])).unwrap_err(), Error::FallsOffEnd);
    b.inst("jmp", &[], Flow::Jump(Label(7)));
    assert_eq!(compile(&b.0, &toy(1, &[])).unwrap_err(), Error::NoSuchLabel { label: 7 });
    b.label(7);
    b.label(7);
    b.ret(x, 0);
    assert_eq!(compile(&b.0, &toy(1, &[])).unwrap_err(), Error::LabelTwice { label: 7 });
}

#[test]
fn three_values_in_two_registers_spill_the_longest_lived() {
    let mut b = Builder::new();
    let (x, y, z, w, v) = (b.var("x"), b.var("y"), b.var("z"), b.var("w"), b.var("v"));
    b.def(x, &[]);
    b.def(y, &[]);
    b.def(z, &[]);
    b.def(w, &[x, y]);
    b.def(v, &[w, z]);
    b.ret(v, 0);
    let c = same(&b.0, &toy(2, &[]), &[]);
    assert_eq!(spilled(&c), ["x", "z"]);
    let ops: Vec<&str> = c.body.iter().filter_map(|i| if let Item::Inst(i) = i { Some(i.op) } else { None }).collect();
    assert_eq!(ops, ["op", "store", "op", "op", "store", "load", "op", "load", "op", "ret"]);
    assert!(spilled(&same(&b.0, &toy(3, &[]), &[])).is_empty());
}

#[test]
fn accumulators_around_a_loop_spill_rather_than_what_a_turn_loads() {
    let mut b = Builder::new();
    let n = b.param("n", 1);
    let acc = [b.var("a0"), b.var("a1"), b.var("a2")];
    for a in acc {
        b.def(a, &[]);
    }
    b.label(1);
    for a in acc {
        let t = b.var("t");
        b.def(t, &[n]);
        b.inst("add", &[(a, Access::ReadWrite), (t, Access::Read)], Flow::Next);
    }
    b.inst("dec", &[(n, Access::ReadWrite)], Flow::Branch(Label(1)));
    let sum = b.var("sum");
    b.def(sum, &acc);
    b.ret(sum, 0);
    for turns in 1..=5 {
        let c = same(&b.0, &toy(4, &[]), &[turns]);
        assert_eq!(spilled(&c), ["a0", "a1"]);
    }
    assert!(spilled(&same(&b.0, &toy(5, &[]), &[3])).is_empty());
}

/// A load or a store in a loop counts ten times one outside it. `inside` is read and written
/// once a turn; `outside`, `k` times before the loop: each is alive across it, beside the count,
/// in two registers. With 9, `outside` costs less for how long it is alive, and spills first,
/// as it would not if a turn counted 9 times; with 10, `inside` does, as it would not if a turn
/// counted 11 times.
#[test]
fn one_in_a_loop_counts_ten_times_one_outside() {
    let first = |k: usize| {
        let mut b = Builder::new();
        let n = b.param("n", 0);
        let (inside, outside) = (b.var("inside"), b.var("outside"));
        b.def(inside, &[]);
        b.def(outside, &[]);
        for _ in 0..k {
            b.def(outside, &[outside]);
        }
        b.label(1);
        b.def(inside, &[inside, n]);
        b.inst("dec", &[(n, Access::ReadWrite)], Flow::Branch(Label(1)));
        let sum = b.var("sum");
        b.def(sum, &[inside, outside]);
        b.ret(sum, 0);
        spilled(&same(&b.0, &toy(2, &[]), &[3]))[0].clone()
    };
    assert_eq!(first(9), "outside");
    assert_eq!(first(10), "inside");
}

#[test]
fn a_parameter_spills_from_the_register_it_arrives_in() {
    let mut b = Builder::new();
    let p = b.param("p", 0);
    let (x, y, w, z) = (b.var("x"), b.var("y"), b.var("w"), b.var("z"));
    b.def(x, &[]);
    b.def(y, &[]);
    b.def(w, &[x, y]);
    b.def(z, &[w, p]);
    b.ret(z, 0);
    let c = same(&b.0, &toy(2, &[]), &[7]);
    assert_eq!(spilled(&c), ["p"]);
    let Item::Inst(first) = &c.body[0] else { panic!("an instruction first") };
    assert_eq!((first.op, first.args[1].operand), ("store", Operand::Reg(r(0), Width::W64)));
}

#[test]
fn a_result_spills_and_comes_back_to_the_register_it_leaves_in() {
    let mut b = Builder::new();
    let (res, x, y, w, x2, y2, w2) =
        (b.var("res"), b.var("x"), b.var("y"), b.var("w"), b.var("x2"), b.var("y2"), b.var("w2"));
    b.def(res, &[]);
    b.def(x, &[]);
    b.def(y, &[]);
    b.def(w, &[x, y]);
    b.def(x2, &[w]);
    b.def(y2, &[w]);
    b.def(w2, &[x2, y2]);
    b.inst("add", &[(res, Access::ReadWrite), (w2, Access::Read)], Flow::Next);
    b.ret(res, 0);
    let c = same(&b.0, &toy(2, &[]), &[]);
    assert_eq!(spilled(&c), ["res"]);
    let last: Vec<_> = c.body.iter().rev().take(2).collect();
    let Item::Inst(load) = last[1] else { panic!("a load before the return") };
    assert_eq!((load.op, load.args[0].operand), ("load", Operand::Reg(r(0), Width::W64)));
    assert_eq!(reg_of(&c, "res"), r(0));
}

#[test]
fn a_value_live_across_a_call_is_kept_in_a_register_the_call_does_not_change() {
    let build = |call: bool| {
        let mut b = Builder::new();
        let (x, y) = (b.var("x"), b.var("y"));
        b.def(x, &[]);
        if call {
            b.call(None, &[], &[0, 1, 2]);
        }
        b.def(y, &[x]);
        b.ret(y, 0);
        b
    };
    assert_eq!(reg_of(&same(&build(false).0, &toy(4, &[3]), &[]), "x"), r(0));
    let c = same(&build(true).0, &toy(4, &[3]), &[]);
    assert_eq!((reg_of(&c, "x"), c.frame.saved.clone()), (r(3), vec![r(3)]));
}

#[test]
fn a_value_live_across_a_call_that_changes_every_register_spills() {
    let mut b = Builder::new();
    let (x, y) = (b.var("x"), b.var("y"));
    b.def(x, &[]);
    b.call(None, &[], &[0, 1, 2]);
    b.def(y, &[x]);
    b.ret(y, 0);
    let c = same(&b.0, &toy(3, &[]), &[]);
    assert_eq!(spilled(&c), ["x"]);
    assert_eq!(ops(&c), ["op", "store", "call", "load", "op", "ret"]);
}

#[test]
fn a_call_is_given_its_arguments_in_their_registers_and_gives_back_its_result() {
    let mut b = Builder::new();
    let (x, y, res) = (b.var("x"), b.var("y"), b.var("res"));
    b.def(x, &[]);
    b.def(y, &[]);
    b.call(Some((res, 0)), &[(x, 1), (y, 2)], &[0, 1, 2]);
    b.ret(res, 0);
    let c = same(&b.0, &toy(4, &[3]), &[]);
    // Each of `x` and `y` is made where its argument goes, so that its copy moves nothing.
    assert_eq!([reg_of(&c, "x"), reg_of(&c, "y"), reg_of(&c, "res")], [r(1), r(2), r(0)]);
    let moves: Vec<(PReg, PReg)> = c
        .body
        .iter()
        .filter_map(|i| match i {
            Item::Inst(Inst { op: "copy", args, .. }) => match (args[0].operand, args[1].operand) {
                (Operand::Reg(to, _), Operand::Reg(from, _)) => Some((to, from)),
                _ => None,
            },
            _ => None,
        })
        .collect();
    assert_eq!(moves, [(r(1), r(1)), (r(2), r(2)), (r(0), r(0))]);
    assert!(c.frame.saved.is_empty());
}

#[test]
fn arguments_that_swap_registers_go_through_a_third() {
    let mut b = Builder::new();
    let (p, q) = (b.param("p", 0), b.param("q", 1));
    let res = b.var("res");
    b.call(Some((res, 0)), &[(q, 0), (p, 1)], &[0, 1, 2]);
    b.ret(res, 0);
    let c = same(&b.0, &toy(3, &[]), &[7, 9]);
    // `p` arrives in r0, which `q` is copied to while `p` is still to be read: so it moves
    // to a variable of its own on entry, in r2, and is copied from there to its argument, in r1.
    assert_eq!(regs_of(&c, "p"), [r(2), r(1), r(0)]);
    assert_eq!(ops(&c), ["copy", "copy", "copy", "call", "copy", "ret"]);
}

#[test]
fn a_parameter_live_across_a_call_moves_on_entry() {
    let mut b = Builder::new();
    let p = b.param("p", 0);
    let y = b.var("y");
    b.call(None, &[], &[0, 1]);
    b.def(y, &[p]);
    b.ret(y, 0);
    let c = same(&b.0, &toy(3, &[2]), &[7]);
    // The variable `p` is made in r2, which the call keeps, and the one it arrives in is
    // copied there.
    assert_eq!(regs_of(&c, "p"), [r(2), r(0)]);
    assert_eq!(ops(&c), ["copy", "call", "op", "ret"]);
    assert_eq!(c.frame.saved, [r(2)]);
}

#[test]
fn a_result_live_across_a_call_moves_into_its_register_before_the_return() {
    let mut b = Builder::new();
    let x = b.var("x");
    b.def(x, &[]);
    b.call(None, &[], &[0, 1]);
    b.ret(x, 0);
    let c = same(&b.0, &toy(3, &[2]), &[]);
    assert_eq!(regs_of(&c, "x"), [r(2), r(0)]);
    assert_eq!(ops(&c), ["op", "call", "copy", "ret"]);
}

#[test]
fn a_parameter_returned_in_another_register_is_copied_there() {
    let mut b = Builder::new();
    let p = b.param("p", 1);
    b.ret(p, 0);
    let c = same(&b.0, &toy(2, &[]), &[3]);
    assert_eq!(regs_of(&c, "p"), [r(1), r(0)]);
    assert_eq!(ops(&c), ["copy", "ret"]);
}

#[test]
fn two_arguments_in_one_register_at_once_are_an_error() {
    let mut b = Builder::new();
    let (x, y) = (b.var("x"), b.var("y"));
    b.def(x, &[]);
    b.def(y, &[]);
    b.call(None, &[(x, 0), (y, 0)], &[0, 1]);
    let z = b.var("z");
    b.def(z, &[]);
    b.ret(z, 0);
    assert_eq!(compile(&b.0, &toy(3, &[])).unwrap_err(), Error::FixedClash { a: "x".into(), b: "y".into(), reg: r(0) });
}

#[test]
fn a_value_that_may_not_spill_and_is_live_across_a_call_that_changes_every_register_is_an_error() {
    // `x` is written as `dec` jumps, which leaves no place for a store.
    let mut b = Builder::new();
    let n = b.param("n", 0);
    let x = b.var("x");
    b.def(x, &[n]);
    b.label(1);
    b.inst("dec", &[(x, Access::ReadWrite)], Flow::Branch(Label(1)));
    b.call(None, &[], &[0, 1]);
    let y = b.var("y");
    b.def(y, &[x]);
    b.ret(y, 0);
    let err = compile(&b.0, &toy(2, &[])).unwrap_err();
    assert_eq!(
        err,
        Error::OutOfRegisters {
            class: Class::Gpr,
            var: "x".into(),
            near: None,
            taken: vec![],
            changed: vec![r(0), r(1)],
            have: 2,
        }
    );
    assert_eq!(
        err.to_string(),
        "no Gpr register left for `x`: an instruction it is live across changes all 2, as a call does"
    );
}

#[test]
fn a_pinned_value_live_across_a_call_that_changes_its_register_is_an_error() {
    let mut b = Builder::new();
    let a = b.var("a");
    b.0.pinned.push((a, r(0)));
    b.def(a, &[]);
    b.call(None, &[], &[0, 1]);
    let y = b.var("y");
    b.def(y, &[a]);
    b.ret(y, 0);
    let err = compile(&b.0, &toy(3, &[])).unwrap_err();
    assert_eq!(err, Error::FixedChanged { var: "a".into(), reg: r(0) });
}

#[test]
fn an_error_names_the_variable_to_blame() {
    let out = Error::OutOfRegisters {
        class: Class::Gpr,
        var: "x".into(),
        near: None,
        taken: vec![("y".into(), r(1))],
        changed: vec![r(0)],
        have: 2,
    };
    let cases = [
        (Error::ReadBeforeWrite { var: "x".into() }, Some("x")),
        (Error::FixedTwice { var: "x".into() }, Some("x")),
        (Error::FixedChanged { var: "x".into(), reg: r(0) }, Some("x")),
        (out, Some("x")),
        // Of two that clash, the first.
        (Error::FixedClash { a: "x".into(), b: "y".into(), reg: r(0) }, Some("x")),
        (Error::LabelTwice { label: 7 }, None),
        (Error::NoSuchLabel { label: 7 }, None),
        (Error::FallsOffEnd, None),
    ];
    for (e, var) in cases {
        assert_eq!(e.var(), var, "{e:?}");
    }
}

#[test]
fn a_variable_pinned_to_two_registers_is_refused() {
    let mut b = Builder::new();
    let a = b.var("a");
    b.0.pinned.push((a, r(0)));
    b.0.pinned.push((a, r(1)));
    b.def(a, &[]);
    let y = b.var("y");
    b.def(y, &[a]);
    b.ret(y, 0);
    let err = compile(&b.0, &toy(3, &[])).unwrap_err();
    assert_eq!(err, Error::FixedTwice { var: "a".into() });
    assert_eq!(err.to_string(), "`a` must be in two different registers");
}

#[test]
fn each_error_says_what_is_wrong_in_words() {
    let out = |near: Option<&str>, taken: Vec<(&str, PReg)>, changed: Vec<PReg>| Error::OutOfRegisters {
        class: Class::Gpr,
        var: "x".into(),
        near: near.map(String::from),
        taken: taken.into_iter().map(|(name, reg)| (name.to_string(), reg)).collect(),
        changed,
        have: 2,
    };
    let cases = [
        (Error::LabelTwice { label: 7 }, "label 7 is placed twice"),
        (Error::NoSuchLabel { label: 7 }, "a jump goes to label 7, which is never placed"),
        (Error::FallsOffEnd, "control runs past the last instruction"),
        (Error::ReadBeforeWrite { var: "x".into() }, "`x` is read before anything writes it"),
        (Error::FixedTwice { var: "a".into() }, "`a` must be in two different registers"),
        (
            Error::FixedClash { a: "x".into(), b: "y".into(), reg: r(0) },
            "`x` and `y` are live at once and both must be in PReg { class: Gpr, hw: 0 }",
        ),
        (
            Error::FixedChanged { var: "a".into(), reg: r(0) },
            "`a` must be in PReg { class: Gpr, hw: 0 }, which an instruction it is live across changes",
        ),
        (
            out(Some("a note"), vec![("y", r(0)), ("z", r(1))], vec![]),
            "no Gpr register left for `x` (a note): all 2 are held by variables live beside it: y=0 z=1",
        ),
        (
            out(None, vec![], vec![r(0), r(1)]),
            "no Gpr register left for `x`: an instruction it is live across changes all 2, as a call does",
        ),
        (
            out(None, vec![("y", r(1))], vec![r(0)]),
            "no Gpr register left for `x`: an instruction it is live across changes 1 of the 2, as a call does, and \
             variables live beside it hold the rest: y=1",
        ),
    ];
    for (e, want) in cases {
        assert_eq!(e.to_string(), want, "{e:?}");
    }
}
