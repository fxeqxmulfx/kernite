//! The use cases on a toy machine: a few general registers, some callee-saved, and opcodes
//! that are only names, which [`run`] runs.

use std::collections::BTreeMap;

use kgen_core::{
    Access, Arg, Base, Class, Compiled, Flow, Imm, Inst, Item, Label, Machine, Mem, Operand, PReg, Program, RegInfo,
    Register, Signature, Slot, SlotId, Spill, VReg, Width,
};

use crate::{Error, compile};

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

impl Spill<&'static str> for Toy {
    fn width(&self, _: Class) -> Width {
        Width::W64
    }

    fn store(&self, slot: SlotId, var: VReg) -> Inst<&'static str, VReg> {
        let args =
            vec![slot_arg(slot, Access::Write), Arg { operand: Operand::Reg(var, Width::W64), access: Access::Read }];
        Inst { op: "store", args, implicit: vec![], flow: Flow::Next }
    }

    fn load(&self, var: VReg, slot: SlotId) -> Inst<&'static str, VReg> {
        let args =
            vec![Arg { operand: Operand::Reg(var, Width::W64), access: Access::Write }, slot_arg(slot, Access::Read)];
        Inst { op: "load", args, implicit: vec![], flow: Flow::Next }
    }
}

/// What `body` returns, run from registers that hold `params` and `slots` slots of memory
/// that hold zero: `op` writes a hash of its immediate and the registers it reads, `add` adds
/// the second to the first, `dec` takes one from its register and jumps unless that leaves
/// zero, `load` and `store` move a register's value to a slot and back, and `ret` returns what
/// it reads. Reading a register nothing wrote, or running past ten thousand instructions,
/// panics.
fn run<R: Register + Ord>(body: &[Item<&'static str, R>], params: &[(R, u64)], slots: usize) -> u64 {
    let mut regs: BTreeMap<R, u64> = params.iter().copied().collect();
    let mut mem = vec![0u64; slots];
    let place = |l: Label| body.iter().position(|item| *item == Item::Label(l)).expect("placed");
    let mut at = 0;
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
        let reg = |a: &Arg<R>| match a.operand {
            Operand::Reg(r, _) => r,
            _ => unreachable!("{:?} is no register", a.operand),
        };
        let mut jump = false;
        match inst.op {
            "op" => {
                let read = inst.args[1..].iter().filter(|a| a.access == Access::Read);
                let hash =
                    read.fold(0xcbf2_9ce4_8422_2325u64, |h, a| (h ^ value(a, &regs)).wrapping_mul(0x100_0000_01b3));
                regs.insert(reg(&inst.args[0]), hash);
            }
            "add" => {
                let sum = value(&inst.args[0], &regs).wrapping_add(value(&inst.args[1], &regs));
                regs.insert(reg(&inst.args[0]), sum);
            }
            "dec" => {
                let left = value(&inst.args[0], &regs) - 1;
                regs.insert(reg(&inst.args[0]), left);
                jump = left != 0;
            }
            "load" => {
                let v = value(&inst.args[1], &regs);
                regs.insert(reg(&inst.args[0]), v);
            }
            "store" => {
                let Operand::Mem(Mem { base: Base::Slot(s), .. }) = inst.args[0].operand else { unreachable!() };
                mem[s.0 as usize] = value(&inst.args[1], &regs);
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
fn same(p: &Program<&'static str>, m: &Toy, args: &[u64]) -> Compiled<&'static str> {
    let c = compile(p, m).unwrap();
    let before: Vec<(VReg, u64)> = p.params.iter().map(|&(v, _)| v).zip(args.iter().copied()).collect();
    let after: Vec<(PReg, u64)> = p.params.iter().map(|&(_, r)| r).zip(args.iter().copied()).collect();
    assert_eq!(run(&c.body, &after, c.frame.slots.len()), run(&p.body, &before, p.slots.len()), "for {args:?}");
    c
}

/// The names of the variables `c` spilled, in the order they did.
fn spilled(c: &Compiled<&'static str>) -> Vec<String> {
    c.frame.slots.iter().filter(|s| s.spill).map(|s| s.name.clone()).collect()
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
                body: vec![],
                slots: vec![],
                consts: vec![],
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
        let args = args.iter().map(|&(v, access)| Arg { operand: Operand::Reg(v, Width::W64), access }).collect();
        self.0.body.push(Item::Inst(Inst { op, args, implicit: vec![], flow }));
    }

    /// `dst = op(srcs)`, with an immediate of its own, so that no two compute the same.
    fn def(&mut self, dst: VReg, srcs: &[VReg]) {
        self.1 += 1;
        let mut args = vec![Arg { operand: Operand::Reg(dst, Width::W64), access: Access::Write }];
        args.push(Arg { operand: Operand::Imm(Imm { value: self.1, hex: false }), access: Access::Read });
        args.extend(srcs.iter().map(|&s| Arg { operand: Operand::Reg(s, Width::W64), access: Access::Read }));
        self.0.body.push(Item::Inst(Inst { op: "op", args, implicit: vec![], flow: Flow::Next }));
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

fn reg_of(c: &kgen_core::Compiled<&'static str>, name: &str) -> PReg {
    c.assignment.iter().find(|(info, _)| info.name == name).and_then(|(_, r)| *r).unwrap()
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
    assert_eq!(reg_of(&compile(&b.0, &toy(1, &[])).unwrap(), "q"), r(0));

    let mut b = Builder::new();
    let p = b.param("p", 0);
    let q = b.var("q");
    b.def(q, &[p]);
    b.inst("use", &[(p, Access::Read)], Flow::Next);
    b.ret(q, 0);
    assert_eq!(compile(&b.0, &toy(2, &[])).unwrap_err(), Error::FixedClash { a: "p".into(), b: "q".into(), reg: r(0) });
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
