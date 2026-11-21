//! The allocator on a toy machine: a few general registers, some callee-saved, and opcodes
//! that are only names, which [`run`] runs.

use std::collections::BTreeMap;

use kernite_core::{
    Access, Arg, Base, Class, Clobber, Flow, Imm, Inst, Item, Label, Mem, Operand, PReg, Program, RegInfo, Register,
    Signature, Slot, SlotId, VReg, Width,
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

fn slot_arg<R>(slot: SlotId, width: Width, access: Access) -> Arg<R> {
    Arg { operand: Operand::Mem(Mem { base: Base::Slot(slot), index: None, disp: 0, width }), access }
}

fn reg_arg<R>(r: R, width: Width, access: Access) -> Arg<R> {
    Arg { operand: Operand::Reg(r, width), access }
}

/// An instruction of the toy's that goes on to the next and changes nothing but its operands.
fn plain<R>(op: &'static str, args: Vec<Arg<R>>) -> Inst<&'static str, R> {
    Inst { op, args, implicit: vec![], flow: Flow::Next, clobbers: vec![] }
}

impl Spill<&'static str> for Toy {
    fn width(&self, _: Class) -> Width {
        Width::W64
    }

    /// A variable that no instruction names more than the low half of is kept in 32 bits.
    fn kept(&self, _: Class, named: Width) -> Width {
        named.max(Width::W32)
    }

    fn store(&self, slot: SlotId, var: VReg, width: Width) -> Inst<&'static str, VReg> {
        plain("store", vec![slot_arg(slot, width, Access::Write), reg_arg(var, width, Access::Read)])
    }

    fn load(&self, var: VReg, slot: SlotId, width: Width) -> Inst<&'static str, VReg> {
        plain("load", vec![reg_arg(var, width, Access::Write), slot_arg(slot, width, Access::Read)])
    }

    fn copy(&self, to: VReg, from: VReg, width: Width) -> Inst<&'static str, VReg> {
        plain("copy", vec![reg_arg(to, width, Access::Write), reg_arg(from, width, Access::Read)])
    }
}

/// What a read at 32 bits sees of a register: its low half.
const LOW: u64 = 0xffff_ffff;

/// The FNV hash of `values`, which `op` and `call` write.
fn hash(values: impl IntoIterator<Item = u64>) -> u64 {
    values.into_iter().fold(0xcbf2_9ce4_8422_2325u64, |h, x| (h ^ x).wrapping_mul(0x100_0000_01b3))
}

/// What `body` returns, run from registers that hold `params` and `slots` slots of memory
/// that hold zero: `op` writes a hash of its immediate and the registers it reads, `add` adds
/// the second to the first, `add32` adds the low halves, 32 bits, of the two to the first's and clears its upper half,
/// `dec` takes one from its register and jumps unless that leaves zero, `load` and `store` move a register's value to a
/// slot and back, `copy` moves one to another, `call` writes a hash of the arguments it reads to its result, if it has
/// one, and `ret` returns what it reads. A read at 32 bits sees the low half of a register, and a read of memory at the
/// address in a register sees all of the register, for the toy has no memory but its slots. Reading a register nothing
/// wrote, or running past ten thousand instructions, panics.
///
/// A call writes junk to each of the machine's registers it changes, its `clobbers`, that
/// `reg` says is a register of `R`'s, as a callee may, or to the upper half of one it clears that of; the result goes
/// in after. So a value kept in one across a call is lost, or its upper half is, and the run returns something else if
/// the value is read after.
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
            Operand::Reg(r, w) => {
                let all = *regs.get(&r).unwrap_or_else(|| panic!("{r:?} is read before anything writes it"));
                if w == Width::W32 { all & LOW } else { all }
            }
            Operand::Mem(Mem { base: Base::Slot(s), .. }) => mem[s.0 as usize],
            Operand::Mem(Mem { base: Base::Reg(r), .. }) => regs[&r],
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
                // Each variable it writes has a value of its own: the first the hash, and one more each after it.
                let others = inst.args[1..].iter().filter(|a| a.access == Access::Write);
                let writes = std::iter::once(&inst.args[0]).chain(others);
                for (k, a) in writes.enumerate() {
                    regs.insert(reg_of(a), h.wrapping_add(k as u64));
                }
            }
            "add" => {
                let sum = value(&inst.args[0], &regs).wrapping_add(value(&inst.args[1], &regs));
                regs.insert(reg_of(&inst.args[0]), sum);
            }
            "add32" => {
                let sum = value(&inst.args[0], &regs) + value(&inst.args[1], &regs);
                regs.insert(reg_of(&inst.args[0]), sum & LOW);
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
                for (k, &clobber) in inst.clobbers.iter().enumerate() {
                    let junk = 0xdead_0000 + calls * 64 + k as u64;
                    match clobber {
                        Clobber::All(c) => {
                            if let Some(c) = reg(c) {
                                regs.insert(c, junk);
                            }
                        }
                        Clobber::Upper(c) => {
                            if let Some(c) = reg(c) {
                                let low = regs.get(&c).map_or(0, |v| v & LOW);
                                regs.insert(c, junk << 32 | low);
                            }
                        }
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

/// The slot each store and load of `c` names, in order.
fn moves(c: &Allocated<&'static str>) -> Vec<(&'static str, u32)> {
    let slot = |i: &Inst<&'static str, PReg>| {
        i.args.iter().find_map(|a| match a.operand {
            Operand::Mem(Mem { base: Base::Slot(s), .. }) => Some(s.0),
            _ => None,
        })
    };
    let moved = |item: &Item<&'static str, PReg>| match item {
        Item::Inst(i) if matches!(i.op, "store" | "load") => slot(i).map(|s| (i.op, s)),
        _ => None,
    };
    c.body.iter().filter_map(moved).collect()
}

/// The registers each copy of `c` writes and reads, in order.
fn copies(c: &Allocated<&'static str>) -> Vec<(PReg, PReg)> {
    let copy = |item: &Item<&'static str, PReg>| match item {
        Item::Inst(i) if i.op == "copy" => match (i.args[0].operand, i.args[1].operand) {
            (Operand::Reg(to, _), Operand::Reg(from, _)) => Some((to, from)),
            _ => unreachable!("a copy of registers"),
        },
        _ => None,
    };
    c.body.iter().filter_map(copy).collect()
}

fn r(hw: u8) -> PReg {
    PReg { class: Class::Gpr, hw }
}

/// `n` registers, the ones in `saved` callee-saved.
fn toy(n: u8, saved: &[u8]) -> Toy {
    Toy { gpr: (0..n).map(r).collect(), saved: saved.iter().map(|&h| r(h)).collect() }
}

/// What a call does that clears the upper halves of the registers `hws` and leaves the lower.
fn upper(hws: &[u8]) -> Vec<Clobber> {
    hws.iter().map(|&h| Clobber::Upper(r(h))).collect()
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
        let args = args.iter().map(|&(v, access)| reg_arg(v, Width::W64, access)).collect();
        self.0.body.push(Item::Inst(Inst { op, args, implicit: vec![], flow, clobbers: vec![] }));
    }

    /// `dst = op(srcs)`, with an immediate of its own, so that no two compute the same.
    fn def(&mut self, dst: VReg, srcs: &[VReg]) {
        self.def_at(Width::W64, dst, srcs);
    }

    /// [`def`](Builder::def) at 32 bits: it reads the low half of each of `srcs`, and writes `dst`, all of it.
    fn def_low(&mut self, dst: VReg, srcs: &[VReg]) {
        self.def_at(Width::W32, dst, srcs);
    }

    /// `dst = op(srcs)`, as [`def`](Builder::def), but for the width each is read or written at.
    fn def_at(&mut self, width: Width, dst: VReg, srcs: &[VReg]) {
        self.1 += 1;
        let arg = |v, access| Arg { operand: Operand::Reg(v, width), access };
        let mut args = vec![arg(dst, Access::Write)];
        args.push(Arg { operand: Operand::Imm(Imm { value: self.1, hex: false }), access: Access::Read });
        args.extend(srcs.iter().map(|&s| arg(s, Access::Read)));
        self.0.body.push(Item::Inst(plain("op", args)));
    }

    /// `dst`'s low half added to, which clears its upper half: a write of part of a register that reads what was there.
    fn add_low(&mut self, dst: VReg, src: VReg) {
        let arg = |v, access| Arg { operand: Operand::Reg(v, Width::W32), access };
        self.0.body.push(Item::Inst(plain("add32", vec![arg(dst, Access::ReadWrite), arg(src, Access::Read)])));
    }

    /// A call that changes the registers `clobbers`, as a machine's code makes one: each of
    /// `args` copied to a variable of its own, pinned in the register given with it, and the
    /// result, if `dst` says where it goes, given back in a variable pinned in the register
    /// given with it and copied to `dst`.
    fn call(&mut self, dst: Option<(VReg, u8)>, args: &[(VReg, u8)], clobbers: &[u8]) {
        self.call_with(dst, args, clobbers.iter().map(|&h| Clobber::All(r(h))).collect());
    }

    /// [`call`](Builder::call) that does to registers what `clobbers` says.
    fn call_with(&mut self, dst: Option<(VReg, u8)>, args: &[(VReg, u8)], clobbers: Vec<Clobber>) {
        self.call_at(Width::W64, dst, args, clobbers);
    }

    /// [`call_with`](Builder::call_with) that copies each argument at `width`, and reads it so.
    fn call_at(&mut self, width: Width, dst: Option<(VReg, u8)>, args: &[(VReg, u8)], clobbers: Vec<Clobber>) {
        let mut implicit = vec![];
        for &(a, hw) in args {
            let name = self.0.regs[a.id as usize].name.clone();
            let pinned = self.var(&name);
            self.0.pinned.push((pinned, r(hw)));
            self.0.body.push(Item::Inst(Toy::copy(&toy(0, &[]), pinned, a, width)));
            implicit.push(reg_arg(pinned, width, Access::Read));
        }
        let back = dst.map(|(d, hw)| {
            let name = self.0.regs[d.id as usize].name.clone();
            let pinned = self.var(&name);
            self.0.pinned.push((pinned, r(hw)));
            implicit.push(reg_arg(pinned, Width::W64, Access::Write));
            (d, pinned)
        });
        self.0.body.push(Item::Inst(Inst { op: "call", args: vec![], implicit, flow: Flow::Next, clobbers }));
        if let Some((d, pinned)) = back {
            self.0.body.push(Item::Inst(Toy::copy(&toy(0, &[]), d, pinned, Width::W64)));
        }
    }

    /// A copy of `width` of `from` to `to`, as a machine's code makes one.
    fn copy(&mut self, to: VReg, from: VReg, width: Width) {
        self.0.body.push(Item::Inst(Toy::copy(&toy(0, &[]), to, from, width)));
    }

    /// A slot of `width`'s bytes: one that a spill keeps a variable in, as the allocator makes it, or one of the
    /// program's own.
    fn slot_at(&mut self, width: Width, name: &str, spill: bool) -> SlotId {
        let bytes = width.bytes();
        self.0.slots.push(Slot { name: name.into(), size: bytes, align: bytes, spill });
        SlotId(self.0.slots.len() as u32 - 1)
    }

    /// [`slot_at`](Builder::slot_at) of 8 bytes.
    fn slot(&mut self, name: &str, spill: bool) -> SlotId {
        self.slot_at(Width::W64, name, spill)
    }

    /// `v` stored to `slot`, `width` of it, as the allocator stores a spilled variable.
    fn store_at(&mut self, width: Width, slot: SlotId, v: VReg) {
        self.0.body.push(Item::Inst(Toy::store(&toy(0, &[]), slot, v, width)));
    }

    /// [`store_at`](Builder::store_at) of all of it.
    fn store(&mut self, slot: SlotId, v: VReg) {
        self.store_at(Width::W64, slot, v);
    }

    /// `v` loaded from `slot`, `width` of it, as the allocator loads a spilled variable.
    fn load_at(&mut self, width: Width, v: VReg, slot: SlotId) {
        self.0.body.push(Item::Inst(Toy::load(&toy(0, &[]), v, slot, width)));
    }

    /// [`load_at`](Builder::load_at) of all of it.
    fn load(&mut self, v: VReg, slot: SlotId) {
        self.load_at(Width::W64, v, slot);
    }

    fn label(&mut self, l: u32) {
        self.0.body.push(Item::Label(Label(l)));
    }

    fn ret(&mut self, v: VReg, hw: u8) {
        self.ret_at(Width::W64, v, hw);
    }

    /// [`ret`](Builder::ret) that reads `v` at `width`.
    fn ret_at(&mut self, width: Width, v: VReg, hw: u8) {
        self.0.results.push((v, r(hw)));
        self.0.signature.result = Some(Class::Gpr);
        let args = vec![reg_arg(v, width, Access::Read)];
        self.0.body.push(Item::Inst(Inst { op: "ret", args, implicit: vec![], flow: Flow::Return, clobbers: vec![] }));
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
            // The variable spilling made for the instruction alone, of `y`'s name.
            vreg: VReg { id: 6, class: Class::Gpr },
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
        // The variable spilling made for the add alone, of `t`'s name.
        vreg: VReg { id: 6, class: Class::Gpr },
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
    assert_eq!(compile(&b.0, &toy(1, &[])).unwrap_err(), Error::ReadBeforeWrite { var: "x".into(), vreg: x });

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
    assert_eq!(compile(&b.0, &toy(2, &[])).unwrap_err(), Error::ReadBeforeWrite { var: "x".into(), vreg: x });

    // And to the next item, where it may.
    let mut b = Builder::new();
    let n = b.param("n", 0);
    let x = b.var("x");
    b.inst("jmp", &[(n, Access::Read)], Flow::Table { to: vec![Label(1)], or_next: true });
    b.inst("use", &[(x, Access::Read)], Flow::Next);
    b.label(1);
    b.def(x, &[]);
    b.ret(x, 1);
    assert_eq!(compile(&b.0, &toy(2, &[])).unwrap_err(), Error::ReadBeforeWrite { var: "x".into(), vreg: x });
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

/// A parameter that spills is stored on entry from the register it arrives in, as much as it is kept in: all of the
/// register, or 32 bits where no instruction names more.
#[test]
fn a_parameter_spills_from_the_register_it_arrives_in() {
    for width in [Width::W64, Width::W32] {
        let mut b = Builder::new();
        let p = b.param("p", 0);
        let (x, y, w, z) = (b.var("x"), b.var("y"), b.var("w"), b.var("z"));
        b.def(x, &[]);
        b.def(y, &[]);
        b.def(w, &[x, y]);
        b.def_at(width, z, &[w, p]);
        b.ret(z, 0);
        let c = same(&b.0, &toy(2, &[]), &[7]);
        assert_eq!(spilled(&c), ["p"]);
        let Item::Inst(first) = &c.body[0] else { panic!("an instruction first") };
        assert_eq!((first.op, first.args[1].operand), ("store", Operand::Reg(r(0), width)));
    }
}

/// A result that spills comes back before the return to the register it leaves in, as much as it is kept in: all of the
/// register, or 32 bits where no instruction names more. It is a load from its slot, or, where a register holds it, a
/// copy from there: the register the last write left it in holds it until a call changes every register.
#[test]
fn a_result_spills_and_comes_back_to_the_register_it_leaves_in() {
    for (width, call) in [(Width::W64, false), (Width::W32, false), (Width::W64, true), (Width::W32, true)] {
        let mut b = Builder::new();
        let (res, x, y, w, x2, y2, w2) =
            (b.var("res"), b.var("x"), b.var("y"), b.var("w"), b.var("x2"), b.var("y2"), b.var("w2"));
        b.def_at(width, res, &[]);
        b.def(x, &[]);
        b.def(y, &[]);
        b.def(w, &[x, y]);
        b.def(x2, &[w]);
        b.def(y2, &[w]);
        b.def(w2, &[x2, y2]);
        let add = vec![reg_arg(res, width, Access::ReadWrite), reg_arg(w2, width, Access::Read)];
        b.0.body.push(Item::Inst(plain("add", add)));
        if call {
            b.call(None, &[], &[0, 1]);
        }
        b.ret_at(width, res, 0);
        let c = same(&b.0, &toy(2, &[]), &[]);
        assert_eq!(spilled(&c), ["res"]);
        let last: Vec<_> = c.body.iter().rev().take(2).collect();
        let Item::Inst(back) = last[1] else { panic!("a load or a copy before the return") };
        assert_eq!((back.op, back.args[0].operand), (if call { "load" } else { "copy" }, Operand::Reg(r(0), width)));
        assert!(regs_of(&c, "res").contains(&r(0)));
    }
}

/// A variable that no instruction names more than the low half of is kept in it: its slot is 4 bytes, and its store and
/// its load are of 32 bits. One that an instruction names whole is kept whole, whatever the others name, and so is one
/// that an address takes, for an address does not say how much of a register it reads.
#[test]
fn a_variable_is_kept_as_wide_as_the_widest_instruction_that_names_it() {
    // The slots and the moves to and from them, of a program whose `last` makes `v` of `w` and `z`.
    let kept = |last: fn(&mut Builder, VReg, VReg, VReg)| {
        let mut b = Builder::new();
        let (x, y, z, w, v) = (b.var("x"), b.var("y"), b.var("z"), b.var("w"), b.var("v"));
        b.def_low(x, &[]);
        b.def_low(y, &[]);
        b.def_low(z, &[]);
        b.def_low(w, &[x, y]);
        last(&mut b, v, w, z);
        b.ret(v, 0);
        let c = same(&b.0, &toy(2, &[]), &[]);
        let slots = c.frame.slots.iter().map(|s| (s.name.as_str(), s.size, s.align)).collect::<Vec<_>>();
        let moves: Vec<(&str, Width)> = c
            .body
            .iter()
            .filter_map(|i| match i {
                Item::Inst(Inst { op: op @ ("store" | "load"), args, .. }) => {
                    args.iter().find_map(|a| match a.operand {
                        Operand::Reg(_, width) => Some((*op, width)),
                        _ => None,
                    })
                }
                _ => None,
            })
            .collect();
        (format!("{slots:?}"), moves)
    };
    let (s32, s64, l32, l64) =
        (("store", Width::W32), ("store", Width::W64), ("load", Width::W32), ("load", Width::W64));
    let low = ("[(\"x\", 4, 4), (\"z\", 4, 4)]".into(), vec![s32, s32, l32, l32]);
    let whole = ("[(\"x\", 4, 4), (\"z\", 8, 8)]".into(), vec![s32, s64, l32, l64]);
    assert_eq!(kept(|b, v, w, z| b.def_low(v, &[w, z])), low);
    assert_eq!(kept(|b, v, w, z| b.def(v, &[w, z])), whole);
    let address = |b: &mut Builder, v, w, z| {
        let mem = Mem { base: Base::Reg(z), index: None, disp: 0, width: Width::W32 };
        let at = Arg { operand: Operand::Mem(mem), access: Access::Read };
        let args = vec![reg_arg(v, Width::W32, Access::Write), reg_arg(w, Width::W32, Access::Read), at];
        b.0.body.push(Item::Inst(plain("op", args)));
    };
    assert_eq!(kept(address), whole);
}

/// A spilled variable is read in the register that holds it, with no load, where one does. A parameter that spills is
/// stored on entry from the register it arrives in, `p` from r1, and that register holds it until an instruction writes
/// the register: `x` reads it there. `b` takes r1, so the read of `p` after it is a load. The same where each
/// instruction that names `p` reads 32 bits of it, and `p` is kept in 32.
#[test]
fn a_spilled_variable_is_read_in_the_register_that_holds_it() {
    for width in [Width::W64, Width::W32] {
        let mut b = Builder::new();
        let p = b.param("p", 1);
        let (x, y, a, d, z, w) = (b.var("x"), b.var("y"), b.var("a"), b.var("b"), b.var("z"), b.var("w"));
        b.def_at(width, x, &[p]);
        b.def(y, &[x]);
        b.def(a, &[y]);
        b.def(d, &[]);
        b.def(z, &[a, d]);
        b.def_at(width, w, &[z, p]);
        b.ret(w, 0);
        let c = same(&b.0, &toy(2, &[]), &[5]);
        assert_eq!(spilled(&c), ["p"]);
        assert_eq!(ops(&c), ["store", "op", "op", "op", "op", "op", "load", "op", "ret"]);
        let Item::Inst(read) = &c.body[1] else { panic!("an instruction") };
        assert_eq!(read.args[2].operand, Operand::Reg(r(1), width));
    }
}

/// A load leaves what it loads in the register it loads to, as a store leaves what it stores in the register it stores
/// from: `w` reads `p` in r1, where its load put it, and `v` reads it there too, with no load of its own.
#[test]
fn a_load_leaves_what_it_loads_in_the_register_it_loads_to() {
    let mut b = Builder::new();
    let p = b.param("p", 1);
    let (x, y, a, d, z, w, v) = (b.var("x"), b.var("y"), b.var("a"), b.var("b"), b.var("z"), b.var("w"), b.var("v"));
    b.def(x, &[p]);
    b.def(y, &[x]);
    b.def(a, &[y]);
    b.def(d, &[]);
    b.def(z, &[a, d]);
    b.def(w, &[z, p]);
    b.def(v, &[w, p]);
    b.ret(v, 0);
    let c = same(&b.0, &toy(2, &[]), &[5]);
    assert_eq!(ops(&c), ["store", "op", "op", "op", "op", "op", "load", "op", "op", "ret"]);
}

/// A store to a slot leaves what it stores in the register it stores from, and in no other that held what the slot kept
/// before: `x` and `y` are stored in turn to a slot, and the load after reads `y`, in r1. `x` is alive to the end in
/// r0, and a read of it would be wrong, for it is not what the slot keeps.
#[test]
fn a_store_leaves_what_it_stores_only_in_the_register_it_stores_from() {
    let mut b = Builder::new();
    let (x, y, t, z, w) = (b.var("x"), b.var("y"), b.var("t"), b.var("z"), b.var("w"));
    let s = b.slot("s", true);
    b.def(x, &[]);
    b.def(y, &[]);
    b.store(s, x);
    b.store(s, y);
    b.load(t, s);
    b.def(z, &[t, x]);
    b.def(w, &[z, y]);
    b.ret(w, 0);
    let c = same(&b.0, &toy(3, &[]), &[]);
    assert_eq!(ops(&c), ["op", "op", "op", "op", "ret"]);
    let Item::Inst(read) = &c.body[2] else { panic!("an instruction") };
    assert_eq!(read.args[2].operand, Operand::Reg(r(1), Width::W64));
}

/// A call leaves in a register what it does not change, and the load after it is left out, and the store before it,
/// and the slot: `x` stays in r0 over the call. A call that changes r0 leaves nothing in it, nor does one that clears
/// its upper half, where the slot keeps all of r0: one kept in 32 bits stays.
#[test]
fn a_call_leaves_in_a_register_what_it_does_not_change() {
    let all = |hw: u8| vec![Clobber::All(r(hw))];
    for (clobbers, width, kept) in [
        (all(1), Width::W64, true),
        (all(0), Width::W64, false),
        (upper(&[0]), Width::W64, false),
        (upper(&[0]), Width::W32, true),
        (all(0), Width::W32, false),
    ] {
        let mut b = Builder::new();
        let (x, t, z) = (b.var("x"), b.var("t"), b.var("z"));
        let s = b.slot_at(width, "s", true);
        b.def_at(width, x, &[]);
        b.store_at(width, s, x);
        b.call_with(None, &[], clobbers);
        b.load_at(width, t, s);
        b.def_at(width, z, &[t]);
        b.ret_at(width, z, 0);
        let c = same(&b.0, &toy(2, &[]), &[]);
        let ops_kept: &[&str] =
            if kept { &["op", "call", "op", "ret"] } else { &["op", "store", "call", "load", "op", "ret"] };
        assert_eq!(ops(&c), ops_kept);
    }
}

/// A store that no load follows is left out, and the slot of the variable with it: the slots after it are the ones the
/// moves name from then on. Four values are alive at once in three registers, so `v3` spills, and `v2`. `v3` is stored
/// from r2, where it was written, and nothing writes r2 before the last `add`, which reads `v3` there: its load is left
/// out, and no load follows its store, nor does its slot hold anything, so that the slot of `v2` is the first.
#[test]
fn a_store_that_no_load_follows_is_left_out_with_its_slot() {
    let mut b = Builder::new();
    let v: Vec<VReg> = (0..4).map(|i| b.var(&format!("v{i}"))).collect();
    for &x in &v {
        b.def(x, &[]);
    }
    for &x in &v[1..] {
        b.inst("add", &[(v[0], Access::ReadWrite), (x, Access::Read)], Flow::Next);
    }
    b.ret(v[0], 0);
    let c = same(&b.0, &toy(3, &[]), &[]);
    assert_eq!(spilled(&c), ["v2"]);
    assert_eq!(ops(&c), ["op", "op", "op", "store", "op", "add", "load", "add", "add", "ret"]);
    assert_eq!(moves(&c), [("store", 0), ("load", 0)]);
    assert_eq!(regs_of(&c, "v3"), [r(2)]);
}

/// A store that another store to its slot follows, with no load between them, is left out too, for no one reads what it
/// stores. `x` and `y` are stored in turn, and a call changes every register before the load, so that only the store of
/// `y` is of use.
#[test]
fn a_store_that_another_store_follows_before_any_load_is_left_out() {
    let mut b = Builder::new();
    let (x, y, t, z) = (b.var("x"), b.var("y"), b.var("t"), b.var("z"));
    let s = b.slot("s", true);
    b.def(x, &[]);
    b.def(y, &[]);
    b.store(s, x);
    b.store(s, y);
    b.call(None, &[], &[0, 1]);
    b.load(t, s);
    b.def(z, &[t]);
    b.ret(z, 0);
    let c = same(&b.0, &toy(2, &[]), &[]);
    assert_eq!(ops(&c), ["op", "op", "store", "call", "load", "op", "ret"]);
    let Item::Inst(store) = &c.body[2] else { panic!("an instruction") };
    assert_eq!(store.args[1].operand, Operand::Reg(r(1), Width::W64));
}

/// A store stays where a load may follow it, on some way if not on every: the load after the jump reads the store
/// before it, if the kernel does not jump. `n` says which way the kernel takes, the jump if 2, as the Toy's `dec` says;
/// it is run both ways. `y` takes the register of `x`, so that no register holds what the slot keeps.
#[test]
fn a_store_stays_where_a_load_follows_it_on_one_way_only() {
    let mut b = Builder::new();
    let (n, x, y, t) = (b.param("n", 1), b.var("x"), b.var("y"), b.var("t"));
    let s = b.slot("s", true);
    b.def(x, &[]);
    b.store(s, x);
    b.def(y, &[]);
    b.inst("dec", &[(n, Access::ReadWrite)], Flow::Branch(Label(1)));
    b.load(t, s);
    b.inst("add", &[(y, Access::ReadWrite), (t, Access::Read)], Flow::Next);
    b.label(1);
    b.ret(y, 0);
    for way in [1, 2] {
        let c = same(&b.0, &toy(3, &[]), &[way]);
        assert_eq!(ops(&c), ["op", "store", "op", "dec", "load", "add", "ret"]);
    }
}

/// A store stays where a load follows it by the jump back of a loop: each turn stores what the next loads. `x` is
/// stored before the loop from r1, where it stays, for the loop reads it, and `t2` in the loop from r2, so that no
/// register holds what the slot keeps at the top of the loop, and the load there stays.
#[test]
fn a_store_stays_where_a_load_follows_it_by_the_jump_back_of_a_loop() {
    let mut b = Builder::new();
    let (n, x, t, t2) = (b.param("n", 0), b.var("x"), b.var("t"), b.var("t2"));
    let s = b.slot("s", true);
    b.def(x, &[]);
    b.store(s, x);
    b.label(1);
    b.load(t, s);
    b.def(t2, &[t, x]);
    b.store(s, t2);
    b.inst("dec", &[(n, Access::ReadWrite)], Flow::Branch(Label(1)));
    b.ret(t2, 2);
    for turns in 1..=3 {
        let c = same(&b.0, &toy(3, &[]), &[turns]);
        assert_eq!(ops(&c), ["op", "store", "load", "op", "store", "dec", "ret"]);
        assert_eq!(regs_of(&c, "x"), [r(1)]);
        assert_eq!(regs_of(&c, "t2"), [r(2)]);
    }
}

/// A register holds what a slot keeps at the top of a loop though each turn brings it in a variable of its own: `x` is
/// stored before the loop from r1, and `t2` in the loop, also from r1, so that r1 holds what the slot keeps at the top
/// on both ways, in `x` on the way in and in `t2` on the jump back, which are made one. The load at the top is left
/// out, and both stores, which no load follows now.
#[test]
fn a_register_holds_what_a_slot_keeps_at_the_top_of_a_loop_though_each_turn_brings_it_in_a_variable_of_its_own() {
    let mut b = Builder::new();
    let (n, x, t, t2) = (b.param("n", 0), b.var("x"), b.var("t"), b.var("t2"));
    let s = b.slot("s", true);
    b.def(x, &[]);
    b.store(s, x);
    b.label(1);
    b.load(t, s);
    b.def(t2, &[t]);
    b.store(s, t2);
    b.inst("dec", &[(n, Access::ReadWrite)], Flow::Branch(Label(1)));
    b.ret(t2, 1);
    for turns in 1..=3 {
        let c = same(&b.0, &toy(2, &[]), &[turns]);
        assert_eq!(ops(&c), ["op", "op", "dec", "ret"]);
        assert_eq!(regs_of(&c, "x").into_iter().chain(regs_of(&c, "t2")).collect::<Vec<_>>(), [r(1)]);
    }
}

/// A slot of the program's own keeps its stores and its number, though no load follows them: only the slots that spills
/// make are left out, and the ones after them are numbered again from where the program's own end.
#[test]
fn a_slot_of_the_programs_own_keeps_its_stores_and_its_number() {
    let mut b = Builder::new();
    let (x, y, u, t, w) = (b.var("x"), b.var("y"), b.var("u"), b.var("t"), b.var("w"));
    let (own, a, k) = (b.slot("own", false), b.slot("a", true), b.slot("k", true));
    b.def(x, &[]);
    b.store(own, x);
    b.store(a, x);
    b.def(y, &[]);
    b.store(k, y);
    b.def(u, &[]);
    b.load(t, k);
    b.def(w, &[t, u]);
    b.ret(w, 0);
    let c = same(&b.0, &toy(3, &[]), &[]);
    assert_eq!(spilled(&c), ["k"]);
    assert_eq!(c.frame.slots.len(), 2);
    assert_eq!(moves(&c), [("store", 0), ("store", 1), ("load", 1)]);
}

/// A load is left out where the instruction after it is the only one that reads the variable it loads, and that
/// instruction reads the variable of the register that holds it: it reads there before it writes, though the register
/// is the one it writes. `z` takes r0 from `x`, which it reads there as `t`, and so it does where it reads `t` twice.
#[test]
fn a_load_is_left_out_where_the_instruction_after_it_is_the_only_one_that_reads_it() {
    for twice in [false, true] {
        let mut b = Builder::new();
        let (x, y, t, z, w) = (b.var("x"), b.var("y"), b.var("t"), b.var("z"), b.var("w"));
        let s = b.slot("s", true);
        b.def(x, &[]);
        b.def(y, &[]);
        b.store(s, x);
        b.load(t, s);
        b.def(z, &[t, x, t][..if twice { 3 } else { 2 }]);
        b.def(w, &[z, y]);
        b.ret(w, 0);
        let c = same(&b.0, &toy(3, &[]), &[]);
        assert_eq!(ops(&c), ["op", "op", "op", "op", "ret"]);
        let Item::Inst(read) = &c.body[2] else { panic!("an instruction") };
        assert!(
            read.args
                .iter()
                .all(|a| a.operand == Operand::Reg(r(0), Width::W64) || matches!(a.operand, Operand::Imm(_)))
        );
    }
}

/// A load is a copy from the register that holds what it loads where the instruction it is for is not the only one
/// that reads what it loads: `t` is read by `z` and by `w`. The copy is of as much as the load was: all of it, or 32
/// bits where no instruction names more.
#[test]
fn a_load_is_a_copy_where_the_instruction_it_is_for_is_not_the_only_one_that_reads_it() {
    for width in [Width::W64, Width::W32] {
        let mut b = Builder::new();
        let (x, t, z, w, v) = (b.var("x"), b.var("t"), b.var("z"), b.var("w"), b.var("v"));
        let s = b.slot("x", true);
        b.def_at(width, x, &[]);
        b.store_at(width, s, x);
        b.load_at(width, t, s);
        b.def_at(width, z, &[t]);
        b.def_at(width, w, &[t, z]);
        b.def_at(width, v, &[w, x]);
        b.ret_at(width, v, 0);
        let c = same(&b.0, &toy(4, &[]), &[]);
        assert_eq!(ops(&c), ["op", "copy", "op", "op", "op", "ret"]);
        assert_eq!(copies(&c), [(r(1), r(0))]);
    }
}

/// A load is a copy where the instruction after it does not read what it loads, though a later one is the only other
/// that does: `y` takes the register that holds `x`, which it reads, and writes it before `z` reads `t`, so `z` cannot
/// read `t` there. The copy keeps `t` in r1, and the store and its slot are gone, as no load follows.
#[test]
fn a_load_is_a_copy_where_the_instruction_after_it_does_not_read_what_it_loads() {
    let mut b = Builder::new();
    let (x, t, y, z) = (b.var("x"), b.var("t"), b.var("y"), b.var("z"));
    let s = b.slot("s", true);
    b.def(x, &[]);
    b.store(s, x);
    b.load(t, s);
    b.def(y, &[x]);
    b.def(z, &[t, y]);
    b.ret(z, 0);
    let c = same(&b.0, &toy(3, &[]), &[]);
    assert_eq!(ops(&c), ["op", "copy", "op", "op", "ret"]);
    assert_eq!(copies(&c), [(r(1), r(0))]);
}

/// A load is left out where the instruction it is for comes after another load, as the first of two does that are for
/// one instruction: `p` and `q` arrive in r0 and r1 and are stored on entry, and `z` reads `t` and `u`, loaded from
/// those slots; each is read in the register it arrived in, and the loads and the stores are left out. So it is where
/// the load between goes to the register that holds what the first loads, as `u` goes to r1 where `q` is, if `t` is
/// loaded from `q`'s slot: the load between is left out as well, and writes no register.
#[test]
fn a_load_is_left_out_where_the_instruction_it_is_for_comes_after_another_load() {
    for swapped in [false, true] {
        let mut b = Builder::new();
        let (p, q) = (b.param("p", 0), b.param("q", 1));
        let (t, u, z) = (b.var("t"), b.var("u"), b.var("z"));
        let (sp, sq) = (b.slot("p", true), b.slot("q", true));
        b.store(sp, p);
        b.store(sq, q);
        let (first, second) = if swapped { (sq, sp) } else { (sp, sq) };
        b.load(t, first);
        b.load(u, second);
        b.def(z, &[t, u]);
        b.ret(z, 0);
        let c = same(&b.0, &toy(4, &[]), &[5, 6]);
        assert_eq!(ops(&c), ["op", "ret"]);
        let Item::Inst(read) = &c.body[0] else { panic!("an instruction") };
        let (at_t, at_u) = if swapped { (r(1), r(0)) } else { (r(0), r(1)) };
        assert_eq!(read.args[2].operand, Operand::Reg(at_t, Width::W64));
        assert_eq!(read.args[3].operand, Operand::Reg(at_u, Width::W64));
    }
}

/// A load is a copy where another load for its instruction writes the register that holds what it loads, as a load
/// does that is not left out: `t` is loaded from `q`'s slot, which r1 holds, and `u`, which `w` reads as well, is
/// loaded to r1 from `p`'s. `t` is copied out of r1 before `u` comes there, and the copy takes r0, where `p` was, so
/// `u` is loaded from its slot.
#[test]
fn a_load_is_a_copy_where_another_load_for_its_instruction_writes_the_register_that_holds_what_it_loads() {
    let mut b = Builder::new();
    let (p, q) = (b.param("p", 0), b.param("q", 1));
    let (t, u, z, w) = (b.var("t"), b.var("u"), b.var("z"), b.var("w"));
    let (sp, sq) = (b.slot("p", true), b.slot("q", true));
    b.store(sp, p);
    b.store(sq, q);
    b.load(t, sq);
    b.load(u, sp);
    b.def(z, &[t, u]);
    b.def(w, &[z, u]);
    b.ret(w, 0);
    let c = same(&b.0, &toy(4, &[]), &[5, 6]);
    assert_eq!(ops(&c), ["store", "copy", "load", "op", "op", "ret"]);
    assert_eq!(copies(&c), [(r(0), r(1))]);
}

/// A load is a copy where a load after it writes the register that holds what it loads, though that load is a copy only
/// as a third writes the register that holds what it loads in turn: `t`, `u` and `v` take r0, r1 and r2, and are loaded
/// from the slots of `pb`, `pc` and `pa`, which r1, r2 and r0 hold. `w` reads `v` as well, so the load of `v` writes
/// r2; `u` is copied out of r2 before it, and `t` out of r1 before that. The copy of `t` takes r0, where `pa` was, so
/// `v` is loaded from its slot.
#[test]
fn a_load_is_a_copy_where_the_loads_after_it_write_the_registers_that_hold_what_each_loads_one_after_another() {
    let mut b = Builder::new();
    let (pa, pb, pc) = (b.param("pa", 0), b.param("pb", 1), b.param("pc", 2));
    let (t, u, v, z, w) = (b.var("t"), b.var("u"), b.var("v"), b.var("z"), b.var("w"));
    let (sa, sb, sc) = (b.slot("pa", true), b.slot("pb", true), b.slot("pc", true));
    b.store(sa, pa);
    b.store(sb, pb);
    b.store(sc, pc);
    b.load(t, sb);
    b.load(u, sc);
    b.load(v, sa);
    b.def(z, &[t, u, v]);
    b.def(w, &[z, v]);
    b.ret(w, 0);
    let c = same(&b.0, &toy(4, &[]), &[5, 6, 7]);
    assert_eq!(ops(&c), ["store", "copy", "copy", "load", "op", "op", "ret"]);
    assert_eq!(copies(&c), [(r(0), r(1)), (r(1), r(2))]);
}

/// A copy from the register it writes is left out, as any copy between one register is: `t` is loaded to r0, where `x`
/// was, and read twice, so the load is a copy of r0 to r0.
#[test]
fn a_load_that_is_a_copy_from_the_register_it_writes_is_left_out() {
    let mut b = Builder::new();
    let (x, t, z, w) = (b.var("x"), b.var("t"), b.var("z"), b.var("w"));
    let s = b.slot("s", true);
    b.def(x, &[]);
    b.store(s, x);
    b.load(t, s);
    b.def(z, &[t]);
    b.def(w, &[t, z]);
    b.ret(w, 0);
    let c = same(&b.0, &toy(2, &[]), &[]);
    assert_eq!(ops(&c), ["op", "op", "op", "ret"]);
    assert!(copies(&c).is_empty());
}

/// A load that is a copy leaves what it loads in the register it loads to, as any load does, and a later load into that
/// register is a copy from it, and left out, where another register holds the same first. Each of the two ways to the
/// label stores `x` from r2 and loads `t` to r3, and each writes `x` of its own, so that at the label what each
/// register holds is not one value to the copies, though the slot keeps the same in both, in `x` and in `t`. `u` is
/// loaded to r3, where the way put `t`, and no copy is made, as one from r2 would be. `n` says which way the kernel
/// takes, the jump if 2, as the Toy's `dec` says; it is run both ways.
#[test]
fn a_load_that_is_a_copy_leaves_what_it_loads_in_the_register_it_loads_to() {
    let mut b = Builder::new();
    let (n, acc, x, t, u) = (b.param("n", 1), b.var("acc"), b.var("x"), b.var("t"), b.var("u"));
    let s = b.slot("s", true);
    let add_twice = |b: &mut Builder, v: VReg| {
        for _ in 0..2 {
            b.inst("add", &[(acc, Access::ReadWrite), (v, Access::Read)], Flow::Next);
        }
    };
    b.def(acc, &[]);
    b.def(x, &[]);
    b.store(s, x);
    b.load(t, s);
    add_twice(&mut b, t);
    b.inst("dec", &[(n, Access::ReadWrite)], Flow::Branch(Label(1)));
    b.def(x, &[]);
    b.store(s, x);
    b.load(t, s);
    add_twice(&mut b, t);
    b.label(1);
    b.load(u, s);
    add_twice(&mut b, u);
    b.inst("add", &[(acc, Access::ReadWrite), (x, Access::Read)], Flow::Next);
    b.inst("add", &[(acc, Access::ReadWrite), (n, Access::Read)], Flow::Next);
    b.ret(acc, 0);
    for way in [1, 2] {
        let c = same(&b.0, &toy(4, &[]), &[way]);
        let ops_kept =
            ["op", "op", "copy", "add", "add", "dec", "op", "copy", "add", "add", "add", "add", "add", "add", "ret"];
        assert_eq!(ops(&c), ops_kept);
        assert_eq!(copies(&c), [(r(3), r(2)), (r(3), r(2))]);
    }
}

/// A register holds what a slot keeps at a label if each way to it brings that. Two ways come to the label, the jump
/// and the code `way` before it, and `x` is stored before them, from r1, where it stays, read after the label. Where
/// `way` leaves r1 as it was, the load after the label is left out; where it stores something else to the slot, the
/// ways bring it in two registers, and the load stays. `n` says which way the kernel takes, the jump if 2, as the Toy's
/// `dec` says; it is run both ways.
#[test]
fn a_register_holds_what_a_slot_keeps_at_a_label_if_every_way_to_it_brings_it() {
    let join = |way: fn(&mut Builder, SlotId)| {
        let mut b = Builder::new();
        let (n, x, t, z) = (b.param("n", 0), b.var("x"), b.var("t"), b.var("z"));
        let s = b.slot("s", true);
        b.def(x, &[]);
        b.store(s, x);
        b.inst("dec", &[(n, Access::ReadWrite)], Flow::Branch(Label(1)));
        way(&mut b, s);
        b.label(1);
        b.load(t, s);
        b.def(z, &[t, x]);
        b.ret(z, 0);
        let (jumps, falls) = (same(&b.0, &toy(4, &[]), &[2]), same(&b.0, &toy(4, &[]), &[1]));
        assert_eq!(ops(&jumps), ops(&falls));
        ops(&falls)
    };
    let agree = |b: &mut Builder, _| {
        let y = b.var("y");
        b.def(y, &[]);
    };
    assert_eq!(join(agree), ["op", "dec", "op", "op", "ret"]);
    let differ = |b: &mut Builder, s| {
        let y = b.var("y");
        b.def(y, &[]);
        b.store(s, y);
    };
    assert_eq!(join(differ), ["op", "store", "dec", "op", "store", "load", "op", "ret"]);
}

/// A register holds what a slot keeps at a label only if each way to it has the register hold that slot's content, and
/// not another's. `w` and `x` are stored to `b` and `a`, and `w` and `n` are read after the label, so `x` has r2, and
/// `y`, which the way that falls through writes before it stores it to `b`, takes r2 where `x` was. At the label r2
/// holds `a` on the jump and `b` on the fall, and the two loads stay, whichever comes first: the first one's register
/// would be r2, and end what it holds. `n` says which way the kernel takes, the jump if 2, as the Toy's `dec` says; it
/// is run both ways.
#[test]
fn a_register_holds_what_a_slot_keeps_at_a_label_only_if_each_way_to_it_has_it_hold_that_slot_and_not_another() {
    for a_first in [true, false] {
        let mut b = Builder::new();
        let (n, w, x, y) = (b.param("n", 0), b.var("w"), b.var("x"), b.var("y"));
        let (t, u, v, z) = (b.var("t"), b.var("u"), b.var("v"), b.var("z"));
        let (sa, sb) = (b.slot("a", true), b.slot("b", true));
        b.def(w, &[]);
        b.def(x, &[]);
        b.store(sb, w);
        b.store(sa, x);
        b.inst("dec", &[(n, Access::ReadWrite)], Flow::Branch(Label(1)));
        b.def(y, &[]);
        b.store(sb, y);
        b.label(1);
        let (first, second) = if a_first { (sa, sb) } else { (sb, sa) };
        b.load(t, first);
        b.def(v, &[t]);
        b.load(u, second);
        b.def(z, &[u, v, w, n]);
        b.ret(z, 0);
        for way in [1, 2] {
            let c = same(&b.0, &toy(5, &[]), &[way]);
            let ops_kept = ["op", "op", "store", "store", "dec", "op", "store", "load", "op", "load", "op", "ret"];
            assert_eq!(ops(&c), ops_kept);
            let slots = [("store", sb.0), ("store", sa.0), ("store", sb.0), ("load", first.0), ("load", second.0)];
            assert_eq!(moves(&c), slots);
            assert_eq!(regs_of(&c, "x").into_iter().chain(regs_of(&c, "y")).collect::<Vec<_>>(), [r(2), r(2)]);
        }
    }
}

/// A register holds what a slot keeps at a label though each way to it brings that in a variable of its own: here both
/// ways bring it in r0, the one in `x` and the other in `y`, which are made one, written on each way. `z` reads it in
/// r0, the load after the label is left out, and so are the stores, which no load follows now. `n` and `m` say which
/// way the kernel takes, as the Toy's `dec` says: the first jump over the code of `x` if `n` is not 1, the second over
/// that of `y` if `m` is not 1; it is run by each of the three ways.
#[test]
fn a_register_holds_what_a_slot_keeps_at_a_label_though_each_way_brings_it_in_a_variable_of_its_own() {
    let mut b = Builder::new();
    let (n, m, x, y, t, z) = (b.param("n", 0), b.param("m", 1), b.var("x"), b.var("y"), b.var("t"), b.var("z"));
    let s = b.slot("s", true);
    b.inst("dec", &[(n, Access::ReadWrite)], Flow::Branch(Label(1)));
    b.def(x, &[]);
    b.store(s, x);
    b.inst("dec", &[(m, Access::ReadWrite)], Flow::Branch(Label(2)));
    b.label(1);
    b.def(y, &[]);
    b.store(s, y);
    b.label(2);
    b.load(t, s);
    b.def(z, &[t]);
    b.ret(z, 0);
    for ways in [[1, 5], [2, 5], [1, 1]] {
        let c = same(&b.0, &toy(3, &[]), &ways);
        assert_eq!(ops(&c), ["dec", "op", "dec", "op", "op", "ret"]);
        assert_eq!(regs_of(&c, "x").into_iter().chain(regs_of(&c, "y")).collect::<Vec<_>>(), [r(0)]);
    }
}

/// A parameter stays as it is where variables are made one. `x` comes after `y` and arrives in r0, stored from there;
/// the way that falls through loads the slot to `y`, in r0 too, so that the ways bring its content in `y` and in `x`.
/// `z` reads it in `x`, the parameter, though `y` is the lower of the two, and `y` has no register left. `n` says which
/// way the kernel takes, the jump if 2, as the Toy's `dec` says; it is run both ways.
#[test]
fn a_parameter_stays_as_it_is_where_variables_are_made_one() {
    let mut b = Builder::new();
    let (n, y, x, t, z) = (b.param("n", 1), b.var("y"), b.param("x", 0), b.var("t"), b.var("z"));
    let s = b.slot("s", true);
    b.store(s, x);
    b.inst("dec", &[(n, Access::ReadWrite)], Flow::Branch(Label(1)));
    b.load(y, s);
    b.label(1);
    b.load(t, s);
    b.def(z, &[t]);
    b.ret(z, 0);
    for way in [1, 2] {
        let c = same(&b.0, &toy(3, &[]), &[way, 7]);
        assert_eq!(ops(&c), ["dec", "op", "ret"]);
        assert_eq!(regs_of(&c, "x"), [r(0)]);
        assert_eq!(regs_of(&c, "y"), []);
    }
}

/// A register holds what a slot keeps in a loop if the way into it and the jump back both bring that. `x` is stored
/// before the loop, from r1, where it stays, read after the loop. Where the body leaves r1 as it was, the load at the
/// top of the loop is left out; where it stores something else to the slot, the jump back brings it in another
/// register, and the load stays.
#[test]
fn a_register_holds_what_a_slot_keeps_in_a_loop_if_the_way_in_and_the_jump_back_both_bring_it() {
    let looped = |body: fn(&mut Builder, SlotId, VReg)| {
        let mut b = Builder::new();
        let (n, x, acc, t, z) = (b.param("n", 0), b.var("x"), b.var("acc"), b.var("t"), b.var("z"));
        let s = b.slot("s", true);
        b.def(x, &[]);
        b.def(acc, &[]);
        b.store(s, x);
        b.label(1);
        b.load(t, s);
        b.inst("add", &[(acc, Access::ReadWrite), (t, Access::Read)], Flow::Next);
        body(&mut b, s, acc);
        b.inst("dec", &[(n, Access::ReadWrite)], Flow::Branch(Label(1)));
        b.def(z, &[acc, x]);
        b.ret(z, 0);
        let ways: Vec<_> = (1..=3).map(|turns| ops(&same(&b.0, &toy(5, &[]), &[turns]))).collect();
        assert!(ways.windows(2).all(|w| w[0] == w[1]));
        ways[0].clone()
    };
    let agree = |b: &mut Builder, _, acc| {
        let y = b.var("y");
        b.def(y, &[acc]);
    };
    assert_eq!(looped(agree), ["op", "op", "add", "op", "dec", "op", "ret"]);
    let differ = |b: &mut Builder, s, acc| {
        let y = b.var("y");
        b.def(y, &[acc]);
        b.store(s, y);
    };
    assert_eq!(looped(differ), ["op", "op", "store", "load", "add", "op", "store", "dec", "op", "ret"]);
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
fn a_value_live_across_a_call_that_clears_upper_halves_spills_if_all_of_it_is_read_after() {
    let mut b = Builder::new();
    let (x, y) = (b.var("x"), b.var("y"));
    b.def(x, &[]);
    b.call_with(None, &[], upper(&[0, 1, 2]));
    b.def(y, &[x]);
    b.ret(y, 0);
    // The run puts junk in the upper half of each register, so one that kept `x` would return another value.
    let c = same(&b.0, &toy(3, &[]), &[]);
    assert_eq!(spilled(&c), ["x"]);
    assert_eq!(ops(&c), ["op", "store", "call", "load", "op", "ret"]);
}

#[test]
fn a_value_live_across_a_call_that_clears_upper_halves_stays_in_a_register_if_its_lower_half_is_read_after() {
    let mut b = Builder::new();
    let (x, y) = (b.var("x"), b.var("y"));
    b.def(x, &[]);
    b.call_with(None, &[], upper(&[0, 1, 2]));
    b.def_low(y, &[x]);
    b.ret(y, 0);
    let c = same(&b.0, &toy(3, &[]), &[]);
    assert!(spilled(&c).is_empty());
    assert_eq!(ops(&c), ["op", "call", "op", "ret"]);
}

#[test]
fn a_read_of_all_of_a_value_before_a_call_that_clears_upper_halves_does_not_spill_it() {
    let mut b = Builder::new();
    let (x, y, z) = (b.var("x"), b.var("y"), b.var("z"));
    b.def(x, &[]);
    b.def(y, &[x]);
    b.call_with(None, &[], upper(&[0, 1, 2]));
    b.def_low(z, &[x, y]);
    b.ret(z, 0);
    let c = same(&b.0, &toy(3, &[]), &[]);
    assert!(spilled(&c).is_empty());
}

#[test]
fn a_value_is_spilled_if_a_path_after_a_call_that_clears_upper_halves_reads_all_of_it() {
    // The path where `dec` falls through adds `x` to `n`; the other does not read it.
    let mut b = Builder::new();
    let n = b.param("n", 0);
    let x = b.var("x");
    b.def(x, &[]);
    b.call_with(None, &[], upper(&[1, 2]));
    b.inst("dec", &[(n, Access::ReadWrite)], Flow::Branch(Label(1)));
    b.inst("add", &[(n, Access::ReadWrite), (x, Access::Read)], Flow::Next);
    b.label(1);
    b.ret(n, 0);
    for arg in [1, 2] {
        assert_eq!(spilled(&same(&b.0, &toy(3, &[]), &[arg])), ["x"], "for {arg}");
    }
}

#[test]
fn a_write_of_all_of_a_value_after_a_call_that_clears_upper_halves_ends_the_need_for_its_upper_half() {
    // The first `x` has its lower half read after the call; the upper half read after that is of the second, which a
    // write of 32 bits replaces all of.
    let mut b = Builder::new();
    let (x, t, y) = (b.var("x"), b.var("t"), b.var("y"));
    b.def(x, &[]);
    b.call_with(None, &[], upper(&[0, 1, 2]));
    b.def_low(t, &[x]);
    b.def_low(x, &[t]);
    b.def(y, &[x]);
    b.ret(y, 0);
    assert!(spilled(&same(&b.0, &toy(3, &[]), &[])).is_empty());
}

#[test]
fn a_write_of_part_of_a_value_after_a_call_that_clears_upper_halves_ends_the_need_for_its_upper_half() {
    // `add32` reads the lower halves of `x` and `z`, and clears the upper half of `x`, which `op` reads after.
    let mut b = Builder::new();
    let (x, y, z) = (b.var("x"), b.var("y"), b.var("z"));
    b.def(x, &[]);
    b.def(z, &[]);
    b.call_with(None, &[], upper(&[0, 1, 2]));
    b.add_low(x, z);
    b.def(y, &[x]);
    b.ret(y, 0);
    assert!(spilled(&same(&b.0, &toy(3, &[]), &[])).is_empty());
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
    // Each of `x` and `y` is made where its argument goes, and `res` where the result comes back, so that no copy moves
    // anything, and the body has none.
    assert_eq!([reg_of(&c, "x"), reg_of(&c, "y"), reg_of(&c, "res")], [r(1), r(2), r(0)]);
    assert_eq!(ops(&c), ["op", "op", "call", "ret"]);
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
    assert_eq!(ops(&c), ["copy", "copy", "copy", "call", "ret"]);
}

/// A parameter that moves on entry moves as much as it is kept in: 32 bits where no instruction reads more.
#[test]
fn a_parameter_that_moves_on_entry_moves_as_much_as_it_is_kept_in() {
    let mut b = Builder::new();
    let p = b.param("p", 0);
    let y = b.var("y");
    b.call(None, &[], &[0, 1]);
    b.def_low(y, &[p]);
    b.ret(y, 0);
    let c = same(&b.0, &toy(3, &[2]), &[7]);
    assert_eq!(regs_of(&c, "p"), [r(2), r(0)]);
    let Item::Inst(first) = &c.body[0] else { panic!("an instruction first") };
    let to = (first.op, first.args[0].operand, first.args[1].operand);
    assert_eq!(to, ("copy", Operand::Reg(r(2), Width::W32), Operand::Reg(r(0), Width::W32)));
}

/// A copy of one variable to another is left out where the two take one register, as one a call's argument or result
/// has, and a copy the program makes of its own: where `x` is dead after it, `y` takes its register. It stays where `x`
/// is read after it, and the two take two.
#[test]
fn a_copy_of_one_variable_to_another_in_one_register_is_left_out() {
    let copy = |to, from, width| Item::Inst(Toy::copy(&toy(0, &[]), to, from, width));
    let program = |both: bool| {
        let mut b = Builder::new();
        let (x, y, z) = (b.var("x"), b.var("y"), b.var("z"));
        b.def(x, &[]);
        b.0.body.push(copy(y, x, Width::W64));
        b.def(z, &if both { vec![x, y] } else { vec![y] });
        b.ret(z, 0);
        b.0
    };
    let c = same(&program(false), &toy(3, &[]), &[]);
    assert_eq!((reg_of(&c, "x"), reg_of(&c, "y")), (r(0), r(0)));
    assert_eq!(ops(&c), ["op", "op", "ret"]);
    let c = same(&program(true), &toy(3, &[]), &[]);
    assert_ne!(reg_of(&c, "x"), reg_of(&c, "y"));
    assert_eq!(ops(&c), ["op", "copy", "op", "ret"]);
}

/// A copy is left out where it moves all that its variable `to` is kept in: of 32 bits, where no instruction names more
/// of `to`. One of less stays, though the two take one register: it moves 32 bits of a variable kept in 64, and clears
/// the upper half.
#[test]
fn a_copy_of_less_than_its_variable_is_kept_in_stays() {
    let copy = |to, from| Item::Inst(Toy::copy(&toy(0, &[]), to, from, Width::W32));
    let program = |whole: bool| {
        let mut b = Builder::new();
        let (x, y, z) = (b.var("x"), b.var("y"), b.var("z"));
        b.def_low(x, &[]);
        b.0.body.push(copy(y, x));
        if whole {
            b.def(z, &[y])
        } else {
            b.def_low(z, &[y])
        }
        b.ret(z, 0);
        b.0
    };
    let c = same(&program(false), &toy(3, &[]), &[]);
    assert_eq!((reg_of(&c, "x"), reg_of(&c, "y")), (r(0), r(0)));
    assert_eq!(ops(&c), ["op", "op", "ret"]);
    let c = same(&program(true), &toy(3, &[]), &[]);
    assert_eq!((reg_of(&c, "x"), reg_of(&c, "y")), (r(0), r(0)));
    assert_eq!(ops(&c), ["op", "copy", "op", "ret"]);
}

/// The copy of a call's result, from the variable pinned in the register it comes back in, is as wide as the call gives
/// the result back: `given` is kept in the 32 bits the callee gives back, though `d` is kept in 64 where an instruction
/// reads all of it. In one register, the copy is left out where no instruction after it reads all of `d` before one
/// writes it, and stays where one does, in its block or in one after: it clears the upper half, which a callee may
/// leave set, as a call does in `run`.
#[test]
fn a_copy_of_a_calls_result_stays_where_all_of_its_variable_is_read_after() {
    // The ops of the program once compiled, where `after` reads `d` into `z`, the result, after the copy.
    let compiled = |after: fn(&mut Builder, VReg, VReg)| {
        let mut b = Builder::new();
        let (d, z, given) = (b.var("d"), b.var("z"), b.var("given"));
        b.0.pinned.push((given, r(0)));
        let implicit = vec![reg_arg(given, Width::W32, Access::Write)];
        b.0.body.push(Item::Inst(Inst { op: "call", args: vec![], implicit, flow: Flow::Next, clobbers: vec![] }));
        b.0.body.push(Item::Inst(Toy::copy(&toy(0, &[]), d, given, Width::W32)));
        after(&mut b, d, z);
        b.ret(z, 0);
        let c = same(&b.0, &toy(3, &[]), &[]);
        assert_eq!(reg_of(&c, "d"), r(0));
        ops(&c)
    };
    assert_eq!(compiled(|b, d, z| b.def_low(z, &[d])), ["call", "op", "ret"]);
    assert_eq!(compiled(|b, d, z| b.def(z, &[d])), ["call", "copy", "op", "ret"]);
    let next_block = |b: &mut Builder, d, z| {
        b.label(1);
        b.def(z, &[d]);
    };
    assert_eq!(compiled(next_block), ["call", "copy", "op", "ret"]);
    let written_first = |b: &mut Builder, d, z| {
        b.def_low(z, &[d]);
        b.def(d, &[z]);
        b.def(z, &[d]);
    };
    assert_eq!(compiled(written_first), ["call", "op", "op", "op", "ret"]);
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

/// A parameter that moves on entry to a register the call keeps, and is passed in the register it arrived in, which
/// holds it still, is not copied back there.
#[test]
fn an_argument_is_not_copied_to_the_register_that_holds_it_already() {
    let mut b = Builder::new();
    let p = b.param("p", 0);
    let y = b.var("y");
    b.call(None, &[(p, 0)], &[0, 1]);
    b.def(y, &[p]);
    b.ret(y, 0);
    let c = same(&b.0, &toy(3, &[2]), &[7]);
    assert_eq!(ops(&c), ["copy", "call", "op", "ret"]);
    assert_eq!(copies(&c), [(r(2), r(0))]);
}

/// A copy that stays reads the register its value arrived in, not the variable that was copied from there, so that the
/// copies that set a call's arguments do not wait for each other, where an instruction after it reads that variable.
/// The last read of a variable stays where it is: another register would leave the copy that wrote it moving what
/// nothing reads.
#[test]
fn an_argument_copy_reads_the_register_the_value_arrived_in() {
    let mut b = Builder::new();
    let p = b.param("p", 0);
    let y = b.var("y");
    b.call(None, &[(p, 0), (p, 1)], &[0, 1]);
    b.def(y, &[p]);
    b.ret(y, 0);
    let c = same(&b.0, &toy(3, &[2]), &[7]);
    assert_eq!(ops(&c), ["copy", "copy", "call", "op", "ret"]);
    assert_eq!(copies(&c), [(r(2), r(0)), (r(1), r(0))]);
    let mut b = Builder::new();
    let (p, y) = (b.param("p", 0), b.var("y"));
    b.call(None, &[(p, 0), (p, 1)], &[0, 1]);
    b.call(None, &[(p, 1)], &[0, 1]);
    b.def(y, &[p]);
    b.ret(y, 0);
    let c = same(&b.0, &toy(3, &[2]), &[7]);
    assert_eq!(copies(&c), [(r(2), r(0)), (r(1), r(0)), (r(1), r(2))]);
    // `q`, a copy of `p`, goes to r0 and `p` to r1, and neither is read after. The copy to r0 writes r0 while `p` is
    // live, so `p` moves on entry to a register of its own, r2, as `m` holds r1 until `z` reads it; and the copy to r1
    // reads `p` there, not in r0, where it arrived.
    let mut b = Builder::new();
    let (p, m, q, z) = (b.param("p", 0), b.param("m", 1), b.var("q"), b.var("z"));
    b.copy(q, p, Width::W64);
    b.def(z, &[m]);
    b.call(None, &[(q, 0), (p, 1)], &[]);
    b.ret(z, 0);
    let c = same(&b.0, &toy(4, &[]), &[7, 9]);
    assert_eq!(copies(&c), [(r(2), r(0)), (r(1), r(2)), (r(0), r(3))]);
}

/// A parameter that a call passes twice, and nothing reads after, stays in the register it arrived in, the first
/// argument's, though `m` holds the second's until the arguments are set: the copy to the first moves nothing, and the
/// copy to the second reads it there, once `m` is read. It is not moved on entry to a register of its own.
#[test]
fn a_parameter_passed_twice_and_read_no_more_stays_where_it_arrived() {
    let mut b = Builder::new();
    let (p, m, z) = (b.param("p", 0), b.param("m", 1), b.var("z"));
    b.def(z, &[m]);
    b.call(None, &[(p, 0), (p, 1)], &[0, 1]);
    b.ret(z, 0);
    let c = same(&b.0, &toy(4, &[3]), &[7, 9]);
    assert_eq!(regs_of(&c, "p"), [r(0), r(0), r(1)]);
    assert_eq!(copies(&c), [(r(1), r(0)), (r(0), r(2))]);
}

/// A parameter that a call takes the low half of stays in the register it arrived in, though the kernel reads all of it
/// after the call: the copy that passes the low half moves nothing, so it is left out, and does not clear the rest.
#[test]
fn a_parameter_passed_in_part_stays_where_it_arrived() {
    let mut b = Builder::new();
    let (p, y) = (b.param("p", 0), b.var("y"));
    b.call_at(Width::W32, None, &[(p, 0)], vec![]);
    b.def(y, &[p]);
    b.ret(y, 0);
    let c = same(&b.0, &toy(3, &[]), &[0x1_0000_0007]);
    assert_eq!(ops(&c), ["call", "op", "ret"]);
}

/// What a register holds is not known after a call that changes it.
#[test]
fn a_register_holds_nothing_known_after_a_call_that_changes_it() {
    let mut b = Builder::new();
    let (p, y) = (b.param("p", 0), b.var("y"));
    b.call(None, &[(p, 0)], &[0, 1]);
    b.call(None, &[(p, 0)], &[0, 1]);
    b.def(y, &[p]);
    b.ret(y, 0);
    let c = same(&b.0, &toy(3, &[2]), &[7]);
    assert_eq!(copies(&c), [(r(2), r(0)), (r(0), r(2))]);
    // Two calls that leave r0 as it was, and a third that changes it, which `p` waits out in r2: the second passes `p`
    // in r0 with no move, and so does the first, for r0 holds `p` as it arrived.
    let mut b = Builder::new();
    let (p, y) = (b.param("p", 0), b.var("y"));
    b.call(None, &[(p, 0)], &[1]);
    b.call(None, &[(p, 0)], &[1]);
    b.call(None, &[], &[0, 1]);
    b.def(y, &[p]);
    b.ret(y, 0);
    let c = same(&b.0, &toy(3, &[2]), &[7]);
    assert_eq!(copies(&c), [(r(2), r(0))]);
}

/// A register holds at a label what every way to it brings. One way comes to the first label, the code before it, and
/// the call after it passes `p` in r0, which the call before it leaves, with no move. A third call changes r0, and `p`
/// waits it out in r2.
#[test]
fn a_label_that_one_way_comes_to_leaves_what_a_register_holds() {
    let mut b = Builder::new();
    let (p, y) = (b.param("p", 0), b.var("y"));
    b.call(None, &[(p, 0)], &[1]);
    b.label(1);
    b.call(None, &[(p, 0)], &[1]);
    b.call(None, &[], &[0, 1]);
    b.def(y, &[p]);
    b.ret(y, 0);
    let c = same(&b.0, &toy(3, &[2]), &[7]);
    assert_eq!(copies(&c), [(r(2), r(0))]);
}

/// Two ways come to a label, a jump and the code before it, and the call there passes `p` in r0, with the move left out
/// where both ways leave r0 as `p` was, and kept where either changes it, whichever way is the first the allocator
/// follows. `n` says which way the kernel takes, the jump if 2, as the Toy's `dec` says; it is run both ways.
#[test]
fn a_register_holds_at_a_label_what_both_ways_to_it_bring() {
    // `way` is the code from the jump to the label, and the call at the label gives back its result in r1. A call that
    // changes r0 follows it, and `p` and `d` are read after it, so that `p` waits it out in a register of its own.
    let join = |way: fn(&mut Builder, VReg, VReg)| {
        let mut b = Builder::new();
        let (p, n, d, z) = (b.param("p", 0), b.param("n", 1), b.var("d"), b.var("z"));
        way(&mut b, p, n);
        b.label(1);
        b.call(Some((d, 1)), &[(p, 0)], &[1]);
        b.call(None, &[], &[0]);
        b.def(z, &[p, d]);
        b.ret(z, 0);
        let (jumps, falls) = (same(&b.0, &toy(4, &[]), &[7, 2]), same(&b.0, &toy(4, &[]), &[7, 1]));
        assert_eq!(copies(&jumps), copies(&falls));
        copies(&falls)
    };
    // The code before the label passes `p` in r0, which that call leaves as it was.
    let agree = |b: &mut Builder, p, n| {
        b.inst("dec", &[(n, Access::ReadWrite)], Flow::Branch(Label(1)));
        b.call(None, &[(p, 0)], &[1]);
    };
    assert_eq!(join(agree), [(r(2), r(0))]);
    // The code before the label calls what changes r0, and the jump skips it.
    let falls = |b: &mut Builder, _, n| {
        b.inst("dec", &[(n, Access::ReadWrite)], Flow::Branch(Label(1)));
        b.call(None, &[], &[0]);
    };
    assert_eq!(join(falls), [(r(2), r(0)), (r(0), r(2))]);
    // The jump is after a call that changes r0, and the code before the label passes `p` in r0 again.
    let jumps = |b: &mut Builder, p, n| {
        b.call(None, &[], &[0]);
        b.inst("dec", &[(n, Access::ReadWrite)], Flow::Branch(Label(1)));
        b.call(None, &[(p, 0)], &[1]);
    };
    assert_eq!(join(jumps), [(r(2), r(0)), (r(0), r(2)), (r(0), r(2))]);
}

/// A register holds in a loop what the way into it and the jump back both bring: the call passes `p` in r0 with no move
/// where it leaves r0 as it was, and has the move each time round where it changes r0. `p` is read after a call that
/// changes r0, so that it waits that call out in a register of its own, where a move on entry puts it.
#[test]
fn a_register_holds_in_a_loop_what_the_way_in_and_the_jump_back_bring() {
    let looped = |clobbers: &[u8]| {
        let mut b = Builder::new();
        let (p, n) = (b.param("p", 0), b.param("n", 1));
        let (acc, d, z) = (b.var("acc"), b.var("d"), b.var("z"));
        b.def(acc, &[]);
        b.label(2);
        b.call(Some((d, 1)), &[(p, 0)], clobbers);
        b.inst("add", &[(acc, Access::ReadWrite), (d, Access::Read)], Flow::Next);
        b.inst("dec", &[(n, Access::ReadWrite)], Flow::Branch(Label(2)));
        b.call(None, &[], &[0]);
        b.def(z, &[acc, p]);
        b.ret(z, 0);
        copies(&same(&b.0, &toy(5, &[]), &[7, 3]))
    };
    // The moves of `p` and `n` on entry, to wait out the calls, and the move of `p` to r0 where the call changes it.
    assert_eq!(looped(&[1]), [(r(2), r(0)), (r(3), r(1))]);
    assert_eq!(looped(&[0, 1]), [(r(2), r(0)), (r(3), r(1)), (r(0), r(2))]);
}

/// Two ways to a label that each move a value written before they parted to one register bring one value, though each
/// by a copy of its own: the move after they meet is left out.
#[test]
fn a_value_written_before_two_ways_part_is_one_value_where_they_meet() {
    for ways in [[1, 1], [1, 2], [2, 1]] {
        let mut b = Builder::new();
        let (n1, n2) = (b.param("n1", 0), b.param("n2", 1));
        let (x, y, z) = (b.var("x"), b.var("y"), b.var("z"));
        b.def(x, &[]);
        b.inst("dec", &[(n1, Access::ReadWrite)], Flow::Branch(Label(1)));
        b.copy(y, x, Width::W64);
        b.inst("dec", &[(n2, Access::ReadWrite)], Flow::Branch(Label(2)));
        b.label(1);
        b.copy(y, x, Width::W64);
        b.label(2);
        b.copy(y, x, Width::W64);
        b.def(z, &[y, x]);
        b.ret(z, 0);
        let c = same(&b.0, &toy(5, &[]), &ways);
        assert_eq!(["x", "y"].map(|v| reg_of(&c, v)), [r(2), r(0)]);
        assert_eq!(copies(&c), [(r(0), r(2)), (r(0), r(2))]);
    }
}

/// A parameter is one value on each way to a label that moves it to one register, as a value written is: it is in the
/// register it arrives in, whichever way the kernel takes, so the move after the ways meet is left out.
#[test]
fn a_parameter_is_one_value_where_two_ways_that_move_it_to_one_register_meet() {
    for ways in [[1, 1], [1, 2], [2, 1]] {
        let mut b = Builder::new();
        let (n1, n2, x) = (b.param("n1", 0), b.param("n2", 1), b.param("x", 2));
        let (y, z) = (b.var("y"), b.var("z"));
        b.inst("dec", &[(n1, Access::ReadWrite)], Flow::Branch(Label(1)));
        b.copy(y, x, Width::W64);
        b.inst("dec", &[(n2, Access::ReadWrite)], Flow::Branch(Label(2)));
        b.label(1);
        b.copy(y, x, Width::W64);
        b.label(2);
        b.copy(y, x, Width::W64);
        b.def(z, &[y, x]);
        b.ret(z, 0);
        let c = same(&b.0, &toy(5, &[]), &[ways[0], ways[1], 7]);
        assert_eq!(["x", "y"].map(|v| reg_of(&c, v)), [r(2), r(0)]);
        assert_eq!(copies(&c), [(r(0), r(2)), (r(0), r(2))]);
    }
}

/// A register holds at a label as many bits of a value as the fewer that either way brings: one way copies all of `x`
/// to `a`, and the other 32 bits of it to `b`, which take one register, and so does `w`, so that the copy of all of `x`
/// to `w` after the ways meet stays. `n` says which way the kernel takes, the jump to the second if 2, as the Toy's
/// `dec` says; it is run both ways.
#[test]
fn a_register_holds_at_a_label_as_much_of_a_value_as_the_least_of_the_ways_bring() {
    let mut b = Builder::new();
    let (n, m, x) = (b.param("n", 0), b.param("m", 1), b.param("x", 2));
    let (a, half, w, u) = (b.var("a"), b.var("b"), b.var("w"), b.var("u"));
    b.inst("dec", &[(n, Access::ReadWrite)], Flow::Branch(Label(1)));
    b.copy(a, x, Width::W64);
    b.inst("dec", &[(m, Access::ReadWrite)], Flow::Branch(Label(2)));
    b.label(1);
    b.copy(half, x, Width::W32);
    b.label(2);
    b.copy(w, x, Width::W64);
    b.def(u, &[w, x]);
    b.ret(u, 0);
    for ways in [[1, 2], [2, 2]] {
        let c = same(&b.0, &toy(6, &[]), &[ways[0], ways[1], 7 | 1 << 40]);
        let [a, half, w] = ["a", "b", "w"].map(|v| reg_of(&c, v));
        assert_eq!([a, half], [w, w]);
        assert_eq!(copies(&c), [(w, r(2)); 3]);
    }
}

/// A register holds at a label a value only if every way to it brings that value: the register of `y` holds `x` on the
/// way that jumps over the second copy, and `z` on the two that make it, so it holds neither where they meet, and the
/// copy of `x` to `w`, which has the register of `y`, stays. `n` and `m` say which way the kernel takes, as the Toy's
/// `dec` says: the first jump if `n` is not 1, the second if `m` is not 1; it is run by each of the three ways.
#[test]
fn a_register_holds_at_a_label_only_the_value_that_every_way_to_it_brings() {
    let mut b = Builder::new();
    let (n, m, x, z) = (b.param("n", 0), b.param("m", 1), b.param("x", 2), b.param("z", 3));
    let (y, w, u) = (b.var("y"), b.var("w"), b.var("u"));
    b.inst("dec", &[(n, Access::ReadWrite)], Flow::Branch(Label(1)));
    b.copy(y, x, Width::W64);
    b.inst("dec", &[(m, Access::ReadWrite)], Flow::Branch(Label(2)));
    b.label(1);
    b.copy(y, z, Width::W64);
    b.label(2);
    b.copy(w, x, Width::W64);
    b.def(u, &[w, x, z]);
    b.ret(u, 0);
    for ways in [[1, 1], [1, 2], [2, 1]] {
        let c = same(&b.0, &toy(6, &[]), &[ways[0], ways[1], 7, 9]);
        let [y, w] = ["y", "w"].map(|v| reg_of(&c, v));
        assert_eq!(y, w);
        assert_eq!(copies(&c), [(y, r(2)), (y, r(3)), (w, r(2))]);
    }
}

/// A copy puts a value in a register and takes out the one that was there: `y` takes `x`, then `z`, then `x` again, and
/// the third copy stays, for the register holds `z` by then.
#[test]
fn a_register_holds_the_value_of_the_last_copy_into_it() {
    let mut b = Builder::new();
    let (x, z) = (b.param("x", 0), b.param("z", 1));
    let (y, w) = (b.var("y"), b.var("w"));
    b.copy(y, x, Width::W64);
    b.def(w, &[y]);
    b.copy(y, z, Width::W64);
    b.def(w, &[w, y]);
    b.copy(y, x, Width::W64);
    b.def(w, &[w, y, x, z]);
    b.ret(w, 0);
    let c = same(&b.0, &toy(4, &[]), &[3, 5]);
    assert_eq!(copies(&c), [(r(2), r(0)), (r(2), r(1)), (r(2), r(0)), (r(0), r(3))]);
}

/// And an instruction that writes it takes out the value too: `y` takes `x`, `op` writes `y`, and the copy of `x` after
/// it stays.
#[test]
fn a_register_an_instruction_writes_holds_nothing_known() {
    let mut b = Builder::new();
    let x = b.param("x", 0);
    let (y, w) = (b.var("y"), b.var("w"));
    b.copy(y, x, Width::W64);
    b.def(w, &[y]);
    b.def(y, &[y, w]);
    b.copy(y, x, Width::W64);
    b.def(w, &[w, y, x]);
    b.ret(w, 0);
    let c = same(&b.0, &toy(4, &[]), &[3]);
    assert_eq!(copies(&c), [(r(1), r(0)), (r(1), r(0)), (r(0), r(2))]);
}

/// A register that a copy of part of a value is put in holds that much of it and no more, as the rest is cleared: the
/// low half of `x` in the register `y` has is no copy of all of `x`, which `z` is, in the same register.
#[test]
fn a_register_a_copy_of_part_of_a_value_is_put_in_holds_that_part() {
    let mut b = Builder::new();
    let x = b.param("x", 0);
    let (t, y, z, u) = (b.var("t"), b.var("y"), b.var("z"), b.var("u"));
    b.def(t, &[]);
    b.copy(y, x, Width::W32);
    b.def_low(t, &[t, y]);
    b.copy(z, x, Width::W64);
    b.def(u, &[z, x, t]);
    b.ret(u, 0);
    let c = same(&b.0, &toy(4, &[]), &[7 | 1 << 40]);
    assert_eq!(["x", "t", "y", "z"].map(|v| reg_of(&c, v)), [r(0), r(1), r(2), r(2)]);
    assert_eq!(copies(&c), [(r(2), r(0)), (r(2), r(0))]);
}

/// And one that a copy of all of it is put in after that holds all of it: the third copy of `x`, in the register of the
/// second, is left out.
#[test]
fn a_register_a_copy_of_all_of_a_value_is_put_in_holds_all_of_it() {
    let mut b = Builder::new();
    let x = b.param("x", 0);
    let (t, y, z, z2, u) = (b.var("t"), b.var("y"), b.var("z"), b.var("z2"), b.var("u"));
    b.def(t, &[]);
    b.copy(y, x, Width::W32);
    b.def_low(t, &[t, y]);
    b.copy(z, x, Width::W64);
    b.def(t, &[t, z]);
    b.copy(z2, x, Width::W64);
    b.def(u, &[z2, x, t]);
    b.ret(u, 0);
    let c = same(&b.0, &toy(4, &[]), &[7 | 1 << 40]);
    assert_eq!(["x", "t", "y", "z", "z2"].map(|v| reg_of(&c, v)), [r(0), r(1), r(2), r(2), r(2)]);
    assert_eq!(copies(&c), [(r(2), r(0)), (r(2), r(0))]);
}

/// A copy of all of a value from a register that holds only part of it is a copy of that part: `given` is 32 bits that
/// a call gives back, and `y` copies them, so that `z`, in the register of `given`, is not `y` whole.
#[test]
fn a_copy_of_all_of_a_value_from_a_register_that_holds_part_of_it_stays() {
    let mut b = Builder::new();
    let (t, y, z, u, given) = (b.var("t"), b.var("y"), b.var("z"), b.var("u"), b.var("given"));
    b.def(t, &[]);
    b.0.pinned.push((given, r(0)));
    let implicit = vec![reg_arg(given, Width::W32, Access::Write)];
    b.0.body.push(Item::Inst(Inst { op: "call", args: vec![], implicit, flow: Flow::Next, clobbers: vec![] }));
    b.copy(y, given, Width::W32);
    b.def_low(t, &[t, given]);
    b.copy(z, y, Width::W64);
    b.def(u, &[z, t]);
    b.ret(u, 0);
    let c = same(&b.0, &toy(3, &[]), &[]);
    assert_eq!(["given", "t", "y", "z"].map(|v| reg_of(&c, v)), [r(0), r(1), r(2), r(0)]);
    assert_eq!(copies(&c), [(r(2), r(0)), (r(0), r(2))]);
}

/// A copy to the register it reads is left out, though the register holds part of the value only: `y` takes 32 bits of
/// what a call gives back, `z` all of `y`, in the same register, since `given` is read after and holds r0.
#[test]
fn a_copy_to_the_register_it_reads_is_left_out_whatever_the_register_holds() {
    let mut b = Builder::new();
    let (t, y, z, u, given) = (b.var("t"), b.var("y"), b.var("z"), b.var("u"), b.var("given"));
    b.def(t, &[]);
    b.0.pinned.push((given, r(0)));
    let implicit = vec![reg_arg(given, Width::W32, Access::Write)];
    b.0.body.push(Item::Inst(Inst { op: "call", args: vec![], implicit, flow: Flow::Next, clobbers: vec![] }));
    b.copy(y, given, Width::W32);
    b.copy(z, y, Width::W64);
    b.def_low(t, &[t, given]);
    b.def(u, &[z, t]);
    b.ret(u, 0);
    let c = same(&b.0, &toy(3, &[]), &[]);
    assert_eq!(["given", "t", "y", "z"].map(|v| reg_of(&c, v)), [r(0), r(1), r(2), r(2)]);
    assert_eq!(copies(&c), [(r(2), r(0))]);
}

/// A register that a copy is put in holds no more of a value than the register it reads: `y` takes the 32 bits that a
/// call gives back in `given`, `z` all of `y`, and `w` all of `z`. `z` holds those 32 bits alone, so the copy to `w`
/// reads `z`, and not the register of `given`, which holds all that the call wrote.
#[test]
fn a_register_a_copy_is_put_in_holds_no_more_of_a_value_than_the_register_it_reads() {
    let mut b = Builder::new();
    let (t, y, z, w, u, given) = (b.var("t"), b.var("y"), b.var("z"), b.var("w"), b.var("u"), b.var("given"));
    b.def(t, &[]);
    b.0.pinned.push((given, r(0)));
    let implicit = vec![reg_arg(given, Width::W32, Access::Write)];
    b.0.body.push(Item::Inst(Inst { op: "call", args: vec![], implicit, flow: Flow::Next, clobbers: vec![] }));
    b.copy(y, given, Width::W32);
    b.copy(z, y, Width::W64);
    b.copy(w, z, Width::W64);
    b.def_low(t, &[t, given]);
    b.def(u, &[w, z, y, t]);
    b.ret(u, 0);
    let c = same(&b.0, &toy(6, &[]), &[]);
    assert_eq!(["given", "t", "y", "z", "w"].map(|v| reg_of(&c, v)), [r(0), r(1), r(2), r(3), r(4)]);
    assert_eq!(copies(&c), [(r(2), r(0)), (r(3), r(2)), (r(4), r(3))]);
}

/// A register that a copy finds a value in, where nothing known is, holds it by the variable the copy reads: `x` is
/// written before a jump and again before the label it goes to, so that its register holds nothing known at the label.
/// There `y` takes 32 bits of `x`, `z` all of it, and `w` all of `z`, and the copy to `w` reads the register of `x`,
/// which held the value first, and not that of `y`, which holds the low half alone.
#[test]
fn a_register_a_copy_finds_a_value_in_holds_it_by_the_variable_the_copy_reads() {
    let mut b = Builder::new();
    let n = b.param("n", 0);
    let (x, y, z, w, u) = (b.var("x"), b.var("y"), b.var("z"), b.var("w"), b.var("u"));
    b.def(x, &[]);
    b.inst("dec", &[(n, Access::ReadWrite)], Flow::Branch(Label(1)));
    b.def(x, &[x]);
    b.label(1);
    b.copy(y, x, Width::W32);
    b.copy(z, x, Width::W64);
    b.copy(w, z, Width::W64);
    b.def(u, &[w, z, y, x]);
    b.ret(u, 0);
    for n in [1, 2] {
        let c = same(&b.0, &toy(6, &[]), &[n]);
        assert_eq!(["x", "y", "z", "w"].map(|v| reg_of(&c, v)), [r(1), r(0), r(2), r(3)]);
        assert_eq!(copies(&c), [(r(0), r(1)), (r(2), r(1)), (r(3), r(1))]);
    }
}

/// An instruction that writes two variables makes two values: a copy of one to the other stays.
#[test]
fn what_an_instruction_writes_to_two_variables_is_two_values() {
    let mut b = Builder::new();
    let (s, one, two, u) = (b.param("s", 0), b.var("one"), b.var("two"), b.var("u"));
    b.inst("op", &[(one, Access::Write), (two, Access::Write), (s, Access::Read)], Flow::Next);
    b.copy(one, two, Width::W64);
    b.def(u, &[one, two]);
    b.ret(u, 0);
    let c = same(&b.0, &toy(4, &[]), &[7]);
    assert_eq!(["one", "two"].map(|v| reg_of(&c, v)), [r(0), r(1)]);
    assert_eq!(copies(&c), [(r(0), r(1))]);
}

/// A call that clears the upper half of r0, and leaves the lower, changes what r0 holds: `y` holds `p` whole, and `z`,
/// in r0, is not `p` whole after the call, so that the copy of `y` to it stays.
#[test]
fn a_register_holds_nothing_known_after_a_call_that_clears_its_upper_half() {
    let mut b = Builder::new();
    let p = b.param("p", 0);
    let (y, z) = (b.var("y"), b.var("z"));
    b.copy(y, p, Width::W64);
    b.call_with(None, &[(p, 0)], upper(&[0]));
    b.copy(z, y, Width::W64);
    b.ret(z, 0);
    let c = same(&b.0, &toy(3, &[]), &[7 | 1 << 40]);
    assert_eq!(["p", "y", "z"].map(|v| reg_of(&c, v)), [r(0), r(1), r(0)]);
    assert_eq!(copies(&c), [(r(1), r(0)), (r(0), r(1))]);
}

/// A copy that finds a register nothing known is in finds all of its value, and the register it copies to holds the
/// same: `x` is written before a loop and written again at its end, so that no register is known to hold it at the top,
/// where `y` takes 32 bits of it, `z` all of it, and `z2` all of it again, which is left out.
#[test]
fn a_copy_finds_what_a_register_holds_where_ways_to_a_label_leave_it_unknown() {
    let mut b = Builder::new();
    let n = b.param("n", 0);
    let (x, t, y, z, z2) = (b.var("x"), b.var("t"), b.var("y"), b.var("z"), b.var("z2"));
    b.def(x, &[]);
    b.def(t, &[]);
    b.label(1);
    b.copy(y, x, Width::W32);
    b.def_low(t, &[t, y]);
    b.copy(z, x, Width::W64);
    b.def(t, &[t, z]);
    b.copy(z2, x, Width::W64);
    b.def(t, &[t, z2]);
    b.def(x, &[x, t]);
    b.inst("dec", &[(n, Access::ReadWrite)], Flow::Branch(Label(1)));
    b.ret(t, 0);
    let c = same(&b.0, &toy(5, &[]), &[3]);
    assert_eq!(["x", "t", "y", "z", "z2"].map(|v| reg_of(&c, v)), [r(1), r(2), r(3), r(3), r(3)]);
    assert_eq!(copies(&c), [(r(3), r(1)), (r(3), r(1)), (r(0), r(2))]);
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
    // The variable the call copies `x` to, which it pins.
    let pinned = VReg { id: 2, class: Class::Gpr };
    assert_eq!(
        compile(&b.0, &toy(3, &[])).unwrap_err(),
        Error::FixedClash { a: "x".into(), vreg: pinned, b: "y".into(), reg: r(0) }
    );
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
            vreg: x,
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
    let (y, a) = (b.var("y"), b.var("a"));
    b.0.pinned.push((a, r(0)));
    b.def(a, &[]);
    b.call(None, &[], &[0, 1]);
    b.def(y, &[a]);
    b.ret(y, 0);
    let err = compile(&b.0, &toy(3, &[])).unwrap_err();
    assert_eq!(err, Error::FixedChanged { var: "a".into(), vreg: a, reg: r(0) });
}

#[test]
fn an_error_names_the_variable_to_blame() {
    let x = VReg { id: 7, class: Class::Gpr };
    let out = Error::OutOfRegisters {
        class: Class::Gpr,
        var: "x".into(),
        vreg: x,
        near: None,
        taken: vec![("y".into(), r(1))],
        changed: vec![r(0)],
        have: 2,
    };
    let cases = [
        (Error::ReadBeforeWrite { var: "x".into(), vreg: x }, Some(x)),
        (Error::FixedTwice { var: "x".into(), vreg: x }, Some(x)),
        (Error::FixedChanged { var: "x".into(), vreg: x, reg: r(0) }, Some(x)),
        (out, Some(x)),
        // Of two that clash, the first.
        (Error::FixedClash { a: "x".into(), vreg: x, b: "y".into(), reg: r(0) }, Some(x)),
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
    let (y, a) = (b.var("y"), b.var("a"));
    b.0.pinned.push((a, r(0)));
    b.0.pinned.push((a, r(1)));
    b.def(a, &[]);
    b.def(y, &[a]);
    b.ret(y, 0);
    let err = compile(&b.0, &toy(3, &[])).unwrap_err();
    assert_eq!(err, Error::FixedTwice { var: "a".into(), vreg: a });
    assert_eq!(err.to_string(), "`a` must be in two different registers");
}

#[test]
fn each_error_says_what_is_wrong_in_words() {
    let v = VReg { id: 0, class: Class::Gpr };
    let out = |near: Option<&str>, taken: Vec<(&str, PReg)>, changed: Vec<PReg>| Error::OutOfRegisters {
        class: Class::Gpr,
        var: "x".into(),
        vreg: v,
        near: near.map(String::from),
        taken: taken.into_iter().map(|(name, reg)| (name.to_string(), reg)).collect(),
        changed,
        have: 2,
    };
    let cases = [
        (Error::LabelTwice { label: 7 }, "label 7 is placed twice"),
        (Error::NoSuchLabel { label: 7 }, "a jump goes to label 7, which is never placed"),
        (Error::FallsOffEnd, "control runs past the last instruction"),
        (Error::ReadBeforeWrite { var: "x".into(), vreg: v }, "`x` is read before anything writes it"),
        (Error::FixedTwice { var: "a".into(), vreg: v }, "`a` must be in two different registers"),
        (
            Error::FixedClash { a: "x".into(), vreg: v, b: "y".into(), reg: r(0) },
            "`x` and `y` are live at once and both must be in PReg { class: Gpr, hw: 0 }",
        ),
        (
            Error::FixedChanged { var: "a".into(), vreg: v, reg: r(0) },
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

/// Which items of the body every path from the start to the end runs, `x` for each that does
/// and `.` for each that does not; with `leaves`, where a thread may end or never come back.
fn paths(b: &Builder, leaves: &[usize]) -> String {
    let on = crate::on_every_path(&b.0, leaves).unwrap();
    on.iter().map(|&on| if on { 'x' } else { '.' }).collect()
}

#[test]
fn an_item_that_a_jump_can_skip_is_not_on_every_path() {
    let mut b = Builder::new();
    let (p, x, y) = (b.var("p"), b.var("x"), b.var("y"));
    b.def(p, &[]);
    b.inst("br", &[(p, Access::Read)], Flow::Branch(Label(1)));
    b.def(x, &[]);
    b.label(1);
    b.def(y, &[]);
    b.inst("ret", &[], Flow::Return);
    assert_eq!(paths(&b, &[]), "xx.xxx");

    // Past a jump that always skips it, an item is on no path at all.
    let mut b = Builder::new();
    let x = b.var("x");
    b.inst("jmp", &[], Flow::Jump(Label(1)));
    b.def(x, &[]);
    b.label(1);
    b.inst("ret", &[], Flow::Return);
    assert_eq!(paths(&b, &[]), "x.xx");
}

#[test]
fn a_jump_through_a_table_is_a_branch_to_each_of_its_labels() {
    let mut b = Builder::new();
    let (n, x, y) = (b.param("n", 0), b.var("x"), b.var("y"));
    b.inst("jmp", &[(n, Access::Read)], Flow::Table { to: vec![Label(1), Label(2)], or_next: false });
    b.label(1);
    b.def(x, &[]);
    b.label(2);
    b.def(y, &[]);
    b.inst("ret", &[], Flow::Return);
    assert_eq!(paths(&b, &[]), "x..xxx");
}

#[test]
fn what_follows_a_leave_is_not_on_every_path() {
    let mut b = Builder::new();
    let (p, x) = (b.var("p"), b.var("x"));
    b.def(p, &[]);
    b.inst("ret_if", &[(p, Access::Read)], Flow::Next);
    b.def(x, &[]);
    b.inst("ret", &[], Flow::Return);
    assert_eq!(paths(&b, &[]), "xxxx");
    assert_eq!(paths(&b, &[1]), "xx..");

    // Nor what follows it past a label: a leave is a way to the end from its block.
    let mut b = Builder::new();
    let (p, x) = (b.var("p"), b.var("x"));
    b.def(p, &[]);
    b.inst("ret_if", &[(p, Access::Read)], Flow::Next);
    b.label(1);
    b.def(x, &[]);
    b.inst("ret", &[], Flow::Return);
    assert_eq!(paths(&b, &[1]), "xx...");

    // The first of two leaves in a block is where the rest stops.
    let mut b = Builder::new();
    let (p, x, y) = (b.var("p"), b.var("x"), b.var("y"));
    b.def(p, &[]);
    b.inst("ret_if", &[(p, Access::Read)], Flow::Next);
    b.def(x, &[]);
    b.inst("ret_if", &[(p, Access::Read)], Flow::Next);
    b.def(y, &[]);
    b.inst("ret", &[], Flow::Return);
    assert_eq!(paths(&b, &[1, 3]), "xx....");
}

#[test]
fn a_loop_with_a_way_out_is_run_by_the_threads_that_leave_it() {
    let mut b = Builder::new();
    let (p, x) = (b.var("p"), b.var("x"));
    b.label(1);
    b.def(p, &[]);
    b.inst("br", &[(p, Access::Read)], Flow::Branch(Label(1)));
    b.def(x, &[]);
    b.inst("ret", &[], Flow::Return);
    assert_eq!(paths(&b, &[]), "xxxxx");
}

#[test]
fn a_loop_with_no_way_out_may_end_where_it_jumps_back() {
    // What each round runs is on every path; what a round may skip is not.
    let mut b = Builder::new();
    let (p, x, y) = (b.var("p"), b.var("x"), b.var("y"));
    b.def(p, &[]);
    b.label(1);
    b.def(x, &[]);
    b.inst("br", &[(p, Access::Read)], Flow::Branch(Label(2)));
    b.def(y, &[]);
    b.label(2);
    b.inst("jmp", &[], Flow::Jump(Label(1)));
    assert_eq!(paths(&b, &[]), "xxxx.xx");

    // Nor is what a thread that goes into the loop never runs, though the loop never ends.
    let mut b = Builder::new();
    let (p, x) = (b.var("p"), b.var("x"));
    b.def(p, &[]);
    b.inst("br", &[(p, Access::Read)], Flow::Branch(Label(1)));
    b.def(x, &[]);
    b.inst("ret", &[], Flow::Return);
    b.label(1);
    b.inst("jmp", &[], Flow::Jump(Label(1)));
    assert_eq!(paths(&b, &[]), "xx....");
}

#[test]
fn control_must_stay_inside_the_body_for_the_paths_too() {
    let mut b = Builder::new();
    let x = b.var("x");
    b.def(x, &[]);
    assert_eq!(crate::on_every_path(&b.0, &[]).unwrap_err(), Error::FallsOffEnd);
    b.inst("jmp", &[], Flow::Jump(Label(7)));
    assert_eq!(crate::on_every_path(&b.0, &[]).unwrap_err(), Error::NoSuchLabel { label: 7 });
}

/// A number for each variable of `b`, followed over its flow by [`crate::forward`]: all 5 at the start, `narrow` sets
/// the first operand's to 1 and `wide` to 5, and `use` fails if it is under 3. The first failure is the item's index
/// and the variable.
fn follow(b: &Builder) -> Result<Option<(usize, VReg)>, Error> {
    let meet = |a: &Vec<u8>, b: &Vec<u8>| a.iter().zip(b).map(|(a, b)| *a.min(b)).collect();
    let said = crate::forward(&b.0, vec![5u8; b.0.regs.len()], meet, |_, inst, now: &mut Vec<u8>| {
        let Some(Operand::Reg(v, _)) = inst.args.first().map(|a| a.operand) else { return Ok(()) };
        match inst.op {
            "narrow" => now[v.id as usize] = 1,
            "wide" => now[v.id as usize] = 5,
            "use" if now[v.id as usize] < 3 => return Err(v),
            _ => {}
        }
        Ok(())
    })?;
    Ok(said.into_iter().enumerate().find_map(|(i, said)| Some((i, said?.err()?))))
}

#[test]
fn a_value_at_a_label_is_the_least_that_any_way_to_it_brings() {
    // One way to the label narrows `x`, and the other does not.
    let mut b = Builder::new();
    let (p, x) = (b.var("p"), b.var("x"));
    b.inst("br", &[(p, Access::Read)], Flow::Branch(Label(1)));
    b.inst("narrow", &[(x, Access::Write)], Flow::Next);
    b.label(1);
    b.inst("use", &[(x, Access::Read)], Flow::Next);
    b.inst("ret", &[], Flow::Return);
    assert_eq!(follow(&b), Ok(Some((3, x))));

    // Both ways write it wide, so that the one way that narrowed it brings nothing.
    let mut b = Builder::new();
    let (p, x) = (b.var("p"), b.var("x"));
    b.inst("narrow", &[(x, Access::Write)], Flow::Next);
    b.inst("br", &[(p, Access::Read)], Flow::Branch(Label(1)));
    b.inst("wide", &[(x, Access::Write)], Flow::Next);
    b.label(1);
    b.inst("use", &[(x, Access::Read)], Flow::Next);
    b.inst("ret", &[], Flow::Return);
    assert_eq!(follow(&b), Ok(Some((4, x))));
}

#[test]
fn a_jump_back_brings_a_value_to_its_label_too() {
    let mut b = Builder::new();
    let (p, x) = (b.var("p"), b.var("x"));
    b.label(1);
    b.inst("use", &[(x, Access::Read)], Flow::Next);
    b.inst("narrow", &[(x, Access::Write)], Flow::Next);
    b.inst("br", &[(p, Access::Read)], Flow::Branch(Label(1)));
    b.inst("ret", &[], Flow::Return);
    assert_eq!(follow(&b), Ok(Some((1, x))));

    // Written wide again before the jump, `x` comes back as it went.
    let mut b = Builder::new();
    let (p, x) = (b.var("p"), b.var("x"));
    b.label(1);
    b.inst("use", &[(x, Access::Read)], Flow::Next);
    b.inst("narrow", &[(x, Access::Write)], Flow::Next);
    b.inst("wide", &[(x, Access::Write)], Flow::Next);
    b.inst("br", &[(p, Access::Read)], Flow::Branch(Label(1)));
    b.inst("ret", &[], Flow::Return);
    assert_eq!(follow(&b), Ok(None));

    // And on past the label, to a read after a branch, which a pass over the blocks in order reaches before the jump
    // back.
    let mut b = Builder::new();
    let (p, x) = (b.var("p"), b.var("x"));
    b.label(1);
    b.inst("br", &[(p, Access::Read)], Flow::Branch(Label(2)));
    b.inst("use", &[(x, Access::Read)], Flow::Next);
    b.inst("narrow", &[(x, Access::Write)], Flow::Next);
    b.inst("jmp", &[], Flow::Jump(Label(1)));
    b.label(2);
    b.inst("ret", &[], Flow::Return);
    assert_eq!(follow(&b), Ok(Some((2, x))));
}

#[test]
fn a_loop_entered_at_its_test_has_the_value_its_one_way_in_brings() {
    // The only way to the top is the jump back from the test, which is entered with `x` written wide.
    let mut b = Builder::new();
    let (p, x) = (b.var("p"), b.var("x"));
    b.inst("narrow", &[(x, Access::Write)], Flow::Next);
    b.inst("wide", &[(x, Access::Write)], Flow::Next);
    b.inst("jmp", &[], Flow::Jump(Label(2)));
    b.label(1);
    b.inst("use", &[(x, Access::Read)], Flow::Next);
    b.label(2);
    b.inst("br", &[(p, Access::Read)], Flow::Branch(Label(1)));
    b.inst("ret", &[], Flow::Return);
    assert_eq!(follow(&b), Ok(None));
}

#[test]
fn an_item_that_no_way_comes_to_is_not_stepped() {
    // Past a jump that always skips it.
    let mut b = Builder::new();
    let x = b.var("x");
    b.inst("narrow", &[(x, Access::Write)], Flow::Next);
    b.inst("jmp", &[], Flow::Jump(Label(1)));
    b.inst("use", &[(x, Access::Read)], Flow::Next);
    b.label(1);
    b.inst("ret", &[], Flow::Return);
    assert_eq!(follow(&b), Ok(None));

    // A loop that no way comes to, after a return.
    let mut b = Builder::new();
    let x = b.var("x");
    b.inst("narrow", &[(x, Access::Write)], Flow::Next);
    b.inst("ret", &[], Flow::Return);
    b.label(1);
    b.inst("use", &[(x, Access::Read)], Flow::Next);
    b.inst("jmp", &[], Flow::Jump(Label(1)));
    assert_eq!(follow(&b), Ok(None));
}

#[test]
fn the_first_failure_of_the_body_is_the_one_given() {
    let mut b = Builder::new();
    let (x, y) = (b.var("x"), b.var("y"));
    b.inst("narrow", &[(y, Access::Write)], Flow::Next);
    b.inst("narrow", &[(x, Access::Write)], Flow::Next);
    b.inst("use", &[(x, Access::Read)], Flow::Next);
    b.inst("use", &[(y, Access::Read)], Flow::Next);
    b.inst("ret", &[], Flow::Return);
    assert_eq!(follow(&b), Ok(Some((2, x))));
}

#[test]
fn control_must_stay_inside_the_body_for_the_values_too() {
    let mut b = Builder::new();
    let x = b.var("x");
    b.inst("narrow", &[(x, Access::Write)], Flow::Next);
    assert_eq!(follow(&b), Err(Error::FallsOffEnd));
    b.inst("jmp", &[], Flow::Jump(Label(7)));
    assert_eq!(follow(&b), Err(Error::NoSuchLabel { label: 7 }));
}
