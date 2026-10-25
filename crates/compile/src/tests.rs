//! The use cases on a toy machine: a few general registers, some callee-saved, and opcodes
//! that are only names.

use kgen_core::{
    Access, Arg, Class, Flow, Inst, Item, Label, Machine, Operand, PReg, Program, RegInfo, Signature, Slot, VReg, Width,
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

fn r(hw: u8) -> PReg {
    PReg { class: Class::Gpr, hw }
}

/// `n` registers, the ones in `saved` callee-saved.
fn toy(n: u8, saved: &[u8]) -> Toy {
    Toy { gpr: (0..n).map(r).collect(), saved: saved.iter().map(|&h| r(h)).collect() }
}

struct Builder(Program<&'static str>);

impl Builder {
    fn new() -> Self {
        Builder(Program {
            name: "toy".into(),
            signature: Signature { params: vec![], result: None },
            regs: vec![],
            params: vec![],
            results: vec![],
            body: vec![],
            slots: vec![],
            consts: vec![],
        })
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

    /// `dst = op(srcs)`.
    fn def(&mut self, dst: VReg, srcs: &[VReg]) {
        let mut args = vec![(dst, Access::Write)];
        args.extend(srcs.iter().map(|&s| (s, Access::Read)));
        self.inst("op", &args, Flow::Next);
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
fn two_live_values_need_two_registers() {
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
    // values are live at once and two registers are not enough.
    assert!(matches!(compile(&b.0, &toy(2, &[])), Err(Error::FixedClash { .. } | Error::OutOfRegisters { .. })));
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
    b.0.slots = vec![Slot { size: 4, align: 4 }, Slot { size: 32, align: 32 }, Slot { size: 8, align: 8 }];
    let frame = compile(&b.0, &toy(1, &[])).unwrap().frame;
    assert_eq!((frame.offsets, frame.size, frame.align), (vec![0, 32, 64], 96, 32));
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
