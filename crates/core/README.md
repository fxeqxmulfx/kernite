# kernite-core

The program kernite works on: instructions over variables, before any register is chosen, each operand marked read or
written. It knows no instruction set, as the opcode is a type of yours: a crate that writes a machine for
kernite-compile adds it. A crate of kernels does not add it: it adds `kernite-macros`.

```sh
cargo add kernite-core
```

```rust
use kernite_core::{Access, Arg, Class, Flow, Inst, Item, Operand, PReg, Program, RegInfo, Signature, VReg, Width};

/// The opcodes of the example's own instruction set.
enum Op {
    Add,
    Ret,
}

fn main() {
    // `fn sum(a, b) -> a + b`: `a` and `b` arrive in `rdi` and `rsi`, and the sum leaves in `rax`.
    let gpr = |id| VReg { id, class: Class::Gpr };
    let (a, b) = (gpr(0), gpr(1));
    let reg = |hw| PReg { class: Class::Gpr, hw };
    let named = |name: &str| RegInfo { class: Class::Gpr, name: name.into() };
    let operand = |v, access| Arg { operand: Operand::Reg(v, Width::W64), access };
    let inst = |op, args, implicit, flow| Item::Inst(Inst { op, args, implicit, flow, clobbers: vec![] });
    let program = Program {
        name: "sum".into(),
        signature: Signature { params: vec![Class::Gpr; 2], result: Some(Class::Gpr) },
        regs: vec![named("a"), named("b")],
        params: vec![(a, reg(7)), (b, reg(6))],
        results: vec![(a, reg(0))],
        pinned: vec![],
        body: vec![
            // `add a, b` reads both and writes `a`.
            inst(Op::Add, vec![operand(a, Access::ReadWrite), operand(b, Access::Read)], vec![], Flow::Next),
            // `ret` reads the result, an operand that it does not name.
            inst(Op::Ret, vec![], vec![operand(a, Access::Read)], Flow::Return),
        ],
        slots: vec![],
        consts: vec![],
        funcs: vec![],
        statics: vec![],
    };
    let [Item::Inst(add), Item::Inst(ret)] = &program.body[..] else { unreachable!() };
    assert_eq!(add.reads().collect::<Vec<_>>(), [a, b]);
    assert_eq!(add.writes().collect::<Vec<_>>(), [a]);
    assert_eq!(ret.reads().collect::<Vec<_>>(), [a]);
}
```

The language, the targets and the guarantees are in
[GUIDE.md](https://github.com/fxeqxmulfx/kernite/blob/master/GUIDE.md), and the API is in
[the docs](https://docs.rs/kernite-core).
