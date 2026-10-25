use crate::{Fault, Guard, ItemKind, Operand, Target, kernel, parse};

/// A target that writes down what it is asked, one line each, and refuses `bad`.
#[derive(Default)]
struct Mock {
    names: Vec<String>,
    lines: Vec<String>,
}

/// A variable by number, whole or seen narrow, as `v.xmm`.
type Var = (usize, bool);

impl Mock {
    fn new_var(&mut self, name: &str) -> Var {
        self.names.push(name.to_string());
        (self.names.len() - 1, false)
    }

    fn show(&self, op: &Operand<Var>) -> String {
        let var = |&(id, narrow): &Var| format!("{}{}", self.names[id], if narrow { ".xmm" } else { "" });
        match op {
            Operand::Var(v) => var(v),
            Operand::Special(name) => format!("%{name}"),
            Operand::Int(x) => x.to_string(),
            Operand::F32(x) => format!("{x}f"),
            Operand::F64(x) => format!("{x}d"),
            Operand::Label(l) => format!("L{l}"),
            Operand::Data(bytes) => format!("data({})", bytes.iter().map(|b| format!("{b:02x}")).collect::<String>()),
            Operand::Mem(m) => {
                let mut parts = vec![];
                parts.extend(m.base.as_ref().map(var));
                parts.extend(m.index.as_ref().map(|(v, k)| format!("{}*{k}", var(v))));
                if m.disp != 0 || parts.is_empty() {
                    parts.push(m.disp.to_string());
                }
                format!("[{}]", parts.join(" + "))
            }
        }
    }
}

impl Target for Mock {
    type Var = Var;

    fn param(&mut self, name: &str, ty: &str) -> Result<Var, String> {
        self.lines.push(format!("param {name}: {ty}"));
        Ok(self.new_var(name))
    }

    fn result(&mut self, ty: &str) -> Result<(), String> {
        self.lines.push(format!("result {ty}"));
        Ok(())
    }

    fn var(&mut self, name: &str, ty: &str) -> Result<Var, String> {
        if ty == "nope" {
            return Err("no such type".into());
        }
        self.lines.push(format!("var {name}: {ty}"));
        Ok(self.new_var(name))
    }

    fn view(&mut self, (id, _): Var, view: &str) -> Result<Var, String> {
        match view {
            "xmm" => Ok((id, true)),
            "ymm" | "gpr" | "pred" => Ok((id, false)),
            _ => Err(format!("no view `{view}`")),
        }
    }

    fn label(&mut self, label: u32) {
        self.lines.push(format!("L{label}:"));
    }

    fn note(&mut self, text: &str) {
        self.lines.push(format!("# {text}"));
    }

    fn inst(&mut self, name: &str, args: &[Operand<Var>], guard: Option<Guard<Var>>) -> Result<(), Fault> {
        if name == "bad" {
            return Err(Fault::at(1, "the second is wrong"));
        }
        let guard =
            guard.map_or(String::new(), |g| format!("@{}{} ", if g.negated { "!" } else { "" }, self.names[g.pred.0]));
        let args: Vec<String> = args.iter().map(|a| self.show(a)).collect();
        self.lines.push(format!("{guard}{name} {}", args.join(", ")).trim_end().to_string());
        Ok(())
    }

    fn end(&mut self) -> Result<(), String> {
        self.lines.push("end".into());
        Ok(())
    }
}

/// What kernel `name` in `src` asks of the target, or the error with its line and column.
fn run(src: &str, name: &str) -> Result<Vec<String>, String> {
    let module = parse(src).map_err(|e| e.at(src))?;
    let f = module
        .items
        .iter()
        .find_map(|item| match &item.kind {
            ItemKind::Fn(f) if f.name == name => Some(f),
            _ => None,
        })
        .expect("the kernel");
    let mut mock = Mock::default();
    kernel(&module, f, &mut mock).map_err(|e| e.at(src))?;
    Ok(mock.lines)
}

fn lines(text: &str) -> Vec<String> {
    text.lines().map(str::trim).filter(|l| !l.is_empty()).map(String::from).collect()
}

#[test]
fn a_kernel_runs_its_helpers_in_place() {
    let src = r#"
        /// The sum of eight floats.
        pub unsafe extern "sysv64" fn sum8(xs: *const f32) -> f32 {
            let v: ymm;
            vmovups(v, [xs]);
            ret_sum(v);
        }

        fn ret_sum(v: ymm) {
            note("the lanes");
            let high: ymm;
            vextractf128(high.xmm, v, 1);
            ret(high.xmm);
        }
    "#;
    let want = lines(
        "
        param xs: *const f32
        result f32
        var v: ymm
        vmovups v, [xs]
        # the lanes
        var high: ymm
        vextractf128 high.xmm, v, 1
        ret high.xmm
        end
        ",
    );
    assert_eq!(run(src, "sum8").unwrap(), want);
}

#[test]
fn loops_unroll_and_number_their_labels_afresh() {
    let src = r#"
        const STEP: usize = 32;

        pub extern "sysv64" fn k(x: *const f32, n: usize) {
            dot::<2>(x, n);
        }

        fn dot<const U: usize>(x: gpr, n: gpr) {
            let acc: [ymm; U];
            for k in 0..U {
                zero(acc[k]);
            }
            label(top);
            for k in 0..U {
                vfmadd231ps(acc[k], acc[k], [x + STEP * k]);
            }
            add(x, STEP * U);
            sub(n, 8 * U);
            jnz(top);
            for k in (0..4).step_by(2).rev() {
                jmp(skip);
                label(skip);
                note("k = {k}, {{in braces}}");
            }
            for a in acc {
                vaddps(a, a, acc[0]);
            }
        }
    "#;
    let want = lines(
        "
        param x: *const f32
        param n: usize
        var acc0: ymm
        var acc1: ymm
        zero acc0
        zero acc1
        L0:
        vfmadd231ps acc0, acc0, [x]
        vfmadd231ps acc1, acc1, [x + 32]
        add x, 64
        sub n, 16
        jnz L0
        jmp L1
        L1:
        # k = 2, {in braces}
        jmp L2
        L2:
        # k = 0, {in braces}
        vaddps acc0, acc0, acc0
        vaddps acc1, acc1, acc0
        end
        ",
    );
    assert_eq!(run(src, "k").unwrap(), want);
}

#[test]
fn brackets_are_memory_at_an_address_or_holding_data() {
    let src = r#"
        pub extern "C" fn m(x: *const f32, i: usize) {
            let v: ymm;
            vmovups(v, [x + i * 4 + 16]);
            vmovups(v, [i * 8 - 4 + x]);
            vmovups(v, [x + (i + 2) * 4]);
            vmovups(v, [x - 8]);
            vmovups(v, [x + i]);
            vmovups(v, [i * 4]);
            vbroadcastss(v, [1.0]);
            vmovups(v, [[0.5; 2]]);
            vmovups(v, [[1u8, 2]]);
            vmovups(v, [[-1, 2]]);
        }
    "#;
    let want = lines(
        "
        param x: *const f32
        param i: usize
        var v: ymm
        vmovups v, [x + i*4 + 16]
        vmovups v, [x + i*8 + -4]
        vmovups v, [x + i*4 + 8]
        vmovups v, [x + -8]
        vmovups v, [x + i*1]
        vmovups v, [i*4]
        vbroadcastss v, data(0000803f)
        vmovups v, data(0000003f0000003f)
        vmovups v, data(0102)
        vmovups v, data(ffffffff02000000)
        end
        ",
    );
    assert_eq!(run(src, "m").unwrap(), want);
}

#[test]
fn a_predicate_guards_what_its_if_holds() {
    let src = r#"
        pub extern "ptx-kernel" fn g(n: u32) {
            let p: pred;
            setp.ge.u32(p, tid.x, n);
            if p {
                ret();
            }
            if !p {
                add.u32(n, n, 1);
            } else {
                sub.u32(n, n, 1.5);
            }
            if 1 + 1 == 2 {
                note("now");
            } else {
                note("never");
            }
        }
    "#;
    let want = lines(
        "
        param n: u32
        var p: pred
        setp.ge.u32 p, %tid.x, n
        @p ret
        @!p add.u32 n, n, 1
        @p sub.u32 n, n, 1.5f
        # now
        end
        ",
    );
    assert_eq!(run(src, "g").unwrap(), want);
}

#[test]
fn values_are_computed_as_rust_computes_them() {
    let src = r#"
        const BITS: u32 = 1 << 31;
        const HALF: f64 = 1.0 / 2.0;
        const TABLE: [i32; 3] = [1, -2, 3];
        const SIGNED: i32 = BITS as i32;
        const TRUNC: u8 = 300 as u8;
        const ROUND: i32 = -2.7 as i32;
        const NEG_INF: f32 = f32::NEG_INFINITY;
        const LAST: i32 = TABLE[2];
        const BITS_OF: u32 = 1.0f32.to_bits();
        const ONE: f32 = f32::from_bits(0x3f80_0000);

        pub extern "C" fn v() {
            let k: usize = 3;
            note("{BITS} {HALF} {SIGNED} {TRUNC} {ROUND} {NEG_INF} {LAST} {BITS_OF} {ONE} {k} {TABLE}");
            mov(k, HALF, -LAST, 0xff_u8);
        }
    "#;
    let want = lines(
        "
        # 2147483648 0.5 -2147483648 44 -2 -inf 3 1065353216 1 3 [1, -2, 3]
        mov 3, 0.5d, -3, 255
        end
        ",
    );
    assert_eq!(run(src, "v").unwrap(), want);
}

#[test]
fn errors_point_where_the_trouble_is() {
    let cases: &[(&str, &str)] = &[
        ("let k = nope;", "1:9: `nope` is not defined"),
        ("let x;", "1:5: `let x;` declares nothing: give it a type, as in `let x: ymm;`, or a value"),
        ("let x: nope;", "1:8: no such type"),
        ("bad(1, 2);", "1:8: the second is wrong"),
        (
            "let p: pred; if p { if p { ret(); } }",
            "1:21: a predicate's `if` inside another's: one predicate guards an instruction",
        ),
        (
            "let p: pred; if p { label(x); }",
            "1:21: a label takes no predicate: control arrives there whatever it holds",
        ),
        ("label(a); label(a);", "1:17: label `a` is placed twice in this block"),
        ("two(1);", "1:1: `two` takes 2 arguments; this gives 1"),
        ("deep();", "5:13: helpers call helpers 64 deep here; does `deep` call itself?"),
        ("const X: u8 = 255 + 1;", "1:19: 256 is out of range for u8"),
        ("let k = A;", "7:18: `A` is defined in terms of itself"),
        ("let a: [ymm; 2]; vaddps(a, a, a);", "1:25: `a` is 2 variables; name one, as in `a[0]`"),
        ("let a: [ymm; 2]; zero(a[2]);", "1:24: `a` has 2 variables; there is no [2]"),
        ("let v: ymm; vmovups(v, [v * v]);", "1:27: an address multiplies a variable by a number, not by a variable"),
        (
            "let v: ymm; let k = v + 1;",
            "1:21: `v` is a variable, known only when the kernel runs; this wants a value known now",
        ),
        ("two::<1>(1, 2);", "1:1: `two` takes 0 values in `::<..>`; this gives 1"),
        ("note(\"{gone}\");", "1:6: `gone` is not defined"),
        ("loop {}", "1:1: a kernel has no `loop`: loops are labels and jumps, and `for` is unrolled"),
    ];
    for (body, want) in cases {
        let src = format!(
            "pub extern \"C\" fn k() {{\n{body}\n}}\nfn two(a: ymm, b: ymm) {{}}\nfn deep() {{ deep(); }}\n\
             const A: usize = B;\nconst B: usize = A;\n"
        );
        let got = run(&src, "k").unwrap_err();
        // The body is on the second line.
        let want = want.replacen("1:", "2:", 1);
        assert_eq!(got, want, "{body}");
    }
}

#[test]
fn a_helper_sees_none_of_its_callers_names() {
    let src = r#"
        pub extern "C" fn k() {
            let secret = 3;
            h();
        }
        fn h() {
            note("{secret}");
        }
    "#;
    assert_eq!(run(src, "k").unwrap_err(), "7:18: `secret` is not defined");
}

#[test]
fn array_parameters_take_variables_declared_together() {
    let src = r#"
        pub extern "C" fn k() {
            let acc: [ymm; 3];
            sum::<3>(acc);
        }
        fn sum<const N: usize>(acc: [xmm; N]) {
            for k in 1..N {
                vaddps(acc[0], acc[0], acc[k]);
            }
        }
    "#;
    let want = lines(
        "
        var acc0: ymm
        var acc1: ymm
        var acc2: ymm
        vaddps acc0.xmm, acc0.xmm, acc1.xmm
        vaddps acc0.xmm, acc0.xmm, acc2.xmm
        end
        ",
    );
    assert_eq!(run(src, "k").unwrap(), want);
}

#[test]
fn the_lexer_reads_what_rust_writes() {
    let src = "a.b(0..n, 1.5e-3, 0x1F_u8, r#type, \"s\\\"q\", 1..=2) /* c /* nested */ */ // end";
    let trees = crate::lex(src).unwrap();
    assert_eq!(crate::text(&trees), r#"a.b(0..n,1.5e-3,0x1F_u8,r#type,"s\"q",1..=2)"#);
}
