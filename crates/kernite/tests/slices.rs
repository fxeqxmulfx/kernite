//! A slice a kernel is given, `&[f32]` to read and `&mut [f32]` to write, is an address and a length, which the kernel
//! has in two registers, `xs` and `xs.len`. A kernel that is no `unsafe fn` reaches an element only where kernite
//! proves the address inside the slice from the compares the kernel makes of an index with the length. What it cannot
//! prove it refuses, at the operand, and the kernel is then an `unsafe fn`, which compiles as it always has.

use std::time::{Duration, Instant};

type Result<T = (), E = Box<dyn std::error::Error + Send + Sync>> = std::result::Result<T, E>;

/// Kernel `k` of `src` as x86, the lines of its code, or the error.
fn code(src: &str) -> Result<Vec<String>, String> {
    kernite::x86_str(src, "k").map(|listing| listing.code().map(str::to_string).collect())
}

/// The error kernel `k` of `src` gives.
fn error(src: &str) -> Result<String> {
    Ok(code(src).err().ok_or("code gave Ok")?)
}

/// `src` with `unsafe` before each `extern`: its kernels are `unsafe fn`s.
fn trusted(src: &str) -> String {
    src.replace("extern \"sysv64\"", "unsafe extern \"sysv64\"")
}

/// Where the first `needle` is in the first `line` of `src`, as an error says it, `line:column`.
fn in_line(src: &str, line: &str, needle: &str) -> Result<String> {
    let (before, _) = src.split_once(line).ok_or("no such line in the source")?;
    let offset = line.find(needle).ok_or("no such text in the line")?;
    let row = before.matches('\n').count() + 1;
    let column = before.rsplit('\n').next().map_or(0, str::len) + offset + 1;
    Ok(format!("{row}:{column}"))
}

/// Where the first `needle` is in `src`, as an error says it, `line:column`.
fn at(src: &str, needle: &str) -> Result<String> {
    in_line(src, needle, needle)
}

/// A kernel `k` of the parameters `params` that gives back a float, `acc`, which `body` adds to: `i` and `j` are 0.
fn kernel(params: &str, body: &[&str]) -> String {
    let body = body.join("\n");
    format!(
        "pub extern \"sysv64\" fn k({params}) -> f32 {{\n\
         let (i, j): (gpr, gpr);\nlet acc: ymm;\nzero(i);\nzero(j);\nzero(acc);\n{body}\nret(acc.xmm);\n}}\n"
    )
}

/// A loop over the floats of `xs` that adds one at a time, as long as `i` is below `len`.
fn scalar(len: &str) -> String {
    let compare = format!("cmp(i, {len});");
    kernel(
        "xs: &[f32]",
        &[
            "label(top);",
            &compare,
            "jae(done);",
            "vaddss(acc.xmm, acc.xmm, [xs + i * 4]);",
            "inc(i);",
            "jmp(top);",
            "label(done);",
        ],
    )
}

/// A loop over the floats of `xs` that adds eight at a time, as long as eight are left.
fn eights() -> String {
    kernel(
        "xs: &[f32]",
        &[
            "label(top);",
            "lea(j, [i + 8]);",
            "cmp(j, xs.len);",
            "ja(done);",
            "vaddps(acc, acc, [xs + i * 4]);",
            "add(i, 8);",
            "jmp(top);",
            "label(done);",
        ],
    )
}

/// A loop over the floats of `xs` that writes eight at a time, as long as eight are left.
fn writes() -> String {
    "pub extern \"sysv64\" fn k(xs: &mut [f32]) {\n\
     let (i, j): (gpr, gpr);\nlet v: ymm;\nzero(i);\nvbroadcastss(v, [1.5]);\nlabel(top);\nlea(j, [i + 8]);\n\
     cmp(j, xs.len);\nja(done);\nvmovups([xs + i * 4], v);\nadd(i, 8);\njmp(top);\nlabel(done);\n}\n"
        .to_string()
}

/// What kernite says of the address `reached` that the instruction `name` reaches, when no compare keeps `keeps`.
fn cannot_keep(place: &str, name: &str, reached: &str, keeps: &str) -> String {
    format!(
        "{place}: `{name}` reaches `{reached}`, which kernite cannot keep inside `xs`: no compare on every way to it \
         keeps {keeps}, so a kernel that does is an `unsafe fn`"
    )
}

/// A slice is the address and the length in the next register, as the pointer and the count are, so a loop that
/// compares with the length is the loop over a pointer and a count, instruction for instruction: one that adds a float
/// at a time, one that adds eight, and one that writes.
#[test]
fn a_loop_over_a_slice_is_what_it_is_with_a_pointer_and_a_count() -> Result {
    let lines = |listing: &str| listing.lines().filter(|l| !l.starts_with('#')).map(str::to_string).collect::<Vec<_>>();
    for src in [scalar("xs.len"), eights(), writes()] {
        let counted = src
            .replace("xs: &[f32]", "xs: *const f32, n: usize")
            .replace("xs: &mut [f32]", "xs: *mut f32, n: usize")
            .replace("xs.len", "n");
        let slice = kernite::x86_str(&src, "k")?.to_string();
        let pointer = kernite::x86_str(&trusted(&counted), "k")?.to_string();
        assert_eq!(lines(&slice), lines(&pointer), "{src}");
        // The slice's length is in the register after its address, where the count was.
        assert!(slice.contains("xs=rdi xs.len=rsi"), "{slice}");
        assert!(pointer.contains("xs=rdi n=rsi"), "{pointer}");
        // A kernel of a slice is as good an `unsafe fn`.
        assert_eq!(code(&trusted(&src))?, code(&src)?);
    }
    Ok(())
}

/// A slice is two registers, so a kernel takes six integers, whatever mixes of slices and numbers they are: the next
/// parameter is in the register after the length, a slice named `_` too, and a seventh integer is one SysV passes on
/// the stack, as is a slice whose length would be the seventh, after five numbers.
#[test]
fn a_slice_is_two_registers() -> Result {
    let src = "pub extern \"sysv64\" fn k(a: &[u8], b: u64, c: &mut [u16], d: f32) {\n}\n";
    let listing = kernite::x86_str(src, "k")?.to_string();
    assert_eq!(listing.lines().next(), Some("# k(gpr, gpr, gpr, gpr, gpr, vec) -> ()"), "{listing}");
    assert!(listing.contains("# a=rdi a.len=rsi b=rdx c=rcx c.len=r8 d=ymm0"), "{listing}");
    let src = "pub extern \"sysv64\" fn k(a: &[u8], b: &[u8], c: &[u8], d: u64) {\n}\n";
    let want = format!("{}: SysV passes a seventh integer on the stack, which kernite does not read", at(src, "u64")?);
    assert_eq!(error(src)?, want);
    let src = "pub extern \"sysv64\" fn k(a: u64, b: u64, c: u64, d: u64, e: u64, xs: &[u8]) {\n}\n";
    let want = format!(
        "{}: SysV passes a seventh integer on the stack, which kernite does not read; a slice is two integers, its \
         address and its length",
        at(src, "&[u8]")?
    );
    assert_eq!(error(src)?, want);
    let src = "pub extern \"sysv64\" fn k(_: &[u8], x: u64) {\n}\n";
    let listing = kernite::x86_str(src, "k")?.to_string();
    assert!(listing.contains("# _=rdi _.len=rsi x=rdx"), "{listing}");
    Ok(())
}

/// An index is kept below the length by a compare on every way to the address. Each of these keeps the address inside
/// where it is reached: a count up with `jae`, one to a limit that is the length less a step, a count down that stops
/// at 0, a displacement from the start once the length is compared with it, the last element, an index that is the
/// shorter of two lengths, one that the length of another slice is compared with, that the first is as long as, and two
/// ways to the address, one that has compared the length with 8, which comes to it first, and one with 1.
#[test]
fn an_index_is_kept_below_the_length_by_a_compare() -> Result {
    let limit = kernel(
        "xs: &[f32]",
        &[
            "let n: gpr;",
            "mov(n, xs.len);",
            "cmp(n, 8);",
            "jb(done);",
            "sub(n, 8);",
            "label(top);",
            "vaddps(acc, acc, [xs + i * 4]);",
            "add(i, 8);",
            "cmp(i, n);",
            "jbe(top);",
            "label(done);",
        ],
    );
    let down = kernel(
        "xs: &[f32]",
        &[
            "let n: gpr;",
            "mov(n, xs.len);",
            "label(top);",
            "test(n, n);",
            "jz(done);",
            "dec(n);",
            "vaddss(acc.xmm, acc.xmm, [xs + n * 4]);",
            "jmp(top);",
            "label(done);",
        ],
    );
    let first =
        kernel("xs: &[f32]", &["cmp(xs.len, 1);", "jb(done);", "vaddss(acc.xmm, acc.xmm, [xs]);", "label(done);"]);
    let last = kernel(
        "xs: &[f32]",
        &[
            "mov(i, xs.len);",
            "test(i, i);",
            "jz(done);",
            "dec(i);",
            "vaddss(acc.xmm, acc.xmm, [xs + i * 4]);",
            "label(done);",
        ],
    );
    let shorter = kernel(
        "xs: &[f32], ys: &[f32]",
        &[
            "let n: gpr;",
            "mov(n, xs.len);",
            "cmp(n, ys.len);",
            "cmova(n, ys.len);",
            "label(top);",
            "cmp(i, n);",
            "jae(done);",
            "vaddss(acc.xmm, acc.xmm, [xs + i * 4]);",
            "vaddss(acc.xmm, acc.xmm, [ys + i * 4]);",
            "inc(i);",
            "jmp(top);",
            "label(done);",
        ],
    );
    let equal = kernel(
        "xs: &[f32], ys: &[f32]",
        &[
            "cmp(xs.len, ys.len);",
            "jne(done);",
            "label(top);",
            "cmp(i, xs.len);",
            "jae(done);",
            "vaddss(acc.xmm, acc.xmm, [ys + i * 4]);",
            "inc(i);",
            "jmp(top);",
            "label(done);",
        ],
    );
    let merged = kernel(
        "xs: &[f32]",
        &[
            "cmp(xs.len, 8);",
            "jb(short);",
            "jmp(both);",
            "label(short);",
            "cmp(xs.len, 1);",
            "jb(done);",
            "label(both);",
            "vaddss(acc.xmm, acc.xmm, [xs + i * 4]);",
            "label(done);",
        ],
    );
    for src in [scalar("xs.len"), limit, down, first, last, shorter, equal, merged] {
        code(&src)?;
    }
    Ok(())
}

/// The address is kept inside by what the compares say of the whole of what it reaches: an instruction that reaches
/// eight floats at the index needs the index and eight at the length or below, and one at a displacement past the index
/// counts the displacement in, up to the last element it reaches a byte of. kernite says how many it counted, and
/// where.
#[test]
fn an_element_past_the_index_is_counted() -> Result {
    // A compare that keeps one element, for a load that reaches eight.
    let src = scalar("xs.len").replace("vaddss(acc.xmm, acc.xmm, ", "vaddps(acc, acc, ");
    let want = cannot_keep(&at(&src, "[xs + i * 4]")?, "vaddps", "[xs + i * 4]", "`i + 8` at or below `xs.len`");
    assert_eq!(error(&src)?, want);
    code(&trusted(&src))?;
    // Past the index by a displacement: one float on is two from the index, and `keeps` of them are kept.
    let src = |keeps: u8| {
        kernel(
            "xs: &[f32]",
            &[
                "label(top);",
                &format!("lea(j, [i + {keeps}]);"),
                "cmp(j, xs.len);",
                "ja(done);",
                "vaddss(acc.xmm, acc.xmm, [xs + i * 4 + 4]);",
                "inc(i);",
                "jmp(top);",
                "label(done);",
            ],
        )
    };
    code(&src(2))?;
    let want =
        cannot_keep(&at(&src(1), "[xs + i * 4 + 4]")?, "vaddss", "[xs + i * 4 + 4]", "`i + 2` at or below `xs.len`");
    assert_eq!(error(&src(1))?, want);
    code(&trusted(&src(1)))?;
    // Two bytes past the index, which reaches a byte of the next float, so two are kept.
    let src = |keeps: u8| {
        kernel(
            "xs: &[f32]",
            &[
                "label(top);",
                &format!("lea(j, [i + {keeps}]);"),
                "cmp(j, xs.len);",
                "ja(done);",
                "vaddss(acc.xmm, acc.xmm, [xs + i * 4 + 2]);",
                "inc(i);",
                "jmp(top);",
                "label(done);",
            ],
        )
    };
    code(&src(2))?;
    let want =
        cannot_keep(&at(&src(1), "[xs + i * 4 + 2]")?, "vaddss", "[xs + i * 4 + 2]", "`i + 2` at or below `xs.len`");
    assert_eq!(error(&src(1))?, want);
    Ok(())
}

/// What kernite cannot prove it refuses, at the first operand it cannot, and says that the kernel is then an
/// `unsafe fn`, which compiles as it did: an index compared with another slice's length, an index that a write moves on
/// after the compare, or that an instruction kernite follows no number through writes, no compare at all, a compare
/// that the jump does not use, and a compare on one way to the address and none on another.
#[test]
fn an_index_no_compare_keeps_below_the_length_is_refused() -> Result {
    let one = "`i + 1` at or below `xs.len`";
    let load = "vaddss(acc.xmm, acc.xmm, [xs + i * 4]);";
    let cases = [
        // The length of another slice, which says nothing of this one.
        (
            kernel(
                "xs: &[f32], ys: &[f32]",
                &["label(top);", "cmp(i, ys.len);", "jae(done);", load, "inc(i);", "jmp(top);", "label(done);"],
            ),
            one,
        ),
        // The index is moved on after the compare, so the compare keeps one element less of what is reached.
        (
            kernel(
                "xs: &[f32]",
                &["label(top);", "cmp(i, xs.len);", "jae(done);", "inc(i);", load, "jmp(top);", "label(done);"],
            ),
            one,
        ),
        // No compare at all.
        (kernel("xs: &[f32]", &[load]), one),
        // A compare that another instruction's flags hide from the jump.
        (kernel("xs: &[f32]", &["cmp(i, xs.len);", "add(j, 1);", "jae(done);", load, "label(done);"]), one),
        // The index is written after the compare by an instruction kernite follows no number through.
        (
            kernel(
                "xs: &[f32]",
                &["label(top);", "cmp(i, xs.len);", "jae(done);", "shl(i, 1);", load, "jmp(top);", "label(done);"],
            ),
            one,
        ),
        // A compare on one way to the address and none on the other.
        (
            kernel(
                "xs: &[f32]",
                &["cmp(j, 1);", "jb(skip);", "cmp(i, xs.len);", "jae(done);", "label(skip);", load, "label(done);"],
            ),
            one,
        ),
    ];
    for (src, keeps) in cases {
        let want = cannot_keep(&at(&src, "[xs + i * 4]")?, "vaddss", "[xs + i * 4]", keeps);
        assert_eq!(error(&src)?, want, "{src}");
        // As an `unsafe fn` the index is the kernel's to keep inside.
        code(&trusted(&src))?;
    }
    Ok(())
}

/// Each condition says what the compare it follows says, and no more: of the unsigned ones, only one that holds where
/// the index is below the length keeps the address inside, whichever way the jump goes, and a signed one, which reads
/// the numbers as signed, says nothing of the unsigned order. The address is reached where the jump falls through, or
/// where it goes.
#[test]
fn a_condition_says_what_its_compare_says_and_no_more() -> Result {
    // The index is a number kernite knows nothing of, `at`, which a caller gives.
    let reached = |jump: &str, goes: bool| {
        let load = "vaddss(acc.xmm, acc.xmm, [xs + at * 4]);";
        let kernel = |body: &[&str]| {
            let body = body.join("\n");
            format!(
                "pub extern \"sysv64\" fn k(xs: &[f32], at: usize) -> f32 {{\nlet acc: ymm;\nzero(acc);\n{body}\n\
                 label(done);\nret(acc.xmm);\n}}\n"
            )
        };
        let compare = format!("{jump}(near);");
        if goes {
            kernel(&["cmp(at, xs.len);", &compare, "jmp(done);", "label(near);", load])
        } else {
            kernel(&["cmp(at, xs.len);", &compare, load, "jmp(done);", "label(near);"])
        }
    };
    // Where `at` is below the length, which `jb` is for, and where it is not, which `jae` is for.
    for (jump, falls, goes) in [
        ("jae", true, false),
        ("jnb", true, false),
        ("jnc", true, false),
        ("jb", false, true),
        ("jc", false, true),
        ("jnae", false, true),
        ("ja", false, false),
        ("jnbe", false, false),
        ("jbe", false, false),
        ("jna", false, false),
        ("je", false, false),
        ("jz", false, false),
        ("jne", false, false),
        ("jnz", false, false),
        ("jl", false, false),
        ("jge", false, false),
        ("jg", false, false),
        ("jle", false, false),
    ] {
        for (src, proven) in [(reached(jump, false), falls), (reached(jump, true), goes)] {
            assert_eq!(code(&src).is_ok(), proven, "{jump}: {src}");
            // As an `unsafe fn` the index is the kernel's to keep inside.
            code(&trusted(&src))?;
        }
    }
    Ok(())
}

/// An instruction that writes part of a variable, or one kernite follows no number through, leaves the variable a
/// number of any value, so the compare before it says nothing: a 32-bit `add` may wrap at 2^32, and a 32-bit `mov` sets
/// the variable to a number of its own. The compare after it is what keeps the address inside.
#[test]
fn a_write_of_part_of_a_variable_is_a_number_of_any_value() -> Result {
    let load = "vaddss(acc.xmm, acc.xmm, [xs + i * 4]);";
    let after = |write: &str| kernel("xs: &[f32]", &["cmp(i, xs.len);", "jae(done);", write, load, "label(done);"]);
    for write in ["add(i.r32, 1);", "mov(i.r32, xs.len.r32);", "shl(i, 1);", "neg(i);", "imul(i, i, 3);"] {
        let src = after(write);
        let want = cannot_keep(&at(&src, "[xs + i * 4]")?, "vaddss", "[xs + i * 4]", "`i + 1` at or below `xs.len`");
        assert_eq!(error(&src)?, want, "{write}");
        code(&trusted(&src))?;
    }
    // The same write before a compare of its own is an index kernite keeps inside.
    let src = kernel("xs: &[f32]", &["mov(i.r32, xs.len.r32);", "cmp(i, xs.len);", "jae(done);", load, "label(done);"]);
    code(&src)?;
    Ok(())
}

/// `xor` of a gpr with itself, whole or its low 32 bits, is 0, as `zero` is; of its low 16, it is a number of any
/// value, as the bits above stay.
#[test]
fn an_index_zeroed_by_xor_is_0() -> Result {
    let src = |xor: &str| {
        kernel(
            "xs: &[f32]",
            &[
                "cmp(xs.len, 1);",
                "jb(done);",
                "mov(i, 5);",
                xor,
                "vaddss(acc.xmm, acc.xmm, [xs + i * 4]);",
                "label(done);",
            ],
        )
    };
    code(&src("xor(i, i);"))?;
    code(&src("xor(i.r32, i.r32);"))?;
    let src = src("xor(i.r16, i.r16);");
    let want = cannot_keep(&at(&src, "[xs + i * 4]")?, "vaddss", "[xs + i * 4]", "`i + 1` at or below `xs.len`");
    assert_eq!(error(&src)?, want);
    code(&trusted(&src))?;
    Ok(())
}

/// A move on a condition is the one or the other as the compare before it says: after `cmp(n, xs.len)`, `cmova`,
/// `cmovae` and `cmovnz` of `xs.len` leave `n` at or below `xs.len`, and the other conditions do not, nor does a signed
/// one, which says nothing.
#[test]
fn a_move_on_a_condition_is_the_one_or_the_other() -> Result {
    let src = |cmov: &str| {
        kernel(
            "xs: &[f32], ys: &[f32]",
            &[
                "let n: gpr;",
                "mov(n, ys.len);",
                "cmp(n, xs.len);",
                &format!("{cmov}(n, xs.len);"),
                "label(top);",
                "cmp(i, n);",
                "jae(done);",
                "vaddss(acc.xmm, acc.xmm, [xs + i * 4]);",
                "inc(i);",
                "jmp(top);",
                "label(done);",
            ],
        )
    };
    for cmov in ["cmova", "cmovae", "cmovnz"] {
        code(&src(cmov))?;
    }
    for cmov in ["cmovb", "cmovbe", "cmovz", "cmovg", "cmovl"] {
        let src = src(cmov);
        let want = cannot_keep(&at(&src, "[xs + i * 4]")?, "vaddss", "[xs + i * 4]", "`i + 1` at or below `xs.len`");
        assert_eq!(error(&src)?, want, "{cmov}");
        code(&trusted(&src))?;
    }
    Ok(())
}

/// What a slice holds is reached through the slice alone: its address is no register a kernel reads or writes, as an
/// array's is not, so no `add` moves it and no `mov` or `lea` takes it, to reach what is not inside by another name;
/// and it has a length, `.len`, and no other part, as a gpr has `.r32`.
#[test]
fn the_address_of_a_slice_is_reached_by_no_register() -> Result {
    let held = |name: &str| format!("`{name}` takes no memory as an operand: `[xs]` is what it holds");
    let put = "`lea` puts the address of `xs` in a register, where kernite cannot check what it reaches, so a kernel \
               that does is an `unsafe fn`";
    let cases = [
        ("add(xs, 4);", "xs", held("add")),
        ("mov(xs, 0);", "xs", held("mov")),
        ("mov(j, xs);", "xs", held("mov")),
        ("cmp(xs, 0);", "xs", held("cmp")),
        ("lea(j, [xs + 4]);", "[xs + 4]", put.to_string()),
        ("lea(j, [xs + i * 4]);", "[xs + i * 4]", put.to_string()),
        ("mov(j, xs.size);", "size", "a slice has a length, `.len`, and no `.size`".to_string()),
        ("mov(j, xs.r32);", "r32", "a slice has a length, `.len`, and no `.r32`".to_string()),
    ];
    for (line, token, said) in cases {
        let src = kernel("xs: &[f32]", &[line]);
        assert_eq!(error(&src)?, format!("{}: {said}", in_line(&src, line, token)?), "{line}");
        if said == put {
            code(&trusted(&src))?;
        }
    }
    Ok(())
}

/// A slice's elements are reached at a displacement known now, `[xs + 4]`, as an array's are, once the kernel has
/// compared the length with what the address reaches; and never before the start, where the index is 0.
#[test]
fn an_address_at_a_displacement_needs_the_length_compared() -> Result {
    let src = |compare: &[&str]| {
        let load: &[&str] = &["vaddss(acc.xmm, acc.xmm, [xs + 4]);", "label(done);"];
        kernel("xs: &[f32]", &[compare, load].concat())
    };
    let want = |src: &str| -> Result<String> {
        Ok(cannot_keep(&at(src, "[xs + 4]")?, "vaddss", "[xs + 4]", "`xs.len` at 2 or more"))
    };
    assert_eq!(error(&src(&[]))?, want(&src(&[]))?);
    // A compare that keeps two, on the way to it, and one that keeps one.
    code(&src(&["cmp(xs.len, 2);", "jb(done);"]))?;
    assert_eq!(error(&src(&["cmp(xs.len, 1);", "jb(done);"]))?, want(&src(&["cmp(xs.len, 1);", "jb(done);"]))?);
    code(&trusted(&src(&[])))?;
    // Before the start of the slice, where the index is 0, whatever the compare: the last element is `xs.len - 1` as
    // an index of its own, `dec(i)`.
    let before = kernel(
        "xs: &[f32]",
        &[
            "label(top);",
            "cmp(i, xs.len);",
            "jae(done);",
            "vaddss(acc.xmm, acc.xmm, [xs + i * 4 - 4]);",
            "inc(i);",
            "jmp(top);",
            "label(done);",
        ],
    );
    let want = format!(
        "{}: `vaddss` reads at `[xs + i * 4 - 4]`, before `xs` starts, when `i` is 0",
        at(&before, "[xs + i * 4 - 4]")?
    );
    assert_eq!(error(&before)?, want);
    Ok(())
}

/// An index that the kernel makes by an addition that may wrap, below 0 by a `sub`, or past `u64::MAX` by an `add`, an
/// `inc` or a `lea`, is a number kernite knows nothing of, and one that cannot wrap is as the compare left it.
#[test]
fn an_index_that_may_wrap_is_a_number_of_any_value() -> Result {
    // Two from the end, where the length may be 0 or 1.
    let src = kernel("xs: &[f32]", &["mov(i, xs.len);", "sub(i, 2);", "vaddss(acc.xmm, acc.xmm, [xs + i * 4]);"]);
    let want = cannot_keep(&at(&src, "[xs + i * 4]")?, "vaddss", "[xs + i * 4]", "`i + 1` at or below `xs.len`");
    assert_eq!(error(&src)?, want);
    // The same, once the length is two or more.
    let src = kernel(
        "xs: &[f32]",
        &[
            "mov(i, xs.len);",
            "cmp(i, 2);",
            "jb(done);",
            "sub(i, 2);",
            "vaddss(acc.xmm, acc.xmm, [xs + i * 4]);",
            "label(done);",
        ],
    );
    code(&src)?;
    // One past a number a caller gives, which may be `u64::MAX`, and `j` then 0.
    for add in [&["lea(j, [at + 1]);"][..], &["mov(j, at);", "add(j, 1);"], &["mov(j, at);", "inc(j);"]] {
        let compare = ["cmp(j, xs.len);", "ja(done);", "vaddss(acc.xmm, acc.xmm, [xs + at * 4]);", "label(done);"];
        let src = kernel("xs: &[f32], at: usize", &[add, &compare].concat());
        let want = cannot_keep(&at(&src, "[xs + at * 4]")?, "vaddss", "[xs + at * 4]", "`at + 1` at or below `xs.len`");
        assert_eq!(error(&src)?, want, "{add:?}");
    }
    Ok(())
}

/// The length of a slice stays as the kernel was given it, for the compares of it are what keep an address inside the
/// slice: an instruction that writes it is refused, and a call that gives back its result to it, which says so and
/// where; what the kernel does with the length is to copy it to a variable that it may change. An `unsafe fn` keeps no
/// length for kernite to prove with, so it may.
#[test]
fn a_slice_length_is_not_written() -> Result {
    let said = |src: &str| -> Result<String> {
        Ok(format!(
            "{}: `xs.len` is the length of a slice, which kernite keeps as it was given to prove each address inside \
             the slice, so a kernel that writes it is an `unsafe fn`; `mov(n, xs.len)` copies it to a variable `n`, \
             which may change",
            at(src, "len")?
        ))
    };
    let bodies: [&[&str]; 5] = [
        &["dec(xs.len);"],
        &["mov(xs.len, 0);"],
        &["sub(xs.len, 1);"],
        &["xor(xs.len, xs.len);"],
        &["add(i, 1);", "cmova(xs.len, i);"],
    ];
    let call = "pub extern \"sysv64\" fn k(xs: &[f32]) -> f32 {\nlet acc: ymm;\nzero(acc);\ncall(xs.len, f, ());\n\
                ret(acc.xmm);\n}\npub extern \"sysv64\" fn f() -> u64 {\nlet g: gpr;\nmov(g, 1);\nret(g);\n}\n";
    for src in bodies.map(|body| kernel("xs: &[f32]", body)).into_iter().chain([call.to_string()]) {
        assert_eq!(error(&src)?, said(&src)?, "{src}");
        code(&trusted(&src))?;
    }
    // A copy is a variable of its own, and the length is read, in a compare, as often as the kernel wants.
    code(&kernel("xs: &[f32]", &["let n: gpr;", "mov(n, xs.len);", "inc(n);", "cmp(xs.len, n);"]))?;
    Ok(())
}

/// A slice is read through `&` and written through `&mut`, as an array is, and the index is scaled by the bytes of an
/// element, as the address is: `[xs + i * 4]` of floats, `[xs + i]` of bytes. Another scale is no index of the slice.
#[test]
fn a_slice_is_read_or_written_and_its_index_is_scaled_by_its_elements() -> Result {
    let store = |ty: &str| {
        format!(
            "pub extern \"sysv64\" fn k(xs: {ty}) {{\nlet i: gpr;\nlet v: xmm;\nzero(i);\nzero(v);\nlabel(top);\n\
             cmp(i, xs.len);\njae(done);\nvmovss([xs + i * 4], v);\ninc(i);\njmp(top);\nlabel(done);\n}}\n"
        )
    };
    // A write through `&`.
    let src = store("&[f32]");
    let want = format!(
        "{}: `vmovss` writes `xs`, which is `&[f32]`: a kernel only reads what it is given by `&`, and writes what it \
         is given by `&mut`",
        at(&src, "[xs + i * 4]")?
    );
    assert_eq!(error(&src)?, want);
    code(&store("&mut [f32]"))?;
    // The scale is the element's bytes.
    let get = |ty: &str, address: &str| {
        format!(
            "pub extern \"sysv64\" fn k(xs: &[{ty}]) {{\nlet (i, g): (gpr, gpr);\nzero(i);\nlabel(top);\n\
             cmp(i, xs.len);\njae(done);\nmovzx(g.r32, byte[{address}]);\ninc(i);\njmp(top);\nlabel(done);\n}}\n"
        )
    };
    for (ty, size) in [("u8", 1), ("i16", 2), ("f32", 4), ("u64", 8)] {
        let scaled = |k: u64| if k == 1 { "xs + i".to_string() } else { format!("xs + i * {k}") };
        // `movzx` reads a byte, whatever the element's size.
        code(&get(ty, &scaled(size)))?;
        // Twice the bytes, or half of them where that is no scale x86 has.
        let other = if size == 8 { 4 } else { size * 2 };
        let wrong = get(ty, &scaled(other));
        let want = format!(
            "{}: `xs` is `&[{ty}]`, so an index into it is scaled by the bytes of one element, {size}, as in \
             `[{}]`; not by {other}",
            at(&wrong, "[xs")?,
            scaled(size)
        );
        assert_eq!(error(&wrong)?, want);
    }
    Ok(())
}

/// A slice holds Rust's numbers, as an array and stack memory do: what else it holds is refused at the type, in an
/// `unsafe fn` as well, for the type is wrong, not what the kernel does. A slice of arrays is one of them, which the
/// message names whole.
#[test]
fn a_slice_holds_rusts_numbers() -> Result {
    let numbers = "a slice by reference holds Rust's numbers, `u8` to `u64`, `i8` to `i64`, `usize`, `isize`, `f32` \
                   and `f64`";
    for (ty, elem) in [
        ("&[bool]", "bool"),
        ("&mut [bool]", "bool"),
        ("&[u128]", "u128"),
        ("&[[u64; 2]]", "[u64; 2]"),
        ("&[*const u8]", "*const u8"),
    ] {
        let src = format!("pub extern \"sysv64\" fn k(xs: {ty}) {{\n}}\n");
        assert_eq!(error(&src)?, format!("{}: {numbers}; not `{elem}`", at(&src, ty)?), "{ty}");
        let src = trusted(&src);
        assert_eq!(error(&src)?, format!("{}: {numbers}; not `{elem}`", at(&src, ty)?), "unsafe {ty}");
    }
    for ty in ["u8", "i8", "u16", "i16", "u32", "i32", "u64", "i64", "usize", "isize", "f32", "f64"] {
        code(&format!("pub extern \"sysv64\" fn k(xs: &[{ty}]) {{\n}}\n"))?;
    }
    Ok(())
}

/// A vector of indices reaches the slice at addresses kernite cannot keep inside it, as it does an array, so a kernel
/// that does is an `unsafe fn`.
#[test]
fn a_vector_of_indices_is_no_index_kernite_keeps_inside_a_slice() -> Result {
    let src = "pub extern \"sysv64\" fn k(xs: &[u32]) {\nlet (v, m, i): (ymm, ymm, ymm);\nzero(v);\nzero(i);\n\
               vpcmpeqd(m, v, v);\nvpgatherdd(v, [xs + i * 4], m);\n}\n";
    let want = format!(
        "{}: `vpgatherdd` reaches `xs` at a vector of indices, which kernite cannot keep inside it, so a kernel that \
         does is an `unsafe fn`",
        at(src, "[xs + i * 4]")?
    );
    assert_eq!(error(src)?, want);
    code(&trusted(src))?;
    Ok(())
}

/// A kernel is given slices and arrays together, and numbers. A slice is no memory a call passes to another kernel,
/// which takes the address and the length as two numbers; and a `#[cooperative]` kernel, whose threads share one
/// memory, takes the same two numbers, not a slice. Each says what to write.
#[test]
fn a_slice_is_given_to_a_kernel_alone() -> Result {
    let both = "pub extern \"sysv64\" fn k(xs: &[f32], out: &mut [f32; 8]) {\nlet (i, j): (gpr, gpr);\nlet v: ymm;\n\
                zero(i);\nzero(v);\nlabel(top);\nlea(j, [i + 8]);\ncmp(j, xs.len);\nja(done);\n\
                vaddps(v, v, [xs + i * 4]);\nadd(i, 8);\njmp(top);\nlabel(done);\nvmovups([out], v);\n}\n";
    code(both)?;
    let called = "pub extern \"sysv64\" fn first(ys: &[f32]) -> f32 {\nlet a: xmm;\nzero(a);\nret(a);\n}\n";
    let src = format!(
        "pub extern \"sysv64\" fn k(xs: &[f32]) -> f32 {{\nlet a: xmm;\ncall(a, first, xs);\nret(a);\n}}\n{called}"
    );
    let want = format!(
        "{}: `first` takes a `&[f32]` `ys`, which a kernel passes to no function: it takes the address and the length \
         as two numbers, `*const f32` and `n: usize`",
        at(&src, "xs);")?
    );
    assert_eq!(error(&src)?, want);
    // The kernel that takes one is called by Rust, as any kernel is.
    assert!(kernite::x86_str(&src, "first").is_ok());
    let src = "#[cooperative]\npub extern \"sysv64\" fn k(xs: &[f32]) {\n}\n";
    let want = format!(
        "{}: a cooperative kernel takes no slice, `&[f32]`: its threads take the address and the length as two \
         numbers, `*const f32` and `n: usize`",
        at(src, "&[f32]")?
    );
    assert_eq!(error(src)?, want);
    Ok(())
}

/// What a safe kernel may not do of an array it may not do of a slice: a bit at an offset in a register, which reaches
/// as far as the offset says, and a non-temporal store that the kernel may return after with no `sfence` between, which
/// Rust's model has a data race. Each name is refused, locked or not, and as an `unsafe fn` the kernel takes it.
#[test]
fn a_slice_takes_no_bit_at_a_register_and_no_store_that_may_come_after() -> Result {
    for name in ["bt", "btc", "btr", "bts", "lock.btc", "lock.btr", "lock.bts"] {
        let src = format!("pub extern \"sysv64\" fn k(xs: &mut [u64], i: u64) {{\n{name}([xs], i);\n}}\n");
        let want = format!(
            "2:{}: `{name}` reaches a bit at the offset in `i`, which kernite cannot keep inside `xs`, so a kernel \
             that does is an `unsafe fn`",
            name.len() + 8
        );
        assert_eq!(error(&src)?, want, "{name}");
        code(&trusted(&src))?;
    }
    let kernel = |body: &str| {
        format!(
            "pub extern \"sysv64\" fn k(xs: &mut [u64], g: u64) {{\n\
             cmp(xs.len, 1);\njb(done);\n{body}\nlabel(done);\n}}\n"
        )
    };
    // The store is ordered on the way to the `ret` that it is on: the compare above skips the store and the fence both.
    for fence in ["sfence", "mfence"] {
        code(&kernel(&format!("movnti([xs], g);\n{fence}();")))?;
    }
    let src = kernel("movnti([xs], g);");
    let want = format!(
        "{}: `movnti` is a non-temporal store, and the kernel may return before an `sfence` orders it, so a kernel \
         that does is an `unsafe fn`",
        at(&src, "movnti")?
    );
    assert_eq!(error(&src)?, want);
    code(&trusted(&src))?;
    // A store of a vector is at a multiple of 32, which no slice is, as its elements are no more than 8 bytes: the
    // address of the slice is a multiple of its element's alignment, and kernite does not say more of it.
    for (elem, store, size) in [("f32", "vmovntps", 4), ("f64", "vmovntpd", 8), ("u64", "vmovntdq", 8)] {
        let src = format!(
            "pub extern \"sysv64\" fn k(xs: &mut [{elem}]) {{\nlet v: ymm;\nzero(v);\ncmp(xs.len, 8);\njb(done);\n\
             {store}([xs], v);\nlabel(done);\n}}\n"
        );
        let want = format!(
            "{}: `{store}` writes 32 bytes at once, at an address that is a multiple of 32; `xs` is at a multiple of \
             {size}",
            at(&src, "[xs], v")?
        );
        assert_eq!(error(&src)?, want, "{store}");
    }
    Ok(())
}

/// A slice is at a multiple of its element's bytes and no more, as Rust lays it out, in a kernel of either kind: an
/// instruction that needs a multiple of 32, `vmovaps` of a `ymm`, is an error in a slice of floats, where `vmovups`
/// reads at any.
#[test]
fn a_slice_is_at_a_multiple_of_its_elements_bytes() -> Result {
    let src = |load: &str| {
        format!(
            "pub extern \"sysv64\" fn k(xs: &[f32]) -> f32 {{\nlet v: ymm;\nzero(v);\ncmp(xs.len, 8);\njb(done);\n\
             {load}(v, [xs]);\nlabel(done);\nret(v.xmm);\n}}\n"
        )
    };
    let aligned = src("vmovaps");
    let want = format!(
        "{}: `vmovaps` reads 32 bytes at once, at an address that is a multiple of 32; `xs` is at a multiple of 4: \
         `vmovups` at any",
        at(&aligned, "[xs]")?
    );
    assert_eq!(error(&aligned)?, want);
    assert_eq!(error(&trusted(&aligned))?, want);
    code(&src("vmovups"))?;
    Ok(())
}

/// What the proof costs grows with the gprs a length or an index is related to, not with all that a kernel has: a
/// kernel of a slice that adds up a thousand gprs, none of them an index, compiles safe in about the time it takes as
/// an `unsafe fn`, within twenty times and a second, where a node for each gpr made it take minutes.
#[test]
fn a_safe_kernel_of_a_thousand_gprs_compiles_in_about_the_time_of_an_unsafe_one() -> Result {
    let mut lines: Vec<String> = ["let s: gpr;", "zero(s);"].map(String::from).into();
    for k in 0..1000 {
        lines.extend([format!("let t{k}: gpr;"), format!("mov(t{k}, {k});"), format!("add(s, t{k});")]);
    }
    lines.extend(["cmp(xs.len, 1);", "jb(done);", "vaddss(acc.xmm, acc.xmm, [xs]);", "label(done);"].map(String::from));
    let src = kernel("xs: &[f32]", &lines.iter().map(String::as_str).collect::<Vec<_>>());
    let time = |src: &str| -> Result<Duration> {
        let start = Instant::now();
        code(src)?;
        Ok(start.elapsed())
    };
    let unsafe_time = time(&trusted(&src))?;
    let safe_time = time(&src)?;
    assert!(safe_time < unsafe_time * 20 + Duration::from_secs(1), "safe {safe_time:?}, unsafe {unsafe_time:?}");
    Ok(())
}
