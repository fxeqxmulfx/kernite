//! A GPU kernel that is no `unsafe fn`: kernite proves, as its instructions come, that each load and store reaches one
//! whole element, inside the memory it is lent, at its alignment, that no two threads write one address, and that none
//! reads what another writes. What the proof leaves for the launch, a count of elements say, the host checks before the
//! kernel runs, and the kernel's code says it. What kernite cannot prove it refuses, at the first instruction or
//! operand it cannot, and says that the kernel is then an `unsafe fn`, which compiles as it always has.

use kernite::Requirement;

type Result<T = (), E = Box<dyn std::error::Error + Send + Sync>> = std::result::Result<T, E>;

/// The helper that `saxpy` of the examples calls: this thread's index among all the grid's.
const GLOBAL_INDEX: &str = r#"
fn global_index(i: u32) {
    let (block, size, thread): (u32, u32, u32);
    mov.u32(block, ctaid.x);
    mov.u32(size, ntid.x);
    mov.u32(thread, tid.x);
    mad.lo.u32(i, block, size, thread);
}
"#;

/// A function a kernel may call if it is an `unsafe fn`.
const ONE: &str = "extern \"ptx-func\" fn one(a: u32) -> u32 {\n    ret(a);\n}\n";

/// The lines that give a thread its index `i` among all the grid's, and end it where `i` is not below `n`.
const BELOW_N: [&str; 7] =
    ["let i: u32;", "global_index(i);", "let past: pred;", "setp.ge.u32(past, i, n);", "if past {", "    ret();", "}"];

/// `saxpy` of the examples, with no `unsafe`.
const SAXPY: &str = r#"pub extern "ptx-kernel" fn saxpy(a: f32, x: *const f32, y: *mut f32, n: u32) {
    let i: u32;
    global_index(i);
    let past: pred;
    setp.ge.u32(past, i, n);
    if past {
        ret();
    }
    let (at_x, at_y): (u64, u64);
    mad.wide.u32(at_x, i, 4, x);
    mad.wide.u32(at_y, i, 4, y);
    let (xi, yi): (f32, f32);
    ld.global.f32(xi, [at_x]);
    ld.global.f32(yi, [at_y]);
    fma.rn.f32(yi, a, xi, yi);
    st.global.f32([at_y], yi);
}
"#;

/// Kernel `k`, with no `unsafe`, of parameters `params` and body `lines`, one to a line, 4 columns in.
fn kernel(params: &str, lines: &[&str]) -> String {
    let body: String = lines.iter().map(|l| format!("    {l}\n")).collect();
    format!("pub extern \"ptx-kernel\" fn k({params}) {{\n{body}}}\n")
}

/// [`BELOW_N`] and then `lines`.
fn below_n(lines: &[&'static str]) -> Vec<&'static str> {
    BELOW_N.iter().chain(lines).copied().collect()
}

/// `kernel` and the helper `global_index`, with `extra` before them.
fn module(extra: &str, kernel: &str) -> String {
    format!("{extra}{kernel}{GLOBAL_INDEX}")
}

/// `src` with `unsafe` before each `extern "ptx-kernel"`: its kernels are `unsafe fn`s.
fn trusted(src: &str) -> String {
    src.replace("pub extern \"ptx-kernel\"", "pub unsafe extern \"ptx-kernel\"")
}

/// What the error of a kernel that does `what`, which kernite cannot prove safe, says.
fn cannot(what: &str) -> String {
    format!("{what}, and kernite cannot prove that this is safe, so a kernel that does is an `unsafe fn`")
}

/// A condition on a launch as one line: its terms, each a factor and the quantities multiplied by it, and what it
/// keeps.
fn show(r: &Requirement) -> String {
    let terms: Vec<String> = r
        .terms
        .iter()
        .map(|(factor, quantities)| {
            let each = quantities.iter().map(|q| format!("{q:?}"));
            std::iter::once(factor.to_string()).chain(each).collect::<Vec<_>>().join("*")
        })
        .collect();
    format!("{} >= 0 ({:?})", terms.join(" + "), r.why)
}

/// What a launch of kernel `name` of `src` must hold, or the error where kernite cannot prove the kernel safe.
fn requires(src: &str, name: &str) -> Result<Vec<String>, String> {
    let module = kernite::lang::parse(src).map_err(|e| e.at(src))?;
    let f = kernite::function(&module, name).ok_or("no such function")?;
    let compiled = kernite::ptx(&module, f, &[], "").map_err(|e| e.at(src))?;
    Ok(compiled.code.requires.iter().map(show).collect())
}

/// Where `needle` first is in `src`, as `line:column`.
fn place(src: &str, needle: &str) -> Result<String> {
    let (line, text) = src.lines().enumerate().find(|(_, l)| l.contains(needle)).ok_or("no such text")?;
    Ok(format!("{}:{}", line + 1, text.find(needle).ok_or("no such text")? + 1))
}

#[test]
fn saxpy_is_safe_and_its_launch_holds_what_it_reaches() -> Result {
    let src = module("", SAXPY);
    // Each of `x` and `y` has as many elements as `n` says.
    assert_eq!(
        requires(&src, "saxpy")?,
        ["-1*Param(3) + 1*Len(1) >= 0 (Inside(1))", "-1*Param(3) + 1*Len(2) >= 0 (Inside(2))"]
    );
    // The PTX is that of the kernel with `unsafe`: kernite adds nothing to it.
    assert_eq!(kernite::ptx_str(&src)?, kernite::ptx_str(&trusted(&src))?);
    Ok(())
}

/// A kernel that loads and stores each element whole, by one access of its size, is proved safe whatever the size of
/// the element, 1, 2, 4 or 8 bytes: a launch holds that each of the memories has as many elements as `n` says, and the
/// PTX is that of the kernel with `unsafe`.
#[test]
fn a_kernel_that_reaches_each_element_whole_is_safe_whatever_its_size() -> Result {
    for (ty, bytes) in [("u8", 1), ("u16", 2), ("u32", 4), ("u64", 8)] {
        // PTX has no register of 8 bits: a byte goes in one of 16.
        let register = if bytes == 1 { "u16" } else { ty };
        let lines = [
            "let i: u32;".to_string(),
            "global_index(i);".to_string(),
            "let past: pred;".to_string(),
            "setp.ge.u32(past, i, n);".to_string(),
            "if past {".to_string(),
            "    ret();".to_string(),
            "}".to_string(),
            "let (at_x, at_y): (u64, u64);".to_string(),
            format!("mad.wide.u32(at_x, i, {bytes}, x);"),
            format!("mad.wide.u32(at_y, i, {bytes}, y);"),
            format!("let v: {register};"),
            format!("ld.global.{ty}(v, [at_x]);"),
            format!("st.global.{ty}([at_y], v);"),
        ];
        let lines: Vec<&str> = lines.iter().map(String::as_str).collect();
        let src = module("", &kernel(&format!("x: *const {ty}, y: *mut {ty}, n: u32"), &lines));
        assert_eq!(
            requires(&src, "k")?,
            ["-1*Param(2) + 1*Len(0) >= 0 (Inside(0))", "-1*Param(2) + 1*Len(1) >= 0 (Inside(1))"],
            "{src}"
        );
        assert_eq!(kernite::ptx_str(&src)?, kernite::ptx_str(&trusted(&src))?, "{src}");
    }
    Ok(())
}

/// What kernite cannot prove a GPU kernel safe of is refused at the first instruction or operand that it cannot, in
/// words that say what the kernel does and that it is then an `unsafe fn`; as one, it compiles, with nothing left for
/// a launch to hold.
#[test]
fn what_kernite_cannot_prove_safe_is_refused_unless_the_kernel_is_unsafe() -> Result {
    // A case is the module, the text where its error is, and the error: at the operand it is about, and else at the
    // last word of the instruction's name, as every error of an instruction's is.
    let case = |extra: &str, params: &str, lines: &[&str], needle: &'static str, error: String| {
        (module(extra, &kernel(params, lines)), needle, error)
    };
    let next = ["let (next, at): (u32, u64);", "add.u32(next, i, 1);", "mad.wide.u32(at, next, 4, y);"];
    let own = ["let i: u32;", "global_index(i);", "let at: u64;", "mad.wide.u32(at, i, 4, y);"];
    let one = ["let one: f32;", "mov.f32(one, 1.0);"];
    let store = ["st.global.f32([at], one);"];
    let store_next = [&next[..], &one, &store].concat();
    let store_own = [&own[..], &one, &store].concat();
    let compare_after = [&store_own[..], &BELOW_N[2..]].concat();
    let load_next = [&next[..], &["let v: f32;", "ld.global.f32(v, [at]);"]].concat();
    let volatile = ["let at: u64;", "mad.wide.u32(at, i, 4, y);", "st.volatile.global.f32([at], 1.0);"];
    let generic = ["let at: u64;", "mad.wide.u32(at, i, 4, y);", "st.f32([at], 1.0);"];
    let moved = ["let at: u64;", "mov.u64(at, 0);", "let v: f32;", "ld.global.f32(v, [at]);"];
    let skewed = ["let at: u64;", "mad.wide.u32(at, i, 4, y);", "st.global.f32([at + 2], 1.0);"];
    let wide = [
        "let at: u64;",
        "mad.wide.u32(at, i, 4, y);",
        "let (a, b, c, d): (f32, f32, f32, f32);",
        "ld.global.v4.f32((a, b, c, d), [at]);",
    ];
    let part_load = ["let at: u64;", "mad.wide.u32(at, i, 4, x);", "let h: u16;", "ld.global.u16(h, [at + 2]);"];
    let byte_load = ["let at: u64;", "mad.wide.u32(at, i, 4, x);", "let h: u16;", "ld.global.u8(h, [at + 1]);"];
    let part_store =
        ["let at: u64;", "mad.wide.u32(at, i, 4, y);", "let h: u16;", "mov.u16(h, 7);", "st.global.u16([at + 2], h);"];
    let lanes =
        ["let at: u64;", "mad.wide.u32(at, i, 4, x);", "let (h, l): (u16, u16);", "ld.global.v2.u16((h, l), [at]);"];
    let half = ["let at: u64;", "mad.wide.u32(at, i, 8, x);", "let h: u32;", "ld.global.u32(h, [at + 4]);"];
    let before = ["let at: u64;", "mad.wide.u32(at, i, 4, y);", "st.global.f32([at - 4], 1.0);"];
    let back = [
        "let (b, j): (u32, u32);",
        "mov.u32(b, ctaid.x);",
        "add.u32(j, b, k);",
        "let at: u64;",
        "mad.wide.u32(at, j, 4, y);",
        "st.global.f32([at], 1.0);",
        "st.global.f32([at - 8], 1.0);",
    ];
    let cut = [
        "let c: u32;",
        "mov.u32(c, ctaid.x);",
        "let (c64, big, v, d, e, off, at): (u64, u64, u64, u64, u64, u64, u64);",
        "cvt.u64.u32(c64, c);",
        "shl.b64(big, c64, 32);",
        "add.u64(v, big, c64);",
        "cvt.u64.u32(d, v);",
        "sub.u64(e, d, big);",
        "shl.b64(off, e, 2);",
        "add.u64(at, y, off);",
        "st.global.f32([at], 1.0);",
    ];
    let not_only = "at an address that is not only its thread's";
    let no_compare = "reaches an element at an index that no compare keeps below the memory's length";
    let cases = [
        // A thread writes an element that is not its own.
        case(
            "",
            "y: *mut f32, n: u32",
            &below_n(&store_next),
            "[at], one",
            cannot(&format!("`st.global.f32` writes {not_only}")),
        ),
        // A block, one float of its own, writes the float of the block two before it.
        case("", "y: *mut f32, k: u32", &back, "[at - 8]", cannot(&format!("`st.global.f32` writes {not_only}"))),
        // It reads an element of memory the kernel may write that is not its own.
        case(
            "",
            "y: *mut f32, n: u32",
            &below_n(&load_next),
            "[at]);",
            cannot(&format!("`ld.global.f32` reads memory the kernel may write, {not_only}")),
        ),
        // No compare keeps the index below the length, or the compare comes after the access.
        case("", "y: *mut f32", &store_own, "[at], one", cannot(&format!("`st.global.f32` {no_compare}"))),
        case("", "y: *mut f32, n: u32", &compare_after, "[at], one", cannot(&format!("`st.global.f32` {no_compare}"))),
        // Shared memory, a barrier, an instruction of a warp's threads together, an atomic, a static and a call.
        case(
            "",
            "y: *mut f32",
            &["let tile: shared<[f32; 32]>;", "let s: u32;", "mov.u32(s, tile);"],
            "tile);",
            "`tile` is shared memory, which the threads of a block reach together, so a kernel that names it is an \
             `unsafe fn`"
                .into(),
        ),
        case(
            "",
            "y: *mut f32",
            &["bar.sync(0);"],
            "sync(0)",
            cannot("`bar.sync` makes threads wait for each other, or orders what they write"),
        ),
        // Shared memory that the launch gives, local memory, and the rest of what makes threads wait or orders what
        // they write.
        case(
            "",
            "y: *mut f32",
            &["let tile: shared<[f32]>;", "let s: u32;", "mov.u32(s, tile);"],
            "tile);",
            "`tile` is shared memory, which the threads of a block reach together, so a kernel that names it is an \
             `unsafe fn`"
                .into(),
        ),
        case(
            "",
            "y: *mut f32",
            &["let own: local<[f32; 8]>;", "let s: u32;", "mov.u32(s, own);"],
            "own);",
            "`own` is local memory, which kernite does not follow, so a kernel that names it is an `unsafe fn`".into(),
        ),
        case(
            "",
            "y: *mut f32",
            &["membar.gl();"],
            "gl()",
            cannot("`membar.gl` makes threads wait for each other, or orders what they write"),
        ),
        case(
            "",
            "y: *mut f32",
            &["fence.sc.gpu();"],
            "gpu()",
            cannot("`fence.sc.gpu` makes threads wait for each other, or orders what they write"),
        ),
        // The matrix multiply of a warp's threads, `cp.async`, which another thread's copy may race, a texture and a
        // surface.
        case(
            "",
            "n: u32",
            &[
                "let (a, b, c, d, e, f): (b32, b32, b32, b32, b32, b32);",
                "mov.b32(a, n); mov.b32(b, n); mov.b32(c, n); mov.b32(d, n); mov.b32(e, n); mov.b32(f, n);",
                "let (p, q, r, s): (f32, f32, f32, f32);",
                "mov.f32(p, 0.0); mov.f32(q, 0.0); mov.f32(r, 0.0); mov.f32(s, 0.0);",
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32((p, q, r, s), (a, b, c, d), (e, f), (p, q, r, s));",
            ],
            "f32((p",
            cannot(
                "`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` works on the registers of the threads of a warp \
                 together",
            ),
        ),
        case(
            "",
            "s: u32, y: *mut f32",
            &["cp.r#async.ca.shared.global([s], [y], 4);"],
            "global([s]",
            cannot("`cp.async.ca.shared.global` reaches memory that other threads reach at once"),
        ),
        case(
            "",
            "t: u64, x: f32, y: f32",
            &["let (r, g, b, a): (f32, f32, f32, f32);", "tex._2d.v4.f32.f32((r, g, b, a), [t, (x, y)]);"],
            "f32((r",
            cannot("`tex.2d.v4.f32.f32` reaches a texture or a surface"),
        ),
        case(
            "",
            "s: u64, n: u32",
            &["let (a, b): (b32, b32);", "suld.b._2d.v2.b32.trap((a, b), [s, (n, n)]);"],
            "trap((a",
            cannot("`suld.b.2d.v2.b32.trap` reaches a texture or a surface"),
        ),
        case(
            "",
            "n: u32",
            &["let v: u32;", "shfl.sync.idx.b32(v, n, 0, 31, 0xffffffff);"],
            "b32(v, n",
            cannot("`shfl.sync.idx.b32` works on the registers of the threads of a warp together"),
        ),
        case(
            "",
            "y: *mut u32",
            &["red.global.add.u32([y], 1);"],
            "u32([y]",
            cannot("`red.global.add.u32` reaches memory that other threads reach at once"),
        ),
        case(
            "static COUNTS: global<[u32; 16]>;\n",
            "y: *mut u32",
            &["let at: u64;", "mov.u64(at, COUNTS);"],
            "COUNTS);",
            "`COUNTS` is memory of the module's, which every kernel and the host reach, so a kernel that names it is \
             an `unsafe fn`"
                .into(),
        ),
        case(ONE, "n: u32", &["let r: u32;", "call(r, one, (n));"], "call(r", cannot("`call` calls a function")),
        // An atomic, a jump, a vote of a warp's threads, and a conversion of an address.
        case(
            "",
            "y: *mut u32",
            &["let old: u32;", "atom.global.add.u32(old, [y], 1);"],
            "u32(old",
            cannot("`atom.global.add.u32` reaches memory that other threads reach at once"),
        ),
        case("", "y: *mut u32", &["label(top);", "bra(top);"], "bra(top)", cannot("`bra` jumps")),
        case(
            "",
            "n: u32",
            &["let p: pred;", "setp.eq.u32(p, n, 0);", "vote.sync.all.pred(p, p, 0xffffffff);"],
            "pred(p, p",
            cannot("`vote.sync.all.pred` works on the registers of the threads of a warp together"),
        ),
        case(
            "",
            "y: *mut f32",
            &["let g: u64;", "cvta.to.global.u64(g, y);"],
            "u64(g, y",
            cannot("`cvta.to.global.u64` reaches memory in a way kernite does not follow"),
        ),
        // A word that makes a load or a store more than a plain one, a generic address, one not from a pointer, a
        // pointer to what is not the access's size, and an address its element does not align.
        case(
            "",
            "y: *mut f32, n: u32",
            &below_n(&volatile),
            "f32([at], 1.0)",
            cannot("`st.volatile.global.f32` has `.volatile`, which a plain load or store has not"),
        ),
        case(
            "",
            "y: *mut f32, n: u32",
            &below_n(&generic),
            "f32([at], 1.0)",
            cannot(
                "`st.f32` reaches memory by a generic address, which may be shared or local memory as well as global",
            ),
        ),
        case(
            "",
            "y: *mut f32",
            &moved,
            "[at]);",
            cannot(
                "`ld.global.f32` reaches memory at an address that kernite cannot follow to a pointer of the kernel's",
            ),
        ),
        // An address worked out from `cvt` of a 64-bit register, which keeps its low 32 bits: the proof does not
        // follow it.
        case(
            "",
            "y: *mut f32",
            &cut,
            "[at], 1.0",
            cannot(
                "`st.global.f32` reaches memory at an address that kernite cannot follow to a pointer of the kernel's",
            ),
        ),
        case(
            "",
            "y: *mut f32, n: u32",
            &below_n(&skewed),
            "[at + 2]",
            cannot("`st.global.f32` reaches an address that is not a multiple of 4 bytes from its pointer"),
        ),
        case(
            "",
            "y: *mut f32, n: u32",
            &below_n(&before),
            "[at - 4]",
            cannot("`st.global.f32` reaches memory before it starts, or past where it ends, whatever the launch is"),
        ),
        case(
            "",
            "y: *mut f32, n: u32",
            &below_n(&wide),
            "[at]);",
            cannot("`ld.global.v4.f32` reaches 16 bytes at once, more than the 4 of what its pointer is to"),
        ),
        // Part of an element, which the host may store whole, by its atomics, while the kernel runs: a scalar of fewer
        // bytes than the element, loaded or stored; and a vector of such scalars, which PTX takes as an access for
        // each of them.
        case(
            "",
            "x: *const u32, n: u32",
            &below_n(&part_load),
            "[at + 2]",
            cannot("`ld.global.u16` reaches part of an element, 2 of the 4 bytes of what its pointer is to"),
        ),
        case(
            "",
            "x: *const u32, n: u32",
            &below_n(&byte_load),
            "[at + 1]",
            cannot("`ld.global.u8` reaches part of an element, 1 of the 4 bytes of what its pointer is to"),
        ),
        case(
            "",
            "y: *mut u32, n: u32",
            &below_n(&part_store),
            "[at + 2]",
            cannot("`st.global.u16` reaches part of an element, 2 of the 4 bytes of what its pointer is to"),
        ),
        case(
            "",
            "x: *const u32, n: u32",
            &below_n(&lanes),
            "[at]);",
            cannot("`ld.global.v2.u16` reaches part of an element, 2 of the 4 bytes of what its pointer is to"),
        ),
        case(
            "",
            "x: *const u64, n: u32",
            &below_n(&half),
            "[at + 4]",
            cannot("`ld.global.u32` reaches part of an element, 4 of the 8 bytes of what its pointer is to"),
        ),
        case(
            "",
            "y: *const f32, n: u32",
            &below_n(&[&own[2..], &one, &store].concat()),
            "[at], one",
            cannot("`st.global.f32` writes through a `*const` pointer, memory lent to be read"),
        ),
    ];
    for (src, needle, error) in cases {
        let want = format!("{}: {error}", place(&src, needle)?);
        assert_eq!(requires(&src, "k").err().ok_or("requires gave Ok")?, want, "{src}");
        // As an `unsafe fn` the kernel is what it is without this, and a launch holds nothing for it.
        assert_eq!(requires(&trusted(&src), "k")?, Vec::<String>::new(), "{src}");
    }
    Ok(())
}

/// The proof works with factors of 127 bits and a sign, and follows no number with a factor past them: `i128::MIN`
/// is one, for its negation is. A kernel that makes the threads of a block and the index of a thread each -2^126 times
/// as many, and adds them, makes a sum whose least over a launch, `-2^127 * ntid.x + 2^126`, has the factor -2^127:
/// kernite does not follow the sum, and an address made of it is one it cannot follow, an error at the address, as for
/// any number it does not follow. With -2^125 it follows the sum. A kernel that does nothing with the sum compiles, and
/// so does the `unsafe fn` that it is, as it always did.
#[test]
fn the_proof_does_not_follow_a_number_with_a_factor_past_127_bits() -> Result {
    // A kernel that works out `w`, `-2^power` times the threads of a block and the index of a thread, and takes `y`
    // that far on, to store at, if it `uses` it.
    let src = |power: u32, uses: bool| {
        let mut lines =
            ["let (n32, t32): (u32, u32);", "mov.u32(n32, ntid.x);", "mov.u32(t32, tid.x);"].map(String::from).to_vec();
        lines.extend(
            ["let (n, t, a, b, w): (u64, u64, u64, u64, u64);", "cvt.u64.u32(n, n32);", "cvt.u64.u32(t, t32);"]
                .map(String::from),
        );
        for (name, from) in [("a", "n"), ("b", "t")] {
            lines.push(format!("shl.b64({name}, {from}, {});", power - 63));
            lines.push(format!("mul.lo.u64({name}, {name}, 9223372036854775808);"));
            lines.push(format!("sub.u64({name}, 0, {name});"));
        }
        lines.push("add.u64(w, a, b);".to_string());
        if uses {
            lines.extend(
                ["let (at, one): (u64, f32);", "add.u64(at, y, w);", "mov.f32(one, 1.0);", "st.global.f32([at], one);"]
                    .map(String::from),
            );
        }
        kernel("y: *mut f32", &lines.iter().map(String::as_str).collect::<Vec<_>>())
    };
    let store = "`st.global.f32` reaches";
    let cases = [
        (126, "memory at an address that kernite cannot follow to a pointer of the kernel's"),
        (125, "an element at an index that no compare keeps below the memory's length"),
    ];
    for (power, error) in cases {
        let src = src(power, true);
        let want = format!("{}: {}", place(&src, "[at], one")?, cannot(&format!("{store} {error}")));
        assert_eq!(requires(&src, "k").err().ok_or("requires gave Ok")?, want, "2^{power}\n{src}");
    }
    for power in [125, 126] {
        let src = src(power, false);
        requires(&src, "k").map_err(|e| format!("2^{power}: {e}"))?;
        requires(&trusted(&src), "k").map_err(|e| format!("2^{power}, unsafe: {e}"))?;
    }
    Ok(())
}

/// What a launch holds, of each shape of a kernel that kernite proves safe: one thread that writes a place of its own;
/// a thread for each row, which reads and writes the second float of its two; and a block for each row of 16 and a
/// thread for each element of it. Each is a sum that is not negative: the memory is as long as the kernel reaches, the
/// threads or the blocks are as few as the part of the memory each has for itself allows, and no number wraps.
#[test]
fn a_launch_holds_what_each_shape_of_a_safe_kernel_reaches() -> Result {
    let holds = |params: &str, lines: &[&str]| -> Result<Vec<String>> {
        let mut holds = requires(&kernel(params, lines), "k")?;
        holds.sort();
        Ok(holds)
    };
    // One thread of one block writes the float at `y`, which has one.
    assert_eq!(
        holds("y: *mut f32", &["let one: f32;", "mov.f32(one, 1.0);", "st.global.f32([y], one);"])?,
        ["-1*Blocks + 1 >= 0 (Apart(0))", "-1*Threads + 1 >= 0 (Apart(0))", "1*Len(0) + -1 >= 0 (Inside(0))",]
    );
    // A thread for each row of 2 floats, from row `t0`, in one block: the second of each. `x` and `y` have as many rows
    // as `t0` and the threads.
    let rows = [
        "let (t, at_x, at_y): (u32, u64, u64);",
        "mov.u32(t, tid.x);",
        "add.u32(t, t, t0);",
        "mad.wide.u32(at_x, t, 8, x);",
        "mad.wide.u32(at_y, t, 8, y);",
        "let v: f32;",
        "ld.global.f32(v, [at_x + 4]);",
        "st.global.f32([at_y + 4], v);",
    ];
    assert_eq!(
        holds("y: *mut f32, x: *const f32, t0: u32", &rows)?,
        [
            "-1*Blocks + 1 >= 0 (Apart(0))",
            "-1*Threads + -1*Param(2) + 4294967296 >= 0 (Fits)",
            "-2*Threads + -2*Param(2) + 1*Len(0) >= 0 (Inside(0))",
            "-2*Threads + -2*Param(2) + 1*Len(1) >= 0 (Inside(1))",
        ]
    );
    // A block for each row of 16 floats and a thread for each, at most 16 in a block.
    let blocks = [
        "let (block, thread, i): (u32, u32, u32);",
        "mov.u32(block, ctaid.x);",
        "mov.u32(thread, tid.x);",
        "mad.lo.u32(i, block, 16, thread);",
        "let (at_x, at_y): (u64, u64);",
        "mad.wide.u32(at_x, i, 4, x);",
        "mad.wide.u32(at_y, i, 4, y);",
        "let v: f32;",
        "ld.global.f32(v, [at_x]);",
        "st.global.f32([at_y], v);",
    ];
    assert_eq!(
        holds("y: *mut f32, x: *const f32", &blocks)?,
        [
            "-1*Threads + -16*Blocks + 1*Len(0) + 16 >= 0 (Inside(0))",
            "-1*Threads + -16*Blocks + 1*Len(1) + 16 >= 0 (Inside(1))",
            "-1*Threads + -16*Blocks + 4294967312 >= 0 (Fits)",
            "-1*Threads + 16 >= 0 (Apart(0))",
        ]
    );
    Ok(())
}

/// What the proof follows of the numbers a thread works out, as it follows `mad`: a sum and a difference, a product
/// and a shift, of 32 bits or of 64, and an end by `exit` or `trap`, as by `ret`. Each kernel here writes its thread's
/// float of `y` alone, and so holds what the kernel that does by `mad.wide` holds. A number of 32 bits that is more
/// than the index adds a condition that no thread wraps it: the block's index times its size, that the blocks less one
/// times the threads of one are fewer than 2^32, and the index shifted left by 2, that the grid has 2^30 threads or
/// fewer.
#[test]
fn the_proof_follows_the_sums_products_and_shifts_of_a_threads_numbers() -> Result {
    let holds = |lines: &[&str]| -> Result<Vec<String>> {
        let mut holds = requires(&module("", &kernel("y: *mut f32, n: u32", lines)), "k")?;
        holds.sort();
        Ok(holds)
    };
    let to = |at: &[&'static str]| -> Vec<&'static str> {
        [&BELOW_N[..], &["let (j, k, w, at): (u32, u32, u64, u64);"], at, &["st.global.f32([at], 1.0);"]].concat()
    };
    let plain = holds(&to(&["mad.wide.u32(at, i, 4, y);"]))?;
    assert_eq!(plain, ["-1*Param(1) + 1*Len(0) >= 0 (Inside(0))"]);
    // A sum and a difference that give the index back.
    assert_eq!(holds(&to(&["add.u32(j, i, 1);", "sub.u32(k, j, 1);", "mad.wide.u32(at, k, 4, y);"]))?, plain);
    // A product of 32 bits, that of the block's index and its size, and the index it adds to.
    let by_mul = [
        "let (block, size, thread): (u32, u32, u32);",
        "mov.u32(block, ctaid.x);",
        "mov.u32(size, ntid.x);",
        "mov.u32(thread, tid.x);",
        "mul.lo.u32(j, block, size);",
        "add.u32(k, j, thread);",
        "mad.wide.u32(at, k, 4, y);",
    ];
    assert_eq!(
        holds(&to(&by_mul))?,
        ["-1*Param(1) + 1*Len(0) >= 0 (Inside(0))", "1*Threads + -1*Threads*Blocks + 4294967295 >= 0 (Fits)"]
    );
    // A product of 64 bits, and the sum of the address and it.
    assert_eq!(holds(&to(&["mul.wide.u32(w, i, 4);", "add.u64(at, y, w);"]))?, plain);
    // A shift, which is the product by 4, in 32 bits or in 64.
    let by_shl = ["shl.b32(j, i, 2);", "cvt.u64.u32(w, j);", "add.u64(at, y, w);"];
    assert_eq!(
        holds(&to(&by_shl))?,
        ["-1*Param(1) + 1*Len(0) >= 0 (Inside(0))", "-1*Threads*Blocks + 1073741824 >= 0 (Fits)"]
    );
    let by_shl = ["cvt.u64.u32(w, i);", "shl.b64(w, w, 2);", "add.u64(at, y, w);"];
    assert_eq!(holds(&to(&by_shl))?, plain);
    // A thread that ends by `exit` or by `trap` is one that has gone on past where `i < n`, as one that ends by `ret`.
    for end in ["exit", "trap"] {
        let src = module("", &kernel("y: *mut f32, n: u32", &to(&["mad.wide.u32(at, i, 4, y);"])));
        let src = src.replace("ret()", &format!("{end}()"));
        assert_eq!(requires(&src, "k")?, plain, "{end}");
    }
    Ok(())
}
