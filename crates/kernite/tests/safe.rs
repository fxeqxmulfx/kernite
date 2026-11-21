//! A kernel that is no `unsafe fn`: an x86 kernel of values alone, or of arrays given by reference, whose every address
//! kernite checks as it does stack memory. What kernite cannot prove safe is refused, at the first place that is not,
//! with a word of what to do: the kernel is an `unsafe fn`, which compiles as it does without this.

type Result<T = (), E = Box<dyn std::error::Error + Send + Sync>> = std::result::Result<T, E>;

/// Kernel `k` of `src` as x86, the code of its listing, or the error.
fn code(src: &str) -> Result<Vec<String>, String> {
    kernite::x86_str(src, "k").map(|listing| listing.code().map(str::to_string).collect())
}

/// Kernel `name` of `src` as x86, the code of its listing, or the error.
fn code_of(src: &str, name: &str) -> Result<Vec<String>, String> {
    kernite::x86_str(src, name).map(|listing| listing.code().map(str::to_string).collect())
}

/// The line of the head of kernel `name`'s listing that says what it needs.
fn needs(src: &str, name: &str) -> Result<String, String> {
    let listing = kernite::x86_str(src, name)?.to_string();
    Ok(listing.lines().find(|line| line.starts_with("# needs:")).unwrap_or_default().to_string())
}

/// `src` with `unsafe` before each `extern`: its kernels are `unsafe fn`s.
fn trusted(src: &str) -> String {
    src.replace("extern \"sysv64\"", "unsafe extern \"sysv64\"")
}

/// The helpers that `sum8` of the examples calls.
const LANES: &str = r#"
fn ret_sum(v: ymm) {
    let sum: ymm;
    add_lanes(sum, v);
    ret(sum.xmm);
}

fn add_lanes(sum: ymm, v: ymm) {
    note("the lanes");
    let high: ymm;
    vextractf128(high.xmm, v, 1);
    vaddps(sum.xmm, v.xmm, high.xmm);
    let pair: ymm;
    vmovhlps(pair.xmm, sum.xmm, sum.xmm);
    vaddps(sum.xmm, sum.xmm, pair.xmm);
    vmovshdup(pair.xmm, sum.xmm);
    vaddss(sum.xmm, sum.xmm, pair.xmm);
}
"#;

/// A kernel of values alone, and one that calls itself, as the examples' `gcd` does, written with no `unsafe`, are
/// what they are with it: the same code in the same registers.
#[test]
fn a_kernel_of_values_is_what_it_is_with_unsafe() -> Result {
    let src = r#"
pub extern "sysv64" fn k(a: u64, b: u64) -> u64 {
    test(b, b);
    jnz(more);
    ret(a);
    label(more);
    sub(a, b);
    call(a, k, (a, b));
    ret(a);
}
"#;
    let safe = kernite::x86_str(src, "k")?.to_string();
    assert_eq!(safe, kernite::x86_str(&trusted(src), "k")?.to_string());
    assert_eq!(safe.lines().next(), Some("# k(gpr, gpr) -> gpr"));
    Ok(())
}

/// A kernel gives back a number, a pointer or nothing, as a value is no memory: it is what it is with `unsafe`. Only a
/// `bool` is refused, below.
#[test]
fn a_kernel_gives_back_a_number_a_pointer_or_nothing() -> Result {
    for (param, result) in [("u64", "u64"), ("f32", "f32"), ("u64", "*const u8"), ("u64", "")] {
        let (arrow, ret) =
            if result.is_empty() { (String::new(), "ret();") } else { (format!(" -> {result}"), "ret(x);") };
        let src = format!("pub extern \"sysv64\" fn k(x: {param}){arrow} {{ {ret} }}");
        assert_eq!(
            kernite::x86_str(&src, "k")?.to_string(),
            kernite::x86_str(&trusted(&src), "k")?.to_string(),
            "{src}"
        );
    }
    Ok(())
}

/// An array given by reference is memory kernite checks: `sum8` and `sum16`, which calls it at the array and at its
/// second half, and a kernel that writes the array it is given, compile as they do with pointers, byte for byte.
#[test]
fn a_kernel_of_arrays_by_reference_is_what_it_is_with_pointers() -> Result {
    let arrays = format!(
        r#"
pub extern "sysv64" fn sum8(xs: &[f32; 8]) -> f32 {{
    let v: ymm;
    vmovups(v, [xs]);
    ret_sum(v);
}}

pub extern "sysv64" fn sum16(xs: &[f32; 16]) -> f32 {{
    let (low, high): (ymm, ymm);
    call(low.xmm, sum8, [xs]);
    call(high.xmm, sum8, [xs + 32]);
    vaddss(high.xmm, low.xmm, high.xmm);
    ret(high.xmm);
}}

pub extern "sysv64" fn fill(xs: &mut [f32; 8]) {{
    let v: ymm;
    vbroadcastss(v, [1.5]);
    vmovups([xs], v);
}}
{LANES}"#
    );
    let pointers = format!(
        r#"
pub unsafe extern "sysv64" fn sum8(xs: *const f32) -> f32 {{
    let v: ymm;
    vmovups(v, [xs]);
    ret_sum(v);
}}

pub unsafe extern "sysv64" fn sum16(xs: *const f32) -> f32 {{
    let (low, high): (ymm, ymm);
    call(low.xmm, sum8, xs);
    add(xs, 32);
    call(high.xmm, sum8, xs);
    vaddss(high.xmm, low.xmm, high.xmm);
    ret(high.xmm);
}}

pub unsafe extern "sysv64" fn fill(xs: *mut f32) {{
    let v: ymm;
    vbroadcastss(v, [1.5]);
    vmovups([xs], v);
}}
{LANES}"#
    );
    for name in ["sum8", "sum16", "fill"] {
        let listing = kernite::x86_str(&arrays, name)?.to_string();
        assert_eq!(listing, kernite::x86_str(&pointers, name)?.to_string(), "{name}");
        // An `unsafe fn` takes an array by reference as well.
        assert_eq!(listing, kernite::x86_str(&trusted(&arrays), name)?.to_string(), "{name}");
    }
    Ok(())
}

/// kernite checks an address in an array where the index is 0, as in stack memory: what it reaches lies inside the
/// array, aligned as the instruction wants, and the array is written only if it is `&mut`. A kernel that is not an
/// `unsafe fn` has no register in an address, so every address is checked whole.
#[test]
fn an_address_in_an_array_is_checked_as_one_in_stack_memory() -> Result {
    let written = "a kernel only reads what it is given by `&`, and writes what it is given by `&mut`";
    let cases = [
        (
            "xs: &[f32; 8]",
            "let v: ymm; vmovups(v, [xs + 4]);",
            "2:24: `vmovups` reads 32 bytes at `[xs + 4]`, past the end of `xs`, 32 bytes".to_string(),
        ),
        (
            "xs: &[f32; 8]",
            "let v: ymm; vmovups(v, [xs + 32]);",
            "2:24: `vmovups` reads 32 bytes at `[xs + 32]`, past the end of `xs`, 32 bytes".to_string(),
        ),
        (
            "xs: &[f32; 8]",
            "let v: ymm; vmovups(v, [xs - 4]);",
            "2:24: `vmovups` reads at `[xs - 4]`, before `xs` starts".to_string(),
        ),
        (
            "xs: &[f32; 8]",
            "let v: ymm; vmovaps(v, [xs]);",
            "2:24: `vmovaps` reads 32 bytes at once, at an address that is a multiple of 32; `xs` is at a multiple of \
             4: `vmovups` at any"
                .to_string(),
        ),
        (
            "xs: &[f32; 16]",
            "let v: xmm; vmovaps(v, [xs + 8]);",
            "2:24: `vmovaps` reads 16 bytes at once, at an address that is a multiple of 16; `xs` is at a multiple of \
             4: `vmovups` at any"
                .to_string(),
        ),
        (
            "xs: &[f32; 8]",
            "let v: ymm; zero(v); vmovups([xs], v);",
            format!("2:30: `vmovups` writes `xs`, which is `&[f32; 8]`: {written}"),
        ),
        (
            "xs: &[f32; 8]",
            "add(dword[xs + 4], 1);",
            format!("2:10: `add` reads and writes `xs`, which is `&[f32; 8]`: {written}"),
        ),
    ];
    for (params, body, want) in cases {
        let src = format!("pub extern \"sysv64\" fn k({params}) {{\n{body}\n}}\n");
        assert_eq!(code(&src).err().ok_or("code gave Ok")?, want, "{body}");
        // The same is true of an `unsafe fn`, where the index is the kernel's to keep inside.
        assert_eq!(code(&trusted(&src)).err().ok_or("code gave Ok")?, want, "unsafe {body}");
    }
    // What lies inside compiles, at any multiple of the element's bytes; `&mut` is written as well as read, and the
    // length of an array is the memory it holds, as its elements are.
    let src = r#"
pub extern "sysv64" fn k(xs: &mut [f32; 16], ys: &[u8; 4], zs: &mut [u64; 2]) {
    let v: ymm;
    vmovups(v, [xs + 32]);
    vmovups([xs + 28], v.xmm);
    let g: gpr;
    movzx(g.r32, byte[ys + 3]);
    mov([zs + 8], g);
}
"#;
    let want = [
        "vmovups ymm0, ymmword ptr [rdi + 32]",
        "vmovups xmmword ptr [rdi + 28], xmm0",
        "movzx eax, byte ptr [rsi + 3]",
        "mov qword ptr [rdx + 8], rax",
        "vzeroupper",
        "ret",
    ];
    assert_eq!(code(src)?, want);
    Ok(())
}

/// Functions that the kernels of a test call: one reads an array, one writes it, one is an `unsafe fn`.
const CALLEES: &str = r#"
pub extern "sysv64" fn eight(x: &[f32; 8]) -> f32 {
    let v: ymm;
    vmovups(v, [x]);
    ret(v.xmm);
}

pub extern "sysv64" fn put(x: &mut [f32; 8]) {
    let v: ymm;
    zero(v);
    vmovups([x], v);
}

pub unsafe extern "sysv64" fn danger(x: *const f32) -> f32 {
    let v: ymm;
    vmovups(v, [x]);
    ret(v.xmm);
}
"#;

/// A call passes an array the kernel is given as it is, `[xs]`, or a part of it, `[xs + 32]`, to a kernel that takes
/// an array by reference; that array lies inside the one passed, at a multiple of its elements' bytes, and is written
/// only if the kernel has `&mut` of what it passes it from. kernite says where the call is wrong, as it does of an
/// instruction.
#[test]
fn a_call_passes_an_array_or_a_part_of_it() -> Result {
    let cases = [
        (
            "xs: &[f32; 16]",
            "call(_, eight, [xs + 36]);",
            "2:16: `eight` takes a `&[f32; 8]`, 32 bytes, at `[xs + 36]`, past the end of `xs`, 64 bytes",
        ),
        (
            "xs: &[f32; 16]",
            "call(_, eight, [xs - 4]);",
            "2:16: `eight` takes a `&[f32; 8]` at `[xs - 4]`, before `xs` starts",
        ),
        (
            "xs: &[f32; 16]",
            "call(_, eight, [xs + 2]);",
            "2:16: `eight` takes a `&[f32; 8]`, at a multiple of 4; `[xs + 2]` is not one",
        ),
        (
            "xs: &[f32; 4]",
            "call(_, eight, [xs]);",
            "2:16: `eight` takes a `&[f32; 8]`, 32 bytes, at `[xs]`, past the end of `xs`, 16 bytes",
        ),
        (
            "xs: &[u8; 64]",
            "call(_, eight, [xs + 4]);",
            "2:16: `eight` takes a `&[f32; 8]`, at a multiple of 4; `xs` is at a multiple of 1",
        ),
        (
            "xs: &[f32; 16]",
            "call(put, [xs]);",
            "2:11: `put` writes through a `&mut [f32; 8]`, and `xs` is a `&[f32; 16]`: a kernel only reads what it is \
             given by `&`",
        ),
        (
            "xs: &[f32; 16]",
            "let p: gpr; call(_, eight, p);",
            "2:28: `eight` takes a `&[f32; 8]`: an array the kernel is given, as `[xs]`, or part of it, as \
             `[xs + 32]`; this is a gpr",
        ),
        (
            "xs: &[f32; 16]",
            "call(_, eight, 5);",
            "2:16: `eight` takes a `&[f32; 8]`: an array the kernel is given, as `[xs]`, or part of it, as \
             `[xs + 32]`; this is the integer 5",
        ),
        (
            "xs: &[f32; 16]",
            "call(_, eight, xs);",
            "2:16: `eight` takes a `&[f32; 8]`: an array the kernel is given, as `[xs]`, or part of it, as \
             `[xs + 32]`; this is an array",
        ),
        (
            "",
            "let b: local<[f32; 8], 32>; call(_, eight, [b]);",
            "2:44: `eight` takes a `&[f32; 8]`: an array the kernel is given, as `[xs]`, or part of it, as \
             `[xs + 32]`; this is memory",
        ),
    ];
    for (params, body, want) in cases {
        let src = format!("pub extern \"sysv64\" fn k({params}) {{\n{body}\n}}\n{CALLEES}");
        assert_eq!(code(&src).err().ok_or("code gave Ok")?, want, "{body}");
    }
    // What lies inside is a call at the array's address plus the displacement, `eight` at the array's end included.
    let src = format!(
        "pub extern \"sysv64\" fn k(xs: &mut [f32; 16], ys: &mut [u64; 8]) {{\n\
         call(_, eight, [xs]);\ncall(_, eight, [xs + 32]);\ncall(put, [xs + 32]);\ncall(put, [ys + 32]);\n}}\n{CALLEES}"
    );
    assert_eq!(code(&src)?.len(), 12);
    Ok(())
}

/// Kernels whose array's length is a const's, `&[u64; L]`, or a compile-time argument's, `&[u64; N]`: `sum` is of two
/// instances, and the one of four calls the one of two on each half of its array.
const SIZED: &str = r#"
const L: usize = 2;

#[instances(4, 2)]
pub extern "sysv64" fn sum<const N: usize>(xs: &[u64; N]) -> u64 {
    let (a, b): (gpr, gpr);
    if N == 2 {
        mov(a, [xs]);
        add(a, [xs + 8]);
        ret(a);
    } else {
        call(a, sum::<2>, [xs]);
        call(b, sum::<2>, [xs + 16]);
        add(a, b);
        ret(a);
    }
}

pub extern "sysv64" fn first(xs: &[u64; L]) -> u64 {
    let g: gpr;
    mov(g, [xs]);
    ret(g);
}
"#;

/// A kernel whose array's length is a const's or a compile-time argument's is called as one whose length is written is:
/// each of its instances takes an array of the length it names, `sum::<2>` a `&[u64; 2]`, and a call is checked against
/// it. So a kernel calls itself, at another instance, on the halves of its array.
#[test]
fn a_kernel_of_an_array_whose_length_is_not_written_is_called_as_any_other_is() -> Result {
    let kernel = |body: &str| {
        format!("pub extern \"sysv64\" fn k(xs: &[u64; 4]) -> u64 {{\nlet g: gpr;\n{body}\nret(g);\n}}\n{SIZED}")
    };
    let src = kernel("call(g, first, [xs + 16]);");
    assert_eq!(
        code_of(&src, "sum::<4>")?,
        [
            "sub rsp, 8",
            "call {f0}",
            "mov rcx, rax",
            "add rdi, 16",
            "call {f0}",
            "add rcx, rax",
            "mov rax, rcx",
            "add rsp, 8",
            "ret"
        ]
    );
    assert_eq!(code_of(&src, "sum::<2>")?, ["mov rax, qword ptr [rdi]", "add rax, qword ptr [rdi + 8]", "ret"]);
    for body in ["call(g, first, [xs + 16]);", "call(g, sum::<2>, [xs + 8]);", "call(g, sum::<4>, [xs]);"] {
        code(&kernel(body))?;
    }
    let cases = [
        (
            "call(g, first, [xs + 24]);",
            "3:16: `first` takes a `&[u64; 2]`, 16 bytes, at `[xs + 24]`, past the end of `xs`, 32 bytes",
        ),
        (
            "call(g, sum::<2>, [xs + 24]);",
            "3:19: `sum::<2>` takes a `&[u64; 2]`, 16 bytes, at `[xs + 24]`, past the end of `xs`, 32 bytes",
        ),
        (
            "call(g, sum::<4>, [xs + 8]);",
            "3:19: `sum::<4>` takes a `&[u64; 4]`, 32 bytes, at `[xs + 8]`, past the end of `xs`, 32 bytes",
        ),
    ];
    for (body, want) in cases {
        assert_eq!(code(&kernel(body)).err().ok_or("code gave Ok")?, want, "{body}");
    }
    Ok(())
}

/// What kernite cannot prove safe, a kernel that is not an `unsafe fn` is refused, at the first place that is not: a
/// pointer it takes, a `bool` it gives back, an array it reaches at an index in a register, a bit of one at an offset
/// in a register, an address it puts in a register, a non-temporal store it may return after with no `sfence` between,
/// a static of the module's, a function it calls that is an `unsafe fn`, or one at an address. The error says the
/// kernel is then an `unsafe fn`, and as one it compiles, as it does without all this.
#[test]
fn what_kernite_cannot_prove_safe_is_refused_unless_the_kernel_is_unsafe() -> Result {
    let kernel = |params: &str, ret: &str, body: &str| {
        format!("pub extern \"sysv64\" fn k({params}){ret} {{\n{body}\n}}\n{CALLEES}")
    };
    let cases = [
        (
            kernel("p: *const f32", "", ""),
            "1:29: a pointer may point anywhere, so a kernel that takes `*const f32` is an `unsafe fn`; kernite checks \
             an array by reference, `&[f32; 8]`, and a slice, `&[f32]`",
        ),
        (
            kernel("a: u64, p: *mut f32", "", ""),
            "1:37: a pointer may point anywhere, so a kernel that takes `*mut f32` is an `unsafe fn`; kernite checks \
             an array by reference, `&[f32; 8]`, and a slice, `&[f32]`",
        ),
        (
            kernel("", " -> bool", "let g: gpr;\nmov(g, 1);\nret(g);"),
            "1:31: a `bool` is 0 or 1, which kernite cannot prove a kernel gives back, so a kernel that gives one back \
             is an `unsafe fn`",
        ),
        (
            kernel("xs: &[f32; 8], i: u64", "", "let v: ymm;\nvmovups(v, [xs + i * 4]);"),
            "3:12: `vmovups` reaches `[xs + i * 4]`, at an index in a register, which kernite cannot keep inside `xs`, \
             so a kernel that does is an `unsafe fn`",
        ),
        (
            kernel("i: u64", "", "let b: local<[u64; 4], 32>;\nmov([b + i * 8], i);"),
            "3:5: `mov` reaches `[b + i * 8]`, at an index in a register, which kernite cannot keep inside `b`, so a \
             kernel that does is an `unsafe fn`",
        ),
        (
            kernel("xs: &[f32; 8]", "", "let g: gpr;\nlea(g, [xs + 8]);"),
            "3:8: `lea` puts the address of `xs` in a register, where kernite cannot check what it reaches, so a \
             kernel that does is an `unsafe fn`",
        ),
        (
            kernel("xs: &mut [u64; 4], i: u64", "", "bts([xs], i);"),
            "2:11: `bts` reaches a bit at the offset in `i`, which kernite cannot keep inside `xs`, so a kernel that \
             does is an `unsafe fn`",
        ),
        (
            kernel("i: u64", "", "let b: local<u64>;\nlock.btr(qword[b], i);"),
            "3:20: `lock.btr` reaches a bit at the offset in `i`, which kernite cannot keep inside `b`, so a kernel \
             that does is an `unsafe fn`",
        ),
        (
            kernel("xs: &mut [u64; 4], g: u64", "", "movnti([xs], g);"),
            "2:1: `movnti` is a non-temporal store, and the kernel may return before an `sfence` orders it, so a \
             kernel that does is an `unsafe fn`",
        ),
        (
            kernel("xs: &[f32; 8]", "", "let g: gpr;\nmov(g, 0);\nlet v: ymm;\nvmovups(v, [g]);"),
            "5:12: `vmovups` reaches memory at the address in `g`, which kernite cannot check, so a kernel that does \
             is an `unsafe fn`",
        ),
        (
            kernel("", "", "call(_, danger, 0);"),
            "2:9: `danger` is an `unsafe fn`, so a kernel that calls it is an `unsafe fn`",
        ),
        (
            kernel("", "", "let fp: gpr;\nlea(fp, [eight]);\ncall(_, fp, ());"),
            "4:9: `call` at the address in a gpr reaches a function kernite cannot name, so a kernel that does is an \
             `unsafe fn`",
        ),
        (
            format!("static S: global<u64>;\n{}", kernel("", "", "let g: gpr;\nmov(g, [S]);")),
            "4:9: `S` is memory of the module's, which every kernel and the host reach, so a kernel that names it is \
             an `unsafe fn`",
        ),
    ];
    for (src, want) in cases {
        assert_eq!(code(&src).err().ok_or("code gave Ok")?, want, "{src}");
        // As an `unsafe fn` the kernel is what it is without this: only the kernel is one, its callees are not.
        let unsafe_k = src.replacen("extern", "unsafe extern", 1);
        code(&unsafe_k).map_err(|e| format!("{e}\n{unsafe_k}"))?;
    }
    // A static the kernel does not name is no matter to it.
    let src = format!("static S: global<u64>;\n{}", kernel("a: u64", "", ""));
    code(&src)?;
    Ok(())
}

/// A bit at an offset in a register lies as far from the memory it tests as 2^60 bytes, which kernite cannot keep
/// inside an array or stack memory, so a safe kernel has none. A bit at an offset known now lies in the operand, as the
/// forms have it, and a bit of a register is no memory: a safe kernel takes both.
#[test]
fn a_bit_at_an_offset_known_now_is_safe_and_one_in_a_register_is_not() -> Result {
    let src = r#"
pub extern "sysv64" fn k(xs: &mut [u64; 4], a: u64, i: u64) -> u64 {
    bt(qword[xs + 8], 3);
    lock.bts(qword[xs], 63);
    btr(qword[xs + 24], 0);
    bts(a, i);
    bt(a, i);
    ret(a);
}
"#;
    let want = [
        "bt qword ptr [rdi + 8], 3",
        "lock bts qword ptr [rdi], 63",
        "btr qword ptr [rdi + 24], 0",
        "bts rsi, rdx",
        "bt rsi, rdx",
        "mov rax, rsi",
        "ret",
    ];
    assert_eq!(code(src)?, want);
    // Each of them at an offset in a register is refused, locked or not. As an `unsafe fn` it takes the bit, which is
    // the kernel's to keep inside the array.
    for name in ["bt", "btc", "btr", "bts", "lock.btc", "lock.btr", "lock.bts"] {
        let src = format!("pub extern \"sysv64\" fn k(xs: &mut [u64; 4], i: u64) {{\n{name}([xs], i);\n}}\n");
        let want = format!(
            "2:{}: `{name}` reaches a bit at the offset in `i`, which kernite cannot keep inside `xs`, so a kernel \
             that does is an `unsafe fn`",
            name.len() + 8
        );
        assert_eq!(code(&src).err().ok_or("code gave Ok")?, want, "{name}");
        code(&trusted(&src))?;
    }
    Ok(())
}

/// A non-temporal store goes around the caches, and x86 lets a later store pass it, so a store the caller may see after
/// what it does next races with it. A safe kernel has an `sfence` or an `mfence` after each such store on every way to
/// a `ret`: kernite refuses the kernel where one may not, at the store, and takes it where one does.
#[test]
fn a_non_temporal_store_is_ordered_before_a_safe_kernel_returns() -> Result {
    let kernel = |params: &str, body: &str| format!("pub extern \"sysv64\" fn k({params}) {{\n{body}\n}}\n");
    let arrays = "xs: &mut [u64; 4], g: u64";
    let ordered = [
        (arrays, "movnti([xs], g);\nsfence();"),
        (arrays, "movnti([xs], g);\nmfence();"),
        ("", "let b: local<[f32; 8], 32>;\nlet v: ymm;\nzero(v);\nvmovntps([b], v);\nmfence();"),
        // A store in a loop, ordered once it ends; and one ordered on each of two ways.
        (arrays, "label(top);\nmovnti([xs], g);\nsub(g, 1);\njnz(top);\nsfence();"),
        (arrays, "movnti([xs], g);\ntest(g, g);\njz(other);\nsfence();\nret();\nlabel(other);\nmfence();"),
        // Stores that are no non-temporal ones need none.
        (arrays, "mov([xs], g);"),
    ];
    for (params, body) in ordered {
        let src = kernel(params, body);
        code(&src).map_err(|e| format!("{e}\n{src}"))?;
    }
    let said = "`movnti` is a non-temporal store, and the kernel may return before an `sfence` orders it, so a kernel \
                that does is an `unsafe fn`";
    let refused = [
        // An `sfence` on one way to the `ret` and not on the other.
        (arrays, "movnti([xs], g);\ntest(g, g);\njz(skip);\nsfence();\nlabel(skip);", format!("2:1: {said}")),
        // An `sfence` before the store alone.
        (arrays, "sfence();\nmovnti([xs], g);", format!("3:1: {said}")),
        // The first of two stores that a way to a `ret` leaves unordered.
        (arrays, "movnti([xs], g);\nsfence();\nmovnti([xs + 8], g);\nmovnti([xs + 16], g);", format!("4:1: {said}")),
    ];
    for (params, body, want) in refused {
        let src = kernel(params, body);
        assert_eq!(code(&src).err().ok_or("code gave Ok")?, want, "{body}");
        // As an `unsafe fn` it is the kernel's to order.
        code(&trusted(&src))?;
    }
    // A store of a vector is refused as `movnti` is, of each type of element.
    for (elem, store) in [("f32", "vmovntps"), ("f64", "vmovntpd"), ("u64", "vmovntdq")] {
        let src = kernel("", &format!("let b: local<[{elem}; 8], 32>;\nlet v: ymm;\nzero(v);\n{store}([b], v);"));
        let want = format!("5:1: {}", said.replacen("movnti", store, 1));
        assert_eq!(code(&src).err().ok_or("code gave Ok")?, want, "{store}");
        code(&trusted(&src))?;
    }
    Ok(())
}

/// An array by reference holds Rust's numbers, as stack memory and a static do, and is at most as many bytes as a
/// displacement of 32 bits, signed, reaches from its address; an array of none is taken, though a static or stack
/// memory of none is refused: a kernel is given one, and each instruction that reaches it is refused. What kernite
/// refuses of an array it says, where the type is, and an `unsafe fn` is refused the same: the array's type is wrong,
/// not what the kernel does.
#[test]
fn an_array_kernite_refuses_is_said_why() -> Result {
    let numbers = "an array by reference holds Rust's numbers, `u8` to `u64`, `i8` to `i64`, `usize`, `isize`, `f32` \
                   and `f64`";
    let cases = [
        ("&[bool; 8]", format!("1:30: {numbers}; not `bool`")),
        ("&mut [bool; 8]", format!("1:30: {numbers}; not `bool`")),
        ("&[[u64; 2]; 2]", format!("1:30: {numbers}; not `[u64; 2]`")),
        ("&[*const u8; 2]", format!("1:30: {numbers}; not `*const u8`")),
        (
            "&[u8; 2147483648]",
            "1:30: `&[u8; 2147483648]` is more than 2147483647 bytes: a kernel reaches an array by reference at a \
             displacement from its address, of 32 bits, signed"
                .to_string(),
        ),
        (
            "&mut [u64; 268435456]",
            "1:30: `&mut [u64; 268435456]` is more than 2147483647 bytes: a kernel reaches an array by reference at a \
             displacement from its address, of 32 bits, signed"
                .to_string(),
        ),
    ];
    for (ty, want) in cases {
        let src = format!("pub extern \"sysv64\" fn k(xs: {ty}) {{\n}}\n");
        assert_eq!(code(&src).err().ok_or("code gave Ok")?, want, "{ty}");
        // `unsafe ` moves the type 7 columns.
        assert_eq!(code(&trusted(&src)).err().ok_or("code gave Ok")?, want.replacen("1:30", "1:37", 1), "unsafe {ty}");
    }
    // The most bytes there are, and none, compile; and what an array of none holds is nothing.
    for ty in ["&[u8; 2147483647]", "&mut [u64; 268435455]", "&[f32; 0]", "&mut [u64; 0]"] {
        let src = format!("pub extern \"sysv64\" fn k(xs: {ty}) {{\n}}\n");
        assert_eq!(code(&src)?, ["ret"], "{ty}");
    }
    let src = "pub extern \"sysv64\" fn k(xs: &[f32; 0]) {\nlet v: xmm;\nvmovss(v, [xs]);\n}\n";
    assert_eq!(
        code(src).err().ok_or("code gave Ok")?,
        "3:11: `vmovss` reads 4 bytes at `[xs]`, past the end of `xs`, 0 bytes"
    );
    Ok(())
}

/// A type a kernel takes may name its compile-time arguments, as in any generic function, `&[u64; N]`, and a const of
/// the module or a number in hex, `&[u64; L]` and `&[u64; 0x10]`, as Rust takes them. The array is as long as the
/// instance says, and kernite checks each address in it against that: `[xs + 8]` of `k::<2>` is inside, and of `k::<1>`
/// past the end.
#[test]
fn the_length_of_an_array_is_a_compile_time_argument_or_a_const() -> Result {
    let src = r#"
const L: usize = 2;

#[instances(2, 1)]
pub extern "sysv64" fn k<const N: usize>(xs: &[u64; N]) -> u64 {
    let g: gpr;
    mov(g, [xs + 8]);
    ret(g);
}

pub extern "sysv64" fn l(xs: &[u64; L]) -> u64 {
    let g: gpr;
    mov(g, [xs + 8]);
    ret(g);
}

pub extern "sysv64" fn hex(xs: &[u64; 0x10]) -> u64 {
    let g: gpr;
    mov(g, [xs + 120]);
    ret(g);
}

pub extern "sysv64" fn past(xs: &[u64; 0x10]) -> u64 {
    let g: gpr;
    mov(g, [xs + 128]);
    ret(g);
}
"#;
    for (name, want) in [("k::<2>", "rdi + 8"), ("l", "rdi + 8"), ("hex", "rdi + 120")] {
        assert_eq!(code_of(src, name)?, [format!("mov rax, qword ptr [{want}]"), "ret".to_string()], "{name}");
    }
    assert_eq!(
        code_of(src, "k::<1>").err().ok_or("code gave Ok")?,
        "7:12: in `k::<1>`: `mov` reads 8 bytes at `[xs + 8]`, past the end of `xs`, 8 bytes"
    );
    assert_eq!(
        code_of(src, "past").err().ok_or("code gave Ok")?,
        "25:12: `mov` reads 8 bytes at `[xs + 128]`, past the end of `xs`, 128 bytes"
    );
    Ok(())
}

/// A hint names what a kernel can take: `mov(g, xs)` of an array says that `[xs]` is what it holds, and, of a kernel
/// that is an `unsafe fn`, that `lea(p, [xs])` puts its address in a register, which a safe kernel is refused.
#[test]
fn a_hint_names_no_way_a_safe_kernel_refuses() -> Result {
    let src = "pub extern \"sysv64\" fn k(xs: &[u64; 1]) {\nlet g: gpr;\nmov(g, xs);\n}\n";
    assert_eq!(
        code(src).err().ok_or("code gave Ok")?,
        "3:8: `mov` takes no memory as an operand: `[xs]` is what it holds"
    );
    assert_eq!(
        code(&trusted(src)).err().ok_or("code gave Ok")?,
        "3:8: `mov` takes no memory as an operand: `lea(p, [xs])` puts the address of `xs` in `p`, and `[xs]` is what \
         it holds"
    );
    Ok(())
}

/// `nop` reaches no memory, whatever its operand names, so a safe kernel takes it at an address in a register that
/// kernite cannot check: no other instruction is.
#[test]
fn a_nop_reaches_no_memory_so_a_safe_kernel_takes_it_at_any_address() -> Result {
    let src = "pub extern \"sysv64\" fn k(g: u64) {\nnop(qword[g]);\n}\n";
    assert_eq!(code(src)?, ["nop qword ptr [rdi]", "ret"]);
    let src = src.replace("nop(qword[g]);", "mov(qword[g], 0);");
    assert_eq!(
        code(&src).err().ok_or("code gave Ok")?,
        "2:10: `mov` reaches memory at the address in `g`, which kernite cannot check, so a kernel that does is an \
         `unsafe fn`"
    );
    Ok(())
}

/// A kernel runs only where each function it calls does, so what it needs of the CPU is what it needs itself and what
/// each function it calls needs, in turn. A safe kernel is called without a check of the CPU at each call, and this is
/// what its wrapper says the CPU must have.
#[test]
fn a_kernel_needs_the_features_of_the_functions_it_calls() -> Result {
    let src = r#"
pub extern "sysv64" fn outer(a: u64, x: f32) -> u64 {
    vaddss(x.xmm, x.xmm, x.xmm);
    call(a, middle, a);
    ret(a);
}

pub extern "sysv64" fn middle(a: u64) -> u64 {
    call(a, inner, a);
    ret(a);
}

pub extern "sysv64" fn inner(a: u64) -> u64 {
    popcnt(a, a);
    ret(a);
}

pub extern "sysv64" fn alone(a: u64) -> u64 {
    ret(a);
}
"#;
    assert_eq!(needs(src, "outer")?, "# needs: avx, popcnt");
    assert_eq!(needs(src, "middle")?, "# needs: popcnt");
    assert_eq!(needs(src, "inner")?, "# needs: popcnt");
    assert_eq!(needs(src, "alone")?, "");
    Ok(())
}
