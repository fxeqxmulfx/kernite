//! What a GPU kernel's memory moves, which the listing says in comments before the kernel: for each load, store,
//! atomic and copy of global memory, the sectors of 32 bytes a warp moves and how many of their bytes it uses, and for
//! each pointer, how many times at most the launch reads a byte of it. The kernels here are `unsafe fn`s, but for
//! `saxpy`: the report is the same for each kernel, as what it counts needs no length.

type Result<T = (), E = Box<dyn std::error::Error + Send + Sync>> = std::result::Result<T, E>;

/// The helper that the kernels call: this thread's index among all the grid's.
const GLOBAL_INDEX: &str = r#"
fn global_index(i: u32) {
    let (block, size, thread): (u32, u32, u32);
    mov.u32(block, ctaid.x);
    mov.u32(size, ntid.x);
    mov.u32(thread, tid.x);
    mad.lo.u32(i, block, size, thread);
}
"#;

/// The first line of a report.
const HEAD: &str = "A warp of 32 threads moves, in sectors of 32 bytes:";

/// Kernel `k`, an `unsafe fn`, of parameters `params` and body `lines`, one to a line, and the helper it calls.
fn kernel(params: &str, lines: &[&str]) -> String {
    let body: String = lines.iter().map(|l| format!("    {l}\n")).collect();
    format!("pub unsafe extern \"ptx-kernel\" fn k({params}) {{\n{body}}}\n{GLOBAL_INDEX}")
}

/// The comments before the kernel `name` of the listing of `src`, each with its `// ` taken off.
fn report_of(src: &str, name: &str) -> Result<Vec<String>> {
    let listing = kernite::ptx_str(src)?;
    let lines: Vec<&str> = listing.lines().collect();
    let entry =
        lines.iter().position(|l| l.starts_with(&format!(".visible .entry {name}("))).ok_or("no such kernel")?;
    let before = lines.iter().take(entry).rev().take_while(|l| l.starts_with("//"));
    let mut report: Vec<String> =
        before.map(|l| l.trim_start_matches("// ").trim_start_matches("//").to_string()).collect();
    report.reverse();
    Ok(report)
}

/// The report of the kernel `k` of `src`.
fn report(src: &str) -> Result<Vec<String>> {
    report_of(src, "k")
}

/// The lines of `report` that say what a warp moves, one to an access, and then those that say of a pointer how many
/// times a byte is read, without the indent the report gives them.
fn lines(report: &[String]) -> Vec<&str> {
    report.iter().skip(1).map(|l| l.trim_start()).collect()
}

/// The lines that give the thread its index `i` among all the grid's, and the address `at` that is `scale` bytes from
/// `x` for each of them.
fn index_at(scale: u32) -> Vec<String> {
    let mad = format!("mad.wide.u32(at, i, {scale}, x);");
    ["let i: u32;", "global_index(i);", "let at: u64;", mad.as_str()].map(String::from).to_vec()
}

/// The kernel `k` of `x: *mut {ty}` whose body is `index_at(scale)` and then `access`.
fn at_index(ty: &str, scale: u32, access: &[&str]) -> String {
    let mut body = index_at(scale);
    body.extend(access.iter().map(|l| l.to_string()));
    let body: Vec<&str> = body.iter().map(String::as_str).collect();
    kernel(&format!("x: *mut {ty}"), &body)
}

/// What a warp moves, as the listing says of each access, in each case: the type of what `x` points to, the bytes
/// from one thread's address to the next's, the lines that access it, and what the listing says of it, as the
/// instruction is listed and then its line.
#[test]
fn a_warp_moves_the_sectors_its_threads_addresses_are_in() -> Result {
    let cases: [(&str, u32, &[&str], &str); 11] = [
        // Each thread its own float: 128 bytes in 4 sectors, all used.
        (
            "f32",
            4,
            &["let v: f32;", "ld.global.f32(v, [at]);"],
            "ld.global.f32 %v, [%at]: 4 sectors for 128 bytes, all used",
        ),
        // Each thread the first float of 8: a sector to a thread, of which an eighth is used.
        (
            "f32",
            32,
            &["let v: f32;", "ld.global.f32(v, [at]);"],
            "ld.global.f32 %v, [%at]: 32 sectors for 128 bytes, an eighth used",
        ),
        // One address for every thread: one sector, for the 4 bytes they all use.
        (
            "f32",
            0,
            &["let v: f32;", "ld.global.f32(v, [at]);"],
            "ld.global.f32 %v, [%at]: 1 sector for 4 bytes, an eighth used",
        ),
        // Vectors, each thread its own: 8 bytes, and 16.
        (
            "f32",
            8,
            &["let (v, w): (f32, f32);", "ld.global.v2.f32((v, w), [at]);"],
            "ld.global.v2.f32 {%v, %w}, [%at]: 8 sectors for 256 bytes, all used",
        ),
        (
            "f32",
            16,
            &["let (v, w, a, b): (f32, f32, f32, f32);", "ld.global.v4.f32((v, w, a, b), [at]);"],
            "ld.global.v4.f32 {%v, %w, %a, %b}, [%at]: 16 sectors for 512 bytes, all used",
        ),
        // The warp starts 4 bytes past a sector, and so moves a fifth.
        (
            "f32",
            4,
            &["let v: f32;", "ld.global.f32(v, [at + 4]);"],
            "ld.global.f32 %v, [%at+4]: 5 sectors for 128 bytes",
        ),
        // The same byte for every thread: a sector, for the 1 byte they all use.
        (
            "u8",
            0,
            &["let v: u8;", "ld.global.u8(v, [at]);"],
            "ld.global.u8 %v, [%at]: 1 sector for 1 byte, a thirty-second used",
        ),
        // Bytes, halves and doubles: 32 bytes in a sector, 64 in 2, 256 in 8.
        ("u8", 1, &["let v: u8;", "ld.global.u8(v, [at]);"], "ld.global.u8 %v, [%at]: 1 sector for 32 bytes, all used"),
        (
            "u16",
            2,
            &["let v: u16;", "ld.global.u16(v, [at]);"],
            "ld.global.u16 %v, [%at]: 2 sectors for 64 bytes, all used",
        ),
        (
            "f64",
            8,
            &["let v: f64;", "ld.global.f64(v, [at]);"],
            "ld.global.f64 %v, [%at]: 8 sectors for 256 bytes, all used",
        ),
        // A store moves what a load does.
        (
            "f32",
            4,
            &["st.global.f32([at], 1.0);"],
            "st.global.f32 [%at], 0f3F800000: 4 sectors for 128 bytes, all used",
        ),
    ];
    for (ty, scale, access, line) in cases {
        let got = report(&at_index(ty, scale, access))?;
        assert_eq!(got.first().map(String::as_str), Some(HEAD), "{line}");
        assert_eq!(lines(&got)[0], line);
    }
    Ok(())
}

/// How much of what a warp moves its threads use is said in a word where one says it: all, half, a quarter, an eighth,
/// a sixteenth or a thirty-second, or a number of them; and in none where none does, as a third is not. Threads whose
/// bytes overlap use each once.
#[test]
fn how_much_of_what_a_warp_moves_is_used_is_said_in_words() -> Result {
    let cases: [(&str, u32, &[&str], &str); 7] = [
        // Every other byte: half of 2 sectors.
        (
            "u8",
            2,
            &["let v: u8;", "ld.global.u8(v, [at]);"],
            "ld.global.u8 %v, [%at]: 2 sectors for 32 bytes, half used",
        ),
        // A float in each 4: 16 sectors, of which a quarter is used.
        (
            "f32",
            16,
            &["let v: f32;", "ld.global.f32(v, [at]);"],
            "ld.global.f32 %v, [%at]: 16 sectors for 128 bytes, a quarter used",
        ),
        // A sector to each thread, of which a half, a sixteenth and a thirty-second are used.
        (
            "f32",
            32,
            &["let (v, w, a, b): (f32, f32, f32, f32);", "ld.global.v4.f32((v, w, a, b), [at]);"],
            "ld.global.v4.f32 {%v, %w, %a, %b}, [%at]: 32 sectors for 512 bytes, half used",
        ),
        (
            "u16",
            32,
            &["let v: u16;", "ld.global.u16(v, [at]);"],
            "ld.global.u16 %v, [%at]: 32 sectors for 64 bytes, a sixteenth used",
        ),
        (
            "u8",
            32,
            &["let v: u8;", "ld.global.u8(v, [at]);"],
            "ld.global.u8 %v, [%at]: 32 sectors for 32 bytes, a thirty-second used",
        ),
        // Words 2 bytes apart overlap: their 66 bytes are in 3 sectors, and 66 of 96 is 11 sixteenths.
        (
            "u32",
            2,
            &["let v: u32;", "ld.global.u32(v, [at]);"],
            "ld.global.u32 %v, [%at]: 3 sectors for 66 bytes, 11 sixteenths used",
        ),
        // A third of 12 sectors is no part that a word says.
        ("f32", 12, &["let v: f32;", "ld.global.f32(v, [at]);"], "ld.global.f32 %v, [%at]: 12 sectors for 128 bytes"),
    ];
    for (ty, scale, access, line) in cases {
        let got = report(&at_index(ty, scale, access))?;
        assert_eq!(lines(&got)[0], line);
    }
    Ok(())
}

/// A thread of a block reads the floats of `x` from the last of 32 to the first: its step is 4 bytes down, and the
/// warp moves the same 4 sectors.
#[test]
fn a_warp_whose_addresses_go_down_moves_the_sectors_they_are_in() -> Result {
    let src = kernel(
        "x: *const f32",
        &[
            "let (block, size, thread, back, i): (u32, u32, u32, u32, u32);",
            "mov.u32(block, ctaid.x);",
            "mov.u32(size, ntid.x);",
            "mov.u32(thread, tid.x);",
            "sub.u32(back, 31, thread);",
            "mad.lo.u32(i, block, size, back);",
            "let (at, v): (u64, f32);",
            "mad.wide.u32(at, i, 4, x);",
            "ld.global.f32(v, [at]);",
        ],
    );
    let got = report(&src)?;
    assert_eq!(lines(&got)[0], "ld.global.f32 %v, [%at]: 4 sectors for 128 bytes, all used");
    Ok(())
}

/// The report of a kernel of `x: *const f32` and `y: *mut f32` that loads from `x` at the addresses `at` makes, each
/// `at` of the instructions that make it, and stores to `y` where each thread is: the line of `x`, after the lines
/// of the accesses.
fn what_is_read(params: &str, loads: &[&[&str]]) -> Result<String> {
    let mut body: Vec<String> =
        vec!["let i: u32;".into(), "global_index(i);".into(), "let (at, v): (u64, f32);".into()];
    for load in loads {
        body.extend(load.iter().map(|l| l.to_string()));
        body.push("ld.global.f32(v, [at]);".into());
    }
    let body: Vec<&str> = body.iter().map(String::as_str).collect();
    let got = report(&kernel(params, &body))?;
    let last = got.last().ok_or("no report")?;
    Ok(last.trim_start().to_string())
}

/// What the report says of a pointer: how many times at most the launch reads a byte of it, as the address says.
#[test]
fn a_pointer_says_how_many_times_a_byte_of_it_is_read() -> Result {
    let own = "mad.wide.u32(at, i, 4, x);";
    let next = ["let j: u32;", "add.u32(j, i, 1);", "mad.wide.u32(at, j, 4, x);"];
    let before = ["mad.wide.u32(at, i, 4, x);", "sub.u64(at, at, 4);"];
    // An index that wraps below 0 for the thread that is the grid's first is none kernite follows.
    let wraps = ["let j: u32;", "sub.u32(j, i, 1);", "mad.wide.u32(at, j, 4, x);"];
    // The block's start, 3 floats for each of its threads and 8 more, which is `j`.
    let apart = [
        "let (b, s, t, j): (u32, u32, u32, u32);",
        "mov.u32(b, ctaid.x);",
        "mov.u32(s, ntid.x);",
        "mad.lo.u32(s, s, 3, 8);",
        "mov.u32(t, tid.x);",
        "mul.lo.u32(t, t, 3);",
        "mad.lo.u32(j, b, s, t);",
        "mad.wide.u32(at, j, 4, x);",
    ];
    let cases: [(&[&[&str]], &str); 13] = [
        // Each thread its own element.
        (&[&[own]], "x: each byte is read at most once"),
        // Each block all of them, as a product of a matrix and a vector reads the vector.
        (
            &[&["let t: u32;", "mov.u32(t, tid.x);", "mad.wide.u32(at, t, 4, x);"]],
            "x: each byte is read at most once by every block",
        ),
        // Each thread its element and the next.
        (&[&[own], &next], "x: each byte is read at most twice"),
        (&[&[own], &next, &before], "x: each byte is read at most 3 times"),
        (&[&[own], &wraps], "x: not told how many times a byte is read, as a load of it is not counted"),
        // The two floats of each pair, by one thread: no byte twice.
        (
            &[&["mad.wide.u32(at, i, 8, x);"], &["mad.wide.u32(at, i, 8, x);", "add.u64(at, at, 4);"]],
            "x: each byte is read at most once",
        ),
        // Its element, and the element its block's thread of that index reads: the blocks step apart, but not alike.
        (
            &[&[own], &["let t: u32;", "mov.u32(t, tid.x);", "mad.wide.u32(at, t, 4, x);"]],
            "x: not told how many times a byte is read, as its loads step differently, which kernite does not compare",
        ),
        // Every thread the same element, and every thread of a block.
        (&[&["mad.wide.u32(at, i, 0, x);"]], "x: each byte is read at most once by every thread"),
        (
            &[&["let b: u32;", "mov.u32(b, ctaid.x);", "mad.wide.u32(at, b, 4, x);"]],
            "x: each byte is read at most once by every thread of a block",
        ),
        // Blocks that may overlap: a block's step is 32 floats, and its threads may be more.
        (
            &[&[
                "let (b, t, k): (u32, u32, u32);",
                "mov.u32(b, ctaid.x);",
                "shl.b32(b, b, 5);",
                "mov.u32(t, tid.x);",
                "add.u32(k, b, t);",
                "mad.wide.u32(at, k, 4, x);",
            ]],
            "x: not told how many times a byte is read, as its blocks may read the same bytes",
        ),
        // A block's step is 32 bytes more than its threads' 12 each, and a load 140 bytes before reaches back past
        // that, so a block reads what the one before it does.
        (
            &[&apart, &["sub.u64(at, at, 140);"]],
            "x: not told how many times a byte is read, as its blocks may read the same bytes",
        ),
        // A block's step is 2 floats, and its two loads reach 3: the blocks read the same float.
        (
            &[&["let b: u32;", "mov.u32(b, ctaid.x);", "mad.wide.u32(at, b, 8, x);"], &["add.u64(at, at, 8);"]],
            "x: not told how many times a byte is read, as its blocks may read the same bytes",
        ),
        // Its element and the element `n` on, which the launch gives: where one starts from the other is not known.
        (
            &[&[own], &["let j: u32;", "add.u32(j, i, n);", "mad.wide.u32(at, j, 4, x);"]],
            "x: not told how many times a byte is read, as its loads start apart by a number the launch gives, which \
             kernite does not compare",
        ),
    ];
    for (loads, line) in cases {
        assert_eq!(what_is_read("x: *const f32, n: u32", loads)?, line, "{loads:?}");
    }
    Ok(())
}

/// The kernel `k` of `unsafe fn` of parameters `params` and body `lines`, which the report of is wanted whole.
fn whole(params: &str, lines: &[&str]) -> Result<Vec<String>> {
    report(&kernel(params, lines))
}

/// What kernite does not follow, it says, in the access's own line, and why; and the rest it counts as it does of a
/// kernel with none of it.
#[test]
fn an_access_that_kernite_cannot_follow_says_why() -> Result {
    let got = whole(
        "x: *const f32, table: *const u64, m: *mut u64",
        &[
            "let i: u32;",
            "global_index(i);",
            // Shared memory, a barrier, and a barrier that a copy arrives at: no access of global memory, and so no
            // line.
            "let tile: shared<[f32; 32]>;",
            "let s: u32;",
            "mov.u32(s, tile);",
            "st.shared.f32([s], 1.0);",
            "bar.sync(0);",
            "cp.r#async.mbarrier.arrive.b64([m]);",
            // A load at an address that was loaded.
            "let (p, v): (u64, f32);",
            "ld.global.u64(p, [table]);",
            "ld.global.f32(v, [p]);",
            // A load at a generic address.
            "let at: u64;",
            "mad.wide.u32(at, i, 4, x);",
            "ld.f32(v, [at]);",
            // A load at an address of `tid.y`.
            "let (row, below): (u32, u64);",
            "mov.u32(row, tid.y);",
            "mad.wide.u32(below, row, 4, x);",
            "ld.global.f32(v, [below]);",
        ],
    )?;
    assert_eq!(
        got,
        [
            HEAD,
            "  ld.global.u64 %p, [%table]: 1 sector for 8 bytes, a quarter used",
            "  ld.global.f32 %v, [%p]: not counted, as its address comes from a value loaded from memory",
            "  ld.f32 %v, [%at]: not counted, as it reaches memory by a generic address, which may be shared or local \
             memory as well as global",
            "  ld.global.f32 %v, [%below]: not counted, as its address comes from `tid.y`, which kernite does not \
             follow",
            "  x: not told how many times a byte is read, as a load of it is not counted",
            "  table: each byte is read at most once by every thread",
        ]
    );
    Ok(())
}

/// What reaches global memory and moves no value, or reaches it by a handle that is no pointer of the kernel's, and
/// what reaches shared, local or constant memory, has no line: a kernel that has one of each has no report.
#[test]
fn what_has_no_line_leaves_a_kernel_with_no_report() -> Result {
    let table = "static TABLE: constant<[f32; 4], 16> = [1.0, 2.0, 0.5, -1.0];\n";
    let src = format!(
        "{table}{}",
        kernel(
            "x: *mut f32, t: u64",
            &[
                // Global memory, and no value moves.
                "prefetch.global.L1([x]);",
                "prefetchu.L1([x]);",
                "applypriority.global.L2__evict_normal([x], 128);",
                "discard.global.L2([x], 128);",
                // Global memory, by a handle: a texture or a surface.
                "let (f, r0, r1, r2, r3): (f32, f32, f32, f32, f32);",
                "mov.f32(f, 0.5);",
                "tex._2d.v4.f32.f32((r0, r1, r2, r3), [t, (f, f)]);",
                "tld4.r._2d.v4.f32.f32((r0, r1, r2, r3), [t, (f, f)]);",
                "let (n, a, b): (u32, u32, u32);",
                "mov.u32(n, 1);",
                "suld.b._2d.v2.b32.trap((a, b), [t, (n, n)]);",
                "sust.b._2d.v2.b32.trap([t, (n, n)], (a, b));",
                "sured.b.add._2d.u32.trap([t, (n, n)], a);",
                // Shared memory, by an atomic; local memory; constant memory.
                "let (tile, own, old): (shared<[u32; 32]>, local<[f32; 4]>, u32);",
                "atom.shared.add.u32(old, [tile], 1);",
                "st.local.f32([own], 1.0);",
                "ld.const.f32(f, [TABLE + 4]);",
            ],
        )
    );
    assert_eq!(report(&src)?, Vec::<String>::new());
    Ok(())
}

/// An atomic, a reduction, a copy by `cp.async` and a load by `ldu` reach global memory at an address as a load does,
/// and the GPU counts what they move as it does a load's: each has a line of its own, in the words of a load's.
#[test]
fn what_an_atomic_a_reduction_a_copy_and_a_uniform_load_move_is_counted_as_a_loads_is() -> Result {
    let cases: [(u32, &[&str], &str); 6] = [
        // Each thread adds to its own word, and has what was there.
        (
            4,
            &["let old: u32;", "atom.global.add.u32(old, [at], 1);"],
            "atom.global.add.u32 %old, [%at], 1: 4 sectors for 128 bytes, all used",
        ),
        // The same, and nothing comes back.
        (4, &["red.global.add.u32([at], 1);"], "red.global.add.u32 [%at], 1: 4 sectors for 128 bytes, all used"),
        // Each thread copies its own word to shared memory.
        (
            4,
            &["let tile: shared<[u32; 32], 16>;", "cp.r#async.ca.shared.global([tile], [at], 4);"],
            "cp.async.ca.shared.global [tile], [%at], 4: 4 sectors for 128 bytes, all used",
        ),
        // Every thread loads the same word, as `ldu` asks.
        (
            0,
            &["let v: u32;", "ldu.global.u32(v, [at]);"],
            "ldu.global.u32 %v, [%at]: 1 sector for 4 bytes, an eighth used",
        ),
        // A copy may read fewer of the bytes it copies, the rest being zeros, by a number of them; and where it reads
        // none, as `n` of 0 says, it reads nothing. It is counted as one that reads all it copies.
        (
            4,
            &[
                "let (tile, n): (shared<[u32; 32], 16>, u32);",
                "mov.u32(n, 2);",
                "cp.r#async.ca.shared.global([tile], [at], 4, n);",
            ],
            "cp.async.ca.shared.global [tile], [%at], 4, %n: 4 sectors for 128 bytes, all used",
        ),
        // The same, by a predicate that has it read none.
        (
            4,
            &[
                "let (tile, p): (shared<[u32; 32], 16>, pred);",
                "setp.ge.u32(p, i, 1);",
                "cp.r#async.ca.shared.global([tile], [at], 4, p);",
            ],
            "cp.async.ca.shared.global [tile], [%at], 4, %p: 4 sectors for 128 bytes, all used",
        ),
    ];
    for (scale, access, line) in cases {
        let got = report(&at_index("u32", scale, access))?;
        assert_eq!(got.first().map(String::as_str), Some(HEAD), "{line}");
        assert_eq!(lines(&got)[0], line);
    }
    Ok(())
}

/// A byte of a pointer is read by an atomic, a reduction, a copy and a load by `ldu` as it is by a load, and so a
/// kernel that has one of them besides a load of the same address reads each byte twice; and one that has a copy and
/// no load has a report.
#[test]
fn a_pointer_is_read_by_what_reads_it_besides_a_load() -> Result {
    let cases: [(u32, &[&str], &str); 4] = [
        (
            4,
            &["let (v, old): (u32, u32);", "ld.global.u32(v, [at]);", "atom.global.add.u32(old, [at], 1);"],
            "x: each byte is read at most twice",
        ),
        (
            4,
            &["let v: u32;", "ld.global.u32(v, [at]);", "red.global.add.u32([at], 1);"],
            "x: each byte is read at most twice",
        ),
        (
            4,
            &[
                "let (v, tile): (u32, shared<[u32; 32], 16>);",
                "ld.global.u32(v, [at]);",
                "cp.r#async.ca.shared.global([tile], [at], 4);",
            ],
            "x: each byte is read at most twice",
        ),
        (
            0,
            &["let (v, w): (u32, u32);", "ld.global.u32(v, [at]);", "ldu.global.u32(w, [at]);"],
            "x: each byte is read at most twice by every thread",
        ),
    ];
    for (scale, access, line) in cases {
        let got = report(&at_index("u32", scale, access))?;
        assert_eq!(got.last().map(|l| l.trim_start()), Some(line), "{access:?}");
    }
    let copy = report(&at_index(
        "u32",
        4,
        &["let tile: shared<[u32; 32], 16>;", "cp.r#async.ca.shared.global([tile], [at], 4);"],
    ))?;
    assert_eq!(copy.last().map(|l| l.trim_start()), Some("x: each byte is read at most once"));
    Ok(())
}

/// A function is not followed: a call that is passed an address of a pointer has a line that says so, and leaves the
/// pointer not told, whether the function loads from it or gives it back for a load to use. A call that is passed no
/// pointer has none.
#[test]
fn a_call_that_is_passed_a_pointer_is_not_followed_and_leaves_it_not_told() -> Result {
    let functions = r#"
extern "ptx-func" fn peek(p: u64) -> f32 {
    let v: f32;
    ld.global.f32(v, [p]);
    ret(v);
}
extern "ptx-func" fn address(p: u64) -> u64 {
    ret(p);
}
extern "ptx-func" fn twice(n: u32) -> u32 {
    let m: u32;
    add.u32(m, n, n);
    ret(m);
}
"#;
    let load = [
        "let i: u32;",
        "global_index(i);",
        "let (at, v, w): (u64, f32, f32);",
        "mad.wide.u32(at, i, 4, x);",
        "ld.global.f32(v, [at]);",
    ];
    let cases: [(&[&str], Vec<&str>); 3] = [
        // What a function loads from the address it is passed.
        (
            &["call(w, peek, at);"],
            vec![
                "call (%w), peek, (%at): not counted, as kernite does not follow a function a kernel calls",
                "x: not told how many times a byte is read, as a load of it is not counted",
            ],
        ),
        // An address that a function gives back, which every thread loads from.
        (
            &["let p: u64;", "call(p, address, x);", "ld.global.f32(w, [p]);"],
            vec![
                "call (%p), address, (%x): not counted, as kernite does not follow a function a kernel calls",
                "ld.global.f32 %w, [%p]: not counted, as its address comes from what a function returns",
                "x: not told how many times a byte is read, as a load of it is not counted",
            ],
        ),
        // A number: the pointer is not passed.
        (&["let n: u32;", "call(n, twice, i);"], vec!["x: each byte is read at most once"]),
    ];
    for (lines_after, tail) in cases {
        let mut body: Vec<&str> = load.to_vec();
        body.extend(lines_after);
        let src = format!("{}{functions}", kernel("x: *const f32", &body));
        let got = report(&src)?;
        let mut want = vec!["ld.global.f32 %v, [%at]: 4 sectors for 128 bytes, all used"];
        want.extend(tail);
        assert_eq!(lines(&got), want, "{lines_after:?}");
    }
    Ok(())
}

/// `wmma` reaches global memory by an address, and kernite does not count what it moves yet: its line says so, and a
/// pointer it loads from is not told.
#[test]
fn what_a_wmma_moves_is_not_counted_yet_and_leaves_a_pointer_it_loads_from_not_told() -> Result {
    let got = whole(
        "x: *const f32, y: *mut f32",
        &[
            "let (a, b): ([f16x2; 8], [f16x2; 8]);",
            "let c: [f32; 8];",
            "wmma.load.c.sync.aligned.row.m16n16k16.f32(c, [x]);",
            "wmma.store.d.sync.aligned.row.m16n16k16.global.f32([y], c);",
        ],
    )?;
    assert_eq!(
        lines(&got),
        [
            "wmma.load.c.sync.aligned.row.m16n16k16.f32 {%c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7}, [%x]: \
             not counted, as kernite does not count what `wmma.load` moves yet",
            "wmma.store.d.sync.aligned.row.m16n16k16.global.f32 [%y], {%c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7}: \
             not counted, as kernite does not count what `wmma.store` moves yet",
            "x: not told how many times a byte is read, as a load of it is not counted",
        ]
    );
    Ok(())
}

/// What the report takes as given of the numbers a kernel works out: one that wraps on some launches, as `i + n` does
/// where `n` is a number the launch gives, is counted as if it never did, and one that wraps on every launch, `i - 1`
/// where `i` may be 0, is not followed.
#[test]
fn a_number_is_taken_not_to_wrap_and_one_that_always_wraps_is_not_followed() -> Result {
    let kernel = |work: &[&str]| -> Result<Vec<String>> {
        let mut body = vec!["let (i, j, v): (u32, u32, f32);", "global_index(i);"];
        body.extend(work);
        body.extend(["let at: u64;", "mad.wide.u32(at, j, 4, x);", "ld.global.f32(v, [at]);"]);
        report(&kernel("x: *const f32, n: u32", &body))
    };
    let got = kernel(&["add.u32(j, i, n);"])?;
    assert_eq!(
        lines(&got),
        ["ld.global.f32 %v, [%at]: 4 sectors for 128 bytes, all used", "x: each byte is read at most once"]
    );
    let got = kernel(&["sub.u32(j, i, 1);"])?;
    assert_eq!(
        lines(&got),
        [
            "ld.global.f32 %v, [%at]: not counted, as its address comes from what `sub.u32` works out, which kernite \
             does not follow",
            "x: not told how many times a byte is read, as a load of it is not counted",
        ]
    );
    Ok(())
}

/// What the report takes as given of the memory of a launch: each pointer is to memory of its own. A load at an
/// address that a kernel loads from memory, or at a number, is counted of no pointer, though the address may be one of
/// a pointer, and so leaves the line of each pointer as the loads kernite follows to it say.
#[test]
fn a_pointer_is_taken_to_be_to_memory_of_its_own() -> Result {
    let got = whole(
        "x: *const f32, table: *const u64",
        &[
            "let (i, at, v): (u32, u64, f32);",
            "global_index(i);",
            "mad.wide.u32(at, i, 4, x);",
            "ld.global.f32(v, [at]);",
            // A load at an address that was loaded, which may be `x`, and one at a number, which may be too.
            "let (p, w, n, u): (u64, f32, u64, f32);",
            "ld.global.u64(p, [table]);",
            "ld.global.f32(w, [p]);",
            "mov.u64(n, 4096);",
            "ld.global.f32(u, [n]);",
        ],
    )?;
    assert_eq!(
        lines(&got),
        [
            "ld.global.f32 %v, [%at]: 4 sectors for 128 bytes, all used",
            "ld.global.u64 %p, [%table]: 1 sector for 8 bytes, a quarter used",
            "ld.global.f32 %w, [%p]: not counted, as its address comes from a value loaded from memory",
            "ld.global.f32 %u, [%n]: not counted, as its address is not an address of a pointer of the kernel's",
            "x: each byte is read at most once",
            "table: each byte is read at most once by every thread",
        ]
    );
    Ok(())
}

/// A kernel that jumps is not followed yet, and says so, whatever it loads.
#[test]
fn a_kernel_that_jumps_says_kernite_does_not_follow_one_yet() -> Result {
    let got = whole(
        "x: *const f32, n: u32",
        &[
            "let (i, p): (u32, pred);",
            "mov.u32(i, 0);",
            "label(top);",
            "let (at, v): (u64, f32);",
            "mad.wide.u32(at, i, 4, x);",
            "ld.global.f32(v, [at]);",
            "add.u32(i, i, 1);",
            "setp.lt.u32(p, i, n);",
            "if p {",
            "    bra(top);",
            "}",
        ],
    )?;
    assert_eq!(got, ["kernite does not follow a kernel that jumps yet: what its memory moves is not counted"]);
    Ok(())
}

/// The report is comments, directly before the kernel's entry, and the only ones but the listing's first line: no line
/// of code says any of it.
#[test]
fn the_report_is_comments_before_its_kernel() -> Result {
    let listing = kernite::ptx_str(&at_index("f32", 4, &["let v: f32;", "ld.global.f32(v, [at]);"]))?;
    let lines: Vec<&str> = listing.lines().collect();
    let entry = lines.iter().position(|l| l.starts_with(".visible .entry k(")).ok_or("no entry")?;
    let mut report: Vec<&str> = lines[..entry].iter().rev().take_while(|l| l.starts_with("//")).copied().collect();
    report.reverse();
    assert_eq!(
        report,
        [
            "// A warp of 32 threads moves, in sectors of 32 bytes:",
            "//   ld.global.f32 %v, [%at]: 4 sectors for 128 bytes, all used",
            "//   x: each byte is read at most once",
        ]
    );
    let others: Vec<&str> = lines.iter().filter(|l| l.starts_with("//") && !report.contains(l)).copied().collect();
    assert_eq!(others, ["// Written by kernite."]);
    assert!(lines.iter().filter(|l| !l.starts_with("//")).all(|l| !l.contains("sector") && !l.contains("warp")));
    Ok(())
}

/// A kernel with no access of global memory has no report, and nor has a function a kernel calls.
#[test]
fn a_kernel_that_moves_no_global_memory_has_no_report() -> Result {
    let src = format!(
        "{}{}",
        kernel(
            "n: u32",
            &[
                "let i: u32;",
                "global_index(i);",
                "let tile: shared<[u32; 32]>;",
                "let s: u32;",
                "mov.u32(s, tile);",
                "st.shared.u32([s], i);"
            ]
        ),
        "extern \"ptx-func\" fn peek(x: u64) -> f32 {\n    let v: f32;\n    ld.global.f32(v, [x]);\n    ret(v);\n}\n"
    );
    let listing = kernite::ptx_str(&src)?;
    assert!(!listing.contains("A warp"), "{listing}");
    assert!(listing.contains(".func (.param .f32 peek_retval) peek("), "{listing}");
    Ok(())
}

/// `saxpy` of the examples, with no `unsafe`: kernite proves it safe, and it says what it moves as it does of the
/// kernel with `unsafe`, whose code it is.
#[test]
fn a_kernel_with_no_unsafe_says_what_it_moves_as_one_with_unsafe_does() -> Result {
    let safe = format!(
        "{}{GLOBAL_INDEX}",
        r#"pub extern "ptx-kernel" fn saxpy(a: f32, x: *const f32, y: *mut f32, n: u32) {
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
"#
    );
    let trusted = safe.replace("pub extern", "pub unsafe extern");
    let want = [
        HEAD,
        "  ld.global.f32 %xi, [%at_x]: 4 sectors for 128 bytes, all used",
        "  ld.global.f32 %yi, [%at_y]: 4 sectors for 128 bytes, all used",
        "  st.global.f32 [%at_y], %yi: 4 sectors for 128 bytes, all used",
        "  x: each byte is read at most once",
        "  y: each byte is read at most once",
    ];
    assert_eq!(report_of(&safe, "saxpy")?, want);
    assert_eq!(report_of(&trusted, "saxpy")?, want);
    Ok(())
}

/// A warp is counted as if each of its 32 threads ran an access, whatever guards it, and where a warp starts is a
/// multiple of 32 bytes from its pointer wherever kernite knows no better: what `ntid.x` multiplies is one, as a
/// block's threads are whole warps, and so is what another number of the launch's does.
#[test]
fn a_guarded_access_is_counted_for_every_thread_and_an_unknown_start_is_a_sector_s_start() -> Result {
    let guarded = whole(
        "x: *mut f32, n: u32",
        &[
            "let i: u32;",
            "global_index(i);",
            "let (p, at): (pred, u64);",
            "setp.lt.u32(p, i, n);",
            "mad.wide.u32(at, i, 4, x);",
            "if p {",
            "    st.global.f32([at], 1.0);",
            "}",
        ],
    )?;
    assert_eq!(lines(&guarded)[0], "@%p st.global.f32 [%at], 0f3F800000: 4 sectors for 128 bytes, all used");
    // `n` floats on: the launch gives where the warp starts, and kernite takes it to be a sector's start.
    let moved = whole(
        "x: *const f32, n: u32",
        &[
            "let (i, j): (u32, u32);",
            "global_index(i);",
            "add.u32(j, i, n);",
            "let (at, v): (u64, f32);",
            "mad.wide.u32(at, j, 4, x);",
            "ld.global.f32(v, [at]);",
        ],
    )?;
    assert_eq!(lines(&moved)[0], "ld.global.f32 %v, [%at]: 4 sectors for 128 bytes, all used");
    // A block's start that is 4 bytes from the last one's moves each warp's start by that: no one count is the
    // warps'.
    let skewed = whole(
        "x: *const f32",
        &[
            "let (b, t, k): (u32, u32, u32);",
            "mov.u32(b, ctaid.x);",
            "mov.u32(t, tid.x);",
            "add.u32(k, b, t);",
            "let (at, v): (u64, f32);",
            "mad.wide.u32(at, k, 4, x);",
            "ld.global.f32(v, [at]);",
        ],
    )?;
    assert_eq!(
        lines(&skewed)[0],
        "ld.global.f32 %v, [%at]: not counted, as the start of its warps moves by 4 bytes from one block to the next"
    );
    Ok(())
}

/// Where kernite does not follow an address it says what the address is: a number, what a function returns, what an
/// instruction works out that kernite does not follow, or memory the module declares; or what is wrong with its steps:
/// a product of the thread's index and the block's, or a step from one thread to the next that the launch gives.
#[test]
fn an_address_that_kernite_does_not_follow_is_told_as_what_it_is() -> Result {
    let loaded = "ld.global.f32 %v, [%at]: not counted, as its address";
    let unread = "x: not told how many times a byte is read, as a load of it is not counted";
    let cases: [(&[&str], Vec<String>); 7] = [
        // A number, where no pointer of the kernel's is.
        (
            &["let (at, v): (u64, f32);", "mov.u64(at, 4096);", "ld.global.f32(v, [at]);"],
            vec![format!("{loaded} is not an address of a pointer of the kernel's")],
        ),
        // What a function returns, which it is given a pointer to give back.
        (
            &["let (at, v): (u64, f32);", "call(at, address, x);", "ld.global.f32(v, [at]);"],
            vec![
                "call (%at), address, (%x): not counted, as kernite does not follow a function a kernel calls".into(),
                format!("{loaded} comes from what a function returns"),
                unread.into(),
            ],
        ),
        // A product of the thread's index and the block's.
        (
            &[
                "let (b, t, k): (u32, u32, u32);",
                "mov.u32(b, ctaid.x);",
                "mov.u32(t, tid.x);",
                "mul.lo.u32(k, b, t);",
                "let (at, v): (u64, f32);",
                "mad.wide.u32(at, k, 4, x);",
                "ld.global.f32(v, [at]);",
            ],
            vec![format!("{loaded} is not a sum of steps of its thread's index and its block's"), unread.into()],
        ),
        // A step from one thread to the next that the launch gives.
        (
            &[
                "let (t, s, k): (u32, u32, u32);",
                "mov.u32(t, tid.x);",
                "mov.u32(s, ntid.x);",
                "mul.lo.u32(k, t, s);",
                "let (at, v): (u64, f32);",
                "mad.wide.u32(at, k, 4, x);",
                "ld.global.f32(v, [at]);",
            ],
            vec![
                "ld.global.f32 %v, [%at]: not counted, as its step from one thread to the next is a number the \
                 launch gives"
                    .into(),
                unread.into(),
            ],
        ),
        // An address that an instruction under a guard may leave as it was.
        (
            &[
                "let (i, p): (u32, pred);",
                "global_index(i);",
                "setp.lt.u32(p, i, 100);",
                "let (at, v): (u64, f32);",
                "mov.u64(at, x);",
                "if p {",
                "    mad.wide.u32(at, i, 4, x);",
                "}",
                "ld.global.f32(v, [at]);",
            ],
            vec![format!("{loaded} comes from an instruction under a guard"), unread.into()],
        ),
        // What an instruction kernite does not follow works out.
        (
            &[
                "let (i, k): (u32, u32);",
                "global_index(i);",
                "div.u32(k, i, 2);",
                "let (at, v): (u64, f32);",
                "mad.wide.u32(at, k, 4, x);",
                "ld.global.f32(v, [at]);",
            ],
            vec![format!("{loaded} comes from what `div.u32` works out, which kernite does not follow"), unread.into()],
        ),
        // No access of global memory: no line.
        (&["let v: f32;", "mov.f32(v, 1.0);"], vec![]),
    ];
    for (body, want) in cases {
        let src = format!(
            "{}extern \"ptx-func\" fn address(x: u64) -> u64 {{\n    ret(x);\n}}\n",
            kernel("x: *const f32", body)
        );
        let got = report(&src)?;
        let want: Vec<&str> = want.iter().map(String::as_str).collect();
        assert_eq!(lines(&got), want, "{body:?}");
    }
    // Memory the module declares, which every kernel and the host reach, and which no pointer of the kernel's is to.
    let body = ["let v: u32;", "ld.global.u32(v, [COUNTS + 4]);", "st.global.u32([y], v);"];
    let got = report(&format!("pub static COUNTS: global<[u32; 16]>;\n{}", kernel("y: *mut u32", &body)))?;
    assert_eq!(
        lines(&got),
        [
            "ld.global.u32 %v, [COUNTS+4]: not counted, as it reaches memory that the module declares, which kernite \
             does not follow yet",
            "st.global.u32 [%y], %v: 1 sector for 4 bytes, an eighth used",
        ]
    );
    Ok(())
}
