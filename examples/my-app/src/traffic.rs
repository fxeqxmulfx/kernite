//! What the listing of a GPU kernel says a warp moves, held to what the GPU counts. Each kernel of `shapes` reaches `x`
//! and `y` at its threads' indexes in the grid, in a shape of its own. NVIDIA's profiler, `ncu`, runs them and counts
//! the sectors of 32 bytes that their loads, stores, atomics and reductions of global memory take, and the requests
//! that ask for them. Each count is what the listing says of a warp, times the warps of the grid: exactly, but for the
//! loads of `overlap`, whose warps share a sector that the GPU may count once.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Command;

use kernite_cuda::{Cuda, Grid, Ptr};

type Result<T = (), E = Box<dyn std::error::Error + Send + Sync>> = std::result::Result<T, E>;

/// Kernels that each reach `x` and `y`, a thread at its index among all the grid's, as the kernel says: in the shape of
/// its address. They are `unsafe fn`s, as every kernel of `*const u8` and `*mut u8` is that reads floats there: what a
/// warp moves is the same for a kernel kernite proves safe.
#[kernite_macros::kernels]
mod shapes {
    /// Each thread its own float: a warp's 128 bytes are in 4 sectors, all used.
    pub unsafe extern "ptx-kernel" fn unit(x: *const u8, y: *mut u8) {
        let i: u32;
        global_index(i);
        let (from, to): (u64, u64);
        mad.wide.u32(from, i, 4, x);
        mad.wide.u32(to, i, 4, y);
        let v: f32;
        ld.global.f32(v, [from]);
        st.global.f32([to], v);
    }

    /// The first float of each 8: a sector a thread, 32 of them, of which an eighth is used.
    pub unsafe extern "ptx-kernel" fn spread(x: *const u8, y: *mut u8) {
        let i: u32;
        global_index(i);
        let (from, to): (u64, u64);
        mad.wide.u32(from, i, 32, x);
        mad.wide.u32(to, i, 32, y);
        let v: f32;
        ld.global.f32(v, [from]);
        st.global.f32([to], v);
    }

    /// The same float for every thread: one sector, for the 4 bytes they all use.
    pub unsafe extern "ptx-kernel" fn same(x: *const u8, y: *mut u8) {
        let v: f32;
        ld.global.f32(v, [x]);
        st.global.f32([y], v);
    }

    /// Two floats a thread, at once: 8 sectors for the warp's 256 bytes, all used.
    pub unsafe extern "ptx-kernel" fn pairs(x: *const u8, y: *mut u8) {
        let i: u32;
        global_index(i);
        let (from, to): (u64, u64);
        mad.wide.u32(from, i, 8, x);
        mad.wide.u32(to, i, 8, y);
        let (a, b): (f32, f32);
        ld.global.v2.f32((a, b), [from]);
        st.global.v2.f32([to], (a, b));
    }

    /// Four floats a thread, at once: 16 sectors for the warp's 512 bytes, all used.
    pub unsafe extern "ptx-kernel" fn quads(x: *const u8, y: *mut u8) {
        let i: u32;
        global_index(i);
        let (from, to): (u64, u64);
        mad.wide.u32(from, i, 16, x);
        mad.wide.u32(to, i, 16, y);
        let (a, b, c, d): (f32, f32, f32, f32);
        ld.global.v4.f32((a, b, c, d), [from]);
        st.global.v4.f32([to], (a, b, c, d));
    }

    /// Each thread the float after its own, in a block of one warp, which starts 512 bytes after the block before it:
    /// the warp starts 4 bytes past a sector, so its 128 bytes are in 5, and no other warp has one of them.
    pub unsafe extern "ptx-kernel" fn past(x: *const u8, y: *mut u8) {
        let (block, thread): (u32, u32);
        mov.u32(block, ctaid.x);
        mov.u32(thread, tid.x);
        let (from, to): (u64, u64);
        mad.wide.u32(from, block, 512, x);
        mad.wide.u32(to, block, 512, y);
        mad.wide.u32(from, thread, 4, from);
        mad.wide.u32(to, thread, 4, to);
        let v: f32;
        ld.global.f32(v, [from + 4]);
        st.global.f32([to + 4], v);
    }

    /// Each thread the float after its own, the warps of a block one after another: a warp starts 4 bytes past a
    /// sector, so its 128 bytes are in 5, and the last is the first of the warp after it.
    pub unsafe extern "ptx-kernel" fn overlap(x: *const u8, y: *mut u8) {
        let i: u32;
        global_index(i);
        let (from, to): (u64, u64);
        mad.wide.u32(from, i, 4, x);
        mad.wide.u32(to, i, 4, y);
        let v: f32;
        ld.global.f32(v, [from + 4]);
        st.global.f32([to + 4], v);
    }

    /// The floats of a block of 32 threads, from the last to the first: the same 4 sectors, all used. A block is one
    /// warp.
    pub unsafe extern "ptx-kernel" fn back(x: *const u8, y: *mut u8) {
        let (block, size, thread, last, i): (u32, u32, u32, u32, u32);
        mov.u32(block, ctaid.x);
        mov.u32(size, ntid.x);
        mov.u32(thread, tid.x);
        sub.u32(last, 31, thread);
        mad.lo.u32(i, block, size, last);
        let (from, to): (u64, u64);
        mad.wide.u32(from, i, 4, x);
        mad.wide.u32(to, i, 4, y);
        let v: f32;
        ld.global.f32(v, [from]);
        st.global.f32([to], v);
    }

    /// Each thread its own byte: 32 bytes are one sector, all used.
    pub unsafe extern "ptx-kernel" fn bytes(x: *const u8, y: *mut u8) {
        let i: u32;
        global_index(i);
        let (from, to): (u64, u64);
        mad.wide.u32(from, i, 1, x);
        mad.wide.u32(to, i, 1, y);
        let v: u8;
        ld.global.u8(v, [from]);
        st.global.u8([to], v);
    }

    /// Each thread its own half: 64 bytes in 2 sectors, all used.
    pub unsafe extern "ptx-kernel" fn halves(x: *const u8, y: *mut u8) {
        let i: u32;
        global_index(i);
        let (from, to): (u64, u64);
        mad.wide.u32(from, i, 2, x);
        mad.wide.u32(to, i, 2, y);
        let v: u16;
        ld.global.u16(v, [from]);
        st.global.u16([to], v);
    }

    /// Each thread its own double: 256 bytes in 8 sectors, all used.
    pub unsafe extern "ptx-kernel" fn doubles(x: *const u8, y: *mut u8) {
        let i: u32;
        global_index(i);
        let (from, to): (u64, u64);
        mad.wide.u32(from, i, 8, x);
        mad.wide.u32(to, i, 8, y);
        let v: f64;
        ld.global.f64(v, [from]);
        st.global.f64([to], v);
    }

    /// The same float for every thread, which `ldu` loads for a warp that reads one address: one sector, for the 4
    /// bytes they all use.
    pub unsafe extern "ptx-kernel" fn uniform(x: *const u8, y: *mut u8) {
        let i: u32;
        global_index(i);
        let to: u64;
        mad.wide.u32(to, i, 4, y);
        let v: f32;
        ldu.global.f32(v, [x]);
        st.global.f32([to], v);
    }

    /// Each thread adds 1 to its own word of `x`, and stores what the word was: the atomic's 128 bytes are in 4
    /// sectors, all used.
    pub unsafe extern "ptx-kernel" fn atoms(x: *mut u8, y: *mut u8) {
        let i: u32;
        global_index(i);
        let (from, to): (u64, u64);
        mad.wide.u32(from, i, 4, x);
        mad.wide.u32(to, i, 4, y);
        let old: u32;
        atom.global.add.u32(old, [from], 1);
        st.global.u32([to], old);
    }

    /// Every thread adds 1 to the same word of `x`, as a counter does: a sector for the warp, for the 4 bytes it
    /// reaches.
    pub unsafe extern "ptx-kernel" fn counts(x: *mut u8, y: *mut u8) {
        let i: u32;
        global_index(i);
        let to: u64;
        mad.wide.u32(to, i, 4, y);
        let old: u32;
        atom.global.add.u32(old, [x], 1);
        st.global.u32([to], old);
    }

    /// Each thread adds 1 to its own word of `x`, as `red` does, which gives nothing back; and stores a word to `y`.
    pub unsafe extern "ptx-kernel" fn reduces(x: *mut u8, y: *mut u8) {
        let i: u32;
        global_index(i);
        let (from, to): (u64, u64);
        mad.wide.u32(from, i, 4, x);
        mad.wide.u32(to, i, 4, y);
        red.global.add.u32([from], 1);
        st.global.u32([to], i);
    }

    /// Each thread copies its own word of `x` into shared memory with `cp.async`, and stores it to `y`.
    pub unsafe extern "ptx-kernel" fn copies(x: *const u8, y: *mut u8) {
        let tile: shared<[u32; 128], 16>;
        let (i, t, s): (u32, u32, u32);
        global_index(i);
        mov.u32(t, tid.x);
        mov.u32(s, tile);
        mad.lo.u32(s, t, 4, s);
        let (from, to): (u64, u64);
        mad.wide.u32(from, i, 4, x);
        mad.wide.u32(to, i, 4, y);
        cp.r#async.ca.shared.global([s], [from], 4);
        cp.r#async.wait_all();
        let v: u32;
        ld.shared.u32(v, [s]);
        st.global.u32([to], v);
    }

    /// The index of this thread among all the grid's.
    fn global_index(i: u32) {
        let (block, size, thread): (u32, u32, u32);
        mov.u32(block, ctaid.x);
        mov.u32(size, ntid.x);
        mov.u32(thread, tid.x);
        mad.lo.u32(i, block, size, thread);
    }
}

/// What a launch of a kernel of `shapes` is: the kernels and the pointers it takes.
type Launch = fn(&shapes::Kernels, Grid, Ptr<u8>, Ptr<u8>) -> std::result::Result<(), kernite_cuda::Error>;

/// A kernel of `shapes` and how it runs.
struct Shape {
    name: &'static str,
    /// The threads of each of its blocks, whole warps.
    threads: u32,
    /// Whether the last sector of a warp is the first of the next: ncu may count a load of it once, and fewer sectors
    /// than the listing says, but never more.
    shares: bool,
    launch: Launch,
}

/// Each kernel of `shapes`. A grid is `BLOCKS` blocks of its threads. A block of `back` and of `past` is one warp, as
/// their kernels say.
fn launches() -> [Shape; 16] {
    macro_rules! shape {
        ($kernel:ident, $threads:expr, $shares:expr) => {
            Shape {
                name: stringify!($kernel),
                threads: $threads,
                shares: $shares,
                launch: |kernels, grid, x, y| {
                    // SAFETY: `x` and `y` are the buffers of `run`, which hold 32 bytes for each thread of the grid
                    // and 4 more, as much as the widest kernel reaches, and last until `run` has waited for the kernel.
                    unsafe { kernels.$kernel(grid, x, y) }
                },
            }
        };
    }
    [
        shape!(unit, 128, false),
        shape!(spread, 128, false),
        shape!(same, 128, false),
        shape!(pairs, 128, false),
        shape!(quads, 128, false),
        shape!(past, 32, false),
        shape!(overlap, 128, true),
        shape!(back, 32, false),
        shape!(bytes, 128, false),
        shape!(halves, 128, false),
        shape!(doubles, 128, false),
        shape!(uniform, 128, false),
        shape!(atoms, 128, false),
        shape!(counts, 128, false),
        shape!(reduces, 128, false),
        shape!(copies, 128, false),
    ]
}

/// The blocks of a grid.
const BLOCKS: u32 = 16;

/// Runs each kernel of `shapes` once, on its own grid of whole warps, and waits for it.
fn run() -> Result {
    let cuda = Cuda::new()?;
    let kernels = shapes::Kernels::load(&cuda)?;
    let threads = launches().iter().map(|s| s.threads).max().unwrap_or(0);
    let bytes = 32 * usize::try_from(BLOCKS * threads)? + 4;
    let (x, y) = (cuda.alloc::<u8>(bytes)?, cuda.alloc::<u8>(bytes)?);
    for shape in launches() {
        (shape.launch)(&kernels, Grid::new(BLOCKS, shape.threads), x.ptr(), y.ptr())?;
        cuda.sync()?;
    }
    Ok(())
}

/// The kinds of access of global memory that the GPU counts apart, each by the name its metrics have and the
/// instructions of the listing that it counts: a copy by `cp.async` is counted with the loads, as is `ldu`.
const KINDS: [(&str, &[&str]); 4] =
    [("ld", &["ld", "ldu", "cp"]), ("st", &["st"]), ("atom", &["atom"]), ("red", &["red"])];

/// What the listing says of a kernel's accesses of each kind: how many it has, and how many sectors a warp moves in all
/// of them.
type Said = BTreeMap<&'static str, (u64, u64)>;

/// What the listing of `shapes` says of each kernel, by the kernel's name in the listing: the report that comes before
/// its `.entry`, an access to a line, as `1 sector for 4 bytes` and `4 sectors for 128 bytes, all used` say it.
fn said() -> Result<BTreeMap<String, Said>> {
    let mut said = BTreeMap::new();
    let mut report: Vec<&str> = vec![];
    for line in shapes::PTX.lines() {
        if let Some(entry) = line.strip_prefix(".visible .entry ") {
            let name = entry.split('(').next().ok_or("an entry has no name")?;
            let mut kernel = Said::new();
            for access in report.iter().filter_map(|l| l.strip_prefix("//   ")) {
                let Some((instruction, count)) = access.split_once(": ") else { continue };
                let op = instruction.split_whitespace().find(|word| !word.starts_with('@')).unwrap_or("");
                let op = op.split('.').next().unwrap_or("");
                // The lines of a pointer, `x: each byte is read at most once`, are no instruction's.
                let Some((kind, _)) = KINDS.iter().find(|(_, ops)| ops.contains(&op)) else { continue };
                let sectors: u64 = count
                    .split_whitespace()
                    .next()
                    .and_then(|n| n.parse().ok())
                    .ok_or_else(|| format!("{name}: `{access}` is not counted"))?;
                let (accesses, all) = kernel.entry(kind).or_default();
                *accesses += 1;
                *all += sectors;
            }
            said.insert(name.to_string(), kernel);
        }
        if line.starts_with("//") {
            report.push(line);
        } else {
            report.clear();
        }
    }
    Ok(said)
}

/// The fields of a line of CSV, in which a field is in quotes where it has commas.
fn fields(line: &str) -> Vec<String> {
    let mut fields = vec![String::new()];
    let mut quoted = false;
    for c in line.chars() {
        match (c, quoted) {
            ('"', _) => quoted = !quoted,
            (',', false) => fields.push(String::new()),
            (c, _) => {
                if let Some(field) = fields.last_mut() {
                    field.push(c);
                }
            }
        }
    }
    fields
}

/// What `ncu` counted of a launch: the warps of its grid, and each metric's value.
#[derive(Debug, Default)]
struct Counted {
    warps: u64,
    metrics: BTreeMap<String, u64>,
}

/// The dimensions in `(x, y, z)`, multiplied.
fn volume(dims: &str) -> Result<u64> {
    let dims = dims.trim_matches(|c| c == '(' || c == ')');
    dims.split(',').try_fold(1, |all, d| -> Result<u64> { Ok(all * d.trim().parse::<u64>()?) })
}

/// What `ncu` counted, from the CSV of its details page, by the name of the kernel: one launch of each.
fn counted(csv: &str) -> Result<BTreeMap<String, Counted>> {
    let mut lines = csv.lines().filter(|l| l.starts_with('"'));
    let header = fields(lines.next().ok_or("ncu printed no table")?);
    let column =
        |name: &str| header.iter().position(|h| h == name).ok_or_else(|| format!("ncu has no `{name}` column"));
    let (kernel, block, grid) = (column("Kernel Name")?, column("Block Size")?, column("Grid Size")?);
    let (metric, value) = (column("Metric Name")?, column("Metric Value")?);
    let mut all: BTreeMap<String, Counted> = BTreeMap::new();
    for line in lines {
        let row = fields(line);
        let get = |at: usize| row.get(at).map(String::as_str).ok_or("a row of ncu's is short");
        let launched = all.entry(get(kernel)?.to_string()).or_default();
        launched.warps = volume(get(grid)?)? * volume(get(block)?)?.div_ceil(32);
        launched.metrics.insert(get(metric)?.to_string(), get(value)?.replace(',', "").parse()?);
    }
    Ok(all)
}

/// The metric that counts the sectors, `what` being `sectors`, or the requests, `what` being `requests`, that the L1
/// serves for accesses of global memory of the kind `kind`.
fn metric(what: &str, kind: &str) -> String {
    format!("l1tex__t_{what}_pipe_lsu_mem_global_op_{kind}.sum")
}

/// What the test sets, for it to run the kernels, in the process `ncu` runs, and not start `ncu` again.
const PROFILED: &str = "MY_APP_PROFILED";

/// Nsight Compute's `ncu`, if CUDA is here: `$NCU`, or on the path, or in a toolkit under `/usr/local`.
fn ncu() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("NCU") {
        return Some(p.into());
    }
    if Command::new("ncu").arg("--version").output().is_ok_and(|o| o.status.success()) {
        return Some("ncu".into());
    }
    let mut found: Vec<PathBuf> = std::fs::read_dir("/usr/local")
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path().join("bin/ncu"))
        .filter(|p| p.exists())
        .collect();
    found.sort();
    found.pop()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What each kernel of `shapes` moves, as the listing says, is what `ncu` counts of the GPU, times the warps.
    /// Nsight Compute runs this test again, with `PROFILED` set, to run the kernels.
    #[test]
    fn ncu_counts_what_the_listing_says_a_warp_moves() -> Result {
        if std::env::var_os(PROFILED).is_some() {
            return run();
        }
        let Some(ncu) = ncu() else {
            eprintln!("no ncu here: skipped; set NCU to run it");
            return Ok(());
        };
        if !matches!(Cuda::count(), Ok(gpus) if gpus > 0) {
            eprintln!("no GPU: skipped");
            return Ok(());
        }
        let log = std::env::temp_dir().join(format!("my-app-ncu-{}.csv", std::process::id()));
        let metrics: Vec<String> =
            KINDS.iter().flat_map(|(kind, _)| [metric("sectors", kind), metric("requests", kind)]).collect();
        let out = Command::new(ncu)
            .args(["--csv", "--page", "details", "--print-kernel-base", "mangled", "--metrics", &metrics.join(",")])
            .arg("--log-file")
            .arg(&log)
            .arg(std::env::current_exe()?)
            .args(["--exact", "traffic::tests::ncu_counts_what_the_listing_says_a_warp_moves", "--nocapture"])
            .env(PROFILED, "1")
            .output()?;
        let said_by_ncu = String::from_utf8_lossy(&out.stdout) + String::from_utf8_lossy(&out.stderr);
        if said_by_ncu.contains("ERR_NVGPUCTRPERM") {
            eprintln!("ncu may not read the GPU's counters here: skipped. {said_by_ncu}");
            return Ok(());
        }
        assert!(out.status.success(), "{said_by_ncu}");
        let csv = std::fs::read_to_string(&log)?;
        std::fs::remove_file(&log)?;
        let (said, counted) = (said()?, counted(&csv)?);
        for Shape { name, threads, shares, .. } in launches() {
            let (entry, kernel) = said
                .iter()
                .find(|(k, _)| k.ends_with(&format!("{}{name}E", name.len())))
                .ok_or_else(|| format!("the listing has no kernel {name}"))?;
            let ncu = counted.get(entry).ok_or_else(|| format!("ncu did not count {name}"))?;
            assert_eq!(ncu.warps, u64::from(BLOCKS) * u64::from(threads / 32), "{name}: the grid's warps");
            assert!(!kernel.is_empty(), "{name}: the listing says nothing of its accesses");
            for (kind, _) in KINDS {
                let got = |what: &str| {
                    let metric = metric(what, kind);
                    ncu.metrics.get(&metric).copied().ok_or_else(|| format!("ncu has no {metric} of {name}"))
                };
                let (accesses, sectors) = kernel.get(kind).copied().unwrap_or_default();
                let (loaded, most) = (got("sectors")?, sectors * ncu.warps);
                if shares && kind == "ld" {
                    // A warp has one sector that is the first of the warp after it, which the GPU may serve once
                    // for both.
                    let least = most - ncu.warps;
                    assert!(
                        (least..=most).contains(&loaded),
                        "{name}: sectors loaded: {loaded}, not {least} to {most}"
                    );
                } else {
                    assert_eq!(loaded, most, "{name}: sectors of {kind}");
                }
                assert_eq!(got("requests")?, accesses * ncu.warps, "{name}: requests of {kind}");
            }
        }
        Ok(())
    }
}
