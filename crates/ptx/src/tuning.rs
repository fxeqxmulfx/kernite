//! A kernel's tuning directives: what a launch of it looks like, for `ptxas` to compile it for.
//!
//! PTX has `.maxntid` and `.reqntid`, the threads of a block at most and exactly, along x, then y and
//! z; `.minnctapersm`, the blocks an SM should hold at once, at least; and `.maxnreg`, the registers
//! a thread may use. They go on an `.entry` between its parameters and its body. `.maxntid` and
//! `.maxnreg` came with PTX 1.3, `.minnctapersm` with 2.0 and `.reqntid` with 2.1, and every module
//! kernite writes is of `.version 3.0` at least, `Needs::FLOOR`, so a directive asks nothing of the
//! module's `.version` and `.target`.
//!
//! kernite takes the values `ptxas` takes and a launch can have, and refuses the rest, at the
//! directive, for `ptxas` cuts or ignores each of them without a word:
//!
//! - `.maxntid` and `.reqntid` take 1 to 3 sizes, each 1 or more, at most 1024 along x and along y
//!   and 64 along z, as a block has them, and 1024 threads in all. `ptxas` takes `.reqntid 32, 33`
//!   and `.reqntid 1, 1, 65`, which no launch matches, and ignores a `.maxntid` past 1024 threads.
//!   It dies of `SIGFPE` on `.maxntid 4294967295`, in ptxas 13.4 and in the driver's own compiler;
//!   kernite writes no size past 1024.
//! - A kernel has `.maxntid` or `.reqntid`, not both: `ptxas` refuses a module that has both.
//! - `.minnctapersm` is 1 to 16, with `.maxntid` or `.reqntid`, and no more than 32 warps in all.
//!   `ptxas` ignores it alone, and ignores a number of blocks, or of warps, that the GPU's SM does
//!   not hold: every GPU kernite writes for holds 16 blocks, and 32 warps, at least. It counts the
//!   warps of the blocks, a warp being 32 threads and the last of a block maybe fewer, and not
//!   their threads: 10 blocks of 96 threads are 30 warps, and 10 of 97 are 40, which Turing's SM
//!   does not hold, though they are 970 threads and its SM holds 1024.
//! - `.maxnreg` is 24 to 255. `ptxas` raises a smaller number to 24, and ignores a bigger. With
//!   `.maxntid` or `.reqntid` it is at most as many registers as a launch gives the threads of a
//!   block. A launch counts the registers of a thread by eights, 32 threads to a warp, and the
//!   warps of a block by fours, and every GPU kernite writes for gives a block 65536 at most:
//!   255 registers for 256 threads are 256 * 32 * 8, and for 257 threads 256 * 32 * 12. The ISA
//!   says `.maxnreg` may be further constrained by `.maxntid`, and ptxas 13.4 does so for
//!   `.maxntid 1024` alone, which gives a kernel 64 registers, and for `.minnctapersm`, but keeps
//!   `.maxnreg 128` beside `.maxntid 1024`: a kernel that uses its 128 registers has no block of
//!   1024 threads that can launch, and the driver says `CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES`.

use kernite_lang::Fault;

/// The most threads a block has along x, y and z, whatever the GPU, from Kepler's on.
const AXES: [(&str, i128); 3] = [("x", 1024), ("y", 1024), ("z", 64)];

/// The most threads a block has in all.
const THREADS: i128 = 1024;

/// The most blocks an SM is asked to hold at once: every GPU's holds 16 at least.
const BLOCKS: i128 = 16;

/// The threads of a warp.
const WARP: i128 = 32;

/// The most warps an SM is asked to hold at once: every GPU's holds 32 at least, Turing's.
const SM_WARPS: i128 = 32;

/// The fewest registers `ptxas` gives a thread, and the most a thread has.
const REGISTERS: (i128, i128) = (24, 255);

/// The most registers a block has: every GPU's gives a block 65536 at most.
const BLOCK_REGISTERS: i128 = 65536;

/// The tuning directives of a kernel's `.entry` as PTX writes them, `.maxntid 16, 16`, one for
/// each directive of `given`, by its name and the whole numbers it gives, in the order written;
/// or which of them is wrong, by its place among them, and what is wrong with it.
pub fn lines(given: &[(&str, Vec<i128>)]) -> Result<Vec<String>, (usize, Fault)> {
    for (i, (name, values)) in given.iter().enumerate() {
        match *name {
            "maxntid" | "reqntid" => block(name, values),
            "minnctapersm" => blocks(values),
            // `maxnreg`: the language has no other name.
            _ => registers(values),
        }
        .map_err(|fault| (i, fault))?;
    }
    let find = |name: &str| given.iter().position(|(n, _)| *n == name);
    // The threads of a block, if the kernel says them, and the one number of a directive that has one.
    let threads = find("maxntid").or(find("reqntid")).map(|i| given[i].1.iter().product::<i128>());
    let number = |name: &str| find(name).map(|i| (i, given[i].1[0]));
    if let (Some(max), Some(req)) = (find("maxntid"), find("reqntid")) {
        let fault = Fault::new(
            "a kernel has `#[maxntid(..)]` or `#[reqntid(..)]`, not both: its blocks have at most these threads, or \
             exactly these",
        );
        return Err((max.max(req), fault));
    }
    if let Some((i, n)) = number("minnctapersm") {
        let Some(threads) = threads else {
            let fault = Fault::new(
                "`#[minnctapersm(..)]` needs `#[maxntid(..)]` or `#[reqntid(..)]`: `ptxas` ignores it where it does \
                 not know the threads of a block",
            );
            return Err((i, fault));
        };
        let w = warps(threads);
        if n * w > SM_WARPS {
            let message = format!(
                "{n} blocks of {threads} threads make {} warps, {w} to a block: an SM holds {SM_WARPS} on some GPUs, \
                 and `ptxas` counts the warps of the blocks, not their threads, and ignores the number on a GPU whose \
                 SM holds fewer",
                n * w
            );
            return Err((i, Fault::at(0, message)));
        }
    }
    if let (Some((i, n)), Some(threads)) = (number("maxnreg"), threads) {
        // A launch counts the registers of a thread by eights, and the warps of a block by fours.
        let (r, w) = (up(n, 8), up(warps(threads), 4));
        if r * WARP * w > BLOCK_REGISTERS {
            let message = format!(
                "{n} registers for each of {threads} threads make {} in a launch, which counts a thread's registers \
                 by eights and a block's warps by fours, {r} * {WARP} * {w}: a block has {BLOCK_REGISTERS} at most, \
                 and `ptxas` does not lower `#[maxnreg(..)]` to fit the threads of a block, so a kernel that used \
                 them all could not launch",
                r * WARP * w
            );
            return Err((i, Fault::at(0, message)));
        }
    }
    let join = |values: &[i128]| values.iter().map(i128::to_string).collect::<Vec<_>>().join(", ");
    Ok(given.iter().map(|(name, values)| format!(".{name} {}", join(values))).collect())
}

/// `n`, 1 or more, rounded up to a multiple of `unit`.
fn up(n: i128, unit: i128) -> i128 {
    (n - 1) / unit * unit + unit
}

/// The warps of a block of `threads` threads, 1 or more: the last has fewer than 32 maybe.
fn warps(threads: i128) -> i128 {
    up(threads, WARP) / WARP
}

/// Whether `values`, the sizes of `maxntid` or `reqntid`, are the threads of a block: one size,
/// two or three, each in its axis's range, and not more than a block has in all.
fn block(name: &str, values: &[i128]) -> Result<(), Fault> {
    if values.is_empty() || values.len() > AXES.len() {
        let fault = format!("`#[{name}(..)]` gives one size, two or three: the threads along x, then y and z");
        return Err(if values.is_empty() { Fault::new(fault) } else { Fault::at(AXES.len(), fault) });
    }
    for (i, &n) in values.iter().enumerate() {
        let (axis, most) = AXES[i];
        if n < 1 {
            return Err(Fault::at(i, format!("{n} threads are too few: a block has 1 at least along each axis")));
        }
        if n > most {
            return Err(Fault::at(i, format!("{n} threads are too many: a block has {most} at most along {axis}")));
        }
    }
    let all: i128 = values.iter().product();
    if all > THREADS {
        return Err(Fault::new(format!("these sizes make {all} threads: a block has {THREADS} at most in all")));
    }
    Ok(())
}

/// Whether `values` are the blocks an SM should hold at once, `minnctapersm`'s one number.
fn blocks(values: &[i128]) -> Result<(), Fault> {
    let n = one("minnctapersm", "the blocks an SM should hold at once, at least", values)?;
    if n < 1 {
        return Err(Fault::at(0, format!("{n} blocks are too few: an SM holds 1 at least")));
    }
    if n > BLOCKS {
        let fault = format!(
            "{n} blocks are too many: an SM holds {BLOCKS} on some GPUs, and `ptxas` ignores the number on a GPU \
             whose SM holds fewer"
        );
        return Err(Fault::at(0, fault));
    }
    Ok(())
}

/// Whether `values` are the registers a thread may use, `maxnreg`'s one number.
fn registers(values: &[i128]) -> Result<(), Fault> {
    let n = one("maxnreg", "the most registers a thread may use", values)?;
    let (least, most) = REGISTERS;
    if n < least {
        let fault = format!(
            "{n} registers are too few: `ptxas` gives a thread {least} at least, and raises a smaller number to it"
        );
        return Err(Fault::at(0, fault));
    }
    if n > most {
        let fault =
            format!("{n} registers are too many: a thread has {most} at most, and `ptxas` ignores a bigger number");
        return Err(Fault::at(0, fault));
    }
    Ok(())
}

/// The number that `values` are for `name`, which gives `what`; or why they are none, or more.
fn one(name: &str, what: &str, values: &[i128]) -> Result<i128, Fault> {
    let fault = format!("`#[{name}(..)]` gives one number: {what}");
    match values {
        [n] => Ok(*n),
        [] => Err(Fault::new(fault)),
        _ => Err(Fault::at(1, fault)),
    }
}
