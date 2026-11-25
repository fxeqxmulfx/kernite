//! What a kernel's memory moves, which the listing says before the kernel: for each load, store, atomic and copy of
//! global memory, the sectors of 32 bytes that a warp moves against the bytes its threads use, and for each pointer,
//! how many times at most the launch reads a byte of it. [`Numbers`] says where each access is, whether the kernel has
//! `unsafe` or not, and this counts. GUIDE.md says what it takes as given.

use std::collections::BTreeSet;

use kernite_build::Quantity;
use kernite_lang::{Mem, Operand as Given};

use crate::forms::Space;
use crate::numbers::{Numbers, Poly, Step, gcd};
use crate::var::Var;

/// The bytes of a sector, which memory serves a warp's access in.
const SECTOR: i128 = 32;

/// The threads of a warp, which are consecutive in `tid.x` and run an access together.
const WARP: i128 = 32;

/// An access of global memory that kernite followed to a pointer: where it reaches, as [`Poly::parts`] splits the bytes
/// from where the pointer points.
#[derive(Clone, Debug, PartialEq, Eq)]
struct At {
    /// The pointer parameter, by its place among the parameters.
    pointer: usize,
    /// The bytes one thread reaches at once.
    bytes: i128,
    /// The launch's numbers that are in the offset, the same for each thread.
    launch: Poly,
    /// The bytes that are in the offset besides, the same for each thread.
    shift: i128,
    /// The offset's step from one block to the next, as the launch's numbers say.
    block: Poly,
    /// The offset's step from one thread to the next, in bytes.
    step: i128,
}

/// Why kernite does not count an access, and the pointers it is of, as far as it knows them.
type Missed = (Vec<usize>, String);

/// An access of global memory: a load, a store, an atomic, a copy by `cp.async`, or a call that may do any of them.
#[derive(Debug)]
struct Access {
    /// The instruction as it is listed, with no `;`.
    text: String,
    /// Whether it reads what is at its address: a load, an atomic, a copy, or a call.
    read: bool,
    /// Where it reaches, or why kernite does not count it.
    at: Result<At, Missed>,
}

/// The accesses of global memory of a kernel so far.
#[derive(Debug, Default)]
pub(crate) struct Traffic {
    accesses: Vec<Access>,
    /// Whether the kernel has a jump.
    jumps: bool,
}

impl Traffic {
    /// Takes in the instruction `i`, which is listed as `line`, before `n` follows it.
    pub(crate) fn step(&mut self, n: &Numbers, i: &Step, line: &str) {
        self.jumps |= matches!(i.bound.form.op, "bra" | "brx");
        if let Some((read, m, space)) = global(i) {
            self.accesses.push(Access { text: listed(line), read, at: locate(n, i, m, space) });
        }
    }

    /// Takes in the call listed as `line`, which is passed the operands `passed`, before `n` follows it. kernite does
    /// not follow a function, so what it moves is not counted, and no pointer it is passed an address of is told.
    pub(crate) fn call(&mut self, n: &Numbers, passed: &[&Given<Var>], line: &str) {
        let pointers = n.pointers_in(passed.iter().copied());
        if !pointers.is_empty() {
            let why = "kernite does not follow a function a kernel calls".to_string();
            self.accesses.push(Access { text: listed(line), read: true, at: Err((pointers, why)) });
        }
    }

    /// The lines the listing says before the kernel: none if it moves no global memory.
    pub(crate) fn report(&self, n: &Numbers) -> Vec<String> {
        if self.accesses.is_empty() {
            return vec![];
        }
        if self.jumps {
            return vec![
                "kernite does not follow a kernel that jumps yet: what its memory moves is not counted".into(),
            ];
        }
        let mut lines = vec![format!("A warp of {WARP} threads moves, in sectors of {SECTOR} bytes:")];
        for a in &self.accesses {
            lines.push(match &a.at {
                Ok(at) => format!("  {}: {}", a.text, moves(at)),
                Err((_, why)) => format!("  {}: not counted, as {why}", a.text),
            });
        }
        for (&pointer, p) in &n.pointers {
            let loads = self.accesses.iter().filter(|a| a.read);
            let followed: Vec<&At> =
                loads.clone().filter_map(|a| a.at.as_ref().ok()).filter(|at| at.pointer == pointer).collect();
            let missed = loads.filter(|a| matches!(&a.at, Err((qs, _)) if qs.contains(&pointer))).count();
            if followed.is_empty() && missed == 0 {
                continue;
            }
            let says = match missed {
                0 => reads(&followed),
                _ => Err("a load of it is not counted".into()),
            };
            lines.push(match says {
                Ok(says) => format!("  {}: {says}", p.name),
                Err(why) => format!("  {}: not told how many times a byte is read, as {why}", p.name),
            });
        }
        lines
    }
}

/// The instruction as it is listed, with no `;`.
fn listed(line: &str) -> String {
    line.strip_suffix(';').unwrap_or(line).to_string()
}

/// How `i` reaches global memory at an address, if it does and the report has it: whether it reads what is there, the
/// address, and the space the address is in, global or generic. The report has what a warp moves of `ld`, `ldu`, `st`,
/// `atom`, `red` and `cp.async`, and says that it does not count `wmma`'s yet. The rest of what reaches global memory
/// has no line: `prefetch`, `prefetchu`, `applypriority` and `discard` move no value, and `tex`, `tld4`, `suld`, `sust`
/// and `sured` reach it by a handle.
fn global<'a>(i: &Step<'a>) -> Option<(bool, &'a Mem<Var>, Space)> {
    let read = match i.bound.form.op {
        "ld" | "ldu" | "atom" | "red" => true,
        "cp" if i.bound.has("global") => true,
        "st" => false,
        "wmma" => i.bound.has("load"),
        _ => return None,
    };
    let addresses = i.args.iter().filter_map(|a| if let Given::Mem(m) = a { Some(m) } else { None });
    let (m, space) = addresses.zip(i.bound.spaces()).find(|(_, s)| matches!(s, Space::Global | Space::Generic))?;
    Some((read, m, space))
}

/// Where the access `i`, at the address `m` in `space`, is, if kernite follows it to a pointer and a step of the
/// thread's index; or why not.
fn locate(n: &Numbers, i: &Step, m: &Mem<Var>, space: Space) -> Result<At, Missed> {
    let found = n.address(m);
    let pointers = || match &found {
        Ok((pointer, _)) => vec![*pointer],
        Err((pointer, _)) => pointer.iter().copied().collect(),
    };
    if i.bound.form.op == "wmma" {
        let what = if i.bound.has("load") { "load" } else { "store" };
        return Err((pointers(), format!("kernite does not count what `wmma.{what}` moves yet")));
    }
    if space == Space::Generic {
        let why = "it reaches memory by a generic address, which may be shared or local memory as well as global";
        return Err((pointers(), why.into()));
    }
    if matches!(m.base, Some(Var::Mem(_))) {
        return Err((vec![], "it reaches memory that the module declares, which kernite does not follow yet".into()));
    }
    let (pointer, offset) = found.map_err(|(pointer, why)| (pointer.into_iter().collect(), why))?;
    let lost = |why: &str| (vec![pointer], why.to_string());
    let (shift, block, thread) = offset
        .parts()
        .ok_or_else(|| lost("its address is not a sum of steps of its thread's index and its block's"))?;
    let step =
        thread.constant().ok_or_else(|| lost("its step from one thread to the next is a number the launch gives"))?;
    let (launch, shift) = shift.split();
    // Where a warp starts moves from one block to the next by what the block's step is: `ntid.x` times anything is a
    // multiple of a warp's threads, and where kernite knows no better, any other number of the launch's is one of a
    // sector's bytes.
    let (_, by) = block.split();
    if step != 0 && by % SECTOR != 0 {
        return Err(lost(&format!("the start of its warps moves by {by} bytes from one block to the next")));
    }
    Ok(At { pointer, bytes: i128::from(i.reach), launch, shift, block, step })
}

/// What one warp moves of `a`, as a line says.
fn moves(a: &At) -> String {
    let start = a.shift.rem_euclid(SECTOR);
    let lanes = (0..WARP).map(|lane| start + a.step * lane);
    let mut spans: Vec<(i128, i128)> = lanes.map(|from| (from, from + a.bytes)).collect();
    let touched = |&(from, to): &(i128, i128)| from.div_euclid(SECTOR)..=(to - 1).div_euclid(SECTOR);
    let sectors = i128::try_from(spans.iter().flat_map(touched).collect::<BTreeSet<_>>().len()).unwrap_or(i128::MAX);
    spans.sort_unstable();
    // The bytes the threads use, each once, even where two threads use it.
    let (mut used, mut end) = (0, i128::MIN);
    for (from, to) in spans {
        used += (to - from.max(end)).max(0);
        end = end.max(to);
    }
    let line = format!("{} for {}", counted(sectors, "sector"), counted(used, "byte"));
    match fraction(used, sectors * SECTOR) {
        Some(part) => format!("{line}, {part} used"),
        None => line,
    }
}

/// `n` of `noun`, which takes an `s` unless `n` is 1: `1 sector`, `4 sectors`.
fn counted(n: i128, noun: &str) -> String {
    let s = if n == 1 { "" } else { "s" };
    format!("{n} {noun}{s}")
}

/// How much of the `moved` bytes the `used` are, in words, if it is one a word says: all, half, or a number of
/// quarters, eighths, sixteenths or thirty-seconds.
fn fraction(used: i128, moved: i128) -> Option<String> {
    let g = gcd(used, moved).max(1);
    let (p, q) = (used / g, moved / g);
    let name = match q {
        1 => return Some("all".into()),
        2 => return Some("half".into()),
        4 => "quarter",
        8 => "eighth",
        16 => "sixteenth",
        32 => "thirty-second",
        _ => return None,
    };
    Some(match (p, q) {
        (1, 8) => "an eighth".into(),
        (1, _) => format!("a {name}"),
        _ => format!("{p} {name}s"),
    })
}

/// Whether `p` is not less than 0 whatever the launch gives, as each of its factors is not less than 0.
fn at_least_0(p: Option<Poly>) -> bool {
    p.is_some_and(|p| p.0.values().all(|&c| c >= 0))
}

/// How many times at most the launch reads a byte of a pointer, as its `loads` say, or why kernite cannot tell.
fn reads(loads: &[&At]) -> Result<String, String> {
    let Some(first) = loads.first() else { return Err("it has no load".into()) };
    if loads.iter().any(|l| (&l.block, l.step) != (&first.block, first.step)) {
        return Err("its loads step differently, which kernite does not compare".into());
    }
    if loads.iter().any(|l| l.launch != first.launch) {
        return Err("its loads start apart by a number the launch gives, which kernite does not compare".into());
    }
    let step = first.step.abs();
    // How many times the loads of one thread, and of the threads in step with it, read the byte `b`.
    let covers = |b: i128| -> i128 {
        loads
            .iter()
            .map(|l| match step {
                0 => i128::from(l.shift <= b && b < l.shift + l.bytes),
                _ => (b - l.shift).div_euclid(step) - (b - l.shift - l.bytes).div_euclid(step),
            })
            .sum()
    };
    let most = loads.iter().map(|l| covers(l.shift)).max().unwrap_or(1);
    let (blocks_differ, threads_differ) = (!first.block.0.is_empty(), first.step != 0);
    // A block's threads reach from the least start of its loads to the most end, `span` bytes on.
    let start = loads.iter().map(|l| l.shift).min().unwrap_or(0);
    let span = loads.iter().map(|l| l.shift + l.bytes).max().unwrap_or(0) - start;
    // Where a block's step is `ntid.x` steps of a thread, the grid's threads are one run of the step, which `covers`
    // counts whole. Other blocks read bytes apart if a block's step is not less than the bytes its threads cover.
    let threads = Poly::launch(Quantity::Threads).scale(step);
    let run = threads.as_ref() == Some(&first.block);
    let cover = threads.and_then(|w| w.add(&Poly::int((span - step).max(0))));
    let apart = match (blocks_differ, threads_differ) {
        (false, _) => true,
        (true, true) => run || at_least_0(cover.and_then(|w| first.block.sub(&w))),
        (true, false) => at_least_0(first.block.sub(&Poly::int(span))),
    };
    if !apart {
        return Err("its blocks may read the same bytes".into());
    }
    let who = match (blocks_differ, threads_differ) {
        (false, true) => " by every block",
        (false, false) => " by every thread",
        (true, false) => " by every thread of a block",
        (true, true) => "",
    };
    let times = match most {
        1 => "once".to_string(),
        2 => "twice".to_string(),
        n => format!("{n} times"),
    };
    Ok(format!("each byte is read at most {times}{who}"))
}
