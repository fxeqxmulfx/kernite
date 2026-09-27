use kgen_core::{Class, Machine, PReg};

const fn gpr(hw: u8) -> PReg {
    PReg { class: Class::Gpr, hw }
}

const fn vec(hw: u8) -> PReg {
    PReg { class: Class::Vec, hw }
}

pub(crate) const RSP: PReg = gpr(4);
pub(crate) const RBP: PReg = gpr(5);

/// Where the integer and pointer parameters arrive, in order.
pub(crate) const INT_PARAMS: [PReg; 6] = [gpr(7), gpr(6), gpr(2), gpr(1), gpr(8), gpr(9)];

/// Where a float result goes back.
pub(crate) const FLOAT_RESULT: PReg = vec(0);

/// The caller-saved registers first, so that a kernel that fits in them saves nothing. The
/// stack pointer is never handed out, and neither is `rbp`, which holds the caller's stack
/// pointer while a frame is realigned.
const GPRS: [PReg; 14] = [
    gpr(0),
    gpr(1),
    gpr(2),
    gpr(6),
    gpr(7),
    gpr(8),
    gpr(9),
    gpr(10),
    gpr(11),
    gpr(3),
    gpr(12),
    gpr(13),
    gpr(14),
    gpr(15),
];

const VECS: [PReg; 16] = [
    vec(0),
    vec(1),
    vec(2),
    vec(3),
    vec(4),
    vec(5),
    vec(6),
    vec(7),
    vec(8),
    vec(9),
    vec(10),
    vec(11),
    vec(12),
    vec(13),
    vec(14),
    vec(15),
];

const SAVED: [PReg; 5] = [gpr(3), gpr(12), gpr(13), gpr(14), gpr(15)];

/// x86-64 under the System V calling convention: sixteen general registers and sixteen
/// vector ones, every vector register the caller's to save.
#[derive(Clone, Copy, Debug, Default)]
pub struct SysV;

impl Machine for SysV {
    fn allocatable(&self, class: Class) -> &[PReg] {
        match class {
            Class::Gpr => &GPRS,
            Class::Vec => &VECS,
        }
    }

    fn callee_saved(&self) -> &[PReg] {
        &SAVED
    }
}
