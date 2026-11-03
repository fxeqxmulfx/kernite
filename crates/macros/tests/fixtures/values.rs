//! Consts the kernels read as values known now, which Rust reads as well: a module of consts of
//! numbers, floats, `bool`s, arrays of them and expressions of these, and a module of kernels
//! that takes them, where Rust sees them by the `use` the macro writes. Rust checks as it builds
//! that a kernel reads what Rust does.

#[kernite_macros::consts]
pub mod sizes {
    /// How far a value moves.
    pub const STEP: u64 = 3;
    /// Whether it moves.
    pub const MOVES: bool = true;
    /// One step and two.
    pub const STEPS: [u64; 2] = [STEP, 2 * STEP];
    /// Half, as a float.
    pub const HALF: f32 = 0.5;
}

#[kernite_macros::kernels(consts = crate::sizes)]
pub mod kernels {
    /// The furthest a value moves, of the steps the module takes.
    pub const FAR: u64 = STEPS[1];

    /// `x` moved as far as it goes.
    pub unsafe extern "sysv64" fn far(x: u64) -> u64 {
        if MOVES {
            add(x, FAR);
        }
        ret(x);
    }

    // Its instances are both values of a `bool`, so no value reaches the arm the macro writes for
    // an instance not listed. Rust says nothing of it.
    /// `x`, moved a step if `M`.
    #[instances(true, false)]
    pub unsafe extern "sysv64" fn moved<const M: bool>(x: u64) -> u64 {
        if M {
            add(x, STEP);
        }
        ret(x);
    }
}

// Rust's `FAR` is 6, and the kernel adds 6. Rust checks both as it builds, the listing as
// bytes: it compares those, and not yet `str`s.
const _: () = assert!(kernels::FAR == 6 && sizes::HALF == 0.5 && sizes::MOVES);
const _: () = assert!(matches!(
    kernels::LISTINGS[0].1.as_bytes(),
    b"# far(gpr) -> gpr
# frame: 0 bytes aligned to 0, saves []
# x=rdi,rax
    add rdi, 6
    mov rax, rdi
    ret
"
));
