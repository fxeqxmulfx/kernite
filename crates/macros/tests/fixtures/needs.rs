//! Kernels that need CPU features past x86-64's own, three, two, one and none, whose `# Safety` says which, after the
//! memory of the module's that a kernel names. A generic kernel's says those of every instance, each once, in the order
//! of their names; a grid's kernel's launcher's, those of the kernel; and the `# Safety` that a kernel's docs have is
//! followed by them, with no second heading.

#[kernite_macros::kernels]
pub mod kernels {
    /// A word of the module's memory that a kernel names.
    pub static WORD: global<u64>;

    /// How many bits of `x` are set, lead and trail, summed.
    pub unsafe extern "sysv64" fn bits(x: u64) -> u64 {
        let sum: gpr;
        let count: gpr;
        popcnt(sum, x);
        lzcnt(count, x);
        add(sum, count);
        tzcnt(count, x);
        add(sum, count);
        ret(sum);
    }

    /// `y += a * x`, eight floats at once.
    pub unsafe extern "sysv64" fn axpy(y: *mut f32, a: *const f32, x: *const f32) {
        let scale: ymm;
        let sum: ymm;
        vbroadcastss(scale, [a]);
        vmovups(sum, [y]);
        vfmadd231ps(sum, scale, [x]);
        vmovups([y], sum);
    }

    /// How many bits of the integer at `x` are set.
    ///
    /// # Safety
    ///
    /// `x` points at an integer.
    pub unsafe extern "sysv64" fn counted(x: *const u64) -> u64 {
        let n: gpr;
        mov(n, [x]);
        popcnt(n, n);
        ret(n);
    }

    /// How many bits of `x` are set.
    pub unsafe extern "sysv64" fn ones(x: u64) -> u64 {
        popcnt(x, x);
        ret(x);
    }

    /// How many bits of `WORD` are set.
    pub unsafe extern "sysv64" fn word_bits() -> u64 {
        let n: gpr;
        mov(n, [WORD]);
        popcnt(n, n);
        ret(n);
    }

    /// `x` and one.
    pub unsafe extern "sysv64" fn next(x: u64) -> u64 {
        add(x, 1);
        ret(x);
    }

    /// How many zeros lead `x`, or how many of its bits are set.
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn pick<const N: u64>(x: u64) -> u64 {
        if N == 1 {
            lzcnt(x, x);
        } else {
            popcnt(x, x);
        }
        ret(x);
    }

    /// How many bits of `x` are set, or how many zeros lead it, when `N` is 2.
    #[instances(1, 2, 3)]
    pub unsafe extern "sysv64" fn count<const N: u64>(x: u64) -> u64 {
        if N == 2 {
            lzcnt(x, x);
        } else {
            popcnt(x, x);
        }
        ret(x);
    }

    /// How many bits are set of the word at `x`, once every thread of the grid is there.
    #[cooperative]
    pub unsafe extern "sysv64" fn together(x: *mut u64) {
        grid.sync();
        let g: gpr;
        mov(g, [x]);
        popcnt(g, g);
        mov([x], g);
    }
}
