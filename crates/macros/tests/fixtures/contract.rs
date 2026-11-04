//! x86 kernels whose safety contracts say what memory of the module's nothing else touches
//! while they run: what each names, and what the functions it calls name. And GPU kernels whose
//! launchers say what a launch keeps to: their own contract, or the macros' where they have none.

#[kernite_macros::kernels]
pub mod cpu {
    /// How many times `run` has run.
    pub static RUNS: global<u64>;
    /// What `scale` multiplies by.
    pub static FACTORS: constant<[u64; 2]> = [3, 5];

    /// Counts a run.
    pub unsafe extern "sysv64" fn run() {
        let n: gpr;
        mov(n, [RUNS]);
        inc(n);
        mov([RUNS], n);
    }

    /// Does nothing.
    pub unsafe extern "sysv64" fn idle() {}

    /// Runs `idle` or `run`, whichever is at the address in `fp`.
    pub unsafe extern "sysv64" fn either(fp: u64) {
        call(fp, (idle, run));
    }

    /// `x[0] *= FACTORS[0]`, counted as a run.
    ///
    /// # Safety
    ///
    /// `x` points at an integer.
    pub unsafe extern "sysv64" fn scale(x: *mut u64) {
        let n: gpr;
        mov(n, [x]);
        imul(n, [FACTORS]);
        mov([x], n);
        call(run);
    }

    /// `x[t]` is how many times `run` has run, for each thread `t`.
    #[cooperative]
    pub unsafe extern "sysv64" fn runs(x: *mut u64) {
        let (t, n): (gpr, gpr);
        mov(t, grid.thread);
        mov(n, [RUNS]);
        mov([x + t * 8], n);
    }
}

#[kernite_macros::kernels]
pub mod gpu {
    /// Writes 1 at `x`.
    pub unsafe extern "ptx-kernel" fn one(x: *mut u32) {
        st.global.u32([x], 1);
    }

    /// Writes 2 at `x`.
    ///
    /// # Safety
    ///
    /// `x` points at an integer.
    pub unsafe extern "ptx-kernel" fn two(x: *mut u32) {
        st.global.u32([x], 2);
    }
}
