//! Kernels that need CPU features past x86-64's own, three, two, one and none, whose docs say
//! which: a generic kernel's say those of every instance, each once, in the order of their
//! names, and a grid's kernel's launcher's those of the kernel.

#[kernite_macros::kernels]
pub mod kernels {
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

    /// How many bits of `x` are set.
    pub unsafe extern "sysv64" fn ones(x: u64) -> u64 {
        popcnt(x, x);
        ret(x);
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
