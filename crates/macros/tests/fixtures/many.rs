//! Launchers of more than seven parameters, as Clippy counts them: a cooperative x86 kernel's, of
//! its eight and the grid, and a GPU kernel's, of its six, `self` and the grid; each in a module
//! that takes consts as well. Clippy says nothing of them, for each launcher allows what it says.

#[kgen_macros::consts]
pub mod sizes {
    /// How far a value moves.
    pub const STEP: u64 = 3;
}

#[kgen_macros::kernels]
pub mod cpu {
    #[cooperative]
    pub unsafe extern "sysv64" fn many(a: u64, b: u64, c: u64, d: u64, e: u64, f: f32, g: f32, h: f32) {
        add(a, b);
        add(c, d);
        add(e, a);
        vaddss(f, f, g);
        vaddss(f, f, h);
    }
}

#[kgen_macros::kernels]
pub mod gpu {
    pub unsafe extern "ptx-kernel" fn many(x: *mut f32, a: u32, b: u32, c: u32, d: u32, e: u32) {
        let at: u64;
        mov.u64(at, x);
        red.global.add.u32([at], a);
        red.global.add.u32([at], b);
        red.global.add.u32([at], c);
        red.global.add.u32([at], d);
        red.global.add.u32([at], e);
    }
}

#[kgen_macros::kernels(consts = crate::sizes)]
pub mod cpu_taking {
    #[cooperative]
    pub unsafe extern "sysv64" fn many(a: u64, b: u64, c: u64, d: u64, e: u64, f: f32, g: f32, h: f32) {
        add(a, STEP);
        add(c, d);
        add(e, a);
        vaddss(f, f, g);
        vaddss(f, f, h);
        add(b, b);
    }
}

#[kgen_macros::kernels(consts = crate::sizes)]
pub mod gpu_taking {
    pub unsafe extern "ptx-kernel" fn many(x: *mut f32, a: u32, b: u32, c: u32, d: u32, e: u32) {
        let at: u64;
        mov.u64(at, x);
        red.global.add.u32([at], a);
        red.global.add.u32([at], b);
        red.global.add.u32([at], c);
        red.global.add.u32([at], d);
        red.global.add.u32([at], e);
        red.global.add.u32([at], STEP);
    }
}
