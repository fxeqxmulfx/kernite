//! Clippy tells of a public x86 kernel whose docs have no `# Safety`, at the kernel, as it tells of an `unsafe fn` of
//! Rust's own: of a plain one, one that gives back a number, and one of instances. It does not tell of a kernel whose
//! docs have one, nor of a kernel that names a static of its module, plain or of instances, whose contract the macros
//! write as one, nor of a kernel that needs a CPU feature past x86-64's own, `avx` here, an `unsafe fn` or not, plain
//! or of instances, whose features the macros write into one, nor of a cooperative kernel, whose launcher has the
//! macros'. A GPU kernel is not told of either: its launcher has the macros' `# Safety`.

/// x86 kernels, in each form, and a static that one names.
#[kernite_macros::kernels]
pub mod cpu {
    /// A static that a kernel names.
    pub static NEAR: global<[u32; 4]>;

    /// A kernel with no `# Safety`.
    pub unsafe extern "sysv64" fn plain() {}

    /// A kernel that gives back a number, with no `# Safety`.
    pub unsafe extern "sysv64" fn number(x: u64) -> u64 {
        ret(x);
    }

    /// Kernels of instances, with no `# Safety`.
    #[instances(2, 3)]
    pub unsafe extern "sysv64" fn many<const N: u64>(x: u64) -> u64 {
        let y: gpr;
        lea(y, [x + N]);
        ret(y);
    }

    /// A kernel with a `# Safety`.
    ///
    /// # Safety
    ///
    /// The kernel reads and writes nothing.
    pub unsafe extern "sysv64" fn documented() {}

    /// A kernel that names a static of its module.
    pub unsafe extern "sysv64" fn names() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Kernels of instances that name a static of their module.
    #[instances(2, 3)]
    pub unsafe extern "sysv64" fn names_many<const N: u64>() {
        let p: gpr;
        lea(p, [NEAR + N]);
    }

    /// A kernel with a `# Safety` that names a static of its module.
    ///
    /// # Safety
    ///
    /// The kernel reads nothing.
    pub unsafe extern "sysv64" fn documented_names() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// A cooperative kernel.
    #[cooperative]
    pub unsafe extern "sysv64" fn together() {}

    /// A kernel that needs `avx`, with no `# Safety`.
    pub unsafe extern "sysv64" fn needs(p: *mut f32) {
        let v: ymm;
        vmovups(v, [p]);
        vmovups([p], v);
    }

    /// A kernel that is no `unsafe fn` and needs `avx`, with no `# Safety`: Rust has it as a function that is
    /// unsafe to call where the caller does not have the feature.
    pub extern "sysv64" fn needs_safely(xs: &mut [f32; 8]) {
        let v: ymm;
        vmovups(v, [xs]);
        vmovups([xs], v);
    }

    /// Kernels of instances that need `avx`, with no `# Safety`.
    #[instances(2, 3)]
    pub unsafe extern "sysv64" fn needs_many<const N: u64>(p: *mut f32) {
        let v: ymm;
        vmovups(v, [p + N]);
        vmovups([p + N], v);
    }

    /// Kernels of instances that are no `unsafe fn` and need `avx`, with no `# Safety`.
    #[instances(8, 16)]
    pub extern "sysv64" fn needs_many_safely<const N: usize>(xs: &mut [f32; N]) {
        let v: ymm;
        vmovups(v, [xs]);
        vmovups([xs], v);
    }
}

/// A GPU kernel.
#[kernite_macros::kernels]
pub mod gpu {
    /// A kernel that does nothing.
    pub unsafe extern "ptx-kernel" fn idle() {}
}
