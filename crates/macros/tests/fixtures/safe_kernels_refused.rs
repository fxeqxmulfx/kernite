//! Rust refuses a call of a kernel where the kernel's contract is not kept: a safe kernel that needs the CPU feature
//! `avx` is called by a function that has it, or in an `unsafe` block, and by no other; and a kernel that is an
//! `unsafe fn` is called in an `unsafe` block, as it was.

#[kernite_macros::kernels]
pub mod cpu {
    /// The first of the four floats at `xs`.
    pub extern "sysv64" fn first(xs: &[f32; 4]) -> f32 {
        let v: xmm;
        vmovss(v, [xs]);
        ret(v);
    }

    /// The same, by `first`, which it calls: it needs `avx` as well.
    pub extern "sysv64" fn again(xs: &[f32; 4]) -> f32 {
        let v: xmm;
        call(v, first, [xs]);
        ret(v);
    }

    /// The first of the four integers at `xs`.
    pub unsafe extern "sysv64" fn load(xs: *const u64) -> u64 {
        let x: gpr;
        mov(x, [xs]);
        ret(x);
    }
}

/// A function with no features calls both.
pub fn without(xs: &[f32; 4]) -> f32 {
    cpu::first(xs) + cpu::again(xs)
}

/// A function that has `avx` calls both with no `unsafe`.
#[target_feature(enable = "avx")]
pub fn with(xs: &[f32; 4]) -> f32 {
    cpu::first(xs) + cpu::again(xs)
}

/// A function that has found `avx` calls them in an `unsafe` block, which says why it holds.
pub fn found(xs: &[f32; 4]) -> Option<f32> {
    if !is_x86_feature_detected!("avx") {
        return None;
    }
    // SAFETY: the CPU has `avx`, as the check above found.
    Some(unsafe { cpu::first(xs) + cpu::again(xs) })
}

/// A kernel that is an `unsafe fn` is called with `unsafe`, whatever the function that calls it has.
#[target_feature(enable = "avx")]
pub fn loads(xs: &[u64; 4]) -> u64 {
    cpu::load(xs.as_ptr())
}
