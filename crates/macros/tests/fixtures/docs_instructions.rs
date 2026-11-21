//! Modules of x86 and of GPU kernels that name instructions, as a crate builds, tests, documents and lints them: the
//! docs of the instructions, which an editor shows, are nothing to Rust.

#[kernite_macros::kernels]
pub mod kernels {
    /// The flags that `claim` sets a bit of: memory of the module's.
    pub static FLAGS: global<[u64; 4]>;

    /// Adds the eight floats of `b` to those of `a`.
    pub unsafe extern "sysv64" fn add_row(a: *mut f32, b: *const f32) {
        let (x, y): (ymm, ymm);
        vmovups(x, [a]);
        vmovups(y, [b]);
        vaddps(x, x, y);
        vmovups([a], x);
    }

    /// Sets bit `i` of the `FLAGS` in one atomic step, and copies it to `i` if it was clear.
    pub unsafe extern "sysv64" fn claim(i: u64) {
        let (flags, bit): (gpr, gpr);
        lea(flags, [FLAGS]);
        mov(bit, i);
        lock.bts([flags], i);
        cmovz(i, bit);
    }

    /// Built for a test and left out of the library.
    #[cfg(test)]
    pub unsafe extern "sysv64" fn in_tests(a: *mut f32, b: *const f32) {
        let (x, y): (ymm, ymm);
        vmovups(x, [a]);
        vmovups(y, [b]);
        vsubps(x, x, y);
        vmovups([a], x);
    }

    /// Left out always.
    #[cfg(any())]
    pub unsafe extern "sysv64" fn left_out(a: *mut f32, b: *const f32) {
        let (x, y): (ymm, ymm);
        vmovups(x, [a]);
        vmovups(y, [b]);
        vmulps(x, x, y);
        vmovups([a], x);
    }
}

#[kernite_macros::kernels]
pub mod gpu {
    /// Adds `a` times the float at `x` to the float at `y`.
    pub unsafe extern "ptx-kernel" fn axpy(a: f32, x: *const f32, y: *mut f32) {
        let (u, v): (f32, f32);
        ld.global.f32(u, [x]);
        ld.global.f32(v, [y]);
        fma.rn.ftz.f32(v, a, u, v);
        st.global.f32([y], v);
    }

    /// Writes to `out` the `v` of the lane whose index differs from this one's in the lowest bit.
    pub unsafe extern "ptx-kernel" fn neighbour(v: u32, out: *mut u32) {
        let w: u32;
        shfl.sync.bfly.b32(w, v, 1, 31, 0xffffffff);
        st.global.u32([out], w);
    }

    /// Built for a test and left out of the library.
    #[cfg(test)]
    pub unsafe extern "ptx-kernel" fn in_tests(y: *mut f32) {
        let v: f32;
        ld.global.f32(v, [y]);
        sub.rn.f32(v, v, v);
        st.global.f32([y], v);
    }

    /// Left out always.
    #[cfg(any())]
    pub unsafe extern "ptx-kernel" fn left_out(y: *mut f32) {
        let v: f32;
        ld.global.f32(v, [y]);
        mul.rn.f32(v, v, v);
        st.global.f32([y], v);
    }
}
