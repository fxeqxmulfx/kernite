//! Modules of x86 and of GPU kernels as an editor opens them, for `tests/hover.rs` to hover at the instructions a
//! kernel names.

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

    /// Not built: `cfg` leaves it out, and with it the docs of the instructions only it names.
    #[cfg(any())]
    pub unsafe extern "sysv64" fn left_out(a: *mut f32, b: *const f32) {
        let (x, y): (ymm, ymm);
        vmovups(x, [a]);
        vmovups(y, [b]);
        vsubps(x, x, y);
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

    /// Multiplies the half floats that the warp's threads hold, `a` and `b` of each, and writes the first float of the
    /// result that this thread holds to `out`.
    pub unsafe extern "ptx-kernel" fn tile(a: u32, b: u32, out: *mut f32) {
        let (a0, a1, a2, a3, b0, b1): (b32, b32, b32, b32, b32, b32);
        mov.b32(a0, a);
        mov.b32(a1, a);
        mov.b32(a2, a);
        mov.b32(a3, a);
        mov.b32(b0, b);
        mov.b32(b1, b);
        let (p, q, r, s): (f32, f32, f32, f32);
        mov.f32(p, 0.0);
        mov.f32(q, 0.0);
        mov.f32(r, 0.0);
        mov.f32(s, 0.0);
        mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32((p, q, r, s), (a0, a1, a2, a3), (b0, b1), (p, q, r, s));
        st.global.f32([out], p);
    }

    /// Not built: `cfg` leaves it out, and with it the docs of the instructions only it names.
    #[cfg(any())]
    pub unsafe extern "ptx-kernel" fn left_out(y: *mut f32) {
        let v: f32;
        ld.global.f32(v, [y]);
        sub.rn.f32(v, v, v);
        st.global.f32([y], v);
    }
}
