//! A module whose PTX is much: one kernel of 32768 instructions, which is 787 KB of PTX. `PTX` is a constant that Rust
//! builds as it builds this crate, and again in each crate that names it. Rust stops a constant that takes more than
//! 2 000 000 steps, as `long_running_const_eval`, an error unless the crate allows it, and this crate does not.

#[kernite_macros::kernels]
pub mod gpu {
    /// Adds a value to itself 32768 times.
    pub unsafe extern "ptx-kernel" fn big(x: *mut f32) {
        let v: f32;
        ld.global.f32(v, [x]);
        for _ in 0..32768 {
            add.rn.f32(v, v, v);
        }
        st.global.f32([x], v);
    }
}
