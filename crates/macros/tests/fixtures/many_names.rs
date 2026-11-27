//! A module whose PTX has the name of its static 30000 times, once for each use in the kernel: 1.5 MB of PTX with 30000
//! holes in one part, which is more pieces than one constant of Rust's takes steps for: it stops at about 24000. The
//! macros cut the part in parts of a few thousand pieces, and `PTX` is them joined.

#[kernite_macros::kernels]
pub mod gpu {
    static COUNTS: global<[u32; 16]>;

    /// Counts 30000 times.
    pub unsafe extern "ptx-kernel" fn count() {
        let at: u64;
        for _ in 0..30000 {
            mov.u64(at, COUNTS);
        }
    }
}
