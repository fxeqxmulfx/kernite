//! A module whose PTX has the name of its static 100000 times, once for each use in the kernel: 5 MB of PTX with 100000
//! holes in one part, which is more pieces than one constant of Rust's takes steps for. The macros cut the part in
//! parts of a few thousand pieces, and `PTX` is them joined.

#[kernite_macros::kernels]
pub mod gpu {
    static COUNTS: global<[u32; 16]>;

    /// Counts 100000 times.
    pub unsafe extern "ptx-kernel" fn count() {
        let at: u64;
        for _ in 0..100000 {
            mov.u64(at, COUNTS);
        }
    }
}
