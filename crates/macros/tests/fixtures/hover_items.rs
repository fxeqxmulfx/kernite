//! A module of x86 kernels as an editor opens it, for `tests/hover.rs` to hover at the names a kernel writes.

#[kernite_macros::kernels]
pub mod kernels {
    /// The counts that `bump` adds one to: memory of the module's, kept from one call to the next.
    pub static COUNTS: global<[u32; 16]>;

    /// Adds one to the first of the `COUNTS`.
    pub unsafe extern "sysv64" fn bump() {
        let (bins, count): (gpr, gpr);
        lea(bins, [COUNTS]);
        mov(count.r32, [bins]);
        inc(count);
        mov([bins], count.r32);
    }
}
