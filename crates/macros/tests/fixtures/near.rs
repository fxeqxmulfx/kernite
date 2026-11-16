//! A static of 1 GiB, the most that is in reach of `rip`: Rust makes it as it makes any static, the linker takes it,
//! and a kernel names its last byte at a displacement. Built under `deny(warnings)`, the crate says nothing, and the
//! kernel runs.
#![deny(warnings)]

#[kernite_macros::kernels]
pub mod cpu {
    /// 1 GiB of zeros, in the `.bss` of the program.
    pub static NEAR: global<[u8; 1073741824]>;

    /// Writes 7 to the first byte of `NEAR` and 9 to the last, each at a displacement from `rip`.
    pub unsafe extern "sysv64" fn ends() {
        mov(byte[NEAR], 7);
        mov(byte[NEAR + 1073741823], 9);
    }
}

#[cfg(test)]
mod tests {
    use super::cpu::{NEAR, ends};

    /// The kernel writes the first byte and the last byte of 1 GiB, which the host reads back, and what it leaves alone
    /// is zeros.
    #[test]
    fn a_kernel_reaches_a_static_of_1_gib_at_a_displacement() {
        assert_eq!((NEAR.len(), NEAR.is_empty()), (1073741824, false));
        // SAFETY: `ends` writes only the memory of the module it names, which nothing else touches.
        unsafe { ends() };
        // SAFETY: Each address is in a static, which lives for the whole program, of atomics the kernel wrote.
        let at = unsafe { |i: usize| *NEAR.at(i) };
        assert_eq!((at(0), at(1), at(536870912), at(1073741822), at(1073741823)), (7, 0, 0, 0, 9));
    }
}
