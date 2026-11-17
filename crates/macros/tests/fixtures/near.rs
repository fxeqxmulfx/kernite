//! A static of 1 GiB, the most that is in reach of `rip`: Rust makes it as it makes any static, the linker takes it,
//! and a kernel names its last byte at a displacement. A second of 1 GiB after it in the module is past reach beside
//! it, so it is the linker's, and a kernel takes its address. Built under `deny(warnings)`, the crate says nothing, and
//! the kernels run.
#![deny(warnings)]

#[kernite_macros::kernels]
pub mod cpu {
    /// 1 GiB of zeros, in the `.bss` of the program.
    pub static NEAR: global<[u8; 1073741824]>;

    /// 1 GiB of zeros more, past reach beside `NEAR`: the linker's.
    pub static PAST: global<[u8; 1073741824]>;

    /// Writes 7 to the first byte of `NEAR` and 9 to the last, each at a displacement from `rip`.
    pub unsafe extern "sysv64" fn ends() {
        mov(byte[NEAR], 7);
        mov(byte[NEAR + 1073741823], 9);
    }

    /// Writes 2 to the last byte of `PAST`, by its address.
    pub unsafe extern "sysv64" fn past() {
        let p: gpr;
        lea(p, [PAST + 1073741823]);
        mov(byte[p], 2);
    }
}

#[cfg(test)]
mod tests {
    use super::cpu::{NEAR, PAST, ends, past};

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

    /// The kernel writes the last byte of the second static of 1 GiB, by its address, which the host reads back, and
    /// what it leaves alone is zeros.
    #[test]
    fn a_kernel_reaches_a_second_static_of_1_gib_by_its_address() {
        assert_eq!(PAST.len(), 1073741824);
        // SAFETY: `past` writes only the memory of the module it names, which nothing else touches.
        unsafe { past() };
        // SAFETY: Each address is in a static, which lives for the whole program, of atomics the kernel wrote.
        let at = unsafe { |i: usize| *PAST.at(i) };
        assert_eq!((at(0), at(1073741822), at(1073741823)), (0, 0, 2));
    }
}
