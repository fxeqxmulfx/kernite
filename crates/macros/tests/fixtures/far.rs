//! Statics of more than 1 GiB, which are far from `rip`, and one in reach beside them. The linker defines each far one,
//! so the crate builds as fast as with a small static, and a kernel takes the address of one with `lea(p, [BIG])` and
//! indexes it. Built under `deny(warnings)`, the crate says nothing, and the kernels run.
#![deny(warnings)]

#[kernite_macros::kernels]
pub mod cpu {
    /// 3 GiB of zeros, the linker's, at a page.
    pub static BIG: global<[u8; 3221225472], 4096>;

    /// 2 GiB of `i16`s with a raw name, which only this crate sees.
    pub(crate) static r#type: global<[i16; 1073741824], 2>;

    /// Left out by `cfg`, so the linker is given no symbol for it.
    #[cfg(any())]
    pub static GONE: global<[u8; 3221225472]>;

    /// Counts how many times `ends` ran, in reach of `rip`.
    pub static RUNS: global<[u64; 1]>;

    /// Just short of 2 GiB: a static of Rust's of that size fails to link, so it is far.
    pub static EDGE: global<[u8; 2147483647]>;

    /// Writes 7 to the first byte of `BIG` and 9 to the byte `last` bytes on, -1 to the last `i16` of `type`, and
    /// counts the run.
    pub unsafe extern "sysv64" fn ends(last: u64) {
        let p: gpr;
        let n: gpr;
        lea(p, [BIG]);
        mov(byte[p], 7);
        mov(byte[p + last], 9);
        lea(p, [r#type]);
        mov(word[p + 2147483646], -1);
        mov(n, [RUNS]);
        inc(n);
        mov([RUNS], n);
    }

    /// Writes 4 to the last byte of `EDGE`, 2147483646 bytes on from its address, which a displacement holds.
    pub unsafe extern "sysv64" fn edge() {
        let p: gpr;
        lea(p, [EDGE + 2147483646]);
        mov(byte[p], 4);
    }

    /// Writes 5 to the byte of `BIG` 2147483648 bytes in, which no displacement holds, and 6 to the byte 8 on.
    pub unsafe extern "sysv64" fn middle() {
        let p: gpr;
        let at: gpr;
        mov(at, 2147483648);
        lea(p, [BIG + 8]);
        mov(byte[p + at], 5);
        mov(byte[p], 6);
    }
}

#[cfg(test)]
mod tests {
    use super::cpu::{BIG, EDGE, RUNS, edge, ends, middle, r#type};

    /// The kernel writes the first byte and the last byte of 3 GiB, which the host reads back, and the middle, and what
    /// it leaves alone is zeros. A static of 2 GiB with a raw name is reached the same way, and so is one of 2147483647
    /// bytes, at its last byte.
    #[test]
    fn a_kernel_reaches_a_static_of_3_gib_by_its_address() {
        assert_eq!((BIG.len(), BIG.is_empty(), r#type.len()), (3221225472, false, 1073741824));
        // SAFETY: `ends` and `middle` write only the memory of the module they name, which nothing else touches.
        unsafe {
            ends(3221225471);
            middle();
        }
        // SAFETY: Each address is in a static, which lives for the whole program, of atomics the kernels wrote.
        let (at, last) = unsafe { (|i: usize| *BIG.at(i), *r#type.at(1073741823)) };
        assert_eq!((at(0), at(3221225471), last), (7, 9, -1));
        assert_eq!((at(8), at(2147483656)), (6, 5));
        assert_eq!((at(1), at(7), at(9), at(1610612736), at(3221225470)), (0, 0, 0, 0, 0));
        // SAFETY: As before.
        assert_eq!(unsafe { *r#type.at(0) }, 0);
        assert_eq!(RUNS.read(), [1]);
        // SAFETY: As before.
        unsafe { ends(100) };
        assert_eq!((at(0), at(100), at(3221225471), RUNS.read()[0]), (7, 9, 9, 2));
        assert_eq!((BIG.ptr().addr() % 4096, r#type.ptr().addr() % 2), (0, 0));
        // SAFETY: `edge` writes only the memory of the module it names, which nothing else touches.
        unsafe { edge() };
        // SAFETY: Each address is in a static, which lives for the whole program, of atomics the kernel wrote.
        let (first, last) = unsafe { (*EDGE.at(0), *EDGE.at(2147483646)) };
        assert_eq!((EDGE.len(), first, last), (2147483647, 0, 4));
    }
}
