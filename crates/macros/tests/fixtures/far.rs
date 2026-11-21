//! Statics of more than 1 GiB, which are far from `rip`, and one in reach beside them. The linker defines each far one,
//! so the crate builds as fast as with a small static, and a kernel takes the address of one with `lea(p, [BIG])` and
//! indexes it. A static of one byte is far too, beside one of 1 GiB that `cfg` leaves out, which counts all the same.
//! Built under `deny(warnings)`, the crate says nothing, and the kernels run.
#![deny(warnings)]

#[kernite_macros::kernels]
pub mod cpu {
    /// 3 GiB of zeros, the linker's, at a page.
    pub static BIG: global<[u8; 3221225472], 4096>;

    /// 2 GiB of `i16`s with a raw name, which only this crate sees.
    #[allow(non_upper_case_globals)]
    pub(crate) static r#type: global<[i16; 1073741824], 2>;

    /// Left out by `cfg`, so the linker is given no symbol for it.
    #[cfg(any())]
    pub static GONE: global<[u8; 3221225472]>;

    /// Counts how many times `ends` ran, in reach of `rip`.
    pub static RUNS: global<[u64; 1]>;

    /// Just short of 2 GiB: a static of Rust's of that size fails to link, so it is far. It takes `#[used]`, as a
    /// static in reach does.
    #[used]
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

/// A static that `cfg` leaves out counts all the same: `B`, of one byte, is past reach beside `A`, the linker's.
#[kernite_macros::kernels]
pub mod left_out {
    /// Left out: 1 GiB, all that is in reach.
    #[cfg(any())]
    pub static A: global<[u8; 1073741824]>;

    /// One byte, past reach beside `A`, as it is where `A` is kept.
    pub static B: global<[u8; 1]>;

    /// Writes 3 to `B`, by its address.
    pub unsafe extern "sysv64" fn touch() {
        let p: gpr;
        lea(p, [B]);
        mov(byte[p], 3);
    }
}

#[cfg(test)]
mod tests {
    use super::cpu::{BIG, EDGE, RUNS, edge, ends, middle, r#type};
    use super::left_out;

    /// The kernel writes the first byte and the last byte of 3 GiB, which the host reads back, and the middle, and what
    /// it leaves alone is zeros. A static of 2 GiB with a raw name is reached the same way, and so is one of 2147483647
    /// bytes, at its last byte.
    #[test]
    fn a_kernel_reaches_a_static_of_3_gib_by_its_address() -> Result<(), Box<dyn std::error::Error>> {
        // To Rust it is a reference to the linker's memory.
        let _: &'static kernite_cpu::Global<u8, 3221225472, 4096> = BIG;
        assert_eq!((BIG.len(), BIG.is_empty(), r#type.len()), (3221225472, false, 1073741824));
        // SAFETY: `ends` and `middle` write only the memory of the module they name, which nothing else touches.
        unsafe {
            ends(3221225471);
            middle();
        }
        // SAFETY: Each address is in a static, which lives for the whole program, of atomics the kernels wrote.
        let at = |i: usize| BIG.at(i).map(|p| unsafe { *p }).ok_or("at gave None");
        // SAFETY: As before.
        let of_type = |i: usize| r#type.at(i).map(|p| unsafe { *p }).ok_or("at gave None");
        assert_eq!((at(0)?, at(3221225471)?, of_type(1073741823)?), (7, 9, -1));
        assert_eq!((at(8)?, at(2147483656)?), (6, 5));
        assert_eq!((at(1)?, at(7)?, at(9)?, at(1610612736)?, at(3221225470)?), (0, 0, 0, 0, 0));
        assert_eq!(of_type(0)?, 0);
        assert_eq!(RUNS.read(), [1]);
        // SAFETY: As before.
        unsafe { ends(100) };
        assert_eq!((at(0)?, at(100)?, at(3221225471)?, RUNS.read()[0]), (7, 9, 9, 2));
        assert_eq!((BIG.ptr().addr() % 4096, r#type.ptr().addr() % 2), (0, 0));
        // SAFETY: `edge` writes only the memory of the module it names, which nothing else touches.
        unsafe { edge() };
        // SAFETY: Each address is in a static, which lives for the whole program, of atomics the kernel wrote.
        let at = |i: usize| EDGE.at(i).map(|p| unsafe { *p }).ok_or("at gave None");
        assert_eq!((EDGE.len(), at(0)?, at(2147483646)?), (2147483647, 0, 4));
        Ok(())
    }

    /// The kernel writes `B`, of one byte, past reach beside a static that `cfg` leaves out, by its address, which the
    /// host reads back.
    #[test]
    fn a_static_beside_one_cfg_leaves_out_is_the_linkers_all_the_same() {
        // SAFETY: `touch` writes only the memory of the module it names, which nothing else touches.
        unsafe { left_out::touch() };
        // SAFETY: The address is in a static, which lives for the whole program, of atomics the kernel wrote.
        assert_eq!(left_out::B.at(0).map(|p| unsafe { *p }), Some(3));
    }
}
