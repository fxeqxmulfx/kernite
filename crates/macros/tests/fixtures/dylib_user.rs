//! A crate that links `dylib`, a crate of the `dylib` kind with a static in reach of `rip` and one past it: its test
//! calls the kernel, and names each static to read the bytes the kernel wrote.
#![deny(warnings)]

#[cfg(test)]
mod tests {
    use dylib::cpu::{BIG, NEAR, ends};

    /// The kernel writes the first byte of each static and the byte 1 on, which the host reads back.
    #[test]
    fn a_kernel_of_a_dylib_writes_its_statics_and_the_host_reads_them_back() {
        // SAFETY: `ends` writes only the memory of its module, which nothing else touches.
        unsafe { ends(1) };
        assert_eq!((NEAR.len(), &NEAR.read()[..3]), (4096, &[7, 9, 0][..]));
        // SAFETY: Each address is in a static, which lives for the whole program, of atomics the kernel wrote.
        let at = |i: usize| unsafe { *BIG.at(i) };
        assert_eq!((BIG.len(), [at(0), at(1), at(2)]), (3221225472, [7, 9, 0]));
    }
}
