//! Raw strings where kernite takes a string: a note's text, `note(r#"say "hi""#)`, and a function's ABI,
//! `extern r"sysv64"` and `extern r#"ptx-kernel"#`. Rust reads each as a string, an ABI too, and the macro is given it
//! as written. Rust builds the crate without a word, and the tests find: each module written with raw strings has the
//! `LISTINGS`, or the PTX, of the one written with plain strings, with each note as its text says, and its kernels run
//! as theirs do. The tests that need a GPU skip where there is none.

/// x86 kernels with the ABI and the notes written plain: one with a note of each kind of text, and one of instances.
#[kernite_macros::kernels]
pub mod plain {
    pub const K: u64 = 3;

    /// `x` and `K` added.
    pub unsafe extern "sysv64" fn notes(x: u64) -> u64 {
        note("hi");
        note("say \"hi\", {K}, {{braces}}");
        note("a \"# b");
        note("a\\nb");
        add(x, K);
        ret(x);
    }

    /// `x` and `N` added.
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tile<const N: u64>(x: u64) -> u64 {
        note("tile {N}");
        add(x, N);
        ret(x);
    }
}

/// The same kernels, with the ABI and the notes written raw, with no `#` and with each number of them.
#[kernite_macros::kernels]
pub mod raw {
    pub const K: u64 = 3;

    /// `x` and `K` added.
    pub unsafe extern r"sysv64" fn notes(x: u64) -> u64 {
        note(r"hi");
        note(r#"say "hi", {K}, {{braces}}"#);
        note(r##"a "# b"##);
        note(r"a\nb");
        add(x, K);
        ret(x);
    }

    /// `x` and `N` added.
    #[instances(1, 2)]
    pub unsafe extern r#"sysv64"# fn tile<const N: u64>(x: u64) -> u64 {
        note(r"tile {N}");
        add(x, N);
        ret(x);
    }
}

/// A GPU kernel with the ABI and the note written plain.
#[kernite_macros::kernels]
pub mod plain_gpu {
    /// Every thread adds 1 to the count at `n`.
    pub unsafe extern "ptx-kernel" fn count(n: *mut u32) {
        note("one \"more\"");
        red.global.add.u32([n], 1);
    }
}

/// The same kernel, with the ABI and the note written raw.
#[kernite_macros::kernels]
pub mod raw_gpu {
    /// Every thread adds 1 to the count at `n`.
    pub unsafe extern r#"ptx-kernel"# fn count(n: *mut u32) {
        note(r#"one "more""#);
        red.global.add.u32([n], 1);
    }
}

/// The same checks for each module of x86 kernels.
#[cfg(test)]
macro_rules! x86_kernels_run {
    ($kernels:ident) => {
        // SAFETY: each kernel takes a number and gives one.
        unsafe {
            assert_eq!($kernels::notes(10), 13);
            assert_eq!(($kernels::tile::<1>(10), $kernels::tile::<2>(10)), (11, 12));
        }
    };
}

#[cfg(test)]
mod tests {
    use kernite_cuda::{Cuda, Grid};

    use super::{plain, plain_gpu, raw, raw_gpu};

    #[test]
    fn a_kernel_with_raw_strings_has_the_listing_of_the_same_kernel_with_plain_ones() {
        // `notes`, and two of `tile`.
        assert_eq!(plain::LISTINGS.len(), 3);
        assert_eq!(raw::LISTINGS, plain::LISTINGS);
        // Each note says what its string holds: a raw string has no escape.
        let (_, notes) = raw::LISTINGS.iter().find(|(name, _)| *name == "notes").unwrap();
        for note in ["# hi", "# say \"hi\", 3, {braces}", "# a \"# b", "# a\\nb"] {
            assert!(notes.lines().any(|line| line.trim() == note), "{note} in the listing:\n{notes}");
        }
    }

    #[test]
    fn a_kernel_with_plain_strings_runs() {
        x86_kernels_run!(plain);
    }

    #[test]
    fn a_kernel_with_raw_strings_runs() {
        x86_kernels_run!(raw);
    }

    #[test]
    fn a_gpu_module_with_raw_strings_is_the_ptx_of_the_one_with_plain_strings() {
        // The PTX names the kernel by its module's path, `9plain_gpu` or `7raw_gpu`, and differs in nothing else.
        assert!(plain_gpu::PTX.contains("// one \"more\"\n"), "the note in the PTX:\n{}", plain_gpu::PTX);
        assert_eq!(raw_gpu::PTX.replace("7raw_gpu", "9plain_gpu"), plain_gpu::PTX);
    }

    #[test]
    fn a_gpu_kernel_with_raw_strings_runs() {
        let Ok(cuda) = Cuda::new().map_err(|e| eprintln!("no GPU: skipped. {e}")) else { return };
        let kernels = raw_gpu::Kernels::load(&cuda).unwrap();
        let mut count = cuda.alloc::<u32>(1).unwrap();
        count.zero().unwrap();
        // SAFETY: the kernel adds to the count it is given, which lives until the launch has run.
        unsafe { kernels.count(Grid { blocks: [2, 1, 1], threads: [64, 1, 1], shared: 0 }, count.ptr()).unwrap() };
        cuda.sync().unwrap();
        assert_eq!(count.read().unwrap(), [128]);
    }
}
