//! Kernels that `cfg` keeps and that name an item `cfg` leaves out. Built without the feature `big`, which
//! keeps each item they name, each name is an error where it is written, as Rust says of a function it
//! leaves out: Rust's own, "cannot find value", of a const, a static of the process's and an x86 kernel that
//! takes no compile-time arguments, which are Rust's items, and kernite's of the rest, a helper, an
//! `extern "ptx-func"` function, an instance of an x86 kernel and a static of the GPU's, which are not.
//! Where `cfg` leaves out the kernel with what it names, nothing is said, nor of what a helper names, where
//! `cfg` leaves out the helper. Built with `big` and `wide`, the crate builds without a word, and the x86
//! kernels run.

#![deny(warnings)]

/// What Rust has a name for: a const, a static of the process's and an x86 kernel. Each name is an error
/// in a kernel's body, in its `#[instances(..)]`, in the type of a static and in its value, and in a const's
/// definition, which is Rust's own; a kernel of instances names what it names once for each place, whatever
/// its instances.
#[kernite_macros::kernels]
pub mod rust_names {
    #[cfg(feature = "big")]
    const SIZE: u64 = 4;

    #[cfg(feature = "big")]
    const LIST: u64 = 2;

    #[cfg(feature = "big")]
    const LEN: usize = 2;

    #[cfg(feature = "big")]
    pub static COUNT: global<[u64; 1]>;

    const TWICE: u64 = SIZE * 2;

    pub static WIDTHS: global<[u32; LEN]>;

    pub static START: global<u64> = SIZE;

    #[cfg(feature = "big")]
    pub unsafe extern "sysv64" fn eight() -> u64 {
        let v: gpr;
        mov(v, 8);
        ret(v);
    }

    pub unsafe extern "sysv64" fn reads_const(p: *mut u64) {
        mov(qword[p], SIZE);
    }

    pub unsafe extern "sysv64" fn reads_twice(p: *mut u64) {
        mov(qword[p], TWICE);
    }

    pub unsafe extern "sysv64" fn reads_static(p: *mut u64) {
        let v: gpr;
        mov(v, qword[COUNT]);
        mov(qword[p], v);
    }

    pub unsafe extern "sysv64" fn calls_kernel(p: *mut u64) {
        let g: gpr;
        call(g, eight, ());
        mov(qword[p], g);
    }

    #[instances(LIST)]
    pub unsafe extern "sysv64" fn listed<const N: u64>(p: *mut u64) {
        mov(qword[p], N);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tile<const N: u64>(p: *mut u64) {
        let v: gpr;
        mov(v, qword[COUNT]);
        add(v, SIZE);
        add(v, SIZE);
        add(v, N);
        mov(qword[p], v);
    }
}

/// What Rust has no name for: a helper, which is no Rust item, and an instance of an x86 kernel, which
/// Rust has hidden. kernite says the error, once for each place, in a kernel of instances too, and for a
/// helper that a helper calls.
#[kernite_macros::kernels]
pub mod kernite_names {
    #[cfg(feature = "big")]
    fn seven(p: gpr) {
        mov(qword[p], 7);
    }

    #[cfg(feature = "big")]
    fn inner(p: gpr) {
        mov(qword[p], 5);
    }

    fn outer(p: gpr) {
        inner(p);
    }

    #[cfg(feature = "big")]
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tile<const N: u64>(x: u64) -> u64 {
        add(x, N);
        ret(x);
    }

    pub unsafe extern "sysv64" fn calls_helper(p: *mut u64) {
        seven(p);
    }

    pub unsafe extern "sysv64" fn calls_through(p: *mut u64) {
        outer(p);
    }

    pub unsafe extern "sysv64" fn calls_instance(p: *mut u64) -> u64 {
        let r: gpr;
        mov(r, qword[p]);
        call(r, tile::<2>, r);
        ret(r);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn calls_helper_of_instances<const N: u64>(p: *mut u64) {
        seven(p);
        seven(p);
    }
}

/// A name in the body of a helper is said where `cfg` keeps the helper as well as the kernel. A kept kernel
/// that calls a helper `cfg` leaves out is told of the helper alone, though the helper names a const, a
/// static, a kernel and a helper that `cfg` leaves out too. One that calls a helper at two places, one in a
/// helper `cfg` leaves out, is told of what the helper names, once.
#[kernite_macros::kernels]
pub mod in_helpers {
    #[cfg(feature = "big")]
    const FOUR: u64 = 4;

    #[cfg(feature = "big")]
    pub static HELD: global<[u64; 1]>;

    #[cfg(feature = "big")]
    pub unsafe extern "sysv64" fn eight() -> u64 {
        let v: gpr;
        mov(v, 8);
        ret(v);
    }

    #[cfg(feature = "big")]
    fn last(p: gpr) {
        add(qword[p], 1);
    }

    #[cfg(feature = "big")]
    fn left_out(p: gpr) {
        let v: gpr;
        call(v, eight, ());
        add(v, FOUR);
        add(v, qword[HELD]);
        mov(qword[p], v);
        last(p);
    }

    fn twice(p: gpr) {
        add(qword[p], FOUR);
    }

    #[cfg(feature = "big")]
    fn calls_twice(p: gpr) {
        twice(p);
    }

    pub unsafe extern "sysv64" fn calls_left_out(p: *mut u64) {
        left_out(p);
    }

    pub unsafe extern "sysv64" fn calls_both_ways(p: *mut u64) {
        calls_twice(p);
        twice(p);
    }
}

/// The same for GPU kernels: a const is Rust's, in a kernel's body, in its directives and in the type of a
/// static of the GPU's; a function the GPU runs and a static of the GPU's have no Rust item, and kernite
/// says the error. A helper `cfg` leaves out is told of alone.
#[kernite_macros::kernels]
pub mod gpu_names {
    #[cfg(feature = "big")]
    const SIZE: u32 = 4;

    #[cfg(feature = "big")]
    const LEN: usize = 2;

    #[cfg(feature = "big")]
    pub static COUNT: global<[u64; 1]>;

    pub static WIDTHS: global<[u32; LEN]>;

    #[cfg(feature = "big")]
    extern "ptx-func" fn nine() -> u32 {
        let v: u32;
        mov.u32(v, 9);
        ret(v);
    }

    extern "ptx-func" fn via() -> u32 {
        let v: u32;
        call(v, nine, ());
        add.u32(v, v, SIZE);
        ret(v);
    }

    #[cfg(feature = "big")]
    fn left_out(p: u64) {
        let (g, at): (u32, u64);
        call(g, nine, ());
        add.u32(g, g, SIZE);
        mov.u64(at, COUNT);
        st.global.u32([p], g);
    }

    pub unsafe extern "ptx-kernel" fn reads_const(p: *mut u32) {
        st.global.u32([p], SIZE);
    }

    pub unsafe extern "ptx-kernel" fn calls_left_out(p: *mut u32) {
        left_out(p);
    }

    pub unsafe extern "ptx-kernel" fn reads_static(p: *mut u64) {
        let at: u64;
        mov.u64(at, COUNT);
        st.global.u64([p], at);
    }

    pub unsafe extern "ptx-kernel" fn calls_function(p: *mut u32) {
        let g: u32;
        call(g, nine, ());
        st.global.u32([p], g);
    }

    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn calls_function_of_instances<const K: u32>(p: *mut u32) {
        let g: u32;
        call(g, nine, ());
        st.global.u32([p], g);
    }

    #[maxntid(SIZE)]
    pub unsafe extern "ptx-kernel" fn limited(p: *mut u32) {
        st.global.u32([p], 1);
    }
}

/// Where `cfg` leaves out the kernel with what it names, or keeps it only where it keeps the item, nothing
/// is said.
#[kernite_macros::kernels]
pub mod together {
    #[cfg(feature = "big")]
    const SIZE: u64 = 4;

    #[cfg(feature = "big")]
    pub static COUNT: global<[u64; 1]>;

    #[cfg(feature = "big")]
    fn seven(p: gpr) {
        mov(qword[p], 7);
    }

    #[cfg(feature = "big")]
    pub unsafe extern "sysv64" fn eight() -> u64 {
        let v: gpr;
        mov(v, 8);
        ret(v);
    }

    #[cfg(feature = "big")]
    pub unsafe extern "sysv64" fn names_each(p: *mut u64) {
        let g: gpr;
        call(g, eight, ());
        add(g, SIZE);
        add(g, qword[COUNT]);
        seven(p);
        mov(qword[p], g);
    }

    #[cfg(all(feature = "big", feature = "wide"))]
    pub unsafe extern "sysv64" fn names_each_when_wide(p: *mut u64) {
        let g: gpr;
        call(g, eight, ());
        add(g, SIZE);
        add(g, qword[COUNT]);
        seven(p);
        mov(qword[p], g);
    }
}

/// A name kept only where `wide` holds as well, which a kernel `big` keeps names: it is gone where `big`
/// holds alone, and the error is said there, not where neither holds nor where both do. So is a helper
/// under a `cfg_attr(..)`, which an unconditional kernel names.
#[kernite_macros::kernels]
pub mod narrower {
    #[cfg(all(feature = "big", feature = "wide"))]
    const WIDE: u64 = 6;

    #[cfg(all(feature = "big", feature = "wide"))]
    fn six(p: gpr) {
        mov(qword[p], 6);
    }

    #[cfg_attr(feature = "big", cfg(feature = "wide"))]
    fn five(p: gpr) {
        mov(qword[p], 5);
    }

    #[cfg(feature = "big")]
    pub unsafe extern "sysv64" fn reads_wide(p: *mut u64) {
        mov(qword[p], WIDE);
    }

    #[cfg(feature = "big")]
    pub unsafe extern "sysv64" fn calls_six(p: *mut u64) {
        six(p);
    }

    pub unsafe extern "sysv64" fn calls_five(p: *mut u64) {
        five(p);
    }
}

/// A helper may call itself, or two may call each other, as long as a value known now ends it. What they
/// name is said as it is for any helper, once for each place, however often the kernel runs them. Here a
/// helper that calls itself names a const `cfg` leaves out, and a kept kernel that calls a pair `cfg`
/// leaves out is told of the helper it calls alone, though the pair names that const too.
#[kernite_macros::kernels]
pub mod recursive {
    #[cfg(feature = "big")]
    const STEP: u64 = 3;

    fn count<const N: u64>(p: gpr) {
        add(qword[p], STEP);
        if N > 0 {
            count::<{ N - 1 }>(p);
        }
    }

    fn ping<const N: u64>(p: gpr) {
        add(qword[p], 1);
        if N > 0 {
            pong::<{ N - 1 }>(p);
        }
    }

    fn pong<const N: u64>(p: gpr) {
        add(qword[p], 10);
        if N > 0 {
            ping::<{ N - 1 }>(p);
        }
    }

    #[cfg(feature = "big")]
    fn tick<const N: u64>(p: gpr) {
        add(qword[p], STEP);
        if N > 0 {
            tock::<{ N - 1 }>(p);
        }
    }

    #[cfg(feature = "big")]
    fn tock<const N: u64>(p: gpr) {
        add(qword[p], 1);
        if N > 0 {
            tick::<{ N - 1 }>(p);
        }
    }

    pub unsafe extern "sysv64" fn counts(p: *mut u64) {
        count::<3>(p);
    }

    pub unsafe extern "sysv64" fn calls_each_other(p: *mut u64) {
        ping::<3>(p);
    }

    pub unsafe extern "sysv64" fn calls_left_out_pair(p: *mut u64) {
        tick::<3>(p);
    }

    // A pair `cfg` keeps, that a kept kernel reaches only through a helper `cfg` leaves out, and a const
    // `cfg` leaves out with that helper: the kernel is told of the helper alone, whatever the pair names.
    #[cfg(feature = "big")]
    const HOP: u64 = 2;

    #[cfg(feature = "big")]
    fn entry(p: gpr) {
        back::<3>(p);
    }

    fn back<const N: u64>(p: gpr) {
        add(qword[p], HOP);
        if N > 0 {
            forth::<{ N - 1 }>(p);
        }
    }

    fn forth<const N: u64>(p: gpr) {
        add(qword[p], 1);
        if N > 0 {
            back::<{ N - 1 }>(p);
        }
    }

    pub unsafe extern "sysv64" fn calls_entry(p: *mut u64) {
        entry(p);
    }
}

#[cfg(all(test, feature = "big", feature = "wide"))]
mod tests {
    /// Each x86 kernel runs: it writes through its pointer what it names.
    #[test]
    fn the_kernels_run_as_they_are_written() {
        let x = &mut 9u64;
        // SAFETY: `x` is the u64 each kernel writes, and nothing else touches `COUNT`, which is 0.
        unsafe {
            super::rust_names::reads_const(x);
            assert_eq!(*x, 4);
            super::rust_names::reads_twice(x);
            assert_eq!(*x, 8);
            super::rust_names::reads_static(x);
            assert_eq!(*x, 0);
            super::rust_names::calls_kernel(x);
            assert_eq!(*x, 8);
            super::rust_names::listed::<2>(x);
            assert_eq!(*x, 2);
            super::rust_names::tile::<2>(x);
            assert_eq!(*x, 10);
            super::kernite_names::calls_helper(x);
            assert_eq!(*x, 7);
            super::kernite_names::calls_through(x);
            assert_eq!(*x, 5);
            *x = 3;
            assert_eq!(super::kernite_names::calls_instance(x), 5);
            super::kernite_names::calls_helper_of_instances::<1>(x);
            assert_eq!(*x, 7);
            super::in_helpers::calls_left_out(x);
            assert_eq!(*x, 13);
            *x = 0;
            super::in_helpers::calls_both_ways(x);
            assert_eq!(*x, 8);
            super::together::names_each(x);
            assert_eq!(*x, 12);
            *x = 0;
            super::together::names_each_when_wide(x);
            assert_eq!(*x, 12);
            super::narrower::reads_wide(x);
            assert_eq!(*x, 6);
            super::narrower::calls_six(x);
            assert_eq!(*x, 6);
            super::narrower::calls_five(x);
            assert_eq!(*x, 5);
            *x = 0;
            super::recursive::counts(x);
            assert_eq!(*x, 12);
            *x = 0;
            super::recursive::calls_each_other(x);
            assert_eq!(*x, 22);
            *x = 0;
            super::recursive::calls_left_out_pair(x);
            assert_eq!(*x, 8);
            *x = 0;
            super::recursive::calls_entry(x);
            assert_eq!(*x, 6);
        }
    }
}
