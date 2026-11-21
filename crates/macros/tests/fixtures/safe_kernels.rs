//! A crate that forbids `unsafe_code` has x86 kernels that are no `unsafe fn`, of numbers, arrays and slices, and calls
//! them as it calls any function: the code the macro writes for one is no `unsafe` of the crate's. One needs a CPU
//! feature past x86-64's own, so Rust lets only a function that has it call it without `unsafe`.

#![forbid(unsafe_code)]

#[kernite_macros::kernels]
pub mod cpu {
    /// Twice `x`.
    pub extern "sysv64" fn twice(x: u64) -> u64 {
        add(x, x);
        ret(x);
    }

    /// The first of the four floats at `xs`.
    pub extern "sysv64" fn first(xs: &[f32; 4]) -> f32 {
        let v: xmm;
        vmovss(v, [xs]);
        ret(v);
    }

    /// Twice `x`, by `twice`, which it calls.
    pub extern "sysv64" fn four_times(x: u64) -> u64 {
        call(x, twice, x);
        call(x, twice, x);
        ret(x);
    }

    /// Sets the two integers at `xs` to `x`.
    pub extern "sysv64" fn fill(xs: &mut [u64; 2], x: u64) {
        mov(qword[xs], x);
        mov(qword[xs + 8], x);
    }

    /// `x` and `N` added, in an instance for each `N`.
    #[instances(1, 2)]
    pub extern "sysv64" fn tile<const N: u64>(x: u64) -> u64 {
        add(x, N);
        ret(x);
    }

    /// The second of the `N` integers at `xs`, in an instance for each `N`: the array's length is a compile-time
    /// argument, which the instance checks the address against.
    #[instances(2, 3)]
    pub extern "sysv64" fn second<const N: usize>(xs: &[u64; N]) -> u64 {
        let x: gpr;
        mov(x, [xs + 8]);
        ret(x);
    }

    /// The sum of the integers of `xs`, of any length: each is added where the index is below the length, as the
    /// compare says, which is all kernite needs to prove each address inside the slice.
    pub extern "sysv64" fn sum(xs: &[u64]) -> u64 {
        let (i, sum): (gpr, gpr);
        zero(i);
        zero(sum);
        label(top);
        cmp(i, xs.len);
        jae(done);
        add(sum, [xs + i * 8]);
        inc(i);
        jmp(top);
        label(done);
        ret(sum);
    }

    /// Sets each integer of `xs` to `x`.
    pub extern "sysv64" fn set(xs: &mut [u64], x: u64) {
        let i: gpr;
        zero(i);
        label(top);
        cmp(i, xs.len);
        jae(done);
        mov([xs + i * 8], x);
        inc(i);
        jmp(top);
        label(done);
    }
}

/// `twice`, `four_times` and `tile` are called with no `unsafe`, and so is `fill`, which writes an array.
pub fn of_values(x: u64, xs: &mut [u64; 2]) -> u64 {
    cpu::fill(xs, 7);
    cpu::twice(x) + cpu::four_times(x) + cpu::tile::<1>(x) + cpu::tile::<2>(x)
}

/// `second` is called with no `unsafe` at the length of the array it is given.
pub fn of_instances(xs: &[u64; 2], ys: &[u64; 3]) -> u64 {
    cpu::second::<2>(xs) + cpu::second::<3>(ys)
}

/// `set` and `sum` are called with a slice of any length, none too, and no `unsafe`.
pub fn of_slices(xs: &mut [u64], ys: &[u64]) -> u64 {
    cpu::set(xs, 7);
    cpu::sum(xs) + cpu::sum(ys)
}

/// `first` needs AVX, so a function that has it calls it with no `unsafe`.
#[target_feature(enable = "avx")]
pub fn with_avx(xs: &[f32; 4]) -> f32 {
    cpu::first(xs)
}
