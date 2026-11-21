//! A use of a deprecated x86 kernel of instances by a kernel is told of by Rust at each place a kernel writes an
//! instance of it, as Rust tells of a use of a deprecated generic function of its own, `old::<2>`: at the name, with
//! the note of the attribute, and not at the module's attribute as well. A place is told of once, whatever kernels have
//! it: a helper that two kernels call, or a kernel of two instances. A use inside the deprecated kernel itself is not
//! told of, as in Rust. `deny` counts the uses told of, where Cargo says a line that two errors share once.
#![deny(deprecated)]

#[kernite_macros::kernels]
pub mod cpu {
    /// Called by kernels, an instance at each place.
    #[deprecated(note = "use `fresh`")]
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn old<const N: u64>(p: *mut u64) {
        mov(qword[p], N);
    }

    /// Called by no kernel.
    #[deprecated]
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn unused<const N: u64>(p: *mut u64) {
        mov(qword[p], N);
    }

    /// Calls `old::<2>`.
    pub unsafe extern "sysv64" fn once(p: *mut u64) {
        call(old::<2>, p);
    }

    /// Calls `old::<1>` and then `old::<2>`, two instances of one function.
    pub unsafe extern "sysv64" fn both(p: *mut u64) {
        call(old::<1>, p);
        call(old::<2>, p);
    }

    /// Calls `old::<2>` twice, at two places.
    pub unsafe extern "sysv64" fn again(p: *mut u64) {
        call(old::<2>, p);
        call(old::<2>, p);
    }

    /// Calls the instance its own argument names, at one place, in each of its two instances.
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn by<const M: u64>(p: *mut u64) {
        call(old::<M>, p);
    }

    /// Calls `old::<1>`, at one place for each kernel that calls it.
    fn touch(p: gpr) {
        call(old::<1>, p);
    }

    /// Calls `touch`.
    pub unsafe extern "sysv64" fn first(p: *mut u64) {
        touch(p);
    }

    /// Calls `touch` as well.
    pub unsafe extern "sysv64" fn second(p: *mut u64) {
        touch(p);
    }

    /// Of one instance of three arguments: a number, a `bool` and a negative number.
    #[deprecated]
    #[instances((4, false, -2))]
    pub unsafe extern "sysv64" fn shapes<const W: usize, const B: bool, const K: i8>(p: *mut u8) {
        mov(byte[p], W);
        mov(byte[p + 1], K);
        if B {
            mov(byte[p + 2], 1);
        }
    }

    /// Calls `shapes::<4, false, -2>`.
    pub unsafe extern "sysv64" fn shaped(p: *mut u8) {
        call(shapes::<4, false, -2>, p);
    }

    /// Calls its own instance for 1, in the instance for 2: Rust does not tell of a use inside the item deprecated.
    #[deprecated]
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn rec<const N: u64>(p: *mut u64) {
        if N == 2 {
            call(rec::<1>, p);
        }
        mov(qword[p], N);
    }
}
