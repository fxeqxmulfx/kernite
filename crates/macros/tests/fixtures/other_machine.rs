//! x86 kernels that only an x86-64 build keeps, of each kind that kernite writes more for than the kernel's own
//! function: one whose code holds a constant, one of compile-time arguments, and one of those whose instances
//! hold a constant. Checked for another machine, which has no `extern "sysv64"`, the crate says nothing; built for
//! x86-64, it keeps the kernels, and they run.
#![deny(warnings)]

#[kernite_macros::kernels]
pub mod cpu {
    #[cfg(target_arch = "x86_64")]
    pub unsafe extern "sysv64" fn quarter(x: *mut f32) {
        let v: ymm;
        vbroadcastss(v, [0.25]);
        vmovups([x], v);
    }

    #[cfg(target_arch = "x86_64")]
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tiles<const N: u64>(p: *mut u64) {
        mov(qword[p], N);
    }

    #[cfg(target_arch = "x86_64")]
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn halves<const N: u64>(x: *mut f32) {
        let v: ymm;
        vbroadcastss(v, [0.5]);
        vmovups([x], v);
    }
}

#[cfg(test)]
mod tests {
    /// Each kernel runs, and writes through its pointer what it holds.
    #[test]
    fn the_kernels_run_where_cfg_keeps_them() {
        let (x, p) = (&mut [0.0f32; 8], &mut 0u64);
        // SAFETY: `x` is the eight floats a kernel writes, and `p` the u64 the other does.
        unsafe {
            super::cpu::quarter(x.as_mut_ptr());
            assert_eq!(*x, [0.25; 8]);
            super::cpu::halves::<2>(x.as_mut_ptr());
            assert_eq!(*x, [0.5; 8]);
            super::cpu::tiles::<2>(p);
            assert_eq!(*p, 2);
        }
    }
}
