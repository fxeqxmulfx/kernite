//! Cooperative x86 kernels that only an x86-64 build keeps, in a crate that takes kernite-cpu on every machine: a
//! plain one, one whose code holds a constant, and one of compile-time arguments. Checked for another machine,
//! which has no `extern "sysv64"`, the crate says nothing; built for x86-64, it keeps the kernels, and they run on
//! a grid.
#![deny(warnings)]

#[kernite_macros::kernels]
pub mod cpu {
    #[cfg(target_arch = "x86_64")]
    #[cooperative]
    pub unsafe extern "sysv64" fn together(p: *mut u64) {
        mov(qword[p], 3);
    }

    #[cfg(target_arch = "x86_64")]
    #[cooperative]
    pub unsafe extern "sysv64" fn quarter(x: *mut f32) {
        let v: ymm;
        vbroadcastss(v, [0.25]);
        vmovups([x], v);
    }

    #[cfg(target_arch = "x86_64")]
    #[cooperative]
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tiles<const N: u64>(p: *mut u64) {
        mov(qword[p], N);
    }
}

#[cfg(test)]
mod tests {
    /// Each kernel runs on a grid, and writes through its pointer what it holds.
    #[test]
    fn the_kernels_run_where_cfg_keeps_them() {
        let (x, p) = (&mut [0.0f32; 8], &mut 0u64);
        // SAFETY: `x` is the eight floats a kernel writes, and `p` the u64 the others do.
        unsafe {
            super::cpu::together(kernite_cpu::Grid::new(1), p).unwrap();
            assert_eq!(*p, 3);
            super::cpu::quarter(kernite_cpu::Grid::new(1), x.as_mut_ptr()).unwrap();
            assert_eq!(*x, [0.25; 8]);
            super::cpu::tiles::<2>(kernite_cpu::Grid::new(1), p).unwrap();
            assert_eq!(*p, 2);
        }
    }
}
