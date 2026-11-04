//! An x86 kernel of instances that kernite refuses, for it reads a register it never wrote, and that only an
//! x86-64 build keeps. Checked for another machine, the crate says nothing; built for x86-64, it is kernite's errors.
#![deny(warnings)]

#[kernite_macros::kernels]
pub mod cpu {
    #[cfg(target_arch = "x86_64")]
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tiles<const N: u64>(p: *mut u64) {
        let v: gpr;
        add(v, N);
        mov(qword[p], v);
    }
}
