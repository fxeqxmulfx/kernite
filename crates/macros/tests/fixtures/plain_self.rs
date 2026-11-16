//! A parameter named `self` or `Self`, written plain, in each kind of function: Rust says what it says of a function
//! that is no method, at the name, and nothing more; of a kernel that `cfg` leaves out, nothing.

#[kernite_macros::kernels]
pub mod gpu_self {
    pub unsafe extern "ptx-kernel" fn f(self: *mut u64) {
        st.global.u64([self], 1);
    }
}

#[kernite_macros::kernels]
pub mod gpu_upper_self {
    pub unsafe extern "ptx-kernel" fn f(Self: *mut u64) {
        st.global.u64([Self], 1);
    }
}

#[kernite_macros::kernels]
pub mod cpu_instances {
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn f<const N: u64>(self: u64) -> u64 {
        add(self, N);
        ret(self);
    }
}

#[kernite_macros::kernels]
pub mod cpu_cooperative {
    #[cooperative]
    pub unsafe extern "sysv64" fn f(self: *mut u64) {
        mov(qword[self], 1);
    }
}

#[kernite_macros::kernels]
pub mod cpu_plain {
    pub unsafe extern "sysv64" fn f(self: u64) -> u64 {
        ret(self);
    }

    pub unsafe extern "sysv64" fn g(Self: u64) -> u64 {
        ret(Self);
    }
}

#[kernite_macros::kernels]
pub mod cpu_not_first {
    pub unsafe extern "sysv64" fn f(a: u64, self: u64) -> u64 {
        add(a, self);
        ret(a);
    }
}

#[kernite_macros::kernels]
pub mod cpu_left_out {
    #[cfg(any())]
    pub unsafe extern "sysv64" fn f(self: u64) -> u64 {
        ret(self);
    }
}

#[kernite_macros::kernels]
pub mod cpu_generic {
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn f<const N: usize>(self: *mut [u64; N]) {
        mov(qword[self], 1);
    }
}

#[kernite_macros::kernels]
pub mod cpu_refused {
    pub unsafe extern "sysv64" fn f(self: f128) {
        ret();
    }

    pub unsafe extern "sysv64" fn g(self: [u64]) {
        ret();
    }
}

#[kernite_macros::kernels]
pub mod gpu_refused {
    pub unsafe extern "ptx-kernel" fn f(self: f128) {
        ret();
    }
}

#[kernite_macros::kernels]
pub mod cpu_helper {
    pub unsafe extern "sysv64" fn k(x: u64) -> u64 {
        h(x);
        g(x);
        ret(x);
    }

    fn h(self: gpr) {
        add(self, 1);
    }

    fn g(Self: gpr) {
        add(Self, 1);
    }
}

#[kernite_macros::kernels]
pub mod gpu_helper {
    pub unsafe extern "ptx-kernel" fn k(p: *mut u32) {
        let s: shared<[u32; 4], 16>;
        h(s);
        let v: u32;
        call(v, g, 1);
        st.global.u32([p], v);
    }

    fn h(self: shared<[u32; 4], 16>) {
        st.shared.u32([self], 1);
    }

    extern "ptx-func" fn g(self: u32) -> u32 {
        ret(self);
    }
}

#[kernite_macros::kernels]
pub mod cpu_beside {
    pub unsafe extern "sysv64" fn f(self: u64, x: u64) -> u64 {
        ret(x);
    }
}

#[kernite_macros::kernels]
pub mod cpu_mut {
    pub unsafe extern "sysv64" fn f(mut self: u64) -> u64 {
        ret(self);
    }

    pub unsafe extern "sysv64" fn g(mut Self: u64) -> u64 {
        ret(Self);
    }
}
