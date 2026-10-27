//! Names written raw, `r#type`, in the Rust the macro writes: a parameter's and a compile-time
//! argument's, in each kind of kernel that passes them on, a kernel's, a function's that a kernel
//! calls, a static's and a const's. Each case is a module of its own, and Rust builds each as
//! written, without a word.

/// A GPU kernel's compile-time argument, which picks its instance, and its parameter, which its
/// launcher passes on.
#[kgen_macros::kernels]
pub mod launched {
    #[instances(1, 2)]
    #[allow(non_upper_case_globals)]
    pub unsafe extern "ptx-kernel" fn k<const r#in: u32>(xs: *mut u32, r#type: u32) {
        let y: u32;
        add.u32(y, r#type, r#in);
        st.global.u32([xs], y);
    }
}

/// An x86 kernel's compile-time argument, which picks its instance, and its parameter, which
/// the instance is given.
#[kgen_macros::kernels]
pub mod instances {
    #[instances(1, 2)]
    #[allow(non_upper_case_globals)]
    pub unsafe extern "sysv64" fn tile<const r#in: u64>(r#type: u64) -> u64 {
        add(r#type, r#in);
        ret(r#type);
    }
}

/// A cooperative x86 kernel's compile-time argument, which picks its instance, and its pointer
/// and number, which its launcher hands each thread.
#[kgen_macros::kernels]
pub mod cooperative {
    #[instances(1, 2)]
    #[cooperative]
    #[allow(non_upper_case_globals)]
    pub unsafe extern "sysv64" fn together<const r#in: u64>(r#ref: *mut u64, r#type: u64) {
        grid.sync();
        add(r#type, r#in);
        mov(qword[r#ref], r#type);
    }
}

/// An x86 function, which a kernel calls.
#[kgen_macros::kernels]
pub mod calls {
    pub unsafe extern "sysv64" fn r#fn(x: u64) -> u64 {
        add(x, 1);
        ret(x);
    }

    pub unsafe extern "sysv64" fn caller(x: u64) -> u64 {
        let r: gpr;
        call(r, r#fn, x);
        ret(r);
    }
}

/// GPU kernels, one of them cooperative: the launcher of each, and the method that says how many
/// blocks a grid of the cooperative one has at most, `loop_blocks`.
#[kgen_macros::kernels]
pub mod gpu_kernels {
    pub unsafe extern "ptx-kernel" fn r#type(xs: *mut u32) {
        st.global.u32([xs], 1);
    }

    #[cooperative]
    pub unsafe extern "ptx-kernel" fn r#loop(xs: *mut u32) {
        st.global.u32([xs], 1);
    }
}

/// x86 kernels: a plain one, one of instances, which the plain one calls, and a cooperative one.
#[kgen_macros::kernels]
pub mod cpu_kernels {
    pub unsafe extern "sysv64" fn r#type(x: u64) -> u64 {
        let r: gpr;
        call(r, r#loop::<1>, x);
        ret(r);
    }

    #[instances(1)]
    pub unsafe extern "sysv64" fn r#loop<const N: u64>(x: u64) -> u64 {
        add(x, N);
        ret(x);
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn r#move(p: *mut u64) {
        grid.sync();
        mov(qword[p], 1);
    }
}

/// An x86 static, which a kernel names.
#[kgen_macros::kernels]
pub mod cpu_memory {
    #[allow(non_upper_case_globals)]
    pub static r#static: global<[u32; 4]>;

    pub unsafe extern "sysv64" fn k() {
        let at: gpr;
        lea(at, [r#static]);
    }
}

/// A GPU static, which a kernel names.
#[kgen_macros::kernels]
pub mod gpu_memory {
    pub static r#static: global<[u32; 4]>;

    pub unsafe extern "ptx-kernel" fn k() {
        let at: u64;
        mov.u64(at, r#static);
        red.global.add.u32([at], 1);
    }
}

/// A const of a module of consts, which a kernel of the module that takes it reads.
#[kgen_macros::consts]
pub mod sizes {
    #[allow(non_upper_case_globals)]
    pub const r#const: u64 = 3;
}

#[kgen_macros::kernels(consts = crate::sizes)]
pub mod reads {
    pub unsafe extern "sysv64" fn k(x: u64) -> u64 {
        add(x, r#const);
        ret(x);
    }
}
