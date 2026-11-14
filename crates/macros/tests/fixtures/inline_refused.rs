//! Each item kernite refuses `#[inline]` on, with `#[inline(never)]`, which wins over the `#[inline(always)]` of the
//! function the macros write to pick an instance; and each other way to write it, on an x86 kernel, where Rust would
//! say that the attribute is incompatible with `#[unsafe(naked)]`. kernite says the same of each, once, at its
//! `inline`, and Rust says nothing. Each case is a module of its own, for an attribute that is wrong stops the module.

#[kernite_macros::kernels]
pub mod x86_helper {
    pub unsafe extern "sysv64" fn k() {
        h();
    }

    #[inline(never)]
    fn h() {}
}

#[kernite_macros::kernels]
pub mod x86_const_helper {
    pub unsafe extern "sysv64" fn k() {
        h();
    }

    #[inline(never)]
    const fn h() {}
}

#[kernite_macros::kernels]
pub mod gpu_helper {
    pub unsafe extern "ptx-kernel" fn k() {
        h();
    }

    #[inline(never)]
    fn h() {}
}

#[kernite_macros::kernels]
pub mod ptx_func {
    #[inline(never)]
    extern "ptx-func" fn f(a: u32) -> u32 {
        add.u32(a, a, 1);
        ret(a);
    }

    pub unsafe extern "ptx-kernel" fn k() {}
}

#[kernite_macros::kernels]
pub mod x86_kernel {
    #[inline(never)]
    pub unsafe extern "sysv64" fn k() {}
}

#[kernite_macros::kernels]
pub mod x86_instances_before {
    #[inline(never)]
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn k<const N: u64>() {}
}

#[kernite_macros::kernels]
pub mod x86_instances_after {
    #[instances(1, 2)]
    #[inline(never)]
    pub unsafe extern "sysv64" fn k<const N: u64>() {}
}

#[kernite_macros::kernels]
pub mod x86_cooperative_before {
    #[inline(never)]
    #[cooperative]
    pub unsafe extern "sysv64" fn k(p: *mut u64) {
        grid.sync();
        mov(qword[p], 1);
    }
}

#[kernite_macros::kernels]
pub mod x86_cooperative_after {
    #[cooperative]
    #[inline(never)]
    pub unsafe extern "sysv64" fn k(p: *mut u64) {
        grid.sync();
        mov(qword[p], 1);
    }
}

#[kernite_macros::kernels]
pub mod gpu_kernel {
    #[inline(never)]
    pub unsafe extern "ptx-kernel" fn k() {}
}

#[kernite_macros::kernels]
pub mod gpu_instances_before {
    #[inline(never)]
    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn k<const N: u32>() {}
}

#[kernite_macros::kernels]
pub mod gpu_instances_after {
    #[instances(1, 2)]
    #[inline(never)]
    pub unsafe extern "ptx-kernel" fn k<const N: u32>() {}
}

#[kernite_macros::kernels]
pub mod gpu_cooperative_before {
    #[inline(never)]
    #[cooperative]
    pub unsafe extern "ptx-kernel" fn k(p: *mut u32) {
        grid.sync();
        st.global.u32([p], 1);
    }
}

#[kernite_macros::kernels]
pub mod gpu_cooperative_after {
    #[cooperative]
    #[inline(never)]
    pub unsafe extern "ptx-kernel" fn k(p: *mut u32) {
        grid.sync();
        st.global.u32([p], 1);
    }
}

#[kernite_macros::kernels]
pub mod x86_static_global {
    #[inline(never)]
    static S: global<[u32; 4]>;

    pub unsafe extern "sysv64" fn k() {}
}

#[kernite_macros::kernels]
pub mod x86_static_constant {
    #[inline(never)]
    pub static C: constant<[f32; 2]> = [0.5, 1.5];

    pub unsafe extern "sysv64" fn k() {}
}

#[kernite_macros::kernels]
pub mod gpu_static_global {
    #[inline(never)]
    static S: global<[u32; 4]>;

    pub unsafe extern "ptx-kernel" fn k() {}
}

#[kernite_macros::kernels]
pub mod gpu_static_constant {
    #[inline(never)]
    pub static C: constant<[f32; 2]> = [0.5, 1.5];

    pub unsafe extern "ptx-kernel" fn k() {}
}

#[kernite_macros::kernels]
pub mod form_plain {
    #[inline]
    pub unsafe extern "sysv64" fn k() {}
}

#[kernite_macros::kernels]
pub mod form_always {
    #[inline(always)]
    pub unsafe extern "sysv64" fn k() {}
}

#[kernite_macros::kernels]
pub mod form_value {
    #[inline = "never"]
    pub unsafe extern "sysv64" fn k() {}
}

#[kernite_macros::kernels]
pub mod form_raw {
    #[r#inline]
    pub unsafe extern "sysv64" fn k() {}
}

#[kernite_macros::kernels]
pub mod form_raw_never {
    #[r#inline(never)]
    pub unsafe extern "sysv64" fn k() {}
}

#[kernite_macros::kernels]
pub mod form_cfg_attr_true {
    #[cfg_attr(all(), inline)]
    pub unsafe extern "sysv64" fn k() {}
}

#[kernite_macros::kernels]
pub mod form_cfg_attr_false {
    #[cfg_attr(any(), inline(never))]
    pub unsafe extern "sysv64" fn k() {}
}

#[kernite_macros::kernels]
pub mod form_cfg_attr_target {
    #[cfg_attr(target_arch = "x86_64", inline(always))]
    pub unsafe extern "sysv64" fn k() {}
}

#[kernite_macros::kernels]
pub mod form_cfg_attr_nested {
    #[cfg_attr(all(), cfg_attr(any(), inline))]
    pub unsafe extern "sysv64" fn k() {}
}

#[kernite_macros::kernels]
pub mod form_cfg_attr_second {
    #[cfg_attr(all(), doc = "x", r#inline)]
    pub unsafe extern "sysv64" fn k() {}
}

#[kernite_macros::kernels]
pub mod form_raw_cfg_attr {
    #[r#cfg_attr(all(), inline)]
    pub unsafe extern "sysv64" fn k() {}
}

#[kernite_macros::kernels]
pub mod form_cfg_attr_raw_value {
    #[cfg_attr(all(), r#inline = "always")]
    pub unsafe extern "sysv64" fn k() {}
}

#[kernite_macros::kernels]
pub mod form_cfg_attr_deep {
    #[cfg_attr(any(), cfg_attr(any(), doc = "x", inline(always)))]
    pub unsafe extern "sysv64" fn k() {}
}
