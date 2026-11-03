//! Each thing `#[kernels]` and `#[consts]` refuse, which the macro says where it is written.

#[kgen_macros::kernels]
pub fn kernels_on_a_function() {}

#[kgen_macros::kernels(sizes)]
pub mod kernels_given_no_module_of_consts {}

#[kgen_macros::consts(sizes)]
pub mod consts_given_an_argument {}

#[kgen_macros::consts]
pub const CONSTS_ON_A_CONST: u64 = 1;

#[kgen_macros::consts]
pub mod a_const_that_is_not_pub {
    const STEP: u64 = 3;
}

#[kgen_macros::consts]
pub mod a_function_among_the_consts {
    pub fn step() {}
}

kgen_macros::__kernels! {}

kgen_macros::__kernels! { [] (sizes) pub fn kernels_called_on_a_function() {} }

#[kgen_macros::kernels]
pub mod an_abi_kgen_does_not_know {
    pub unsafe extern "C" fn f() {}
}

#[kgen_macros::kernels]
pub mod a_kernel_that_is_not_unsafe {
    pub extern "sysv64" fn f() {}
}

#[kgen_macros::consts]
pub mod sizes {
    pub const STEP: u64 = 3;
}

#[kgen_macros::kernels(consts = crate::sizes)]
pub mod a_const_given_and_its_own {
    pub const STEP: u64 = 4;
}

#[kgen_macros::kernels]
pub mod a_method_of_kernels_named_twice {
    #[cooperative]
    pub unsafe extern "ptx-kernel" fn k() {}

    pub unsafe extern "ptx-kernel" fn k_blocks() {}
}

#[kgen_macros::kernels]
pub mod kernels_no_module_of_ptx_has_both {
    pub unsafe extern "ptx-kernel" fn old(n: u32) {
        let i: u32;
        shfl.idx.b32(i, n, 0, 31);
    }

    pub unsafe extern "ptx-kernel" fn new(f: f32) {
        let v: f32;
        tanh.approx.f32(v, f);
    }
}

#[kgen_macros::kernels(consts = crate::sizes)]
pub mod two_functions_of_a_name_given_consts {
    fn h() {}
    fn h() {}
}
