//! An integer where an integer of another type is wanted is told by its type, "this is a u16, not u8": "an integer"
//! would say no fault, for a `u8` is one too. So it is for a const, for an element of an array const, told at the
//! array, for a compile-time argument of `#[instances(..)]`, on x86 and on a GPU, and for the value of a static, which
//! says its type in its own words. Where the type wanted is no integer's, `bool`, what is wrong is that the value is
//! an integer, and kernite says "an integer". Rust's own error stands beside kernite's, at the const.

#[kernite_macros::kernels]
pub mod consts {
    pub const A: u8 = 1u16;
    pub const B: i64 = 2u32;
    pub const C: [u8; 2] = [1u8, 2u16];
    pub const D: bool = 1u16;

    pub unsafe extern "sysv64" fn a(x: u64) -> u64 {
        add(x, A);
        ret(x);
    }

    pub unsafe extern "sysv64" fn b(x: u64) -> u64 {
        add(x, B);
        ret(x);
    }

    pub unsafe extern "sysv64" fn c(x: u64) -> u64 {
        add(x, C[1]);
        ret(x);
    }

    pub unsafe extern "sysv64" fn d(x: u64) -> u64 {
        add(x, D);
        ret(x);
    }
}

#[kernite_macros::kernels]
pub mod instances_x86 {
    #[instances(1u16)]
    pub unsafe extern "sysv64" fn a<const N: u8>(x: u64) -> u64 {
        add(x, N);
        ret(x);
    }
}

#[kernite_macros::kernels]
pub mod instances_gpu {
    #[instances(1, 2i32)]
    pub unsafe extern "ptx-kernel" fn a<const N: u8>(p: *mut u32) {
        let v: u32;
        mov.u32(v, N);
        st.global.u32([p], v);
    }
}

#[kernite_macros::kernels]
pub mod statics_x86 {
    pub static A: global<u32> = 1u16;

    pub unsafe extern "sysv64" fn a(x: u64) -> u64 {
        ret(x);
    }
}

#[kernite_macros::kernels]
pub mod statics_gpu {
    pub static A: global<u32> = 1u16;

    pub unsafe extern "ptx-kernel" fn a(p: *mut u32) {
        st.global.u32([p], 1);
    }
}
