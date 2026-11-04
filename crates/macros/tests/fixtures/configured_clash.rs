//! Kernels and functions that no one PTX module can hold are said where `cfg` keeps them, and of those
//! alone: `old` is for GPUs older than `sm_70` from PTX 6.4 on, `mid` and the function `fast` need `sm_75`
//! and PTX 7.0, and `top` needs `sm_80`, which came with PTX 7.0. The crate has a feature for each, and
//! builds with those that no module has together, once they are named. The one kernel that calls `fast`,
//! `shine`, is left out by every `cfg`. The error is at the module's first GPU item, `first`, which needs
//! nothing and is left out unless its feature keeps it.

#![deny(warnings)]

#[kernite_macros::kernels]
pub mod trio {
    #[cfg(feature = "first")]
    pub unsafe extern "ptx-kernel" fn first(out: *mut u32) {
        st.global.u32([out], 1);
    }

    #[cfg(feature = "old")]
    pub unsafe extern "ptx-kernel" fn old(n: u32) {
        let i: u32;
        shfl.idx.b32(i, n, 0, 31);
    }

    #[cfg(feature = "mid")]
    pub unsafe extern "ptx-kernel" fn mid(f: f32) {
        let v: f32;
        tanh.approx.f32(v, f);
    }

    #[cfg(feature = "top")]
    pub unsafe extern "ptx-kernel" fn top(n: u32) {
        let i: u32;
        redux.sync.add.u32(i, n, n);
    }

    #[cfg(feature = "fast")]
    extern "ptx-func" fn fast(f: f32) -> f32 {
        let v: f32;
        tanh.approx.f32(v, f);
        ret(v);
    }

    #[cfg(any())]
    pub unsafe extern "ptx-kernel" fn shine(x: *mut f32, f: f32) {
        let v: f32;
        call(v, fast, f);
        st.global.f32([x], v);
    }
}
