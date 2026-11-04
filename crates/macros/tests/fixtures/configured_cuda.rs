//! Modules of GPU kernels whose GPU items are all under the feature `cuda`, which brings
//! `kernite-cuda`: built without it, the crate has no `kernite-cuda`, and builds without a word,
//! under `deny(warnings)`, as it does a crate that never had the modules; built with it, the
//! modules are there. Each module has one GPU item, a kernel, a function that kernels call or a
//! static, under `cfg` and under `cfg_attr(.., cfg(..))`.

#![deny(warnings)]

/// A kernel, under `cfg`.
#[kernite_macros::kernels]
pub mod kernel_by_cfg {
    #[cfg(feature = "cuda")]
    pub unsafe extern "ptx-kernel" fn one(p: *mut u32) {
        st.global.u32([p], 1);
    }
}

/// A kernel, under a `cfg_attr(..)`.
#[kernite_macros::kernels]
pub mod kernel_by_attr {
    #[cfg_attr(not(feature = "cuda"), cfg(any()))]
    pub unsafe extern "ptx-kernel" fn one(p: *mut u32) {
        st.global.u32([p], 1);
    }
}

/// A function that kernels call, under `cfg`.
#[kernite_macros::kernels]
pub mod func_by_cfg {
    #[cfg(feature = "cuda")]
    extern "ptx-func" fn one() -> u32 {
        let v: u32;
        mov.u32(v, 1);
        ret(v);
    }
}

/// A function that kernels call, under a `cfg_attr(..)`.
#[kernite_macros::kernels]
pub mod func_by_attr {
    #[cfg_attr(not(feature = "cuda"), cfg(any()))]
    extern "ptx-func" fn one() -> u32 {
        let v: u32;
        mov.u32(v, 1);
        ret(v);
    }
}

/// A static, under `cfg`.
#[kernite_macros::kernels]
pub mod static_by_cfg {
    #[cfg(feature = "cuda")]
    pub static COUNTS: global<[u32; 4]>;
}

/// A static, under a `cfg_attr(..)`.
#[kernite_macros::kernels]
pub mod static_by_attr {
    #[cfg_attr(not(feature = "cuda"), cfg(any()))]
    pub static COUNTS: global<[u32; 4]>;
}

/// With the feature, each module's kernels load on a GPU, and each module is PTX.
#[cfg(feature = "cuda")]
pub fn load(cuda: &kernite_cuda::Cuda) -> Result<usize, kernite_cuda::Error> {
    kernel_by_cfg::Kernels::load(cuda)?;
    kernel_by_attr::Kernels::load(cuda)?;
    func_by_cfg::Kernels::load(cuda)?;
    func_by_attr::Kernels::load(cuda)?;
    static_by_cfg::Kernels::load(cuda)?;
    static_by_attr::Kernels::load(cuda)?;
    Ok(kernel_by_cfg::PTX.len())
}
