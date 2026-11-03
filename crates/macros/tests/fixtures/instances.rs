//! Kernels called for an instance their `#[instances(..)]` does not list, which `cargo build`
//! refuses where Rust compiles the call.

#[kernite_macros::kernels]
pub mod cpu {
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn k<const N: u64>(x: u64) -> u64 {
        add(x, N);
        ret(x);
    }
}

#[kernite_macros::kernels]
pub mod gpu {
    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn k<const N: u32>(x: *mut u32) {
        st.global.u32([x], N);
    }
}

/// `cpu::k` for an instance it is not compiled for.
pub fn on_the_cpu(x: u64) -> u64 {
    unsafe { cpu::k::<3>(x) }
}

/// `gpu::k` for an instance it is not compiled for.
pub fn on_the_gpu(kernels: &gpu::Kernels, x: kernite_cuda::Ptr<u32>) -> Result<(), kernite_cuda::Error> {
    unsafe { kernels.k::<3>(kernite_cuda::Grid::new(1, 1), x) }
}
