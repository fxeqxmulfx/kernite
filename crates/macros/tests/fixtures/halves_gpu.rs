//! A GPU kernel that takes an `f16` and a `bf16`, by value and by pointer, which Rust has not and
//! the launcher names as kernite-cuda does. Rust builds the crate without a word, and the test runs
//! the kernel on the GPU. It skips where there is none.

#[kernite_macros::kernels]
pub mod gpu {
    /// Writes `half` at `out` and `brain` at `other`, as they are.
    pub unsafe extern "ptx-kernel" fn store(out: *mut f16, half: f16, brain: bf16, other: *mut bf16) {
        st.global.b16([out], half);
        st.global.b16([other], brain);
    }
}

#[cfg(test)]
mod tests {
    use kernite_cuda::{Cuda, Grid, bf16, f16};

    use super::gpu;

    #[test]
    fn a_launcher_takes_the_halves_as_kernite_cuda_has_them() {
        let Ok(cuda) = Cuda::new().map_err(|e| eprintln!("no GPU: skipped. {e}")) else { return };
        let kernels = gpu::Kernels::load(&cuda).unwrap();
        let (out, other) = (cuda.alloc::<f16>(1).unwrap(), cuda.alloc::<bf16>(1).unwrap());
        let (half, brain) = (f16::from_f32(1.5), bf16::from_f32(-2.5));
        // SAFETY: the kernel writes one half to each buffer, which holds one, and the launch has ended
        // when they are read.
        unsafe { kernels.store(Grid::new(1, 1), out.ptr(), half, brain, other.ptr()).unwrap() };
        cuda.sync().unwrap();
        assert_eq!(out.read().unwrap()[0].to_bits(), half.to_bits());
        assert_eq!(other.read().unwrap()[0].to_bits(), brain.to_bits());
    }
}
