//! A GPU kernel that takes an `f16` and a `bf16`, by value and by pointer, which Rust has not and
//! the launcher names as kernite-cuda does, and one that takes them written raw, `r#f16` and `r#bf16`,
//! which Rust reads as the same names. Rust builds the crate without a word, and the tests run each
//! kernel on the GPU. They skip where there is none.

#[kernite_macros::kernels]
pub mod gpu {
    /// Writes `half` at `out` and `brain` at `other`, as they are.
    pub unsafe extern "ptx-kernel" fn store(out: *mut f16, half: f16, brain: bf16, other: *mut bf16) {
        st.global.b16([out], half);
        st.global.b16([other], brain);
    }

    /// The same, with the types written raw.
    pub unsafe extern "ptx-kernel" fn store_raw(out: *mut r#f16, half: r#f16, brain: r#bf16, other: *mut r#bf16) {
        st.global.b16([out], half);
        st.global.b16([other], brain);
    }
}

#[cfg(test)]
mod tests {
    use kernite_cuda::{Cuda, Grid, bf16, f16};

    use super::gpu;

    /// Launches `store`, or `store_raw` if `raw`, and gives back the bits of what it wrote and of what it was
    /// given: the half written, the half given, the brain written, the brain given.
    fn stored(raw: bool) -> Result<Option<[u16; 4]>, Box<dyn std::error::Error>> {
        let Ok(cuda) = Cuda::new().map_err(|e| eprintln!("no GPU: skipped. {e}")) else { return Ok(None) };
        let kernels = gpu::Kernels::load(&cuda)?;
        let (out, other) = (cuda.alloc::<f16>(1)?, cuda.alloc::<bf16>(1)?);
        let (half, brain) = (f16::from_f32(1.5), bf16::from_f32(-2.5));
        let grid = Grid::new(1, 1);
        // SAFETY: the kernel writes one half to each buffer, which holds one, and the launch has ended
        // when they are read.
        unsafe {
            if raw {
                kernels.store_raw(grid, out.ptr(), half, brain, other.ptr())?;
            } else {
                kernels.store(grid, out.ptr(), half, brain, other.ptr())?;
            }
        }
        cuda.sync()?;
        Ok(Some([out.read()?[0].to_bits(), half.to_bits(), other.read()?[0].to_bits(), brain.to_bits()]))
    }

    #[test]
    fn a_launcher_takes_the_halves_as_kernite_cuda_has_them() -> Result<(), Box<dyn std::error::Error>> {
        if let Some([out, half, other, brain]) = stored(false)? {
            assert_eq!((out, other), (half, brain));
        }
        Ok(())
    }

    #[test]
    fn a_launcher_takes_the_halves_written_raw_as_kernite_cuda_has_them() -> Result<(), Box<dyn std::error::Error>> {
        if let Some([out, half, other, brain]) = stored(true)? {
            assert_eq!((out, other), (half, brain));
        }
        Ok(())
    }
}
