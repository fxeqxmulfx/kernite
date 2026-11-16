//! A static of global memory that a module has no room for, as `ptxas` writes the size of a module's global memory in
//! 32 bits, is outside the module: the module declares a `.u64` of constant memory for its address, and `Kernels::load`
//! has the driver allocate the memory, zeros. `big` has one of 6 GB, which its kernels write and read at the first byte
//! and the last. `left_out` has two of 3 GB, the first of which `cfg` leaves out: which static is outside is the
//! source's, which `cfg` does not change, so the second is outside beside it all the same. The crate builds without a
//! word under `deny(warnings)`, and its tests assert how the modules declare their statics and, on a machine with a
//! GPU, load `big`: where the GPU has the memory, its kernels read what they wrote, and where it has not, the load is
//! an error that names the static and its bytes.

#![deny(warnings)]

/// 6 GB of global memory, and the kernels that write and read its first byte and its last.
#[kernite_macros::kernels]
pub mod big {
    /// More than a module has room for.
    pub static BIG: global<[u8; 6442450944]>;

    /// Writes 1 to the first byte of `BIG` and 2 to its last.
    pub unsafe extern "ptx-kernel" fn write() {
        let v: u16;
        mov.u16(v, 1);
        st.global.u8([BIG], v);
        mov.u16(v, 2);
        st.global.u8([BIG + 6442450943], v);
    }

    /// Writes the first byte of `BIG` to `out`, and its last to `out + 1`.
    pub unsafe extern "ptx-kernel" fn read(out: *mut u8) {
        let v: u16;
        ld.global.u8(v, [BIG]);
        st.global.u8([out], v);
        ld.global.u8(v, [BIG + 6442450943]);
        st.global.u8([out + 1], v);
    }
}

/// `A` is left out by `cfg`, and `B` has no room beside it all the same.
#[kernite_macros::kernels]
pub mod left_out {
    /// Left out: 3 GB, which `B` makes 6 GB with.
    #[cfg(any())]
    pub static A: global<[u8; 3221225472]>;

    /// Kept: it is outside the module, as it is where `A` is kept.
    pub static B: global<[u8; 3221225472]>;

    /// Writes 1 to the first byte of `B`.
    pub unsafe extern "ptx-kernel" fn touch() {
        let v: u16;
        mov.u16(v, 1);
        st.global.u8([B], v);
    }
}

#[cfg(test)]
mod tests {
    use kernite_cuda::{Cuda, Grid};

    use super::{big, left_out};

    /// The module declares `BIG` as the `.u64` of constant memory that holds its address, not as global memory, and
    /// each kernel that names it loads that address as it starts.
    #[test]
    fn the_module_declares_the_static_it_has_no_room_for_as_the_slot_of_its_address() {
        let ptx = big::PTX;
        assert!(ptx.contains(".const .align 8 .b8 _ZN7project7outside3big3BIGE[8];"), "{ptx}");
        assert!(!ptx.contains(".global .align"), "{ptx}");
        assert_eq!(ptx.matches("ld.const.u64 %BIG, [_ZN7project7outside3big3BIGE];").count(), 2, "{ptx}");
    }

    /// `B` is the slot of an address, though the static before it is left out.
    #[test]
    fn a_static_beside_one_cfg_leaves_out_is_outside_the_module_all_the_same() {
        let ptx = left_out::PTX;
        assert!(ptx.contains(".const .align 8 .b8 _ZN7project7outside8left_out1BE[8];"), "{ptx}");
        assert!(!ptx.contains(".global .align"), "{ptx}");
    }

    /// Where the GPU has 6 GB, the module loads and the kernels write and read the first byte and the last of `BIG`;
    /// where it has not, the load is an error that names the static and its bytes, and the run is skipped.
    #[test]
    fn the_module_loads_where_the_gpu_has_the_memory_of_the_static_outside_it() {
        let Ok(cuda) = Cuda::new().inspect_err(|e| eprintln!("no GPU: skipped. {e}")) else { return };
        let kernels = match big::Kernels::load(&cuda) {
            Ok(kernels) => kernels,
            Err(error) => {
                let error = error.to_string();
                assert!(error.contains("out of memory"), "{error}");
                assert!(error.contains("3BIGE`, 6442450944 bytes"), "{error}");
                eprintln!("a GPU with more than 6 GB: skipped, this one has less, so the load is refused");
                return;
            }
        };
        let out = cuda.alloc::<u8>(2).unwrap();
        // SAFETY: `write` writes the module's own memory, and `read` the two bytes of `out`.
        unsafe {
            kernels.write(Grid::new(1, 1)).unwrap();
            kernels.read(Grid::new(1, 1), out.ptr()).unwrap();
        }
        assert_eq!(out.read().unwrap(), [1, 2]);
    }
}
