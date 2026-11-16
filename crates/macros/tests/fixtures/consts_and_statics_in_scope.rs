//! Modules of kernels with a const or a static in scope named as a local that the Rust the macros write bound for
//! itself before: consts named `__0`, `grid`, `threads` and `shared`, which the modules import, and `cuda`, the GPU
//! module's own, and a static named `grid`, the x86 module's own. Rust reads a binding named as a const in scope as
//! that const, and refuses one named as a static, and said so at the module's attribute, where nothing can be mended.
//! The types of the kernels and of a static name the consts, as the length of an array. Each module is built as
//! written, a kernel of each kind, without a word.

#![allow(non_upper_case_globals)]

pub const __0: usize = 4;
pub const grid: usize = 1;
pub const threads: usize = 2;
pub const shared: usize = 3;

/// x86 kernels: `plain` takes a pointer, which the macro asks Rust of; `tiles` is of instances; and `together` is
/// cooperative, so its launcher binds the grid, beside the module's static `grid`.
#[kernite_macros::kernels]
pub mod cpu {
    use super::{__0, shared, threads};

    pub static grid: global<[u64; 2]>;

    pub unsafe extern "sysv64" fn plain(x: *mut [u64; __0]) {
        mov(qword[x], 7);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tiles<const N: u64>(x: *mut [u64; threads]) {
        mov(qword[x], N);
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn together(x: *mut [u64; threads], y: *mut [u64; shared]) {
        mov(qword[x], 1);
        mov(qword[y], 1);
    }
}

/// GPU kernels: `plain` takes a pointer; `tiles` is of instances; and `together` is cooperative, so its method binds
/// the grid, and another says how many blocks its grid has at most. `Kernels::load` binds the `Cuda`, and `MEMORY`
/// gives the memory of a static whose type names `cuda`.
#[kernite_macros::kernels]
pub mod gpu {
    use super::{__0, grid, shared, threads};

    const cuda: usize = 4;

    pub static MEMORY: global<[u32; cuda]>;

    pub unsafe extern "ptx-kernel" fn plain(x: *mut [u32; __0]) {
        st.global.u32([x], 7);
    }

    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn tiles<const N: u32>(x: *mut [u32; grid]) {
        st.global.u32([x], N);
    }

    #[cooperative]
    pub unsafe extern "ptx-kernel" fn together(x: *mut [u32; threads], y: *mut [u32; shared]) {
        st.global.u32([x], 1);
        st.global.u32([y], 1);
    }
}
