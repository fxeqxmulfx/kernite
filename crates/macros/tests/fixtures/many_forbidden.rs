//! A crate of kernels that forbids Clippy's `too_many_arguments`: each launcher allows it, of
//! however many parameters, so Clippy refuses each module that has one, at its attribute.

#![forbid(clippy::too_many_arguments)]

#[kernite_macros::kernels]
pub mod cpu {
    #[cooperative]
    pub unsafe extern "sysv64" fn together(x: *mut u32) {
        grid.sync();
        mov(dword[x], 1);
    }
}

#[kernite_macros::kernels]
pub mod gpu {
    pub unsafe extern "ptx-kernel" fn one(x: *mut u32) {
        st.global.u32([x], 1);
    }
}
