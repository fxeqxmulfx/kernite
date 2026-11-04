//! The kernels that say an error in a type once already, each with a type that is not there: a
//! plain x86 kernel, in a parameter and in a result, and a GPU kernel of instances.

#[kernite_macros::kernels]
pub mod cpu {
    pub unsafe extern "sysv64" fn parameter(p: *mut Gone) {
        mov(byte[p], 1);
    }

    pub unsafe extern "sysv64" fn result(p: *mut u8) -> *mut Away {
        mov(byte[p], 1);
        ret(p);
    }
}

#[kernite_macros::kernels]
pub mod gpu {
    #[instances(2, 3)]
    pub unsafe extern "ptx-kernel" fn parameter<const N: u32>(p: *mut Elsewhere) {
        st.global.u32([p], N);
    }
}
