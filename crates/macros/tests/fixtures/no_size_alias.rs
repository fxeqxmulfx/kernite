//! Kernels that take or give back a pointer to a type that kernite cannot tell has no size by what is written, for a
//! type alias names it: `Shown`, a trait object, and `Bytes`, a slice. Rust refuses the type and says that a size
//! cannot be known, once, at the type, in a kernel of every kind: plain, of compile-time arguments, cooperative, and
//! cooperative of compile-time arguments, on x86 and on a GPU, and in one that returns the pointer. It says nothing of
//! a plain function's pointer, which has two words. Two pointers to one type that a plain kernel takes have it said
//! once, at the first; two to two types, at each.

pub type Shown = dyn core::fmt::Debug + Send;
pub type Bytes = [u8];

pub fn plain(_x: *const Shown, _y: *mut Bytes) {}

#[kernite_macros::kernels]
pub mod cpu {
    pub unsafe extern "sysv64" fn read(x: *const super::Shown) {
        ret();
    }

    pub unsafe extern "sysv64" fn write(x: *mut super::Bytes) {
        ret();
    }

    pub unsafe extern "sysv64" fn different(x: *const u8, y: *const super::Shown, z: *mut super::Bytes) {
        ret();
    }

    pub unsafe extern "sysv64" fn same(x: *const super::Shown, y: *mut super::Shown) {
        ret();
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tiles<const N: u64>(x: *const super::Shown) {
        mov(qword[x], N);
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn together(x: *const super::Shown) {
        ret();
    }

    #[cooperative]
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn together_tiles<const N: u64>(x: *const super::Shown) {
        mov(qword[x], N);
    }

    pub unsafe extern "sysv64" fn result(x: u64) -> *const super::Shown {
        ret(x);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn result_tiles<const N: u64>(x: u64) -> *mut super::Bytes {
        ret(x);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn result_shown_tiles<const N: u64>(x: u64) -> *const super::Shown {
        ret(x);
    }
}

#[kernite_macros::kernels]
pub mod gpu {
    pub unsafe extern "ptx-kernel" fn read(x: *const super::Shown) {
        ret();
    }

    pub unsafe extern "ptx-kernel" fn write(x: *mut super::Bytes) {
        ret();
    }

    pub unsafe extern "ptx-kernel" fn different(x: *const u8, y: *const super::Shown, z: *mut super::Bytes) {
        ret();
    }

    pub unsafe extern "ptx-kernel" fn same(x: *const super::Shown, y: *mut super::Shown) {
        ret();
    }

    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn tiles<const N: u32>(x: *const super::Shown) {
        st.global.u32([x], N);
    }

    #[cooperative]
    pub unsafe extern "ptx-kernel" fn together(x: *const super::Shown) {
        ret();
    }

    #[cooperative]
    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn together_tiles<const N: u32>(x: *const super::Shown) {
        st.global.u32([x], N);
    }
}
