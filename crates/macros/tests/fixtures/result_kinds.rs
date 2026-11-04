//! The result of a kernel that can have none. The threads of a cooperative x86 kernel return nothing, and so
//! does a GPU kernel, and kernite says so, at the result's type. Rust sees that type as it sees a plain x86
//! kernel's, which is beside them for the words to be the same: it says what it says of any function's result,
//! once, at the same place, and counts an import that only the result names as used. The crate's functions
//! call every kernel as Rust calls it, and Rust says nothing there. A GPU kernel's launcher is a method, so a
//! `&` in its result takes the lifetime of `self`, and Rust has none to ask for.

pub struct Ret(pub u64);

pub struct NeedsCopy<T: Copy>(pub T);

#[kernite_macros::kernels]
pub mod plain {
    use super::{NeedsCopy, Ret};

    pub unsafe extern "sysv64" fn size(p: *const u8) -> *const core::ffi::CStr {
        ret(p);
    }

    pub unsafe extern "sysv64" fn bound(p: *const u8) -> *const NeedsCopy<String> {
        ret(p);
    }

    pub unsafe extern "sysv64" fn reference(p: *const u8) -> &u64 {
        ret(p);
    }

    pub unsafe extern "sysv64" fn lifetime(p: *const u8) -> &'a u64 {
        ret(p);
    }

    pub unsafe extern "sysv64" fn blank(p: *const u8) -> *const _ {
        ret(p);
    }

    pub unsafe extern "sysv64" fn module(p: *const u8) -> core::f16 {
        ret(p);
    }

    pub unsafe extern "sysv64" fn wide(p: *const u8) -> *const f128 {
        ret(p);
    }

    pub unsafe extern "sysv64" fn import(p: *const u8) -> Ret {
        ret(p);
    }
}

#[kernite_macros::kernels]
pub mod cpu {
    use super::{NeedsCopy, Ret};

    #[cooperative]
    pub unsafe extern "sysv64" fn size(p: *const u8) -> *const core::ffi::CStr {
        ret(p);
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn bound(p: *const u8) -> *const NeedsCopy<String> {
        ret(p);
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn reference(p: *const u8) -> &u64 {
        ret(p);
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn lifetime(p: *const u8) -> &'a u64 {
        ret(p);
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn blank(p: *const u8) -> *const _ {
        ret(p);
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn module(p: *const u8) -> core::f16 {
        ret(p);
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn wide(p: *const u8) -> *const f128 {
        ret(p);
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn import(p: *const u8) -> Ret {
        ret(p);
    }
}

#[kernite_macros::kernels]
pub mod cpu_tiles {
    use super::Ret;

    #[cooperative]
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn size<const N: u64>(p: *const u8) -> *const core::ffi::CStr {
        add(p, N);
        ret(p);
    }

    #[cooperative]
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn array<const N: usize>(p: *const u8) -> *const [u8; N] {
        ret(p);
    }

    #[cooperative]
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn import<const N: u64>(p: *const u8) -> Ret {
        add(p, N);
        ret(p);
    }
}

#[kernite_macros::kernels]
pub mod gpu {
    use super::{NeedsCopy, Ret};

    pub unsafe extern "ptx-kernel" fn size(p: *mut u32, n: u32) -> *mut core::ffi::CStr {
        st.global.u32([p], n);
    }

    pub unsafe extern "ptx-kernel" fn bound(p: *mut u32, n: u32) -> *mut NeedsCopy<String> {
        st.global.u32([p], n);
    }

    pub unsafe extern "ptx-kernel" fn reference(p: *mut u32, n: u32) -> &u64 {
        st.global.u32([p], n);
    }

    pub unsafe extern "ptx-kernel" fn lifetime(p: *mut u32, n: u32) -> &'a u64 {
        st.global.u32([p], n);
    }

    pub unsafe extern "ptx-kernel" fn blank(p: *mut u32, n: u32) -> *mut _ {
        st.global.u32([p], n);
    }

    pub unsafe extern "ptx-kernel" fn module(p: *mut u32, n: u32) -> core::f16 {
        st.global.u32([p], n);
    }

    pub unsafe extern "ptx-kernel" fn wide(p: *mut u32, n: u32) -> *mut f128 {
        st.global.u32([p], n);
    }

    pub unsafe extern "ptx-kernel" fn import(p: *mut u32, n: u32) -> Ret {
        st.global.u32([p], n);
    }
}

#[kernite_macros::kernels]
pub mod gpu_tiles {
    use super::Ret;

    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn size<const N: u32>(p: *mut u32) -> *mut core::ffi::CStr {
        st.global.u32([p], N);
    }

    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn array<const N: usize>(p: *mut u32) -> *mut [u8; N] {
        st.global.u32([p], 1);
    }

    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn import<const N: u32>(p: *mut u32) -> Ret {
        st.global.u32([p], N);
    }
}

#[kernite_macros::kernels]
pub mod gpu_coop {
    use super::Ret;

    #[cooperative]
    pub unsafe extern "ptx-kernel" fn size(p: *mut u32, n: u32) -> *mut core::ffi::CStr {
        st.global.u32([p], n);
    }

    #[cooperative]
    pub unsafe extern "ptx-kernel" fn import(p: *mut u32, n: u32) -> Ret {
        st.global.u32([p], n);
    }
}

/// Kernels that give back `()`, which is no result: they are kernels, and their launchers give back the
/// `Result` of the launch.
#[kernite_macros::kernels]
pub mod cpu_unit {
    #[cooperative]
    pub unsafe extern "sysv64" fn unit(p: *mut u32) -> () {
        mov(dword[p], 1);
    }
}

#[kernite_macros::kernels]
pub mod gpu_unit {
    pub unsafe extern "ptx-kernel" fn unit(p: *mut u32, n: u32) -> () {
        st.global.u32([p], n);
    }
}

/// A pointer of two words to a type Rust refuses by itself, `*mut dyn Copy`: Rust says so of a plain kernel's
/// result, and of a launcher's it does not.
#[kernite_macros::kernels]
pub mod plain_object {
    pub unsafe extern "sysv64" fn object(p: *const u8) -> *mut dyn Copy {
        ret(p);
    }
}

#[kernite_macros::kernels]
pub mod cpu_object {
    #[cooperative]
    pub unsafe extern "sysv64" fn object(p: *const u8) -> *mut dyn Copy {
        ret(p);
    }
}

#[kernite_macros::kernels]
pub mod gpu_object {
    pub unsafe extern "ptx-kernel" fn object(p: *mut u32, n: u32) -> *mut dyn Copy {
        st.global.u32([p], n);
    }
}

/// `!` is no result of these kernels either, and a pointer to it, `*const !`, is a type that Rust has not made
/// stable: it says so at the `!` of a plain kernel's result, and of a cooperative kernel's and a GPU kernel's.
#[kernite_macros::kernels]
pub mod plain_never {
    pub unsafe extern "sysv64" fn pointer(p: *const u8) -> *const ! {
        ret(p);
    }
}

#[kernite_macros::kernels]
pub mod cpu_never {
    #[cooperative]
    pub unsafe extern "sysv64" fn forever(p: *mut u32) -> ! {
        mov(dword[p], 1);
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn pointer(p: *const u8) -> *const ! {
        ret(p);
    }
}

#[kernite_macros::kernels]
pub mod gpu_never {
    pub unsafe extern "ptx-kernel" fn forever(p: *mut u32, n: u32) -> ! {
        st.global.u32([p], n);
    }

    pub unsafe extern "ptx-kernel" fn pointer(p: *mut u32, n: u32) -> *mut ! {
        st.global.u32([p], n);
    }
}

/// Each x86 kernel, called as Rust calls it.
pub fn on_the_cpu(p: *const u8, q: *mut u32) -> Result<(), kernite_cpu::Error> {
    let grid = || kernite_cpu::Grid::new(1);
    unsafe {
        let _: u64 = plain::reference(p);
        plain::import(p);
        cpu::size(grid(), p)?;
        let _ = cpu::bound(grid(), p)?;
        cpu::reference(grid(), p).unwrap();
        let _: Result<(), kernite_cpu::Error> = cpu::lifetime(grid(), p);
        let _: u64 = cpu::blank(grid(), p);
        cpu::module(grid(), p)?;
        cpu::wide(grid(), p)?;
        match cpu::import(grid(), p) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        cpu_tiles::size::<1>(grid(), p)?;
        cpu_tiles::array::<2>(grid(), p).unwrap();
        cpu_tiles::import::<2>(grid(), p)?;
        let done: Result<(), kernite_cpu::Error> = cpu_unit::unit(grid(), q);
        done?;
        cpu_object::object(grid(), p)?;
        plain_never::pointer(p);
        cpu_never::forever(grid(), q)?;
        cpu_never::pointer(grid(), p)?;
    }
    Ok(())
}

/// Each GPU kernel, launched as Rust launches it.
pub fn on_the_gpu(
    gpu: &gpu::Kernels,
    tiles: &gpu_tiles::Kernels,
    coop: &gpu_coop::Kernels,
    nothing: &gpu_unit::Kernels,
    object: &gpu_object::Kernels,
    never: &gpu_never::Kernels,
    p: kernite_cuda::Ptr<u32>,
) -> Result<(), kernite_cuda::Error> {
    let grid = || kernite_cuda::Grid::new(1, 1);
    unsafe {
        gpu.size(grid(), p, 1)?;
        let _ = gpu.bound(grid(), p, 1)?;
        gpu.reference(grid(), p, 1).unwrap();
        let _: Result<(), kernite_cuda::Error> = gpu.lifetime(grid(), p, 1);
        let _: u64 = gpu.blank(grid(), p, 1);
        gpu.module(grid(), p, 1)?;
        gpu.wide(grid(), p, 1)?;
        match gpu.import(grid(), p, 1) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        tiles.size::<1>(grid(), p)?;
        tiles.array::<2>(grid(), p).unwrap();
        tiles.import::<2>(grid(), p)?;
        coop.size(grid(), p, 1)?;
        coop.import(grid(), p, 1)?;
        coop.size_blocks(256, 0)?;
        let done: Result<(), kernite_cuda::Error> = nothing.unit(grid(), p, 1);
        done?;
        object.object(grid(), p, 1)?;
        never.forever(grid(), p, 1)?;
        never.pointer(grid(), p, 1)?;
    }
    Ok(())
}
