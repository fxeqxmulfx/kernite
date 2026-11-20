//! Rust tells of an item with no doc, in a module of kernels that denies `missing_docs`, at the item, as it tells of an
//! item written in plain Rust, and not at the module's `#[kernite_macros::kernels]`, where nothing can be mended. Each
//! is told of at its own place: the statics of GPU kernels, which are methods of `Kernels`; the statics of x86 kernels,
//! near `rip` and far from it; and the x86 kernels, plain, giving back a number, a pointer or `()`, and of instances.
//! What has a doc is not told of, nor is a GPU kernel, whose launcher the macros write with a doc, nor a cooperative
//! x86 kernel, whose launcher they write with one, nor an x86 kernel that names a static of its module, whose safety
//! contract they write as a doc.

/// GPU statics of no doc, and one with a doc, and a GPU kernel.
#[kernite_macros::kernels]
pub mod gpu {
    #![deny(missing_docs)]

    pub static ONE: global<[u32; 4]>;

    /// A static with a doc.
    pub static DOCUMENTED: global<[u32; 4]>;

    pub static TWO: constant<[f32; 4]> = [1.0, 2.0, 3.0, 4.0];

    pub unsafe extern "ptx-kernel" fn bump() {
        red.global.add.u32([ONE], 1);
    }
}

/// x86 statics of no doc, a static with a doc, and kernels of no doc, in each form.
#[kernite_macros::kernels]
pub mod cpu {
    #![deny(missing_docs)]

    pub static NEAR: global<[u32; 4]>;

    pub static FAR: global<[u8; 3221225472]>;

    /// A static with a doc.
    pub static DOCUMENTED: global<[u32; 4]>;

    pub unsafe extern "sysv64" fn plain() {}

    pub unsafe extern "sysv64" fn number(x: u64) -> u64 {
        ret(x);
    }

    pub unsafe extern "sysv64" fn pointer(p: *const u8) -> *const u8 {
        ret(p);
    }

    pub unsafe extern "sysv64" fn unit() -> () {}

    #[instances(2, 3)]
    pub unsafe extern "sysv64" fn many<const N: u64>(x: u64) -> u64 {
        let y: gpr;
        lea(y, [x + N]);
        ret(y);
    }

    #[instances(2, 3)]
    pub unsafe extern "sysv64" fn many_unit<const N: u64>() {}

    /// A kernel with a doc.
    pub unsafe extern "sysv64" fn documented() {}

    pub unsafe extern "sysv64" fn names() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn together() {}
}
