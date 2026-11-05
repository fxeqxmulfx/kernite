//! A GPU module whose statics have no room beside each other where `cfg` keeps both: `A` and `B` are 48000 bytes
//! of constant memory each, and a module has 64 KB. Rust decides which `cfg` holds, so `__Room` counts the room,
//! and shows Rust the types of the helpers that `cfg` keeps where it says the error: `h` takes `Ret`, a type of
//! the crate's that the module imports, `lifetimes` a lifetime it has not declared, and `language` the language's
//! types, of which Rust says nothing. Where the room is right, the module builds, and shows none of them. `broken`
//! is a kernel that does not compile, where `d` holds, and Rust says of the types once beside both errors.

pub struct Ret(pub u64);

#[kernite_macros::kernels]
pub mod room {
    use super::Ret;

    #[cfg(feature = "a")]
    pub static A: constant<[u32; 12000]> = [0; 12000];

    #[cfg(feature = "b")]
    pub static B: constant<[u32; 12000]> = [0; 12000];

    #[cfg(feature = "c")]
    fn h(x: Ret) {
        nop();
    }

    #[cfg(feature = "c")]
    fn lifetimes(x: &'a Ret) {
        nop();
    }

    fn language(a: gpr, b: ymm, c: local<[u32; 4], 16>) -> r32 {
        nop();
    }

    pub unsafe extern "ptx-kernel" fn nothing() {
        ret();
    }

    #[cfg(feature = "d")]
    pub unsafe extern "ptx-kernel" fn broken() {
        nop(true);
    }
}
