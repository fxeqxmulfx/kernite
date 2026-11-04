//! Kernels whose types hold a block among their arguments, `Arr<{ 3 }>`, which Rust lints as it lints any braces
//! around a number. Five give back a pointer to such a type: plain, of compile-time arguments, with the block beside
//! another argument, in a type nested in another, and inside an array. One takes a pointer to it. The array and the
//! parameter were read whole before. Rust builds the crate with its lint on each pair of braces, at the braces, and
//! kernite says nothing. The tests call each x86 kernel: those that give back a pointer give back the one they are
//! given, and the one of compile-time arguments adds its argument to it first; the one that takes a pointer writes
//! the low byte of its argument through it.

/// A type with a number as its argument.
#[repr(C)]
pub struct Arr<const N: usize>(pub [u8; N]);

/// A type with a type and a number as its arguments.
#[repr(C)]
pub struct Pair<T, const N: usize>(pub T, pub [u8; N]);

/// x86 kernels.
#[kernite_macros::kernels]
pub mod cpu {
    pub unsafe extern "sysv64" fn plain(p: *const u8) -> *const crate::Arr<{ 3 }> {
        ret(p);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tiles<const N: u64>(p: *const u8) -> *const crate::Arr<{ 3 }> {
        add(p, N);
        ret(p);
    }

    pub unsafe extern "sysv64" fn beside(p: *const u8) -> *const crate::Pair<u8, { 3 }> {
        ret(p);
    }

    pub unsafe extern "sysv64" fn nested(p: *const u8) -> *const crate::Pair<crate::Arr<{ 3 }>, 4> {
        ret(p);
    }

    pub unsafe extern "sysv64" fn parameter(p: *mut crate::Arr<{ 3 }>, n: u64) {
        mov(byte[p], n.r8);
    }

    pub unsafe extern "sysv64" fn array(p: *const u8) -> *const [crate::Arr<{ 3 }>; 1] {
        ret(p);
    }
}

#[test]
fn each_kernel_with_a_result_gives_back_its_pointer() {
    let bytes = [7u8; 3];
    let p = bytes.as_ptr();
    unsafe {
        assert_eq!(cpu::plain(p).cast::<u8>(), p);
        assert_eq!(cpu::tiles::<1>(p).cast::<u8>(), p.wrapping_add(1));
        assert_eq!(cpu::tiles::<2>(p).cast::<u8>(), p.wrapping_add(2));
        assert_eq!(cpu::beside(p).cast::<u8>(), p);
        assert_eq!(cpu::nested(p).cast::<u8>(), p);
        assert_eq!(cpu::array(p).cast::<u8>(), p);
    }
}

#[test]
fn the_pointer_taken_is_written_through() {
    let mut arr = Arr([0u8; 3]);
    unsafe { cpu::parameter(&mut arr, 0x1234) };
    assert_eq!(arr.0, [0x34, 0, 0]);
}
