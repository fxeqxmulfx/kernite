//! Kernels of instances, and a cooperative one, whose types hold braces Rust finds unnecessary,
//! `[u8; { N }]`, and which set the lint's level: `deny`, `forbid`, `warn`, `expect`, or `allow` in
//! a `cfg_attr`. Rust says the lint once, at the braces, at the level the kernel sets, as of any
//! function; an `expect` of it that is met says nothing, and one that is not says so, once.

#[kernite_macros::kernels]
pub mod cpu {
    #[deny(unused_braces)]
    #[instances(2, 3)]
    pub unsafe extern "sysv64" fn denied<const N: usize>(p: *mut [u8; { N }]) {
        mov(byte[p], N);
    }

    #[forbid(unused_braces)]
    #[cooperative]
    #[instances(2, 3)]
    pub unsafe extern "sysv64" fn forbidden<const N: usize>(p: *mut [u8; { N }]) {
        mov(byte[p], N);
    }

    #[warn(unused_braces)]
    #[instances(2, 3)]
    pub unsafe extern "sysv64" fn warned<const N: usize>(p: *mut [u8; { N }]) {
        mov(byte[p], N);
    }

    #[expect(unused_braces)]
    #[instances(2, 3)]
    pub unsafe extern "sysv64" fn met<const N: usize>(p: *mut [u8; { N }]) {
        mov(byte[p], N);
    }

    #[expect(unused_braces)]
    #[instances(2, 3)]
    pub unsafe extern "sysv64" fn unmet<const N: usize>(p: *mut [u8; N]) {
        mov(byte[p], N);
    }

    #[cfg_attr(all(), allow(unused_braces))]
    #[instances(2, 3)]
    pub unsafe extern "sysv64" fn configured<const N: usize>(p: *mut [u8; { N }]) {
        mov(byte[p], N);
    }
}
