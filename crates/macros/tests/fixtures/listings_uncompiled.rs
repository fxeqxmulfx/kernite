//! A module of x86 kernels has `LISTINGS` whether its kernels compile or not: the listing of each that compiles and
//! `cfg` keeps, and of no other. So Rust says nothing of the code that names it, a glob's included, beside the errors
//! of the kernels that do not compile; a module of GPU kernels alone has none. The consts assert what each `LISTINGS`
//! holds: Rust evaluates them though a kernel has not compiled, and one that fails is Rust's error, beside theirs.

/// One kernel compiles, one does not, and `cfg` leaves out one that would.
#[kernite_macros::kernels]
pub mod mixed {
    pub unsafe extern "sysv64" fn good(x: u64) -> u64 {
        add(x, 1);
        ret(x);
    }

    pub unsafe extern "sysv64" fn broken(x: u64) -> u64 {
        add(x, nope);
        ret(x);
    }

    #[cfg(any())]
    pub unsafe extern "sysv64" fn off(x: u64) -> u64 {
        ret(x);
    }
}

/// No kernel compiles.
#[kernite_macros::kernels]
pub mod none {
    pub unsafe extern "sysv64" fn broken(x: u64) -> u64 {
        add(x, nope);
        ret(x);
    }
}

/// A module of GPU kernels alone.
#[kernite_macros::kernels]
pub mod gpu {
    pub unsafe extern "ptx-kernel" fn scale(x: *mut f32, a: f32) {
        let v: f32;
        ld.global.f32(v, [x]);
        mul.rn.f32(v, v, a);
        st.global.f32([x], v);
    }
}

/// A `LISTINGS` of its own, for a glob of `gpu` to clash with if `gpu` had one.
pub mod mine {
    pub const LISTINGS: usize = 0;
}

const fn same(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    let mut i = 0;
    while i < a.len() && i < b.len() && a[i] == b[i] {
        i += 1;
    }
    i == a.len() && i == b.len()
}

const _: () = assert!(mixed::LISTINGS.len() == 1 && same(mixed::LISTINGS[0].0, "good"));
const _: () = assert!(!mixed::LISTINGS[0].1.is_empty());
const _: () = assert!(none::LISTINGS.is_empty());

/// By the names a glob brings in, `LISTINGS` among them, though no kernel of `none` compiles.
pub mod globbed {
    use crate::none::*;

    pub fn all() -> usize {
        LISTINGS.len() + unsafe { broken(1) } as usize
    }
}

/// By the names two globs bring in: `LISTINGS` is `mine` alone, for `gpu` has none.
pub mod beside_gpu {
    use crate::gpu::*;
    use crate::mine::*;

    pub fn all(_: Option<&Kernels>) -> usize {
        LISTINGS
    }
}
