//! A module of kernels whose `LISTINGS` holds each x86 kernel's listing by its name: a generic
//! kernel's, an instance each, as `plus::<2>`, a grid's kernel's, one whose note breaks lines, which the listing shows
//! as one, and one whose note and ABI hold escapes, which the listing shows as the characters they stand for. Rust
//! checks each as it builds.

#[kernite_macros::kernels]
pub mod kernels {
    /// `x` and `N`.
    #[instances(2, 3)]
    pub unsafe extern "sysv64" fn plus<const N: u64>(x: u64) -> u64 {
        add(x, N);
        ret(x);
    }

    /// One at `x`, from each thread of the grid.
    #[cooperative]
    pub unsafe extern "sysv64" fn together(x: *mut u32) {
        mov(dword[x], 1);
    }

    /// A note of three line breaks, `\n`, `\r\n` and `\r`, which the listing shows as three spaces.
    pub unsafe extern "sysv64" fn noted() {
        note("a\nint3\r\nint3\rint3");
    }

    /// A note of escapes: `\x41`, `\u{e9}` and a `\` that ends a line, which leave `A`, `é` and `ab`.
    pub unsafe extern "sysv64" fn escaped() {
        note("\x41 caf\u{e9} a\
              b");
    }

    /// An ABI of escapes, `"sysv\x364"`, which is `sysv64`.
    pub unsafe extern "sysv\x364" fn spelled() {
        note("spelled");
    }
}

// Each name and listing, as bytes: Rust compares those as it builds, and not yet `str`s.
const fn listings() {
    let [two, three, together, noted, escaped, spelled] = kernels::LISTINGS else { return };
    assert!(matches!(two.0.as_bytes(), b"plus::<2>"));
    assert!(matches!(
        two.1.as_bytes(),
        b"# plus::<2>(gpr) -> gpr
# frame: 0 bytes aligned to 0, saves []
# x=rdi,rax
    add rdi, 2
    mov rax, rdi
    ret
"
    ));
    assert!(matches!(three.0.as_bytes(), b"plus::<3>"));
    assert!(matches!(
        three.1.as_bytes(),
        b"# plus::<3>(gpr) -> gpr
# frame: 0 bytes aligned to 0, saves []
# x=rdi,rax
    add rdi, 3
    mov rax, rdi
    ret
"
    ));
    assert!(matches!(together.0.as_bytes(), b"together"));
    assert!(matches!(
        together.1.as_bytes(),
        b"# together(gpr, gpr) -> ()
# frame: 0 bytes aligned to 0, saves []
# x=rdi grid=rsi
    mov dword ptr [rdi], 1
    ret
"
    ));
    assert!(matches!(noted.0.as_bytes(), b"noted"));
    assert!(matches!(
        noted.1.as_bytes(),
        b"# noted() -> ()
# frame: 0 bytes aligned to 0, saves []
    # a int3 int3 int3
    ret
"
    ));
    assert!(matches!(escaped.0.as_bytes(), b"escaped"));
    assert!(matches!(
        escaped.1.as_bytes(),
        b"# escaped() -> ()
# frame: 0 bytes aligned to 0, saves []
    # A caf\xc3\xa9 ab
    ret
"
    ));
    assert!(matches!(spelled.0.as_bytes(), b"spelled"));
    assert!(matches!(
        spelled.1.as_bytes(),
        b"# spelled() -> ()
# frame: 0 bytes aligned to 0, saves []
    # spelled
    ret
"
    ));
}

const _: () = {
    assert!(kernels::LISTINGS.len() == 6, "a listing for each instance and kernel");
    listings();
};
