//! A module of kernels whose `LISTINGS` holds each x86 kernel's listing by its name: a generic
//! kernel's, an instance each, as `plus::<2>`, and a grid's kernel's. Rust checks each as it
//! builds.

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
}

// Each name and listing, as bytes: Rust compares those as it builds, and not yet `str`s.
const _: () = {
    let [two, three, together] = kernels::LISTINGS else { panic!("a listing for each instance and kernel") };
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
};
