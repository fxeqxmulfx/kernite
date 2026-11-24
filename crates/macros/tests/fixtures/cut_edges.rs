//! A module of two kernels, whose PTX the macros cut at 4096 pieces: `even` has 8192 pieces, two cuts exactly, and
//! `odd` has 8193, two cuts and one more piece. A hole is where each name goes: the kernel's own, and each use of the
//! static, so `even` has 8190 uses and `odd` 8191.

#[kernite_macros::kernels]
pub mod gpu {
    static COUNTS: global<[u32; 16]>;

    /// Has 8192 pieces of PTX.
    pub unsafe extern "ptx-kernel" fn even() {
        for _ in 0..8190 {
            red.global.add.u32([COUNTS + 8], 1);
        }
    }

    /// Has 8193 pieces of PTX.
    pub unsafe extern "ptx-kernel" fn odd() {
        for _ in 0..8191 {
            red.global.add.u32([COUNTS + 8], 1);
        }
    }
}

#[test]
fn each_hole_has_the_names_of_the_path_whichever_the_cut() {
    let ptx = gpu::PTX;
    // What fills a hole: the crate's name and the module's. The workspace's, `7project`, is no hole.
    let hole = "9cut_edges3gpu";
    // The holes: the name of each kernel, each use of the static, and the static's declaration.
    assert_eq!(ptx.matches(hole).count(), (1 + 8190) + (1 + 8191) + 1, "the names of the path in the PTX");
    assert_eq!(ptx.matches(&format!("[_ZN7project{hole}6COUNTSE+8]")).count(), 8190 + 8191, "the uses of the static");
    assert!(ptx.contains(&format!(".entry _ZN7project{hole}4evenE(")), "no kernel `even`");
    assert!(ptx.contains(&format!(".entry _ZN7project{hole}3oddE(")), "no kernel `odd`");
    assert!(!ptx.contains('\0'), "a hole is left in the PTX");
}
