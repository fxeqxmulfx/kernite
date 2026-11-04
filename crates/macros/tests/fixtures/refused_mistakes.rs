//! Modules of x86 kernels, of GPU kernels and of both kinds, which `#[kernels]` refuses at its attribute, `consts = 5`.
//! Their items hold a mistake of each kind kernite finds in an item; a compile-time argument is named as a parameter
//! of a launcher or of a `_blocks` method, and a kernel as a cooperative kernel's `_blocks` method. kernite checks
//! none of the items, so each module is told its refusal and those names, and none of the mistakes. With the feature
//! `taken` the same modules are not refused, and each mistake is said.

#[cfg_attr(not(feature = "taken"), kernite_macros::kernels(consts = 5))]
#[cfg_attr(feature = "taken", kernite_macros::kernels)]
pub mod cpu {
    // A kernel that is not `unsafe`.
    pub extern "sysv64" fn plain(x: u64) -> u64 {
        ret(x);
    }

    // `#[instances(..)]` on a kernel with no compile-time argument.
    #[instances(1)]
    pub unsafe extern "sysv64" fn once(x: u64) -> u64 {
        ret(x);
    }

    // A name that is not defined.
    pub unsafe extern "sysv64" fn broken(x: u64) -> u64 {
        add(x, nope);
        ret(x);
    }

    // `#[instances(..)]` on a helper.
    #[instances(1)]
    fn helper(x: u64) -> u64 {
        ret(x);
    }

    // An ABI kernite does not compile for.
    pub unsafe extern "C" fn other() {}

    // A static that holds nothing.
    pub static EMPTY: global<[u32; 0]>;

    // A compile-time argument named as the parameter of the launcher.
    #[instances(1)]
    #[cooperative]
    #[allow(non_upper_case_globals)]
    pub unsafe extern "sysv64" fn together<const grid: u64>(x: *mut u64) {
        mov(qword[x], 1);
    }
}

#[cfg_attr(not(feature = "taken"), kernite_macros::kernels(consts = 5))]
#[cfg_attr(feature = "taken", kernite_macros::kernels)]
pub mod gpu {
    pub extern "ptx-kernel" fn plain(x: *mut u32) {
        st.global.u32([x], 1);
    }

    #[instances(1)]
    pub unsafe extern "ptx-kernel" fn once(x: *mut u32) {
        st.global.u32([x], 1);
    }

    pub unsafe extern "ptx-kernel" fn broken(x: *mut u32) {
        st.global.u32([x], nope);
    }

    pub static EMPTY: global<[u32; 0]>;

    #[instances(1)]
    #[allow(non_upper_case_globals)]
    pub unsafe extern "ptx-kernel" fn launched<const grid: u32>(x: *mut u32) {
        st.global.u32([x], 1);
    }

    // Compile-time arguments named as the parameters of `together_blocks`.
    #[instances((1, 2))]
    #[cooperative]
    #[allow(non_upper_case_globals)]
    pub unsafe extern "ptx-kernel" fn together<const threads: u32, const shared: u32>(x: *mut u32) {
        st.global.u32([x], 1);
    }

    // A kernel named as the method `cut_blocks`, which says how many blocks a grid of `cut` has at most.
    #[cooperative]
    pub unsafe extern "ptx-kernel" fn cut(x: *mut u32) {
        grid.sync();
    }

    pub unsafe extern "ptx-kernel" fn cut_blocks(x: *mut u32) {
        st.global.u32([x], 1);
    }
}

// Kernels of both kinds and a static, which makes it two modules.
#[cfg_attr(not(feature = "taken"), kernite_macros::kernels(consts = 5))]
#[cfg_attr(feature = "taken", kernite_macros::kernels)]
pub mod both {
    pub static B: global<u32>;

    pub unsafe extern "sysv64" fn x() {
        ret();
    }

    pub unsafe extern "ptx-kernel" fn k(p: *mut u32) {
        st.global.u32([p], 1);
    }
}
