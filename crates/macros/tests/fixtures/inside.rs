//! Errors of the language inside a module, each where it is in the module.

#[kgen_macros::kernels]
pub mod both {
    pub unsafe extern "sysv64" fn add(n: u64) {
        let v: ymm;
        vaddps(v, v, n);
    }

    fn broken() {
        let ;
    }

    #[instances(1)]
    fn helper() {}

    pub unsafe extern "ptx-kernel" fn scale(x: *mut f32) {}

    pub static COUNTS: global<[u32; 16]>;
}
