macro_rules! files {
    ($module:ident, $a:ident, $b:ident) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "sysv64" fn first(x: u64) -> &$a::$b {
                ret(x);
            }
            pub unsafe extern "sysv64" fn second(x: u64) -> &$a::$b {
                ret(x);
            }
        }
    };
}
