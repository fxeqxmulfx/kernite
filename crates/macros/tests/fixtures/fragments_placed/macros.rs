macro_rules! files {
    ($module:ident, $t:ty) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "sysv64" fn first(x: u64) -> &$t {
                ret(x);
            }
            pub unsafe extern "sysv64" fn second(x: u64) -> Option<&$t> {
                ret(x);
            }
        }
    };
}

macro_rules! alone {
    ($module:ident, $t:ty) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "sysv64" fn only(x: u64) -> &$t {
                ret(x);
            }
        }
    };
}
