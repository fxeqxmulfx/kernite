macro_rules! files {
    ($module:ident, $t:ident) => {
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
    ($module:ident, $t:ident) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "sysv64" fn only(x: u64) -> &$t {
                ret(x);
            }
        }
    };
}

macro_rules! files_tree {
    ($module:ident, $t:tt) => {
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

macro_rules! whole {
    ($module:ident, $r:ty) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "sysv64" fn first(x: u64) -> $r {
                ret(x);
            }
            pub unsafe extern "sysv64" fn second(x: u64) -> $r {
                ret(x);
            }
        }
    };
}
