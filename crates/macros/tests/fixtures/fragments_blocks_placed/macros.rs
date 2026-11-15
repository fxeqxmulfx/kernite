macro_rules! files {
    ($module:ident, $b:block, $c:block) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "sysv64" fn first(x: u64) -> &u8 $b
            pub unsafe extern "sysv64" fn second(x: u64) -> &u8 $c
        }
    };
}

macro_rules! files_idents {
    ($module:ident, $t:ident, $b:block, $c:block) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "sysv64" fn first(x: u64) -> &$t $b
            pub unsafe extern "sysv64" fn second(x: u64) -> &$t $c
        }
    };
}
