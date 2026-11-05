//! Each thing wrong with an attribute kernite reads, written plain and written raw, and two of one attribute
//! written one plain and one raw, which are two all the same. kernite says the same of each, in the same words,
//! at the same column of its line, and Rust says nothing. Each case is a module of its own, for an attribute that
//! is wrong stops the module.

#[kernite_macros::kernels]
pub mod plain_instances_no_list {
    #[instances]
    pub unsafe extern "sysv64" fn k<const N: u64>() {}
}

#[kernite_macros::kernels]
pub mod raw_instances_no_list {
    #[r#instances]
    pub unsafe extern "sysv64" fn k<const N: u64>() {}
}

#[kernite_macros::kernels]
pub mod plain_instances_twice {
    #[instances(1)]
    #[instances(2)]
    pub unsafe extern "sysv64" fn k<const N: u64>() {}
}

#[kernite_macros::kernels]
pub mod raw_instances_twice {
    #[r#instances(1)]
    #[r#instances(2)]
    pub unsafe extern "sysv64" fn k<const N: u64>() {}
}

#[kernite_macros::kernels]
pub mod mixed_instances_twice {
    #[instances(1)]
    #[r#instances(2)]
    pub unsafe extern "sysv64" fn k<const N: u64>() {}
}

#[kernite_macros::kernels]
pub mod plain_cooperative_argument {
    #[cooperative(1)]
    pub unsafe extern "sysv64" fn k() {}
}

#[kernite_macros::kernels]
pub mod raw_cooperative_argument {
    #[r#cooperative(1)]
    pub unsafe extern "sysv64" fn k() {}
}

#[kernite_macros::kernels]
pub mod plain_cooperative_twice {
    #[cooperative]
    #[cooperative]
    pub unsafe extern "sysv64" fn k() {}
}

#[kernite_macros::kernels]
pub mod raw_cooperative_twice {
    #[r#cooperative]
    #[r#cooperative]
    pub unsafe extern "sysv64" fn k() {}
}

#[kernite_macros::kernels]
pub mod mixed_cooperative_twice {
    #[cooperative]
    #[r#cooperative]
    pub unsafe extern "sysv64" fn k() {}
}

#[kernite_macros::kernels]
pub mod plain_directive_no_list {
    #[maxnreg]
    pub unsafe extern "ptx-kernel" fn k() {}
}

#[kernite_macros::kernels]
pub mod raw_directive_no_list {
    #[r#maxnreg]
    pub unsafe extern "ptx-kernel" fn k() {}
}

#[kernite_macros::kernels]
pub mod plain_directive_twice {
    #[maxntid(256)]
    #[maxntid(128)]
    pub unsafe extern "ptx-kernel" fn k() {}
}

#[kernite_macros::kernels]
pub mod raw_directive_twice {
    #[r#maxntid(256)]
    #[r#maxntid(128)]
    pub unsafe extern "ptx-kernel" fn k() {}
}

#[kernite_macros::kernels]
pub mod mixed_directive_twice {
    #[maxntid(256)]
    #[r#maxntid(128)]
    pub unsafe extern "ptx-kernel" fn k() {}
}

#[kernite_macros::kernels]
pub mod plain_old_directive {
    #[maxnctapersm(2)]
    pub unsafe extern "ptx-kernel" fn k() {}
}

#[kernite_macros::kernels]
pub mod raw_old_directive {
    #[r#maxnctapersm(2)]
    pub unsafe extern "ptx-kernel" fn k() {}
}

#[kernite_macros::kernels]
pub mod plain_directive_on_x86 {
    #[maxntid(256)]
    pub unsafe extern "sysv64" fn k() {}
}

#[kernite_macros::kernels]
pub mod raw_directive_on_x86 {
    #[r#maxntid(256)]
    pub unsafe extern "sysv64" fn k() {}
}

#[kernite_macros::kernels(consts = 5)]
pub mod plain_consts_refused {
    pub unsafe extern "sysv64" fn k() {}
}

#[kernite_macros::kernels(r#consts = 5)]
pub mod raw_consts_refused {
    pub unsafe extern "sysv64" fn k() {}
}
