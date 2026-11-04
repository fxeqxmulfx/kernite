//! GPU kernels that take a type kernite refuses, one that Rust has a word for as well: `f128`, an `f16` in a
//! type, `!`, and a type with no size; or one that holds an `f16`, an `f128` or a `!` that is no type, or is a
//! type of the crate's so named. kernite says each at the type, once for each instance, and Rust nothing more: not
//! that the type is unstable or has no size, though the macros leave a launcher for the kernel that never runs,
//! nor anything at a call of the launcher, which the crate's own function makes as it makes any call.

#[kernite_macros::kernels]
pub mod wide {
    pub unsafe extern "ptx-kernel" fn plain(x: f128) {
        ret();
    }

    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn tiles<const N: u32>(x: f128) {
        ret();
    }

    #[cooperative]
    pub unsafe extern "ptx-kernel" fn together(x: f128) {
        ret();
    }

    #[cooperative]
    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn together_tiles<const N: u32>(x: f128) {
        ret();
    }
}

#[kernite_macros::kernels]
pub mod array {
    pub unsafe extern "ptx-kernel" fn plain(x: [f16; 4]) {
        ret();
    }

    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn tiles<const N: u32>(x: [f16; 4]) {
        ret();
    }

    #[cooperative]
    pub unsafe extern "ptx-kernel" fn together(x: [f16; 4]) {
        ret();
    }

    #[cooperative]
    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn together_tiles<const N: u32>(x: [f16; 4]) {
        ret();
    }
}

#[kernite_macros::kernels]
pub mod never {
    pub unsafe extern "ptx-kernel" fn plain(x: !) {
        ret();
    }

    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn tiles<const N: u32>(x: !) {
        ret();
    }

    #[cooperative]
    pub unsafe extern "ptx-kernel" fn together(x: !) {
        ret();
    }

    #[cooperative]
    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn together_tiles<const N: u32>(x: !) {
        ret();
    }
}

#[kernite_macros::kernels]
pub mod slice {
    pub unsafe extern "ptx-kernel" fn plain(x: [u64]) {
        ret();
    }

    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn tiles<const N: u32>(x: [u64]) {
        ret();
    }

    #[cooperative]
    pub unsafe extern "ptx-kernel" fn together(x: [u64]) {
        ret();
    }

    #[cooperative]
    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn together_tiles<const N: u32>(x: [u64]) {
        ret();
    }
}

#[kernite_macros::kernels]
pub mod text {
    pub unsafe extern "ptx-kernel" fn plain(x: str) {
        ret();
    }

    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn tiles<const N: u32>(x: str) {
        ret();
    }

    #[cooperative]
    pub unsafe extern "ptx-kernel" fn together(x: str) {
        ret();
    }

    #[cooperative]
    #[instances(1, 2)]
    pub unsafe extern "ptx-kernel" fn together_tiles<const N: u32>(x: str) {
        ret();
    }
}

#[kernite_macros::kernels]
pub mod several {
    /// A parameter kernite takes, one it refuses, a pointer, and one it refuses: kernite says the first
    /// it refuses, and Rust says nothing of the other.
    pub unsafe extern "ptx-kernel" fn mixed(a: u32, x: f128, p: *mut u8, y: [u64]) {
        ret();
    }
}

/// A type of the crate's, which a kernel takes by value.
pub struct Params(pub u64);

#[kernite_macros::kernels]
pub mod imported {
    use super::Params;

    /// `Params` is named by this kernel alone, which kernite refuses: the import is used all the same.
    pub unsafe extern "ptx-kernel" fn plain(x: Params) {
        ret();
    }
}

#[kernite_macros::kernels]
pub mod beside {
    use super::Params;

    /// `Params` is named beside an `f16`, which Rust says is unstable wherever it finds it: kernite's error is
    /// alone, and the import is used.
    pub unsafe extern "ptx-kernel" fn plain(x: (Params, f16)) {
        ret();
    }
}

macro_rules! named {
    ($i:ident) => {
        u8
    };
}

/// A type of the crate's with a const argument.
pub struct Marker<const N: usize>;

#[kernite_macros::kernels]
pub mod not_length {
    /// A type with a `!` that is no type: an operator, in an array's length.
    pub unsafe extern "ptx-kernel" fn plain(x: [u8; !0usize & 3]) {
        ret();
    }
}

#[kernite_macros::kernels]
pub mod unequal_length {
    /// A `!` that is half of `!=`, in a group of an array's length.
    pub unsafe extern "ptx-kernel" fn plain(x: [u8; (1 != 2) as usize]) {
        ret();
    }
}

#[kernite_macros::kernels]
pub mod block_argument {
    use super::Marker;

    /// A `!` that is an operator, in a block: the const argument of a type.
    pub unsafe extern "ptx-kernel" fn plain(x: Marker<{ !0usize & 3 }>) {
        ret();
    }
}

#[kernite_macros::kernels]
pub mod macro_argument {
    /// An `f16` that is a name, among a macro's arguments: the macro takes a name, and writes a type.
    pub unsafe extern "ptx-kernel" fn plain(x: named!(f16)) {
        ret();
    }
}

/// A module of the crate, named as a half is.
pub mod f16 {
    /// A type of it.
    pub struct X;
}

#[kernite_macros::kernels]
pub mod parameter_name {
    /// An `f128` that is the name of a function pointer's parameter: no type.
    pub unsafe extern "ptx-kernel" fn plain(x: fn(f128: u8)) {
        ret();
    }
}

#[kernite_macros::kernels]
pub mod module_name {
    use super::f16;

    /// An `f16` that is the first segment of a path, the module's name: no type.
    pub unsafe extern "ptx-kernel" fn plain(x: f16::X) {
        ret();
    }
}

#[kernite_macros::kernels]
pub mod no_module_f16 {
    /// An `f16` that is the first segment of a path, and the crate's module of this name is not in scope: no
    /// type.
    pub unsafe extern "ptx-kernel" fn plain(x: f16::X) {
        ret();
    }
}

#[kernite_macros::kernels]
pub mod no_module_f128 {
    /// An `f128` that is the first segment of a path, and no module of this name is in scope: no type.
    pub unsafe extern "ptx-kernel" fn plain(x: f128::X) {
        ret();
    }
}

macro_rules! f16 {
    () => {
        u64
    };
}

#[kernite_macros::kernels]
pub mod macro_name {
    /// An `f16` that is the name of a macro: no type, and the macro writes a `u64`.
    pub unsafe extern "ptx-kernel" fn plain(x: f16!()) {
        ret();
    }
}

/// A const of the crate, named as an unstable type is.
#[allow(non_upper_case_globals)]
pub const f128: usize = 2;

#[kernite_macros::kernels]
pub mod length_name {
    use super::f128;

    /// An `f128` that is the whole length of an array, a const's name: no type. The module does not hold the
    /// const, so kernite says the name is not defined.
    pub unsafe extern "ptx-kernel" fn plain(x: [u8; f128]) {
        ret();
    }
}

/// Types of the crate, named as unstable types are.
#[allow(non_camel_case_types)]
pub mod types {
    /// A type named as a half is.
    pub struct f16;

    /// A type named as a wide float is.
    pub struct f128;
}

#[kernite_macros::kernels]
pub mod crate_f128 {
    use super::types::f128;

    /// An `f128` that is a type of the crate's, which the module imports: no unstable type. The import is used.
    pub unsafe extern "ptx-kernel" fn plain(x: f128) {
        ret();
    }
}

#[kernite_macros::kernels]
pub mod crate_array {
    use super::types::f16;

    /// An `f16` in an array that is a type of the crate's, which the module imports. The import is used.
    pub unsafe extern "ptx-kernel" fn plain(x: [f16; 4]) {
        ret();
    }
}

#[kernite_macros::kernels]
pub mod crate_path {
    use super::types::f16;

    /// An `f16` that is the later segment of a path, `self::f16`, to the type the module imports: a name, no
    /// type. A bare `f16` is a half to a GPU kernel, so it is the path that is refused. The import is used.
    pub unsafe extern "ptx-kernel" fn plain(x: self::f16) {
        ret();
    }
}

#[kernite_macros::kernels]
pub mod core_path {
    /// An `f128` that is the later segment of a path, `core::primitive::f128`: a name, no type.
    pub unsafe extern "ptx-kernel" fn plain(x: core::primitive::f128) {
        ret();
    }
}

#[kernite_macros::kernels]
pub mod parameter_path {
    /// An `f128` that is the name of a function pointer's parameter, whose type is a path from the root,
    /// `::core::primitive::u8`: no type. The `:` of the name is not the first of a `::`.
    pub unsafe extern "ptx-kernel" fn plain(x: fn(f128: ::core::primitive::u8)) {
        ret();
    }
}

/// Each kernel, launched as Rust launches it: with a value of any type where the kernel takes one of a type
/// it refuses.
pub fn on_the_gpu(
    wide: &wide::Kernels,
    array: &array::Kernels,
    never: &never::Kernels,
    slice: &slice::Kernels,
    text: &text::Kernels,
    several: &several::Kernels,
    imported: &imported::Kernels,
    beside: &beside::Kernels,
    not_length: &not_length::Kernels,
    unequal_length: &unequal_length::Kernels,
    block_argument: &block_argument::Kernels,
    macro_argument: &macro_argument::Kernels,
    parameter_name: &parameter_name::Kernels,
    module_name: &module_name::Kernels,
    no_module_f16: &no_module_f16::Kernels,
    no_module_f128: &no_module_f128::Kernels,
    macro_name: &macro_name::Kernels,
    length_name: &length_name::Kernels,
    crate_f128: &crate_f128::Kernels,
    crate_array: &crate_array::Kernels,
    crate_path: &crate_path::Kernels,
    core_path: &core_path::Kernels,
    parameter_path: &parameter_path::Kernels,
    bytes: kernite_cuda::Ptr<u8>,
) -> Result<(), kernite_cuda::Error> {
    let grid = || kernite_cuda::Grid::new(1, 1);
    unsafe {
        wide.plain(grid(), 1.0)?;
        wide.tiles::<1>(grid(), 1.0)?;
        wide.together(grid(), 1.0)?;
        wide.together_tiles::<2>(grid(), 1.0)?;
        array.plain(grid(), [0.0; 4])?;
        array.tiles::<1>(grid(), [0.0; 4])?;
        array.together(grid(), [0.0; 4])?;
        array.together_tiles::<2>(grid(), [0.0; 4])?;
        never.plain(grid(), 1)?;
        never.tiles::<1>(grid(), 1)?;
        never.together(grid(), 1)?;
        never.together_tiles::<2>(grid(), 1)?;
        slice.plain(grid(), 1)?;
        slice.tiles::<1>(grid(), 1)?;
        slice.together(grid(), 1)?;
        slice.together_tiles::<2>(grid(), 1)?;
        text.plain(grid(), 1)?;
        text.tiles::<1>(grid(), 1)?;
        text.together(grid(), 1)?;
        text.together_tiles::<2>(grid(), 1)?;
        several.mixed(grid(), 1, 1.0, bytes, 1)?;
        imported.plain(grid(), Params(1))?;
        beside.plain(grid(), (Params(1), 0.0))?;
        not_length.plain(grid(), [0; 3])?;
        unequal_length.plain(grid(), [0])?;
        block_argument.plain(grid(), Marker)?;
        macro_argument.plain(grid(), 1)?;
        parameter_name.plain(grid(), drop::<u8>)?;
        module_name.plain(grid(), f16::X)?;
        no_module_f16.plain(grid(), 1)?;
        no_module_f128.plain(grid(), 1)?;
        macro_name.plain(grid(), 1)?;
        length_name.plain(grid(), [0; 2])?;
        crate_f128.plain(grid(), types::f128)?;
        crate_array.plain(grid(), [types::f16, types::f16, types::f16, types::f16])?;
        crate_path.plain(grid(), types::f16)?;
        core_path.plain(grid(), 1)?;
        parameter_path.plain(grid(), drop::<u8>)?;
    }
    Ok(())
}
