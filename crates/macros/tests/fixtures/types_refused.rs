//! x86 kernels that take or give back a type kernite refuses, one that Rust has a word for as well: `f128`,
//! an `f16` in a type, `!` as a parameter, and a type with no size; or one that holds an `f16`, an `f128` or a
//! `!` that is no type, or is a type of the crate's so named. kernite says each at the type, once for each
//! instance, and Rust nothing more: not that the type is unstable or has no size, though the macros leave a stub
//! for the kernel, nor anything at a call of the kernel, which the crate's own function makes as it makes any call.

#[kernite_macros::kernels]
pub mod wide {
    pub unsafe extern "sysv64" fn plain(x: f128) {
        ret();
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tiles<const N: u64>(x: f128) {
        ret();
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn together(x: f128) {
        ret();
    }

    #[cooperative]
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn together_tiles<const N: u64>(x: f128) {
        ret();
    }
}

#[kernite_macros::kernels]
pub mod array {
    pub unsafe extern "sysv64" fn plain(x: [f16; 4]) {
        ret();
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tiles<const N: u64>(x: [f16; 4]) {
        ret();
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn together(x: [f16; 4]) {
        ret();
    }

    #[cooperative]
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn together_tiles<const N: u64>(x: [f16; 4]) {
        ret();
    }
}

#[kernite_macros::kernels]
pub mod tuple {
    pub unsafe extern "sysv64" fn plain(x: (f16, u64)) {
        ret();
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tiles<const N: u64>(x: (f16, u64)) {
        ret();
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn together(x: (f16, u64)) {
        ret();
    }

    #[cooperative]
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn together_tiles<const N: u64>(x: (f16, u64)) {
        ret();
    }
}

#[kernite_macros::kernels]
pub mod reference {
    pub unsafe extern "sysv64" fn plain(x: &f16) {
        ret();
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tiles<const N: u64>(x: &f16) {
        ret();
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn together(x: &f16) {
        ret();
    }

    #[cooperative]
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn together_tiles<const N: u64>(x: &f16) {
        ret();
    }
}

#[kernite_macros::kernels]
pub mod never {
    pub unsafe extern "sysv64" fn plain(x: !) {
        ret();
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tiles<const N: u64>(x: !) {
        ret();
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn together(x: !) {
        ret();
    }

    #[cooperative]
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn together_tiles<const N: u64>(x: !) {
        ret();
    }
}

#[kernite_macros::kernels]
pub mod slice {
    pub unsafe extern "sysv64" fn plain(x: [u64]) {
        ret();
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tiles<const N: u64>(x: [u64]) {
        ret();
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn together(x: [u64]) {
        ret();
    }

    #[cooperative]
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn together_tiles<const N: u64>(x: [u64]) {
        ret();
    }
}

#[kernite_macros::kernels]
pub mod text {
    pub unsafe extern "sysv64" fn plain(x: str) {
        ret();
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tiles<const N: u64>(x: str) {
        ret();
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn together(x: str) {
        ret();
    }

    #[cooperative]
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn together_tiles<const N: u64>(x: str) {
        ret();
    }
}

#[kernite_macros::kernels]
pub mod object {
    pub unsafe extern "sysv64" fn plain(x: dyn Fn()) {
        ret();
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tiles<const N: u64>(x: dyn Fn()) {
        ret();
    }

    #[cooperative]
    pub unsafe extern "sysv64" fn together(x: dyn Fn()) {
        ret();
    }

    #[cooperative]
    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn together_tiles<const N: u64>(x: dyn Fn()) {
        ret();
    }
}

#[kernite_macros::kernels]
pub mod several {
    /// A parameter kernite takes, one it refuses, a pointer, and one it refuses: kernite says the first
    /// it refuses, and Rust says nothing of the other.
    pub unsafe extern "sysv64" fn mixed(a: u64, x: f128, p: *mut u8, y: [u64]) {
        ret();
    }
}

/// A type of the crate's, which a kernel takes by value.
pub struct Params(pub u64);

#[kernite_macros::kernels]
pub mod imported {
    use super::Params;

    /// `Params` is named by this kernel alone, which kernite refuses: the import is used all the same.
    pub unsafe extern "sysv64" fn plain(x: Params) {
        ret();
    }
}

#[kernite_macros::kernels]
pub mod beside {
    use super::Params;

    /// `Params` is named beside an `f16`, which Rust says is unstable wherever it finds it: kernite's error is
    /// alone, and the import is used.
    pub unsafe extern "sysv64" fn plain(x: (Params, f16)) {
        ret();
    }
}

#[kernite_macros::kernels]
pub mod gives_wide {
    pub unsafe extern "sysv64" fn plain(x: u64) -> f128 {
        ret(x);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tiles<const N: u64>(x: u64) -> f128 {
        ret(x);
    }
}

#[kernite_macros::kernels]
pub mod gives_slice {
    pub unsafe extern "sysv64" fn plain(x: u64) -> [u64] {
        ret(x);
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tiles<const N: u64>(x: u64) -> [u64] {
        ret(x);
    }
}

#[kernite_macros::kernels]
pub mod deep {
    /// An `f16` and a `!` deep in a type, in angle brackets, a function pointer and its result.
    pub unsafe extern "sysv64" fn plain(x: Option<fn(Vec<f16>) -> !>) {
        ret();
    }
}

macro_rules! gone {
    () => {
        u64
    };
}

#[kernite_macros::kernels]
pub mod made {
    /// A type a macro writes: its `!` is the macro's, and Rust expands it.
    pub unsafe extern "sysv64" fn plain(x: gone!()) {
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
    pub unsafe extern "sysv64" fn plain(x: [u8; !0usize & 3]) {
        ret();
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tiles<const N: u64>(x: [u8; !0usize & 3]) {
        ret();
    }
}

#[kernite_macros::kernels]
pub mod unequal_length {
    /// A `!` that is half of `!=`, in a group of an array's length.
    pub unsafe extern "sysv64" fn plain(x: [u8; (1 != 2) as usize]) {
        ret();
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tiles<const N: u64>(x: [u8; (1 != 2) as usize]) {
        ret();
    }
}

#[kernite_macros::kernels]
pub mod block_argument {
    use super::Marker;

    /// A `!` that is an operator, in a block: the const argument of a type.
    pub unsafe extern "sysv64" fn plain(x: Marker<{ !0usize & 3 }>) {
        ret();
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tiles<const N: u64>(x: Marker<{ !0usize & 3 }>) {
        ret();
    }
}

#[kernite_macros::kernels]
pub mod macro_argument {
    /// An `f16` that is a name, among a macro's arguments: the macro takes a name, and writes a type.
    pub unsafe extern "sysv64" fn plain(x: named!(f16)) {
        ret();
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tiles<const N: u64>(x: named!(f16)) {
        ret();
    }
}

/// A module of the crate, named as an unstable type is.
pub mod f128 {
    /// A type of it.
    pub struct X;
}

#[kernite_macros::kernels]
pub mod parameter_name {
    /// An `f16` that is the name of a function pointer's parameter: no type.
    pub unsafe extern "sysv64" fn plain(x: fn(f16: u8)) {
        ret();
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tiles<const N: u64>(x: fn(f16: u8)) {
        ret();
    }
}

#[kernite_macros::kernels]
pub mod module_name {
    use super::f128;

    /// An `f128` that is the first segment of a path, the module's name: no type.
    pub unsafe extern "sysv64" fn plain(x: f128::X) {
        ret();
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tiles<const N: u64>(x: f128::X) {
        ret();
    }
}

#[kernite_macros::kernels]
pub mod no_module_f16 {
    /// An `f16` that is the first segment of a path, and no module of this name is in scope: no type.
    pub unsafe extern "sysv64" fn plain(x: f16::X) {
        ret();
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tiles<const N: u64>(x: f16::X) {
        ret();
    }
}

#[kernite_macros::kernels]
pub mod no_module_f128 {
    /// An `f128` that is the first segment of a path, and the crate's module of this name is not in scope: no
    /// type.
    pub unsafe extern "sysv64" fn plain(x: f128::X) {
        ret();
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tiles<const N: u64>(x: f128::X) {
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
    pub unsafe extern "sysv64" fn plain(x: f16!()) {
        ret();
    }

    #[instances(1, 2)]
    pub unsafe extern "sysv64" fn tiles<const N: u64>(x: f16!()) {
        ret();
    }
}

/// A const of the crate, named as an unstable type is.
#[allow(non_upper_case_globals)]
pub const f16: usize = 2;

#[kernite_macros::kernels]
pub mod length_name {
    use super::f16;

    /// An `f16` that is the whole length of an array, a const's name: no type. The module does not hold the
    /// const, so kernite says the name is not defined.
    pub unsafe extern "sysv64" fn plain(x: [u8; f16]) {
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
pub mod crate_f16 {
    use super::types::f16;

    /// An `f16` that is a type of the crate's, which the module imports: no unstable type. The import is used.
    pub unsafe extern "sysv64" fn plain(x: f16) {
        ret();
    }
}

#[kernite_macros::kernels]
pub mod crate_f128 {
    use super::types::f128;

    /// An `f128` that is a type of the crate's, which the module imports: no unstable type. The import is used.
    pub unsafe extern "sysv64" fn plain(x: f128) {
        ret();
    }
}

#[kernite_macros::kernels]
pub mod crate_array {
    use super::types::f16;

    /// An `f16` in an array that is a type of the crate's, which the module imports. The import is used.
    pub unsafe extern "sysv64" fn plain(x: [f16; 4]) {
        ret();
    }
}

#[kernite_macros::kernels]
pub mod crate_path {
    use super::types::f16;

    /// An `f16` that is the later segment of a path, `self::f16`, to the type the module imports: a name, no
    /// type. The import is used.
    pub unsafe extern "sysv64" fn plain(x: self::f16) {
        ret();
    }
}

#[kernite_macros::kernels]
pub mod core_path {
    /// An `f16` that is the later segment of a path, `core::primitive::f16`: a name, no type.
    pub unsafe extern "sysv64" fn plain(x: core::primitive::f16) {
        ret();
    }
}

#[kernite_macros::kernels]
pub mod parameter_path {
    /// An `f16` that is the name of a function pointer's parameter, whose type is a path from the root,
    /// `::core::primitive::u8`: no type. The `:` of the name is not the first of a `::`.
    pub unsafe extern "sysv64" fn plain(x: fn(f16: ::core::primitive::u8)) {
        ret();
    }
}

/// Each kernel, called as Rust calls it: with a value of any type where the kernel takes one of a type it
/// refuses, and its result read as a `u64` where it gives one back.
pub fn on_the_cpu() -> Result<(), kernite_cpu::Error> {
    unsafe {
        wide::plain(1.0);
        wide::tiles::<1>(1.0);
        wide::together(kernite_cpu::Grid::new(1), 1.0)?;
        wide::together_tiles::<2>(kernite_cpu::Grid::new(1), 1.0)?;
        array::plain([0.0; 4]);
        array::tiles::<1>([0.0; 4]);
        array::together(kernite_cpu::Grid::new(1), [0.0; 4])?;
        array::together_tiles::<2>(kernite_cpu::Grid::new(1), [0.0; 4])?;
        tuple::plain((0.0, 1));
        tuple::tiles::<1>((0.0, 1));
        tuple::together(kernite_cpu::Grid::new(1), (0.0, 1))?;
        tuple::together_tiles::<2>(kernite_cpu::Grid::new(1), (0.0, 1))?;
        reference::plain(&1.0);
        reference::tiles::<1>(&1.0);
        reference::together(kernite_cpu::Grid::new(1), &1.0)?;
        reference::together_tiles::<2>(kernite_cpu::Grid::new(1), &1.0)?;
        never::plain(1);
        never::tiles::<1>(1);
        never::together(kernite_cpu::Grid::new(1), 1)?;
        never::together_tiles::<2>(kernite_cpu::Grid::new(1), 1)?;
        slice::plain(1);
        slice::tiles::<1>(1);
        slice::together(kernite_cpu::Grid::new(1), 1)?;
        slice::together_tiles::<2>(kernite_cpu::Grid::new(1), 1)?;
        text::plain(1);
        text::tiles::<1>(1);
        text::together(kernite_cpu::Grid::new(1), 1)?;
        text::together_tiles::<2>(kernite_cpu::Grid::new(1), 1)?;
        object::plain(1);
        object::tiles::<1>(1);
        object::together(kernite_cpu::Grid::new(1), 1)?;
        object::together_tiles::<2>(kernite_cpu::Grid::new(1), 1)?;
        several::mixed(1, 1.0, std::ptr::null_mut(), 1);
        imported::plain(Params(1));
        beside::plain((Params(1), 0.0));
        deep::plain(None);
        made::plain(1);
        not_length::plain([0; 3]);
        not_length::tiles::<1>([0; 3]);
        unequal_length::plain([0]);
        unequal_length::tiles::<1>([0]);
        block_argument::plain(Marker);
        block_argument::tiles::<2>(Marker);
        macro_argument::plain(1);
        macro_argument::tiles::<2>(1);
        parameter_name::plain(drop::<u8>);
        parameter_name::tiles::<2>(drop::<u8>);
        module_name::plain(f128::X);
        module_name::tiles::<2>(f128::X);
        no_module_f16::plain(1);
        no_module_f16::tiles::<2>(1);
        no_module_f128::plain(1);
        no_module_f128::tiles::<2>(1);
        macro_name::plain(1);
        macro_name::tiles::<2>(1);
        length_name::plain([0; 2]);
        crate_f16::plain(types::f16);
        crate_f128::plain(types::f128);
        crate_array::plain([types::f16, types::f16, types::f16, types::f16]);
        crate_path::plain(types::f16);
        core_path::plain(1);
        parameter_path::plain(drop::<u8>);
        let _: u64 = gives_wide::plain(1);
        let _: u64 = gives_wide::tiles::<2>(1);
        let _: u64 = gives_slice::plain(1);
        let _: u64 = gives_slice::tiles::<2>(1);
    }
    Ok(())
}
