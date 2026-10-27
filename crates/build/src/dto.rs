//! The data kgen-build hands on.

/// A kernel compiled, or a static, and the constants of its module it read. Rust never sees
/// those reads, so whatever writes the module out for Rust has to say they happened.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Compiled<T> {
    /// What it compiled to: its listing, its `.entry`, or the memory a static is.
    pub code: T,
    /// The constants, by name, in the module's order.
    pub consts: Vec<String>,
}
