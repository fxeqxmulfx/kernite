//! The data kgen-build and the machines hand on: plain values, whatever the machine.

/// A kernel compiled, or a static, and the constants of its module it read. Rust never sees
/// those reads, so whatever writes the module out for Rust has to say they happened.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Compiled<T> {
    /// What it compiled to: its listing, its `.entry`, or the memory a static is.
    pub code: T,
    /// The constants, by name, in the module's order.
    pub consts: Vec<String>,
}

/// A kernel's code as `naked_asm!` takes it, for Rust to call in the process. Constant `i`'s
/// address reads `{ci}` in `code`, function `i`'s `{fi}`, and static `NAME`'s `{NAME}`: the
/// operands of `naked_asm!` that name them.
#[derive(Debug, PartialEq)]
pub struct Naked {
    /// The code, a line per instruction or label.
    pub code: Vec<String>,
    /// Its listing, for people and for snapshot files.
    pub text: String,
    /// The CPU features its instructions need past the machine's own, each once, in the order
    /// of their names, as Rust's `target_feature` writes them: `avx2`.
    pub needs: Vec<String>,
    /// The constants it reads, by index.
    pub consts: Vec<Bytes>,
    /// The functions it names, by index, each by what the source calls it: `gcd`, `tile::<4>`.
    pub funcs: Vec<String>,
    /// The statics it names, each by what the source calls it: `COUNTS`.
    pub statics: Vec<String>,
}

/// A constant's bytes, and what its address is a multiple of.
#[derive(Debug, PartialEq)]
pub struct Bytes {
    /// The bytes, the lowest first.
    pub bytes: Vec<u8>,
    /// What its address is a multiple of, a power of two.
    pub align: u32,
}

/// A static of the process's memory, as Rust lays it out.
#[derive(Debug, PartialEq)]
pub struct HostStatic {
    /// The number each element is, by the language's name: `u32`.
    pub elem: String,
    /// The bytes of each element.
    pub size: u64,
    /// How many elements it has.
    pub len: u64,
    /// What its address is a multiple of, a power of two.
    pub align: u64,
    /// What it holds until something writes it, as bytes, the lowest first; zeros past their
    /// end.
    pub init: Vec<u8>,
}

/// A static of the GPU's memory: its symbol, and the number the host reads its elements as.
#[derive(Debug, PartialEq)]
pub struct DeviceStatic {
    /// The name the driver finds it by.
    pub symbol: String,
    /// The number the host reads each element as, by the language's name: `u32` for one PTX
    /// calls `b32` or `u32`, or `f16`.
    pub host: String,
}
