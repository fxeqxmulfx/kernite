//! The data kgen-build and the machines hand on: plain values, whatever the machine.

use kgen_lang::{Error, Module, Value};

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
    /// The functions a call at an address may reach, each once, by what the source calls it,
    /// `gcd` or `tile::<4>`. The code names none of them, so `funcs` does not.
    pub reach: Vec<String>,
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

/// The consts a module takes from a module of consts, `#[kernels(consts = crate::sizes)]`: its
/// first items, which its kernels read as they read its own.
pub struct Given {
    /// How many of the module's tokens are theirs.
    pub before: usize,
    /// The path of their module, as written: `crate::sizes`.
    pub from: String,
}

/// A module of kernels, its kernels compiled, for whatever writes it out for Rust: what each of
/// its items is, and what the module as a whole needs.
pub struct ModuleOut {
    /// The module, without each const given that an item of its own is named as too.
    pub module: Module,
    /// An error at each item of the module's own named as a const given is.
    pub given: Vec<Error>,
    /// Why the module is two, if it has both kinds of kernel and a static.
    pub owner: Option<Error>,
    /// What each item of `module` is, in order.
    pub items: Vec<ItemOut>,
    /// The consts something read, by their index among `module`'s items, in order, each once.
    /// Rust sees none of these reads, so it would find them unused.
    pub read: Vec<usize>,
    /// Each instance written of each function the process runs, by name, `tile::<4>`, and its
    /// listing.
    pub listings: Vec<(String, String)>,
    /// The module the GPU loads, if a static of the module's or a function of it is the GPU's.
    pub device: Option<DeviceOut>,
}

/// What an item of a module is, its kernels compiled.
#[derive(Debug)]
pub enum ItemOut {
    /// Rust, which passes through: a const of the module's own, or what kgen does not read.
    Rust,
    /// A const given, Rust's where its module is.
    Given,
    /// A static of the process's memory, as Rust lays it out, or why it is none.
    HostStatic(Result<HostStatic, Error>),
    /// A static of the GPU's memory, as the host reads it, or why it is none. The module the GPU
    /// loads has it.
    DeviceStatic(Result<DeviceStatic, Error>),
    /// A helper, and what is wrong with it whether it is called or not: its body, its
    /// `#[instances(..)]`, then its `#[cooperative]`.
    Helper(Vec<Error>),
    /// A function of an ABI kgen compiles nothing for, which says so.
    Unknown(Error),
    /// A function the process runs.
    Native(NativeOut),
    /// A function the GPU runs.
    Device(DeviceFnOut),
}

/// A function the process runs, compiled.
#[derive(Debug)]
pub struct NativeOut {
    /// Why it is not a kernel Rust may call: it is not an `unsafe fn`.
    pub not_unsafe: Option<Error>,
    /// What is wrong with its `#[instances(..)]`; then nothing of it is compiled.
    pub said: Option<Error>,
    /// Each instance that compiled, in order.
    pub instances: Vec<NativeInstance>,
    /// Whether Rust gets its code: it is an `unsafe fn`, its attributes hold, each of its
    /// instances compiled, and it reaches the statics it names.
    pub written: bool,
    /// How many instances it lists, compiled or not; none if its attributes are wrong.
    pub count: usize,
    /// The statics its code names, those the functions it names or may reach at an address
    /// name, and on, in the module's order; none unless the process owns the statics.
    pub touched: Vec<String>,
    /// Why each instance that did not compile did not, in order.
    pub errors: Vec<Error>,
}

/// An instance of a function the process runs, compiled.
#[derive(Debug)]
pub struct NativeInstance {
    /// Its compile-time arguments; none for a function that takes none.
    pub args: Vec<Value>,
    /// Its code.
    pub naked: Naked,
    /// Each function its code names, `{fi}`, by index; none unless Rust gets its code.
    pub calls: Vec<Call>,
}

/// A function a kernel's code names, `{fi}`: by its name, and its instance's index among those
/// its `#[instances(..)]` lists, if it takes compile-time arguments.
#[derive(Debug, PartialEq)]
pub struct Call {
    /// The function's name, `tile`.
    pub function: String,
    /// The instance's index in its function's `#[instances(..)]`; none for a function that
    /// takes no compile-time arguments.
    pub instance: Option<usize>,
}

/// A function the GPU runs, compiled into the module it loads.
#[derive(Debug)]
pub struct DeviceFnOut {
    /// Why it is not a kernel Rust may launch: it is not an `unsafe fn`.
    pub not_unsafe: Option<Error>,
    /// What is wrong with its `#[cooperative]`, else with its `#[instances(..)]`; then nothing
    /// of it is compiled.
    pub said: Option<Error>,
    /// Why each instance that did not compile did not, in order.
    pub errors: Vec<Error>,
    /// How Rust launches it, if it is a kernel; none for a function the kernels call.
    pub launcher: Option<Launcher>,
}

/// How Rust launches a kernel of the GPU's.
#[derive(Debug, PartialEq)]
pub enum Launcher {
    /// Never: the kernel is not an `unsafe fn`, its attributes are wrong, or one of its
    /// instances did not compile.
    Never,
    /// Each instance by its compile-time arguments, and the index of its symbol in
    /// [`DeviceOut::symbols`].
    Of(Vec<(Vec<Value>, usize)>),
}

/// The module the GPU loads.
pub struct DeviceOut {
    /// Its text; empty if it is none.
    pub text: String,
    /// Why it is none, at the module's first item that is the GPU's: a static of its memory, or
    /// a function it runs.
    pub error: Option<Error>,
    /// The symbol of each instance of each kernel Rust launches, by index.
    pub symbols: Vec<String>,
}
