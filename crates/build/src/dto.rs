//! The data kernite-build and the machines hand on: plain values, whatever the machine.

use kernite_lang::{Error, Mention, Module, Span, Value};

use crate::Assembly;

/// A kernel compiled, or a static, and where it named the items of its module. Rust never sees
/// a kernel's body run, so whatever writes the module out for Rust has to say what it named.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Compiled<T> {
    /// What it compiled to: its listing, its `.entry`, or the memory a static is.
    pub code: T,
    /// Where it named an item of its module.
    pub mentions: Vec<Mention>,
}

/// A static that does not compile: why not, and what stands for it in the code that names it.
#[derive(Debug)]
pub struct Failed<T> {
    /// Why it does not compile, where it can be mended; none in a refused module, which says nothing of
    /// its items.
    pub error: Option<Error>,
    /// The static as its type has it, holding zeros, if the language works its type out and the machine
    /// takes its name and its type, whatever it holds; else none.
    pub stub: Option<T>,
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

/// A static of the GPU's memory: its symbol, its type, and the number the host reads its elements as.
#[derive(Debug, PartialEq)]
pub struct DeviceStatic {
    /// The name the driver finds it by.
    pub symbol: String,
    /// Its type as the module declares it, with the values it names known now: `constant<[u32; 16]>`.
    pub ty: String,
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
/// its items is, and what the module as a whole needs. A module that is refused whole is not
/// compiled: each of its items is what it is otherwise, and nothing is said of an item.
pub struct ModuleOut {
    /// The module, without each const given that an item of its own is named as too.
    pub module: Module,
    /// An error at each item of the module's own named as a const given is.
    pub given: Vec<Error>,
    /// Why the module is two, if it has both kinds of kernel and a static.
    pub owner: Option<Error>,
    /// What each item of `module` is, in order.
    pub items: Vec<ItemOut>,
    /// The module the GPU loads, if a static of the module's or a function of it is the GPU's;
    /// empty, in a refused module.
    pub device: Option<DeviceOut>,
}

/// What an item of a module is, its kernels compiled.
#[derive(Debug)]
pub enum ItemOut {
    /// Rust, which passes through: a const of the module's own, or what kernite does not read.
    Rust,
    /// A const given, Rust's where its module is.
    Given,
    /// A static of the process's memory, as Rust lays it out, and where it named the module's items;
    /// or why it is none, and what stands for it. A refused module lays out none: each stands as its
    /// type has it, with no error.
    HostStatic(Result<Compiled<HostStatic>, Failed<HostStatic>>),
    /// A static of the GPU's memory, as the host reads it, and where it named the module's items; or
    /// why it is none, and what stands for it. The module the GPU loads has it. A refused module lays
    /// out none, as for the process's.
    DeviceStatic(Result<Compiled<DeviceStatic>, Failed<DeviceStatic>>),
    /// A helper, and what is wrong with it whether it is called or not: its body, its
    /// `#[instances(..)]`, then its attributes, `#[cooperative]` and the directives.
    Helper(Vec<Error>),
    /// A function of an ABI kernite compiles nothing for, which says so, unless its module is refused.
    Unknown(Option<Error>),
    /// A function the process runs; in a refused module, one that is not compiled: it has nothing
    /// to say, no instance, and is not written.
    Native(NativeOut),
    /// A function the GPU runs; in a refused module, one that is not compiled: it has nothing to
    /// say, and is never launched.
    Device(DeviceFnOut),
}

impl ItemOut {
    /// Where the item named an item of its module: a kernel's, in its body, in its
    /// `#[instances(..)]`, in its directives or in a helper it calls, and an `extern "ptx-func"`
    /// function's, in its body, whichever of the instances of either named them; a static's, in
    /// its type or what it holds. A helper has none of its own: the kernel or function that
    /// calls it named what it names. A const's definition is Rust's own, and the names in it
    /// are not the item's. Rust never sees these names, so whatever writes the item out for
    /// Rust says each at its place, where the item runs the code that has it.
    pub fn mentions(&self) -> &[Mention] {
        match self {
            ItemOut::HostStatic(Ok(s)) => &s.mentions,
            ItemOut::DeviceStatic(Ok(s)) => &s.mentions,
            ItemOut::Native(k) => &k.mentions,
            ItemOut::Device(k) => &k.mentions,
            _ => &[],
        }
    }

    /// Whether kernite says an error of the item itself: a static it cannot lay out; a helper
    /// whose body, `#[instances(..)]` or attributes are wrong; a function of an ABI it compiles
    /// nothing for; a kernel that is not an `unsafe fn`; or a kernel or an `extern "ptx-func"`
    /// function whose attributes are wrong, or of which an instance does not compile. An error of
    /// the module as a whole is none of an item's, and a module refused whole says none of its items'.
    pub fn failed(&self) -> bool {
        match self {
            ItemOut::Rust | ItemOut::Given => false,
            ItemOut::HostStatic(laid) => laid.as_ref().err().is_some_and(|failed| failed.error.is_some()),
            ItemOut::DeviceStatic(laid) => laid.as_ref().err().is_some_and(|failed| failed.error.is_some()),
            ItemOut::Helper(errors) => !errors.is_empty(),
            ItemOut::Unknown(error) => error.is_some(),
            ItemOut::Native(k) => k.not_unsafe.is_some() || k.said.is_some() || !k.errors.is_empty(),
            ItemOut::Device(k) => k.not_unsafe.is_some() || k.said.is_some() || !k.errors.is_empty(),
        }
    }
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
    /// Where the instances that compiled named an item of the module.
    pub mentions: Vec<Mention>,
    /// Whether Rust gets its code: it is an `unsafe fn`, its attributes hold, each of its
    /// instances compiled, and it reaches the statics it names.
    pub written: bool,
    /// How many instances it lists, compiled or not; none if its attributes are wrong.
    pub count: usize,
    /// Each instance Rust gets the code of, by name, `tile::<4>`, and its listing, in order; none
    /// unless `written`.
    pub listings: Vec<(String, String)>,
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
    /// What is wrong with its `#[cooperative]` or its directives, else with its
    /// `#[instances(..)]`; then nothing of it is compiled.
    pub said: Option<Error>,
    /// Why each instance that did not compile did not, in order.
    pub errors: Vec<Error>,
    /// Where the instances that compiled named an item of the module.
    pub mentions: Vec<Mention>,
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

/// The module the GPU loads, put together from what the program keeps of the module's items.
pub struct DeviceOut {
    /// The module as the machine says it, for the program to put together.
    pub assembly: Assembly,
    /// Where the module's errors go: at the module's first item that is the GPU's, a static of
    /// its memory or a function it runs.
    pub at: Span,
    /// Each instance of each kernel Rust launches, by index: the code it is, by
    /// [`Kept::Code`](crate::Kept::Code), and its symbol.
    pub symbols: Vec<(usize, String)>,
    /// The item of the module that each of the assembly's code is of, by
    /// [`Kept::Code`](crate::Kept::Code): an instance of a function the GPU runs, in the order of
    /// the items.
    pub code_items: Vec<usize>,
    /// The item of the module that each of the assembly's statics is of, by
    /// [`Kept::Static`](crate::Kept::Static): a static of the GPU's memory that was laid out, in
    /// the order of the items.
    pub static_items: Vec<usize>,
}
