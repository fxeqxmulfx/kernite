//! `#[kernels]` on a module: each function in it that names an ABI is a kernel, or a function
//! kernels call, written in the language of `kernite-lang` and compiled for the machine the ABI
//! belongs to, in place.
//!
//! ```ignore
//! #[kernite_macros::kernels]
//! mod cpu {
//!     /// The sum of the eight floats at `xs`.
//!     pub unsafe extern "sysv64" fn sum8(xs: *const f32) -> f32 {
//!         let v: ymm;
//!         vmovups(v, [xs]);
//!         ret_sum(v);
//!     }
//!
//!     fn ret_sum(v: ymm) {
//!         let high: ymm;
//!         vextractf128(high.xmm, v, 1);
//!         ..
//!     }
//! }
//! ```
//!
//! - An `unsafe extern "sysv64" fn` is compiled for x86-64 and becomes an `#[unsafe(naked)]`
//!   function of the signature it declares, its body the kernel's `naked_asm!`; constants it
//!   reads are statics beside it. `LISTINGS` holds each one's listing, by name, for snapshots.
//!   The module's kernels call each other, themselves among them, as `call(r, gcd, (a, b))`
//!   does, or at the address `lea(fp, [gcd])` takes; all but a `#[cooperative]` one, which
//!   its launcher runs. One that never returns, `-> !`, ends each way out in `ud2()` or in a
//!   call of one that does not.
//! - An `unsafe extern "ptx-kernel" fn` is compiled to PTX, for NVIDIA GPUs. The module's are
//!   one PTX module, `PTX`, and `Kernels::load(&cuda)` has the driver compile it for the GPU
//!   it finds; then each kernel is launched by the method of its name, which takes a
//!   `kernite_cuda::Grid` and the kernel's parameters, a pointer as the `kernite_cuda::Ptr` a
//!   buffer on the GPU gives, and an `f16` or a `bf16` as kernite-cuda's, Rust having neither.
//!   A crate with a module of GPU kernels needs a dependency named `kernite-cuda` for these, not
//!   renamed in `Cargo.toml`, and the error is at the module when it has none and `cfg` keeps one
//!   of its GPU items: a feature can bring `kernite-cuda` and keep them, and the crate builds
//!   without it.
//! - An `extern "ptx-func" fn` is a `.func` in `PTX`, which the module's GPU kernels and
//!   functions call, itself among them, as `call(r, gcd, (a, b))` does. It takes and
//!   returns PTX's types, `u32` or `f16x2`, or never returns, `-> !`, and leaves no function
//!   behind for Rust, which does not call it.
//! - A `static` is memory of the module's, which each of its kernels names as it names memory
//!   it declares: `static COUNTS: global<[u32; 16]>;` in global memory, or
//!   `static WEIGHTS: constant<[f32; 4]> = [1.0, 0.5, 0.25, 2.0];` in constant memory, which
//!   the kernels only read. It holds what it says until something writes it, or zeros, and
//!   keeps what is written from one call or launch to the next. One that does not compile is an
//!   error where the trouble is, and a static stands for it in the code that uses it. If only
//!   what it holds is wrong, it stands as a static of its type, holding zeros, and the code that
//!   uses it is read as it is for one that compiles: it need not say the type of the elements, and
//!   a use that the type does not fit is Rust's error, beside kernite's. If its type or its name
//!   is wrong, it stands as one whose elements are of any type, and the code that uses it says
//!   the type of the elements, or Rust asks for it, beside kernite's error. kernite says the first
//!   error it finds, and reads what a static holds before the machine reads its name and its type.
//!   So its error can be at what the static holds while the machine refuses the type or the name
//!   too, as for `static EMPTY: global<[u32; 0]> = [nope];`: that static stands as one whose
//!   elements are of any type as well, and Rust asks for the type of a use that says none, beside
//!   kernite's error. It is its kernels' memory:
//!   - In a module of GPU kernels it is on the GPU, declared once in `PTX`. `Kernels` has a
//!     method of its name that gives it there, a `kernite_cuda::Global` of the type of Rust's its
//!     elements are read as, `u32` for a `b32`, for the host to read and write:
//!     `kernels.COUNTS().read()?`. A module of statics alone is the GPU's. The statics together
//!     fit the module: it has 64 KB of constant memory and less than 4 GB of global memory, and
//!     the error is at the type of a static that does not fit beside the statics before it that fit.
//!     The room is that of the statics `cfg` keeps, counted as Rust leaves them: the macro writes
//!     a struct with a field for each static, under the static's `cfg`, and a derive of
//!     `kernite-macros` counts the fields that are there. A crate with a static in such a module
//!     needs a dependency named `kernite-macros` as well, not renamed in `Cargo.toml`, and the
//!     error is at the module when it has none and `cfg` keeps a static.
//!   - In a module of x86 kernels it is in the process's memory, which the kernels reach at a
//!     displacement from `rip`, `[COUNTS + 8]`: a `static` of Rust's, a `kernite_cpu::Global` of
//!     its elements, for the host to read and write: `cpu::COUNTS.read()`. A crate with a static
//!     in such a module needs a dependency named `kernite-cpu` for it, not renamed in
//!     `Cargo.toml`, and the error is at the static when it has none.
//!   - A module of both kinds of kernel and a static is an error: it is two modules, one of
//!     each kind of kernel, with the statics each names.
//! - A function without an ABI is a helper, inlined where a kernel calls it, and leaves no
//!   function behind. Constants and every other item but a `static` stay as written, for Rust
//!   to use as well.
//!   A constant a kernel reads counts as used, as it would if Rust had seen the read, under the
//!   `cfg` of each item that reads it, a helper's under that of the kernel that calls it and that
//!   of the helper: only one that nothing reads, or that only what `cfg` leaves out reads, is
//!   `dead_code`. Rust has no `bf16`, and its `f16` is unstable, so a constant of either goes in
//!   the kernel that reads it, where `const` is the language's alone.
//! - A kernel `cfg` keeps names only what `cfg` keeps. What it names is written for Rust at the
//!   name, for Rust to say what it says of any name it cannot find, at the name: a constant, a
//!   static of the process's, an x86 kernel that takes no compile-time arguments. Of the rest, a
//!   helper, an instance of an x86 kernel, an `extern "ptx-func"` function and a static of the
//!   GPU's, which Rust has no item for, it is an error of kernite's at the name, under
//!   `cfg(all(<the kernel's>, not(<the item's>)))`. Both are under the kernel's `cfg`, and, for a
//!   name in a helper's body, the helper's and that of a place that calls the helper. The naked
//!   function of an x86 kernel names the same items, and where `cfg` leaves one out it names
//!   `const 0`, for Rust to say a name once, at the name.
//!
//! ```ignore
//! #[kernite_macros::kernels]
//! mod gpu {
//!     /// `x[i] *= a` for each `i < n`.
//!     pub unsafe extern "ptx-kernel" fn scale(x: *mut f32, a: f32, n: u32) {
//!         ..
//!     }
//! }
//!
//! let cuda = kernite_cuda::Cuda::new()?;
//! let xs = cuda.upload(&[1.0, 2.0, 3.0])?;
//! let kernels = gpu::Kernels::load(&cuda)?;
//! unsafe { kernels.scale(kernite_cuda::Grid::new(1, 32), xs.ptr(), 2.0, 3)? };
//! ```
//!
//! A kernel with const generics, integers or `bool`s, is compiled for each instance its
//! `#[instances(..)]` lists, each a kernel of its own, of values known now, the module's
//! constants among them: `#[instances(256, 1024)]` on `fn block_sums<const N: u32>`, or
//! `#[instances((3, true), (-2, false))]` for two. On x86 the kernel is a generic function,
//! `cpu::dot::<4>(x, y, n)`, and `LISTINGS` holds each instance's listing by its name,
//! `dot::<4>`; on a GPU its launcher is generic, `kernels.block_sums::<1024>(grid, x, sums)`.
//! An instance it does not list is a compile error where Rust compiles the call, in
//! `cargo build`; `cargo check` evaluates no generic function's constants, and misses it.
//!
//! A kernel's parameters keep the names they are written with beside those the Rust the
//! macro writes binds for itself: `grid`, the launcher's, and `threads` and `shared`, those
//! of a cooperative GPU kernel's `_blocks` method. A compile-time argument may not be named
//! as one of them, for Rust cannot keep a const apart from a parameter of its name: that is
//! an error at its name.
//!
//! A parameter named `_` is given and not read, in a kernel of every kind. Rust cannot pass `_` on,
//! so the macros name it, with a name of their own that no parameter or compile-time argument of
//! the kernel has. The function Rust calls the kernel by takes it under that name, and its docs
//! show it.
//!
//! A type in a kernel's parameters or return means what it means in any function of the module,
//! `super::W` and `self::W` among them, and it may name the kernel's compile-time arguments, as in
//! any generic function: `*mut [u8; N]` in `fn arr<const N: usize>`. An error in one is Rust's, a
//! pointer's among them, said once, at the type, in a kernel of every kind but the case below that
//! has two, and so is a lint, at the level the kernel's `allow`, `warn`, `deny`, `forbid` or
//! `expect` gives it. So is a pointer to a type with no size, `*const CStr` say, which has two
//! words where a kernel's has one: the macros ask Rust of each pointer a kernel takes or gives
//! back, and it says that the size cannot be known, at the type, whatever the type is called and
//! holds. So is a pointer to a type Rust refuses by itself, `*const NeedsCopy<String>` where
//! `struct NeedsCopy<T: Copy>`, or `*const [CStr; 3]`: the macros ask Rust whether a pointer has a
//! size by a coercion and never by a call, which would say the error of the type again. And so is a
//! number's name that the module gives another type, `pub type u64 = u32;` and a `use` of it:
//! kernite reads `u64` as 64 bits where Rust would pass a `u32`, so the macros ask Rust of each
//! number a kernel takes or gives back whether it is that number, and it says that it is not, at
//! the type, once. A type with no size, `pub type u8 = [u8];`, has Rust say as well that its size
//! cannot be known, at the type, and nothing at the module's attribute, where nothing could be
//! mended: the Rust the macros write for the kernel itself takes the number and not the name. One
//! case has two errors: a kernel of compile-time arguments casts back a pointer it gives, so for a
//! type that holds the part Rust refuses, `-> *const (u8, [CStr; 3])`, Rust says the part in the
//! signature, and the whole type at the cast.
//! A generic is a value: `<T>` is an error at the `T`, and so is `impl` in a parameter's type or
//! the result's, wherever in it, `x: impl Copy`, `Option<impl Copy>` or `-> *mut impl Copy`, at the
//! `impl`: in a parameter it makes the function generic over a type, and in the result it hides
//! one. kernite says both in the same words.
//! The Rust the macros write names the numbers and `str` by their paths, `::core::primitive::u64`,
//! so a module may give any of those names another type and build, so long as none of its kernels
//! takes or gives that name as a number.
//! A number written by its path, `core::primitive::u64` or `::std::primitive::u64`, is that number,
//! and Rust is asked of the path as of the name: it goes through `core` or `std`, so a module that
//! gives either another meaning, `use crate::fake as core;`, has Rust say that the type is not the
//! number, at the type, once.
//! Any other type a kernel takes or gives back is kernite's error alone, at the type: `f128`,
//! `[f16; 4]`, `!` as a parameter, `[u64]`, or a type of the crate's named `f16` or `f128`. The
//! macros write a name Rust cannot find for it, with the type as its argument, so that Rust says
//! neither that the type is unstable or has no size, nor anything at a call of the kernel. An
//! `f16` or an `f128` in the argument is written `self::f16` or `self::f128`: a type of the module
//! so named is found there, and Rust does not call its `use` unused. What Rust says of any
//! function's signature it still says: a lifetime it cannot find, a `_` in a result, and a module
//! where a type is written, as `core::f16` is in `x: core::f16`, or an `f16` where the module
//! imports a module so named, `use crate::f16;` with `x: f16`: Rust expected a type and found a
//! module. An `f16` or an `f128` that the macros do not see as a type keeps Rust's word that it
//! is unstable: one that a macro writes, `half!()`, or one in an expression of the type, as in an
//! array's length, `[u8; size_of::<f16>()]`.
//! A result that a cooperative x86 kernel or a GPU kernel cannot have is kernite's error at the
//! type, and Rust sees the type as it sees a plain kernel's: it says what it says of any function's
//! result, once, at the type, in the same words. So a pointer whose type has no size,
//! `-> *const CStr`, a `&` with no lifetime to take, a lifetime not declared and a `_` are Rust's
//! errors there as they are in a plain kernel's, and an import that only the result names is used.
//! A GPU kernel's launcher is a method, so a `&` in its result takes the lifetime of `self`, and
//! Rust has none to ask for. A pointer of two words to a type Rust refuses by itself,
//! `-> *mut dyn Copy`, is kernite's error alone, though Rust says its own of a plain kernel's
//! result. A result of `()` is no result, and no error.
//! A name written raw is the name Rust reads, in a type as in a value's name: `r#u64` is the number
//! `u64`, `r#f16` the half a GPU kernel's launcher takes as kernite-cuda has it, and `r#f128` a type
//! kernite refuses, as the plain names are. A keyword written raw is a name too: a pointer to
//! `r#dyn::Foo` has one word, as Rust has it, where a pointer to `dyn Foo` has two. An error says the
//! keyword as it is written, `r#dyn`. An attribute kernite reads, written raw, is the attribute:
//! `#[r#instances(1, 2)]`, `#[r#cooperative]`, a GPU kernel's `#[r#maxnreg(32)]`, and the module's
//! `#[kernels(r#consts = ..)]`. Rust is given none of them, and an error of one is kernite's alone.
//!
//! A helper is compiled where a kernel calls it, and an `extern "ptx-func"` function is a `.func`
//! of the GPU's module, so the macros write no function for either. Where kernite finds any error
//! in the module, they write the types of every helper and function of the module, one that no
//! kernel calls too: each compile-time argument's, each parameter's and the result's, as they write
//! the type of a kernel that does not compile, `let _: ..;` in a block of the module, under the
//! `cfg` of the function and where the error is said. Every error is an item of the module, as Rust
//! is silent about a name it cannot find, as `gpr` and `local<[u32; 4], 16>` are, only in a module
//! that has one. So the types are shown nowhere else. In such a module Rust says nothing of any
//! other name it cannot find either, a const that `cfg` leaves out say, until the errors are
//! mended. Rust finds the names in the types, so an import that only they name is used, and it says
//! of them only that a lifetime is not declared, that a module is where a type is written, and what
//! a macro writes; it says the same of a helper that no kernel calls. The room of the statics is
//! the one error that Rust decides, by `cfg`. `__Room` is derived on a struct with a field for each
//! static and each helper, under the `cfg` of its item. It counts the room of the statics Rust
//! keeps, and writes the error, with the types of the helpers Rust keeps beside it. A module that
//! builds is shown none of them, so an import that only a helper names is unused to Rust.
//!
//! The Rust the macro writes needs no `allow` of a lint of Rust's, so a crate may `forbid` one,
//! `dead_code` say, and build its kernels: Rust lints the names and the code a kernel is
//! written with, and none that the macro adds. Clippy's `too_many_arguments` is the one lint a
//! crate of kernels may not forbid: Clippy says it of a launcher of more than seven parameters,
//! whoever wrote it, and each launcher allows it.
//!
//! A module of kernels takes the consts it shares with others from a module `#[consts]` is
//! on, which it names by its path, as Rust reads one where the module is:
//! `#[kernels(consts = crate::model::sizes)]`, or `consts = sizes` for one beside it. Its
//! kernels read them as the module's own, and Rust sees them there by a `use` the macro
//! writes; a const, function or static of the module's own of one of their names is an error.
//! A module of consts holds `pub` consts alone, of what kernels read: numbers, floats,
//! `bool`s, arrays of them, and expressions of these. It is of the same crate, for the macro
//! that hands its consts on is the crate's alone. That macro calls `kernite-macros` back by
//! its name, so a crate with a module of kernels given consts needs a dependency named
//! `kernite-macros`, not renamed in `Cargo.toml`, and the error is at the module of kernels
//! when it has none and `cfg` keeps it.
//!
//! ```
//! #[kernite_macros::consts]
//! pub mod sizes {
//!     /// How far a value moves.
//!     pub const STEP: u64 = 3;
//! }
//!
//! #[kernite_macros::kernels(consts = crate::sizes)]
//! pub mod cpu {
//!     /// `x` a step on.
//!     pub unsafe extern "sysv64" fn step(x: u64) -> u64 {
//!         add(x, STEP);
//!         ret(x);
//!     }
//!
//!     /// Two steps, as Rust counts them.
//!     pub const TWO: u64 = 2 * STEP;
//! }
//!
//! fn main() {
//!     assert_eq!(unsafe { cpu::step(1) }, 4);
//!     assert_eq!(cpu::TWO, 6);
//! }
//! ```
//!
//! ```compile_fail
//! #[kernite_macros::consts]
//! pub mod sizes {
//!     pub const STEP: u64 = 3;
//! }
//!
//! #[kernite_macros::kernels(consts = crate::sizes)]
//! pub mod cpu {
//!     // `STEP` names one of the consts the module takes from `crate::sizes` as well.
//!     pub const STEP: u64 = 4;
//! }
//!
//! fn main() {}
//! ```
//!
//! A GPU kernel whose blocks wait for each other, at `grid.sync()`, is `#[cooperative]`: its
//! launcher launches it cooperatively, every block of the grid at once, so the grid has at
//! most as many blocks as the GPU runs at once, which the method of its name and `_blocks`
//! says, of so many threads a block and so many bytes of the launch's shared memory:
//! `kernels.normalize_blocks(256, 0)?`. A grid of more is an error, and so is a GPU before
//! Pascal's, `sm_60`, which launches none cooperatively.
//!
//! A GPU kernel says how it is launched with `#[maxntid(16, 16)]`, the most threads a block has,
//! `#[reqntid(8, 8, 4)]`, the threads it has, `#[minnctapersm(2)]`, the blocks an SM should hold at
//! once, and `#[maxnreg(32)]`, the most registers a thread may use: whole numbers known now, the
//! module's consts among them, and the compile-time arguments of an instance. They are PTX's
//! directives of the same names, which `ptxas` holds the kernel's code to, and they are not Rust's:
//! the kernel's launcher is the same, and a launch over a block the directives refuse is an error
//! that names the directive and the threads of the block.
//!
//! An x86 kernel whose threads wait for each other is `#[cooperative]` too, and runs on the
//! threads of a grid, each given its place there, `grid.thread` of `grid.threads`: it becomes
//! the function of its name that takes a `kernite_cpu::Grid` and the kernel's parameters, runs
//! the kernel on every thread at once, and returns when all have ended:
//! `cpu::normalize(kernite_cpu::Grid::new(4), x.as_mut_ptr(), sums.as_mut_ptr(), n)?`. A thread
//! spins at `grid.sync()` until the last comes, so a grid has at most as many threads as the
//! machine runs at once, `kernite_cpu::most()`, and more is an error. A crate with such a kernel
//! needs a dependency named `kernite-cpu` for it, not renamed in `Cargo.toml`, and the error is at
//! the kernel when it has none.
//!
//! A GPU kernel is named in PTX as C++ names a function in namespaces, those of the
//! workspace, the crate and its modules, mangled under Itanium's ABI:
//! `_ZN7kernite6my_app3gpu5scaleE`, which `cu++filt`, Nsight and `cuda-gdb` show as
//! `kernite::my_app::gpu::scale`, an instance as a template's,
//! `kernite::my_app::gpu::block_sums<(unsigned int)1024>`, and a static as a variable of the
//! namespaces, `kernite::my_app::gpu::COUNTS`. The workspace's manifest is the one Cargo
//! takes for the package. kernite reads a relative `CARGO_HOME` from the directory rustc runs
//! in, and Cargo from the one it was run in, which may be another. The workspace is named as
//! the directory of its manifest is, or as the manifest names it, which a clone keeps
//! wherever it is. `name = ""` leaves the workspace out of the names. A `name` that is not a
//! string is an error at the module. The error names the manifest. A package with no
//! workspace's manifest is one of its own, named as the package is.
//!
//! ```toml
//! [workspace.metadata.kernite]
//! name = "kernite"
//! ```
//!
//! A kernel is `unsafe`: it trusts the pointers it is given, and that nothing else touches the
//! memory of the module's it names while it runs, as an x86 kernel's safety contract says, by
//! name, of the statics it and the functions it calls name, whether it calls them by name or
//! at an address. One that does not compile is a `compile_error!` where the trouble is, beside
//! a stub of its signature or its launcher, so that its callers are not errors as well. Both
//! are under the kernel's `cfg`, as a helper's errors and a static's are under theirs, and so
//! are the function of each of its instances and the module of its constants: what `cfg` leaves
//! out says nothing. A module of x86 kernels has `LISTINGS` whether its kernels compile or not,
//! with the listing of each that compiles and `cfg` keeps, and of no other; a module of GPU
//! kernels alone has none.
//!
//! kernite refuses a module of kernels whole, at its attribute, for an argument of `#[kernels]` that is not
//! `consts = path`, such as `consts = 5`, and for consts given in a crate with no dependency named `kernite-macros`. It
//! says so once, there, and the module stays. Each of its kernels is a stub, as one that does not compile is. Each of
//! its statics is a static of its type, holding zeros, if kernite works its type out and takes its name and its type,
//! and one whose elements are of any type if not. So the code that names the module's items, wherever it is, is not an
//! error as well, but for two kinds of use, beside the refusal. A use that the type of the first kind of static does
//! not fit is Rust's error, as it is for a static that compiles. A use of the second kind that says no type for its
//! elements has Rust ask for it. The module is given no consts, so a type that reads one is of the second kind. kernite
//! checks none of the items, so it says nothing it would find in them, such as a kernel that is not `unsafe`, an ABI it
//! does not know, what a static holds, or what compiling a kernel finds. It compiles nothing of the module, so a
//! workspace `name` that is not a string is no error at it. No room is counted for the statics, so a crate with no
//! dependency named `kernite-macros` is not told it needs one for them. What the macro finds in writing the module for
//! Rust is said, as for any module: the dependency the crate needs, `kernite-cuda` for a GPU item that `cfg` keeps and
//! `kernite-cpu` for an x86 static or cooperative kernel; a compile-time argument named as a parameter of a launcher or
//! of a `_blocks` method, `grid`, `threads` or `shared`; and a GPU kernel or a static named as a cooperative kernel's
//! `_blocks` method.
//!
//! A module kernite cannot read is refused whole as well, and stays with every item kernite can read. A `const` or a
//! `static` it cannot read ends at its `;`, for its value may hold a `{..}`, as in `= { 1 };`. Any other item ends at
//! its first `;`, or at its first `{..}` outside `<..>`: a function at its body. The items after it are read as they
//! are. If items share a name, it knows one of them: a const before a function, a function before a static, else the
//! first. If it cannot read an item and reads no function that names an ABI, it knows no static, for the kernels say
//! whose a static is, the process's or the GPU's. If brackets nest too deep, it knows no item. Its items are as in a
//! module refused at its attribute, with two differences: the module is given the consts it takes, and kernite says
//! nothing of an item of its own that has the name of one. The error is said once, where it can be mended. If the
//! module's attribute is refused too, that refusal is said instead. So Rust says nothing of the code that names an item
//! kernite knows, by a path, by a `use` or by a glob, beside the error. Of a name kernite does not know, as that of an
//! item it cannot read, Rust says nothing of a path or a `use`, for the error is in the module. It cannot find the name
//! by a glob or as a method of `Kernels`, as it cannot find a name that no module has. A glob that finds none of what
//! it names is unused, and Rust warns of it.
//!
//! A path to a module of consts that Rust cannot find, as in `consts = crate::model::sizs`, is told once, by Rust, at
//! the name it cannot find. So is a path to a module with no `#[kernite_macros::consts]` on it, at the module's name:
//! Rust says the module has no `__kernite_consts`, which that attribute writes. Rust's error is an unresolved import,
//! of the path as the macro writes it, from a module of its own in the module of kernels. A path from `self`, `super`
//! or a name is said from there: `nope` is `super::super::nope`. A name after the first is said in the module before
//! it, which the macro names `__`, that module's name, `_` and a number: `crate::hidden::nope` is `__hidden_3::nope`.
//! The macro cannot tell that the path fails, for Rust finds the path after the macro runs. The module of kernels
//! stays, with nothing in it. Rust says nothing of the code that names its items by a path, wherever it is. A glob of
//! the module brings in nothing, and Rust warns of that alone. A `use` of an item by its name, `use crate::cpu::step;`,
//! is Rust's error as well, for the module has no `step` until the path is mended.
//!
//! An argument of `#[consts]`, as in `#[consts(x)]`, is told where it is, and changes nothing else. The
//! module is the one it is without the argument: its other errors are said, and the code that names its
//! consts, or a module of kernels that takes them, is not an error as well.

extern crate proc_macro;

mod cond;
mod expand;
mod gpu;
mod room;
mod rust;
mod workspace;
mod x86;

use proc_macro::{Delimiter, Group, Span, TokenStream, TokenTree};

use kernite::ConstItem;
use kernite::lang;

use crate::expand::expand;
use crate::rust::{
    MACROS, Site, attrs_len, braced, depends_on, error, flatten, for_rust, inner_attribute, is_ident, is_path,
    is_punct, lex, module, named, needs_dependency, parse, path_uses, site,
};

/// See the crate documentation.
#[proc_macro_attribute]
pub fn kernels(attr: TokenStream, item: TokenStream) -> TokenStream {
    let attr: Vec<TokenTree> = attr.into_iter().collect();
    let tokens: Vec<TokenTree> = item.into_iter().collect();
    let Some((braces, head)) = braced(&tokens) else {
        let span = tokens.first().map_or_else(Span::call_site, TokenTree::span);
        return error(
            span,
            "`#[kernels]` goes on a module written in braces, `mod name { .. }`, where the kernels are",
        );
    };
    if attr.is_empty() {
        return module(head, braces, expand(None, None, braces.stream()));
    }
    // The module kernite refuses at its attribute is still there, with the error in it, for the code that
    // names it to find: `expand` writes it as it writes any module, but kernite checks none of its items,
    // so it says nothing it would find in them, and it counts no room, so the crate is not told it needs
    // `kernite-macros` for a static. What `expand` finds in writing the module for Rust, such as the
    // dependency on `kernite-cuda` or `kernite-cpu` it needs, is said, as for any module.
    let refused = |why: TokenStream| module(head, braces, expand(None, Some(why), braces.stream()));
    let path = match &attr[..] {
        [c, eq, path @ ..] if named(c, "consts") && is_punct(eq, '=') && is_path(path) => path,
        _ => {
            let message = "`#[kernels]` takes the module of consts its kernels read, as `consts = crate::model::sizes`, \
                           a module `#[kernite_macros::consts]` is on; or nothing";
            return refused(error(attr[0].span(), message));
        }
    };
    // The macro the consts' module has calls `__kernels!` back by the name `kernite-macros`, which a
    // crate that has the dependency by another name has not: that is said here, at the module, in place
    // of the call. Rust takes out a module that `cfg` leaves out before this runs, so none is told.
    if !depends_on(MACROS) {
        return refused(error(site(), &needs_dependency("a module of kernels given consts", MACROS)));
    }
    // The module is written here, with the call in it, and not the call in its place: Rust finds the path only
    // after this runs, and where it cannot, the module is still there, with nothing in it, for the code that
    // names it. Its own attributes, `#![..]`, three tokens each, stay first, where Rust has them. The path is
    // found in a module of its own, `__consts`, which the docs hide, written at the path's last name, for an
    // error of Rust's in it to point there.
    let body = flatten(braces.stream());
    let attrs = body.as_chunks::<3>().0.iter().take_while(|attr| inner_attribute(*attr)).count() * 3;
    let mut inner: TokenStream = body[..attrs].iter().cloned().collect();
    let site = Site::at(path.last().map_or_else(Span::call_site, |t| t.span().resolved_at(Span::call_site())));
    inner.extend(parse("#[doc(hidden)] mod __consts"));
    inner.extend([TokenTree::Group(Group::new(Delimiter::Brace, path_uses(path)))]);
    // The macro the consts' module has calls `__kernels!` back with the consts, their module's
    // path, and what is in this module.
    inner.extend(parse("__consts::__kernite_consts!"));
    let given = [Group::new(Delimiter::Parenthesis, path.iter().cloned().collect()), braces.clone()];
    let call = Group::new(Delimiter::Brace, given.into_iter().map(TokenTree::Group).collect());
    inner.extend([TokenTree::Group(call)]);
    drop(site);
    module(head, braces, inner)
}

/// See the crate documentation.
#[proc_macro_attribute]
pub fn consts(attr: TokenStream, item: TokenStream) -> TokenStream {
    let tokens: Vec<TokenTree> = item.into_iter().collect();
    let Some((braces, head)) = braced(&tokens) else {
        let span = tokens.first().map_or_else(Span::call_site, TokenTree::span);
        return error(
            span,
            "`#[consts]` goes on a module written in braces, `mod name { .. }`, of the consts kernels read",
        );
    };
    let top = flatten(braces.stream());
    let (trees, spans) = lex(&top);
    let top = for_rust(top);
    let at = |s: lang::Span| spans.get(s.0 as usize).copied().unwrap_or_else(Span::call_site);
    // The module as it is, for Rust; and the consts kernite reads, for the kernels. The argument is told where
    // it is, and changes nothing else: the module is the one it would be without it, for the modules of kernels
    // that take its consts to read them.
    let mut inner: TokenStream = top.iter().cloned().collect();
    inner.extend(attr.into_iter().next().map(|t| error(t.span(), "`#[consts]` takes no arguments")));
    let mut given = TokenStream::new();
    match trees.and_then(|trees| lang::module(&trees, 0)) {
        Err(e) => inner.extend(error(at(e.span), &e.message)),
        Ok(module) => {
            for (item, verdict) in module.items.iter().zip(kernite::consts(&module)) {
                let tokens = &top[item.tokens.clone()];
                let named = &tokens[attrs_len(tokens)..];
                match verdict {
                    ConstItem::Given | ConstItem::Wrong(_) if !named.first().is_some_and(|t| is_ident(t, "pub")) => {
                        let message = "a const of a module of consts is `pub`, for the modules of kernels that take \
                                       it to see";
                        inner.extend(error(named[0].span(), message));
                    }
                    ConstItem::Given => given.extend(tokens.iter().cloned()),
                    ConstItem::Wrong(e) => inner.extend(error(at(e.span), &e.message)),
                    // The module's own attribute, `#![..]`, is Rust's.
                    ConstItem::NotAConst if inner_attribute(tokens) => {}
                    ConstItem::NotAConst => {
                        let message = "a module of consts holds the consts kernels read, as `pub const E: usize = \
                                       16;`, and nothing else";
                        inner.extend(error(named.first().unwrap_or(&tokens[0]).span(), message));
                    }
                }
            }
        }
    }
    // The macro `#[kernels(consts = ..)]` calls, by the module's path, which calls `__kernels!`
    // back with the consts that compile first. `#[kernels(consts = ..)]` has seen that the crate
    // has `kernite-macros`, by that name, for the call to name it.
    let mut call = parse("::kernite_macros::__kernels!");
    let mut args: TokenStream = [TokenTree::Group(Group::new(Delimiter::Bracket, given))].into_iter().collect();
    args.extend(parse("$($module)*"));
    call.extend([TokenTree::Group(Group::new(Delimiter::Brace, args))]);
    let mut rule = parse("($($module:tt)*) =>");
    rule.extend([TokenTree::Group(Group::new(Delimiter::Brace, call))]);
    rule.extend(parse(";"));
    inner.extend(parse("#[doc(hidden)] macro_rules! __kernite_consts"));
    inner.extend([TokenTree::Group(Group::new(Delimiter::Brace, rule))]);
    inner.extend(parse("#[doc(hidden)] pub(crate) use __kernite_consts;"));
    // A module with nothing in it, which `#[kernels(consts = ..)]` imports a glob of: a module with no `#[consts]` on
    // it has none, and the glob is the failure Rust says nothing more after.
    inner.extend(parse("#[doc(hidden)] pub mod __kernite {}"));
    module(head, braces, inner)
}

/// What `#[kernels(consts = path)]` on a module writes in it once the macro of the module of consts
/// at `path` has handed it their tokens: `[consts] (path) { what is in the module }`.
#[doc(hidden)]
#[proc_macro]
pub fn __kernels(input: TokenStream) -> TokenStream {
    let tokens: Vec<TokenTree> = input.into_iter().collect();
    let [TokenTree::Group(consts), TokenTree::Group(path), TokenTree::Group(braces)] = &tokens[..] else {
        return error(Span::call_site(), "`__kernels!` is how `#[kernels(consts = ..)]` is given its consts");
    };
    // What the macro writes is of the hygiene of the group `#[kernels]` made of the path, which is
    // the attribute's: it resolves where the module is, not inside the consts' macro, and Rust,
    // which takes it for the macro's, lints none of it. It is where the path begins, for an error
    // to point at.
    let at = path.span();
    let path: Vec<TokenTree> = path.stream().into_iter().collect();
    let site = Site::at(path.first().map_or(at, |t| t.span().resolved_at(at)));
    let inner = expand(Some((consts.stream(), &path)), None, braces.stream());
    drop(site);
    inner
}

/// What Rust counts of a module's GPU statics once `cfg` has left out those it leaves out: their
/// room. `#[kernels]` writes a struct with a field for each, under its `cfg`, for this to be derived
/// on, and one for each helper, to show Rust the types of its signature where there is an error.
#[doc(hidden)]
#[proc_macro_derive(__Room, attributes(kernite_room, kernite_shown))]
pub fn __room(input: TokenStream) -> TokenStream {
    room::derive(input)
}
