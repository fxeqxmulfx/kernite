# kernite

Kernels written where they are called, in Rust's syntax read as instructions over variables: compiled for x86-64
to assembly that changes only when they do, and for NVIDIA GPUs to PTX.

## A kernel

`#[kernite_macros::kernels]` on a module compiles each function in it that names an ABI. Its body is instructions,
called by name, over variables declared with a type and no value; kernite allocates the registers.

```rust
#[kernite_macros::kernels]
mod kernels {
    /// The sum of the eight floats at `xs`.
    pub extern "sysv64" fn sum8(xs: &[f32; 8]) -> f32 {
        let v: ymm;
        vmovups(v, [xs]);
        let high: ymm;
        vextractf128(high.xmm, v, 1);
        let sum: ymm;
        vaddps(sum.xmm, v.xmm, high.xmm);
        let pair: ymm;
        vmovhlps(pair.xmm, sum.xmm, sum.xmm);
        vaddps(sum.xmm, sum.xmm, pair.xmm);
        vmovshdup(pair.xmm, sum.xmm);
        vaddss(sum.xmm, sum.xmm, pair.xmm);
        ret(sum.xmm);
    }
}

/// `sum8` needs AVX, so a function that has it calls it with no `unsafe`.
#[target_feature(enable = "avx")]
fn sum(xs: &[f32; 8]) -> f32 {
    kernels::sum8(xs)
}
```

`sum8` is a function of the signature it declares, and Rust calls it with no `unsafe`: kernite proves it safe, as "A
safe kernel" below says, and a kernel it cannot prove safe is an `unsafe fn`. Its body is the kernel in Intel syntax,
in an `#[unsafe(naked)]` function, and `kernels::LISTINGS` holds each kernel's listing for snapshots. Everything that is
not an instruction happens while the kernel compiles: `for` unrolls, `if` on a bool chooses, a function without an ABI
is a helper inlined where it is called, and `<const N: usize>` makes one generic. Each parameter and generic of a
function has a name of its own, but `_`. A parameter named `_` is given and not read, as in Rust, in a kernel of every
kind: the next parameter is read where the ABI puts it. Rust cannot pass `_` on, so the macros name it for Rust, and the
function Rust calls the kernel by takes it under that name. A generic kernel is compiled for each value its
`#[instances(1, 4)]` lists, each a kernel of its own, which Rust calls as it calls a generic function,
`kernels::dot::<4>(x, y, n)`; a value it does not list fails to build. A value known now holds at most 1048576 elements
in all, however its arrays nest, and a range as many. Each number, float and `bool` is an element, and so is each array
of none or one item, which costs memory of its own. A kernel declares at most 65536 variables, however its loops and
helpers repeat them. Working out a function takes at most 4194304 steps: each turn of a `for`, each instruction, call,
label or note, each number of a range, each item an array lists, as `[a, b]`, and each element of each `[x; n]`, of each
array a method makes, of each one copied from a name, as `let b: [u32; 4] = a;` and `for x in a` copy it, and of each
value an index takes or a note shows. A number read by name or worked out costs none of its own. Nesting is held to 64
levels, so that no kernel can run the compiler out of stack. Brackets, types, expressions, rows of `else if`s, consts
that name consts and a value's arrays each nest 64 deep at most, and each operator in `a + b + c` is a level of its
expression. A block, a `for` and an `if` are a level for what they hold, and count together with the expression inside
them, along the helpers that call one another. [`kernite-lang`](crates/lang/src/lib.rs) describes the language.

A helper is always inlined, and nothing else is, so `#[inline]` changes nothing kernite compiles: kernite refuses it, in
every form, on a function or a static, at the `inline`. `#[inline(never)]`, `#[r#inline]` and
`#[cfg_attr(.., inline(always))]`, whatever the condition, are refused as `#[inline]` is, and Rust says none of its own.
On a const the attribute is Rust's, and Rust says what is wrong of it.

A type in a kernel's parameters or return means what it means in any function of the module: `super::W` is the `W`
beside the module of kernels, and `self::W` the one in it. It may name the kernel's compile-time arguments, as in any
generic function: in `fn arr<const N: usize>(p: *mut [u8; N])`, `arr::<2>` takes a pointer to two bytes. Rust takes the
argument in a type alone, `N` or `{ N }`, and no expression of it, `N + 1`. An error in a type is Rust's, a pointer's
among them, said once, at the type, in a kernel of every kind, and so is a lint, at the level the kernel's `allow`,
`warn`, `deny`, `forbid` or `expect` gives it: a kernel of instances, or a cooperative one, on x86 or on a GPU, is no
different from any function. Rust's lint of needless parentheses is one: `*mut ([u8])` has its warning, on x86 and on a
GPU as in a plain function, and `*const (dyn Debug + Send)`, where the pointer needs them, has none. The one exception
is a pointer that a kernel of instances gives back, which the next paragraph tells. A generic is a value: `<T>` is an
error at the `T`, and so is `impl` in a parameter's type or the result's, wherever in it, `x: impl Copy`,
`Option<impl Copy>` or `-> *mut impl Copy`, at the `impl`: in a parameter it makes the function generic over a type, and
in the result it hides one. kernite says both in the same words. The level of a lint holds at each name in the kernel's
code too, a deprecated item's, which a paragraph below tells.

A kernel's pointer to a type with a size, `*const T` or `*mut T`, is one register: `*const [usize; 3]` and
`*mut *const f32` are taken as `*const f32` is, on x86 and on a GPU. A pointer to a type with no size has two words, an
address and a length or a table, so it is an error at its type: take the address and the length as two parameters.
kernite tells a slice, a `str` or a `dyn`, or a tuple that ends in one, in parentheses or not, by what is written, and
says so in its words. Rust tells any other, whatever it holds and whatever it is called: `CStr`, `Floats` where
`type Floats = [f32];`, `Shown` where `type Shown = dyn Debug + Send;`, a struct of yours whose last field is a slice,
`(&u8, CStr)`. It says that the size cannot be known, at the type, for a pointer a kernel returns as well as for one it
takes, in a kernel of every kind. A type that Rust refuses by itself is its error alone, said once, at the type, in a
kernel of every kind: an argument that does not meet its struct's bound, `*const NeedsCopy<String>` where
`struct NeedsCopy<T: Copy>`, or an array or a tuple that holds a type with no size, `*const [CStr; 3]` or
`*const (CStr, u8)`. One case has two errors: in a kernel of instances, `-> *const (u8, [CStr; 3])` has Rust's error
twice, at `[CStr; 3]` and at the whole type. A type in a message is written as Rust writes it, `[usize; 3]` and
`fn(f16: u8)`. A block and the arguments of a macro are written as their source has them, as Rust writes a block it does
not work out: `Marker<{N*2}>` from `{N*2}`, `Marker<{ N * 2 }>` from `{ N * 2 }` and `m!(a+b)`. A run of spaces, line
breaks and comments is one space. The length of an array is worked out in the type of a parameter or a variable that is
an array, a reference to one or a name with arguments, and in those of its elements and arguments that are: with `N` as
3, `[usize; 6]` from `[usize;N*2]`, `&[usize; 6]` from `&[usize;N*2]` and `Wrap<[usize; 6]>` from `Wrap<[usize;N*2]>`.
In any other type, and in a result, it is written as its source has it: `(u8, [usize; N*2])` from `(u8,[usize;N*2])`.

A number's name in a kernel's parameters or return, `u8` to `u64`, `i8` to `i64`, `usize`, `isize`, `f32`, `f64` or
`bool`, means that number: `u64` is 64 bits. A module that gives the name another type, `pub type u64 = u32;` and a
`use` of it, would have Rust pass a `u32` where the listing reads 64 bits, so it is an error at the type. Rust says it,
once, in a kernel of every kind: it expected `u64` and found `u32`. A type with no size, `pub type u8 = [u8];`, adds
Rust's error that its size cannot be known, at the type too. Rust says nothing at the module's attribute.

A module may give the name of a primitive, `u8` or `str`, another type, and its kernels build as they do without, so
long as none takes or gives that name as a number: the Rust kernite writes names each primitive by its path,
`::core::primitive::u8`.

Rust names each primitive by a path as well as by its name, and kernite reads the path as the name:
`core::primitive::u64`, `::core::primitive::u64`, `std::primitive::u64` and `::std::primitive::u64` are all `u64`. A
name of the path written raw is the name Rust reads, so `r#core::primitive::u64` and `::core::primitive::r#u64` are
`u64` too. So it is for each of the 17 types of `core::primitive`: `bool`, `char`, `f32`, `f64`, `i8` to `i128`,
`isize`, `str`, `u8` to `u128` and `usize`. It holds wherever a type is written: the parameters and the result of a
kernel, a helper or an `extern "ptx-func"` function; the type of a compile-time argument,
`fn k<const N: core::primitive::u32>`; the type of a variable, a const or a static; and the type a cast is to. A kernel
written with the paths compiles to the listing, or the PTX, of the same kernel written with the names, and a message
says the name: `core::primitive::str` is `str` in it.

A path in a value that begins with a primitive's, and goes on by a name or none, is read from the primitive's name too:
`core::primitive::u32::MAX` is `u32::MAX`, `::std::primitive::f32::INFINITY` is `f32::INFINITY`, and
`core::primitive::f32::from_bits(..)` is `f32::from_bits(..)`. It holds wherever a value is written: a const, a static's
data, the length of an array, a compile-time argument, the values an instance lists, and the operand of an instruction,
on x86 and on a GPU. Here too a kernel compiles to the listing, or the PTX, it has with the names, and a message says
the name: `core::primitive::u32::NOPE` is `u32::NOPE` in "`u32::NOPE` is not a constant kernite knows".

kernite reads no other path as a primitive, and quotes it as written: `my::core::primitive::u64` and `crate::Word`,
which are paths to types of yours; `core::primitive::f16` and `core::primitive::f128`, which name no type, for
`core::primitive` has neither in the Rust kernite builds with; and, where a type is written,
`core::primitive::u64::MAX`, which names a constant. Nor does it read a name of yours as the type it stands for: `Word`
after `type Word = u64;` is quoted as `Word`. A place that takes no type of yours, a kernel's parameter say, refuses
each at the type, in the words it has for `Word`: "a kernel takes pointers, 64-bit integers and floats, not
`crate::Word`".

Rust is asked of the path as it is of the name, in a kernel's parameters and result. A module that gives `core` or `std`
another meaning, `use crate::fake as core;`, has Rust say, once, at the type, that it is not the number, as it does when
the module gives `u64` another type.

A value names a number's constants by a path, in a const, a static's data and the operand of an instruction, on x86 and
on a GPU. An integer has `MIN`, `MAX` and `BITS`: `u8::MAX`, `i64::MIN` and `usize::BITS`, of the ten integers kernite
has, `u8` to `u64`, `i8` to `i64`, `usize` and `isize`. `f32` and `f64` have `INFINITY`, `NEG_INFINITY`, `MAX`, `MIN`,
`MIN_POSITIVE` and `EPSILON`, with Rust's own values, and `f16` and `bf16` have the same six, with the bits that IEEE
754's rules give a float of their widths, 5 bits of exponent and 10 of fraction, and 8 and 7, for Rust has no such
floats to take them from. `core::f32::consts` and `core::f64::consts` have 27 constants, 21 that Rust has stable: `PI`,
`TAU`, `E`, `EULER_GAMMA`, `GOLDEN_RATIO`, `FRAC_PI_2`, `FRAC_PI_3`, `FRAC_PI_4`, `FRAC_PI_6`, `FRAC_PI_8`, `FRAC_1_PI`,
`FRAC_2_PI`, `FRAC_2_SQRT_PI`, `SQRT_2`, `FRAC_1_SQRT_2`, `LN_2`, `LN_10`, `LOG2_E`, `LOG10_E`, `LOG2_10` and `LOG10_2`;
and six that it has unstable, `SQRT_3`, `SQRT_5`, `FRAC_1_SQRT_2PI`, `FRAC_1_SQRT_3`, `FRAC_1_SQRT_5` and
`FRAC_1_SQRT_PI`. kernite names them as Rust does, by the path from `core` or `std`, `core::f32::consts::PI` or
`::std::f64::consts::LOG2_E`, with a name of the path written raw too, `std::r#f64::consts::r#E`. Each is Rust's own
value, bit for bit, and a kernel that names it compiles to the listing, or the PTX, of one that makes the same bits,
`f32::from_bits(0x40490fdb)`. Rust says of each of the six, at the path, "use of unstable library feature
`more_float_constants`", where the crate does not enable the feature, and kernite says nothing: its values are the bits
that Rust's nightly gives them.

In a module of kernels Rust holds each path to a constant to the bits kernite takes for it: kernite writes a check at
the path, in a const, a static's data and an operand, on x86 and on a GPU, and Rust runs it as it builds the module. A
module that gives a name in the path another meaning, `use crate::fake as std;`, where the meaning has a
`f32::consts::PI` of its own, has Rust's constant in kernite and its own in Rust, and Rust says, at the path,
"`std::f32::consts::PI` is not Rust's constant, which kernite takes, with the bits 0x40490fdb: a name in the path means
another thing in this module; write `::core::f32::consts::PI`". It says so of `std`, of `core`, and of an `f32` or an
`f64` that the module names a module of its own. Where the meaning has no constant at the path, Rust says what it says
of any path, that it cannot find `f32` in `std`, at the name, and kernite says nothing. Of a path that begins with
`::std` or `::core`, which name the crates whatever the module gives, Rust says nothing. A const of a module of consts
is Rust's where its module is, and holds nothing that gives `std` another meaning: a module of kernels that takes it is
asked nothing of its paths, whatever the module means by `std`.

The path without its root, `f32::consts::PI`, is Rust's only after `use std::f32;`: where the module has no such `use`,
Rust says "ambiguous associated type" at it, once for each place it is written. A path through the primitive,
`core::primitive::f32::consts::PI`, is no path of Rust's, `use std::f32;` or not, for `core::primitive::f32` is the
type, which has no `consts`: Rust says the same of it, as it is written. kernite takes both for Rust to refuse, and says
nothing, in a module of kernels and in a module of consts, and in the kernels that take its consts. No Rust reads the
text that `x86_str` and `ptx_str` are given, and kernite reads no `use`, so there it refuses the path without its root,
at its first token, and says the path to write: "`f32::consts::PI` is Rust's only after `use std::f32;`, which kernite
does not read: write `core::f32::consts::PI`". It refuses the path through the primitive as it is written:
"`core::primitive::f32::consts::PI` is not a constant kernite knows".

A path that names no constant kernite knows is kernite's error alone, at the path, and Rust is not asked of it:
"`std::f32::consts::NOPE` is not a constant kernite knows". So it is for a name that the two modules have not; a type
that has no `consts`, `f16::consts::PI`; a constant that Rust has in a module of its type too, `std::u32::MAX` or
`core::f32::MAX`; and a number's own constant that kernite has no value for, `f32::NAN`, `f64::DIGITS` or `u128::MAX`.
A number's own constant, `u32::MAX`, is not asked of Rust either: a module may give `u32` another type, and its kernels
build as they do without.

Any other type a kernel takes or gives back is kernite's error alone, at the type, once for each instance: `f128`,
`[f16; 4]`, `(f16, u64)`, `&f16`, `!` as a parameter, `[u64]`, `str`, `dyn Fn()`, an `f16` or a `bf16` on x86, or a type
of yours, whatever it is called: a type of the crate's named `f16` or `f128` is one, and Rust does not call its `use`
unused. Rust says nothing of the type beside it: not that it is unstable or has no size, as it says wherever the type is
named, and nothing at a call of the kernel, which the crate's own code makes as it makes any call. What Rust says of any
function's signature it still says: a lifetime it cannot find, as for a result `&u64` with none to take, on x86, or for
an `'a` not declared; a `_` in a result; and a module where a type is written. Rust expected a type and found a module
in `x: core::f16`, or in `[core::f128; 2]` on a GPU. It does so too for an `f16` or an `f128` where the kernel's module
imports a module so named, `use crate::f16;` with `x: f16` or `x: [f16; 4]`: Rust reads the name as the module. A
pointer to such a type is a pointer, and is said as above: `*mut f128` is Rust's error that `f128` is unstable, and
`*mut [u64]` is kernite's, for it has two words. An `f16` or an `f128` that kernite does not see as a type keeps Rust's
word that it is unstable, beside kernite's error: one that a macro writes, `half!()`, or one in an expression of the
type, as in an array's length, `[u8; size_of::<f16>()]`.

A cooperative x86 kernel's threads return nothing, and so does a GPU kernel: a result is kernite's error, at the type.
Rust sees the type as it sees a plain x86 kernel's, and says what it says of any function's result, once, at the type,
in the same words. A pointer to a type with no size, `-> *const CStr`, has Rust say that the size cannot be known, and
`-> *const NeedsCopy<String>` that the bound is not met. A lifetime it cannot find, `-> &'a u64`, a `_`, `-> *const _`,
a module where a type is written, `-> core::f16`, and an `f128` behind a pointer, `-> *const f128`, are Rust's errors
beside kernite's. An import that only the result names is used, so Rust does not call it unused, and Rust says nothing
at a call of the kernel. A GPU kernel's launcher is a method, so the `&` of a result, `-> &u64`, takes the lifetime of
`self`, and Rust has none to ask for. A pointer of two words to a type Rust refuses by itself, `-> *mut dyn Copy`, is
kernite's error alone, though Rust says its own of a plain kernel's result. A result of `()` is no result, and no
error.

A helper is compiled where a kernel calls it, and an `extern "ptx-func"` function is a `.func` of the GPU's module, so
Rust sees neither. Where kernite finds any error in the module, Rust is shown the types of every helper and function of
the module, one that no kernel calls too: each compile-time argument's, each parameter's and the result's, as it sees a
kernel's. An import that only these types name is used, so Rust does not call `use super::Ret;` unused beside kernite's
error at `Ret`, which is said once: at the argument of a call of the helper, at the result of a helper that is called,
or at the type of a function. Rust says of the types only that a lifetime is not declared, `x: &'a Ret`; that a module
is where a type is written, `x: core::f16` or `x: core`; and what a macro writes, `x: half!()` with an `f16`, which is
unstable. It says the same of the types of a helper that no kernel calls, of which kernite says nothing. It says nothing
of a type it refuses by itself, `[u64]`, `f128` or `_`. In a module with an error of kernite's, it says nothing of any
name it cannot find, `gpr` or `local<[u32; 4], 16>` in these types, or a const that `cfg` leaves out, until the errors
are mended. A function's types are shown where `cfg` keeps the function and the error is said. A module that builds is
shown none of them, so an import that only a helper names is unused to Rust, which says so as it does of any import that
nothing names.

A helper and an `extern "ptx-func"` function are no symbol of the host's, nor a function of Rust's, so each takes the
attributes that say how Rust has it, and no other: a doc, `cfg`, `deprecated`, the level of a lint, `allow`, `warn`,
`deny`, `forbid` or `expect`, and a `rustfmt` attribute, each alone or in a `cfg_attr(..)`, whatever its condition.
kernite refuses any other, as an error at its name, which says that the item is a helper or an `extern "ptx-func"`
function, which the attribute is not for: `#[unsafe(no_mangle)]`, `#[unsafe(export_name = ..)]`,
`#[unsafe(link_section = ..)]` and `#[used]` are for a symbol of the host's, `#[cold]` and `#[track_caller]` for a
function of Rust's, `#[must_use]` and the rest of Rust's for an item of its own, and `#[foo]`, which nothing defines,
for none. Rust reads the `deprecated` of each as it reads a function of its own, though no kernel names it: a
`deprecated` that is malformed is Rust's error at the attribute.

A name written raw is the name Rust reads, in a type as in a value's name: `r#u64` is `u64`, `*const r#u8` is
`*const u8` and `[r#f32; 4]` is `[f32; 4]`, in a parameter, a result, a variable, a compile-time argument's type,
`<const N: r#u64>`, a const and a static. A kernel written so compiles to the listing, or the PTX, of the one written
plain, and the launcher of a GPU kernel that takes `r#f16` or `r#bf16` takes kernite-cuda's half, as it does for `f16`
and `bf16`. A type kernite refuses is refused as it is written plain: `r#f16`, `[r#f16; 4]` and `r#f128` are its error
alone, at the type, once for each instance, as `f16`, `[f16; 4]` and `f128` are, and it says the type as Rust reads it,
`f16` and not `r#f16`.

A keyword written raw is a name, and no keyword: `r#dyn::Foo` is a struct in a module named `dyn`, `r#mut<u64>` is a
generic struct named `mut`, and a pointer to `r#dyn::Foo` has one word, as Rust has it, where a pointer to `dyn Foo` has
two. So it is for every keyword Rust lets be written raw, `r#for`, `r#true` and `r#type` as `r#dyn`. A kernel that names
a type so compiles to the listing, or the PTX, of one that names types of the same layout by words that are no keywords,
`dyn_::Foo` say. An error says the keyword as it is written, `r#dyn::Foo` and `r#mut`: without its `r#` the word is the
keyword, and not the name.

An attribute of kernite's written raw is the attribute: a kernel's `#[r#instances(1, 2)]` and `#[r#cooperative]`, a
GPU kernel's `#[r#maxntid(..)]`, `#[r#reqntid(..)]`, `#[r#minnctapersm(..)]` and `#[r#maxnreg(..)]`, and a module's
`#[kernite_macros::kernels(r#consts = crate::model::sizes)]`. A kernel written so compiles to the listing, or the
PTX, of the one written plain. An attribute written raw is wrong as the plain one is: the same error, in the same
words, at its name, which starts at the same `line:column`, and Rust adds none of its own. So `#[r#maxnctapersm(..)]`
is refused as `#[maxnctapersm(..)]` is.

Five names cannot be written raw, as in Rust: `self`, `Self`, `super`, `crate` and `_`. In a module of kernels Rust
refuses one written so at each place, in its own words, "`self` cannot be a raw identifier", and an `r#` that no name
follows as well. Of a parameter or a compile-time argument so named it says that alone: the macros name it for Rust, as
they name a parameter `_`, and write the name no more. kernite refuses each where it reads a module from text, as
`kernite::x86_str` and `kernite::ptx_str` do: at its `r#`, "`self` cannot be written raw" and "`r#` comes before a
name, as in `r#u64`", wherever a name is written, an attribute's too. Any other name can be written raw, a weak keyword
or a reserved word as well, `r#union` and `r#gen`.

A `mut` before a parameter's name is taken and means nothing, in a function of every kind: an instruction may write any
variable, which Rust cannot see. So the macros write no `mut` for Rust, which would say that the variable does not need
to be mutable, and in a naked function that patterns are not allowed there.

A parameter named `self` or `Self`, written plain, is Rust's error as of any function that is no method, at the name,
and nothing more: that `self` is only allowed in associated functions, or is unexpected where it is not the first
parameter, or that `Self` is not found in this scope. So it is in a kernel of every kind, in a helper and in an
`extern "ptx-func"` function, with `mut` before the name too, of which Rust says no more than of a function's: before
`Self`, that `mut` must be followed by a named binding. The macros name it for Rust, as they name a parameter `_`, and
give Rust a function of its own that never runs, to say it in. That function has `_` in the place of each other
parameter, and each is of the type `()`, so Rust says nothing there of a type as written, that it cannot find `gpr` or
that `f128` is unstable, nor that a parameter is unused.

A space is what Rust reads as one: U+0009 to U+000D, U+0020, U+0085, U+200E, U+200F, U+2028 and U+2029, which Rust's
lexer calls Pattern_White_Space. Each other character that Unicode calls a space, U+00A0, U+1680, U+2000 to U+200A,
U+202F, U+205F or U+3000, is no space, as in Rust, and is refused at the character: "unexpected `\u{a0}`". A character
that is not printable ASCII is shown in that error as Rust shows it, `\u{a0}`, and one that is, as it is.

A name is what Rust reads as one: `_`, or a character that Unicode says starts an identifier, XID_Start, and then each
that goes on one, XID_Continue, in the Unicode 17.0.0 of Rust 1.100. So `a·b`, `a‿b` and an `e` with U+0301 after it are
names, and `a½` and `a²` are not: the name ends at the `a`, and the character after it is refused at the character,
"unexpected `\u{bd}`". `·` and U+0301 go on a name and start none, so `·a` is refused at the `·`. A name is read in NFC,
as Rust reads it, so each spelling of a name is that name: `café` with `é` as U+00E9 and with `e` and U+0301 is one
helper, one variable, one const or one kernel, which `x86_str` finds by its name in either spelling. Two functions so
named are refused as two of one spelling are, "`café` names another function of the module as well", and a listing
writes the name in NFC.

A string is plain or raw, as in Rust, wherever kernite takes one: a note's text, `note(r#"a "b""#)`, and the ABI of a
function, `extern r"sysv64"`. A raw string has no escapes, and ends at the first `"` that has as many `#` after it as
the string began with, so `r##"a "# b"##` holds a `"#`. A string that is never closed, plain or raw, is refused where it
starts: "this string is never closed". So is a raw string of more than 255 `#`, as Rust refuses it. A CRLF in a string,
plain or raw, is a LF, as Rust reads it. A CR that no LF follows is refused at the CR, as Rust refuses it, in a string,
plain or raw, and in a doc comment: "a CR on its own is not allowed in a string, as in Rust: write `\r`", "a CR on its
own is not allowed in a raw string, as in Rust" or "a CR on its own is not allowed in a doc comment, as in Rust". A
comment that is no doc comment takes it, as Rust does, and so does a string that writes it by the escape `\r`.

A plain string takes the escapes that Rust reads: `\n`, `\r`, `\t`, `\0`, `\\`, `\'` and `\"`; `\x41`, of two hex
digits, 7f at most; `\u{e9}`, of one to six hex digits in braces, with `_` allowed after the first, that make a
character; and a `\` that ends a line, which leaves out the line break and the spaces, tabs and line breaks after it, a
CR on its own too. So `note("caf\u{e9}")` shows `café`, and `extern "sysv\x364"` is `sysv64`. Any other escape is
refused at its `\`, as Rust refuses it, with one of three messages: "`\q` is no escape, as in Rust: a string has `\n`,
`\r`, `\t`, `\0`, `\\`, `\'`, `\"`, `\x41`, `\u{e9}` and a `\` that ends a line", "`\x` takes two hex digits, 7f at
most, as in Rust: write a greater character as `\u{80}`" or "`\u` takes one to six hex digits in braces, as `\u{e9}`, of
a character: not d800 to dfff, and 10ffff at most, as in Rust".

A note's text holds no NUL, on any machine, as `ptxas` reads a NUL as the end of the module. It is refused at the
string, whether the NUL is an escape of a plain string, `\0`, `\x00` or `\u{0}`, or the character itself, in a plain or
a raw string: "a NUL is not allowed in a note, on any machine: `ptxas` reads it as the end of the module".

A note shows a value by its name in braces, `note("acc {k}")`, and reads the name as `format!` does: it comes right
after the `{`, and spaces, as `char::is_whitespace` says, may come after it, before the `}`. So `{k }` shows `k`, and
`{ k}` is refused at the string: "a note's name comes right after its `{`, with no space before it: `{k}`". A note has
no arguments, so braces with no name in them, `{}` or `{ }`, are refused at the string too: "a note has no arguments for
`{}`: its braces hold a value's name, `{k}`". Any other character in the braces is refused at the string, the `:` of a
format too, `{k:x}`, for a note has none. It is shown as `format!` shows it, by `char::escape_debug`, so `§` as it is
and U+200E, which shows as nothing, as `\u{200e}`: "unexpected `\u{200e}` in the braces of a note, which hold a name and
then spaces". The name is not read in NFC, as `format!`'s is not: `{café}` with `é` as U+00E9 finds `café` declared in
either spelling, and `{café}` with `e` and U+0301 finds neither.

A macro's fragment is one operand, or one type, wherever kernite reads it, as Rust reads it. With `$n:expr` as `1 + 2`,
`$n * 2` is 6, and not the 5 that `1 + 2 * 2` is. It is so in a kernel's expression, on x86 and on a GPU, under a `-`,
in the length of an array, in `#[instances(..)]`, and in a const, which is 6 to Rust too. A `$t:ty` is one type in a
parameter's type, in a result and in an argument of a type, `*const Wrap<$t>`. It is one behind a pointer, a reference
or an arrow too: `*const $t` with `$t` as `dyn Debug + Send` is a pointer to one type, as `*const (dyn Debug + Send)`
is, and `fn() -> $t` a function pointer that gives back one, and Rust does not call the `+` of either ambiguous.
kernite's errors name such a type with its bounds in parentheses behind a pointer, a reference or an arrow,
`&(dyn Debug + Send)`, as if they were written so. Anywhere else, and for a fragment with no bounds of its own, they say
the fragment as it is written: `Wrap<dyn Debug + Send>`, `*const [u8]`. An error in a fragment is said where the
fragment is written, in the call of the macro. Rust says its error of the whole fragment where the fragment starts: that
`1u16 + 2` is no `u8`, at the `1u16`. Where kernels of one macro take the same `$t:ty`, `$t:path`, `$t:ident` or
`$t:tt`, a result with a `&` before it, `&$t`, `Option<&$t>` or `&$a::$t`, has Rust say, at the `&` of each, that it has
no lifetime to take, as for kernels written out, wherever the macro is defined, and whether it writes the body or its
call gives it, a `$b:block`. Its help to return an owned value takes out that `&`, and the `mut` after it, and no more
of the macro. An `$i:ident` or a `$t:tt`, a fragment that holds no operator, `u8::MAX` or `st.global`, and one that no
operator is next to, `let k = 2 + 3` as a `$s:stmt`, are read as they are written.

The kernels read the module's consts as values known now, and so does Rust. Consts that modules share, as a model's
CPU kernels and GPU kernels share its sizes, are a module of their own, which `#[kernite_macros::consts]` is on, and a
module of kernels takes them by its path, `#[kernite_macros::kernels(consts = crate::model::sizes)]`: its kernels read
them as the module's own, and Rust sees them there by a `use` the macro writes. A module of consts is the crate's own,
and holds `pub` consts of what kernels read, numbers, floats, `bool`s, arrays of them and expressions of these; a
module that takes them may not have a const, a function or a static of its own of one of their names. Each const,
function and static of a module has a name of its own.

kernite refuses a module of kernels whole, at its attribute, for an argument of `#[kernels]` that is not
`consts = path`, such as `consts = 5`, and for consts given in a crate with no dependency named `kernite-macros`. It
says so once, there, and the module stays. Each of its kernels is a stub, as one that does not compile is. Each of its
statics is a static of its type, holding zeros, if kernite works its type out and takes its name and its type, and one
whose elements are of any type if not. So Rust says nothing of the code that names the module's items, wherever it is,
but for two kinds of use, beside the refusal. A use that the type of the first kind of static does not fit is Rust's
error, as it is for a static that compiles. A use of the second kind that says no type for its elements has Rust ask for
it. The module is given no consts, so a type that reads one is of the second kind. kernite checks none of the items, so
it says nothing it would find in them, such as a kernel that must be `unsafe` and is not, an ABI it does not know, what
a static holds, or what compiling a kernel finds. It compiles nothing of the module, so a workspace `name` that is not a
string is no error at it. No room is counted for the statics, so a crate with no dependency named `kernite-macros` is
not told it needs one for them. What the macro finds in writing the module for Rust is said, as for any module: the
dependency the crate needs, `kernite-cuda` for a GPU item that `cfg` keeps and `kernite-cpu` for an x86 static or
cooperative kernel; a compile-time argument named as a parameter of a launcher or of a `_blocks` method,
`__kernite_grid`, `__kernite_threads` or `__kernite_shared`; and a GPU kernel or a static named as a cooperative
kernel's `_blocks` method.

A module kernite cannot read is refused whole as well, and stays with every item kernite can read. A `const` or a
`static` it cannot read ends at its `;`, for its value may hold a `{..}`, as in `= { 1 };`. Any other item ends at its
first `;`, or at its first `{..}` outside `<..>`: a function at its body. The items after it are read as they are. If
items share a name, it knows one of them: a const before a function, a function before a static, else the first. If it
cannot read an item and reads no function that names an ABI, it knows no static, for the kernels say whose a static is,
the process's or the GPU's. If brackets nest too deep, it knows no item. Its items are as in a module refused at its
attribute, with two differences: the module is given the consts it takes, and kernite says nothing of an item of its own
that has the name of one. The error is said once, where it can be mended. If the module's attribute is refused too, that
refusal is said instead. So Rust says nothing of the code that names an item kernite knows, by a path, by a `use` or by
a glob, beside the error. Of a name kernite does not know, as that of an item it cannot read, Rust says nothing of a
path or a `use`, for the error is in the module. It cannot find the name by a glob or as a method of `Kernels`, as it
cannot find a name that no module has. A glob that finds none of what it names is unused, and Rust warns of it.

A path to a module of consts that Rust cannot find, as in `consts = crate::model::sizs`, is told once, by Rust, at the
name it cannot find. So is a path to a module with no `#[kernite_macros::consts]` on it, at the module's name: Rust says
the module has no `__kernite_consts`, which that attribute writes. Rust's error is an unresolved import, of the path as
the macro writes it, from a module of its own in the module of kernels. A path from `self`, `super` or a name is said
from there: `nope` is `super::super::nope`. A name after the first is said in the module before it, which the macro
names `__`, that module's name, `_` and a number: `crate::hidden::nope` is `__hidden_3::nope`. The macro cannot tell
that the path fails, for Rust finds the path after the macro runs. The module of kernels stays, with nothing in it. Rust
says nothing of the code that names its items by a path, wherever it is. A glob of the module brings in nothing, and
Rust warns of that alone. A `use` of an item by its name, `use crate::cpu::step;`, is Rust's error as well, for the
module has no `step` until the path is mended.

An argument of `#[consts]`, as in `#[consts(x)]`, is told where it is, and changes nothing else. The module is the one
it is without the argument: its other errors are said, and Rust says nothing of the code that names its consts, nor of
the modules of kernels that take them.

A function or a static that `cfg` leaves out says nothing, as Rust says nothing of a function it leaves out. A macro
runs before Rust knows which `cfg` holds, so kernite compiles each one, and puts each error it finds in one under the
same `#[cfg(..)]` or `#[cfg_attr(.., cfg(..))]`: a kernel that does not compile, or is not `unsafe` where it must be, a
helper, a static. Where `cfg` leaves them out, the crate builds without a word; where it keeps them, each error is said
where it is. A crate that keeps its x86 kernels on x86-64 alone, by `cfg(target_arch = "x86_64")`, builds for another
machine without a word, though only x86-64 has `extern "sysv64"`: each function kernite writes for a kernel is under the
kernel's `cfg` too, as one for each instance of a kernel of compile-time arguments or of a cooperative one.

A kernel names the items of its module that it uses: a const it reads, a static, a helper, a kernel or an
`extern "ptx-func"` function it calls. A kernel names an item in its body, in its `#[instances(..)]`, in its directives
or in a helper it calls; an `extern "ptx-func"` function names one in its body; a static, in its type or in what it
holds. A helper names nothing of its own: the kernel or function that calls it names what it names, where the helper
writes it, and only where `cfg` keeps the helper as well as the kernel. What a const names in its definition is Rust's
own, and not the kernel's.

A kernel that `cfg` keeps names only what `cfg` keeps. Where it names an item `cfg` leaves out, the name is an error,
said at the name, once for each place it is written, in a kernel of instances too. Rust never sees a kernel run, so
kernite says it. Of the items that are Rust's, a const, a static in the process's memory and an x86 kernel that takes no
compile-time arguments, kernite writes the name for Rust where the kernel writes it, under the kernel's `cfg`, and Rust
says what it says of any name it cannot find, `cannot find value`, in a module where kernite says no error. Of the rest,
which Rust has no item for, a helper, an instance of an x86 kernel, an `extern "ptx-func"` function and a static in the
GPU's memory, kernite says that `seven` is left out by `cfg` where `calls_seven`, which names it, is kept. Where `cfg`
leaves out the kernel too, or keeps the item wherever it keeps the kernel, nothing is said. A kept kernel that calls a
helper `cfg` leaves out, itself or through a helper `cfg` keeps, is told of the helper alone, whatever the helper names.
One that calls a helper at two places, one of them in a helper `cfg` leaves out, is told of what the helper names, once.
A const that `cfg` leaves out with every item that names it is not said of, and one that `cfg` keeps and only items
`cfg` leaves out name is unused to Rust, which says so as it does of any const that nothing names.

Rust tells of an item of a module of kernels that has no doc, where `missing_docs` is on, as it tells of an item of
plain Rust: at the item's `pub`, and not at the module's attribute. That holds of a static of x86 kernels, of a static
of GPU kernels, which Rust has as a method of `Kernels`, and of an x86 kernel. Clippy tells there too of a public x86
kernel that is an `unsafe fn` and whose docs have no `# Safety`, as it does of any public `unsafe fn`. Neither tells of
a GPU kernel or a cooperative x86 kernel, whose launcher the macros write with a doc, a `# Safety` where it is an
`unsafe fn`, nor of an x86 kernel that names a static of its module, or whose code needs a CPU feature past x86-64's
own, an `unsafe fn` or not, plain or of instances: the macros write its `# Safety` where its docs have none, and add to
the one they have.

The macros take a heading of the word Safety alone for the docs' `# Safety`: `Safety` or `SAFETY`, in emphasis or not,
of any level, written as Markdown has a heading, with `#`s or with a line of `=` or `-` under it, in a block quote or a
list item or not, as rustdoc reads the docs. The words in a sentence are no heading, nor is a title that says more,
`# Safety notes`, nor is a line of a code block, fenced or indented, or of a block of HTML. Clippy takes more for a
`# Safety`: a title that has the word among other text, `# Safety *now*`, a title of two lines, one of them the word,
and a title in a footnote or right after the definition of a link. Where it takes one that the macros do not, the docs
have a `# Safety` of the macros' own as well. What the macros add to the docs' `# Safety`, the sentence on the module's
memory and the CPU features, is its last paragraphs, before the next heading of its level or higher; where the docs have
none, the macros write a `# Safety` of their own at the end of the docs. A paragraph of theirs that is no part of a
`# Safety`, the grid of a cooperative kernel that runs at once and the launch of a safe GPU kernel that is checked
first, comes at the end of the text before the docs' first heading, so that no heading of the user's has it. Each
paragraph of theirs has a blank line after it, as before it, so that a title with a line of `=` under it, or a list,
that comes next is not a line of it. None goes in a block quote or a list item, which it would end: a heading in one is
where its block begins, and what the macros add to a `# Safety` in one comes after the block, even where the block has a
heading after it. Where the docs end in a code block or a block of HTML that they leave open, what the macros add comes
after it. A doc of a macro's text, `#[doc = include_str!("..")]` say, the macros cannot read: they take it for a line of
text with no space before it. Where it holds more, a heading or a block that it leaves open, they may read the docs
otherwise than rustdoc does, and Clippy may tell of the kernel.

A deprecated item that a kernel names is told of by Rust at the name, once for each place the kernel writes it.
`lea(p, [COUNTS]);` of a deprecated static is a warning at `COUNTS`, with the note of the attribute, `call(v, old, ());`
of a deprecated x86 kernel is one at `old`, and `call(old::<2>, p);` of an instance of one that takes compile-time
arguments is one at `old` too, as a use of a generic function of Rust's is. A static in the GPU's memory is told of as a
use of its method, `Kernels::COUNTS`. Rust never sees the code of a helper, an `extern "ptx-func"` function or a GPU
kernel, so a deprecated one is told of as a use of a function of its name that kernite writes, and Rust says its path
with an `_`, `cpu::_::touch`: `touch(p);` of a helper, `call(v, gcd, (a, b));` and `mov.u64(fp, gcd);` of a function,
and `mov.u64(fp, run);` of a kernel are each a warning at the name, with the note of the attribute. A use inside the
deprecated item itself is not told of, as in Rust. Rust says nothing at the module's attribute, nor of an item that no
kernel names, but for an error in its `deprecated`, which Rust reads as it does of a function of its own: it says the
error once, at the attribute, whatever places name the item. An x86 kernel of instances is a function for each instance,
and the code of a helper is part of each x86 kernel that calls it. Where the build is incremental, as the dev profile's
is, Rust tells of the first place each function writes the name: once for each instance, and for a place in a helper,
once for each kernel that calls it. `old::<2>`, a helper, an `extern "ptx-func"` function and a GPU kernel are told of
once at each place, whatever instances of a kernel write it and whatever kernels call the helper it is in.

The level of a lint that a kernel sets holds at each of these places, as in a function of Rust's. `#[allow(deprecated)]`
on a kernel has Rust say nothing at any of them, on x86 and on a GPU; `warn`, `deny` and `forbid` make each a warning or
an error, though the crate allows the lint or denies it; and `expect` is met by each: Rust says nothing of them, and of
a kernel whose code names no deprecated item, that the expectation is unfulfilled, once, at the attribute. A GPU kernel
in a crate without `kernite-cuda` has its places at its level too, beside the error that says the crate needs it. Where
a kernel with a level calls a deprecated helper, GPU helper or `extern "ptx-func"` function, Rust says the kernel's name
in the path, `cpu::warned::_::touch`, as the function of the item's name is in the kernel's own function, one for all
the kernel's places. Where the build is incremental, Rust counts an error in the item's `deprecated` once for each
function that has it: the function of each kernel, helper or function with a level whose code names the item, the one of
all the other places that name it, and the launcher of a GPU kernel. A helper, a GPU helper and an `extern "ptx-func"`
function have a level of their own for their code, and the level of the kernel that calls them does not reach in:
`touch(p);` in a helper with `#[allow(deprecated)]` says nothing, whatever kernel calls it, and in one with no level it
is told of at the module's level, though the kernel allows the lint. A helper is compiled only where a kernel that `cfg`
keeps calls it, so Rust says nothing of the `expect` of a helper or a GPU helper that no such kernel calls, whatever its
code names; an `extern "ptx-func"` function is compiled whether a kernel calls it or not, and its `expect` holds as a
function's does.

One place does not have the level of its code: the first an x86 kernel writes the name of a static, or of a kernel that
takes no compile-time arguments. The name is an operand of the kernel's function, so Rust tells of it at the level of
the kernel, whichever code writes it: a place that only a helper writes first is told of at the kernel's level, or at
the module's where the kernel sets none, though the helper has a level of its own. Where the kernel allows the lint or
expects it, Rust says nothing of the operand, and the place has the level of its code. The function is naked, and Rust
takes no `expect` on a naked function: a plain x86 kernel with an `expect` is Rust's error, "the `expect` attribute is
incompatible with `#[unsafe(naked)]`". A kernel of instances, or a cooperative one, takes `expect`, as the function of
each instance allows the lint instead: where the crate forbids the lint, Rust says at the attribute that neither the
`expect` nor the `allow` can overrule the `forbid`.

The level of a lint that a static sets holds at each place its type or its data names an item, as in a static of Rust's:
a static of x86 kernels, in reach of `rip` or far from it, and a static in the GPU's memory. `#[allow(deprecated)]` on
`static S: global<[u8; N]> = [K, K];`, with `N` and `K` deprecated consts of the module, has Rust say nothing of any of
the three places, though the crate denies the lint. `warn`, `deny` and `forbid` make each a warning or an error, though
the crate allows the lint or denies it; and `expect` is met by each: Rust says nothing of them, and of a static whose
type and data name no deprecated item, that the expectation is unfulfilled, once, at the attribute. A static that sets
no level is told of at the module's level. A static in a crate without the crate it needs, `kernite-cpu` for one of x86
kernels and `kernite-cuda` for one in the GPU's memory, has its places at its level too, beside the error that says the
crate needs it.

A module holds what `cfg` keeps: Rust puts it together, where it knows which `cfg` holds. `LISTINGS` has the listing
of each x86 kernel that compiles and `cfg` keeps, and of no other. A module of x86 kernels has it whether its kernels
compile or not, so Rust says nothing of the code that names it, and it is empty where none compiles. A module of GPU
kernels alone has none. `PTX` has each GPU kernel, function and static that `cfg` keeps, for the PTX and the GPU that
the kernels and functions kept need, and `Kernels::load` asks the driver for the names of those alone.

kernite reads every item before Rust knows which `cfg` holds, as Rust parses every item before it does, so an item it
cannot read is said whatever `cfg` holds, as Rust says a syntax error in a function it leaves out: the head of a
function, as `fn k<T>()`, rests on no other item. What rests on all the items together is said whatever `cfg` leaves
out: a name that two items share, though each is under a `cfg` the other is not.

The x86 target knows every instruction of x86-64-v3, in each form `llvm-mc` takes, [`forms`](crates/x86/src/forms.rs):
x86-64's own on integers of 8 to 64 bits, SSE's to SSE4.2's as AVX writes them, `vaddps` and not `addps`, and AVX's,
AVX2's, FMA's, F16C's, BMI1's, BMI2's, LZCNT's, POPCNT's and MOVBE's. It leaves out the stack's, which is the frame's,
the strings', `loop`, x87's, MMX's and the system's; and it does not yet write those that take registers of their own,
as `mul` takes `rax` and `rdx` and `shl` may take `cl`, which it names as such. A gpr is `g`, or `g.r32`, `g.r16` or
`g.r8`, and a write to `r16` or `r8` keeps the rest of it, as x86 does. Memory is as much as the other operands say, or,
where an instruction's forms differ by that alone, as Intel's word before the brackets says, `movzx(g.r32, byte[x])`,
which kernite checks wherever it is given. An address is a base gpr, an index gpr times 1, 2, 4 or 8, and a number,
`[x + i * 4 + 8]`. The base may be left out, `[i * 4 + 8]`, and x86 writes it so, with a 32-bit displacement. A gpr that
comes twice has its scales added, so `[x + x * 3]` is `[x * 4]`. A gpr whose scales come to 0 is left out, so
`[x + i * N]` is `[x]` where the const `N` is 0. With no base, a gpr times 3, 5 or 9 is that gpr added to itself times
2, 4 or 8, as x86 has it: `[i * 5]` is `[i + i * 4]`. Any other scale is an error, `[x + i * 3]` or `[i * 6]`. A gather,
`vpgatherdd(v, [x + i * 4], mask)`, has a vector of indices in `i`, and its base may be left out too, `[i * 4]`. It
takes its result, indices and mask in three registers, as x86 faults otherwise. The listing's header and the function's
`# Safety` name the CPU features a kernel needs, `# needs: avx, fma`, which nothing checks for when it runs; and a
kernel that loads MXCSR stores the caller's first and loads it back before it returns, as SysV says. Where `llvm-mc` and
GNU `as` are installed, a test hands them every form, with each integer the least the form takes and the most: `llvm-mc`
must take it, and GNU `as` must take it at the feature kernite says it needs, without a warning, and refuse it at the
features before.

An editor shows an instruction's doc where a kernel names it: hover at `vaddps` in rust-analyzer. The doc says what the
instruction does, in a sentence of kernite's own; each form it takes, as a kernel writes it, `vaddps(=ymm, ymm,
ymm/m256)`, with `=` before each operand it writes, `+` before each it reads and writes, and `~` before each it reads
nothing of; the CPU feature it needs; and where Intel's SDM says the rest. Each instruction of `forms` has a doc, and a
family's, `cmovz`, `vcmpltps` or `vfmadd231ps`, says its own condition, predicate or order. The macros write a doc as a
function that nothing calls, in a block of the module, and a `let _ = ..;` at each name that names it, under the `cfg`
of the item the name is in. So a doc adds no code and changes no listing and no error, and needs no `allow`: a name
that is no instruction has its own error and no doc, and a function that `cfg` leaves out has none for a name only it
names.

A kernel gives back its result as SysV does: `ret(g)` of a gpr for an integer, a `bool` or a pointer, which SysV has in
`rax`; `ret(v)` of a vector for a float, which it has in `xmm0`; and `ret()` for none. Any other `ret` is an error at
the `ret`, in a kernel and in a function of the module alike: `ret(v)` in one that gives back an integer, a `bool` or a
pointer, `ret(g)` in one that gives back a float, `ret()` in one that gives back a value, and `ret(g)` or `ret(v)` in
one that gives back nothing.

SysV passes an `f32` in the low 32 bits of a vector register and an `f64` in the low 64, and leaves the rest as the
caller had it, which may be different at each call. A call gives a float back so, and leaves the rest as its callee had
it. kernite holds each vector variable to how many of its low bits are the kernel's own: a float's 32 or 64 to begin
with, and all 256 of any other; and then what each instruction that writes it makes of it. An instruction that reads
more of it, on any way to the instruction, is an error at the instruction, which names the variable: `vmovups([p], x)`
of an `f32` `x` reads 128 bits of it and `vmovups([p], x.ymm)` 256, where `vmovss([p], x)` reads its 32. The variable is
whole once the kernel has written all of it, with `zero(x)`, a load, or an instruction that computes all of an xmm or a
ymm, as `vaddps` does. A scalar instruction writes the float alone, and carries the rest of its first source to the same
place of its result: `vaddss(y, x, z)` leaves `y` the bits of `x` past the float, where `vaddss(y, w, z)` makes all of
`y` the kernel's own if `w` is. An instruction that carries bits does not read them, and its result is the kernel's own
as far as every bit it carries is: a move of a whole register, `vmovaps(y, x)`; the logic of bits, `vandps(x, x, mask)`,
with which Rust takes the absolute value of a float, and `vxorps(x, x, x)`, which carries the bits of `x` though it
computes zero, where `zero(x)` writes all of it; a blend, `vblendps(y, x, z, 1)` by a number, which takes each lane from
one source, and `vblendvps(y, x, z, m)` by a register, whose selector counts in whole lanes; and what a merge keeps: the
rest of the first source of a scalar instruction, the half that `vmovhps`, `vmovlps` and `vmovhlps` keep, the lanes that
`vinsertps`, `vpinsrd` and `vinsertf128` leave, the element of each half of its first source that `vunpcklpd` leaves in
place, and of its second that `vunpckhpd` does, the last element of the second source of `vunpckhps`, and the lanes of
its result that a gather's mask leaves out. An instruction that picks lanes by a number known now reads the lanes it
picks and no more: `vextractps(g, x, 0)` and `vpshufd(y, x, 0)` read 32 bits of `x`, and `vdpps(y, x, z, 0x11)`, which
multiplies the first lanes alone, 32 of each. One that takes part of each 128 bits reads as far as the last part it
takes: `vunpcklps` and `vmovddup` the low 64 bits of each, and `vmovsldup` and `vpmuludq` the even lanes, so 64 and 96
bits of an xmm, and 192 and 224 of a ymm. A function of the module is held to it as a kernel is. A call reads each
argument as far as its callee takes it, 32 bits of an `f32` and 64 of an `f64`, so `call(r, scale, (x, n))` of an `f32`
`x`, where `scale` takes an `f64`, is an error; at an address with no functions given, it reads a float as the float,
and any other vector as the call names it, an xmm or all of a ymm, so `x.ymm` of a float is an error there. What a call
gives back is the kernel's own as far as the call would read it as an argument: 32 or 64 bits of a float that its callee
gives back, and at an address with no functions given, a float as the float, and any other vector as the call names it.
`ret(v)` reads `v` as far as the kernel gives it back, 32 bits of an `f32` and 64 of an `f64`, so `ret(x)` of an `f32`
`x`, where the kernel gives back an `f64`, is an error.

Stack memory a kernel declares is a variable, as a GPU kernel's local memory is: `let tables: local<[u32; 1024], 32>;`
is 4 KB of its frame at a multiple of 32, reached by its name, `mov(c.r32, [tables + b * 4])`, and `lea(p, [tables])`
takes its address. Until written, it holds whatever was there. kernite checks an address in it where the index is 0:
what it reaches lies inside, and `vmovaps` and `vmovdqa` reach it at a multiple of what they move; the index is the
kernel's to keep inside. A frame of more than a page is touched a page at a time on the way down, as Rust's functions
are, so that one past the end of the stack faults at the guard page there rather than reach past it: no byte of a frame
lies more than a page below memory the kernel has touched, those below the stack pointer counted. The stack pointer is 8
past a multiple of 16 on entry, as SysV says. A kernel that calls nothing, or whose calls are all jumps, keeps up to 128
bytes of a frame aligned to 16 or less below the stack pointer, in the red zone that SysV lets a function use without
moving the stack pointer, as rustc does of its own. A frame of 128 bytes or less, with the few bytes more that align it,
lies there whole, with no `sub rsp` or `add rsp`, and the slot of a frame of 8 bytes is at `[rsp - 8]`. A bigger one has
`sub rsp` move the stack pointer by what is past 128 bytes, or by the pages the kernel touches where they are more, and
`add rsp` take it back: a frame of 136 bytes has `sub rsp, 8`, its first word at `[rsp - 128]` and its last at `[rsp]`,
and one of 4097 bytes has `sub rsp, 4096` for the page the kernel touches, which leaves one byte below the stack
pointer. Linux, as SysV asks, runs a signal handler below the red zone, which it leaves as it is. On bare metal and
under UEFI an interrupt may write it, and rustc's `x86_64-unknown-none` and `x86_64-unknown-uefi` keep their own
functions out of it. kernite writes one listing for every target, so a kernel whose frame lies in part below the stack
pointer runs on those only where no interrupt writes the stack it runs on. A call writes its return address in the red
zone, so `sub rsp` makes the whole frame of a kernel that calls, aligned to 16 or less, with a few bytes more where a
call or the frame needs the stack pointer at a multiple of 16, and `add rsp` takes it down. Only a frame aligned past 16
keeps the caller's stack pointer in `rbp`: the kernel aligns the stack pointer with `and rsp`, and takes the frame down
by `mov rsp, rbp`.

When the variables alive together do not fit in the registers, kernite spills one to a slot of the frame. It stores it
after an instruction that writes it only where a load of it follows before it is stored again, and loads it before an
instruction that reads it only where no register holds it. It spills, of the variable left without a register and those
holding the ones it could have, the one whose loads and stores are fewest for how long it is alive, one in a loop
counting ten times one outside, and spills again until all fit. The listing names each spill that keeps a slot, in its
header and at each load and store. Sixteen accumulators of `dot` and the load beside them take 17 of the 16 vector
registers, and two spill:

```
# dot::<16>(gpr, gpr, gpr) -> vec
# needs: avx, fma
# frame: 64 bytes aligned to 32, saves []
# acc15, spilled: 32 bytes at rsp
# acc14, spilled: 32 bytes at rsp + 32
...
    vmovups ymm14, ymmword ptr [rdi + 480]
    # acc15 reloaded
    vmovups ymm15, ymmword ptr [rsp]
    vfmadd231ps ymm15, ymm14, ymmword ptr [rsi + 480]
    # acc15 spilled
    vmovups ymmword ptr [rsp], ymm15
```

A store of a spilled variable leaves it in the register it stores from, and a load of it in the register it loads to,
and the kernel reads it there, with no load, until an instruction writes that register, a call changes what it holds of
the variable, or the variable is stored again from another register. A call changes the registers its callee writes, and
the upper half of each vector register if the callee runs `vzeroupper`, which holds no part of an xmm or a float. An
instruction that reads the variable and writes it too takes a `mov` from the register that holds it, and not a load.
Where two ways meet, at a label, a register holds the variable only if each way brings it there. A store is left out if
no way from it comes to a load of the variable before another store of it, and so is a slot that no move uses. After its
loop, `dot::<16>` reads `acc15` in `ymm15`, where the last turn left it, and does not load it.

A spill moves a register with `mov` or `vmovups`, which leave the flags as they were, so it may go between a
comparison and its jump. Only when no spill makes room, as when one instruction names more variables than there are
registers, is it an error, which names the variables at the declaration of the one that found none.

A kernel whose threads wait for each other is `#[cooperative]`, and runs on the threads of a grid, as a GPU kernel runs
on its blocks: each thread runs all of it, and reads where it is as memory, `grid.thread` of `grid.threads`, as in
`mov(t, grid.thread)`. Its `grid.sync()` waits until every thread has come to it, so that what each wrote before, every
one reads after. It is code of the kernel's own, which calls nothing: a thread counts itself in with `lock xadd`, and
the last to come lets the others go, who spin with `pause` until it does. kernite refuses one under an `if`, where a
thread might not come to it.

```rust
#[kernite_macros::kernels]
mod kernels {
    /// `x[i] /= s` for each `i < n`, `s` the sum of `x`, on the threads of a grid.
    #[cooperative]
    pub unsafe extern "sysv64" fn normalize(x: *mut f32, sums: *mut f32, n: usize) {
        let c: gpr;
        mov(c, grid.thread);
        ..
        grid.sync();
        ..
    }
}

let mut sums = vec![0.0; x.len().div_ceil(1024)];
unsafe { kernels::normalize(kernite_cpu::Grid::new(4), x.as_mut_ptr(), sums.as_mut_ptr(), x.len())? };
```

`normalize` becomes the function that launches it, which takes a [`kernite_cpu::Grid`](crates/cpu/src/lib.rs) and the
kernel's parameters, runs the kernel on a thread for each of the grid's, the caller's the first, and returns when all
have ended. A thread that waits spins, so it needs a core: a grid of more threads than the process runs at once,
`kernite_cpu::most()`, is an error, as a GPU's grid of more blocks than it runs at once is.

The grid's other threads are kept. The first launch from a thread starts them, each with the stack Rust gives a thread,
and the launches after it from that thread hand them the kernel and start none. On a Ryzen 5 5600H a launch of a kernel
that returns at once costs 22 ns on one thread, which makes no system call, 0.2 microseconds on two, 0.8 on six and 2.3
on twelve. A kept thread with no kernel to run spins for 200 microseconds, `kernite_cpu::SPIN`, for the next to come,
and then sleeps, using no CPU. A launch that finds the threads asleep costs 5 microseconds on two threads, 9 on six and
16 on twelve, and some 100 when the machine has been idle for milliseconds, as the system wakes its cores. The threads
end when the thread that launched does.

Where a grid's threads run is the system's to say, unless the caller places them. A `Grid::new(4)` runs each thread on
any CPU the caller's thread may run on, and the system moves it from one to another as it likes. A
`kernite_cpu::Grid::on(&[0, 2, 4, 6])` runs thread 0 on CPU 0, thread 1 on CPU 2, thread 2 on CPU 4 and thread 3 on CPU
6, each on its CPU alone, in every launch. On a Ryzen 5 5600H, whose CPUs 0 and 1 are the two threads of one core, that
is a thread a core. `kernite_cpu::cpus()` lists the CPUs to choose from: those the caller's thread may run on, which
`numactl --physcpubind` or `taskset` set for a process when they start it. `kernite_cpu::most()` counts them, or fewer
where a cgroup limits the process to fewer: under `systemd-run --user --scope -p CPUQuota=200%` it is 2, though `cpus()`
lists every CPU the thread may run on. Under `numactl --physcpubind=0-5` it is 6, a grid of 7 threads is an error, and
so is a thread placed on CPU 6:
`thread 0 of the grid is placed on CPU 6, and the caller's thread may run on CPUs 0-5`. So is a CPU given to two
threads, since a thread that waits needs a CPU of its own. When a placement is an error, no kernel runs and no thread
moves.

The caller's thread is the grid's first. The first launch that places it puts it on the first CPU of the grid, and it
stays there, as the grid's other threads stay where they are placed, until a launch places them elsewhere, or places
none, which puts the caller's thread and every thread kept for it back on the CPUs the caller's thread could run on. A
thread the caller starts meanwhile starts on that one CPU, as a thread starts on the CPUs of its creator.
`kernite_cpu::cpus()` from that thread still lists the CPUs it could run on before, for `Grid::on(&cpus()?[..4])` to
give the same grid in launch after launch. If the program moves the caller's thread itself in between, kernite does not
know: a launch that places it on the CPU kernite left it on does not move it.

A launch makes one call to the system for each thread it moves, and none when none moves, so a thousand launches of one
placement ask it nothing about CPUs. The first launch that places the caller's thread, and the first after one that
placed none, also asks once which CPUs the thread may run on. A thread that may run on one CPU alone cannot run while
another spins there, and a spin that no launch ends lasts 200 microseconds, so a launch that moves threads first ends
the spin of each kept thread on a CPU a thread moves to, and of each the system places: it sleeps until a launch uses
it. A launch that moves none ends no spin.

A page of memory is on the NUMA node of the CPU that first writes it, if that node has room, whichever thread asked for
the buffer. `vec![0.0; n]`, for a large `n`, writes no page, so the first thread to write one gives it its node, and
`vec![1.0; n]` writes them all, so the caller's thread gives them its own. A grid's threads are near their memory, then,
if each writes its part of a buffer first, and a `Grid::on` of the same CPUs keeps each by its part in the launches
after. A page is the least that has a node, so a part is near its thread but for the pages it shares with the parts next
to it.

kernite places no memory and sets no policy: to spread a buffer over the nodes, or keep it on one, run the process under
`numactl --interleave=all` or `--membind=1`. The system says which CPUs are on a node, and which are the threads of one
core, in `/sys/devices/system/node/node0/cpulist` and `/sys/devices/system/cpu/cpu0/topology/thread_siblings_list`, for
a program to read and give to `Grid::on`. The grid's barrier is one cache line for the whole grid, whichever nodes its
threads are on, so a `grid.sync()` over threads on two nodes passes that line between them.

The kernel's parameters keep the names they are written with, but `_`, which the macros name for Rust. The launcher
names its grid `__kernite_grid`, and a compile-time argument may not be named so, for Rust cannot keep a const apart
from a parameter of its name: that is an error at its name. A crate with a cooperative kernel needs a dependency named
`kernite-cpu` for the grid, not renamed in `Cargo.toml`. kernite says at the kernel when it has none.

A kernel calls the module's others, itself among them, as SysV calls a function: `call(g, gcd, (a, b))` moves `a` to
`rdi` and `b` to `rsi`, and `g` from `rax`, and is checked against what `gcd` takes and returns. A move to a register
that holds the value already, on every way to it, a jump back among them, is left out. So is the kernel's own move of
all of a variable, as wide as kernite moves it, which a paragraph below tells: `mov` of a gpr, and `vmovaps`, `vmovapd`,
`vmovups`, `vmovupd`, `vmovdqa` or `vmovdqu` of a vector. Its move of a variable to itself is left out where it moves
all of the register, a gpr's 64 bits or a ymm's 256, and stays where it moves less, which may change the rest:
`mov(x.r32, x.r32)` clears the upper half of the gpr, and `vmovaps(x, x)` of a float the upper lanes of its ymm. A value
that goes to two registers goes to each from the register that held it first, where the variable it comes from is read
after the move, so that no move waits for another. A value that a call takes in the register that holds it stays there,
though the call takes it again: `x` in `rdi` stays there for `call(r, gcd, (x, x))`, which moves it to `rsi`. One alive
across the call stays there through it, if the call does not change that register. `lea(fp, [gcd])` takes an address to
call at, `call(g, fp, (a, b), (gcd, sum))`, checked against each function given. A value alive across a call is kept in
a register the call does not change. A call changes the registers its callee writes and does not restore, and those of
each function the callee calls, by name or at an address with the functions it may reach given. kernite cannot check
that the address is that of one of them, and trusts it: a value alive across the call is kept in a register none of them
changes, which another function may change. So `gcds`, which calls `gcd` in a loop, keeps its values in registers `gcd`
does not write, and saves none for its own caller. A call at an address with no functions given may reach any function,
and one that comes back to the function it is in, as `gcd`'s call of `gcd`, reaches code that is not compiled yet: each
changes every register SysV lets a callee change, so an integer alive across it is kept in `rbx` or `r12` to `r15`,
which every callee leaves as it was, and which the kernel saves for its own caller, and a vector spills. A vector alive
across any other call stays in a register none of the functions it may reach writes, as an integer does. But a function
that runs `vzeroupper` clears the upper half of every vector register, and leaves the lower half: a vector of which an
instruction after the call reads all, as one on all of a ymm does, spills, and one of which the kernel reads the lower
half alone, an xmm or a float, stays in its register. So does one whose lower half the kernel writes before it reads all
of it: an instruction that writes an xmm or a float clears the upper half of its ymm, even one that reads it too, as an
FMA does. A vector that an address takes its indices from counts as read whole, though a gather of an xmm of indices
reads its lower half alone. The stack pointer is a multiple of 16 at each call. A kernel writes `vzeroupper` where it
may have set the upper half of a vector register: before it returns, so that the SSE code of its caller does not pay for
it, and before it calls a function that may run SSE code, so that the SSE code of the callee does not. It may have
where, on some way to it, the kernel runs an instruction on all of a ymm, or calls at an address with no functions given
and passes all of a ymm or takes all of one back, with no `vzeroupper` since. A call at an address with no functions
given may run SSE code: the kernel does not know its callee. So may a call that comes back to the function it is in
through another of the module's: it reaches code that is not compiled yet. So may a function of the module that makes
such a call, or calls one of the module's that may, unless the call passes all of a ymm or takes all of one back: its
callee runs AVX instructions, and `vzeroupper` would clear the upper half of a ymm it passes. A call of the function the
kernel is in may where one of the kernel's other calls may. The instructions of the module's other functions are AVX's,
which pay nothing for an upper half set, and the kernel clears it before it returns. kernite moves and spills a vector
register as wide as an instruction uses its variable: as an xmm, `vmovaps xmm0, xmm1`, where none uses it as all of a
ymm, and else as all of a ymm, `vmovaps ymm0, ymm1`. A call of the module's functions, by name or at an address with
them given, passes a float and takes one back as an xmm: `call(v.xmm, half, (x))` writes `v` as an instruction that
writes an xmm does, though only the float of it is the kernel's own. A call at an address with no functions given passes
and takes back each vector as the call names it: an xmm, or all of a ymm. A float, an `xmm` or the `.xmm` of a ymm says
that its callee takes or gives back no ymm, and all of a ymm that it does, as the type of a function pointer does in
Rust. So a kernel of xmm registers alone writes no `vzeroupper`, though it moves and spills them and calls functions, by
name or at an address: each clears what it sets before it returns. A call that `ret` follows at once is a jump: the
kernel takes its frame down and writes `vzeroupper` as `ret` does, and jumps to the callee, which returns to the
kernel's caller. So `call(g, gcd, (a, b)); ret(g);` is a loop in `gcd`: the stack does not grow. A label between the
call and the `ret` leaves the `ret` for the other ways that come to it, and an address to call at in a register the
kernel saves goes first to `r11`, which taking the frame down leaves as it is. The call stays a call where an
instruction comes between it and the `ret`, `call(r, gcd, (a, b)); add(r, 1); ret(r);`. It stays one too where the jump
would not do what the call does: where the kernel has taken the address of its stack memory, which the callee may use,
and the frame the jump takes down holds; and where it calls at an address with no functions given and takes all of a ymm
back: the kernel's `vzeroupper`, as it takes its frame down, runs before the callee sets the upper half of what it gives
back. A jump that passes all of a ymm has no `vzeroupper` before it: the callee, passed all of a ymm, clears the upper
halves before it returns. A kernel that never returns, `-> !`, ends each way out in `ud2()` or in a call of one that
does not.

Memory of the module's is a `static` beside its kernels, in the process's memory, which each kernel names as it names
its own, and which keeps what they write from one call to the next: `static COUNTS: global<[u32; 16]>;` is 16 words the
kernels read and write, and `static WEIGHTS: constant<[f32; 4]> = [1.0, 0.5, 0.25, 2.0];` four floats they only read,
holding what it says until something writes it. A kernel reaches a static at a displacement from `rip`,
`vbroadcastf128(w, [WEIGHTS])`, and `lea(p, [COUNTS])` takes its address, for an index, `[p + i * 4]`. kernite checks an
address in it as it does one in stack memory, and refuses a write to constant memory. The host reads and writes each as
the module has it, a [`kernite_cpu::Global`](crates/cpu/src/global.rs) of Rust's type for its elements:
`kernels::COUNTS.read()`, `kernels::WEIGHTS.write(&[2.0; 4])`. Its elements are atomics, which the host loads and
stores, and a kernel's safety contract says that nothing else touches the memory of the module's it names while it runs,
nor any that the functions it calls name, whether it calls them by name or at an address. A module's statics are its
kernels': on the GPU for GPU kernels, so a module with kernels of both kinds and a static is an error, which says to
make it two. A crate with a static in a module of x86 kernels needs a dependency named `kernite-cpu` for it, not renamed
in `Cargo.toml`. kernite says at the static when it has none. A crate of the `dylib` kind may have such a static, which
it exports as it exports any: a program that links the crate names it as it names one of its own,
`my_lib::kernels::COUNTS`.

A static may be as big as a Rust type may be: its size, rounded up to its alignment, is less than 2^61 bytes. A kernel
names a static at a displacement from `rip`, 32 bits, signed, which reaches 2 GiB. The rest of the program's code and
data take room in those 2 GiB too, for the code reaches its data so, and the linker lays the statics out among that
data: so the statics of a module are in reach of `rip` up to 1073741824 bytes together, half of it, and a kernel names
each at a displacement. A static is in reach where it and the statics in reach before it in its module, each with the
padding its alignment may need, one byte less than it, are 1073741824 bytes at most:
`static A: global<[u8; 1073741824]>;` is in reach, and a second of 1 GiB after it is not. Every static counts, whatever
`cfg` keeps, so a static is in reach or not whichever features are on. The statics of other modules and the rest of the
program are the linker's to fit: it refuses a program where they do not fit in the 2 GiB. A static that is not in reach
is far, as one of more than 1073741824 bytes is, and a kernel names it only by its address: `lea(p, [BIG])` puts the
address in `p`, and `[p + i]` is the byte `i`. `lea(p, [BIG + 16])` is the address of the byte 16, for a displacement up
to 2147483647; a kernel indexes past that. Any other operand that names the static is an error, which says to take its
address. A far static takes none of the room in reach, so one after it may be in reach. It holds zeros at first, so a
value is an error, which says the host writes what it should hold.

A far static is the linker's, in the `.lbss` section of x86-64's ELF, which the linker lays out past the code and the
rest of the data, so that they stay in reach of `rip`. The section holds zeros and takes no room in the file, where Rust
makes a static of its own in time and memory that grow with its size, 6 milliseconds and 2 MiB for each MiB: a crate
with a static of 3 GiB builds as fast as a crate with one of 4 KiB. kernite defines it so on Linux alone, and refuses it
on another system, at its name. To Rust the static is a reference to that memory, a `&'static Global<..>`, which the
host uses as it uses a `Global`: `kernels::BIG.len()`. A kernel loads the address from it, as code for a
position-independent executable loads one from a slot. A crate of the `dylib` kind exports the reference, and keeps the
memory to itself, so a program that links the crate reaches the memory through it. The memory's symbol is the path of
its module and its name, as `module_path!` gives them, so two such statics of one path and one name in a program are one
symbol defined twice, which the assembler or the linker refuses: two modules of one name in two functions have one path,
and so do two versions of one crate.

## A safe kernel

An x86 kernel that is no `unsafe fn` is one kernite proves safe where it compiles it, and Rust calls it as it calls any
function, with no `unsafe`. It takes numbers, arrays by reference, `&[f32; 8]` for an array it reads and `&mut [f32; 8]`
for one it writes, and slices, `&[f32]` and `&mut [f32]`, and gives back a number, a pointer or nothing. An array holds
Rust's numbers, as stack memory does, and is 2147483647 bytes at most, for a kernel reaches each element at a
displacement from the array's address, of 32 bits, signed; its length may be a compile-time argument, `&[f32; N]`, or a
const. Rust says an array is as long as its type says, and kernite checks each address the kernel reaches in one as it
checks one in stack memory: inside the array, at a displacement known now, and at the alignment its instruction needs.
So nothing is checked when the kernel runs, and there is nothing for the caller to promise. The code of a safe kernel is
that of the same kernel written with `unsafe` and a pointer for each array, and a pointer and a count for each slice:
kernite adds none.

```rust
#[kernite_macros::kernels]
mod kernels {
    /// The sum of the sixteen floats at `xs`: `sum8` of each half, added.
    pub extern "sysv64" fn sum16(xs: &[f32; 16]) -> f32 {
        let (low, high): (ymm, ymm);
        call(low.xmm, sum8, [xs]);
        call(high.xmm, sum8, [xs + 32]);
        vaddss(high.xmm, low.xmm, high.xmm);
        ret(high.xmm);
    }
}
```

A call passes an array of the callee's type, or a part of one: `[xs]` is `xs`, and `[xs + 32]` the array 32 bytes on,
which must lie inside `xs`, at a multiple of the callee's element size. The callee's length may be a compile-time
argument or a const as well: `sum::<4>` takes a `&[u64; 4]` and calls `sum::<2>` on each half. A write through `&[T; N]`
is an error, as one to constant memory is, and an array is passed to a callee that writes only from `&mut [T; N]`. A
kernel may call itself and the safe kernels of its module, and so reaches no memory but the arrays and slices it is
given and its own frames, which it touches a page at a time, so that an overflow of the stack faults at the guard page,
as above.

A slice is two registers: its address, in one, and its length in elements, in the next, which the kernel names `xs.len`.
It takes two of the six registers SysV passes integers in, so a kernel takes six integers in all, slices and numbers
together, and the function the macro writes takes the slice and passes its address and its length. A slice holds Rust's
numbers, as an array does, and has any length, none too. It is at a multiple of its element's bytes, as Rust lays it
out, and kernite takes it to be at no more: `vmovaps` of a `ymm`, which needs a multiple of 32, is an error in a slice
of floats, where `vmovups` reads at any. kernite does not know the length, so it proves each address the kernel reaches
in the slice from the compares the kernel makes of an index with `xs.len`:

```rust
#[kernite_macros::kernels]
mod kernels {
    /// The sum of the integers of `xs`.
    pub extern "sysv64" fn sum(xs: &[u64]) -> u64 {
        let (i, sum): (gpr, gpr);
        zero(i);
        zero(sum);
        label(top);
        cmp(i, xs.len);
        jae(done);
        add(sum, [xs + i * 8]);
        inc(i);
        jmp(top);
        label(done);
        ret(sum);
    }
}
```

The index is a gpr, scaled by the bytes of an element, `[xs + i * 8]`, with a displacement known now beside it if the
kernel wants, `[xs + i * 4 + 32]`. The address is inside the slice where, on every way to it, the compares have kept
`i + n` at or below `xs.len`, `n` the elements the instruction reaches from the index: one for the `add` above, eight
for a `vaddps` of a `ymm` of floats, so `lea(j, [i + 8])`, `cmp(j, xs.len)` and `ja(done)` keep eight. The displacement
counts in, and the last element it reaches is rounded up. An address with no index, `[xs + 4]`, is inside where `xs.len`
has been compared with 2 or more. None is before the start, which the index at 0 would reach: `[xs + i * 4 - 4]` is an
error, as it is in an array.

A compare is a `cmp` of a gpr with another or with a number known now, or a `test` of a gpr with itself, each gpr whole,
that a conditional jump or a `cmov` reads, with nothing between them but `mov`, `lea`, jumps and `cmov`s, which leave
the flags as they are. kernite follows both ways of the jump, and holds each way to what the compare says of the numbers
as unsigned ones: `jb`, `jae`, `ja`, `jbe`, `je` and `jne`, and their other names, say what they say, and a signed
condition, `jl` say, says nothing. It follows the numbers the kernel keeps in whole gprs: `mov` of a gpr or of a number
known now, `lea` of a gpr and a displacement, `add` and `sub` of a number known now, `inc`, `dec`, `zero` and `xor` of a
gpr with itself write a number it follows, and so does a `cmov` of a gpr, which is the one or the other as the compare
says: `mov(n, xs.len)` copies the length, and `cmp(n, ys.len)` then `cmova(n, ys.len)` make `n` the shorter of two. A
number below a length is far below 2^64, for Rust has no slice longer than `isize::MAX` bytes, so adding to it does not
wrap. Any other instruction that writes a gpr, or a part of one, and an `add`, a `sub`, an `inc`, a `dec` or a `lea`
that may wrap, makes it a number of any value until a compare says more. A loop has what holds on each way round it.

The length stays as the kernel was given it, for the compares are of it: `mov(n, xs.len)` copies it to a variable that
may change, and an instruction that writes `xs.len` is an error. A kernel passes no slice to a function it calls: the
callee takes the address and the length as two numbers, `*const f32` and `n: usize`, and is an `unsafe fn`, as is a
`#[cooperative]` kernel, which takes no slice but the two numbers.

What kernite cannot prove it refuses, at the first parameter, operand or call it cannot, and the error says that the
kernel is then an `unsafe fn`, which compiles as it always has, and which Rust calls in an `unsafe` block. A kernel is
an `unsafe fn` when it:

- takes a raw pointer, `*const f32`, which may point anywhere;
- reaches an array at an index in a register, `[xs + i * 4]`, or at an address it has taken, `lea(p, [xs])`, or at an
  address in a register, where kernite cannot keep it inside the array;
- reaches a slice at an index that no compare on every way to the address keeps below the length, at a vector of
  indices, or at an address it has taken, `lea(p, [xs])`, or writes the length of a slice, `dec(xs.len)`, which the
  compares rest on;
- reaches a bit of an array or a slice at an offset in a register, `bts([xs], i)`, which may lie anywhere in memory;
- makes a non-temporal store, `movnti([xs], g)`, which goes around the caches, and may return before an `sfence` or
  `mfence` orders it: Rust holds the store unfinished until one does, and what the caller does next races with it;
- names a static of the module's, which every kernel and the host reach;
- calls a kernel that is an `unsafe fn`, or calls at an address, which reaches a function kernite cannot name;
- gives a `bool` back, which must be 0 or 1, and kernite cannot prove that it is;
- runs on a grid of threads, `#[cooperative]`, whose threads reach one memory at once.

The code of a kernel may need CPU features past x86-64's own, as `sum8`'s `vmovups` of a `ymm` needs `avx`, and so does
a kernel that calls one that does. Rust checks them where the kernel is called: the function it calls is
`#[target_feature(enable = "avx")]`, so only a function that has `avx` the same way calls it with no `unsafe`, and any
other calls it in an `unsafe` block, which says why the CPU has `avx`. The features the build is configured with,
`-C target-cpu=x86-64-v3`, do not count: Rust asks for the attribute. Nothing is checked when the kernel is called: Rust
has seen that the caller has the features, or the caller says in an `unsafe` block that the CPU does. A kernel that is
an `unsafe fn` is the same as it was: what it needs is for its caller to know. The kernel's `# Safety` names the
features: the macros write one where its docs have none, and add to the one they have. Where they write it, a safe
kernel's says nothing else, and an `unsafe fn`'s says first that the kernel trusts the pointers it is given. Clippy asks
any public function that is `#[target_feature]` for a `# Safety`, as it asks an `unsafe fn`, and tells of no such
kernel, as each has one.

A crate may `forbid(unsafe_code)` and have safe kernels, and call them: the Rust the macro writes for one is no
`unsafe` of the crate's.

## On a GPU

An `extern "ptx-kernel"` function is compiled to PTX. The module's kernels are one PTX module, `PTX`, which
`Kernels::load` hands to the CUDA driver; each kernel is then a method that launches it on a grid.
[`kernite-cuda`](crates/cuda/src/lib.rs) is the driver, opened when a GPU is asked for, so that a program built with
GPU kernels still runs where there is none. A crate with a module of GPU kernels needs a dependency named
`kernite-cuda`, not renamed in `Cargo.toml`, for the GPU items that `cfg` keeps: kernite says at the module when it has
none and one of them is kept. A feature can bring the crate and keep them, `cuda = ["dep:kernite-cuda"]`, and the crate
builds without the feature. A crate with a static in such a module needs a dependency named `kernite-macros` as well,
not renamed, for Rust to count the room of the statics `cfg` keeps: kernite says at the module when it has none and a
static is kept. A crate with a module of kernels given consts, of either kind, needs a dependency named
`kernite-macros` too, not renamed, for the module of consts to hand them on: kernite says at the module of kernels when
it has none and `cfg` keeps it.

```rust
#[kernite_macros::kernels]
mod gpu {
    /// `y[i] = a * x[i] + y[i]` for each `i < n`, rounded once.
    pub extern "ptx-kernel" fn saxpy(a: f32, x: *const f32, y: *mut f32, n: u32) {
        let (block, size, thread, i): (u32, u32, u32, u32);
        mov.u32(block, ctaid.x);
        mov.u32(size, ntid.x);
        mov.u32(thread, tid.x);
        mad.lo.u32(i, block, size, thread);
        let past: pred;
        setp.ge.u32(past, i, n);
        if past {
            ret();
        }
        ..
    }
}

let cuda = kernite_cuda::Cuda::new()?;
let kernels = gpu::Kernels::load(&cuda)?;
let (xs, mut ys) = (cuda.upload(&x)?, cuda.upload(&y)?);
kernels.saxpy(kernite_cuda::Grid::new(n.div_ceil(256), 256), 2.0, &xs, &mut ys, n)?;
ys.read_into(&mut y)?;
```

`saxpy` is no `unsafe fn`: kernite proves that each thread reads and writes inside `x` and `y`, and writes only its own
element of `y`, as "A safe GPU kernel", below, says. Its launcher takes the memory, `&xs` to read and `&mut ys` to
write, and Rust calls it with no `unsafe`.

A machine may have several GPUs. The GPUs are those the driver shows the program: all of the machine's, unless
`CUDA_VISIBLE_DEVICES` chooses which, and the driver numbers them from 0 in its order. `Cuda::count()` says how many
there are, and `Cuda::on(i)` takes GPU `i`; `Cuda::new()` is `Cuda::on(0)`. `Kernels::load(&cuda)` loads the kernels on
the GPU of `cuda`, and a buffer is on the GPU of the `Cuda` that made it.

An index past the last GPU is an error that says how many there are:
`cuDeviceGet: CUDA_ERROR_INVALID_DEVICE (101): invalid device ordinal: the driver shows 1 GPU, and none is numbered 1`.
Where the driver shows none, because the machine has no GPU or `CUDA_VISIBLE_DEVICES` chooses none, `Cuda::count()` is 0
and `Cuda::new()` is the error `cuInit` gives: `cuInit: CUDA_ERROR_NO_DEVICE (100): no CUDA-capable device is detected`.
Where there is no driver, or `cuInit` fails any other way, `Cuda::count()` is an error, as `Cuda::new()` is.

A `Cuda`, and the buffers and `Kernels` made on it, are for the thread that made it. A thread takes the GPU it uses with
`Cuda::on`, and threads may take the same GPU. One thread may use the `Cuda` of each GPU in turn: each call, a launch, a
copy or `sync`, goes to the GPU of the `Cuda`, the buffer or the `Kernels` it is made on, whichever GPU the call before
it went to. A kernel is given the buffers of the GPU it runs on. The driver does a call's work in the context current on
the thread, so kernite-cuda keeps which GPU's it made current last. A call to that GPU asks the driver for nothing it
did not ask before, and a call to another GPU asks it once more, to make that GPU's current, `cuCtxSetCurrent`.

`to.copy_from(&from)` copies a buffer to one on another GPU, after the kernels launched before it on either GPU have
run, and before those launched after it on either run. Where the two GPUs have a link between them, it asks the driver
for peer access and the copy goes over the link; where they have none, the driver sends it through the host.

A buffer is the GPU's own memory, which the host reaches by copies. Unified memory is reached by the host and by the
GPU's kernels at one address, with no copy, and it is of one of three kinds. `cuda.managed::<f32>(len)` is `len` zeros
of managed memory, which the driver moves to whichever of the host and the GPU touches it. `cuda.pinned::<f32>(len)` is
`len` zeros of page-locked memory, which stays in the host's memory, and a kernel reads and writes it across the bus.
`cuda.lend(&mut xs)` lends a slice of the host's own memory to the GPU, a `Vec`'s or a `Box`'s: a kernel reads and
writes its elements where they are, and the host has the slice back when it drops what `lend` returned. A kernel is
given `memory.ptr()`, and the host reads and writes `memory.host()` and `memory.host_mut()`, which are slices of it. The
slices are at the address `ptr` gives, so a kernel for the CPU that is given a slice's pointer works on the memory that
a kernel for the GPU is given: [`examples/my-app`](examples/my-app/src/main.rs) makes managed memory with
`managed_uninit`, fills it with an x86 kernel, squares it with a PTX kernel and adds it up with an x86 kernel, bit for
bit, with no copy.

The zeros cost. The GPU writes the zeros of managed memory, so the driver has each page at the GPU when the host first
writes it, and moves it back; the host writes the zeros of page-locked memory in a pass of its own, before its fill.
Memory that the host or a kernel writes whole before anything reads it needs none: `cuda.managed_uninit::<f32>(len)` and
`cuda.pinned_uninit::<f32>(len)` leave the bytes as the driver does, and cost what `cuMemAllocManaged` and
`cuMemHostAlloc` do. They are `unsafe`, because the bytes may be those of memory freed before, which the driver hands
out again where a buffer beside it holds the page: the host or a kernel writes each element before anything reads it.
Memory that is read before it is all written is made with `managed` and `pinned`, which are safe.

A GPU does not do all three, so each asks the GPU first. Where the GPU cannot, the error says so in these words, and no
kernel is left to fault: `cuMemAllocManaged: the GPU cannot allocate managed memory: its
CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY is 0` for `managed` and `managed_uninit`; `cuMemHostAlloc: the GPU cannot reach
page-locked memory at the host's addresses: its CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING is 0` for `pinned` and
`pinned_uninit`; and `cuDeviceGetAttribute: the GPU cannot reach pageable memory, a slice of the host's: its
CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS is 0` for `lend`, which a GPU with HMM can do. Each error names the call it
refuses, as `Error::call` does; `lend` makes no call of the driver's to refuse, so its error names
`cuDeviceGetAttribute`, which told that the GPU cannot.

The driver here, NVIDIA's open kernel module 615.71.10 on Linux 7.0, faults a kernel on a lent element that has a guard
page in its aligned 64 KiB, a page set with `madvise(MADV_GUARD_INSTALL)`: the kernel fails with
`CUDA_ERROR_ILLEGAL_ADDRESS (700)`, as does every call after it. glibc 2.43 sets such a page below each thread's
stack. So `lend` the memory of a `Vec`, a `Box` or a mapping, and not an array on a thread's stack.

`host()` and `host_mut()` wait for every kernel launched on the GPU to finish, and say whether one failed, before they
lend the slice, so a slice taken after a launch is never under its kernel, and the host reads what a kernel wrote with
no `sync` of its own. The compiler refuses to take a pointer for a kernel while the host holds a slice it may write, as
`ptr` borrows the memory and `host_mut` borrows it mutably, and it refuses the host a touch of a slice that is lent, as
`lend` borrows the slice. Dropping the memory waits for the kernels that use it, as dropping a buffer does, and then
frees it, or gives a slice lent back to the host.

A pointer taken before the slice is `Copy`, and the compiler cannot refuse a launch on it, so the launch of a kernel
that is an `unsafe fn`, which takes pointers, is `unsafe`, and its contract is this: each pointer is to a buffer or to
unified memory of the GPU, as long as the kernel reads and writes through it; from the launch until the kernel is done,
the host holds no slice of memory the kernel writes, and no slice it may write of memory the kernel reads; and where
`cuda.concurrent_managed_access()` is false, the GPU not reaching managed memory while the host does, the host holds no
slice of managed memory at all. The method is the GPU's `CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS`, and a GPU where
it is false faults the host on managed memory while any kernel runs. `Module::launch` says the contract in its
`# Safety`, and the macros write it as the `# Safety` of the launcher of a GPU kernel that is an `unsafe fn` and has
none. The launcher of a kernel that is no `unsafe fn` takes the memory, not a pointer, and has no contract for its
caller to keep: "A safe GPU kernel", below, says what it checks.

Managed memory is moved ahead of its use, and advised how it is used, over a range of its elements; a slice lent is
advised too, but not moved. `memory.prefetch(0..len, Location::Gpu)` moves them to the GPU, ahead of the kernels that
use them, on the stream the kernels are launched on; the places are `Gpu`, `Host`, `Numa(node)` and `NumaOfThread`.
`memory.advise(range, advice)` tells the driver that the range is `ReadMostly` or `NotReadMostly`, that it has a
`PreferredLocation(place)` or `NoPreferredLocation`, or that the GPU or the host is `AccessedBy` it or `NotAccessedBy`
it, which the driver may follow or not. `memory.placement(range)` says what the driver has of both: `read_mostly`,
`preferred`, `accessed_by` and `last_prefetch`, each only where every page of the range agrees, and `None` or empty
where they do not. A prefetch, and the advice that names the GPU, need the GPU to reach managed memory while the host
does, and where it cannot the error says so in these words: `cuMemPrefetchAsync: the GPU cannot reach managed memory
while the host does: its CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS is 0`, or `cuMemAdvise:` and the same for the
advice.

`prefetch`, `advise` and `placement` need a driver of CUDA 12.2 (535) or later, which brought `cuMemPrefetchAsync_v2`,
`cuMemAdvise_v2` and the attributes of the preferred place and of the last prefetch. `Cuda::new()` opens an older driver
all the same, and each of the three says so when it is called, in the words of the loader: ``dlsym: the driver has no
`cuMemPrefetchAsync_v2`: libcuda.so.1: undefined symbol: cuMemPrefetchAsync_v2`` for `prefetch`, and the same of
`cuMemAdvise_v2` for `advise` and `placement`.

The driver prefetches managed memory alone, and tells the place of it alone, so `prefetch` and `placement` refuse
page-locked memory and a slice lent, and say why: `cuMemPrefetchAsync: the driver prefetches managed memory alone: this
is page-locked memory`, or `this is a slice lent`, and `cuMemRangeGetAttribute: the driver tells the place of managed
memory alone: this is a slice lent`. It advises managed memory, and pageable memory where the GPU reaches it, which
`lend` asked, so `advise` takes a slice lent as it takes managed memory, and refuses page-locked memory: `cuMemAdvise:
the driver advises managed and pageable memory alone: this is page-locked memory`.

A NUMA node does not touch memory, so it is refused as an accessor: `cuMemAdvise: only the GPU and the host access
memory, not a NUMA node`; and the driver refuses a node the host does not have: `cuMemPrefetchAsync:
CUDA_ERROR_INVALID_VALUE (1): invalid argument`. A range with no elements, or one past the end, is refused before the
driver is asked: `cuMemPrefetchAsync: the range 2..2 has no elements`, or `the range 3..5 goes past the end of the 4
elements`; `advise` and `placement` say it with their own calls.

The driver here, NVIDIA's open kernel module 615.71.10 on Linux 7.0, takes a NUMA node as the preferred location,
`PreferredLocation(Numa(node))` or `PreferredLocation(NumaOfThread)`, and changes nothing: the memory prefers what it
did before, the GPU, the host or no place, and `preferred` says so. CUDA's header says the driver sets the node and says
it back as a NUMA node, which this one does not.

The driver works on whole pages, 4 KiB. A range reaches all of each page it touches, the rest of the page too, whether
it is the memory of the same buffer or of another. A managed buffer starts at a multiple of 512 bytes, not at a page, so
it shares the pages at its ends with the buffers beside it, and a range from a page to a page reaches just its own.

The driver here, NVIDIA's open kernel module 615.71.10 on Linux 7.0, does not always clear the advice and the last
prefetch of a buffer it frees, when threads use it at once: a buffer made at that address may start with them, and
`placement` says so. When one thread uses it, it clears them. They are hints, so no value changes.

`saxpy` becomes the method that launches it, which takes the grid and then the kernel's parameters. They keep the names
they are written with, but `_`, which the macros name for Rust. The method names the grid `__kernite_grid`, and a
compile-time argument may not be named so, nor, on a cooperative kernel, `__kernite_threads` or `__kernite_shared`,
which are parameters of the method that says how many blocks its grid has at most, below. Rust cannot keep a const apart
from a parameter of its name: that is an error at its name.

A grid has as many blocks and threads as the GPU takes: at least 1 along each of x, y and z, and, from Kepler's `sm_30`
on, at most 2^31 - 1 blocks along x, 65535 along y and 65535 along z, and 1024 threads in a block, at most 1024 of them
along x, 1024 along y and 64 along z. A grid past one of these is an error that says which size is past which limit:
`cuLaunchKernel: CUDA_ERROR_INVALID_VALUE (1): invalid argument: the grid has 65536 blocks along y, and the GPU takes
65535 at most`, or, for a grid of none, `... the grid has 0 blocks along x, and the GPU takes 1 at least`. The driver
says no more than `CUDA_ERROR_INVALID_VALUE` of any of them, and kernite adds its sentence to that error alone: another,
such as `CUDA_ERROR_ILLEGAL_ADDRESS` from a kernel before it, is said as the driver says it.

A block has shared memory of the kernel's own, which it declares, `let tile: shared<[f32; 256], 16>;`, at most 48 KB,
and of the launch's beside it, `with_shared(bytes)`. Together they may be all the GPU gives a block, which
`Cuda::max_shared` says: 99 KB on an RTX 30, so that a launch gives a kernel of 20 KB of its own 79 KB. A block of more
is an error that says how much it has: `cuFuncSetAttribute: CUDA_ERROR_INVALID_VALUE (1): invalid argument: each block
has 102400 bytes of shared memory, 20480 of the kernel's and 81920 of the launch's, and the GPU takes 101376 at most`.
A cooperative launch says the same, and so does the method that says how many blocks its grid has at most, below. A
grid past more than one limit is told of the first of these: the blocks along x, y and z, the threads along x, y and z,
the threads in all, the shared memory, and last the kernel's `.maxntid` or `.reqntid`, below.

The PTX target knows every instruction of PTX's for the GPUs from Fermi, `sm_20`, to Ampere, `sm_86`, in each form
`ptxas` takes, [`forms`](crates/ptx/src/forms.rs), but the forms PTX deprecated, below, and one.
`txq.force_unnormalized_coords` asks of a sampler apart
from its texture, in the independent mode, which kernite does not write: kernite writes no `.samplerref`, so the ISA
leaves the answer undefined. kernite checks each operand against the forms, and a `call`'s against what the function it
calls takes and returns. The operands of a vector, `(a, b)`, are of one size, and of types that go together, as `ptxas`
takes them: the same type, bits beside any, or an integer beside an integer. A value known now goes by its kind: an
integer beside bits and integers, `(i, 1)`, and a float beside bits, an `f32` and an `f64`, `(f, 1.5)`. An `f16` or a
`bf16` is written as its bits, and goes as an integer. In a vector of bits, a float has the size it is written in, 32
bits for `1.5` and 64 for `1.5f64`, and goes beside registers of that size. In a vector of floats, a value known now is
written as a float of the vector's type, and goes beside any operand the vector takes: `(f, 1)`, and `(q, 1.5)`, a `b64`
beside an `f32`, in `st.global.v2.f32`. A special register goes beside a register of any type of its size, `(f, tid.x)`,
as `ptxas` checks its size alone, and beside a value known now only as a whole one, `(laneid, 1)`: a part of a vector's,
as `tid.x` is of `tid`, takes none, `(tid.x, 1)`. A special register of more bits than the operands of the vector, as
`gridid` is, 64 bits in a vector of 32, `(gridid, 1)`, is an error in `mov`, `mma`, `wmma`, the coordinates of a
surface, and the coordinates and the offsets of a texture, as `ptxas` refuses it: `mov.u32(n, gridid)` puts 32 of its
bits in a variable. `ptxas` takes it after a value known now, `(1, gridid)`, in `mov`, in the accumulator of `mma`, and
in coordinates and offsets, as it checks each operand against the one before it alone, which looks like a bug of its
own: kernite refuses it there too. Any special register goes where a `cvt` to an integer reads an integer source, and
`st` and `sust` the data of their vector, as an integer, or bits, of no more bits than it has: `cvt.u32.u8(i, laneid)`,
`(laneid, laneid)` in a vector of 16 bits, and `(clock64, clock64)` in one of 32. An integer, or bits, of more bits is
an error, `cvt.u64.u64(u, laneid)` and `(laneid, laneid)` in a vector of 64, as `ptxas` refuses it. A special register
goes in a vector of floats too, by its size, as bits: `(f, tid.x)` and `(laneid, laneid)` in `st.global.v2.f32`,
`(d, gridid)` in `st.global.v2.f64`, and `[s, (f, tid.x)]` in the coordinates of `tex` and `tld4`, which take one of as
many bits as the float, and `st` one of more too, `(gridid, gridid)` in `st.global.v2.f32`. In a vector of floats,
`ptxas` takes one of bits the vector does not take after a float known now, `(1.5, gridid)` in the coordinates of `tex`
and `(1.5f64, laneid)` in `st.global.v2.f64`, as it takes `(1, gridid)` in a vector of 32 bits, and refuses it before
one: kernite refuses it in both. In a vector of floats, `ptxas` takes an integer variable of the float's size too,
beside no float: `(i, i)` as the data of `st.global.v2.f32`, as the result of `ld` and `ldu`, and in the coordinates and
the result of `tex` and `tld4`. It checks the operands of a vector against each other, and not against the instruction's
type, and refuses one alone, in `st.global.f32`, and one beside a float, `(f, i)`. That looks like a bug of its own:
kernite refuses it in every place. An `f32` beside a `u32`, or a `u16` beside a `u32`, is an error at the operand that
does not go with one before it, in `st`, `ld`, `ldu`, `mov`, `suld.b`, `sust.b`, `sust.p`, `ldmatrix` and `mma`, as
`ptxas` refuses it. So is an `f32` beside an integer known now in a vector of bits, `(f, 1)`, a `u64` beside `tid.x`, or
`tid.x` beside a value, in a vector that takes values or special registers. The layer of an array of textures, the first
of its vector, is apart: an integer beside floats. `ptxas` checks each operand against the one before it alone, and
takes `(f, b, i, i)`, an `f32` or an `f16x2` and a `u32` with a `b32` between, `(f, b, 1, 1)` and `(f, tid.x, i, i)`,
which looks like a bug of its own: kernite checks each against each before it, and refuses it. In a vector of bits,
`ptxas` refuses a float known now first beside a register of 64 bits, `(1.5, d)`, and takes it elsewhere, `(d, 1.5)` and
`(d, d, 1.5, d)`: kernite refuses it in every place. PTX keeps a `bf16`, `bf16x2` or `tf32` variable in a `b16` or `b32`
register, and kernite takes it wherever it takes one, as `ptxas` does: for an integer or a float of its size, in an
instruction, beside an `f16x2` in a vector, as an address, as an argument of a helper and of a call, as the address a
call calls at, as the result of a call, and in `ret`. `add.u32(i, b, i)` and `add.rn.f32(v, b, v)` write `b` as the
operand, with no `mov` beside it. `fma.rn.bf16x2(b, x, b, b)` takes a `tf32` or a `b32` for `x`, and no `u32`, `f32` or
`f16x2`, as `ptxas` takes none. Where `ptxas` takes more than PTX's ISA has, and reads it as the ISA reads the rest,
kernite takes it too: `sust.p` with a cache operation, as `sust.b` has, or an `mma` fragment `(a, 8)`, a value beside a
variable. An `f16x2` variable is one more, where an instruction takes a `u32` or an `s32` variable: `ptxas` reads its 32
bits as a `b32`'s, and so does kernite. `add.u32(i, h, n)` writes `h` as the operand, with no `mov` beside it, and
`vadd.u32.u32.u32(i, h.b0, n)` writes a part of it. kernite refuses it where `ptxas` does: negated, as an array's layer,
beside a register of another type in a vector, and in each operand the table marks with `~`, as the count of a shift, a
result of `clz`, a fragment of `mma`, the time of `nanosleep.u32`, and the last operand of `vadd4.u32.u32.u32.add`,
which `ptxas` makes a `dp4a` of. In a vector it is one only where each other operand goes with an `f16x2`: an `f16x2`, a
`b32`, `_` or a special register, `(h, tid.x)`, and no `u32`, `s32` or value known now, `(h, i)` or `(h, 1)`, as it is a
float of its own. kernite takes none in the vector of `tex.a2dms`: `ptxas` checks no type of its sample, and takes a
predicate there, which looks like a bug of its own. An instruction that wants an `f16x2` takes no integer, one that
wants an `f32` takes no `f16x2`, and none takes an `f16x2` for a predicate, where `ptxas` reads 32 bits, not one. A call
takes an `f16x2` for a `u32`, an `s32` or an `f32`, as an argument and as a result. `ret` gives one back for a `u32` or
an `s32`, not for an `f32`, as `ptxas` takes it. A function of an `f16x2`, a `bf16x2` or a `tf32`, whose `.param` is a
`.b32`, takes a variable of any type of 32 bits for it, writes one with its result, and gives back one with `ret`. Where
a value known now would be cut to fewer bits, by `ptxas` or by the instruction as the ISA has it, or is one the ISA
leaves undefined, kernite refuses it, and the table says so: `lop3`'s table is 8 bits, and `ptxas` takes a bigger one
and keeps its low 8; `shfl` reads its lane in the low 5 bits of `b`; and a barrier counts 1024 threads at most, the most
a CTA has. A mask of lanes, which names the threads of a warp that take part, may not leave out lane 0 where thread 0
runs the instruction: thread 0 is in lane 0, and the ISA leaves the instruction undefined in a thread outside its mask.
Thread 0 runs the instruction where no `if` guards it and no `ret`, `exit`, `trap`, call or jump keeps thread 0 from it.
kernite refuses a mask known now that leaves out lane 0 there, and takes one in a variable. It does not look inside a
function that kernels call. Where `ptxas` takes less, kernite does too: `st` stores a special register only as one of a
vector's, `cvt` reads none to a float or from one, `cvt.rzi.u32.f32(i, envreg0)`, and `mov.f32` and `mov.f64` read none,
not `envreg0`, a `b32`, for an `f32`: `mov.b32` and `cvt.u32.s32` read it. And kernite checks what `ptxas` leaves to the
program: that an access stays in the memory it names, at a multiple of the bytes it moves. Each form, and each of its
words, says the oldest version of PTX and the oldest GPU `ptxas` takes it at, back to PTX 1.0, which is at times older
than the ISA says: `ptxas` takes `ld.mmio.acquire` for `sm_70`, which the ISA has from `sm_75`. A module's `.version`
and `.target` are the oldest that have every instruction of its kernels and functions, those that `cfg` keeps, and never
older than PTX 3.0 and `sm_30`: the driver starts at most 65535 blocks along x of a module written for `sm_20` or
`sm_21`, Fermi's, whatever the GPU it runs on, and 2^31 - 1 of one written for `sm_30`. Where `ptxas` is installed, a
test hands it every form at what kernite says it needs, at the newest version for each GPU from its own to `sm_86`, and
for each newer GPU to `sm_86`, which the driver compiles a module for when it runs there, and `ptxas` must take it each
time, and say nothing of it: a warning of `ptxas` fails the test as an error does, but the warning that it cannot size
the stack of a kernel with `alloca` or `stackrestore`, which size it as they run. At the version before and on the GPU
before, `ptxas` must refuse it. PTX has no 5.1: 6.0 came after 5.0. `ptxas` 13.5 reads a module of 5.1 as one of 6.0,
but the ISA says nothing of what a module of 5.1 is, so kernite writes none: a module that needs 6.0 says 6.0. A test
hands `ptxas` a module of each version from 1.0 to the end of the major version after the newest, 10.9 now, and fails
where it reads a version that the tests of each form do not try, or does not read one that they do; 5.1 is the one it
reads and they do not try.

What the newest PTX and `ptxas` refuse, kernite refuses at every version of PTX. What they take, kernite takes for the
newest: an older `ptxas` or driver may refuse it, and kernite does not guarantee one. kernite refuses what PTX
deprecates, though the ISA still has it. PTX deprecated `shfl` and `vote` without `.sync` in 6.0, and from 6.4 `ptxas`
refuses them for `sm_70` and newer, where the threads of a warp need not run together; for the rest it takes them, and
warns, but says nothing of `vote.ballot.b32` in PTX 6.0 and 6.1, which looks like a bug. kernite refuses them whatever
the version of PTX and the GPU: `shfl.up.b32`, `shfl.down.b32`, `shfl.bfly.b32`, `shfl.idx.b32`, `vote.all.pred`,
`vote.any.pred`, `vote.uni.pred` and `vote.ballot.b32` are errors at their names, which say why, and name the form with
`.sync`, as `shfl.sync.idx.b32(i, n, 0, 31, 0xffff_ffff)`, whose last operand is a mask of lanes.

An `extern "ptx-func"` function is a `.func` that the module's kernels and functions call, itself among them. It
takes and returns PTX's types, and `call(g, gcd, (a, b))` calls it by its name, or `call(g, fp, (a, b))` at the
address `mov.u64(fp, gcd)` put in `fp`. Each call is checked against what the function takes and returns, and a call
of one that never returns, `-> !`, ends its caller's path there. A helper, a function without an ABI, is inlined
instead, and leaves no call behind. `#[inline]` changes neither, and kernite refuses it on both.

An instruction's name is Rust's words: a word that starts with a digit takes a `_`, a `::` is `__`, and a keyword
takes `r#`, so `tex.2d` is `tex._2d`, `ld.global.L2::64B` is `ld.global.L2__64B`, and `cp.async` is `cp.r#async`. A
texture or a surface is its handle, the `u64` of a CUDA texture or surface object, in brackets with its coordinates:
`tex._2d.v4.f32.f32((r, g, b, a), [t, (x, y)])`.

An editor shows the doc of a PTX instruction where a kernel names it, as it does for x86: hover at any word of
`ld.global.f32(v, [p])` in rust-analyzer. The doc says what the instruction does, in a sentence of kernite's own; what
each word of its name means, in a phrase; each form it takes, as a kernel writes it, `ld.global.f32(=f32/_, [u64])`,
with `=` before each operand it writes, `+` before each it reads and writes, `%` before each that must be a variable,
`_` where a sink may stand, and the limits of a number known now, `u32[0,31]` or `#4|8|16`; the PTX and the GPU it
needs, when more than PTX 3.0 and `sm_30`, and `@7.5 sm_80` after a form that needs more than the first; and where
PTX's ISA says the rest. Each instruction of `forms` has a doc and each word of its name a phrase, which a test holds
the table to. `shfl` and `vote` without `.sync` have none, as kernite refuses them, and a name that is no instruction
has its own error and no doc. The macros write a doc as they write x86's, so it adds no code and changes no listing and
no error.

Memory a kernel declares is a variable: `let tile: shared<[f32; 256], 16>;` is 256 floats of shared memory at a
multiple of 16, reached by its name, `ld.shared.f32(v, [tile + 16])`, and `shared<[f32]>` is as many as the launch
gives, `Grid::new(blocks, threads).with_shared(bytes)`. A kernel's `f16` and `bf16`, which Rust has not, are
kernite-cuda's [`f16` and `bf16`](crates/cuda/src/half.rs): their bits, converted to and from `f32` as `cvt` converts
them.

Memory of the module's is a `static` beside its kernels, declared once in `PTX`, which each kernel names as it names its
own, and which keeps what they write from one launch to the next: `static COUNTS: global<[u32; 16]>;` is 16 words of
global memory, `red.global.add.u32([COUNTS + 8], 1)`, and `static WEIGHTS: constant<[f32; 4]> = [1.0, 0.5, 0.25, 2.0];`
four floats of constant memory, which the kernels only read, `ld.r#const.f32(w, [WEIGHTS + 4])`, holding what it says
until something writes it. The host reads and writes each as `Kernels` gives it, a [`Global`](crates/cuda/src/lib.rs) of
Rust's type for its elements: `kernels.COUNTS()?.read()?`, `kernels.WEIGHTS()?.write(&[2.0; 4])?`. An accessor is an
error only by a mistake of kernite's own: the module has not the memory it names, or not a whole number of its elements.
A module has 64 KB of constant memory, and kernite refuses a static that does not fit in it beside the statics before it
that fit, at its type. It takes no room, so a static after it may fit. The room is that of the statics `cfg` keeps: two
tables of 48000 bytes, one under `cfg(feature = "new")` and one under `cfg(not(feature = "new"))`, fit in a module, and
both under `cfg(feature = "new")` do not.

A module's global memory may be of any size that the GPU has the memory for. `ptxas` writes the size of a module's
global memory in 32 bits, and cuts a bigger one without a word, so a static of global memory that makes it 4 GB, its
bytes with those of the statics before it that are in the module, each with the padding its alignment may need, is
outside the module, `static BIG: global<[u8; 6442450944]>;` say: kernite-cuda allocates it, zeros, at its alignment,
when it loads the module, and frees it with the module. Every static counts, whatever `cfg` keeps, so a static is in the
module or outside it whichever features are on. A GPU that has not the memory is an error at the load, which names the
static. A kernel reaches it as it reaches a static in the module, `st.global.u8([BIG + 6442450943], v)`, and at the same
cost: it reads the address from the module's constant memory, where the address of one in the module is read from the
driver's, and its instructions are the same. It holds zeros, so a value is an error. The host reads and writes it as it
does a static in the module. Its address takes 8 bytes of the 64 KB.

A static is memory on the GPU, and no symbol or code of the host's, so the method that gives it has the attributes that
say how Rust has it, and no other: a doc, `cfg`, `deprecated`, the level of a lint, `allow`, `warn`, `deny`, `forbid` or
`expect`, and a `rustfmt` attribute, each alone or in a `cfg_attr(..)`, whatever its condition. kernite refuses any
other, as an error at its name, which says that the static is memory on the GPU, which the attribute is not for:
`#[unsafe(no_mangle)]`, `#[unsafe(export_name = ..)]`, `#[unsafe(link_section = ..)]` and `#[used]` are for a symbol of
the host's, `#[cold]` and `#[track_caller]` for its code, and the rest of Rust's for an item of its own. A static of x86
kernels is a `static` of Rust's, and its attributes are Rust's.

A kernel whose blocks wait for each other is `#[cooperative]`, and its `grid.sync()` waits until every thread of the
grid has come to it, as CUDA's does. The driver launches it cooperatively, every block of the grid at once, so what a
thread keeps in its registers, or a block in its shared memory, lasts across the wait, where two kernels would write it
to memory and read it back. Its grid has at most as many blocks as the GPU runs at once, which `Kernels` says by a
method of the kernel's name: `kernels.normalize_blocks(256, 0)?` for blocks of 256 threads and no shared memory from
the launch. A grid of more is an error, which says how many. Such a kernel needs a GPU from Pascal's on, `sm_60`, and
kernite refuses `grid.sync()` under an `if`, in an `extern "ptx-func"` function, or in a kernel that is not
cooperative. A jump around one is the kernel's to see to: every thread of the grid takes it, or none does.

A GPU kernel tells `ptxas` what its launches look like with PTX's tuning directives, spelled as `#[instances(..)]` and
`#[cooperative]` are. `#[maxntid(16, 16)]` is the most threads a block has, along x, then y and z: one size, two or
three. `#[reqntid(8, 8, 4)]` is the threads it has exactly. `#[minnctapersm(2)]`, beside either, is the blocks an SM
should hold at once, at least. `#[maxnreg(32)]` is the most registers a thread may use. Each gives whole numbers known
now, consts or the `N` of an instance, and the kernel's `.entry` says them as the ISA does, one to a line between its
parameters and its body: `.maxntid 16, 16`. kernite refuses what the ISA forbids and what `ptxas` would cut or ignore
without a word, as an error at the number or the name that is wrong:

- a size of 0, or past what a block has: 1024 threads along x and along y, 64 along z, and 1024 in all. `ptxas` ignores
  a `.maxntid` past 1024 threads, and takes `.reqntid 32, 33`, which no launch matches;
- `#[maxntid(..)]` with `#[reqntid(..)]`, which `ptxas` refuses;
- a directive twice: `ptxas` keeps the last, and says nothing of the first;
- `#[minnctapersm(..)]` without either, which `ptxas` ignores; more than 16 blocks; and blocks that make more than 32
  warps. An SM of some GPUs holds no more, and `ptxas` ignores the number on one. It counts the warps of the blocks, not
  their threads, a warp being 32 threads and the last of a block maybe fewer: 10 blocks of 96 threads make 30 warps, and
  10 of 97 make 40, which Turing's SM does not hold, though they are 970 threads and it holds 1024;
- `#[maxnreg(..)]` of fewer than 24 registers, which `ptxas` raises to 24, or more than 255, which it ignores; and more
  registers for the threads of a block than a launch gives it. A launch counts the registers of a thread by eights, 32
  threads to a warp, and the warps of a block by fours, and a block has 65536 at most: 255 registers for 256 threads
  are 256 * 32 * 8 = 65536, and for 257 threads 256 * 32 * 12 = 98304. `ptxas` does not lower `#[maxnreg(..)]` to fit
  the threads of a block: `#[maxntid(1024)]` alone gives the kernel the 64 registers that 1024 threads have, but it
  keeps `#[maxnreg(128)]` beside it, and a kernel that used them all could not launch;
- `#[maxnctapersm(..)]`, which PTX has not from 2.1 on, in any form and wherever it is written; and any other
  directive on what is no GPU kernel: a helper, an `extern "ptx-func"` function or an x86 kernel.

A launch over a block that the kernel's `#[maxntid(..)]` or `#[reqntid(..)]` refuses is an error that says the directive
and the threads, as a grid's does:
``... each block has 128 threads along x, and the kernel's `.reqntid 256` takes 256 exactly``, or
``... each block has 512 threads, and the kernel's `.maxntid 256` takes 256 at most``. The driver holds `.maxntid` to
the threads of a block in all, of any shape, and `.reqntid` to the size along each of x, y and z. It lets a block of one
thread through to a kernel of `.reqntid`, and starts the kernel's own threads in it.

A kernel is named in PTX as C++ names a function in the namespaces of its workspace, its crate and its modules, so
that a profiler tells two crates' `saxpy` apart: `_ZN7kernite6my_app3gpu5saxpyE`, which `cu++filt`, Nsight and
`cuda-gdb` show as `kernite::my_app::gpu::saxpy`. An instance of a generic kernel is named as a template's,
`kernite::my_app::gpu::block_sums<(unsigned int)1024>`, and launched as Rust calls a generic method,
`kernels.block_sums::<1024>(grid, x, sums)`; the shared memory the launch gives is named as the kernel's static,
`kernite::my_app::gpu::reverse_runs::run`, and a static of the module's as a variable of its namespace,
`kernite::my_app::gpu::COUNTS`. The workspace's manifest is the one Cargo takes for the package. kernite reads a
relative `CARGO_HOME` from the directory rustc runs in, and Cargo from the one it was run in, which may be another.
The workspace is named as the directory of its manifest is, unless the manifest names it, as kernite's does, so that
a clone keeps the names wherever it is:

```toml
[workspace.metadata.kernite]
name = "kernite"
```

`name = ""` leaves the workspace out of the names: `my_app::gpu::saxpy`. A `name` that is not a string is an error at
the module. The error names the manifest. A package with no workspace's manifest is one of its own, named as the package
is.

## A safe GPU kernel

A GPU kernel that is no `unsafe fn` is one kernite proves safe where it compiles it, and Rust calls its launcher as it
calls any function, with no `unsafe`. The launcher takes memory where the kernel takes a pointer: `&xs` where the kernel
takes a `*const f32`, memory it reads, and `&mut ys` where it takes a `*mut f32`, memory it writes. The memory is a
`Buffer`, managed or page-locked memory, a slice the host lends for good, `cuda.lend(Vec::leak(v))`, or, for a `*const`,
the elements of a buffer from one on, `xs.tail(i)`. A slice lent for less is not, as a `mem::forget` of what `lend` gave
would give it back to the host while the kernel may still write it. Rust keeps the host from touching `ys` while the
launcher has it, and from lending the same memory twice; the kernel keeps what it is lent until it is done, so that
`read`, `write`, `host` and dropping the memory wait for the kernel first. What is left is the kernel's own threads, and
kernite proves, as each instruction comes, that they keep to what they are lent:

- each load and store is of global memory, at an address worked out from one of the kernel's pointers, at a multiple of
  the bytes it moves, and inside the memory the launcher lent for it;
- no two threads write one address, and none reads what another may write: each thread writes only its own part of the
  memory, and reads of memory the kernel writes only that part.

What keeps the threads apart is the thread's index among the grid's, `ctaid.x * ntid.x + tid.x`, which `global_index`
works out in [`examples/my-app`](examples/my-app/src/main.rs), and what keeps it inside the memory is a compare: the
threads whose index is not below `n` end, `setp.ge.u32(past, i, n); if past { ret(); }`, and kernite knows that a thread
that goes on has `i < n`, as it does after any `ret`, `exit` or `trap` under an `if`. kernite follows whole numbers: the
thread's index in its block, the block's in its grid, the sizes of both, the kernel's integers and numbers known now, by
`mov`, `add`, `sub`, `mul`, `mad`, `shl` and `cvt` from 32 bits to 64, and the addresses that `mad.wide` and `add` make
of a pointer and a number. What it cannot know, because the launch gives it, is left as a condition on the launch: that
the memory lent for a pointer has the elements the kernel reaches, `x.len() >= n` for `saxpy`; that a thread's part of
the memory it writes holds its bytes; and that no number the kernel works out wraps. The launcher checks each of them,
and the grid, before the kernel runs. An error says which does not hold, as Rust writes it, with the value of each
number in it, and no kernel runs: ``cuLaunchKernel: `saxpy` is safe on this launch only if `x.len() >= n`, so that each
element it reaches of `x` is inside it, and that does not hold: `x.len()` is 999, `n` is 1000``.

The grid of a safe kernel runs along x alone, with fewer than 2^32 threads, so that `ctaid.x * ntid.x + tid.x` numbers
each of them apart in 32 bits. A grid with more than one block or thread along y or z is an error before the kernel
runs, ``cuLaunchKernel: `saxpy` is safe on a grid that runs along x alone, for kernite proves that its threads reach
memory apart by their index along x: the grid has 2 blocks along y``, and so is one of 2^32 threads or more,
``cuLaunchKernel: `saxpy` numbers its threads in 32 bits, so its grid has fewer than 4294967296 of them: the grid has
16777216 blocks of 256 threads, 4294967296 in all``. Memory of another GPU than the kernels are loaded on is an error
too, ``cuLaunchKernel: the memory is on another GPU than the kernels are loaded on``, and so is managed memory on a GPU
that cannot reach it while the host does, which the host may touch while the kernel runs: ``cuLaunchKernel: a kernel
that is no `unsafe fn` is not lent managed memory by a GPU that cannot reach it while the host does: its
CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS is 0``. The code of a safe kernel is that of the same kernel written with
`unsafe`: kernite adds none to the kernel, and the launcher's checks run on the host.

What kernite cannot prove it refuses, at the first instruction or operand it cannot, and the error says what kernite
cannot prove there, and that a kernel that does it is an `unsafe fn`, which compiles as it always has, and whose
launcher takes pointers, `xs.ptr()`, and is `unsafe`. A kernel is an `unsafe fn` when it:

- writes at an address that is not only its thread's, `[y + (i + 1) * 4]`, or reads memory the kernel may write at one
  that is not, so that a thread may race with another;
- reaches an element at an index that no compare keeps below the memory's length, or one that a compare keeps below it
  only after the access, or reaches memory before it starts, `[at - 4]`;
- reaches an address that kernite cannot follow to a pointer of the kernel's, as a number or a loaded word is, one that
  is not a multiple of the bytes it moves from its pointer, or more bytes at once than the type its pointer is to has,
  `ld.global.v4.f32` through a `*const f32`; or writes through a `*const` pointer, which is memory lent to be read;
- loads or stores more than plainly, `st.volatile.global.f32`, or reaches memory by a generic address, `st.f32`, which
  may be shared or local memory, or any memory but global;
- names shared memory, the kernel's own or the launch's, local memory, or a static of the module's, which the threads of
  a block, or every kernel and the host, reach;
- makes threads wait for each other, or orders what they write, `bar`, `membar`, `fence`; or works on the registers of
  the threads of a warp together, `shfl`, `vote`, `mma`;
- reaches memory that other threads reach at once, as an atomic, `atom` and `red`, or `cp.async`, does;
- reaches a texture or a surface, jumps, or calls a function, `call`, which kernite does not follow;
- is `#[cooperative]`, whose blocks wait for each other, so that kernite cannot prove that they reach memory apart.

A crate may `forbid(unsafe_code)` and have safe GPU kernels, and launch them: the Rust the macro writes for a launcher
is no `unsafe` of the crate's.

## Examples

[`examples/my-app`](examples/my-app/src/main.rs) uses kernite as a crate outside it would: for the CPU `sum8`, a safe
function of an array, `dot` for one accumulator, for four and for sixteen, two of which spill, `count_bytes`, by tables
in its stack memory, `sum16`, a safe one too, which calls `sum8` for each half, `gcds`, a safe function of three slices,
which calls Stein's `gcd`, a safe function of two numbers, which calls itself, `histogram`, which counts in the module's
memory from call to call, `weigh`, by weights in its constant memory, and `normalize` and `diffuse` on the threads of a
grid, which compute what the GPU's do, bit for bit; and for the GPU `saxpy`, a safe kernel, launched with no `unsafe`,
`block_sums` for blocks of 256 threads and of 1024 and `reverse_runs` in shared memory, `scale_f16`, written in place, a
safe one too, `gcds`, which calls `gcd`, an `extern "ptx-func"` function that calls itself, `histogram`, which counts in
the module's global memory from launch to launch, and `weigh`, by weights in its constant memory, with safe functions
over them; and `matmul` on the tensor cores, by `cp.async`, `ldmatrix` and `mma`, in a module of its own, since it needs
an Ampere GPU, `sm_80`; and in another, for Pascal's, `sm_60`, `normalize`, whose threads keep their floats in registers
while the grid waits for every block's sum, and `diffuse`, which takes any number of steps in one launch, the grid
waiting between them. Its tests hold each kernel to a scalar reference that does the same float operations in the same
order, bit for bit, or the tensor cores, which add in an order of their own, to products of integers, exact in any
order; and its listings to [`snapshots/`](examples/my-app/snapshots/).

```
cargo run -p my-app
KERNITE_BLESS=1 cargo test -p my-app   # rewrites the snapshots after a deliberate change, for review
```

[`examples/microgpt`](examples/microgpt/src/main.rs) is Karpathy's
[microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95): a GPT of one layer that learns from
32,033 names to make up new ones. It runs on 25 kernels in x86 for the CPU, or on 25 in PTX for the GPU, and on a
plain Rust reference that does the same float operations in the same order; all three read the model's sizes and
`exp`'s constants from one module of consts. Its tests hold the three to each other bit for bit, one operation at a
time.

Its weights come from the same Mersenne Twister draws as Python's, and it trains on the same names in the same
order, so it prints what microgpt.py prints: the same twenty names, and the same loss at 998 of its 1000 steps. At
the other two the fourth decimal differs by one, where `f32` rounds differently from Python's `f64`. The tests hold
it to what Python drew and printed, in [`fixtures/`](examples/microgpt/fixtures/). It downloads the names on its
first run, as microgpt.py does.

```
cargo run --release -p microgpt              # the x86 kernels
cargo run --release -p microgpt -- gpu       # the PTX kernels, on the GPU
cargo run --release -p microgpt -- reference # plain Rust
```

[`examples/matmul`](examples/matmul/src/main.rs) multiplies matrices, `c = a b` into `f32`: `sgemm` in x86 on the
CPU's threads, `sgemm` in PTX on the GPU's cores, and `hgemm` from `f16`, for the tensor cores of an Ampere GPU,
`sm_80`, with an `n` and a `k` that are multiples of 8. Its tests hold each kernel to a scalar reference, bit for bit,
or `hgemm` to products of integers, exact in any order; and its listings to
[`snapshots/`](examples/matmul/snapshots/). It times them, in GFLOPS:

```
cargo run --release -p matmul -- cpu             # the x86 kernel, on 1, 6 and 12 threads, those the machine runs
cargo run --release -p matmul -- gpu 1024 4096   # the PTX kernels, on the GPU, at these sizes
```

## Layers

Each crate depends only on the layers inside it. `crates/kernite/tests/architecture.rs` checks this rule.

| layer | crate | what it knows |
|---|---|---|
| entities | `kernite-core` | programs over variables and over registers |
| language | `kernite-lang` | kernels in Rust's syntax: parsing, evaluation, the `Target` port |
| entity service | `kernite-compile` | the allocator: liveness, register allocation, spilling, the frame's layout; which items every path of a program runs; the `Machine` and `Spill` ports; tested on a toy machine |
| use cases | `kernite-build` | what kernite does with a module: ABIs, its rules, compiling each function; the ports a machine implements |
| adapters | `kernite-x86`, `kernite-ptx` | x86-64: its instructions, SysV, Intel syntax; PTX for NVIDIA GPUs |
| composition | `kernite` | which machine and syntax each kernel compiles with |
| delivery | `kernite-macros` | kernels as functions: `#[unsafe(naked)]`, and the safe functions that call them; launchers of a grid's threads, or launchers of PTX |
| runtime | `kernite-cuda` | the CUDA driver: the GPUs, memory on each and memory the host reaches too, modules, launches and what a safe kernel's launch must hold, copies between GPUs; `f16` and `bf16`; depends on no other crate |
| runtime | `kernite-cpu` | the threads of a cooperative x86 kernel's grid and the CPUs they run on, and the memory of a module of x86 kernels; depends on no other crate |

## Guarantees

The output is deterministic:
- no crate depends on anything outside the workspace;
- clippy forbids hash maps and hash sets.

The Rust the macro writes needs no `allow` of a lint of Rust's but one, so a crate may `forbid` the others, `dead_code`
say, and build its kernels: Rust lints the names and the code a kernel is written with, and none that the macro adds. A
crate of kernels may not forbid two lints: Rust's `improper_ctypes_definitions` where an x86 kernel takes a slice, which
Rust says of a `sysv64` function that takes one, and which the function the macro writes for the kernel allows; and
Clippy's `too_many_arguments`, which Clippy says of a launcher of more than seven parameters, whoever wrote it, and
which each launcher allows.

Each name that the Rust the macro writes binds for itself starts with `__kernite_`, for Rust reads a binding named as a
const in scope as that const, and refuses one named as a static. So a module of kernels may have a const or a static of
any other name in scope, `grid` say.

Checking the generated kernels is for the crates that hold them: a snapshot of every listing, so that any change
to any layer that moves an instruction shows in review, and runs against scalar references, bit for bit, as in
the examples.

## Commands

```
./make.py check    # formatting, clippy, every test, and the docs
./make.py fmt      # formats every crate
./make.py publish  # publishes each crate crates.io lacks at its version, one at a time
```

kernite builds with Rust 1.100 and later, as the workspace's `rust-version` says. `make.py` runs under
[uv](https://docs.astral.sh/uv/).
