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
    pub unsafe extern "sysv64" fn sum8(xs: *const f32) -> f32 {
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

let sum = unsafe { kernels::sum8(xs.as_ptr()) };
```

`sum8` becomes an `#[unsafe(naked)]` function of the signature it declares, its body the kernel in Intel syntax,
and `kernels::LISTINGS` holds each kernel's listing for snapshots. Everything that is not an instruction happens
while the kernel compiles: `for` unrolls, `if` on a bool chooses, a function without an ABI is a helper inlined
where it is called, and `<const N: usize>` makes one generic. Each parameter and generic of a function has a
name of its own, but `_`. A parameter named `_` is given and not read, as in Rust, in a kernel of every kind: the next
parameter is read where the ABI puts it. Rust cannot pass `_` on, so the macros name it for Rust, and the function Rust
calls the kernel by takes it under that name. A generic kernel is compiled for each value its `#[instances(1, 4)]`
lists, each a kernel of its own, which Rust calls as it calls a generic function, `kernels::dot::<4>(x, y, n)`; a value
it does not list fails to build. A value known now holds at most 1048576 elements in all, however its arrays nest, and a
range as many. Each number, float and `bool` is an element, and so is each array of none or one item,
which costs memory of its own. A kernel declares at most 65536 variables, however its loops and helpers repeat
them. Working out a function takes at most 4194304 steps: each turn of a `for`, each instruction, call, label or
note, each number of a range, each item an array lists, as `[a, b]`, and each element of each `[x; n]`, of each
array a method makes, of each one copied from a name, as `let b: [u32; 4] = a;` and `for x in a` copy it, and of
each value an index takes or a note shows. A number read by name or worked out costs none of its own. Nesting is
held to 64 levels, so that no kernel can run the compiler out of stack. Brackets, types, expressions, rows of
`else if`s, consts that name consts and a value's arrays each nest 64 deep at most, and each operator in
`a + b + c` is a level of its expression. A block, a `for` and an `if` are a level for what they hold, and count
together with the expression inside them, along the helpers that call one another.
[`kernite-lang`](crates/lang/src/lib.rs) describes the language.

A type in a kernel's parameters or return means what it means in any function of the module: `super::W` is the `W`
beside the module of kernels, and `self::W` the one in it. It may name the kernel's compile-time arguments, as in any
generic function: in `fn arr<const N: usize>(p: *mut [u8; N])`, `arr::<2>` takes a pointer to two bytes. Rust takes the
argument in a type alone, `N` or `{ N }`, and no expression of it, `N + 1`. An error in a type is Rust's, a pointer's
among them, said once, at the type, in a kernel of every kind, and so is a lint, at the level the kernel's `allow`,
`warn`, `deny`, `forbid` or `expect` gives it: a kernel of instances, or a cooperative one, on x86 or on a GPU, is no
different from any function. The one exception is a pointer that a kernel of instances gives back, which the next
paragraph tells. A generic is a value: `<T>` is an error at the `T`, and so is `impl` in a parameter's type or the
result's, wherever in it, `x: impl Copy`, `Option<impl Copy>` or `-> *mut impl Copy`, at the `impl`: in a parameter it
makes the function generic over a type, and in the result it hides one. kernite says both in the same words.

A kernel's pointer to a type with a size, `*const T` or `*mut T`, is one register: `*const [usize; 3]` and
`*mut *const f32` are taken as `*const f32` is, on x86 and on a GPU. A pointer to a type with no size has two words, an
address and a length or a table, so it is an error at its type: take the address and the length as two parameters.
kernite tells a slice, a `str` or a `dyn`, or a tuple that ends in one, in parentheses or not, by what is written, and
says so in its words. Rust tells any other, whatever it holds and whatever it is called: `CStr`, `Floats` where
`type Floats = [f32];`, a struct of yours whose last field is a slice, `(&u8, CStr)`. It says that the size cannot be
known, at the type, for a pointer a kernel returns as well as for one it takes, in a kernel of every kind. A type that
Rust refuses by itself is its error alone, said once, at the type, in a kernel of every kind: an argument that does not
meet its struct's bound, `*const NeedsCopy<String>` where `struct NeedsCopy<T: Copy>`, or an array or a tuple that holds
a type with no size, `*const [CStr; 3]` or `*const (CStr, u8)`. One case has two errors: in a kernel of instances,
`-> *const (u8, [CStr; 3])` has Rust's error twice, at `[CStr; 3]` and at the whole type. A type in a message is written
as Rust writes it, `[usize; 3]`.

A number's name in a kernel's parameters or return, `u8` to `u64`, `i8` to `i64`, `usize`, `isize`, `f32`, `f64` or
`bool`, means that number: `u64` is 64 bits. A module that gives the name another type, `pub type u64 = u32;` and a
`use` of it, would have Rust pass a `u32` where the listing reads 64 bits, so it is an error at the type. Rust says it,
once, in a kernel of every kind: it expected `u64` and found `u32`. A type with no size, `pub type u8 = [u8];`, adds
Rust's error that its size cannot be known, at the type too. Rust says nothing at the module's attribute.

A module may give the name of a primitive, `u8` or `str`, another type, and its kernels build as they do without, so
long as none takes or gives that name as a number: the Rust kernite writes names each primitive by its path,
`::core::primitive::u8`.

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
Rust sees neither. Where kernite finds an error in compiling a function or a static, or refuses the module as a whole,
Rust is shown the types of every helper and function of the module, one that no kernel calls too: each compile-time
argument's, each parameter's and the result's, as it sees a kernel's. kernite refuses the module as a whole when it is
two modules, when no module of PTX holds its GPU kernels and statics, when an item of its own has the name of a const it
takes, when the workspace's name is not a string, and when its attribute is refused, as below. An import that only these
types name is used, so Rust does not call `use super::Ret;` unused beside kernite's error at `Ret`, which is said once:
at the argument of a call of the helper, at the result of a helper that is called, or at the type of a function. Rust
says of the types only that a lifetime is not declared, `x: &'a Ret`; that a module is where a type is written,
`x: core::f16` or `x: core`; and what a macro writes, `x: half!()` with an `f16`, which is unstable. It says the same of
the types of a helper that no kernel calls, of which kernite says nothing. It says nothing of a name it cannot find,
such as `gpr` or `local<[u32; 4], 16>`, nor of a type it refuses by itself, `[u64]`, `f128` or `_`. A function's types
are shown where `cfg` keeps the function and the error is said. A module that builds is shown none of them, so an import
that only a helper names is unused to Rust, which says so as it does of any import that nothing names.

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
it says nothing it would find in them, such as a kernel that is not `unsafe`, an ABI it does not know, what a static
holds, or what compiling a kernel finds. It compiles nothing of the module, so a workspace `name` that is not a string
is no error at it. No room is counted for the statics, so a crate with no dependency named `kernite-macros` is not told
it needs one for them. What the macro finds in writing the module for Rust is said, as for any module: the dependency
the crate needs, `kernite-cuda` for a GPU item that `cfg` keeps and `kernite-cpu` for an x86 static or cooperative
kernel; a compile-time argument named as a parameter of a launcher or of a `_blocks` method, `grid`, `threads` or
`shared`; and a GPU kernel or a static named as a cooperative kernel's `_blocks` method. A module kernite cannot read
holds its refusal alone, and Rust says nothing of the code that names its items, for the error is in the module.

An argument of `#[consts]`, as in `#[consts(x)]`, is told where it is, and changes nothing else. The module is the one
it is without the argument: its other errors are said, and Rust says nothing of the code that names its consts, nor of
the modules of kernels that take them.

A function or a static that `cfg` leaves out says nothing, as Rust says nothing of a function it leaves out. A macro
runs before Rust knows which `cfg` holds, so kernite compiles each one, and puts each error it finds in one under the
same `#[cfg(..)]` or `#[cfg_attr(.., cfg(..))]`: a kernel that does not compile, or is not `unsafe`, a helper, a static.
Where `cfg` leaves them out, the crate builds without a word; where it keeps them, each error is said where it is.
A crate that keeps its x86 kernels on x86-64 alone, by `cfg(target_arch = "x86_64")`, builds for another machine
without a word, though only x86-64 has `extern "sysv64"`: each function kernite writes for a kernel is under the
kernel's `cfg` too, as one for each instance of a kernel of compile-time arguments or of a cooperative one.

A kernel names the items of its module that it uses: a const it reads, a static, a helper, a kernel or an
`extern "ptx-func"` function it calls. A kernel names an item in its body, in its `#[instances(..)]`, in its directives
or in a helper it calls; an `extern "ptx-func"` function names one in its body; a static, in its type or in what it
holds. A helper names nothing of its own: the kernel or function that calls it names what it names, where the helper
writes it, and only where `cfg` keeps the helper as well as the kernel. What a const names in its definition is Rust's
own, and not the kernel's.

A kernel that `cfg` keeps names only what `cfg` keeps. Where it names an item `cfg` leaves out, the name is an error,
said at the name, once for each place it is written, in a kernel of instances too. Rust never sees a kernel run, so
kernite says it. Of the items that are Rust's, a const, a static in the process's memory and an x86 kernel that takes
no compile-time arguments, kernite writes the name for Rust where the kernel writes it, under the kernel's `cfg`, and
Rust says what it says of any name it cannot find: `cannot find value`. Of the rest, which Rust has no item for, a
helper, an instance of an x86 kernel, an `extern "ptx-func"` function and a static in the GPU's memory, kernite says
that `seven` is left out by `cfg` where `calls_seven`, which names it, is kept. Where `cfg` leaves out the kernel too,
or keeps the item wherever it keeps the kernel, nothing is said. A kept kernel that calls a helper `cfg` leaves out,
itself or through a helper `cfg` keeps, is told of the helper alone, whatever the helper names. One that calls a helper
at two places, one of them in a helper `cfg` leaves out, is told of what the helper names, once. A const that `cfg`
leaves out with every item that names it is not said of, and one that `cfg` keeps and only items `cfg` leaves out name
is unused to Rust, which says so as it does of any const that nothing names.

A module holds what `cfg` keeps: Rust puts it together, where it knows which `cfg` holds. `LISTINGS` has the listing
of each x86 kernel that compiles and `cfg` keeps, and of no other. A module of x86 kernels has it whether its kernels
compile or not, so Rust says nothing of the code that names it, and it is empty where none compiles. A module of GPU
kernels alone has none. `PTX` has each GPU kernel, function and static that `cfg` keeps, for the PTX and the GPU that
the kernels and functions kept need, and `Kernels::load` asks the driver for the names of those alone.

kernite reads every item before Rust knows which `cfg` holds, as Rust parses every item before it does, so an item it
cannot read is said whatever `cfg` holds, as Rust says a syntax error in a function it leaves out: the head of a
function, as `fn k<T>()`, rests on no other item. What rests on all the items together is said whatever `cfg` leaves
out: a name that two items share, though each is under a `cfg` the other is not. GPU kernels and functions that no one
PTX module can hold, for each sets the module's `.version` and `.target`, are said where `cfg` keeps them: once, at the
module's first GPU item, of those that clash and no others.

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
doc name the CPU features a kernel needs, `# needs: avx, fma`, which nothing checks for when it runs; and a kernel that
loads MXCSR stores the caller's first and loads it back before it returns, as SysV says. Where `llvm-mc` and GNU `as`
are installed, a test hands them every form, with each integer the least the form takes and the most: `llvm-mc` must
take it, and GNU `as` must take it at the feature kernite says it needs, without a warning, and refuse it at the
features before.

Stack memory a kernel declares is a variable, as a GPU kernel's local memory is: `let tables: local<[u32; 1024], 32>;`
is 4 KB of its frame at a multiple of 32, reached by its name, `mov(c.r32, [tables + b * 4])`, and `lea(p, [tables])`
takes its address. Until written, it holds whatever was there. kernite checks an address in it where the index is 0:
what it reaches lies inside, and `vmovaps` and `vmovdqa` reach it at a multiple of what they move; the index is the
kernel's to keep inside. A frame of more than a page is touched a page at a time on the way down, as Rust's functions
are, so that one past the end of the stack faults at the guard page there rather than reach past it.

When the variables alive together do not fit in the registers, kernite spills one to a slot of the frame: it stores it
after each instruction that writes it and loads it before each one that reads it. It spills, of the variable left
without a register and those holding the ones it could have, the one whose loads and stores are fewest for how long
it is alive, one in a loop counting ten times one outside, and spills again until all fit. The listing names each
spill, in its header and at each load and store. Sixteen accumulators of `dot` and the load beside them take 17 of the
16 vector registers, and two spill:

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

The kernel's parameters keep the names they are written with, `grid` among them, but `_`, which the macros name
for Rust. A compile-time argument may not be named `grid`, for Rust cannot keep a const apart from a parameter of
its name: that is an error at its name. A crate with a cooperative kernel needs a dependency named `kernite-cpu`
for the grid, not renamed in `Cargo.toml`. kernite says at the kernel when it has none.

A kernel calls the module's others, itself among them, as SysV calls a function: `call(g, gcd, (a, b))` moves `a` to
`rdi` and `b` to `rsi`, and `g` from `rax`, and is checked against what `gcd` takes and returns. `lea(fp, [gcd])` takes
an address to call at, `call(g, fp, (a, b), (gcd, sum))`, checked against each function given. A value alive across a
call is kept in a register the callee leaves as it was, `rbx` or `r12` to `r15`, which the kernel saves for its own
caller; a vector spills, since the callee may change any. The stack pointer is a multiple of 16 at each call. A kernel
that never returns, `-> !`, ends each way out in `ud2()` or in a call of one that does not.

Memory of the module's is a `static` beside its kernels, in the process's memory, which each kernel names as it names its
own, and which keeps what they write from one call to the next: `static COUNTS: global<[u32; 16]>;` is 16 words the
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
in `Cargo.toml`. kernite says at the static when it has none.

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
    pub unsafe extern "ptx-kernel" fn saxpy(a: f32, x: *const f32, y: *mut f32, n: u32) {
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
let (xs, ys) = (cuda.upload(&x)?, cuda.upload(&y)?);
unsafe { kernels.saxpy(kernite_cuda::Grid::new(n.div_ceil(256), 256), 2.0, xs.ptr(), ys.ptr(), n)? };
ys.read_into(&mut y)?;
```

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

A pointer taken before the slice is `Copy`, and the compiler cannot refuse a launch on it, so a launch is `unsafe`, and
its contract is this: each pointer is to a buffer or to unified memory of the GPU, as long as the kernel reads and
writes through it; from the launch until the kernel is done, the host holds no slice of memory the kernel writes, and no
slice it may write of memory the kernel reads; and where `cuda.concurrent_managed_access()` is false, the GPU not
reaching managed memory while the host does, the host holds no slice of managed memory at all. The method is the GPU's
`CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS`, and a GPU where it is false faults the host on managed memory while any
kernel runs. `Module::launch` says the contract in its `# Safety`, and the macros write it as the `# Safety` of the
launcher of a GPU kernel that has none.

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
CUDA_ERROR_INVALID_VALUE (1): invalid argument`. A range with no elements, or one past the end, panics.

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
they are written with, `grid` among them, but `_`, which the macros name for Rust. A compile-time argument may not
be named `grid`, nor, on a cooperative kernel, `threads` or `shared`, which are parameters of the method that says
how many blocks its grid has at most, below. Rust cannot keep a const apart from a parameter of its name: that is
an error at its name.

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
`ptxas` takes, [`forms`](crates/ptx/src/forms.rs), but one. `txq.force_unnormalized_coords` asks of a sampler apart
from its texture, in the independent mode, which kernite does not write: kernite writes no `.samplerref`, so the ISA
leaves the answer undefined. kernite checks each operand against the forms, and a `call`'s against what the function it
calls takes and returns. Where `ptxas` takes more than PTX's ISA has, and reads it as the ISA reads the rest, kernite
takes it too: `sust.p` with a cache operation, as `sust.b` has, or an `mma` fragment `(a, 8)`, a value beside a
variable. Where a value known now would be cut to fewer bits, by `ptxas` or by the instruction as the ISA has it, or is
one the ISA leaves undefined, kernite refuses it, and the table says so: `lop3`'s table is 8 bits, and `ptxas` takes a
bigger one and keeps its low 8; `shfl` reads its lane in the low 5 bits of `b`; and a barrier counts 1024 threads at
most, the most a CTA has. A mask of lanes, which names the threads of a warp that take part, may not leave out lane 0
where thread 0 runs the instruction: thread 0 is in lane 0, and the ISA leaves the instruction undefined in a thread
outside its mask. Thread 0 runs the instruction where no `if` guards it and no `ret`, `exit`, `trap`, call or jump
keeps thread 0 from it. kernite refuses a mask known now that leaves out lane 0 there, and takes one in a variable. It
does not look inside a function that kernels call. Where `ptxas` takes less, kernite does too: `st` stores a special
register only as one of a vector's. And kernite checks what `ptxas` leaves to the program: that an access stays in the
memory it names, at a multiple of the bytes it moves. Each form, and each of its words, says the oldest version of PTX
and the oldest GPU `ptxas` takes it at, back to PTX 1.0, which is at times older than the ISA says: `ptxas` takes
`ld.mmio.acquire` for `sm_70`, which the ISA has from `sm_75`. A module's `.version` and `.target` are the oldest that
have every instruction of its kernels and functions, those that `cfg` keeps, and never older than PTX 3.0 and `sm_30`:
the driver starts at most 65535 blocks along x of a module written for `sm_20` or `sm_21`, Fermi's, whatever the GPU it
runs on, and 2^31 - 1 of one written for `sm_30`. Where `ptxas` is installed, a test hands it every form at what kernite
says it needs, at the newest version, and for each newer GPU to `sm_86`, which the driver compiles a module for when it
runs there, and `ptxas` must take it each time; and at the version before and on the GPU before, which it must refuse.

An `extern "ptx-func"` function is a `.func` that the module's kernels and functions call, itself among them. It
takes and returns PTX's types, and `call(g, gcd, (a, b))` calls it by its name, or `call(g, fp, (a, b))` at the
address `mov.u64(fp, gcd)` put in `fp`. Each call is checked against what the function takes and returns, and a call
of one that never returns, `-> !`, ends its caller's path there. A helper, a function without an ABI, is inlined
instead, and leaves no call behind.

An instruction's name is Rust's words: a word that starts with a digit takes a `_`, a `::` is `__`, and a keyword
takes `r#`, so `tex.2d` is `tex._2d`, `ld.global.L2::64B` is `ld.global.L2__64B`, and `cp.async` is `cp.r#async`. A
texture or a surface is its handle, the `u64` of a CUDA texture or surface object, in brackets with its coordinates:
`tex._2d.v4.f32.f32((r, g, b, a), [t, (x, y)])`.

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
Rust's type for its elements: `kernels.COUNTS().read()?`, `kernels.WEIGHTS().write(&[2.0; 4])?`. kernite refuses what a
module cannot hold: more than the 64 KB of constant memory it has, or 4 GB of global memory, whose size `ptxas` writes
in 32 bits. The error is at the type of a static that does not fit beside the statics before it that fit. It takes no
room, so a static after it may fit. The room is that of the statics `cfg` keeps: two tables of 48000 bytes, one under
`cfg(feature = "new")` and one under `cfg(not(feature = "new"))`, fit in a module, and both under
`cfg(feature = "new")` do not.

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

## Examples

[`examples/my-app`](examples/my-app/src/main.rs) uses kernite as a crate outside it would: for the CPU `sum8`, `dot` for
one accumulator, for four and for sixteen, two of which spill, `count_bytes`, by tables in its stack memory, `sum16`,
which calls `sum8` for each half, `gcds`, which calls Stein's `gcd`, which calls itself, `histogram`, which counts in
the module's memory from call to call, `weigh`, by weights in its constant memory, and `normalize` and `diffuse` on the
threads of a grid, which compute what the GPU's do, bit for bit; and for the GPU
`saxpy`, `block_sums` for blocks of 256 threads and of 1024 and `reverse_runs` in shared memory, `scale_f16`, written in place, `gcds`, which calls `gcd`, an `extern "ptx-func"`
function that calls itself, `histogram`, which counts in the module's global memory from launch to launch, and
`weigh`, by weights in its constant memory, with safe functions over them; and
`matmul` on the tensor cores, by `cp.async`, `ldmatrix` and `mma`, in a module of its own, since it needs an Ampere
GPU, `sm_80`; and in another, for Pascal's, `sm_60`, `normalize`, whose threads keep their floats in registers while
the grid waits for every block's sum, and `diffuse`, which takes any number of steps in one launch, the grid waiting
between them. Its tests hold each kernel to a scalar reference that does the same float operations in the same order,
bit for bit, or the tensor cores, which add in an order of their own, to products of integers, exact in any order;
and its listings to [`snapshots/`](examples/my-app/snapshots/).

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
| delivery | `kernite-macros` | kernels as functions: `#[unsafe(naked)]`, launchers of a grid's threads, or launchers of PTX |
| runtime | `kernite-cuda` | the CUDA driver: the GPUs, memory on each and memory the host reaches too, modules, launches, copies between GPUs; `f16` and `bf16`; depends on no other crate |
| runtime | `kernite-cpu` | the threads of a cooperative x86 kernel's grid and the CPUs they run on, and the memory of a module of x86 kernels; depends on no other crate |

## Guarantees

The output is deterministic:
- no crate depends on anything outside the workspace;
- clippy forbids hash maps and hash sets.

The Rust the macro writes needs no `allow` of a lint of Rust's, so a crate may `forbid` one, `dead_code` say, and build
its kernels: Rust lints the names and the code a kernel is written with, and none that the macro adds. Clippy's
`too_many_arguments` is the one lint a crate of kernels may not forbid: Clippy says it of a launcher of more than seven
parameters, whoever wrote it, and each launcher allows it.

Checking the generated kernels is for the crates that hold them: a snapshot of every listing, so that any change
to any layer that moves an instruction shows in review, and runs against scalar references, bit for bit, as in
the examples.

## Commands

```
./make.py check    # formatting, clippy, every test, and the docs
./make.py fmt      # formats every crate
```

kernite builds with Rust 1.99.1 and later, as the workspace's `rust-version` says. `make.py` runs under
[uv](https://docs.astral.sh/uv/).
