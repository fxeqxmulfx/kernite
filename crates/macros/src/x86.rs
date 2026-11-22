//! What x86 kernels become in Rust: naked functions and their safety contracts, the launchers
//! of cooperative ones, stubs for those that do not compile, and the module's statics.

use proc_macro::{Delimiter, Group, Span, TokenStream, TokenTree};

use kernite::lang::{self, Fn, Value};
use kernite::{Call, HostStatic, Naked, NativeInstance, NativeOut};

use crate::cond::Cond;
use crate::rust::{
    Written, array_ref, asks, attrs, body_at, cfg_of, clashes, depends_on, ending_at, error_under, given_back,
    gives_placeholder, hygienic, launcher_result, missing, naked_levels, name_after, name_at, never, parse, pick,
    pointer, position, primitive, probe, respan, result, signature, span_of, stub, unnamed, value_params, visibility,
    within, written, written_params,
};

/// The crate the Rust of a static or a cooperative kernel names.
const CPU: &str = "kernite-cpu";

/// `f` as an `#[unsafe(naked)]` function whose body is the listing of its one instance, and the
/// statics its constants are, in a hidden module beside it. For a function of compile-time
/// arguments, each instance is such a function, of the name [`instance_name`] gives it, and `f`
/// is one that calls the instance its arguments pick. So is each instance of a cooperative
/// kernel, which takes the thread's place in the grid after `f`'s parameters, and `f` is its
/// [`grid_launcher`]. So is the instance of a kernel that is no `unsafe fn`, and `f` is the
/// function Rust calls without `unsafe`: `#[inline(always)]`, or where the code needs CPU
/// features `#[target_feature(enable = "..")]` and `#[inline]`, for Rust lets no naked function
/// be `#[target_feature]`; and only a function that has the features calls `f` without
/// `unsafe`. An instance is a function of the module of kernels, as `f` is, and not of a module
/// of its own. It takes what `f` takes, but a pointer or an array by reference is `*const ()`
/// or `*mut ()` to it, whatever it points to, and `f` casts the ones it passes on and the
/// pointer it gets back; and a number is the number in both, `::core::primitive::u64`, and not
/// the name the kernel writes. So no type of `f`'s is written twice: Rust reads each once, in
/// `f`, where it means what it means in any function of the module, `super::W` or `[u8; N]`,
/// whose `N` is a compile-time argument, and a number's name in the questions `f` asks of it;
/// it says an error in one once, at the type, and lints it at the level `f` sets. Each instance
/// names the functions of `module` it calls by the operands [`callees`] gives, and its statics
/// by those [`reached`] gives. Its safety contract says that nothing else touches `named`, the
/// module's memory, while it runs, and which CPU features the code needs. A cooperative kernel of a crate without
/// `kernite-cpu` is [`x86_stub`]'s instead, with the error that says so, and so is a kernel that gives back a type with
/// a `_` in it, which is Rust's error alone. All this writes is under `cfg`, `f`'s own, the module of its statics
/// included: an instance is `extern "sysv64"`, which only x86-64 has, so a crate that keeps `f` there alone builds for
/// another machine. `at` says where `f` first writes the name of an item of the module, for the operands to be written
/// there. Its code ends at `f`'s name, for Rust to tell of `f` at `f`: see [`ending_at`].
///
/// A lint is at the level `f` sets at each place its code names an item, as in a function of Rust's. `places` are the
/// places of `f`'s own that Rust is told of by a statement, and they are in the body of `f`, or of its launcher, in an
/// item of no name, where the level `f` sets holds, `expect` too. An operand is a place too, in the function of each
/// instance, which has the levels `f` sets, an `expect` as an `allow`, for Rust refuses `expect` on a naked function.
pub(crate) fn naked(
    module: &lang::Module,
    f: &Fn,
    k: &NativeOut,
    cfg: &TokenStream,
    top: &[TokenTree],
    at: &dyn std::ops::Fn(&str) -> Span,
    places: TokenStream,
) -> TokenStream {
    let (instances, named) = (&k.instances[..], &k.touched[..]);
    // Where the code of `f` ends, its name: see [`ending_at`].
    let end = span_of(name_after(&f.head, top));
    let hidden = format!("__kernite_{}", f.name);
    let cooperative = f.cooperative.is_some();
    if (cooperative && !depends_on(CPU)) || gives_placeholder(f, top) {
        return x86_stub(f, instances.len(), named, cfg, top, places);
    }
    let mut out = TokenStream::new();
    let mut statics = vec![];
    let safe = !f.unsafe_fn;
    // A slice is two words, its address and its length, which Rust passes in two registers, but it is no type of a
    // function of the `sysv64` ABI, so a function that takes one calls the naked function that takes the two.
    let slices = written_params(f, top).iter().any(|p| array_ref(&p.ty).is_some_and(|(.., slice)| slice));
    if let [one] = instances
        && one.args.is_empty()
        && !cooperative
        && !safe
        && !slices
    {
        let listing = &one.naked;
        out.extend(contract(attrs(f, top), named, &features(instances), safe, end));
        out.extend(parse("#[unsafe(naked)]"));
        out.extend(signature(f, top));
        let code = asm(
            listing,
            &format!("{hidden}::C"),
            callees(module, &one.calls, top, at),
            reached(module, listing, top, at),
            probe(f, top),
        );
        let mut body = unnamed(places);
        body.extend(code);
        out.extend([body_at(body, end)]);
        statics.push(("C".to_string(), listing));
    } else {
        // Each instance's parameters and their types: `f`'s, then a cooperative kernel's place. `f`
        // passes its own on by their names, before the colon, and a pointer cast.
        let (mut params, mut names) = (vec![], vec![]);
        for p in written_params(f, top) {
            let (ty, cast) = erased(&p.ty);
            params.push(format!("{}: {ty}", p.name));
            names.push(format!("{}{cast}", p.name));
            if array_ref(&p.ty).is_some_and(|(.., slice)| slice) {
                params.push("_: usize".to_string());
                names.push(format!("{}.len()", p.name));
            }
        }
        if cooperative {
            params.push("_: &::kernite_cpu::Place".to_string());
        }
        // Its result, which `f` casts back if it is a pointer.
        let ret = f.ret.as_ref().map_or(String::new(), |r| format!("-> {}", erased(within(top, &r.tokens)).0));
        let (mut arms, mut fns) = (vec![], TokenStream::new());
        for (k, instance) in instances.iter().enumerate() {
            let (listing, name) = (&instance.naked, instance_name(&f.name, k));
            let code = asm(
                listing,
                &format!("{hidden}::I{k}C"),
                callees(module, &instance.calls, top, at),
                reached(module, listing, top, at),
                None,
            );
            fns.extend(cfg.clone());
            fns.extend(naked_levels(attrs(f, top)));
            fns.extend(parse(&format!(
                "#[doc(hidden)]\n#[unsafe(naked)]\nunsafe extern \"sysv64\" fn {name}({}) {ret}",
                params.join(", ")
            )));
            fns.extend([TokenTree::Group(Group::new(Delimiter::Brace, code))]);
            statics.push((format!("I{k}C"), listing));
            arms.push((instance.args.clone(), name));
        }
        let features = features(instances);
        if cooperative {
            out.extend(grid_launcher(f, Some(&arms), &features, named, cfg, top, places));
        } else {
            out.extend(contract(attrs(f, top), named, &features, safe, end));
            out.extend(match (safe, features.is_empty()) {
                (true, false) => parse(&format!("#[target_feature(enable = {:?})]\n#[inline]", features.join(","))),
                _ => parse("#[inline(always)]"),
            });
            // Rust warns of a slice in the signature of a `sysv64` function, which only Rust calls, as `f` is.
            if slices {
                out.extend(parse("#[allow(improper_ctypes_definitions)]"));
            }
            out.extend(signature(f, top));
            let mut body = asks(f, top, written);
            body.extend(unnamed(places));
            let call = parse(&format!("({})({})", chosen(f, &arms, top), names.join(", ")));
            body.extend(result(f, top, call));
            out.extend([body_at(body, end)]);
        }
        out.extend(fns);
    }
    let mut items = String::new();
    let mut aligns: Vec<u32> = statics.iter().flat_map(|(_, l)| l.consts.iter().map(|c| c.align)).collect();
    aligns.sort();
    aligns.dedup();
    for a in &aligns {
        items += &format!("#[repr(C, align({a}))]\npub struct A{a}<T>(pub T);\n");
    }
    for (prefix, listing) in &statics {
        for (i, c) in listing.consts.iter().enumerate() {
            let bytes: Vec<String> = c.bytes.iter().map(u8::to_string).collect();
            items += &format!(
                "pub static {prefix}{i}: A{a}<[u8; {n}]> = A{a}([{bytes}]);\n",
                a = c.align,
                n = c.bytes.len(),
                bytes = bytes.join(", ")
            );
        }
    }
    if !items.is_empty() {
        out.extend(cfg.clone());
        out.extend(parse(&format!("#[doc(hidden)]\nmod {hidden} {{\n{items}}}\n")));
    }
    out
}

/// The body of a naked function whose code is `listing`'s, its constants the statics
/// `{statics}0`, `{statics}1` and on, the functions it calls by the operands `funcs`, and the
/// module's memory it names, which the operands `memory` give by the name of each static. The
/// types a plain kernel takes or gives back are asked of in `probe`, a `const` operand, which a
/// comment of the code names, as the code is all a naked function has.
fn asm(
    listing: &Naked,
    statics: &str,
    funcs: TokenStream,
    memory: TokenStream,
    probe: Option<TokenStream>,
) -> TokenStream {
    let mut asm = String::new();
    for line in &listing.code {
        asm += &format!("{line:?}, ");
    }
    if probe.is_some() {
        asm += "\"# {__kernite_sized}\", ";
    }
    for i in 0..listing.consts.len() {
        asm += &format!("c{i} = sym {statics}{i}, ");
    }
    let mut args = parse(&asm);
    args.extend(funcs);
    args.extend(memory);
    if let Some(block) = probe {
        args.extend(parse("__kernite_sized = const"));
        args.extend([TokenTree::Group(Group::new(Delimiter::Brace, block))]);
    }
    let mut out = parse("::core::arch::naked_asm!");
    out.extend([TokenTree::Group(Group::new(Delimiter::Parenthesis, args))]);
    out
}

/// `label = sym path,`, the operand of `naked_asm!` by which the code names `item`, an item of the
/// module that `top` holds. Where `cfg` leaves `item` out, `label = const 0,` is the operand instead:
/// the code is text, and names the item whether `cfg` keeps it or not. The error is said where the
/// kernel writes the name, and is not said here as well.
fn operand(label: &str, path: TokenStream, item: &lang::Item, top: &[TokenTree]) -> TokenStream {
    let mut sym = parse(&format!("{label} = sym"));
    sym.extend(path);
    sym.extend(parse(","));
    let kept = Cond::of(&cfg_of(item, top));
    // An item `cfg` never leaves out is named as it is.
    let (Some(then), Some(otherwise)) = (kept.guard(), kept.not().guard()) else { return sym };
    let mut out = then;
    out.extend(sym);
    out.extend(otherwise);
    out.extend(parse(&format!("{label} = const 0,")));
    out
}

/// The mentions of kernel `k` at which it first writes the name of each item that its code names by an operand: a
/// static, or a function of no compile-time arguments; the first that `own` says `k` writes itself, if it writes the
/// name at all, for the operand is in the function of each instance, where the levels of `k` hold, and not a helper's.
/// [`reached`] and [`callees`] write the operand at that place, in the function of each instance that names the item,
/// so Rust tells of the item there: a deprecated one say. `named` has Rust find the item at each other mention. None
/// where Rust does not get the code of `k`, which has no operand.
pub(crate) fn operands<'a>(k: &'a NativeOut, own: &dyn std::ops::Fn(&lang::Mention) -> bool) -> Vec<&'a lang::Mention> {
    if !k.written {
        return vec![];
    }
    let by_operand = |name: &String| {
        k.instances.iter().any(|one| {
            one.naked.statics.contains(name) || one.calls.iter().any(|c| c.instance.is_none() && c.function == *name)
        })
    };
    let mut firsts: Vec<&lang::Mention> = vec![];
    for m in k.mentions.iter().filter(|m| by_operand(&m.name)) {
        match firsts.iter_mut().find(|first| first.name == m.name) {
            None => firsts.push(m),
            Some(first) if own(m) && !own(first) => *first = m,
            Some(_) => {}
        }
    }
    firsts
}

/// What Rust reads for a use of `item`, an x86 kernel of compile-time arguments, by a kernel that names its instance of
/// `args` at `span`: `old::<2>`, the function that has the attributes of the kernel, so Rust tells of the use as it
/// tells of any use of a function of its own, a deprecated one at that place. The code of a kernel names the function
/// of the instance, which has none of them, and Rust has no name for the kernel alone, which it cannot work out the
/// arguments of.
pub(crate) fn instance_used(item: &lang::Item, args: &[Value], top: &[TokenTree], span: Span) -> Option<TokenStream> {
    let args: Vec<String> = args.iter().map(Value::to_string).collect();
    let mut path = TokenStream::from(name_at(item, top, span)?);
    path.extend(parse(&format!("::<{}>", args.join(", "))));
    Some(path)
}

/// The operands by which the code of `listing`, of a kernel of `module`, names each static it
/// names, which is the static's name in `module` as written, `COUNTS`, or `r#type`, whose
/// `{type}` in the code names it. Each is at the place `at` says the kernel first writes it.
fn reached(
    module: &lang::Module,
    listing: &Naked,
    top: &[TokenTree],
    at: &dyn std::ops::Fn(&str) -> Span,
) -> TokenStream {
    let mut out = TokenStream::new();
    for name in &listing.statics {
        let Some(item) = position(module, name).and_then(|i| module.items.get(i)) else { continue };
        let Some(path) = name_at(item, top, at(name)) else { continue };
        out.extend(operand(&path.to_string(), path.into(), item, top));
    }
    out
}

/// `docs`, a kernel's attributes, with its safety contract: what it says of `named`, the memory of the module's that
/// the kernel names, that nothing else touches it while the kernel runs, and, after it, of the CPU `features` its code
/// needs, as [`needs`] says. A kernel that names no memory and needs no feature has none. Where the docs have no
/// `# Safety`, the contract is one of its own, which says first, of a kernel that is an `unsafe fn`, that it trusts the
/// pointers it is given; else it follows the one they have. It is located at `at`, a token of the user's: Clippy leaves
/// out the attributes of a macro, and would ask the kernel for the `# Safety` that it has.
fn contract(mut docs: TokenStream, named: &[String], features: &[String], safe: bool, at: Span) -> TokenStream {
    let needs = needs(features, safe);
    if named.is_empty() && needs.is_empty() {
        return docs;
    }
    let own = !docs.to_string().contains("# Safety");
    let mut first = String::new();
    if own && !safe {
        first += "The kernel trusts the pointers it is given: each must be to memory as long as the kernel reads and \
                  writes through it. ";
    }
    if !named.is_empty() {
        first += &format!("Nothing else touches the module's memory, {}, while it runs.", list(named));
    }
    let mut written = String::new();
    if own {
        written += "#[doc = \"\"]\n#[doc = \" # Safety\"]\n";
    }
    for paragraph in [first, needs] {
        let paragraph = paragraph.trim();
        if !paragraph.is_empty() {
            written += &format!("#[doc = \"\"]\n#[doc = {:?}]\n", format!(" {paragraph}"));
        }
    }
    docs.extend(respan(parse(&written), at));
    docs
}

/// `names` in words, each in backticks: `` `A` ``, `` `A` and `B` ``, `` `A`, `B` and `C` ``.
fn list(names: &[String]) -> String {
    let quoted: Vec<String> = names.iter().map(|n| format!("`{n}`")).collect();
    match quoted.split_last() {
        Some((last, [])) => last.clone(),
        Some((last, rest)) => format!("{} and {last}", rest.join(", ")),
        None => String::new(),
    }
}

/// Static `s` of a module of x86 kernels as Rust has it, as written but for its type and what it holds: `made`, a
/// `kernite_cpu::Global` of its elements at the multiple its type says, holding what it says or zeros; for a static
/// that did not compile, beside its error, `made` is the static as its type has it, holding zeros, if only what it
/// holds is wrong; if not, `None`, and a `kernite_cpu::Stub` stands for it, which leaves the type of its elements to
/// the code that uses it. In a crate without `kernite-cpu`, its type is [`missing`]'s, with the error that says so
/// after it, under `cfg`, the static's own. A far static, past `rip`'s reach, is a reference to memory that [`linked`]
/// defines past the code and data in reach of `rip`, and a static of an `extern` block declares: Rust would lay a
/// static of its own out among them, and make it in time and memory that grow with its size. The kernels load the
/// address from the reference. A crate of the `dylib` kind exports the reference, as it exports only the symbols that
/// Rust defines, which the memory is not. What each static holds, a far one's reference and a stub too, is in a block
/// with a `global_asm!` that makes its symbol protected: [`initializer`] says why. On a system that is not Linux, it is
/// an error at its name. In each case what it holds comes after the `places` of the static, those of its type and of
/// what it holds that name an item: [`assigned`] says why.
pub(crate) fn cpu_static(
    s: &lang::Static,
    made: Option<&HostStatic>,
    cfg: &TokenStream,
    top: &[TokenTree],
    places: TokenStream,
) -> TokenStream {
    // Its name as written, after `static`.
    let Some(name) = name_after(&s.vis, top) else { return TokenStream::new() };
    let far = made.filter(|m| m.far && depends_on(CPU));
    let mut out: TokenStream = within(top, &s.attrs).iter().cloned().collect();
    out.extend(within(top, &s.vis).iter().cloned());
    out.extend(parse("static"));
    out.extend([name.clone()]);
    if !depends_on(CPU) {
        let (ty, error) = missing(cfg, name.span(), "a static of x86 kernels", CPU);
        out.extend(parse(":"));
        out.extend(ty);
        out.extend(assigned(places, "[]"));
        out.extend(error);
        return out;
    }
    let global = |m: &HostStatic| format!("::kernite_cpu::Global<{}, {}, {}>", primitive(&m.elem), m.len, m.align);
    if let Some(m) = far {
        let sym = symbol(name);
        let memory = format!("__kernite_{sym}");
        out.extend(parse(&format!(": &{}", global(m))));
        out.extend(initializer(name, &format!("&{memory}"), places));
        let under = |attr: &str| cfg.clone().into_iter().chain(parse(attr)).collect::<TokenStream>();
        let mut declared = under(r#"#[cfg(target_os = "linux")]"#);
        declared.extend(linked(&sym, m));
        let message = format!(
            "`{name}` is past what is in reach of `rip` beside the statics before it, and kernite defines such a \
             static for Linux's linker alone: build for Linux, or make the statics smaller"
        );
        declared.extend(error_under(&under(r#"#[cfg(not(target_os = "linux"))]"#), name.span(), &message));
        declared.extend(cfg.clone());
        declared.extend(parse("unsafe extern \"C\""));
        let item = format!("#[link_name = concat!(module_path!(), \"::{sym}\")] safe static {memory}: {};", global(m));
        declared.extend([TokenTree::Group(Group::new(Delimiter::Brace, parse(&item)))]);
        declared.extend(out);
        return declared;
    }
    let (ty, value) = match made {
        Some(m) => {
            let ty = global(m);
            let value = match m.init.iter().all(|&b| b == 0) {
                true => format!("<{ty}>::zeroed()"),
                false => format!("<{ty}>::new([{}])", elements(m).join(", ")),
            };
            (ty, value)
        }
        None => ("::kernite_cpu::Stub".to_string(), "::kernite_cpu::Stub".to_string()),
    };
    out.extend(parse(&format!(": {ty}")));
    out.extend(initializer(name, &value, places));
    out
}

/// What the static `name` holds, `value`, after its `places`, in a block with a `global_asm!` that makes the symbol of
/// the static protected, on Linux. A symbol that a crate of the `dylib` kind exports is default, which a program may
/// replace, so the linker refuses a kernel's displacement from `rip` to it:
/// `relocation R_X86_64_PC32 cannot be used against symbol`. A protected symbol is exported as well, and the crate's
/// own kernels bind to the crate's own. The `global_asm!` names the static, which may be deprecated. Rust tells of no
/// use of a deprecated item inside the item itself, so the `global_asm!` is in the static's own initializer, and no
/// `allow` is written, which a crate that forbids `deprecated` would refuse. It ends the static, at its name: see
/// [`ending_at`].
fn initializer(name: &TokenTree, value: &str, places: TokenStream) -> TokenStream {
    let text = format!(
        "#[cfg(target_os = \"linux\")] mod __kernite_protected {{ \
         ::core::arch::global_asm!(\".protected {{}}\", sym super::{name}); }} {value}"
    );
    ending_at(assigned(places, &text), name.span())
}

/// `= { .. };`, what a static holds: the `places` of the static, in an item of no name, and then `value`. The block is
/// in the static, so Rust tells of each place at the level the static sets, as it does in a static of its own, and an
/// `expect` of the static is met by a place of it.
fn assigned(places: TokenStream, value: &str) -> TokenStream {
    let mut block = unnamed(places);
    block.extend(parse(value));
    let mut out = parse("=");
    out.extend([TokenTree::Group(Group::new(Delimiter::Brace, block))]);
    out.extend(parse(";"));
    out
}

/// The name of a static as the linker has it after the path of its module, which `module_path!` gives, for the statics
/// of two modules to be two: as written, but without the `r#` of a raw name.
fn symbol(name: &TokenTree) -> String {
    name.to_string().trim_start_matches("r#").to_string()
}

/// The `global_asm!` that defines `m`, a far static, as the symbol `name` after its module's path, in a section
/// `.lbss`, which ELF has for what an x86-64 program reaches by its address and no displacement: the linker places it
/// last, past the code and the rest of the data, which stay in reach of `rip`. It takes no room in the file, and is
/// zeros when the program starts. Its symbol is an object of its size, as a static of Rust's is, and hidden, which
/// keeps it out of the dynamic symbol table, so the linker resolves each reference to it.
fn linked(name: &str, m: &HostStatic) -> TokenStream {
    let sym = |before: &str, after: &str| format!(r#"concat!("{before}", module_path!(), "::{name}{after}")"#);
    let bytes = (m.len * m.size).next_multiple_of(m.align);
    parse(&format!(
        "::core::arch::global_asm!({}, \".balign {}\", {}, {}, {}, {}, {}, \".zero {bytes}\", \".popsection\");",
        sym(r#".pushsection \".lbss."#, r#"\",\"awl\",@nobits"#),
        m.align,
        sym(r#".globl \""#, r#"\""#),
        sym(r#".hidden \""#, r#"\""#),
        sym(r#".type \""#, r#"\", @object"#),
        sym(r#".size \""#, &format!(r#"\", {bytes}"#)),
        sym(r#"\""#, r#"\":"#),
    ))
}

/// What each element of `m` holds at first, as Rust writes it: `7u32`, `-1i8`, or a float by its
/// bits, `f32::from_bits(0x3f800000)`, so that it is the same bits.
fn elements(m: &HostStatic) -> Vec<String> {
    let size = m.size as usize;
    Vec::from_fn(m.len as usize, |i| {
        let bytes = m.init.get(i * size..(i + 1) * size).unwrap_or_default();
        let bits = bytes.iter().rev().fold(0, |bits, &byte| bits << 8 | u64::from(byte));
        // The bits as a signed number of the element's width.
        let signed = ((bits << (64 - 8 * size)) as i64) >> (64 - 8 * size);
        match m.elem.as_str() {
            t @ ("f32" | "f64") => format!("{}::from_bits({bits:#x})", primitive(t)),
            t if t.starts_with('i') => format!("{signed}{t}"),
            t => format!("{bits}{t}"),
        }
    })
}

/// The operands by which the code of an instance of a kernel of `module` names each function it
/// `calls`: a function of no compile-time arguments by its name as written, `r#fn` as well, and an
/// instance, `tile::<4>`, by the name [`instance_name`] gives the function [`naked`] makes of it. A function of no
/// compile-time arguments is at the place `at` says the kernel first writes its name.
fn callees(
    module: &lang::Module,
    calls: &[Call],
    top: &[TokenTree],
    at: &dyn std::ops::Fn(&str) -> Span,
) -> TokenStream {
    let mut out = TokenStream::new();
    for (i, call) in calls.iter().enumerate() {
        let Some(item) = position(module, &call.function).and_then(|j| module.items.get(j)) else { continue };
        let path = match (call.instance, name_at(item, top, at(&call.function))) {
            (Some(k), _) => parse(&instance_name(&call.function, k)),
            (None, Some(name)) => name.into(),
            (None, None) => continue,
        };
        out.extend(operand(&format!("f{i}"), path, item, top));
    }
    out
}

/// The name of the function that [`naked`] makes of instance `k` of the kernel `function` of
/// compile-time arguments. The code of a kernel that calls the instance names it by this name.
fn instance_name(function: &str, k: usize) -> String {
    format!("__kernite_{function}_I{k}")
}

/// `ty`, a type of `f`'s, as an instance writes it, and what `f` writes after a value of it to pass
/// it on: a pointer is `*const ()` or `*mut ()` to the instance, whatever it points to, and
/// `as *const ()` or `as *mut ()` in `f`, which is a cast and not a call, so that Rust says nothing
/// of the type it points to; an array or a slice by reference is the same, after a cast to the
/// pointer to it, `as *const _`, which Rust makes of a reference alone; a number is the primitive,
/// as it is in `f`, and nothing.
fn erased(ty: &[TokenTree]) -> (String, String) {
    match (pointer(ty), array_ref(ty)) {
        (Some((kind, _)), _) => (format!("*{kind} ()"), format!(" as *{kind} ()")),
        (_, Some((kind, ..))) => (format!("*{kind} ()"), format!(" as *{kind} _ as *{kind} ()")),
        (None, None) => (given_back(ty).to_string(), String::new()),
    }
}

/// The instance `f` calls, as a Rust expression: the only one, `arms` having one that takes no arguments, by its name,
/// else the one `f`'s compile-time arguments [`pick`].
fn chosen(f: &Fn, arms: &[(Vec<Value>, String)], top: &[TokenTree]) -> String {
    match arms {
        [(args, kernel)] if args.is_empty() => kernel.clone(),
        _ => pick(f, arms, top),
    }
}

/// The function that runs `f`, a cooperative x86 kernel, on the threads of a grid, of `f`'s
/// name and generics: it takes the grid, then `f`'s parameters, and each thread calls the
/// instance `kernels` has for `f`'s compile-time arguments, given its place in the grid. A
/// pointer reaches the threads as the address it is, which each makes a pointer again, as
/// Rust's pointers do not cross threads. Without `kernels`, it never runs, for a kernel that
/// did not compile. Its safety contract says that nothing else touches `named`, the module's memory, while the threads
/// run, and, after it, which CPU `features` the kernels need, as [`needs`] says.
/// It binds `__kernite_grid` for itself, which a parameter of `f` may be named as well; a compile-time argument may
/// not, which is an error at its name, after the function, under `cfg`, `f`'s own; and the function never runs, as one
/// of a kernel that did not compile. In a crate without `kernite-cpu`, it takes any grid, gives [`missing`]'s type,
/// with the error that says so after it, and never runs. Its result is [`launcher_result`]'s. Its body has the `places`
/// of `f`'s code that Rust is told of by a statement, as [`naked`] says.
fn grid_launcher(
    f: &Fn,
    kernels: Option<&[(Vec<Value>, String)]>,
    features: &[String],
    named: &[String],
    cfg: &TokenStream,
    top: &[TokenTree],
    places: TokenStream,
) -> TokenStream {
    let clashes = clashes(f, cfg, &["__kernite_grid"], "the launcher", top);
    let kernels = if clashes.is_empty() { kernels } else { None };
    let mut out = attrs(f, top);
    out.extend(parse(
        "#[doc = \"\"]\n#[doc = \" Every thread of the grid runs at once, as `grid.sync()` needs, each on a core, so a \
         grid has at most as many threads as the process runs at once, [`kernite_cpu::most`]: the CPUs the caller's \
         thread may run on, or fewer where a cgroup limits the process to fewer. More is an error.\"]",
    ));
    let memory = match named {
        [] => String::new(),
        named => format!(" Nothing else touches the module's memory, {}, while they run.", list(named)),
    };
    if !out.to_string().contains("# Safety") {
        out.extend(parse(&format!(
            "#[doc = \"\"]\n#[doc = \" # Safety\"]\n#[doc = \"\"]\n#[doc = {:?}]",
            format!(
                " The kernel trusts the pointers it is given: each must be to memory as long as the kernel's threads \
                 read and write through it, which nothing else touches while they run.{memory}"
            )
        )));
    } else if !memory.is_empty() {
        out.extend(parse(&format!("#[doc = \"\"]\n#[doc = {memory:?}]")));
    }
    // The launcher is an `unsafe fn`, as a cooperative kernel is.
    let needs = needs(features, false);
    if !needs.is_empty() {
        out.extend(parse(&format!("#[doc = \"\"]\n#[doc = {:?}]", format!(" {needs}"))));
    }
    // Clippy says "too many arguments" of a function of more than seven parameters wherever its
    // tokens come from, and the launcher takes the grid as well: so it allows that, and a crate of
    // kernels may not forbid `clippy::too_many_arguments`.
    out.extend(parse("#[allow(clippy::too_many_arguments)]"));
    // Its visibility, then `unsafe fn` and its name and generics as written.
    out.extend(visibility(f, top).iter().cloned());
    out.extend(parse("unsafe fn"));
    out.extend(top.get(f.head.end + 1..f.params_at).unwrap_or_default().iter().cloned());
    // Its types are `kernite_cpu`'s, but where the crate has none: any grid goes in, and the error,
    // said once, comes out.
    let (grid, result, missed) = match depends_on(CPU) {
        true => (
            parse("::kernite_cpu::Grid"),
            parse("::core::result::Result<(), ::kernite_cpu::Error>"),
            TokenStream::new(),
        ),
        false => {
            let name = span_of(name_after(&f.head, top));
            let (result, missed) = missing(cfg, name, "a cooperative x86 kernel", CPU);
            (parse("impl ::core::marker::Sized"), result, missed)
        }
    };
    let mut params = hygienic("$__kernite_grid:", kernels.is_some());
    params.extend(grid);
    params.extend(parse(","));
    params.extend(value_params(f, top));
    out.extend([TokenTree::Group(Group::new(Delimiter::Parenthesis, params))]);
    out.extend(parse("->"));
    out.extend(launcher_result(f, top, result));
    let text = match kernels {
        Some(kernels) => {
            let kernel = chosen(f, kernels, top);
            let (mut addresses, mut args) = (String::new(), vec![]);
            for Written { name, ty, .. } in written_params(f, top) {
                match pointer(&ty) {
                    Some((kind, _)) => {
                        let again = if kind == "mut" { "_mut" } else { "" };
                        addresses += &format!("let {name} = ({name} as *{kind} ()).expose_provenance();\n");
                        args.push(format!("::core::ptr::with_exposed_provenance{again}({name})"));
                    }
                    None => args.push(name.to_string()),
                }
            }
            args.push("$__kernite_grid".to_string());
            format!(
                "{addresses}::kernite_cpu::launch($__kernite_grid, \
                 move |$__kernite_grid: &::kernite_cpu::Place| unsafe {{ ({kernel})({}) }})",
                args.join(", ")
            )
        }
        None => format!("{}::core::unreachable!()", never(f, top)),
    };
    let mut body = asks(f, top, written);
    body.extend(unnamed(places));
    body.extend(hygienic(&text, kernels.is_some()));
    out.extend([TokenTree::Group(Group::new(Delimiter::Brace, body))]);
    out.extend(missed);
    out.extend(clashes);
    out
}

/// The CPU features past x86-64's own the code of `instances` needs, each once, by name.
fn features(instances: &[NativeInstance]) -> Vec<String> {
    let mut features: Vec<String> = instances.iter().flat_map(|k| k.naked.needs.iter().cloned()).collect();
    features.sort();
    features.dedup();
    features
}

/// The paragraph that names the CPU `features`, past x86-64's own, which the code of a kernel needs; none if it needs
/// none. For a kernel that is an `unsafe fn` it says that nothing checks for them when it runs; for one that is not,
/// that Rust checks where it is called, for only a function that has them calls it without `unsafe`, and any other
/// calls it in an `unsafe` block, which says why the CPU has them.
fn needs(features: &[String], safe: bool) -> String {
    if features.is_empty() {
        return String::new();
    }
    let (noun, them) = if let [_] = features { ("feature", "it") } else { ("features", "them") };
    let checked = match safe {
        true => format!(
            "which Rust checks where it is called: only a function that has {them}, \
             `#[target_feature(enable = \"{}\")]`, calls it without `unsafe`, and any other calls it in an `unsafe` \
             block, which says why the CPU has {them}",
            features.join(",")
        ),
        false => "which nothing checks for when it runs".to_string(),
    };
    format!("Needs the CPU {noun} {}, past x86-64's own, {checked}.", list(features))
}

/// `f`, an x86 kernel that does not compile, or that gives back a type with a `_` in it, or a
/// cooperative one in a crate without `kernite-cpu`, as a function that never runs: of `f`'s
/// signature, or of its launcher's, for a cooperative kernel. One of compile-time arguments has its
/// `instances` as functions too, named as [`naked`] would name them, for kernels that call one to
/// name. Its safety contract says what the kernel's would of `named`, the module's memory. Each
/// instance is under `cfg`, `f`'s own, as `f` is.
///
/// Its body has the `places` of `f`'s code that Rust is told of by a statement, as [`naked`] says.
pub(crate) fn x86_stub(
    f: &Fn,
    instances: usize,
    named: &[String],
    cfg: &TokenStream,
    top: &[TokenTree],
    places: TokenStream,
) -> TokenStream {
    if f.cooperative.is_some() {
        return grid_launcher(f, None, &[], named, cfg, top, places);
    }
    let mut out = stub(f, top, places);
    if !f.generics.is_empty() {
        for k in 0..instances {
            let name = instance_name(&f.name, k);
            out.extend(cfg.clone());
            out.extend(parse(&format!(
                "#[doc(hidden)]\nunsafe extern \"sysv64\" fn {name}() {{ ::core::unreachable!() }}\n"
            )));
        }
    }
    out
}
