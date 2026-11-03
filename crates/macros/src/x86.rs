//! What x86 kernels become in Rust: naked functions and their safety contracts, the launchers
//! of cooperative ones, stubs for those that do not compile, and the module's statics.

use proc_macro::{Delimiter, Group, Span, TokenStream, TokenTree};

use kernite::lang::{self, Fn, ItemKind, Value};
use kernite::{Call, HostStatic, Naked, NativeInstance};

use crate::rust::{
    attrs, clashes, error, hygienic, is_ident, is_punct, never, parse, pick, signature, stub, visibility,
    written_params,
};

/// `f` as an `#[unsafe(naked)]` function whose body is the listing of its one instance, and
/// the statics its constants are, in a module beside it. For a function of compile-time
/// arguments, each instance is such a function in that module, and `f` is one that calls the
/// instance its arguments pick. So is each instance of a cooperative kernel, which takes the
/// thread's place in the grid after `f`'s parameters, and `f` is its [`grid_launcher`]. Each
/// names the functions of `module` it calls by the paths [`callees`] gives, and its statics by
/// those [`reached`] gives. Its safety contract says that nothing else touches `named`, the
/// module's memory, while it runs. A cooperative kernel of a crate without `kernite-cpu` is
/// [`x86_stub`]'s instead, with the error that says so.
pub(crate) fn naked(
    module: &lang::Module,
    f: &Fn,
    instances: &[NativeInstance],
    named: &[String],
    top: &[TokenTree],
) -> TokenStream {
    let hidden = format!("__kernite_{}", f.name);
    let cooperative = f.cooperative.is_some();
    if cooperative && !has_cpu() {
        return x86_stub(f, instances.len(), named, top);
    }
    let mut out = TokenStream::new();
    let mut items = String::new();
    let mut statics = vec![];
    if let [one] = instances
        && one.args.is_empty()
        && !cooperative
    {
        let listing = &one.naked;
        let mut docs = attrs(f, top);
        docs.extend(needs(instances));
        out.extend(contract(docs, named));
        out.extend(parse("#[unsafe(naked)]"));
        out.extend(signature(f, top));
        let code = asm(
            listing,
            &format!("{hidden}::C"),
            &callees(module, &one.calls, false, top),
            &reached(module, listing, false, top),
        );
        out.extend([TokenTree::Group(Group::new(Delimiter::Brace, parse(&code)))]);
        statics.push(("C".to_string(), listing));
    } else {
        // Each instance's parameters and their types: `f`'s, then a cooperative kernel's place.
        // `f` passes its own on by their names as written, before the colon.
        let (mut params, mut tys, mut names) = (vec![], vec![], vec![]);
        for (param, colon) in written_params(f, top) {
            params.push(param.iter().cloned().collect::<TokenStream>().to_string());
            tys.push(param[colon + 1..].iter().cloned().collect::<TokenStream>().to_string());
            names.push(param[colon - 1].to_string());
        }
        if cooperative {
            params.push("_: &::kernite_cpu::Place".to_string());
            tys.push("&::kernite_cpu::Place".to_string());
        }
        let ret: TokenStream = f.ret.as_ref().map_or_else(TokenStream::new, |r| {
            parse("->").into_iter().chain(top[r.tokens.clone()].iter().cloned()).collect()
        });
        let ty = format!("unsafe extern \"sysv64\" fn({}) {ret}", tys.join(", "));
        let mut arms = vec![];
        for (k, instance) in instances.iter().enumerate() {
            let listing = &instance.naked;
            items += &format!(
                "#[unsafe(naked)]\npub unsafe extern \"sysv64\" fn I{k}({}) {ret} {{ {} }}\n",
                params.join(", "),
                asm(
                    listing,
                    &format!("I{k}C"),
                    &callees(module, &instance.calls, true, top),
                    &reached(module, listing, true, top)
                )
            );
            statics.push((format!("I{k}C"), listing));
            arms.push((instance.args.clone(), format!("{hidden}::I{k} as {ty}")));
        }
        if cooperative {
            out.extend(grid_launcher(f, Some(&arms), needs(instances), named, top));
        } else {
            let mut docs = attrs(f, top);
            docs.extend(needs(instances));
            out.extend(contract(docs, named));
            out.extend(parse("#[inline(always)]"));
            out.extend(signature(f, top));
            let body = format!("unsafe {{ ({})({}) }}", pick(f, &arms, top), names.join(", "));
            out.extend([TokenTree::Group(Group::new(Delimiter::Brace, parse(&body)))]);
        }
    }
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
        out.extend(parse(&format!("#[doc(hidden)]\nmod {hidden} {{\n{items}}}\n")));
    }
    out
}

/// The body of a naked function whose code is `listing`'s, its constants the statics
/// `{statics}0`, `{statics}1` and on, the functions it calls at the paths `funcs`, and the
/// module's memory it names as `memory` has it: each operand's name, and the path to it.
fn asm(listing: &Naked, statics: &str, funcs: &[String], memory: &[(String, String)]) -> String {
    let mut asm = String::from("::core::arch::naked_asm!(");
    for line in &listing.code {
        asm += &format!("{line:?}, ");
    }
    for i in 0..listing.consts.len() {
        asm += &format!("c{i} = sym {statics}{i}, ");
    }
    for (i, path) in funcs.iter().enumerate() {
        asm += &format!("f{i} = sym {path}, ");
    }
    for (name, path) in memory {
        asm += &format!("{name} = sym {path}, ");
    }
    asm + ")"
}

/// The operand by which the code of `listing`, of a kernel of `module`, names each static it
/// names, and the path to the static from beside the kernels or from `inside` a kernel's hidden
/// module: its name as written, `COUNTS`, or `r#type`, whose `{type}` in the code names it.
fn reached(module: &lang::Module, listing: &Naked, inside: bool, top: &[TokenTree]) -> Vec<(String, String)> {
    let up = if inside { "super::" } else { "" };
    let written = |name: &str| {
        let s = module.items.iter().find_map(|item| match &item.kind {
            ItemKind::Static(s) if s.name == name => Some(s),
            _ => None,
        });
        top[s.expect("a kernel names a static of its module").vis.end + 1].to_string()
    };
    listing.statics.iter().map(|name| written(name)).map(|w| (w.clone(), format!("{up}{w}"))).collect()
}

/// `docs`, a kernel's attributes, with what its safety contract says of `named`, the memory of
/// the module's: that nothing else touches it while the kernel runs. A contract the docs do not
/// have says that the kernel trusts the pointers it is given as well.
fn contract(mut docs: TokenStream, named: &[String]) -> TokenStream {
    if named.is_empty() {
        return docs;
    }
    let memory = format!("Nothing else touches the module's memory, {}, while it runs.", list(named));
    if !docs.to_string().contains("# Safety") {
        docs.extend(parse(&format!(
            "#[doc = \"\"]\n#[doc = \" # Safety\"]\n#[doc = \"\"]\n#[doc = {:?}]",
            format!(
                " The kernel trusts the pointers it is given: each must be to memory as long as the kernel reads and \
                 writes through it. {memory}"
            )
        )));
    } else {
        docs.extend(parse(&format!("#[doc = \"\"]\n#[doc = {:?}]", format!(" {memory}"))));
    }
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

/// Static `s` of a module of x86 kernels as Rust has it, as written but for its type and what it
/// holds: `made`, a `kernite_cpu::Global` of its elements at the multiple its type says, holding
/// what it says or zeros; or what stands for a static that did not compile, beside its error.
/// In a crate without `kernite-cpu`, its type is [`missing`]'s, with the error that says so.
pub(crate) fn cpu_static(s: &lang::Static, made: Option<&HostStatic>, top: &[TokenTree]) -> TokenStream {
    let mut out: TokenStream = s.attrs.clone().map(|i| top[i].clone()).collect();
    out.extend(top[s.vis.clone()].iter().cloned());
    out.extend(parse("static"));
    // Its name as written, after `static`.
    let name = &top[s.vis.end + 1];
    out.extend([name.clone()]);
    if !has_cpu() {
        out.extend(parse(":"));
        out.extend(missing(name.span(), "a static of x86 kernels"));
        out.extend(parse("= [];"));
        return out;
    }
    out.extend(parse(&match made {
        Some(m) => {
            let ty = format!("::kernite_cpu::Global<{}, {}, {}>", m.elem, m.len, m.align);
            match m.init.iter().all(|&b| b == 0) {
                true => format!(": {ty} = <{ty}>::zeroed();"),
                false => format!(": {ty} = <{ty}>::new([{}]);", elements(m).join(", ")),
            }
        }
        None => ": ::kernite_cpu::Stub = ::kernite_cpu::Stub;".to_string(),
    }));
    out
}

/// Whether the crate being compiled depends on `kernite-cpu`, which the Rust of a static or a
/// cooperative kernel names.
fn has_cpu() -> bool {
    given(&std::env::args_os().map(|a| a.to_string_lossy().into_owned()).collect::<Vec<_>>())
}

/// Whether `args`, the arguments Rust runs with, give the crate `kernite_cpu`: `--extern
/// kernite_cpu=..`. When they give no crate at all, as when an editor runs the macro in a process
/// of its own, nothing says it is missing.
fn given(args: &[String]) -> bool {
    !args.iter().any(|a| a.starts_with("--extern")) || args.iter().any(|a| a.contains("kernite_cpu"))
}

/// A type that stands for one of `kernite_cpu`'s in a crate without it, and says at `span` that
/// `what` needs the crate. It is an array whose length is the error, which Rust reports once and
/// takes for a type it has reported, so that it says nothing of the code that uses it.
fn missing(span: Span, what: &str) -> TokenStream {
    let mut length = parse("();");
    length.extend(error(span, &format!("{what} needs `kernite-cpu`, which this crate does not depend on")));
    TokenTree::Group(Group::new(Delimiter::Bracket, length)).into()
}

/// What each element of `m` holds at first, as Rust writes it: `7u32`, `-1i8`, or a float by its
/// bits, `f32::from_bits(0x3f800000)`, so that it is the same bits.
fn elements(m: &HostStatic) -> Vec<String> {
    let size = m.size as usize;
    (0..m.len as usize)
        .map(|i| {
            let mut word = [0; 8];
            if let Some(bytes) = m.init.get(i * size..(i + 1) * size) {
                word[..size].copy_from_slice(bytes);
            }
            let bits = u64::from_le_bytes(word);
            // The bits as a signed number of the element's width.
            let signed = ((bits << (64 - 8 * size)) as i64) >> (64 - 8 * size);
            match m.elem.as_str() {
                t @ ("f32" | "f64") => format!("{t}::from_bits({bits:#x})"),
                t if t.starts_with('i') => format!("{signed}{t}"),
                t => format!("{bits}{t}"),
            }
        })
        .collect()
}

/// The paths by which the code of an instance of a kernel of `module` names each function it
/// `calls`, from beside the kernels or from `inside` a kernel's hidden module: a function of no
/// compile-time arguments by its name as written, `r#fn` as well, and an instance, `tile::<4>`, as
/// the function in its function's hidden module, `__kernite_tile::I0`, that [`naked`] makes it.
fn callees(module: &lang::Module, calls: &[Call], inside: bool, top: &[TokenTree]) -> Vec<String> {
    let up = if inside { "super::" } else { "" };
    let path = |call: &Call| match call.instance {
        Some(k) => format!("{up}__kernite_{}::I{k}", call.function),
        None => {
            let g = kernite::function(module, &call.function).expect("a kernel calls a function of its module");
            // Its name as written, after `fn`.
            format!("{up}{}", top[g.head.end + 1])
        }
    };
    calls.iter().map(path).collect()
}

/// The function that runs `f`, a cooperative x86 kernel, on the threads of a grid, of `f`'s
/// name and generics: it takes the grid, then `f`'s parameters, and each thread calls the
/// instance `kernels` has for `f`'s compile-time arguments, given its place in the grid. A
/// pointer reaches the threads as the address it is, which each makes a pointer again, as
/// Rust's pointers do not cross threads. Without `kernels`, it never runs, for a kernel that
/// did not compile. `needs` is the doc of the CPU features the kernels need. Its safety
/// contract says that nothing else touches `named`, the module's memory, while the threads run.
/// It binds `grid` for itself, which a parameter of `f` may be named as well; a compile-time
/// argument may not, which is an error at its name, and the function never runs, as one of a
/// kernel that did not compile. In a crate without `kernite-cpu`, it takes any grid, gives
/// [`missing`]'s type, with the error that says so, and never runs.
fn grid_launcher(
    f: &Fn,
    kernels: Option<&[(Vec<Value>, String)]>,
    needs: TokenStream,
    named: &[String],
    top: &[TokenTree],
) -> TokenStream {
    let clashes = clashes(f, &["grid"], "the launcher", top);
    let kernels = if clashes.is_empty() { kernels } else { None };
    let mut out = attrs(f, top);
    out.extend(needs);
    out.extend(parse(
        "#[doc = \"\"]\n#[doc = \" Every thread of the grid runs at once, as `grid.sync()` needs, each on a core, so a \
         grid of more threads than [`kernite_cpu::most`] says the machine runs is an error.\"]",
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
    // Clippy says "too many arguments" of a function of more than seven parameters wherever its
    // tokens come from, and the launcher takes the grid as well: so it allows that, and a crate of
    // kernels may not forbid `clippy::too_many_arguments`.
    out.extend(parse("#[allow(clippy::too_many_arguments)]"));
    // Its visibility, then `unsafe fn` and its name and generics as written.
    out.extend(visibility(f, top).iter().cloned());
    out.extend(parse("unsafe fn"));
    out.extend(top[f.head.end + 1..f.params_at].iter().cloned());
    // Its types are `kernite_cpu`'s, but where the crate has none: any grid goes in, and the error,
    // said once, comes out.
    let (grid, result) = match has_cpu() {
        true => (parse("::kernite_cpu::Grid"), parse("::core::result::Result<(), ::kernite_cpu::Error>")),
        false => (parse("impl ::core::marker::Sized"), missing(top[f.head.end + 1].span(), "a cooperative x86 kernel")),
    };
    let mut params = hygienic("$grid:", kernels.is_some());
    params.extend(grid);
    params.extend(parse(","));
    for (param, _) in written_params(f, top) {
        params.extend(param);
        params.extend(parse(","));
    }
    out.extend([TokenTree::Group(Group::new(Delimiter::Parenthesis, params))]);
    out.extend(parse("->"));
    out.extend(result);
    let body = match kernels {
        Some(kernels) => {
            let kernel = match kernels {
                [(args, kernel)] if args.is_empty() => kernel.clone(),
                _ => pick(f, kernels, top),
            };
            let (mut addresses, mut args) = (String::new(), vec![]);
            for (param, colon) in written_params(f, top) {
                // Its name as written, before the colon.
                let name = &param[colon - 1];
                match &param[colon + 1..] {
                    [star, kind, ..] if is_punct(star, '*') && (is_ident(kind, "const") || is_ident(kind, "mut")) => {
                        let again = if is_ident(kind, "mut") { "_mut" } else { "" };
                        addresses += &format!("let {name} = {name}.expose_provenance();\n");
                        args.push(format!("::core::ptr::with_exposed_provenance{again}({name})"));
                    }
                    _ => args.push(name.to_string()),
                }
            }
            args.push("$grid".to_string());
            format!(
                "{addresses}::kernite_cpu::launch($grid, &|$grid: &::kernite_cpu::Place| unsafe {{ ({kernel})({}) }})",
                args.join(", ")
            )
        }
        None => never(f, top),
    };
    out.extend([TokenTree::Group(Group::new(Delimiter::Brace, hygienic(&body, kernels.is_some())))]);
    out.extend(clashes);
    out
}

/// The doc that names the CPU features past x86-64's own the code of `instances` needs, which
/// nothing checks for when it runs; none if it needs none.
fn needs(instances: &[NativeInstance]) -> TokenStream {
    let mut features: Vec<&str> = instances.iter().flat_map(|k| k.naked.needs.iter().map(String::as_str)).collect();
    features.sort();
    features.dedup();
    let named: Vec<String> = features.iter().map(|f| format!("`{f}`")).collect();
    let (list, noun) = match named.as_slice() {
        [] => return TokenStream::new(),
        [one] => (one.clone(), "feature"),
        [rest @ .., last] => (format!("{} and {last}", rest.join(", ")), "features"),
    };
    parse(&format!(
        "#[doc = \"\"]\n#[doc = \" Needs the CPU {noun} {list}, past x86-64's own, which nothing checks for when it \
         runs.\"]"
    ))
}

/// `f`, an x86 kernel that does not compile, or a cooperative one in a crate without
/// `kernite-cpu`, as a function that never runs: of `f`'s signature, or of its launcher's, for a
/// cooperative kernel. One of compile-time arguments has its `instances` in its hidden module
/// too, as [`naked`] would put them there, for kernels that call one to name. Its safety contract
/// says what the kernel's would of `named`, the module's memory.
pub(crate) fn x86_stub(f: &Fn, instances: usize, named: &[String], top: &[TokenTree]) -> TokenStream {
    if f.cooperative.is_some() {
        return grid_launcher(f, None, TokenStream::new(), named, top);
    }
    let mut out = stub(f, top);
    if !f.generics.is_empty() && instances > 0 {
        let items: String = (0..instances)
            .map(|k| format!("pub unsafe extern \"sysv64\" fn I{k}() {{ ::core::unreachable!() }}\n"))
            .collect();
        out.extend(parse(&format!("#[doc(hidden)]\nmod __kernite_{} {{\n{items}}}\n", f.name)));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::given;

    /// Whether `given` says the crate has `kernite_cpu`, of the arguments `args` writes.
    fn run(args: &str) -> bool {
        given(&args.split_whitespace().map(String::from).collect::<Vec<_>>())
    }

    #[test]
    fn a_crate_has_kernite_cpu_when_rust_gives_it_and_when_it_gives_no_crate_at_all() {
        let build = "rustc --crate-name app --extern kernite_macros=/t/libkernite_macros.so";
        assert!(run(&format!("{build} --extern kernite_cpu=/t/libkernite_cpu.rlib")));
        assert!(run(&format!("{build} --extern=kernite_cpu=/t/libkernite_cpu.rlib")));
        assert!(!run(build));
        // An editor runs the macro in a process of its own, which Rust's arguments do not reach.
        assert!(run("rust-analyzer-proc-macro-srv"));
    }
}
