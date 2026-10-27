//! What x86 kernels become in Rust: naked functions and their safety contracts, the launchers
//! of cooperative ones, stubs for those that do not compile, and the module's statics.

use proc_macro::{Delimiter, Group, TokenStream, TokenTree};

use kgen::X86Static;
use kgen::lang::{self, Fn, ItemKind, Value};

use crate::rust::{attrs, is_ident, is_punct, parse, pick, signature, stub, visibility, written_params};

/// `f` as an `#[unsafe(naked)]` function whose body is the listing of its one instance, and
/// the statics its constants are, in a module beside it. For a function of compile-time
/// arguments, each instance is such a function in that module, and `f` is one that calls the
/// instance its arguments pick. So is each instance of a cooperative kernel, which takes the
/// thread's place in the grid after `f`'s parameters, and `f` is its [`grid_launcher`]. Each
/// names the functions of `module` it calls by the paths [`callees`] gives, and its statics by
/// those [`reached`] gives. Its safety contract says that nothing else touches `named`, the
/// module's memory, while it runs.
pub(crate) fn naked(
    module: &lang::Module,
    f: &Fn,
    instances: &[(Vec<Value>, kgen::Listing)],
    named: &[&str],
    top: &[TokenTree],
) -> TokenStream {
    let hidden = format!("__kgen_{}", f.name);
    let cooperative = f.cooperative.is_some();
    let mut out = TokenStream::new();
    let mut items = String::new();
    let mut statics = vec![];
    if let [(args, listing)] = instances
        && args.is_empty()
        && !cooperative
    {
        let mut docs = attrs(f, top);
        docs.extend(needs(instances));
        out.extend(contract(docs, named));
        out.extend(parse("#[unsafe(naked)]"));
        out.extend(signature(f, top));
        let code = asm(
            listing,
            &format!("{hidden}::C"),
            &callees(module, listing, false, top),
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
            params.push("_: &::kgen_cpu::Place".to_string());
            tys.push("&::kgen_cpu::Place".to_string());
        }
        let ret: TokenStream = f.ret.as_ref().map_or_else(TokenStream::new, |r| {
            parse("->").into_iter().chain(top[r.tokens.clone()].iter().cloned()).collect()
        });
        let ty = format!("unsafe extern \"sysv64\" fn({}) {ret}", tys.join(", "));
        let mut arms = vec![];
        for (k, (args, listing)) in instances.iter().enumerate() {
            items += &format!(
                "#[unsafe(naked)]\npub unsafe extern \"sysv64\" fn I{k}({}) {ret} {{ {} }}\n",
                params.join(", "),
                asm(
                    listing,
                    &format!("I{k}C"),
                    &callees(module, listing, true, top),
                    &reached(module, listing, true, top)
                )
            );
            statics.push((format!("I{k}C"), listing));
            arms.push((args.clone(), format!("{hidden}::I{k} as {ty}")));
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
        out.extend(parse(&format!(
            "#[doc(hidden)]\n#[allow(dead_code, non_snake_case)]\nmod {hidden} {{\n{items}}}\n"
        )));
    }
    out
}

/// The body of a naked function whose code is `listing`'s, its constants the statics
/// `{statics}0`, `{statics}1` and on, the functions it calls at the paths `funcs`, and the
/// module's memory it names as `memory` has it: each operand's name, and the path to it.
fn asm(listing: &kgen::Listing, statics: &str, funcs: &[String], memory: &[(String, String)]) -> String {
    let mut asm = String::from("::core::arch::naked_asm!(");
    for line in listing.code() {
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
fn reached(module: &lang::Module, listing: &kgen::Listing, inside: bool, top: &[TokenTree]) -> Vec<(String, String)> {
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
fn contract(mut docs: TokenStream, named: &[&str]) -> TokenStream {
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
fn list(names: &[&str]) -> String {
    let quoted: Vec<String> = names.iter().map(|n| format!("`{n}`")).collect();
    match quoted.split_last() {
        Some((last, [])) => last.clone(),
        Some((last, rest)) => format!("{} and {last}", rest.join(", ")),
        None => String::new(),
    }
}

/// Static `s` of a module of x86 kernels as Rust has it, as written but for its type and what it
/// holds: `made`, a `kgen_cpu::Global` of its elements at the multiple its type says, holding
/// what it says or zeros; or what stands for a static that did not compile, beside its error.
pub(crate) fn cpu_static(s: &lang::Static, made: Option<&X86Static>, top: &[TokenTree]) -> TokenStream {
    let mut out: TokenStream = s.attrs.clone().map(|i| top[i].clone()).collect();
    out.extend(top[s.vis.clone()].iter().cloned());
    out.extend(parse("static"));
    // Its name as written, after `static`.
    out.extend([top[s.vis.end + 1].clone()]);
    out.extend(parse(&match made {
        Some(m) => {
            let ty = format!("::kgen_cpu::Global<{}, {}, {}>", m.elem, m.len, m.align);
            match m.init.iter().all(|&b| b == 0) {
                true => format!(": {ty} = <{ty}>::zeroed();"),
                false => format!(": {ty} = <{ty}>::new([{}]);", elements(m).join(", ")),
            }
        }
        None => ": ::kgen_cpu::Stub = ::kgen_cpu::Stub;".to_string(),
    }));
    out
}

/// What each element of `m` holds at first, as Rust writes it: `7u32`, `-1i8`, or a float by its
/// bits, `f32::from_bits(0x3f800000)`, so that it is the same bits.
fn elements(m: &X86Static) -> Vec<String> {
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
            match m.elem {
                t @ ("f32" | "f64") => format!("{t}::from_bits({bits:#x})"),
                t if t.starts_with('i') => format!("{signed}{t}"),
                t => format!("{bits}{t}"),
            }
        })
        .collect()
}

/// The paths by which the code of `listing`, of a kernel of `module`, names each function it
/// calls, from beside the kernels or from `inside` a kernel's hidden module: a function of no
/// compile-time arguments by its name as written, `r#fn` as well, and an instance, `tile::<4>`,
/// as the function in its function's hidden module, `__kgen_tile::I0`, that [`naked`] makes it.
fn callees(module: &lang::Module, listing: &kgen::Listing, inside: bool, top: &[TokenTree]) -> Vec<String> {
    let up = if inside { "super::" } else { "" };
    let path = |name: &str| {
        let base = name.split("::<").next().unwrap_or(name);
        let g = kgen::function(module, base).expect("a kernel calls a function of its module");
        if g.generics.is_empty() {
            // Its name as written, after `fn`.
            return format!("{up}{}", top[g.head.end + 1]);
        }
        let instances = lang::instances(module, g).unwrap_or_default();
        let k = instances.iter().position(|args| lang::instance(g, args).is_ok_and(|n| n == name));
        format!("{up}__kgen_{base}::I{}", k.expect("a kernel calls an instance its function lists"))
    };
    listing.funcs.iter().map(|name| path(name)).collect()
}

/// The function that runs `f`, a cooperative x86 kernel, on the threads of a grid, of `f`'s
/// name and generics: it takes the grid, then `f`'s parameters, and each thread calls the
/// instance `kernels` has for `f`'s compile-time arguments, given its place in the grid. A
/// pointer reaches the threads as the address it is, which each makes a pointer again, as
/// Rust's pointers do not cross threads. Without `kernels`, it never runs, for a kernel that
/// did not compile. `needs` is the doc of the CPU features the kernels need. Its safety
/// contract says that nothing else touches `named`, the module's memory, while the threads run.
fn grid_launcher(
    f: &Fn,
    kernels: Option<&[(Vec<Value>, String)]>,
    needs: TokenStream,
    named: &[&str],
    top: &[TokenTree],
) -> TokenStream {
    let mut out = attrs(f, top);
    out.extend(needs);
    out.extend(parse(
        "#[doc = \"\"]\n#[doc = \" Every thread of the grid runs at once, as `grid.sync()` needs, each on a core, so a \
         grid of more threads than [`kgen_cpu::most`] says the machine runs is an error.\"]",
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
    out.extend(parse(if kernels.is_some() {
        "#[allow(clippy::too_many_arguments)]"
    } else {
        "#[allow(clippy::too_many_arguments, unused_variables)]"
    }));
    // Its visibility, then `unsafe fn` and its name and generics as written.
    out.extend(visibility(f, top).iter().cloned());
    out.extend(parse("unsafe fn"));
    out.extend(top[f.head.end + 1..f.params_at].iter().cloned());
    let mut params = parse("grid: ::kgen_cpu::Grid, ");
    for (param, _) in written_params(f, top) {
        params.extend(param);
        params.extend(parse(","));
    }
    out.extend([TokenTree::Group(Group::new(Delimiter::Parenthesis, params))]);
    out.extend(parse("-> ::core::result::Result<(), ::kgen_cpu::Error>"));
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
            args.push("grid".to_string());
            format!(
                "{addresses}::kgen_cpu::launch(grid, &|grid: &::kgen_cpu::Place| unsafe {{ ({kernel})({}) }})",
                args.join(", ")
            )
        }
        None => "::core::unreachable!()".to_string(),
    };
    out.extend([TokenTree::Group(Group::new(Delimiter::Brace, parse(&body)))]);
    out
}

/// The doc that names the CPU features past x86-64's own the code of `instances` needs, which
/// nothing checks for when it runs; none if it needs none.
fn needs(instances: &[(Vec<Value>, kgen::Listing)]) -> TokenStream {
    let mut features: Vec<&str> = instances.iter().flat_map(|(_, l)| l.features.iter().map(|f| f.name())).collect();
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

/// `f`, an x86 kernel that does not compile, as a function that never runs: of `f`'s
/// signature, or of its launcher's, for a cooperative kernel. One of compile-time arguments has
/// its `instances` in its hidden module too, as [`naked`] would put them there, for kernels
/// that call one to name. Its safety contract says what the kernel's would of `named`, the
/// module's memory.
pub(crate) fn x86_stub(f: &Fn, instances: usize, named: &[&str], top: &[TokenTree]) -> TokenStream {
    if f.cooperative.is_some() {
        return grid_launcher(f, None, TokenStream::new(), named, top);
    }
    let mut out = stub(f, top);
    if !f.generics.is_empty() && instances > 0 {
        let items: String = (0..instances)
            .map(|k| format!("pub unsafe extern \"sysv64\" fn I{k}() {{ ::core::unreachable!() }}\n"))
            .collect();
        out.extend(parse(&format!(
            "#[doc(hidden)]\n#[allow(dead_code, non_snake_case)]\nmod __kgen_{} {{\n{items}}}\n",
            f.name
        )));
    }
    out
}
