//! `#[kernel(name(key = value, ...))]` on a bodiless `unsafe extern "sysv64" fn`: the
//! kernel, rendered by `kgen`, as the function's `naked_asm!` body.
//!
//! ```ignore
//! #[kgen_macros::kernel(iq2_xxs(ahead = 3))]
//! pub unsafe extern "sysv64" fn dot(
//!     row: *const u8, sbs: usize, quants: *const i8, scales: *const f32, grid: *const u64,
//! ) -> f32;
//! ```
//!
//! The declared signature is checked against the kernel's: as many parameters, each of a
//! type SysV passes where the kernel reads it (a pointer or an integer for a general
//! register), and the result likewise. The kernel's constants become statics beside the
//! function, reached through `sym` operands.

use proc_macro::{Delimiter, Group, TokenStream, TokenTree};

use kgen::{Class, Listing};

/// See the crate documentation.
#[proc_macro_attribute]
pub fn kernel(attr: TokenStream, item: TokenStream) -> TokenStream {
    match expand(attr, item) {
        Ok(tokens) => tokens,
        Err(message) => format!("compile_error!({message:?});").parse().expect("a string literal"),
    }
}

fn expand(attr: TokenStream, item: TokenStream) -> Result<TokenStream, String> {
    let (name, params) = kernel_name(attr)?;
    let listing = kgen::render_named(&name, &params).map_err(|e| format!("kgen: {e}"))?;
    let f = declaration(item)?;
    check(&f, &listing)?;
    emit(&f, &listing).parse().map_err(|e| format!("kgen emitted tokens that do not parse: {e}"))
}

/// `name` or `name(key = value, ...)`.
fn kernel_name(attr: TokenStream) -> Result<(String, Vec<(String, i64)>), String> {
    let usage = "expected a kernel, as in `#[kernel(iq2_xxs(ahead = 3))]`";
    let mut tokens = attr.into_iter();
    let Some(TokenTree::Ident(name)) = tokens.next() else {
        return Err(usage.into());
    };
    let mut params = vec![];
    match tokens.next() {
        None => {}
        Some(TokenTree::Group(g)) if g.delimiter() == Delimiter::Parenthesis => {
            let inner: Vec<TokenTree> = g.stream().into_iter().collect();
            for part in inner.split(|t| matches!(t, TokenTree::Punct(p) if p.as_char() == ',')) {
                match part {
                    [] => {}
                    [TokenTree::Ident(key), TokenTree::Punct(eq), TokenTree::Literal(value)] if eq.as_char() == '=' => {
                        let text = value.to_string().replace('_', "");
                        let value = text.parse().map_err(|_| format!("`{key}` wants an integer, not `{text}`"))?;
                        params.push((key.to_string(), value));
                    }
                    _ => return Err(format!("expected `key = integer`, found `{}`", stream(part))),
                }
            }
        }
        Some(_) => return Err(usage.into()),
    }
    match tokens.next() {
        None => Ok((name.to_string(), params)),
        Some(_) => Err(usage.into()),
    }
}

/// A bodiless function declaration, in pieces.
struct Declaration {
    /// Attributes, visibility and qualifiers: everything before `fn`.
    head: String,
    name: String,
    params: Group,
    /// Each parameter's type.
    types: Vec<String>,
    result: Option<String>,
}

fn declaration(item: TokenStream) -> Result<Declaration, String> {
    let usage = "expected `unsafe extern \"sysv64\" fn name(...) -> ...;`, without a body";
    let tokens: Vec<TokenTree> = item.into_iter().collect();
    let at = tokens.iter().position(|t| is_ident(t, "fn")).ok_or(usage)?;
    let head = &tokens[..at];
    let abi = head.windows(2).any(|w| is_ident(&w[0], "extern") && w[1].to_string() == "\"sysv64\"");
    if !head.iter().any(|t| is_ident(t, "unsafe")) || !abi {
        return Err(format!("{usage}: a kernel trusts its pointers and follows SysV"));
    }
    let (Some(TokenTree::Ident(name)), Some(TokenTree::Group(params))) = (tokens.get(at + 1), tokens.get(at + 2))
    else {
        return Err(usage.into());
    };
    if params.delimiter() != Delimiter::Parenthesis {
        return Err(usage.into());
    }
    let mut types = vec![];
    let inner: Vec<TokenTree> = params.stream().into_iter().collect();
    for part in inner.split(|t| matches!(t, TokenTree::Punct(p) if p.as_char() == ',')) {
        if part.is_empty() {
            continue;
        }
        let colon = part.iter().position(|t| matches!(t, TokenTree::Punct(p) if p.as_char() == ':')).ok_or(usage)?;
        types.push(stream(&part[colon + 1..]));
    }
    let rest = &tokens[at + 3..];
    let (result, end) = match rest {
        [TokenTree::Punct(minus), TokenTree::Punct(gt), ty @ .., end]
            if minus.as_char() == '-' && gt.as_char() == '>' =>
        {
            (Some(stream(ty)), end)
        }
        [end] => (None, end),
        _ => return Err(usage.into()),
    };
    if !matches!(end, TokenTree::Punct(p) if p.as_char() == ';') {
        return Err(usage.into());
    }
    Ok(Declaration { head: stream(head), name: name.to_string(), params: params.clone(), types, result })
}

/// The register file SysV passes a value of type `ty` in, if it is one kgen knows.
fn class(ty: &str) -> Option<Class> {
    let ints = ["usize", "isize", "u64", "i64", "u32", "i32", "u16", "i16", "u8", "i8"];
    if ty.starts_with('*') || ints.contains(&ty) {
        Some(Class::Gpr)
    } else if ty == "f32" {
        Some(Class::Vec)
    } else {
        None
    }
}

fn check(f: &Declaration, l: &Listing) -> Result<(), String> {
    let want = &l.signature;
    if f.types.len() != want.params.len() {
        return Err(format!(
            "`{}` takes {} parameters; `{}` declares {}",
            l.name,
            want.params.len(),
            f.name,
            f.types.len()
        ));
    }
    for (i, (ty, class_wanted)) in f.types.iter().zip(&want.params).enumerate() {
        if class(ty) != Some(*class_wanted) {
            return Err(format!(
                "parameter {i} of `{}` arrives in a {class_wanted:?} register, which `{ty}` does not",
                l.name
            ));
        }
    }
    let result = f.result.as_deref().map(|ty| class(ty).ok_or(format!("kgen returns no `{ty}`"))).transpose()?;
    if result != want.result {
        return Err(format!("`{}` returns {:?}; `{}` declares {:?}", l.name, want.result, f.name, f.result));
    }
    Ok(())
}

fn emit(f: &Declaration, l: &Listing) -> String {
    let module = format!("__kgen_{}", f.name);
    let mut out = String::new();
    if !l.consts.is_empty() {
        let mut aligns: Vec<u32> = l.consts.iter().map(|c| c.align).collect();
        aligns.sort();
        aligns.dedup();
        out += &format!("#[doc(hidden)]\n#[allow(non_snake_case)]\nmod {module} {{\n");
        for a in &aligns {
            out += &format!("    #[repr(C, align({a}))]\n    pub struct A{a}<T>(pub T);\n");
        }
        for (i, c) in l.consts.iter().enumerate() {
            let bytes: Vec<String> = c.bytes.iter().map(u8::to_string).collect();
            out += &format!(
                "    pub static C{i}: A{a}<[u8; {n}]> = A{a}([{bytes}]);\n",
                a = c.align,
                n = c.bytes.len(),
                bytes = bytes.join(", ")
            );
        }
        out += "}\n";
    }
    out += &format!("#[unsafe(naked)]\n{} fn {}{}", f.head, f.name, f.params);
    if let Some(result) = &f.result {
        out += &format!(" -> {result}");
    }
    out += " {\n    ::core::arch::naked_asm!(\n";
    for line in l.code() {
        out += &format!("        {line:?},\n");
    }
    for i in 0..l.consts.len() {
        out += &format!("        c{i} = sym {module}::C{i},\n");
    }
    out += "    )\n}\n";
    out
}

fn is_ident(t: &TokenTree, name: &str) -> bool {
    matches!(t, TokenTree::Ident(i) if i.to_string() == name)
}

fn stream(tokens: &[TokenTree]) -> String {
    tokens.iter().cloned().collect::<TokenStream>().to_string()
}
