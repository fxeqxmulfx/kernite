//! The room of a module's GPU statics, counted over those `cfg` keeps. A macro runs before Rust
//! knows which `cfg` holds, but a derive is given a struct as `cfg` leaves it: `#[kernels]` writes a
//! struct with a field for each static, under the static's `cfg`, and `__Room` counts the room of
//! the fields Rust keeps, by the rule kernite has for it. The struct is never built.

use proc_macro::{Delimiter, Group, Literal, Span, TokenStream, TokenTree};

use crate::cond::Cond;
use crate::rust::{depends_on, error, error_under, flatten, is_ident, parse, site};

/// The crate whose derive counts the room, named by its path in the Rust the macro writes.
const MACROS: &str = "kernite-macros";

/// The field of static `i`, whose type is `ty`, which is at `at`: under the static's `cfg`, with the
/// type for `__Room` to count, written where the static's type is, for an error to point at.
pub(crate) fn field(cfg: &TokenStream, i: usize, ty: &str, at: Span) -> TokenStream {
    let mut ty = Literal::string(ty);
    ty.set_span(at);
    let mut attr = parse("kernite_room");
    attr.extend([TokenTree::Group(Group::new(Delimiter::Parenthesis, TokenTree::Literal(ty).into()))]);
    let mut out = cfg.clone();
    out.extend(parse("#"));
    out.extend([TokenTree::Group(Group::new(Delimiter::Bracket, attr))]);
    out.extend(parse(&format!("_{i}: (),")));
    out
}

/// The struct `__Room` is derived on, with `fields`, in a block of its own, where any of the statics
/// is `kept`. In a crate without a dependency named `kernite-macros` it is an error, at the module,
/// where `kept` holds.
pub(crate) fn carrier(fields: TokenStream, kept: &Cond) -> TokenStream {
    let Some(guard) = kept.guard() else { return TokenStream::new() };
    if !depends_on(MACROS) {
        let message =
            format!("a static of GPU kernels needs a dependency named `{MACROS}`, which this crate does not have");
        return error_under(&guard, site(), &message);
    }
    let mut out = parse("const _: () =");
    let mut block = parse("#[derive(::kernite_macros::__Room)] struct _Room");
    block.extend([TokenTree::Group(Group::new(Delimiter::Brace, fields))]);
    out.extend([TokenTree::Group(Group::new(Delimiter::Brace, block))]);
    out.extend(parse(";"));
    out
}

/// What `__Room` writes for the struct [`carrier`] made, as `cfg` left it: the error that is the
/// room's, at the type of each static that has no room beside those before it that have.
pub(crate) fn derive(input: TokenStream) -> TokenStream {
    let mut statics = vec![];
    types(input, &mut statics);
    let rooms = kernite::ptx_room(&statics.iter().map(|(ty, _)| ty.as_str()).collect::<Vec<_>>());
    statics.iter().zip(rooms).filter_map(|((_, at), why)| why.map(|message| error(*at, &message))).collect()
}

/// Each type `#[kernite_room("..")]` gives in `stream`, in order, with where it is written.
fn types(stream: TokenStream, found: &mut Vec<(String, Span)>) {
    let tokens = flatten(stream);
    for (i, t) in tokens.iter().enumerate() {
        match t {
            TokenTree::Group(g) if g.delimiter() == Delimiter::Brace => types(g.stream(), found),
            TokenTree::Punct(p) if p.as_char() == '#' => {
                let Some(TokenTree::Group(attr)) = tokens.get(i + 1) else { continue };
                if let [name, TokenTree::Group(args)] = &flatten(attr.stream())[..]
                    && is_ident(name, "kernite_room")
                    && let [TokenTree::Literal(ty)] = &flatten(args.stream())[..]
                {
                    found.push((ty.to_string().trim_matches('"').to_string(), ty.span()));
                }
            }
            _ => {}
        }
    }
}
