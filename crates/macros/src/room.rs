//! The room of a module's GPU statics, counted over those `cfg` keeps. A macro runs before Rust
//! knows which `cfg` holds, but a derive is given a struct as `cfg` leaves it: `#[kernels]` writes a
//! struct with a field for each static, under the static's `cfg`, and `__Room` counts the room of
//! the fields Rust keeps, by the rule kernite has for it. The struct is never built.
//!
//! The error of a static with no room is an item of the module, as every error the macros find is,
//! and so Rust says nothing of a name it cannot find: the types a helper's signature names, which
//! no kernel needs to compile, are shown to Rust beside it. The struct has a field for each helper
//! too, under the helper's `cfg`; and `__Room` writes the types of the fields it is given where it
//! writes the error.

use proc_macro::{Delimiter, Group, Literal, Span, TokenStream, TokenTree};

use crate::cond::Cond;
use crate::rust::{MACROS, depends_on, error, error_under, flatten, is_ident, needs_dependency, parse, site};

/// The field of item `i`, under `cfg`, which `attr` is on, with `args` for `__Room` to read.
fn field(cfg: &TokenStream, i: usize, attr: &str, args: TokenStream) -> TokenStream {
    let mut meta = parse(attr);
    meta.extend([TokenTree::Group(Group::new(Delimiter::Parenthesis, args))]);
    let mut out = cfg.clone();
    out.extend(parse("#"));
    out.extend([TokenTree::Group(Group::new(Delimiter::Bracket, meta))]);
    out.extend(parse(&format!("_{i}: (),")));
    out
}

/// The field of static `i`, whose type is `ty`, which is at `at`: under the static's `cfg`, with the
/// type for `__Room` to count, written where the static's type is, for an error to point at.
pub(crate) fn room(cfg: &TokenStream, i: usize, ty: &str, at: Span) -> TokenStream {
    let mut ty = Literal::string(ty);
    ty.set_span(at);
    field(cfg, i, "kernite_room", TokenTree::Literal(ty).into())
}

/// The field of helper `i`, under `cfg`, which is where it is kept: with `lines`, the types of its signature, for
/// `__Room` to write where it writes the error of a static with no room.
pub(crate) fn shown(cfg: &TokenStream, i: usize, lines: TokenStream) -> TokenStream {
    field(cfg, i, "kernite_shown", lines)
}

/// What says, in a crate without a dependency named `kernite-macros`, where `kept` holds, that a static
/// of GPU kernels needs one: an error at the module. The crate has no `__Room` to count the room, so
/// there is no struct, and no error of a static with no room.
pub(crate) fn lacking(kept: &Cond) -> TokenStream {
    match (kept.guard(), depends_on(MACROS)) {
        (Some(guard), false) => error_under(&guard, site(), &needs_dependency("a static of GPU kernels", MACROS)),
        _ => TokenStream::new(),
    }
}

/// The struct `__Room` is derived on, with the fields of the statics that are laid out, `rooms`, and of the helpers,
/// `shown`; none where there is no static, or in a crate without a dependency named `kernite-macros`, which
/// [`lacking`] says. It is an item of the module, for the errors `__Room` writes to be: Rust is silent about a
/// name it cannot find only where the module has an error as an item.
pub(crate) fn carrier(mut rooms: TokenStream, shown: TokenStream) -> TokenStream {
    if rooms.is_empty() || !depends_on(MACROS) {
        return TokenStream::new();
    }
    rooms.extend(shown);
    let mut out = parse("#[doc(hidden)] #[derive(::kernite_macros::__Room)] struct __KerniteRoom");
    out.extend([TokenTree::Group(Group::new(Delimiter::Brace, rooms))]);
    out
}

/// What `__Room` writes for the struct [`carrier`] made, as `cfg` left it: the error that is the
/// room's, at the type of each static that has no room beside those before it that have, and, if
/// there is one, the types the fields of helpers show.
pub(crate) fn derive(input: TokenStream) -> TokenStream {
    let (mut statics, mut lines) = (vec![], TokenStream::new());
    fields(input, &mut statics, &mut lines);
    let rooms = kernite::ptx_room(&statics.iter().map(|(ty, _)| ty.as_str()).collect::<Vec<_>>());
    let mut out: TokenStream =
        statics.iter().zip(rooms).filter_map(|((_, at), why)| why.map(|message| error(*at, &message))).collect();
    if !out.is_empty() && !lines.is_empty() {
        out.extend(parse("const _: () ="));
        out.extend([TokenTree::Group(Group::new(Delimiter::Brace, lines))]);
        out.extend(parse(";"));
    }
    out
}

/// Each type `#[kernite_room("..")]` gives in `stream`, in order, with where it is written, and
/// the tokens of each `#[kernite_shown(..)]`.
fn fields(stream: TokenStream, statics: &mut Vec<(String, Span)>, lines: &mut TokenStream) {
    let tokens = flatten(stream);
    for (i, t) in tokens.iter().enumerate() {
        match t {
            TokenTree::Group(g) if g.delimiter() == Delimiter::Brace => fields(g.stream(), statics, lines),
            TokenTree::Punct(p) if p.as_char() == '#' => {
                let Some(TokenTree::Group(attr)) = tokens.get(i + 1) else { continue };
                let [name, TokenTree::Group(args)] = &flatten(attr.stream())[..] else { continue };
                if is_ident(name, "kernite_room")
                    && let [TokenTree::Literal(ty)] = &flatten(args.stream())[..]
                {
                    statics.push((ty.to_string().trim_matches('"').to_string(), ty.span()));
                } else if is_ident(name, "kernite_shown") {
                    lines.extend(args.stream());
                }
            }
            _ => {}
        }
    }
}
