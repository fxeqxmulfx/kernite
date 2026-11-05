//! When what the macro writes is there. Only Rust knows which `cfg` holds, so the macro writes each
//! piece of a module under the `cfg` of the item it stands for, and Rust puts the module together.

use proc_macro::{Delimiter, Group, TokenStream, TokenTree};

use kernite::{Kept, When};

use crate::rust::{attrs_len, flatten, inner_attribute, is_error, is_punct, named, parse};

/// When something the macro writes is there: always, never, or when a predicate of `cfg` holds,
/// `feature = "cuda"`. The macro does not evaluate a predicate; it writes it for Rust to.
#[derive(Clone)]
pub(crate) enum Cond {
    /// Whatever `cfg` holds.
    Always,
    /// Under no `cfg`: what is written for it is left out.
    Never,
    /// When the predicate holds.
    If(TokenStream),
}

impl Cond {
    /// The condition under which an item is there, from `cfg`, what [`carried`](crate::rust::carried)
    /// keeps of its attributes when it offers nothing: its `cfg`s as one predicate, `all(..)` of them,
    /// where a `cfg_attr(p, cfg(q))` is `any(not(p), q)`, and one in another likewise. `Always` for an
    /// item with none.
    pub(crate) fn of(cfg: &TokenStream) -> Cond {
        let attrs: Vec<TokenTree> = cfg.clone().into_iter().collect();
        Cond::all(attrs.chunks(2).filter_map(|attr| {
            let [_, TokenTree::Group(meta)] = attr else { unreachable!("an attribute is `#` and brackets") };
            predicate(&flatten(meta.stream())).map(Cond::If)
        }))
    }

    /// When `module`, the tokens of a module the macros write, has an error as an item of it: the `cfg` before each
    /// [`error`](crate::rust::error) at its top, as one condition. Only there is Rust silent about a name it cannot
    /// find in the module, so only there can it be shown the types of a signature that nothing else writes. An item
    /// begins at the start, or after the `;` or the braces that end the one before.
    pub(crate) fn errors_in(module: &TokenStream) -> Cond {
        let tokens: Vec<TokenTree> = module.clone().into_iter().collect();
        let ends =
            |t: &TokenTree| is_punct(t, ';') || matches!(t, TokenTree::Group(g) if g.delimiter() == Delimiter::Brace);
        let (mut found, mut attrs, mut start, mut i) = (vec![], TokenStream::new(), true, 0);
        while i < tokens.len() {
            let (rest, n) = (&tokens[i..], attrs_len(&tokens[i..]));
            if start && inner_attribute(rest) {
                i += 3;
            } else if start && n > 0 {
                attrs.extend(rest[..n].iter().cloned());
                i += n;
            } else {
                if start && is_error(rest) {
                    found.push(Cond::of(&attrs));
                }
                (attrs, start) = (TokenStream::new(), ends(&tokens[i]));
                i += 1;
            }
        }
        Cond::any(found)
    }

    /// When any of `parts` holds, each once: `Always` if one always does, `Never` for none.
    pub(crate) fn any(parts: impl IntoIterator<Item = Cond>) -> Cond {
        let mut kept = vec![];
        for part in parts {
            match part {
                Cond::Always => return Cond::Always,
                Cond::Never => {}
                Cond::If(p) => kept.push(p),
            }
        }
        match kept.is_empty() {
            true => Cond::Never,
            false => Cond::If(join("any", distinct(kept))),
        }
    }

    /// When all of `parts` hold, each once: `Never` if one never does, `Always` for none.
    pub(crate) fn all(parts: impl IntoIterator<Item = Cond>) -> Cond {
        let mut kept = vec![];
        for part in parts {
            match part {
                Cond::Always => {}
                Cond::Never => return Cond::Never,
                Cond::If(p) => kept.push(p),
            }
        }
        match kept.is_empty() {
            true => Cond::Always,
            false => Cond::If(join("all", distinct(kept))),
        }
    }

    /// When `when` holds, `kept` saying when each code and static it names is kept.
    pub(crate) fn when(when: &When, kept: &dyn Fn(Kept) -> Cond) -> Cond {
        match when {
            When::Kept(it) => kept(*it),
            When::Any(parts) => Cond::any(parts.iter().map(|part| Cond::when(part, kept))),
            When::All(parts) => Cond::all(parts.iter().map(|part| Cond::when(part, kept))),
            When::Not(part) => Cond::when(part, kept).not(),
        }
    }

    /// When this does not hold.
    pub(crate) fn not(self) -> Cond {
        match self {
            Cond::Always => Cond::Never,
            Cond::Never => Cond::Always,
            Cond::If(p) => Cond::If(call("not", vec![p])),
        }
    }

    /// How many of `conds` hold, as an expression of Rust: a number, if none is asked of `cfg`, else
    /// a `const` block that sums `cfg!(..)` for each, which Rust works out as it compiles.
    pub(crate) fn count<'a>(conds: impl Iterator<Item = &'a Cond>) -> String {
        let (mut always, mut asked) = (0, Vec::<(String, usize)>::new());
        for cond in conds {
            match cond {
                Cond::Always => always += 1,
                Cond::Never => {}
                Cond::If(p) => {
                    let text = p.to_string();
                    match asked.iter_mut().find(|(q, _)| *q == text) {
                        Some((_, n)) => *n += 1,
                        None => asked.push((text, 1)),
                    }
                }
            }
        }
        if asked.is_empty() {
            return always.to_string();
        }
        let terms: Vec<String> =
            asked.iter().map(|(p, n)| format!("::core::cfg!({p}) as ::core::primitive::usize * {n}")).collect();
        format!("const {{ {always} + {} }}", terms.join(" + "))
    }

    /// The attribute that puts what is written under it where this holds: `#[cfg(..)]`, or nothing
    /// where it always does. `None` where it never does, for then nothing is written.
    pub(crate) fn guard(&self) -> Option<TokenStream> {
        match self {
            Cond::Always => Some(TokenStream::new()),
            Cond::Never => None,
            Cond::If(p) => {
                let mut out = parse("#");
                out.extend([TokenTree::Group(Group::new(Delimiter::Bracket, call("cfg", vec![p.clone()])))]);
                Some(out)
            }
        }
    }
}

/// `parts` with each once: of those that are written the same, the first. Each is written out once,
/// for the comparison.
fn distinct(parts: Vec<TokenStream>) -> Vec<TokenStream> {
    let (mut seen, mut once) = (vec![], vec![]);
    for part in parts {
        let text = part.to_string();
        if !seen.contains(&text) {
            seen.push(text);
            once.push(part);
        }
    }
    once
}

/// The predicate that one attribute [`carried`](crate::rust::carried) keeps says, `meta` the tokens in
/// its brackets: `cfg(p)` says `p`, and `cfg_attr(p, ..)` that what it keeps of the list holds, or `p`
/// does not. `None` for an attribute that says nothing, as a `cfg` Rust cannot read: it says what is
/// wrong with it where it is written, and takes the item to be there.
fn predicate(meta: &[TokenTree]) -> Option<TokenStream> {
    let [name, TokenTree::Group(list)] = meta else { return None };
    let tokens = flatten(list.stream());
    if named(name, "cfg") {
        return Some(tokens.into_iter().collect());
    }
    let mut parts = tokens.split(|t| is_punct(t, ','));
    let when: TokenStream = parts.next().expect("a split has a first part").iter().cloned().collect();
    let then: Vec<TokenStream> = parts.filter_map(predicate).collect();
    (!then.is_empty()).then(|| call("any", vec![call("not", vec![when]), join("all", then)]))
}

/// `name(parts)` for a predicate's `all` or `any`, or the part alone where there is one.
fn join(name: &str, mut parts: Vec<TokenStream>) -> TokenStream {
    match parts.len() {
        1 => parts.remove(0),
        _ => call(name, parts),
    }
}

/// `name(a, b, ..)`, the `parts` apart by commas.
fn call(name: &str, parts: Vec<TokenStream>) -> TokenStream {
    let mut args = TokenStream::new();
    for (i, part) in parts.into_iter().enumerate() {
        if i > 0 {
            args.extend(parse(","));
        }
        args.extend(part);
    }
    let mut out = parse(name);
    out.extend([TokenTree::Group(Group::new(Delimiter::Parenthesis, args))]);
    out
}
