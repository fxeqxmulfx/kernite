//! Tokens as `proc_macro` hands them over, with numbered spans in place of its own, and a lexer
//! that reads the same tokens out of a string.

use crate::{Error, nested};

/// Where a token came from, as whoever made the tokens numbers it: an index into the spans of a
/// procedural macro's input, or a byte offset into a string.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Span(pub u32);

/// The brackets around a group.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Delim {
    /// `( )`.
    Paren,
    /// `[ ]`.
    Bracket,
    /// `{ }`.
    Brace,
}

/// A token, or a group of them in brackets. Each says whether the source has a space before it, for
/// [`text`] to write the part of a type that is written as the source has it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Tree {
    /// A name or a keyword.
    Ident {
        /// Its text.
        name: String,
        /// Whether the source has a space before it: a space, a line break or a comment.
        space: bool,
        /// Where it is.
        span: Span,
    },
    /// One punctuation character.
    Punct {
        /// The character.
        ch: char,
        /// Whether the source has a space before it. With none, it touches the token before it, as the
        /// second `<` of `<<` touches the first.
        space: bool,
        /// Where it is.
        span: Span,
    },
    /// A number, string or character, as written.
    Lit {
        /// Its text, quotes and suffix included.
        text: String,
        /// Whether the source has a space before it.
        space: bool,
        /// Where it is.
        span: Span,
    },
    /// Tokens in brackets.
    Group {
        /// The brackets.
        delim: Delim,
        /// What is inside.
        trees: Vec<Tree>,
        /// Whether the source has a space before the opening bracket.
        space: bool,
        /// Whether the source has a space before the closing bracket.
        space_close: bool,
        /// Where it opens.
        span: Span,
    },
}

impl Tree {
    /// Where it is.
    pub fn span(&self) -> Span {
        match self {
            Tree::Ident { span, .. } | Tree::Punct { span, .. } | Tree::Lit { span, .. } | Tree::Group { span, .. } => {
                *span
            }
        }
    }

    /// Whether the source has a space before it: a space, a line break or a comment.
    pub fn space(&self) -> bool {
        match self {
            Tree::Ident { space, .. }
            | Tree::Punct { space, .. }
            | Tree::Lit { space, .. }
            | Tree::Group { space, .. } => *space,
        }
    }
}

/// The name Rust reads in `ident`, an identifier as written: `u64` in `r#u64`. A keyword is asked of `ident` as
/// written, for `r#let` is a name and not `let`; any other name, of what this gives.
pub fn unraw(ident: &str) -> &str {
    ident.strip_prefix("r#").unwrap_or(ident)
}

/// The keywords Rust lets be written raw, the strict and the reserved ones of every edition: each is a name only if it
/// is written raw. The weak ones, `union` and `macro_rules` say, are names as they are.
const KEYWORDS: &[&str] = &[
    "as", "async", "await", "break", "const", "continue", "dyn", "else", "enum", "extern", "false", "fn", "for", "if",
    "impl", "in", "let", "loop", "match", "mod", "move", "mut", "pub", "ref", "return", "static", "struct", "trait",
    "true", "type", "unsafe", "use", "where", "while", // strict
    "abstract", "become", "box", "do", "final", "gen", "macro", "override", "priv", "try", "typeof", "unsized",
    "virtual", "yield", // reserved
];

/// `trees` as the language reads a type: [`text`], but with each name written raw at any depth as the name Rust
/// reads, `[r#f16; 4]` as `[f16; 4]`. A keyword written raw is a name too, and stays as written, `r#dyn::Foo`: the
/// readers of the text ask it of the keyword, [`pointer()`] of `dyn` and [`text`] of `const`, `mut` and `for`, and a
/// name that lost its `r#` would be taken for the keyword. So it is for the 48 keywords Rust lets be written raw, the
/// strict and the reserved ones. A weak keyword, `union` say, is a name as it is, and loses its `r#` as any name does.
pub fn read_type(trees: &[Tree]) -> String {
    fn plain(trees: &mut [Tree]) {
        for tree in trees {
            match tree {
                Tree::Ident { name, .. } if !KEYWORDS.contains(&unraw(name)) => *name = unraw(name).to_string(),
                Tree::Group { trees, .. } => plain(trees),
                _ => {}
            }
        }
    }
    let mut trees = trees.to_vec();
    plain(&mut trees);
    text(&trees)
}

/// `trees` as Rust writes a type in a message.
///
/// The type's own tokens are spaced as the printer of rustc spaces them, whatever the source has:
/// `*const [usize; 3]`, `(u32, u32)`, `&mut [f32]`, `fn(x: u8) -> u8`, `dyn A + B`, `Item<X> = u8` from
/// `Item<X>=u8` too, and a space before the braces of a macro, `byte! {..}`.
///
/// What is not the type's own is written as the source has it, as the messages of rustc write it: a block,
/// `Marker<{N*2}>` or `Marker<{ N * 2 }>`; the length of an array, `[u8; N*2]`; and the arguments of a macro,
/// `m!(a+b)` or `m!(a + b)`. There is one space where the source has a space, a line break or a comment, and
/// none where it has none. Nothing is worked out here: where the evaluator takes a type apart, it works out the
/// length of an array in it, `[u8; 6]`.
pub fn text(trees: &[Tree]) -> String {
    written(trees, false)
}

/// `trees` as [`text`] writes them. `source` says that they are all written as the source has them.
fn written(trees: &[Tree], source: bool) -> String {
    let mut out = String::new();
    for (i, t) in trees.iter().enumerate() {
        if i > 0 && (if source { t.space() } else { spaced(trees, i) }) {
            out.push(' ');
        }
        match t {
            Tree::Ident { name: s, .. } | Tree::Lit { text: s, .. } => out += s,
            // The length of an array is all that follows its `;`.
            Tree::Punct { ch: ';', .. } if !source => {
                out.push(';');
                if i + 1 < trees.len() {
                    out += &format!(" {}", written(&trees[i + 1..], true));
                }
                break;
            }
            Tree::Punct { ch, .. } => out.push(*ch),
            Tree::Group { delim, trees: inside, space_close, .. } => {
                let (open, close) = match delim {
                    Delim::Paren => ('(', ')'),
                    Delim::Bracket => ('[', ']'),
                    Delim::Brace => ('{', '}'),
                };
                // The brackets of a block and of a macro's arguments are as the source has them too.
                let source = source || *delim == Delim::Brace || macro_args(trees, i);
                let pad = |space: bool| if source && space { " " } else { "" };
                let first = inside.first().is_some_and(Tree::space);
                out += &format!("{open}{}{}{}{close}", pad(first), written(inside, source), pad(*space_close));
            }
        }
    }
    out
}

/// Whether `trees[i]` is the arguments of a macro, `(a + b)` in `m!(a + b)`: a group after a `!`, which in
/// a type is the macro's and no other's.
fn macro_args(trees: &[Tree], i: usize) -> bool {
    i > 0 && matches!(trees[i - 1], Tree::Punct { ch: '!', .. })
}

/// Whether `trees[i]` is the `>` of an arrow, `->`, as in `fn(u8) -> u8`: one operator, whose `>` closes
/// no `<`.
fn arrow_head(trees: &[Tree], i: usize) -> bool {
    matches!(
        i.checked_sub(1).and_then(|j| trees.get(j..=i)),
        Some([Tree::Punct { ch: '-', .. }, Tree::Punct { ch: '>', .. }])
    )
}

/// Whether `trees[i]` is the `>` that ends a binder, as in `for<'a> fn(&'a u8)`: the nearest angle
/// bracket before it is a `<`, and a `for` stands before that.
fn binder_end(trees: &[Tree], i: usize) -> bool {
    matches!(trees[i], Tree::Punct { ch: '>', .. })
        && trees[..i].iter().rposition(|t| matches!(t, Tree::Punct { ch: '<' | '>', .. })).is_some_and(|open| {
            matches!(trees[open], Tree::Punct { ch: '<', .. })
                && matches!(trees[..open].last(), Some(Tree::Ident { name, .. }) if name == "for")
        })
}

/// Whether `trees[j]` is a `:` that is not half of a `::`, as in `fn(x: u8)` and `Item: Copy`.
fn colon(trees: &[Tree], j: usize) -> bool {
    let path = |a: usize| {
        matches!(trees.get(a..=a + 1), Some([Tree::Punct { ch: ':', .. }, Tree::Punct { ch: ':', space: false, .. }]))
    };
    matches!(trees[j], Tree::Punct { ch: ':', .. }) && !path(j) && !(j > 0 && path(j - 1))
}

/// Whether `trees[j]` is the name of a lifetime, as in `&'a [u8]`: a word after a `'`.
fn lifetime(trees: &[Tree], j: usize) -> bool {
    matches!(trees[j], Tree::Ident { .. }) && j > 0 && matches!(trees[j - 1], Tree::Punct { ch: '\'', .. })
}

/// Whether `name` is a keyword that a type or a bound follows, as in `*const u8`, `&mut [u8]`, `dyn 'a + A`
/// and `impl (A)`.
fn keyword(name: &str) -> bool {
    matches!(name, "as" | "const" | "dyn" | "impl" | "mut")
}

/// Whether the printer of rustc writes a space between `trees[i - 1]` and `trees[i]`, of a type's own tokens:
/// after a keyword that a type or a bound follows, `dyn`, `impl`, `const`, `mut`, `as`, and before `as`,
/// `<A<'a> as B>::C`; after `,`; after a `:` that is not half of a `::`; after the `>` of a binder,
/// `for<'a> fn(&'a u8)`; between words; after the name of a lifetime, `&'a [u8]` and `&'a !`; on each side of
/// a `+` or a `=` that no operator holds, as in `dyn A + B` and `Item = &u8`; and on each side of an arrow,
/// `fn(u8) -> u8`. The operators that hold a `+` or a `=`, `+=`, `==`, `=>`, `<=` and `..=`, are written
/// close, and so are the two characters of an arrow. A `>` before a `=` is no operator's: it closes the
/// arguments of a name, so `Item<X>=u8` and `Item<X> = u8` are both written `Item<X> = u8`. The braces of a
/// macro have a space before them, `byte! {..}`. Other punctuation and brackets are against the word before
/// them.
fn spaced(trees: &[Tree], i: usize) -> bool {
    // Only a `=` or a `>` after a `=`, and a `=` after a `+`, make an operator: `=&` and `+?` are none, as
    // the `&` after the `=` of `Item=&u8` shows. No operator ends in `+`, and `+=` begins with it; `==` and
    // `=>` begin with `=`, and `+=`, `<=` and `..=` end in it. A `>` before a `=` is no operator's: it closes
    // the arguments of a name, as in `Item<X>=u8`.
    let apart = |j: usize| {
        let next = |chars: &str| matches!(trees.get(j + 1), Some(Tree::Punct { ch, .. }) if chars.contains(*ch));
        match trees[j] {
            Tree::Punct { ch: '+', .. } => !next("="),
            Tree::Punct { ch: '=', .. } => {
                !next("=>") && !matches!(trees[..j].last(), Some(Tree::Punct { ch, .. }) if *ch != '>')
            }
            _ => false,
        }
    };
    arrow_head(trees, i - 1)
        || arrow_head(trees, i + 1)
        || binder_end(trees, i - 1)
        || colon(trees, i - 1)
        || apart(i - 1)
        || apart(i)
        || match (&trees[i - 1], &trees[i]) {
            (Tree::Ident { name, .. }, _) if keyword(name) => true,
            (_, Tree::Ident { name, .. }) if name == "as" => true,
            (Tree::Ident { .. }, after) if lifetime(trees, i - 1) => {
                !matches!(after, Tree::Punct { ch: ',' | '>' | ':', .. })
            }
            (Tree::Punct { ch, .. }, after) => {
                *ch == ',' || *ch == '!' && matches!(after, Tree::Group { delim: Delim::Brace, .. })
            }
            (_, Tree::Punct { .. } | Tree::Group { .. }) => false,
            _ => true,
        }
}

/// Whether `ty`, a type as [`read_type`] gives it, is a pointer, `*const T` or `*mut T`; or why it is
/// not one a machine passes in a register: a pointer to a type with no size is two words, and one
/// to a type kernite cannot read, as `&'static str` is, is refused in the lexer's words. A type that
/// has no size but is named, as `CStr` is, is not told from one that has.
pub fn pointer(ty: &str) -> Result<bool, String> {
    let Some(to) = ty.strip_prefix("*const ").or_else(|| ty.strip_prefix("*mut ")) else {
        return Ok(false);
    };
    // The macro passes a lifetime on as punctuation, where the lexer refuses one in a string, so `to`
    // may not lex: what kernite cannot read is not taken for a word.
    let trees = lex(to).map_err(|e| e.message)?;
    if no_size(&trees) {
        return Err(format!(
            "a pointer to `{to}` has two words, an address and a length or a table, and a kernel's pointer has one"
        ));
    }
    Ok(true)
}

/// Whether the type `ty` has no size: a slice, a `str` or a `dyn`, or a tuple that ends in one.
fn no_size(ty: &[Tree]) -> bool {
    match ty {
        // A slice is brackets with no `;` of their own, `[[u8; 4]]`; an array has one, `[u8; 4]`.
        [Tree::Group { delim: Delim::Bracket, trees, .. }] => {
            !trees.iter().any(|t| matches!(t, Tree::Punct { ch: ';', .. }))
        }
        // Parentheses around one type, `(dyn A + Send)`, are as the type is; a tuple, as its last field.
        [Tree::Group { delim: Delim::Paren, trees, .. }] => no_size(last_field(trees)),
        [Tree::Ident { name, .. }] => name == "str",
        [Tree::Ident { name, .. }, ..] => name == "dyn",
        _ => false,
    }
}

/// The last field of a tuple whose fields are `fields`: what follows the last `,` that neither ends
/// the tuple, `(u8,)`, nor is among the arguments of a type, `Pair<u8, 2>`.
fn last_field(mut fields: &[Tree]) -> &[Tree] {
    if let [rest @ .., Tree::Punct { ch: ',', .. }] = fields {
        fields = rest;
    }
    let (mut open, mut from) = (0usize, 0);
    for (i, t) in fields.iter().enumerate() {
        match t {
            Tree::Punct { ch: '<', .. } => open += 1,
            // The `>` of an arrow closes no `<`; nor does one with none open, which Rust refuses.
            Tree::Punct { ch: '>', .. } if !arrow_head(fields, i) => open = open.saturating_sub(1),
            Tree::Punct { ch: ',', .. } if open == 0 => from = i + 1,
            _ => {}
        }
    }
    &fields[from..]
}

/// The characters that are punctuation.
const PUNCT: &str = "+-*/%^!&|=<>@.,;:#$?~";

/// The tokens of `src`, each spanned by its byte offset, and each with whether a space, a line break or a
/// comment comes before it. Comments are skipped, doc comments too.
pub fn lex(src: &str) -> Result<Vec<Tree>, Error> {
    Lexer { src, pos: 0 }.trees(None, 0).map(|(trees, _)| trees)
}

struct Lexer<'a> {
    src: &'a str,
    pos: usize,
}

impl Lexer<'_> {
    fn peek(&self) -> Option<char> {
        self.src[self.pos..].chars().next()
    }

    fn peek2(&self) -> Option<char> {
        self.src[self.pos..].chars().nth(1)
    }

    fn span(&self) -> Span {
        Span(self.pos as u32)
    }

    fn error(&self, message: impl Into<String>) -> Error {
        Error { span: self.span(), message: message.into() }
    }

    /// Tokens up to `close`, or to the end of the input if there is none to wait for, `depth`
    /// brackets down; and whether a space comes before the end.
    fn trees(&mut self, close: Option<(char, Span)>, depth: usize) -> Result<(Vec<Tree>, bool), Error> {
        let mut out = vec![];
        loop {
            let before = self.pos;
            self.skip_space()?;
            let space = self.pos > before;
            let span = self.span();
            let Some(c) = self.peek() else {
                return match close {
                    None => Ok((out, space)),
                    Some((ch, open)) => Err(Error { span: open, message: format!("this is never closed by `{ch}`") }),
                };
            };
            match c {
                '(' | '[' | '{' => {
                    nested(depth, span)?;
                    self.pos += 1;
                    let (delim, closer) = match c {
                        '(' => (Delim::Paren, ')'),
                        '[' => (Delim::Bracket, ']'),
                        _ => (Delim::Brace, '}'),
                    };
                    let (trees, space_close) = self.trees(Some((closer, span)), depth + 1)?;
                    out.push(Tree::Group { delim, trees, space, space_close, span });
                }
                ')' | ']' | '}' => {
                    if close.map(|(ch, _)| ch) != Some(c) {
                        return Err(self.error(format!("unexpected `{c}`")));
                    }
                    self.pos += 1;
                    return Ok((out, space));
                }
                '"' => out.push(Tree::Lit { text: self.string()?, space, span }),
                '\'' => out.push(Tree::Lit { text: self.char()?, space, span }),
                c if c.is_ascii_digit() => out.push(Tree::Lit { text: self.number(), space, span }),
                c if c == '_' || c.is_alphabetic() => {
                    let start = self.pos;
                    if self.src[start..].starts_with("r#") {
                        self.pos += 2;
                    }
                    while let Some(c) = self.peek().filter(|&c| c == '_' || c.is_alphanumeric()) {
                        self.pos += c.len_utf8();
                    }
                    out.push(Tree::Ident { name: self.src[start..self.pos].to_string(), space, span });
                }
                c if PUNCT.contains(c) => {
                    self.pos += 1;
                    out.push(Tree::Punct { ch: c, space, span });
                }
                c => return Err(self.error(format!("unexpected `{c}`"))),
            }
        }
    }

    fn skip_space(&mut self) -> Result<(), Error> {
        loop {
            let rest = &self.src[self.pos..];
            if let Some(c) = rest.chars().next()
                && c.is_whitespace()
            {
                self.pos += c.len_utf8();
            } else if rest.starts_with("//") {
                self.pos += rest.find('\n').unwrap_or(rest.len());
            } else if rest.starts_with("/*") {
                let start = self.span();
                let mut depth = 0;
                loop {
                    let rest = &self.src[self.pos..];
                    if rest.starts_with("/*") {
                        depth += 1;
                        self.pos += 2;
                    } else if rest.starts_with("*/") {
                        depth -= 1;
                        self.pos += 2;
                        if depth == 0 {
                            break;
                        }
                    } else if let Some(c) = rest.chars().next() {
                        self.pos += c.len_utf8();
                    } else {
                        return Err(Error { span: start, message: "this comment is never closed".into() });
                    }
                }
            } else {
                return Ok(());
            }
        }
    }

    fn string(&mut self) -> Result<String, Error> {
        let (start, span) = (self.pos, self.span());
        self.pos += 1;
        loop {
            match self.peek() {
                None => return Err(Error { span, message: "this string is never closed".into() }),
                Some('\\') => {
                    self.pos += 1;
                    if let Some(c) = self.peek() {
                        self.pos += c.len_utf8();
                    }
                }
                Some('"') => {
                    self.pos += 1;
                    return Ok(self.src[start..self.pos].to_string());
                }
                Some(c) => self.pos += c.len_utf8(),
            }
        }
    }

    fn char(&mut self) -> Result<String, Error> {
        let (start, span) = (self.pos, self.span());
        self.pos += 1;
        if self.peek() == Some('\\') {
            self.pos += 1;
        }
        match self.peek() {
            Some(c) => self.pos += c.len_utf8(),
            None => return Err(self.error("a character literal is never closed")),
        }
        if self.peek() != Some('\'') {
            return Err(Error { span, message: "lifetimes and labels have no place in a kernel".into() });
        }
        self.pos += 1;
        Ok(self.src[start..self.pos].to_string())
    }

    /// A number as Rust writes it: `16`, `0x3f`, `1_000`, `0.5`, `1e-5`, `8u8`, `2.0f32`. A
    /// `.` after the digits belongs to the number only if a digit follows, so that `0..n` is a
    /// range.
    fn number(&mut self) -> String {
        let start = self.pos;
        let word = |c: char| c == '_' || c.is_ascii_alphanumeric();
        let radix = ["0x", "0o", "0b"].iter().any(|p| self.src[start..].starts_with(p));
        if radix {
            self.pos += 2;
            while self.peek().is_some_and(word) {
                self.pos += 1;
            }
            return self.src[start..self.pos].to_string();
        }
        while self.peek().is_some_and(|c| c == '_' || c.is_ascii_digit()) {
            self.pos += 1;
        }
        if self.peek() == Some('.') && self.peek2().is_some_and(|c| c.is_ascii_digit()) {
            self.pos += 1;
            while self.peek().is_some_and(|c| c == '_' || c.is_ascii_digit()) {
                self.pos += 1;
            }
        }
        if matches!(self.peek(), Some('e' | 'E')) {
            let rest = &self.src[self.pos + 1..];
            let digits = rest.strip_prefix(['+', '-']).unwrap_or(rest);
            if digits.starts_with(|c: char| c.is_ascii_digit()) {
                self.pos += 1 + (rest.len() - digits.len());
                while self.peek().is_some_and(|c| c == '_' || c.is_ascii_digit()) {
                    self.pos += 1;
                }
            }
        }
        while self.peek().is_some_and(word) {
            self.pos += 1;
        }
        self.src[start..self.pos].to_string()
    }
}
