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

/// A token, or a group of them in brackets.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Tree {
    /// A name or a keyword.
    Ident {
        /// Its text.
        name: String,
        /// Where it is.
        span: Span,
    },
    /// One punctuation character.
    Punct {
        /// The character.
        ch: char,
        /// Whether the next token is punctuation right after it, as in the first `<` of `<<`.
        joint: bool,
        /// Where it is.
        span: Span,
    },
    /// A number, string or character, as written.
    Lit {
        /// Its text, quotes and suffix included.
        text: String,
        /// Where it is.
        span: Span,
    },
    /// Tokens in brackets.
    Group {
        /// The brackets.
        delim: Delim,
        /// What is inside.
        trees: Vec<Tree>,
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
}

/// `trees` as Rust writes a type: `*const [usize; 3]`, `(u32, u32)`, `&mut [f32]`, `fn(u8) -> u8`.
pub fn text(trees: &[Tree]) -> String {
    let mut out = String::new();
    for (i, t) in trees.iter().enumerate() {
        if i > 0 && spaced(trees, i) {
            out.push(' ');
        }
        match t {
            Tree::Ident { name: s, .. } | Tree::Lit { text: s, .. } => out += s,
            Tree::Punct { ch, .. } => out.push(*ch),
            Tree::Group { delim, trees, .. } => {
                let (open, close) = match delim {
                    Delim::Paren => ('(', ')'),
                    Delim::Bracket => ('[', ']'),
                    Delim::Brace => ('{', '}'),
                };
                out.push(open);
                out += &text(trees);
                out.push(close);
            }
        }
    }
    out
}

/// Whether `trees[i]` is the `>` of an arrow, `->`, as in `fn(u8) -> u8`: one operator, whose `>`
/// closes no `<`.
fn arrow_head(trees: &[Tree], i: usize) -> bool {
    matches!(
        i.checked_sub(1).and_then(|j| trees.get(j..=i)),
        Some([Tree::Punct { ch: '-', joint: true, .. }, Tree::Punct { ch: '>', .. }])
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

/// Whether Rust writes a space between `trees[i - 1]` and `trees[i]`: after `const` and `mut`,
/// which a type follows, after `,` and `;`, after the `>` of a binder, `for<'a> fn(&'a u8)`, between
/// words, on each side of a `+` or a `=` that no operator holds, as in `dyn A + B` and `Item = &u8`,
/// and on each side of an arrow, `fn(u8) -> u8`. The operators that hold a `+` or a `=`, `+=`, `==`,
/// `=>`, `<=` and `..=`, are written close, and so are the two characters of an arrow. Other
/// punctuation and brackets are against the word before them.
fn spaced(trees: &[Tree], i: usize) -> bool {
    // The `joint` mark says that punctuation follows, not that an operator is made, as the `&` after
    // the `=` of `Item=&u8` shows, so the next character is read too. No operator ends in `+`, and
    // `+=` begins with it; `==` and `=>` begin with `=`, and `+=`, `<=` and `..=` end in it.
    let apart = |j: usize| {
        let next = |chars: &str| matches!(trees.get(j + 1), Some(Tree::Punct { ch, .. }) if chars.contains(*ch));
        match trees[j] {
            Tree::Punct { ch: '+', joint, .. } => !(joint && next("=")),
            Tree::Punct { ch: '=', joint, .. } => {
                !(joint && next("=>")) && !matches!(trees[..j].last(), Some(Tree::Punct { joint: true, .. }))
            }
            _ => false,
        }
    };
    arrow_head(trees, i - 1)
        || arrow_head(trees, i + 1)
        || binder_end(trees, i - 1)
        || apart(i - 1)
        || apart(i)
        || match (&trees[i - 1], &trees[i]) {
            (Tree::Ident { name, .. }, _) if name == "const" || name == "mut" => true,
            (Tree::Punct { ch, .. }, _) => matches!(ch, ',' | ';'),
            (_, Tree::Punct { .. } | Tree::Group { .. }) => false,
            _ => true,
        }
}

/// Whether `ty`, a type as [`text`] writes it, is a pointer, `*const T` or `*mut T`; or why it is
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

/// The tokens of `src`, each spanned by its byte offset. Comments are skipped, doc comments too.
pub fn lex(src: &str) -> Result<Vec<Tree>, Error> {
    Lexer { src, pos: 0 }.trees(None, 0)
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
    /// brackets down.
    fn trees(&mut self, close: Option<(char, Span)>, depth: usize) -> Result<Vec<Tree>, Error> {
        let mut out = vec![];
        loop {
            self.skip_space()?;
            let span = self.span();
            let Some(c) = self.peek() else {
                return match close {
                    None => Ok(out),
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
                    let trees = self.trees(Some((closer, span)), depth + 1)?;
                    out.push(Tree::Group { delim, trees, span });
                }
                ')' | ']' | '}' => {
                    if close.map(|(ch, _)| ch) != Some(c) {
                        return Err(self.error(format!("unexpected `{c}`")));
                    }
                    self.pos += 1;
                    return Ok(out);
                }
                '"' => out.push(self.string()?),
                '\'' => out.push(self.char()?),
                c if c.is_ascii_digit() => out.push(self.number()),
                c if c == '_' || c.is_alphabetic() => {
                    let start = self.pos;
                    if self.src[start..].starts_with("r#") {
                        self.pos += 2;
                    }
                    while let Some(c) = self.peek().filter(|&c| c == '_' || c.is_alphanumeric()) {
                        self.pos += c.len_utf8();
                    }
                    out.push(Tree::Ident { name: self.src[start..self.pos].to_string(), span });
                }
                c if PUNCT.contains(c) => {
                    self.pos += 1;
                    let joint = self.peek().is_some_and(|n| PUNCT.contains(n));
                    out.push(Tree::Punct { ch: c, joint, span });
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

    fn string(&mut self) -> Result<Tree, Error> {
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
                    return Ok(Tree::Lit { text: self.src[start..self.pos].to_string(), span });
                }
                Some(c) => self.pos += c.len_utf8(),
            }
        }
    }

    fn char(&mut self) -> Result<Tree, Error> {
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
        Ok(Tree::Lit { text: self.src[start..self.pos].to_string(), span })
    }

    /// A number as Rust writes it: `16`, `0x3f`, `1_000`, `0.5`, `1e-5`, `8u8`, `2.0f32`. A
    /// `.` after the digits belongs to the number only if a digit follows, so that `0..n` is a
    /// range.
    fn number(&mut self) -> Tree {
        let (start, span) = (self.pos, self.span());
        let word = |c: char| c == '_' || c.is_ascii_alphanumeric();
        let radix = ["0x", "0o", "0b"].iter().any(|p| self.src[start..].starts_with(p));
        if radix {
            self.pos += 2;
            while self.peek().is_some_and(word) {
                self.pos += 1;
            }
            return Tree::Lit { text: self.src[start..self.pos].to_string(), span };
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
        Tree::Lit { text: self.src[start..self.pos].to_string(), span }
    }
}
