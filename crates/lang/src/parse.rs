//! The parser: a module's tokens to its items, a body's to statements.

use crate::Error;
use crate::ast::{
    BinOp, Block, Const, Expr, Fn, Generic, Instances, Item, ItemKind, Module, Param, Ret, Static, Stmt, Type, UnOp,
};
use crate::token::{Delim, Span, Tree, text};

/// The items of a module, from the tokens inside its braces.
pub fn module(trees: &[Tree]) -> Result<Module, Error> {
    let mut p = Parser::new(trees, trees.last().map_or(Span::default(), Tree::span));
    let mut items = vec![];
    while !p.done() {
        items.push(p.item()?);
    }
    Ok(Module { items })
}

/// The binary operators, longest first where one starts another, with their precedence.
const BINARY: &[(&str, BinOp, u8)] = &[
    ("||", BinOp::OrOr, 1),
    ("&&", BinOp::AndAnd, 2),
    ("==", BinOp::Eq, 3),
    ("!=", BinOp::Ne, 3),
    ("<=", BinOp::Le, 3),
    (">=", BinOp::Ge, 3),
    ("<<", BinOp::Shl, 7),
    (">>", BinOp::Shr, 7),
    ("<", BinOp::Lt, 3),
    (">", BinOp::Gt, 3),
    ("|", BinOp::Or, 4),
    ("^", BinOp::Xor, 5),
    ("&", BinOp::And, 6),
    ("+", BinOp::Add, 8),
    ("-", BinOp::Sub, 8),
    ("*", BinOp::Mul, 9),
    ("/", BinOp::Div, 9),
    ("%", BinOp::Rem, 9),
];

struct Parser<'a> {
    trees: &'a [Tree],
    pos: usize,
    /// Where errors at the end point.
    end: Span,
}

impl<'a> Parser<'a> {
    fn new(trees: &'a [Tree], end: Span) -> Self {
        Parser { trees, pos: 0, end }
    }

    fn done(&self) -> bool {
        self.pos >= self.trees.len()
    }

    fn peek(&self) -> Option<&'a Tree> {
        self.trees.get(self.pos)
    }

    fn peek_at(&self, n: usize) -> Option<&'a Tree> {
        self.trees.get(self.pos + n)
    }

    fn span(&self) -> Span {
        self.peek().map_or(self.end, Tree::span)
    }

    fn bump(&mut self) -> Option<&'a Tree> {
        let t = self.peek();
        self.pos += 1;
        t
    }

    fn error<T>(&self, message: impl Into<String>) -> Result<T, Error> {
        Err(Error { span: self.span(), message: message.into() })
    }

    /// What the next token is, for messages.
    fn found(&self) -> String {
        match self.peek() {
            None => "the end".into(),
            Some(t) => format!("`{}`", text(std::slice::from_ref(t))),
        }
    }

    fn is_ident(&self, name: &str) -> bool {
        matches!(self.peek(), Some(Tree::Ident { name: n, .. }) if n == name)
    }

    fn eat_ident(&mut self, name: &str) -> bool {
        let yes = self.is_ident(name);
        if yes {
            self.pos += 1;
        }
        yes
    }

    fn ident(&mut self) -> Result<(String, Span), Error> {
        match self.peek() {
            Some(Tree::Ident { name, span }) => {
                self.pos += 1;
                Ok((name.strip_prefix("r#").unwrap_or(name).to_string(), *span))
            }
            _ => self.error(format!("expected a name, found {}", self.found())),
        }
    }

    /// Whether the next tokens are the punctuation `op`, joined as one operator.
    fn is_op(&self, op: &str) -> bool {
        let n = op.chars().count();
        op.chars().enumerate().all(|(i, c)| {
            matches!(self.peek_at(i), Some(Tree::Punct { ch, joint, .. }) if *ch == c && (i + 1 == n || *joint))
        })
    }

    fn eat_op(&mut self, op: &str) -> bool {
        let yes = self.is_op(op);
        if yes {
            self.pos += op.chars().count();
        }
        yes
    }

    fn expect_op(&mut self, op: &str) -> Result<(), Error> {
        if self.eat_op(op) { Ok(()) } else { self.error(format!("expected `{op}`, found {}", self.found())) }
    }

    fn group(&mut self, delim: Delim) -> Option<(&'a [Tree], Span)> {
        match self.peek() {
            Some(Tree::Group { delim: d, trees, span }) if *d == delim => {
                self.pos += 1;
                Some((trees, *span))
            }
            _ => None,
        }
    }

    fn item(&mut self) -> Result<Item, Error> {
        let start = self.pos;
        // The module's own attribute, `#![..]`, passes through where it is.
        if self.is_op("#")
            && matches!(self.peek_at(1), Some(Tree::Punct { ch: '!', .. }))
            && matches!(self.peek_at(2), Some(Tree::Group { delim: Delim::Bracket, .. }))
        {
            self.pos += 3;
            return Ok(Item { tokens: start..self.pos, kind: ItemKind::Other });
        }
        let mut instances = None;
        while self.is_op("#") && matches!(self.peek_at(1), Some(Tree::Group { delim: Delim::Bracket, .. })) {
            if let Some(Tree::Group { trees, .. }) = self.peek_at(1)
                && let [Tree::Ident { name, span }, rest @ ..] = trees.as_slice()
                && name == "instances"
            {
                let at = |message: &str| Err(Error { span: *span, message: message.into() });
                let [Tree::Group { delim: Delim::Paren, trees: values, .. }] = rest else {
                    return at("`#[instances(..)]` lists what a function is compiled for, as in `#[instances(4, 8)]`");
                };
                if instances.is_some() {
                    return at("a function has one `#[instances(..)]`, which lists each instance it is compiled for");
                }
                instances = Some(Instances { values: args(values)?, span: *span, tokens: self.pos..self.pos + 2 });
            }
            self.pos += 2;
        }
        let attrs = start..self.pos;
        let misplaced = instances.as_ref().map(|i| Error {
            span: i.span,
            message: "`#[instances(..)]` goes on a function that names an ABI, which it lists the instances of".into(),
        });
        let head = self.pos;
        if self.eat_ident("pub") {
            self.group(Delim::Paren);
        }
        if self.eat_ident("static") {
            if let Some(e) = misplaced {
                return Err(e);
            }
            let s = self.static_rest(attrs, head..self.pos - 1)?;
            return Ok(Item { tokens: start..self.pos, kind: ItemKind::Static(s) });
        }
        let qualifier = |t: Option<&Tree>| matches!(t, Some(Tree::Ident { name, .. }) if ["fn", "unsafe", "extern", "async"].contains(&name.as_str()));
        if self.is_ident("const") && !qualifier(self.peek_at(1)) {
            if let Some(e) = misplaced {
                return Err(e);
            }
            self.pos += 1;
            let c = self.const_rest()?;
            return Ok(Item { tokens: start..self.pos, kind: ItemKind::Const(c) });
        }
        let qualifiers = self.pos;
        let mut abi = None;
        loop {
            if self.eat_ident("const") || self.eat_ident("unsafe") || self.eat_ident("async") {
                continue;
            }
            if self.is_ident("extern") {
                let span = self.span();
                self.pos += 1;
                abi = Some(match self.peek() {
                    Some(Tree::Lit { text, span }) if text.starts_with('"') => {
                        self.pos += 1;
                        (text.trim_matches('"').to_string(), *span)
                    }
                    _ => ("C".to_string(), span),
                });
                continue;
            }
            break;
        }
        if self.eat_ident("fn") {
            let f = self.function(attrs, instances, head..self.pos - 1, abi)?;
            return Ok(Item { tokens: start..self.pos, kind: ItemKind::Fn(f) });
        }
        if let Some(e) = misplaced {
            return Err(e);
        }
        // Anything else ends at its first `;` or brace group. An item cut short there, as `use
        // a::{b, c};` is, goes on as another; they pass through in order all the same.
        self.pos = qualifiers;
        while let Some(t) = self.bump() {
            if matches!(t, Tree::Punct { ch: ';', .. } | Tree::Group { delim: Delim::Brace, .. }) {
                break;
            }
        }
        Ok(Item { tokens: start..self.pos, kind: ItemKind::Other })
    }

    /// What follows `const`: `NAME: TYPE = VALUE;`.
    fn const_rest(&mut self) -> Result<Const, Error> {
        let (name, span) = self.ident()?;
        self.expect_op(":")?;
        let ty = self.ty()?;
        self.expect_op("=")?;
        let value = self.expr()?;
        self.expect_op(";")?;
        Ok(Const { name, span, ty, value })
    }

    /// What follows `static`: `NAME: TYPE;` or `NAME: TYPE = VALUE;`.
    fn static_rest(&mut self, attrs: std::ops::Range<usize>, vis: std::ops::Range<usize>) -> Result<Static, Error> {
        if self.is_ident("mut") {
            return self.error(
                "a static of the module is memory, and its type says what writes it, as in `static COUNTS: \
                 global<[u32; 16]>;`: it takes no `mut`",
            );
        }
        let (name, span) = self.ident()?;
        self.expect_op(":")?;
        let ty = self.ty()?;
        let value = if self.eat_op("=") { Some(self.expr()?) } else { None };
        self.expect_op(";")?;
        Ok(Static { attrs, vis, name, span, ty, value })
    }

    fn function(
        &mut self,
        attrs: std::ops::Range<usize>,
        instances: Option<Instances>,
        head: std::ops::Range<usize>,
        abi: Option<(String, Span)>,
    ) -> Result<Fn, Error> {
        let (name, span) = self.ident()?;
        let mut generics = vec![];
        if self.eat_op("<") {
            while !self.eat_op(">") {
                if !self.eat_ident("const") {
                    return self.error("a function's generics are values, as in `<const N: usize>`");
                }
                let (name, span) = self.ident()?;
                self.expect_op(":")?;
                let (ty, _) = self.ident()?;
                generics.push(Generic { name, span, ty });
                if !self.eat_op(",") {
                    self.expect_op(">")?;
                    break;
                }
            }
        }
        let params_at = self.pos;
        let Some((inner, _)) = self.group(Delim::Paren) else {
            return self.error(format!("expected `(`, found {}", self.found()));
        };
        let params = params(inner)?;
        let ret = if self.eat_op("->") {
            let (from, span) = (self.pos, self.span());
            while !self.done() && !matches!(self.peek(), Some(Tree::Group { delim: Delim::Brace, .. })) {
                if self.is_ident("where") || self.is_op(";") {
                    break;
                }
                self.pos += 1;
            }
            Some(Ret { ty: text(&self.trees[from..self.pos]), span, tokens: from..self.pos })
        } else {
            None
        };
        if self.is_ident("where") {
            return self.error("a kernel has no `where`");
        }
        let Some((body, span_body)) = self.group(Delim::Brace) else {
            return self.error(format!("a kernel is written in its body, and `{name}` has none"));
        };
        let body = block(body, span_body);
        Ok(Fn { attrs, instances, head, abi, name, span, generics, params, params_at, ret, body })
    }

    fn ty(&mut self) -> Result<Type, Error> {
        if let Some((inner, span)) = self.group(Delim::Bracket) {
            let mut p = Parser::new(inner, span);
            let elem = p.ty()?;
            if p.done() {
                return Ok(Type::Slice(Box::new(elem), span));
            }
            p.expect_op(";")?;
            let len = p.expr()?;
            p.finish()?;
            return Ok(Type::Array(Box::new(elem), len, span));
        }
        if let Some(Tree::Ident { name, span }) = self.peek()
            && matches!(self.peek_at(1), Some(Tree::Punct { ch: '<', .. }))
        {
            self.pos += 2;
            let mut args = vec![];
            while !self.eat_op(">") {
                args.push(self.ty()?);
                if !self.eat_op(",") {
                    self.expect_op(">")?;
                    break;
                }
            }
            return Ok(Type::Generic(name.strip_prefix("r#").unwrap_or(name).to_string(), args, *span));
        }
        let (from, span) = (self.pos, self.span());
        while !self.done() && !["=", ";", ",", ">"].iter().any(|op| self.is_op(op)) {
            self.pos += 1;
        }
        if from == self.pos {
            return self.error(format!("expected a type, found {}", self.found()));
        }
        Ok(Type::Name(text(&self.trees[from..self.pos]), span))
    }

    fn finish(&self) -> Result<(), Error> {
        if self.done() { Ok(()) } else { self.error(format!("unexpected {}", self.found())) }
    }

    /// `let (a, b): (T, U);` declares a variable of each type, and `let (a, b) = (x, y);`
    /// names each value: so many `let`s, in order.
    fn lets(&mut self) -> Result<Vec<Stmt>, Error> {
        self.pos += 1;
        let (inner, span) = self.group(Delim::Paren).expect("a tuple after `let`");
        let mut p = Parser::new(inner, span);
        let mut names = vec![];
        while !p.done() {
            p.eat_ident("mut");
            names.push(p.ident()?);
            if !p.done() {
                p.expect_op(",")?;
            }
        }
        let n = names.len();
        let tuple = |p: &mut Self, what: &str| -> Result<(&'a [Tree], Span), Error> {
            match p.group(Delim::Paren) {
                Some((inner, span)) => Ok((inner, span)),
                None => p.error(format!("expected the {what} of {n} names in parentheses, found {}", p.found())),
            }
        };
        let mut tys = vec![None; n];
        if self.eat_op(":") {
            let (inner, span) = tuple(self, "types")?;
            let mut p = Parser::new(inner, span);
            let mut given = vec![];
            while !p.done() {
                given.push(Some(p.ty()?));
                if !p.done() {
                    p.expect_op(",")?;
                }
            }
            if given.len() != n {
                return Err(Error { span, message: format!("{n} names, and {} types for them", given.len()) });
            }
            tys = given;
        }
        let mut values = vec![None; n];
        if self.eat_op("=") {
            let (inner, span) = tuple(self, "values")?;
            let given = args(inner)?;
            if given.len() != n {
                return Err(Error { span, message: format!("{n} names, and {} values for them", given.len()) });
            }
            values = given.into_iter().map(Some).collect();
        }
        self.expect_op(";")?;
        let lets = names.into_iter().zip(tys).zip(values);
        Ok(lets.map(|(((name, span), ty), value)| Stmt::Let { name, span, ty, value }).collect())
    }

    fn stmt(&mut self) -> Result<Stmt, Error> {
        if self.eat_ident("let") {
            self.eat_ident("mut");
            let (name, span) = self.ident()?;
            let ty = if self.eat_op(":") { Some(self.ty()?) } else { None };
            let value = if self.eat_op("=") { Some(self.expr()?) } else { None };
            self.expect_op(";")?;
            return Ok(Stmt::Let { name, span, ty, value });
        }
        if self.eat_ident("const") {
            return Ok(Stmt::Const(self.const_rest()?));
        }
        if self.eat_ident("for") {
            let (var, span) = self.ident()?;
            if !self.eat_ident("in") {
                return self.error(format!("expected `in`, found {}", self.found()));
            }
            let iter = self.expr()?;
            let body = self.braced()?;
            return Ok(Stmt::For { var, span, iter, body });
        }
        if self.is_ident("if") {
            return self.if_stmt();
        }
        if matches!(self.peek(), Some(Tree::Group { delim: Delim::Brace, .. })) {
            return Ok(Stmt::Block(self.braced()?));
        }
        if let Some(Tree::Ident { name, span }) = self.peek()
            && ["while", "loop", "match", "return", "break", "continue"].contains(&name.as_str())
        {
            return Err(Error {
                span: *span,
                message: format!("a kernel has no `{name}`: loops are labels and jumps, and `for` is unrolled"),
            });
        }
        let e = self.expr()?;
        if !self.done() {
            self.expect_op(";")?;
        }
        Ok(Stmt::Expr(e))
    }

    fn if_stmt(&mut self) -> Result<Stmt, Error> {
        let span = self.span();
        self.pos += 1;
        let cond = self.expr()?;
        let then = self.braced()?;
        let otherwise = if self.eat_ident("else") {
            Some(Box::new(if self.is_ident("if") { self.if_stmt()? } else { Stmt::Block(self.braced()?) }))
        } else {
            None
        };
        Ok(Stmt::If { cond, then, otherwise, span })
    }

    fn braced(&mut self) -> Result<Block, Error> {
        match self.group(Delim::Brace) {
            Some((trees, span)) => block(trees, span),
            None => self.error(format!("expected `{{`, found {}", self.found())),
        }
    }

    fn expr(&mut self) -> Result<Expr, Error> {
        let lo = self.binary(0)?;
        let span = self.span();
        let inclusive = if self.eat_op("..=") {
            true
        } else if self.eat_op("..") {
            false
        } else {
            return Ok(lo);
        };
        let hi = self.binary(0)?;
        Ok(Expr::Range { lo: Box::new(lo), hi: Box::new(hi), inclusive, span })
    }

    /// Operators of precedence `min` and up, left to right.
    fn binary(&mut self, min: u8) -> Result<Expr, Error> {
        let mut lhs = self.cast()?;
        while let Some(&(op_text, op, prec)) = BINARY.iter().find(|(t, ..)| self.is_op(t)) {
            if prec < min {
                break;
            }
            let span = self.span();
            self.pos += op_text.len();
            let rhs = self.binary(prec + 1)?;
            lhs = Expr::Binary(op, Box::new(lhs), Box::new(rhs), span);
        }
        Ok(lhs)
    }

    fn cast(&mut self) -> Result<Expr, Error> {
        let mut e = self.unary()?;
        while self.is_ident("as") {
            let span = self.span();
            self.pos += 1;
            let (ty, _) = self.ident()?;
            e = Expr::Cast(Box::new(e), ty, span);
        }
        Ok(e)
    }

    fn unary(&mut self) -> Result<Expr, Error> {
        let span = self.span();
        let op = if self.is_op("-") {
            UnOp::Neg
        } else if self.is_op("!") && !self.is_op("!=") {
            UnOp::Not
        } else {
            let e = self.primary()?;
            return self.postfix(e);
        };
        self.pos += 1;
        Ok(Expr::Unary(op, Box::new(self.unary()?), span))
    }

    fn primary(&mut self) -> Result<Expr, Error> {
        let span = self.span();
        let Some(t) = self.bump() else {
            return Err(Error { span, message: "expected an expression, found the end".into() });
        };
        match t {
            Tree::Lit { text, span } if text.starts_with('"') || text.starts_with("r\"") => {
                Ok(Expr::Str(unquote(text, *span)?, *span))
            }
            Tree::Lit { text, span } if text.starts_with('\'') => {
                Err(Error { span: *span, message: format!("a kernel has no use for the character {text}") })
            }
            Tree::Lit { text, span } => Ok(Expr::Lit(text.clone(), *span)),
            Tree::Ident { name, span } if name == "true" || name == "false" => Ok(Expr::Bool(name == "true", *span)),
            Tree::Ident { name, span } => {
                let mut path = vec![name.strip_prefix("r#").unwrap_or(name).to_string()];
                while self.is_op("::") && matches!(self.peek_at(2), Some(Tree::Ident { .. })) {
                    self.pos += 2;
                    path.push(self.ident()?.0);
                }
                Ok(Expr::Path(path, *span))
            }
            Tree::Group { delim: Delim::Paren, trees, span } => {
                let mut p = Parser::new(trees, *span);
                if p.done() {
                    return Ok(Expr::Tuple(vec![], *span));
                }
                let first = p.expr()?;
                if p.done() {
                    return Ok(Expr::Paren(Box::new(first), *span));
                }
                let mut items = vec![first];
                while p.eat_op(",") && !p.done() {
                    items.push(p.expr()?);
                }
                p.finish()?;
                Ok(Expr::Tuple(items, *span))
            }
            Tree::Group { delim: Delim::Bracket, trees, span } => {
                let mut p = Parser::new(trees, *span);
                if p.done() {
                    return Ok(Expr::Array(vec![], *span));
                }
                let first = p.expr()?;
                if p.eat_op(";") {
                    let len = p.expr()?;
                    p.finish()?;
                    return Ok(Expr::Repeat(Box::new(first), Box::new(len), *span));
                }
                let mut items = vec![first];
                while p.eat_op(",") && !p.done() {
                    items.push(p.expr()?);
                }
                p.finish()?;
                Ok(Expr::Array(items, *span))
            }
            Tree::Group { delim: Delim::Brace, span, .. } => {
                Err(Error { span: *span, message: "a block is a statement, not a value".into() })
            }
            Tree::Punct { ch, span, .. } => {
                Err(Error { span: *span, message: format!("expected an expression, found `{ch}`") })
            }
        }
    }

    fn postfix(&mut self, mut e: Expr) -> Result<Expr, Error> {
        loop {
            if self.is_op(".") && !self.is_op("..") {
                self.pos += 1;
                if let Some(Tree::Lit { text, span }) = self.peek() {
                    return Err(Error {
                        span: *span,
                        message: format!(
                            "a word after a dot starts with a letter or `_`: `.{text}` is written `._{text}`"
                        ),
                    });
                }
                let (name, span) = self.ident()?;
                e = match self.group(Delim::Paren) {
                    Some((trees, _)) => Expr::Method { recv: Box::new(e), name, args: args(trees)?, span },
                    None => Expr::Field(Box::new(e), name, span),
                };
            } else if let Some((trees, span)) = self.group(Delim::Bracket) {
                let mut p = Parser::new(trees, span);
                let index = p.expr()?;
                p.finish()?;
                e = Expr::Index(Box::new(e), Box::new(index), span);
            } else if let Some((trees, _)) = self.group(Delim::Paren) {
                let span = e.span();
                e = Expr::Call { func: Box::new(e), generics: vec![], args: args(trees)?, span };
            } else if self.is_op("::") && matches!(self.peek_at(2), Some(Tree::Punct { ch: '<', .. })) {
                self.pos += 3;
                let generics = self.generic_args()?;
                let span = e.span();
                e = match self.group(Delim::Paren) {
                    Some((trees, _)) => Expr::Call { func: Box::new(e), generics, args: args(trees)?, span },
                    None => Expr::Instance { func: Box::new(e), generics, span },
                };
            } else if self.is_op("::") && matches!(e, Expr::Field(..)) {
                return self.error("a word of a name is joined to the next by `__`, which PTX writes `::`: `L2__64B`");
            } else {
                return Ok(e);
            }
        }
    }

    /// What follows `::<`: literals, names and `{ expressions }`, up to `>`.
    fn generic_args(&mut self) -> Result<Vec<Expr>, Error> {
        let mut out = vec![];
        while !self.eat_op(">") {
            let arg = if let Some((trees, span)) = self.group(Delim::Brace) {
                let mut p = Parser::new(trees, span);
                let e = p.expr()?;
                p.finish()?;
                e
            } else if self.is_op("-") {
                self.unary()?
            } else {
                self.primary()?
            };
            out.push(arg);
            if !self.eat_op(",") {
                self.expect_op(">")?;
                break;
            }
        }
        Ok(out)
    }
}

/// The statements in a block.
fn block(trees: &[Tree], span: Span) -> Result<Block, Error> {
    let mut p = Parser::new(trees, span);
    let mut stmts = vec![];
    while !p.done() {
        if p.eat_op(";") {
            continue;
        }
        if p.is_ident("let") && matches!(p.peek_at(1), Some(Tree::Group { delim: Delim::Paren, .. })) {
            stmts.extend(p.lets()?);
        } else {
            stmts.push(p.stmt()?);
        }
    }
    Ok(Block { stmts, span })
}

/// Arguments, separated by commas.
fn args(trees: &[Tree]) -> Result<Vec<Expr>, Error> {
    let mut p = Parser::new(trees, trees.last().map_or(Span::default(), Tree::span));
    let mut out = vec![];
    while !p.done() {
        out.push(p.expr()?);
        if !p.done() {
            p.expect_op(",")?;
        }
    }
    Ok(out)
}

/// A function's parameters: `name: type`, separated by commas.
fn params(trees: &[Tree]) -> Result<Vec<Param>, Error> {
    let mut p = Parser::new(trees, trees.last().map_or(Span::default(), Tree::span));
    let mut out = vec![];
    while !p.done() {
        p.eat_ident("mut");
        let (name, span) = p.ident()?;
        p.expect_op(":")?;
        let ty = p.ty()?;
        out.push(Param { name, span, ty });
        if !p.done() {
            p.expect_op(",")?;
        }
    }
    Ok(out)
}

/// A string literal's text.
fn unquote(lit: &str, span: Span) -> Result<String, Error> {
    if let Some(raw) = lit.strip_prefix('r') {
        let hashes = raw.len() - raw.trim_start_matches('#').len();
        return Ok(raw[hashes + 1..raw.len() - hashes - 1].to_string());
    }
    let mut out = String::new();
    let mut chars = lit[1..lit.len() - 1].chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        out.push(match chars.next() {
            Some('n') => '\n',
            Some('t') => '\t',
            Some('r') => '\r',
            Some('0') => '\0',
            Some(c @ ('\\' | '"' | '\'')) => c,
            other => {
                return Err(Error {
                    span,
                    message: format!("the escape `\\{}` is not one kgen reads", other.unwrap_or(' ')),
                });
            }
        });
    }
    Ok(out)
}
