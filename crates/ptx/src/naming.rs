//! The names kgen gives in PTX: in the characters PTX takes, and mangled as C++ names them.

use kgen_lang::{IntTy, Value};

/// `name` in the characters PTX takes in a name: ASCII's letters, digits and `_` as they are;
/// one beyond ASCII as Rust's legacy mangling writes it, `$u3c3$` for `σ`, which is no other
/// name's, since no name of Rust's holds a `$`; and any other `_`, as Cargo writes a crate's
/// `-`. `ptxas` takes `$` anywhere in a name, and nothing beyond ASCII.
pub fn ident(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for c in name.chars() {
        match c {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '_' => out.push(c),
            _ if c.is_ascii() => out.push('_'),
            _ => out += &format!("$u{:x}$", u32::from(c)),
        }
    }
    out
}

/// `path`, namespaces each in the one before, as C++ mangles their names for [`symbol`]:
/// `<length><name>` each, `["kgen", "my_app"]` as `4kgen6my_app`. Each name is as [`ident`]
/// writes it, and one that starts with a digit has `_` before it.
pub fn scope(path: &[&str]) -> String {
    let mut out = String::new();
    for name in path {
        let mut name = ident(name);
        if name.starts_with(|c: char| c.is_ascii_digit()) {
            name.insert(0, '_');
        }
        out += &format!("{}{name}", name.len());
    }
    out
}

/// The symbol of `name`, a kernel or a function, in `scope`, which [`scope`] writes, for the
/// compile-time arguments `args`: as C++ mangles the name of a template's instance there,
/// under Itanium's ABI, which the tools that read PTX demangle. `cu++filt`, as Nsight and
/// `cuda-gdb` do, shows `_ZN4kgen6my_app3gpu5saxpyE` as `kgen::my_app::gpu::saxpy`, and
/// `_ZN4kgen3gpu4tileILm4EEE` as `kgen::gpu::tile<(unsigned long)4>`. Without a scope and
/// without arguments, the symbol is `name`, as [`ident`] writes it.
///
/// `args` are integers and `bool`s, as a kernel's compile-time arguments are: an integer of
/// its type, C++'s of its size, and one of no type an `i32`, as Rust has it.
pub fn symbol(scope: &str, name: &str, args: &[Value]) -> String {
    if scope.is_empty() && args.is_empty() {
        return ident(name);
    }
    let mut out = match scope {
        "" => "_Z".to_string(),
        _ => format!("_ZN{scope}"),
    };
    out += &self::scope(&[name]);
    if !args.is_empty() {
        out += "I";
        for a in args {
            out += &match a {
                Value::Bool(b) => format!("Lb{}E", u8::from(*b)),
                Value::Int(x, ty) => {
                    let ty = match ty.unwrap_or(IntTy::I32) {
                        IntTy::I8 => 'a',
                        IntTy::I16 => 's',
                        IntTy::I32 => 'i',
                        IntTy::I64 | IntTy::Isize => 'l',
                        IntTy::U8 => 'h',
                        IntTy::U16 => 't',
                        IntTy::U32 => 'j',
                        IntTy::U64 | IntTy::Usize => 'm',
                    };
                    let sign = if *x < 0 { "n" } else { "" };
                    format!("L{ty}{sign}{}E", x.unsigned_abs())
                }
                other => unreachable!("a kernel's compile-time argument is an integer or a bool, not {}", other.kind()),
            };
        }
        out += "E";
    }
    if !scope.is_empty() {
        out += "E";
    }
    out
}

/// The symbol of `name`, memory of the function whose symbol is `function` that is the
/// module's rather than the function's, as C++ names a function's static and `nvcc` a
/// kernel's `__shared__` memory: `_ZZ`, the function, `E`, the name. `cu++filt` shows
/// `_ZZN4kgen3gpu7reverseEE3buf` as `kgen::gpu::reverse::buf`, and `_ZZ7reverseE3buf` as
/// `reverse::buf`.
pub(crate) fn local(function: &str, name: &str) -> String {
    // A symbol that is not mangled is a name as `ident` writes it already.
    let function = match function.strip_prefix("_Z") {
        Some(mangled) => mangled.to_string(),
        None => format!("{}{function}", function.len()),
    };
    format!("_ZZ{function}E{}", scope(&[name]))
}
