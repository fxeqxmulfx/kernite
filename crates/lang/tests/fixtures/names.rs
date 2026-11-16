//! The names that rustc makes of strings: a proc macro that kernite-lang's tests build and run with the `rustc` of the
//! machine, to hold what the lexer takes as a name to what Rust takes.

extern crate proc_macro;

use proc_macro::{Ident, Span, TokenStream};

/// The name that rustc makes of `string`, in NFC as it reads a name, or `-` if it makes none: `Ident::new` says so by a
/// panic.
fn name(string: &str) -> String {
    std::panic::catch_unwind(|| Ident::new(string, Span::call_site()).to_string()).unwrap_or_else(|_| "-".into())
}

/// Writes the name of each line of the file `$NAMES_IN` to the file `$NAMES_OUT`, a line each.
#[proc_macro]
pub fn names(_: TokenStream) -> TokenStream {
    // A panic that is caught says nothing.
    std::panic::set_hook(Box::new(|_| {}));
    let lines = std::fs::read_to_string(std::env::var("NAMES_IN").unwrap()).unwrap();
    let names: String = lines.lines().map(|line| name(line) + "\n").collect();
    std::fs::write(std::env::var("NAMES_OUT").unwrap(), names).unwrap();
    TokenStream::new()
}
