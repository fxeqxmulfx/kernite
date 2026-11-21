//! Each attribute that a static of GPU kernels takes, which its method, `Kernels::COUNTS`, has, and Rust says of the
//! method what it says of any: a doc, `cfg`, `cfg_attr(..)`, `deprecated`, the level of a lint, and `rustfmt`'s. A
//! method with no doc is told of by `missing_docs` at the static, as any item of the author's is, whether its module
//! has one static or more. A deprecated method is told of at its use, in `uses`. A static of x86 kernels, in `cpu`, is
//! a static of Rust's: Rust takes `#[used]` of it, which a static of GPU kernels refuses, and tells of its use as it
//! tells of a static's.

#![deny(deprecated)]

/// Statics with a doc, in each form, or with a level that lets a static go without one: no method is told of.
#[kernite_macros::kernels]
pub mod silent {
    #![deny(missing_docs)]

    /// A doc.
    pub static DOCUMENTED: global<[u32; 16]>;

    #[doc = "A doc as an attribute."]
    pub static ATTRIBUTE: global<[u32; 16]>;

    #[doc(hidden)]
    pub static HIDDEN: global<[u32; 16]>;

    #[cfg_attr(all(), doc = "A doc in a `cfg_attr(..)`.")]
    pub static CONFIGURED: global<[u32; 16]>;

    #[allow(missing_docs)]
    pub static ALLOWED: global<[u32; 16]>;

    #[cfg_attr(all(), allow(missing_docs))]
    pub static ALLOWED_CONFIGURED: global<[u32; 16]>;

    #[expect(missing_docs)]
    pub static EXPECTED: global<[u32; 16]>;
}

/// A static with none of them: the method is told of, as any method with no doc is.
#[kernite_macros::kernels]
pub mod bare {
    #![deny(missing_docs)]

    pub static BARE: global<[u32; 16]>;
}

/// A static whose doc and level are in a `cfg_attr(..)` that does not hold: Rust leaves them out of the method, as it
/// does of any item, and tells of it.
#[kernite_macros::kernels]
pub mod off {
    #![deny(missing_docs)]

    #[cfg_attr(any(), doc = "A doc that the condition leaves out.")]
    #[cfg_attr(any(), allow(missing_docs))]
    pub static OFF: global<[u32; 16]>;
}

/// A level that no module gives, which each static gives its method: `warn` says a warning, `deny` and `forbid` an
/// error, and an `expect` that no lint meets is a warning, at the attribute.
#[kernite_macros::kernels]
pub mod warned {
    #[warn(missing_docs)]
    pub static WARNED: global<[u32; 16]>;
}

/// The same, at `deny`.
#[kernite_macros::kernels]
pub mod denied {
    #[deny(missing_docs)]
    pub static DENIED: global<[u32; 16]>;
}

/// The same, at `forbid`.
#[kernite_macros::kernels]
pub mod forbidden {
    #[forbid(missing_docs)]
    pub static FORBIDDEN: global<[u32; 16]>;
}

/// An `expect` that nothing meets.
#[kernite_macros::kernels]
pub mod unmet {
    #[expect(unused_variables)]
    pub static UNMET: global<[u32; 16]>;
}

/// `cfg` leaves out a static and its method, which would be told of, for it has no doc; `deprecated` is told of at each
/// use of the method; and a table that `rustfmt` skips is a static as any is.
#[kernite_macros::kernels]
pub mod kept {
    #![deny(missing_docs)]

    /// A static that `cfg` keeps.
    #[cfg(all())]
    pub static HERE: global<[u32; 16]>;

    #[cfg(any())]
    pub static GONE: global<[u32; 16]>;

    /// A static that is deprecated.
    #[deprecated = "use `HERE`"]
    pub static OLD: global<[u32; 16]>;

    /// A table, as it is written.
    #[rustfmt::skip]
    pub static TABLE: constant<[f32; 4]> = [ 1.0,  2.0,
                                             3.0,  4.0 ];
}

/// A static of x86 kernels has the attributes it is written with, as a static of Rust's does.
#[kernite_macros::kernels]
pub mod cpu {
    /// A static that the program keeps, though no kernel names it.
    #[used]
    pub static KEPT: global<[u32; 16]>;

    /// A static that is deprecated.
    #[deprecated = "use `KEPT`"]
    pub static OLD: global<[u32; 16]>;

    /// Does nothing, in a module of x86 kernels.
    pub unsafe extern "sysv64" fn nothing() {}
}

/// Uses each static there is: the methods of `kept`, and the statics of `cpu`.
pub fn uses(kernels: &kept::Kernels) {
    let _ = (kernels.HERE(), kernels.OLD(), kernels.TABLE(), cpu::KEPT.len(), cpu::OLD.len());
}
