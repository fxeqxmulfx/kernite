//! What a function takes and returns, and the types PTX passes each value as.

use crate::ty::Ty;

/// What a function takes and returns, as the kernels that call it see it: PTX's
/// prototype of it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Prototype {
    /// The type of each parameter, in order.
    pub params: Vec<Ty>,
    /// What it returns.
    pub returns: Returns,
}

/// What a function returns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Returns {
    /// Nothing, `()`.
    Nothing,
    /// A value of this type, which `ret(v)` gives back.
    Value(Ty),
    /// It never returns, `-> !`: PTX's `.noreturn`.
    Never,
}

impl Prototype {
    /// The prototype of a function whose parameters are of the types `params` and which
    /// returns `result`, as Rust writes them: PTX's types, and `!` for one that never
    /// returns; or why it is none PTX has.
    pub fn of(params: &[&str], result: Option<&str>) -> Result<Prototype, String> {
        let params = params.iter().map(|t| passable(t)).collect::<Result<_, _>>()?;
        let returns = result.map_or(Ok(Returns::Nothing), returns)?;
        Ok(Prototype { params, returns })
    }
}

/// The type of a function's parameter written `ty`, if it is one: one of PTX's, but a
/// predicate, which PTX passes to no function.
pub fn passable(ty: &str) -> Result<Ty, String> {
    match Ty::parse(ty) {
        Some(Ty::Pred) => Err(NO_PRED.into()),
        Some(t) => Ok(t),
        None => {
            let ptx = match ty {
                _ if kernite_lang::pointer(ty) != Ok(false) => "an address is a `u64`",
                "i8" => "PTX writes it `s8`",
                "i16" => "PTX writes it `s16`",
                "i32" => "PTX writes it `s32`",
                "i64" | "isize" => "PTX writes it `s64`",
                "usize" => "PTX writes it `u64`",
                _ => "as its variables are: `u32`, `f32`, `b64`",
            };
            Err(format!("a function takes and returns PTX's types, {ptx}; not `{ty}`"))
        }
    }
}

/// What a function returns, as Rust writes it: `()`, one of PTX's types, or `!` for never.
pub(crate) fn returns(ty: &str) -> Result<Returns, String> {
    match ty {
        "()" => Ok(Returns::Nothing),
        "!" => Ok(Returns::Never),
        _ if ty.starts_with('(') => {
            Err("a function returns one value at most, as PTX's do from Fermi on, not a tuple".into())
        }
        _ => passable(ty).map(Returns::Value),
    }
}

/// The type a value of type `t` is passed as, and loaded and stored by `ld.param` and
/// `st.param`: its register's, but bits for a float of 16 bits or two, which `ld` and `st`
/// have no type for.
pub fn passed(t: Ty) -> Ty {
    match t {
        Ty::F16 => Ty::B16,
        Ty::F16x2 => Ty::B32,
        t => t.decl(),
    }
}

/// The type of a function's `.param` for a value of type `t`: bits for one of 8 or 16 bits,
/// which `ptxas` passes to a function only as bits, and otherwise as [`passed`] has it.
pub fn formal(t: Ty) -> Ty {
    match t.bits() {
        8 => Ty::B8,
        16 => Ty::B16,
        _ => passed(t),
    }
}

/// Why a function takes no predicate.
pub(crate) const NO_PRED: &str =
    "PTX passes no `pred` to a function, nor back: `selp.u32(x, 1, 0, p)` makes it a `u32`";
