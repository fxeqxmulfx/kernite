//! Memory a kernel declares, and the module's, a `static`: its shape, what it holds, and
//! whether the module has room for it.

use kgen_lang::{IntTy, Value};

use crate::forms::Space;
use crate::ty::{Kind, Ty};

/// Memory a kernel declares, bytes in shared or local memory, or the module's, a [`Static`],
/// in global or constant memory: at an address that is a multiple of `align`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Memory {
    /// Its name in the PTX: unique in the entry, and in the module if it has no size or is
    /// the module's.
    pub name: String,
    /// [`Space::Shared`] or [`Space::Local`], or for the module's, [`Space::Global`] or
    /// [`Space::Const`].
    pub space: Space,
    /// What its address is a multiple of, in bytes: a power of two.
    pub align: u64,
    /// How many bytes it is; `None` for the shared memory the launch says the size of.
    pub bytes: Option<u64>,
}

/// Memory of the module's, which a `static` of the source declares: bytes in global or
/// constant memory that each of its kernels names, declared once, at the module's scope, which
/// the host reads and writes by its symbol, and which keeps what is written from one launch
/// to the next. Constant memory the kernels only read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Static {
    /// What the source calls it, which a kernel names it by: `COUNTS`.
    pub name: String,
    /// Its type, as the source declares it: `global<[u32; 16]>`.
    pub ty: String,
    /// Its memory, named by its symbol, with its size.
    pub memory: Memory,
    /// What each of its elements is.
    pub elem: Ty,
    /// What it holds until something writes it, as bytes; zeros past their end.
    pub init: Vec<u8>,
}

/// The constant memory a module has: 64 KB.
const CONSTANT: u64 = 64 << 10;

impl Static {
    /// The static `name`, of the type `ty` as the source declares it, `global<[u32; 16]>`, or
    /// `constant<[f32; 4], 16>` at a multiple of 16, named `symbol` in PTX: holding zeros until
    /// [`init`](Static::init) says otherwise. Or why it is none.
    pub fn new(name: &str, symbol: &str, ty: &str) -> Result<Static, String> {
        let (space, word) = match ty.split_once('<').map(|(word, _)| word) {
            Some("global") => (Space::Global, "global"),
            Some("constant") => (Space::Const, "constant"),
            _ => {
                return Err(format!(
                    "a static of the module is memory: in global memory, `global<[u32; 16]>`, or in constant memory, \
                     which its kernels only read, `constant<[f32; 4]>`; not `{ty}`"
                ));
            }
        };
        let (elem, bytes, align) = shape(ty, word)?;
        if elem.host().is_none() {
            return Err(format!(
                "a static holds numbers the host reads as Rust's: `b8` to `b64`, `u8` to `u64`, `s8` to `s64`, \
                 `f16`, `bf16`, `f32` and `f64`; not `{}`",
                elem.name()
            ));
        }
        let Some(bytes) = bytes else {
            return Err(format!(
                "`{ty}` has no size, which only shared memory may leave to the launch: `{word}<[{}; 64]>`",
                elem.name()
            ));
        };
        let memory = Memory { name: symbol.to_string(), space, align, bytes: Some(bytes) };
        Ok(Static { name: name.to_string(), ty: ty.to_string(), memory, elem, init: vec![] })
    }

    /// Makes it hold `value` until something writes it: an array of as many elements as it
    /// has, each of its element's type, `[1.5, 2.0]`; or one, for memory of one that is no
    /// array, `global<f32>`. Or why it cannot.
    pub fn init(&mut self, value: &Value) -> Result<(), String> {
        let (ty, elem) = (&self.ty, self.elem.name());
        let len = self.memory.bytes.unwrap_or(0) / (u64::from(self.elem.bits()) / 8);
        let array = ty.split_once('<').is_some_and(|(_, args)| args.starts_with('['));
        let items = match (value, array) {
            (Value::Array(xs), true) if xs.len() as u64 == len => xs.as_slice(),
            (Value::Array(xs), true) => return Err(format!("`{ty}` holds {len} elements; this has {}", xs.len())),
            (x, true) => return Err(format!("`{ty}` holds an array of `{elem}`; this is {}", x.kind())),
            (x, false) => std::slice::from_ref(x),
        };
        let mut init = vec![];
        for (i, x) in items.iter().enumerate() {
            init.extend(datum(self.elem, x, &if array { format!("element {i}") } else { "this".into() })?);
        }
        self.init = init;
        Ok(())
    }

    /// Its declaration, at the module's scope, `.global .align 4 .b8 COUNTS[64];`: with what it
    /// holds until something writes it, but the zeros that ends in, which PTX fills in.
    pub fn declaration(&self) -> String {
        let m = &self.memory;
        let mut out = format!(".{} .align {} .b8 {}[{}]", m.space.word(), m.align, m.name, m.bytes.unwrap_or(0));
        let end = self.init.iter().rposition(|&b| b != 0).map_or(0, |i| i + 1);
        if end > 0 {
            let bytes: Vec<String> = self.init[..end].iter().map(u8::to_string).collect();
            out += &format!(" = {{{}}}", bytes.join(", "));
        }
        out + ";\n"
    }
}

/// `x` as the bytes of a `t` in memory, the lowest first; or why it is none, of `x` as `what`
/// names it, `this` or `element 2`. An integer fits the bits of a `u` type as a number of no
/// sign, of an `s` type as a signed one, and of a `b` type as either; a float is one of its own
/// type, or bits of its size.
fn datum(t: Ty, x: &Value, what: &str) -> Result<Vec<u8>, String> {
    let (name, bits) = (t.name(), t.bits());
    // A type's name as it is read: an `s8`, an `f32`, an `i64`, a `u8`.
    let a = |name: &str| if name.starts_with(['f', 'i', 's']) { "an" } else { "a" };
    match *x {
        Value::Int(v, ty) if t.int() || t.kind() == Kind::Bit => {
            let fits = |ty: IntTy| match t.kind() {
                Kind::Unsigned => !ty.signed(),
                Kind::Signed => ty.signed(),
                _ => true,
            };
            if let Some(ty) = ty
                && (ty.bits() != bits || !fits(ty))
            {
                return Err(format!("{what} is {} `{}`, not {} `{name}`", a(ty.name()), ty.name(), a(name)));
            }
            let (lo, hi) = match t.kind() {
                Kind::Unsigned => (0, (1i128 << bits) - 1),
                Kind::Signed => (-(1i128 << (bits - 1)), (1i128 << (bits - 1)) - 1),
                _ => (-(1i128 << (bits - 1)), (1i128 << bits) - 1),
            };
            if !(lo..=hi).contains(&v) {
                return Err(format!("{what} is {v}, out of range for `{name}`"));
            }
            Ok(v.to_le_bytes()[..bits as usize / 8].to_vec())
        }
        Value::Int(v, _) if t.kind() == Kind::Float => {
            Err(format!("{what} is an integer, and {} `{name}` a float: `{v}.0`", a(name)))
        }
        Value::F32(v) if matches!(t, Ty::F32 | Ty::B32) => Ok(v.to_le_bytes().to_vec()),
        Value::F64(v) if matches!(t, Ty::F64 | Ty::B64) => Ok(v.to_le_bytes().to_vec()),
        Value::F16(v) if matches!(t, Ty::F16 | Ty::B16) => Ok(v.to_le_bytes().to_vec()),
        Value::Bf16(v) if matches!(t, Ty::Bf16 | Ty::B16) => Ok(v.to_le_bytes().to_vec()),
        ref x => Err(format!("{what} is {}, not {} `{name}`", x.kind(), a(name))),
    }
}

/// Why `s` does not fit beside the statics `before` it, if it does not: in the 64 KB of
/// constant memory a module has, which `ptxas` lays out in the order the module declares it,
/// each at the multiple its address is; or in the global memory `ptxas` lays out with it.
pub fn fits(before: &[Static], s: &Static) -> Result<(), String> {
    let space = s.memory.space;
    let mine = before.iter().chain([s]).filter(|b| b.memory.space == space).map(|b| &b.memory);
    if space == Space::Const {
        let all = mine.fold(0u64, |end, m| end.next_multiple_of(m.align).saturating_add(m.bytes.unwrap_or(0)));
        if all > CONSTANT {
            return Err(format!("this makes the module's constant memory {all} bytes, and a module has 64 KB"));
        }
        return Ok(());
    }
    // ptxas 13.4 writes the size of the section it lays a module's global memory out in,
    // `.nv.global`, or `.nv.global.init` for what holds more than zeros, in 32 bits, and cuts
    // one of 4 GB or more to them without a word: the memory past is no memory's, and the
    // driver's copies run past what it allocated. It cuts a symbol's own size to 61 bits, and
    // writes 0 for 2^63; only at 2^64 does it refuse. The bytes of both sections at once,
    // each memory's padding counted, are under 4 GB.
    let all = mine.fold(0u64, |sum, m| sum.saturating_add(m.bytes.unwrap_or(0)).saturating_add(m.align - 1));
    if all >= 1 << 32 {
        return Err(format!(
            "this makes the module's global memory as much as {all} bytes, each static aligned, and `ptxas` \
             writes its size in 32 bits: a module has less than 4 GB"
        ));
    }
    Ok(())
}

/// The types memory holds, for an error about one that is none.
const HELD: &str = "`b8` to `b128`, `u8` to `u64`, `s8` to `s64`, `f16`, `f16x2`, `bf16`, `bf16x2`, `tf32`, `f32` and \
                    `f64`";

/// What memory of the type `ty`, `word<args>`, holds: the type of its elements; how many bytes,
/// `None` for an array of no length, `[f32]`; and what its address is a multiple of, a power
/// of two, its element's bytes unless it says more: `shared<[f32; 256], 16>`.
pub(crate) fn shape(ty: &str, word: &str) -> Result<(Ty, Option<u64>, u64), String> {
    let wrong = || {
        format!(
            "`{word}<..>` is an array of PTX's types and what its address is a multiple of, `{word}<[f32; 256], 16>`, \
             or one, `{word}<f32>`; not `{ty}`"
        )
    };
    let args = ty.strip_prefix(word).and_then(|t| t.strip_prefix('<')).and_then(|t| t.strip_suffix('>'));
    let args = args.ok_or_else(wrong)?;
    let (array, align) = match args.rsplit_once(", ") {
        Some((array, align)) => (array, Some(align)),
        None => (args, None),
    };
    if array.contains(", ") {
        return Err(wrong());
    }
    let (elem, len) = match array.strip_prefix('[').and_then(|a| a.strip_suffix(']')) {
        Some(inner) => match inner.split_once("; ") {
            Some((elem, len)) => (elem, Some(len.parse::<u64>().map_err(|_| wrong())?)),
            None => (inner, None),
        },
        None => (array, Some(1)),
    };
    let elem = match Ty::parse(elem) {
        Some(Ty::Pred) => return Err("memory holds no `pred`, which is a register's alone: a `u8` holds a bool".into()),
        None => return Err(format!("memory holds {HELD}; not `{elem}`")),
        Some(t) => t,
    };
    let size = u64::from(elem.bits()) / 8;
    let align = match align {
        None => size,
        Some(a) => match a.parse::<u64>() {
            Ok(a) if a.is_power_of_two() && a >= size => a,
            Ok(a) if a.is_power_of_two() => {
                return Err(format!("an array of `{}` is aligned to {size} at least, not {a}", elem.name()));
            }
            _ => return Err(format!("what an address is a multiple of is a power of two, not {a}")),
        },
    };
    let bytes = match len {
        Some(0) => return Err(format!("`{ty}` holds nothing")),
        Some(n) => Some(n.checked_mul(size).ok_or_else(|| format!("`{ty}` is more bytes than there are"))?),
        None => None,
    };
    Ok((elem, bytes, align))
}
