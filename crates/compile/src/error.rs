use std::fmt;

use kgen_core::{Class, PReg};

/// Why a program does not compile. Every case names the variables involved by the names the
/// program gave them, and where it could, the last note before the trouble.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// A label placed twice.
    LabelTwice {
        /// Its number.
        label: u32,
    },
    /// A jump to a label never placed.
    NoSuchLabel {
        /// Its number.
        label: u32,
    },
    /// Control runs past the last item.
    FallsOffEnd,
    /// A variable some path reads before anything writes it.
    ReadBeforeWrite {
        /// The variable.
        var: String,
    },
    /// A variable the calling convention places twice, in different registers.
    FixedTwice {
        /// The variable.
        var: String,
    },
    /// Two variables live at once that the calling convention puts in the same register.
    FixedClash {
        /// One.
        a: String,
        /// The other.
        b: String,
        /// The register both need.
        reg: PReg,
    },
    /// More variables of a file live at once than it has registers. The program has to keep
    /// fewer alive; nothing here spills.
    OutOfRegisters {
        /// The file.
        class: Class,
        /// The variable left without one.
        var: String,
        /// The last note before its first write.
        near: Option<String>,
        /// The variables live beside it that hold one, with the register each holds.
        taken: Vec<(String, PReg)>,
        /// Registers of the file there are.
        have: usize,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::LabelTwice { label } => write!(f, "label {label} is placed twice"),
            Error::NoSuchLabel { label } => write!(f, "a jump goes to label {label}, which is never placed"),
            Error::FallsOffEnd => write!(f, "control runs past the last instruction"),
            Error::ReadBeforeWrite { var } => write!(f, "`{var}` is read before anything writes it"),
            Error::FixedTwice { var } => write!(f, "`{var}` must be in two different registers"),
            Error::FixedClash { a, b, reg } => {
                write!(f, "`{a}` and `{b}` are live at once and both must be in {reg:?}")
            }
            Error::OutOfRegisters { class, var, near, taken, have } => {
                write!(f, "no {class:?} register left for `{var}`")?;
                if let Some(near) = near {
                    write!(f, " ({near})")?;
                }
                write!(f, ": all {have} are held by variables live beside it:")?;
                for (name, reg) in taken {
                    write!(f, " {name}={}", reg.hw)?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for Error {}
