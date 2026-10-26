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
    /// A variable the calling convention places twice, in different registers, which cannot
    /// move to another: one placed for a while alone, as a call's argument is.
    FixedTwice {
        /// The variable.
        var: String,
    },
    /// Two variables live at once that the calling convention puts in the same register, and
    /// neither of which can move to another.
    FixedClash {
        /// One.
        a: String,
        /// The other.
        b: String,
        /// The register both need.
        reg: PReg,
    },
    /// A variable the calling convention puts in a register that an instruction it is live
    /// across changes, as a call does, and which cannot move to another.
    FixedChanged {
        /// The variable.
        var: String,
        /// Its register.
        reg: PReg,
    },
    /// More variables of a file live at once than it has registers, and none of them may
    /// spill to make room, as when one instruction names more than there are: those spilling
    /// made are as short as they can be, and one an instruction writes as it jumps has no
    /// place for its store. The program has to keep fewer alive.
    OutOfRegisters {
        /// The file.
        class: Class,
        /// The variable left without one.
        var: String,
        /// The last note before its first write.
        near: Option<String>,
        /// The variables live beside it that hold one it could have had, with the register
        /// each holds.
        taken: Vec<(String, PReg)>,
        /// The registers it may not have, which an instruction it is live across changes, as a
        /// call does.
        changed: Vec<PReg>,
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
            Error::FixedChanged { var, reg } => {
                write!(f, "`{var}` must be in {reg:?}, which an instruction it is live across changes")
            }
            Error::OutOfRegisters { class, var, near, taken, changed, have } => {
                write!(f, "no {class:?} register left for `{var}`")?;
                if let Some(near) = near {
                    write!(f, " ({near})")?;
                }
                match (changed.len(), taken.is_empty()) {
                    (0, _) => write!(f, ": all {have} are held by variables live beside it:")?,
                    // What the call leaves, variables beside it hold: with none of those, it changes all.
                    (_, true) => write!(f, ": an instruction it is live across changes all {have}, as a call does")?,
                    (n, false) => write!(
                        f,
                        ": an instruction it is live across changes {n} of the {have}, as a call does, and variables \
                         live beside it hold the rest:"
                    )?,
                }
                for (name, reg) in taken {
                    write!(f, " {name}={}", reg.hw)?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for Error {}
