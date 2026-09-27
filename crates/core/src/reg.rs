use std::fmt::Debug;

/// A register file. Registers of different files never share storage, so they never
/// compete for the same register.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Class {
    /// General purpose: integers, counters, addresses.
    Gpr,
    /// Vector.
    Vec,
}

/// A variable of a program before allocation. It may be written any number of times; it
/// gets one register for all of them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VReg {
    /// Its index in [`crate::Program::regs`].
    pub id: u32,
    /// The file its register comes from.
    pub class: Class,
}

/// A machine register: its file and its number in the instruction encoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PReg {
    /// The file.
    pub class: Class,
    /// The number the encoding gives it.
    pub hw: u8,
}

/// What an operand can name: a variable before allocation, a machine register after.
pub trait Register: Copy + Eq + Debug {
    /// The file it belongs to.
    fn class(self) -> Class;
}

impl Register for VReg {
    fn class(self) -> Class {
        self.class
    }
}

impl Register for PReg {
    fn class(self) -> Class {
        self.class
    }
}
