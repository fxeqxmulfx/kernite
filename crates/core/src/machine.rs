use crate::reg::{Class, PReg};

/// What the use cases need to know about a machine and its calling convention: which
/// registers they may hand out, and which of those a kernel must give back as it found them.
///
/// The instruction sets implement it; the allocator's tests implement it with a handful of
/// registers, to run out on purpose.
pub trait Machine {
    /// The registers of `class` a kernel may use, in the order the allocator tries them.
    fn allocatable(&self, class: Class) -> &[PReg];

    /// The registers a kernel must restore before it returns.
    fn callee_saved(&self) -> &[PReg];
}
