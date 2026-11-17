//! Each attribute that is not for memory on the GPU, on a static of GPU kernels: kernite's error at the attribute,
//! once, in the words of one, and Rust says nothing, for the method that gives the memory, `Kernels::COUNTS`, is
//! written without it. The first six are the attributes of the host's symbols and code that it would have exported,
//! placed or compiled as its own; each is a module of its own, a static and no kernel, which is a module of the GPU's.

#[kernite_macros::kernels]
pub mod no_mangle {
    #[unsafe(no_mangle)]
    pub static COUNTS: global<[u32; 16]>;
}

#[kernite_macros::kernels]
pub mod export_name {
    #[unsafe(export_name = "EXPORTED")]
    pub static COUNTS: global<[u32; 16]>;
}

#[kernite_macros::kernels]
pub mod link_section {
    #[unsafe(link_section = ".mine")]
    pub static COUNTS: global<[u32; 16]>;
}

#[kernite_macros::kernels]
pub mod cold {
    #[cold]
    pub static COUNTS: global<[u32; 16]>;
}

#[kernite_macros::kernels]
pub mod track_caller {
    #[track_caller]
    pub static COUNTS: global<[u32; 16]>;
}

#[kernite_macros::kernels]
pub mod used {
    #[used]
    pub static COUNTS: global<[u32; 16]>;
}

// The same, in the forms Rust reads as the same attribute: written raw, on a static named raw, which the error says as
// written; in a `cfg_attr(..)` whatever its condition, beside one the method takes, and inside another `cfg_attr(..)`.
#[kernite_macros::kernels]
pub mod forms {
    #[r#used]
    pub static r#type: global<[u32; 16]>;

    #[unsafe(r#no_mangle)]
    pub static RAW_UNSAFE: global<[u32; 16]>;

    #[cfg_attr(all(), unsafe(no_mangle))]
    pub static HOLDS: global<[u32; 16]>;

    #[cfg_attr(any(), used)]
    pub static FALSE: global<[u32; 16]>;

    #[cfg_attr(all(), doc = "A doc, which the method takes.", used)]
    pub static BESIDE: global<[u32; 16]>;

    #[cfg_attr(all(), cfg_attr(any(), cold))]
    pub static NESTED: global<[u32; 16]>;
}

// Any other attribute is refused as well, for an item of Rust's of its own, a tool's that is not `rustfmt`'s, and one
// that no tool or macro reads: a path is said as written, and an attribute's arguments and value are not.
#[kernite_macros::kernels]
pub mod others {
    #[must_use]
    pub static MUST_USE: global<[u32; 16]>;

    #[derive(Debug)]
    pub static DERIVE: global<[u32; 16]>;

    #[repr(C)]
    pub static REPR: global<[u32; 16]>;

    #[clippy::cognitive_complexity = "5"]
    pub static CLIPPY: global<[u32; 16]>;

    #[nothing_reads_this]
    pub static UNKNOWN: global<[u32; 16]>;
}

// Each attribute of a static is judged, in the order written, beside the ones the method takes; and a constant is
// memory on the GPU as global memory is. A static that `cfg` leaves out says nothing, and one it keeps says it.
#[kernite_macros::kernels]
pub mod beside {
    /// A doc, which the method takes.
    #[used]
    #[allow(unused)]
    #[cold]
    pub static MANY: global<[u32; 16]>;

    #[used]
    pub static WEIGHTS: constant<[f32; 2]> = [0.5, 1.5];

    #[cfg(any())]
    #[used]
    pub static GONE: global<[u32; 16]>;

    #[cfg(all())]
    #[used]
    pub static HERE: global<[u32; 16]>;
}

// A static whose attribute is refused is memory as it was, which a kernel names and the host reads: the method that
// gives it is there, and Rust, which reads `read` below, says nothing of its use.
#[kernite_macros::kernels]
pub mod used_as_memory {
    #[used]
    pub static COUNTS: global<[u32; 16]>;

    pub unsafe extern "ptx-kernel" fn bump() {
        red.global.add.u32([COUNTS], 1);
    }
}

/// Reads the memory, in Rust and outside the module, where rustc checks the use of the method as it does any.
pub fn read(kernels: &used_as_memory::Kernels) -> Vec<u32> {
    kernels.COUNTS().read().unwrap()
}
