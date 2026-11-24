//! A crate that names the PTX of `much_ptx`, as the crate of a program that launches the kernel does. Rust makes the
//! constant again here, and counts its steps again: it says nothing of it, and the text is the one the macros wrote.

/// The bytes of the PTX, known now: Rust makes the constant, and counts its steps, in this crate.
pub const LEN: usize = much_ptx::gpu::PTX.len();

#[test]
fn the_ptx_is_the_whole_of_the_kernel_with_its_name_in_it() {
    let ptx = much_ptx::gpu::PTX;
    assert_eq!(ptx.len(), LEN);
    assert!(ptx.len() > 700_000, "a kernel of 32768 instructions is 787 KB, and the PTX is {} bytes", ptx.len());
    assert_eq!(ptx.matches("add.rn.f32").count(), 32768);
    // The name has the workspace's, the crate's and the module's, as C++ names a function in its namespaces.
    assert!(ptx.contains(".entry _ZN7project8much_ptx3gpu3bigE("), "no kernel `project::much_ptx::gpu::big`");
    assert!(!ptx.contains('\0'), "a hole is left in the PTX");
}
