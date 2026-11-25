//! A crate that names the PTX of `many_names`, as the crate of a program that launches the kernel does. Rust makes the
//! constant again here, and counts its steps again: it says nothing of it, and each use of the static has its name.

/// The bytes of the PTX, known now: Rust makes the constant, and counts its steps, in this crate.
pub const LEN: usize = many_names::gpu::PTX.len();

#[test]
fn each_use_of_the_static_has_its_name_in_the_ptx_whole() {
    let ptx = many_names::gpu::PTX;
    assert_eq!(ptx.len(), LEN);
    assert!(LEN > 5_000_000, "the module is {LEN} bytes, as `many_names` says: 5 MB");
    // The name has the workspace's, the crate's and the module's, then the static's own.
    let name = "_ZN7project10many_names3gpu6COUNTSE";
    assert_eq!(ptx.matches(&format!(", {name};")).count(), 100_000, "the uses of the static in the PTX");
    // Each name is a whole one: none is cut short where the part is cut, and no hole is left.
    assert_eq!(ptx.matches("COUNTS").count(), ptx.matches(name).count(), "a name of the static is not whole");
    assert!(!ptx.contains('\0'), "a hole is left in the PTX");
}
