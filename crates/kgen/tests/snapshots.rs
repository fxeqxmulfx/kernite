//! Every kernel's listing against the one in `snapshots/`. With `KGEN_BLESS=1` the files
//! are written instead, for the diff to be reviewed and committed.

use std::fs;
use std::path::Path;

use kgen::{Kernel, render};

#[test]
fn listings_match_their_snapshots() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../snapshots");
    let bless = std::env::var_os("KGEN_BLESS").is_some();
    if bless {
        fs::create_dir_all(&dir).unwrap();
    }
    let mut changed = vec![];
    let mut names = vec![];
    for kernel in Kernel::snapshots() {
        let listing = render(&kernel).unwrap_or_else(|e| panic!("{kernel:?}: {e}"));
        let file = format!("{}.s", listing.name);
        let text = listing.text();
        let path = dir.join(&file);
        if bless {
            fs::write(&path, &text).unwrap();
        } else if fs::read_to_string(&path).ok().as_deref() != Some(text.as_str()) {
            changed.push(file.clone());
        }
        names.push(file);
    }
    let mut extra: Vec<String> = fs::read_dir(&dir)
        .map(|entries| entries.map(|e| e.unwrap().file_name().into_string().unwrap()).collect())
        .unwrap_or_default();
    extra.retain(|file| !names.contains(file));
    assert!(extra.is_empty(), "snapshots of no kernel: {extra:?}");
    assert!(changed.is_empty(), "listings changed: {changed:?}; rerun with KGEN_BLESS=1 and review the diff");
}

#[test]
fn rendering_is_a_function_of_the_kernel() {
    for kernel in Kernel::snapshots() {
        assert_eq!(render(&kernel).unwrap(), render(&kernel).unwrap());
    }
}
