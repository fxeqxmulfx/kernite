//! What kernels need of PTX and of the GPU, put together. `module` finds why no module has a set of
//! kernels among the sets of at most four, and says it where all of a set are kept, so it holds
//! that kernels added to a set that no module has never make a module of it.

use std::collections::BTreeSet;

use kernite_ptx::forms::forms;
use kernite_ptx::{GPUS, Needs};

/// A set of kernels that no module has is one that no kernels added to it make a module of, for
/// every need the forms of PTX have, and every GPU kernite writes for: the adding is `and`.
#[test]
fn kernels_added_to_a_set_that_no_module_has_never_make_a_module_of_it() {
    let (mut ptxs, mut marks) = (BTreeSet::from([0]), BTreeSet::from([(0, 0)]));
    for form in forms() {
        let alts = form.elems.iter().flat_map(|elem| elem.alts.iter().map(|alt| alt.needs));
        for needs in std::iter::once(form.needs).chain(alts) {
            ptxs.insert(needs.ptx);
            marks.insert((needs.below, needs.since));
        }
    }
    let mut all = vec![];
    for &ptx in &ptxs {
        for sm in std::iter::once(0).chain(GPUS.iter().map(|&(g, _)| g)) {
            all.extend(marks.iter().map(|&(below, since)| Needs { ptx, sm, below, since }));
        }
    }
    let none = all.iter().filter(|n| !n.possible()).count();
    assert!(none > 0, "some needs have no module");
    for &set in all.iter().filter(|n| !n.possible()) {
        for &more in &all {
            assert!(!set.and(more).possible(), "{more:?} added to {set:?}, which no module has");
        }
    }
}
