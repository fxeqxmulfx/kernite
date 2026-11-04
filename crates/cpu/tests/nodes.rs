//! Where the memory of a grid's parts is. A page is on the NUMA node of the CPU of the thread that
//! first writes it, so a buffer is near the threads that use it when each writes its own part
//! first. The test asks Linux which node each page is on with `move_pages`, which the C library has
//! no function for, so it makes the call itself, in an `unsafe` block, and the grid's threads write
//! their parts by address, in another: these are the only `unsafe` here. kernite-cpu makes no such
//! call, for it places no memory: the pages are the system's, on the node of the thread that writes
//! them first.

#![cfg(all(target_os = "linux", target_arch = "x86_64"))]
#![allow(unsafe_code)]

mod common;

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{c_int, c_long};
use std::{fs, io, ptr};

use common::listed;
use kernite_cpu::{Grid, cpus, launch, most};

// The C library's way into the system's calls, for the one it has no function for.
unsafe extern "C" {
    fn syscall(number: c_long, ...) -> c_long;
}

/// The number of `move_pages` on x86-64.
const MOVE_PAGES: c_long = 279;

/// What `move_pages` says of a page that no thread has written: the error number, negated.
const NOT_THERE: c_int = -2;

/// The size of a page on x86-64.
const PAGE: usize = 4096;

/// The pages of a part of a buffer, the part a thread writes: 8 MB.
const PART: usize = 2048;

/// The pages at each end of a part that the test does not look at: where the system gives 512
/// pages at once, one may hold the end of a part and the start of the next.
const EDGE: usize = 512;

/// The size of a large buffer: above the 32 MB up to which the C library may serve a buffer from
/// memory it holds already, where a thread may have written the pages. The pages of a large one
/// are the system's own, and no thread has written them.
const LARGE: usize = 33 << 20;

/// The pages of a part that the test looks at.
const LOOKED_AT: usize = PART - 2 * EDGE;

/// How many elements a large buffer of at least `parts` parts has, with a page to spare for its
/// address to be rounded up to a page.
fn elements(parts: usize) -> usize {
    ((parts * PART + 1) * PAGE).max(LARGE) / size_of::<f32>()
}

/// The address of the first page of a buffer.
fn start(buffer: &[f32]) -> usize {
    buffer.as_ptr().addr().next_multiple_of(PAGE)
}

/// The addresses of the pages of part `t` of the buffer that begins at `start`, that the test looks
/// at: those not at the ends of the part.
fn looked_at(start: usize, t: usize) -> Vec<usize> {
    (EDGE..PART - EDGE).map(|page| start + (t * PART + page) * PAGE).collect()
}

/// For each of `pages`, the node it is on, or why it is on none: `NOT_THERE` where no thread has
/// written it. The system's error if it does not say.
fn nodes_of(pages: &[usize]) -> io::Result<Vec<c_int>> {
    let mut nodes = vec![0; pages.len()];
    // SAFETY: `syscall` is the C library's, and with no nodes to move to, `move_pages` only says
    // where the pages are. It reads `pages.len()` addresses from `pages` and writes as many
    // numbers to `nodes`, which have room for them, and it reads no page.
    let said = unsafe {
        let (count, to) = (pages.len() as c_long, ptr::null::<c_int>());
        syscall(MOVE_PAGES, 0 as c_long, count, pages.as_ptr(), to, nodes.as_mut_ptr(), 0 as c_long)
    };
    if said == 0 { Ok(nodes) } else { Err(io::Error::last_os_error()) }
}

/// Whether the system says where pages are, which a container may keep a program from asking.
fn says_where_pages_are() -> bool {
    let says = nodes_of(&[PAGE]);
    if let Err(e) = &says {
        eprintln!("the system does not say which node a page is on, for {e}: skipped");
    }
    says.is_ok()
}

/// The node a CPU is on, from the directory the system keeps for it: `cpu0/node0`.
fn node_of(cpu: usize) -> Option<usize> {
    let mut entries = fs::read_dir(format!("/sys/devices/system/cpu/cpu{cpu}")).ok()?;
    entries.find_map(|entry| entry.ok()?.file_name().to_str()?.strip_prefix("node")?.parse().ok())
}

/// The nodes the process may take memory from, if the system says.
fn mems() -> Option<Vec<usize>> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    let list = status.lines().find_map(|line| line.strip_prefix("Mems_allowed_list:"))?;
    Some(listed(list.trim()))
}

/// The CPUs a grid is placed on, each with its node: up to `per_node` of each node, the nodes in
/// turn, so that a grid cut short at `most` threads has some of every node it can.
fn placement(nodes: &BTreeMap<usize, Vec<usize>>, per_node: usize, most: usize) -> Vec<(usize, usize)> {
    let mut placed = vec![];
    for round in 0..per_node {
        for (&node, cpus) in nodes {
            placed.extend(cpus.get(round).map(|&cpu| (cpu, node)));
        }
    }
    placed.truncate(most);
    placed
}

/// A grid has a thread on each node in turn, as many on each as it has CPUs, up to the number
/// asked for, and no more than `most`.
#[test]
fn a_grid_is_placed_on_the_nodes_in_turn() {
    let two = BTreeMap::from([(0, vec![0, 1, 2, 3]), (1, vec![4, 5, 6, 7])]);
    assert_eq!(placement(&two, 2, 8), [(0, 0), (4, 1), (1, 0), (5, 1)]);
    assert_eq!(placement(&two, 2, 3), [(0, 0), (4, 1), (1, 0)]);
    assert_eq!(placement(&two, 1, 8), [(0, 0), (4, 1)]);
    let uneven = BTreeMap::from([(0, vec![0]), (2, vec![4, 5, 6]), (3, vec![8, 9])]);
    assert_eq!(placement(&uneven, 2, 9), [(0, 0), (4, 2), (8, 3), (5, 2), (9, 3)]);
    assert_eq!(placement(&BTreeMap::new(), 2, 9), []);
}

/// A large buffer made with `vec![0.0; n]` has no page until a thread writes one, and one made
/// with `vec![1.0; n]` has them all, since the caller wrote them. So a grid's threads place the
/// pages of the first, and the caller has placed those of the second.
#[test]
fn a_buffer_of_zeros_has_no_page_until_a_thread_writes_it_and_one_the_caller_fills_has_them_all() {
    if !says_where_pages_are() {
        return;
    }
    let (zeros, ones) = (vec![0.0f32; elements(1)], vec![1.0f32; elements(1)]);
    let (of_zeros, of_ones) =
        (nodes_of(&looked_at(start(&zeros), 0)).unwrap(), nodes_of(&looked_at(start(&ones), 0)).unwrap());
    let there = of_zeros.iter().filter(|&&node| node != NOT_THERE).count();
    assert_eq!(there, 0, "of {LOOKED_AT} pages of the zeros");
    let missing = of_ones.iter().filter(|&&node| node < 0).count();
    assert_eq!(missing, 0, "of {LOOKED_AT} pages of the ones");
}

/// A grid placed on the CPUs of the machine's nodes, each thread of which writes its part of a
/// buffer first: each page of a part is on the node of the thread's CPU. With one node, which is
/// what this machine has, every page is on it, and the test says it skipped what needs two.
#[test]
fn each_page_of_a_buffer_the_threads_of_a_placed_grid_write_first_is_on_the_node_of_its_thread() {
    if !says_where_pages_are() {
        return;
    }
    let Some(mems) = mems() else {
        return eprintln!("the system does not say which NUMA nodes the process may use: skipped");
    };
    let mut nodes: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for cpu in cpus().unwrap() {
        if let Some(node) = node_of(cpu).filter(|node| mems.contains(node)) {
            nodes.entry(node).or_default().push(cpu);
        }
    }
    let placed = placement(&nodes, 2, most().unwrap());
    if placed.is_empty() {
        return eprintln!("the system does not say which NUMA node a CPU is on: skipped");
    }
    let on: Vec<usize> = placed.iter().map(|&(cpu, _)| cpu).collect();

    let mut buffer = vec![0.0f32; elements(placed.len())];
    let first = start(&buffer);
    let pages: Vec<usize> = (0..placed.len()).flat_map(|t| looked_at(first, t)).collect();
    let before = nodes_of(&pages).unwrap();
    assert!(before.iter().all(|&node| node == NOT_THERE), "a page of the buffer is there before a thread writes it");

    // The grid's threads write to the buffer by address.
    let _ = buffer.as_mut_ptr().expose_provenance();
    launch(Grid::on(&on), move |place| {
        let part = ptr::with_exposed_provenance_mut::<u8>(first + place.thread() as usize * PART * PAGE);
        for page in 0..PART {
            // SAFETY: the page is in the buffer, which the test keeps until the launch returns, and
            // only this thread writes the part it is in.
            unsafe { part.add(page * PAGE).write(1) };
        }
    })
    .unwrap();

    let after = nodes_of(&pages).unwrap();
    for (t, &(cpu, node)) in placed.iter().enumerate() {
        let found: BTreeSet<c_int> = after[t * LOOKED_AT..][..LOOKED_AT].iter().copied().collect();
        let message = format!("the nodes of the pages thread {t} wrote, on CPU {cpu} of node {node}");
        assert_eq!(found, BTreeSet::from([node as c_int]), "{message}");
    }
    if nodes.len() < 2 {
        eprintln!(
            "this machine has one NUMA node, so no page could be on another than its thread's: skipped, for want of two NUMA nodes"
        );
    }
}
