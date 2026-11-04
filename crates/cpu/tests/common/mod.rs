//! What the tests that ask Linux where a thread or a page is share.

/// The numbers a list names, as Linux writes a list of CPUs or of NUMA nodes: `0-5,8`.
pub fn listed(list: &str) -> Vec<usize> {
    let runs = list.split(',').map(|run| run.split_once('-').unwrap_or((run, run)));
    runs.flat_map(|(first, last)| first.parse::<usize>().unwrap()..=last.parse().unwrap()).collect()
}
