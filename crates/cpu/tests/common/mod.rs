//! What the tests that ask Linux where a thread or a page is share.

use std::num::ParseIntError;

/// The numbers a list names, as Linux writes a list of CPUs or of NUMA nodes: `0-5,8`.
pub fn listed(list: &str) -> Result<Vec<usize>, ParseIntError> {
    let mut numbers = vec![];
    for run in list.split(',') {
        let (first, last) = run.split_once('-').unwrap_or((run, run));
        numbers.extend(first.parse::<usize>()?..=last.parse()?);
    }
    Ok(numbers)
}
