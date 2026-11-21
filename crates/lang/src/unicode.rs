//! What Rust reads as a name, for which `std` has no tables: the characters a name is made of, which Unicode calls
//! XID_Start and XID_Continue, and the one spelling that Rust gives names that are one, NFC. The tables are those of
//! Unicode 17.0.0, which Rust 1.100 has; `unicode.py` writes them, and a test holds them to the rustc on the machine.

mod tables;
#[cfg(test)]
mod tests;

use std::ops::RangeInclusive;
use tables::{COMBINING, COMPOSITION, DECOMPOSITION, XID_CONTINUE, XID_START};

/// Whether `c` is in one of `ranges`, which are in order.
fn within(ranges: &[RangeInclusive<char>], c: char) -> bool {
    ranges.get(ranges.partition_point(|run| *run.end() < c)).is_some_and(|run| run.contains(&c))
}

/// Whether a name can start with `c`, but for `_`: XID_Start.
pub(crate) fn is_xid_start(c: char) -> bool {
    if c.is_ascii() { c.is_ascii_alphabetic() } else { within(&XID_START, c) }
}

/// Whether a name can go on with `c`: XID_Continue, which has `_`.
pub(crate) fn is_xid_continue(c: char) -> bool {
    if c.is_ascii() { c.is_ascii_alphanumeric() || c == '_' } else { within(&XID_CONTINUE, c) }
}

/// The combining class of `c`, which orders the marks of a character: 0 for a character that is no mark.
fn class(c: char) -> u8 {
    COMBINING.binary_search_by_key(&c, |&(mark, _)| mark).ok().and_then(|at| COMBINING.get(at)).map_or(0, |m| m.1)
}

// The Hangul syllables are not in the tables: each is a leading consonant, a vowel and perhaps a trailing consonant,
// numbered from U+AC00 in that order. A trailing consonant is numbered from U+11A7, 0 for none, so there are 28.
const SYLLABLE: u32 = 0xAC00;
const LEADING: u32 = 0x1100;
const VOWEL: u32 = 0x1161;
const TRAILING: u32 = 0x11A7;
const LEADINGS: u32 = 19;
const VOWELS: u32 = 21;
const TRAILINGS: u32 = 28;

/// The character that `a` then `b` make, if Unicode says they make one.
fn compose(a: char, b: char) -> Option<char> {
    let (first, second) = (u32::from(a), u32::from(b));
    let syllable = first.wrapping_sub(SYLLABLE);
    if (LEADING..LEADING + LEADINGS).contains(&first) && (VOWEL..VOWEL + VOWELS).contains(&second) {
        char::from_u32(SYLLABLE + ((first - LEADING) * VOWELS + second - VOWEL) * TRAILINGS)
    } else if syllable < LEADINGS * VOWELS * TRAILINGS
        && syllable % TRAILINGS == 0
        && (TRAILING + 1..TRAILING + TRAILINGS).contains(&second)
    {
        char::from_u32(first + second - TRAILING)
    } else {
        COMPOSITION
            .binary_search_by_key(&(a, b), |&(pair, _)| pair)
            .ok()
            .and_then(|at| COMPOSITION.get(at))
            .map(|c| c.1)
    }
}

/// `name` in NFC, as Rust reads a name: one spelling for the names that are one, `é` as U+00E9 and not as `e` and
/// U+0301. A character that is in no name is left as it is, for the tables hold those of names alone.
pub fn nfc(name: &str) -> String {
    // Each character that Unicode says is other characters is those, all the way down. A Hangul syllable is left whole:
    // taking it apart and putting it together again would give the same.
    let mut chars = Vec::new();
    for c in name.chars() {
        match DECOMPOSITION.binary_search_by_key(&c, |&(whole, _)| whole).ok().and_then(|at| DECOMPOSITION.get(at)) {
            Some((_, parts)) => chars.extend(parts.chars()),
            None => chars.push(c),
        }
    }
    // The marks after a character go in the order of their classes, those of one class as they came.
    for run in chars.chunk_by_mut(|&a, &b| (class(a) == 0) == (class(b) == 0)) {
        run.sort_by_key(|&c| class(c));
    }
    // A mark joins the last character that is no mark, if Unicode says the two make one and no character between them
    // has the class of the mark or a greater one; two characters that are no marks, one after the other, may do so too.
    let mut out: Vec<char> = Vec::new();
    let (mut starter, mut last) = (None, 0);
    for c in chars {
        let class = class(c);
        if let Some(at) = starter
            && (out.len() == at + 1 || last < class)
            && let Some(slot) = out.get_mut(at)
            && let Some(whole) = compose(*slot, c)
        {
            *slot = whole;
            continue;
        }
        if class == 0 {
            starter = Some(out.len());
        }
        last = class;
        out.push(c);
    }
    out.into_iter().collect()
}
