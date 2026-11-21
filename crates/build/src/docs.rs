//! What a machine writes the docs of its instructions from.

use std::collections::BTreeMap;

/// The texts of `table` by name: a line for each set of names, then ` : `, then the text, which goes on in the lines
/// after it that start with a space. A machine writes the docs of its instructions from such tables: the sentence of
/// each, say. A table that is wrong gives its first wrong line, and why: a line with no ` : `, a line that goes on
/// from nothing, or a name that two lines give.
pub fn texts(table: &'static str) -> Result<BTreeMap<&'static str, String>, String> {
    let mut entries: Vec<(&'static str, String)> = vec![];
    for line in table.lines().filter(|line| !line.trim().is_empty()) {
        match (line.starts_with(' '), entries.last_mut()) {
            (true, Some((_, text))) => {
                text.push(' ');
                text.push_str(line.trim());
            }
            (true, None) => return Err(format!("`{}` goes on from no text", line.trim())),
            (false, _) => {
                let (names, text) = line.split_once(" : ").ok_or_else(|| format!("`{line}` has no ` : `"))?;
                entries.push((names, text.to_string()));
            }
        }
    }
    let mut out = BTreeMap::new();
    for (names, text) in entries {
        for name in names.split_whitespace() {
            if out.insert(name, text.clone()).is_some() {
                return Err(format!("`{name}` has two texts"));
            }
        }
    }
    Ok(out)
}
