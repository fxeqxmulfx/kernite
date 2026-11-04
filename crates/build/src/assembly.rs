//! A device's module, said in parts. Only the program knows which of the module's items it keeps, as
//! only Rust knows which `cfg` holds, so a machine says its module as parts, each there where a
//! condition on the items holds. Put together for any set of items that is kept, the parts are the
//! module the machine makes of those items alone.

/// Something of a module that the program keeps or leaves out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kept {
    /// The code at this index among those the machine was given: a kernel or a function.
    Code(usize),
    /// The static at this index among those the machine was given.
    Static(usize),
}

/// A condition on what is kept.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum When {
    /// This is kept.
    Kept(Kept),
    /// One of these holds; never, if there are none.
    Any(Vec<When>),
    /// Each of these holds; always, if there are none.
    All(Vec<When>),
    /// This does not hold.
    Not(Box<When>),
}

impl When {
    /// A condition that holds whatever is kept.
    pub fn always() -> When {
        When::All(vec![])
    }

    /// Whether the condition holds, `kept` saying what is kept.
    pub fn holds(&self, kept: &dyn Fn(Kept) -> bool) -> bool {
        match self {
            When::Kept(it) => kept(*it),
            When::Any(parts) => parts.iter().any(|part| part.holds(kept)),
            When::All(parts) => parts.iter().all(|part| part.holds(kept)),
            When::Not(part) => !part.holds(kept),
        }
    }
}

/// A device's module, said in parts: its text, part by part, in order, each there where its condition
/// holds, and why it is none where a refusal's does. The default has no part and no refusal.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Assembly {
    /// The text of the module, part by part.
    pub parts: Vec<(When, String)>,
    /// Why the module is none, each where its condition holds. The first that holds is said.
    pub refusals: Vec<(When, String)>,
}

impl Assembly {
    /// The module of what is kept, `kept` saying what is; or why it is none.
    pub fn text(&self, kept: &dyn Fn(Kept) -> bool) -> Result<String, String> {
        if let Some((_, why)) = self.refusals.iter().find(|(when, _)| when.holds(kept)) {
            return Err(why.clone());
        }
        Ok(self.parts.iter().filter(|(when, _)| when.holds(kept)).map(|(_, text)| text.as_str()).collect())
    }

    /// The module of everything; or why it is none.
    pub fn whole(&self) -> Result<String, String> {
        self.text(&|_| true)
    }
}
