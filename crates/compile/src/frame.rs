use kgen_core::{PReg, Slot};

use crate::{Frame, Machine};

/// The frame a kernel needs: the callee-saved registers among `used`, in the machine's
/// order, and the slots one after another, each at its alignment.
pub(crate) fn layout(slots: &[Slot], used: &[PReg], m: &dyn Machine) -> Frame {
    let saved = m.callee_saved().iter().filter(|r| used.contains(r)).copied().collect();
    let (mut offsets, mut end, mut align) = (Vec::with_capacity(slots.len()), 0, 0);
    for slot in slots {
        let at = u32::next_multiple_of(end, slot.align);
        offsets.push(at);
        end = at + slot.size;
        align = align.max(slot.align);
    }
    let size = if align == 0 { 0 } else { end.next_multiple_of(align) };
    Frame { saved, slots: slots.to_vec(), offsets, size, align }
}
