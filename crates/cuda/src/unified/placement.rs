//! Where managed memory is: moving it before it is used, advice about how it is used, which a slice
//! lent takes too, and what the driver has of both.

use std::ffi::{c_int, c_void};
use std::ops::Range;
use std::ptr;

use super::{CONCURRENT_MANAGED_ACCESS, Kind, Unified};
use crate::{Error, MemLocation, Plain};

/// The `CUmem_range_attribute`s that `placement` asks for.
const READ_MOSTLY: c_int = 1;
const ACCESSED_BY: c_int = 3;
const PREFERRED_LOCATION_KIND: c_int = 5;
const PREFERRED_LOCATION_ID: c_int = 6;
const LAST_PREFETCH_LOCATION_KIND: c_int = 7;
const LAST_PREFETCH_LOCATION_ID: c_int = 8;

/// How many devices `ACCESSED_BY` has room to name: the GPU and the host, with room to spare.
const ACCESSED_BY_ROOM: usize = 8;

/// `CU_DEVICE_CPU`, which `ACCESSED_BY` names the host by.
const HOST_DEVICE: c_int = -1;

/// A place for managed memory to be.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Location {
    /// The memory of the GPU that made it.
    Gpu,
    /// The host's memory, on whichever NUMA node.
    Host,
    /// The host's memory on NUMA node `n`. The driver refuses a node the host does not have, and
    /// any past what an `int` holds.
    Numa(u32),
    /// The host's memory on the NUMA node of the thread that asks.
    NumaOfThread,
}

impl Location {
    /// What the driver is told, for the GPU `device`.
    fn raw(self, device: c_int) -> MemLocation {
        match self {
            Location::Gpu => MemLocation { kind: 1, id: device },
            Location::Host => MemLocation { kind: 2, id: 0 },
            Location::Numa(node) => {
                // A node past what an `int` holds is past the host's, and the driver says so of the
                // greatest.
                MemLocation { kind: 3, id: c_int::try_from(node).unwrap_or(c_int::MAX) }
            }
            Location::NumaOfThread => MemLocation { kind: 4, id: 0 },
        }
    }

    /// The place the driver says, for the GPU `device`: `None` for no place, and for another GPU.
    fn from_raw(MemLocation { kind, id }: MemLocation, device: c_int) -> Option<Location> {
        match kind {
            1 if id == device => Some(Location::Gpu),
            2 => Some(Location::Host),
            3 => Some(Location::Numa(id as u32)),
            4 => Some(Location::NumaOfThread),
            _ => None,
        }
    }
}

/// What the host tells the driver of how managed memory is used, which the driver may follow or
/// not.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Advice {
    /// The memory is read far more than it is written: each processor that reads it may keep a
    /// copy.
    ReadMostly,
    /// The memory is not read-mostly: the copies are collapsed into one.
    NotReadMostly,
    /// The memory belongs at this place: when another processor touches it, the driver maps it
    /// there where it can, and does not move it. The driver here, NVIDIA's open kernel module
    /// 615.71.10 on Linux 7.0, takes a NUMA node as the place and changes nothing: the memory
    /// prefers what it did before, the GPU, the host or none. CUDA's header says the driver sets
    /// the node and says it back as a NUMA node, which this one does not.
    PreferredLocation(Location),
    /// The memory belongs at no place.
    NoPreferredLocation,
    /// The GPU or the host touches the memory, so the driver keeps it mapped for them where it
    /// can, and they do not fault on it. A NUMA node does not touch memory: it is refused.
    AccessedBy(Location),
    /// The GPU or the host need not have the memory kept mapped for them.
    NotAccessedBy(Location),
}

impl Advice {
    /// What the driver is told, for the GPU `device`: the number of the advice, and its place,
    /// which the advice that has none leaves as the host's.
    fn raw(self, device: c_int) -> Result<(c_int, MemLocation), Error> {
        let none = Location::Host.raw(device);
        Ok(match self {
            Advice::ReadMostly => (1, none),
            Advice::NotReadMostly => (2, none),
            Advice::PreferredLocation(place) => (3, place.raw(device)),
            Advice::NoPreferredLocation => (4, none),
            Advice::AccessedBy(place) => (5, accessor(place, device)?),
            Advice::NotAccessedBy(place) => (6, accessor(place, device)?),
        })
    }
}

/// `place`, as the driver is told of what touches memory: the GPU or the host, or the error that
/// says a NUMA node does not.
fn accessor(place: Location, device: c_int) -> Result<MemLocation, Error> {
    match place {
        Location::Gpu | Location::Host => Ok(place.raw(device)),
        Location::Numa(_) | Location::NumaOfThread => {
            Err(Error::new("cuMemAdvise", "only the GPU and the host access memory, not a NUMA node".into()))
        }
    }
}

/// What the driver has of a range of managed memory: the advice in force for the whole of it, and
/// where it was last prefetched to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Placement {
    /// Whether every page of the range is [`ReadMostly`](Advice::ReadMostly).
    pub read_mostly: bool,
    /// The one place every page of the range prefers; `None` where the pages do not agree, or
    /// prefer none. The driver here, NVIDIA's open kernel module 615.71.10 on Linux 7.0, never
    /// says a NUMA node, as it sets none: see [`PreferredLocation`](Advice::PreferredLocation).
    pub preferred: Option<Location>,
    /// The GPU and the host, whichever of them every page of the range is
    /// [`AccessedBy`](Advice::AccessedBy).
    pub accessed_by: Vec<Location>,
    /// The one place the whole range was last prefetched to; `None` where the pages were not
    /// prefetched to one.
    pub last_prefetch: Option<Location>,
}

impl<T: Plain> Unified<'_, T> {
    /// `Ok` if the driver does the `call` to this memory: it does so to managed memory, and to a
    /// slice lent if `lent`. If not, the error that says what the driver does, as `rule`, and that
    /// this memory is not it.
    fn require_kind(&self, call: &'static str, rule: &str, lent: bool) -> Result<(), Error> {
        let this = match self.kind {
            Kind::Managed => return Ok(()),
            Kind::Lent if lent => return Ok(()),
            Kind::Pinned | Kind::Registered => "page-locked memory",
            Kind::Lent => "a slice lent",
        };
        Err(Error::new(call, format!("the driver {rule}: this is {this}")))
    }

    /// `Ok` if this GPU reaches managed memory while the host does; or the error that says the
    /// `call` is refused, as it does not.
    fn require_concurrent(&self, call: &'static str) -> Result<(), Error> {
        self.driver.require(call, CONCURRENT_MANAGED_ACCESS, "the GPU cannot reach managed memory while the host does")
    }

    /// The address and the bytes of the elements `range`; or the error that says the `call` is refused, as `range` has
    /// no elements, or goes past the end.
    fn span(&self, call: &'static str, range: &Range<usize>) -> Result<(u64, usize), Error> {
        let Range { start, end } = *range;
        let said = if start >= end {
            format!("the range {start}..{end} has no elements")
        } else if end > self.len {
            format!("the range {start}..{end} goes past the end of the {} elements", self.len)
        } else {
            return Ok((self.addr() + (start * size_of::<T>()) as u64, (end - start) * size_of::<T>()));
        };
        Err(Error::new(call, said))
    }

    /// Moves the elements `range` to `to`, ahead of what will touch them, on the stream the
    /// kernels are launched on. It moves whole pages, as [`Unified`] says. The GPU must reach
    /// managed memory while the host does; the error says so of one that does not:
    /// `cuMemPrefetchAsync: the GPU cannot reach managed memory while the host does: its
    /// CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS is 0`. The driver prefetches managed memory
    /// alone, so page-locked memory and a slice lent are refused: `cuMemPrefetchAsync: the driver
    /// prefetches managed memory alone: this is page-locked memory`, or `this is a slice lent`.
    /// The driver refuses a NUMA node the host does not have: `cuMemPrefetchAsync:
    /// CUDA_ERROR_INVALID_VALUE (1): invalid argument`.
    /// It needs a driver of CUDA 12.2 or later; an older one is the error of the lookup of its
    /// function: ``dlsym: the driver has no `cuMemPrefetchAsync_v2`: libcuda.so.1: undefined
    /// symbol: cuMemPrefetchAsync_v2``.
    /// A range with no elements, or one past the end, is refused before the driver is asked: `cuMemPrefetchAsync: the
    /// range 2..2 has no elements`, or `the range 3..5 goes past the end of the 4 elements`.
    ///
    /// ```no_run
    /// # use kernite_cuda::{Cuda, Location};
    /// let cuda = Cuda::new()?;
    /// let mut xs = cuda.managed::<f32>(1 << 20)?;
    /// xs.host_mut()?.fill(1.0);
    /// // The kernel that reads `xs` finds it in the GPU's memory, not faulting it there.
    /// xs.prefetch(0..xs.len(), Location::Gpu)?;
    /// // A kernel reads it here, launched on `xs.ptr()`.
    /// # Ok::<(), kernite_cuda::Error>(())
    /// ```
    pub fn prefetch(&self, range: Range<usize>, to: Location) -> Result<(), Error> {
        self.require_kind("cuMemPrefetchAsync", "prefetches managed memory alone", false)?;
        let (addr, count) = self.span("cuMemPrefetchAsync", &range)?;
        self.require_concurrent("cuMemPrefetchAsync")?;
        let (api, to) = (self.driver.current()?, to.raw(self.driver.device));
        let prefetch = api.prefetch.clone()?;
        // SAFETY: it reads no memory, and moves the managed memory of `count` bytes at `addr`. 0 is
        // its flags, of which there are none yet, and the stream is the one the kernels are launched on.
        api.check("cuMemPrefetchAsync", unsafe { prefetch(addr, count, to, 0, self.driver.stream()) })
    }

    /// Tells the driver how the elements `range` are used, in whole pages, as [`Unified`] says.
    /// The GPU must reach managed memory while the host does for the advice that names it, a
    /// preferred location or an accessor; the error is the one of [`prefetch`](Unified::prefetch),
    /// with `cuMemAdvise` for the call it refuses.
    /// A NUMA node the host does not have is refused by the driver, as it is for a prefetch. A NUMA
    /// node is no accessor: `cuMemAdvise: only the GPU and the host access memory, not a NUMA
    /// node`. The driver advises managed memory, and pageable memory where the GPU reaches it,
    /// which `lend` asked: so a slice lent is advised as managed memory is, and page-locked memory
    /// is refused: `cuMemAdvise: the driver advises managed and pageable memory alone: this is
    /// page-locked memory`.
    /// It needs a driver of CUDA 12.2 or later; an older one is the error of the lookup of its
    /// function: ``dlsym: the driver has no `cuMemAdvise_v2`: libcuda.so.1: undefined symbol:
    /// cuMemAdvise_v2``.
    /// A range with no elements, or one past the end, is refused as it is for a prefetch.
    pub fn advise(&self, range: Range<usize>, advice: Advice) -> Result<(), Error> {
        self.require_kind("cuMemAdvise", "advises managed and pageable memory alone", true)?;
        let (addr, count) = self.span("cuMemAdvise", &range)?;
        let (number, place) = advice.raw(self.driver.device)?;
        if matches!(advice, Advice::PreferredLocation(Location::Gpu) | Advice::AccessedBy(Location::Gpu)) {
            self.require_concurrent("cuMemAdvise")?;
        }
        let api = self.driver.current()?;
        let advise = api.advise.clone()?;
        // SAFETY: it reads no memory, and advises about the managed memory of `count` bytes at
        // `addr`.
        api.check("cuMemAdvise", unsafe { advise(addr, count, number, place) })
    }

    /// What the driver has of the elements `range`. The whole range is asked, so a field says
    /// something only where every page of it does. The driver tells the place of managed memory
    /// alone, so page-locked memory and a slice lent are refused: `cuMemRangeGetAttribute: the
    /// driver tells the place of managed memory alone: this is a slice lent`.
    /// The driver here, NVIDIA's open kernel module 615.71.10 on Linux 7.0, does not always clear
    /// the advice and the last prefetch of a buffer it frees, when threads use it at once, so a
    /// buffer made at that address may start with them.
    /// It needs a driver of CUDA 12.2 or later, which brought the attributes of the preferred
    /// place and of the last prefetch with `cuMemAdvise_v2`; an older one is the error of the
    /// lookup of that function, as for [`advise`](Unified::advise). A range with no elements, or one past the end, is
    /// refused as it is for a prefetch.
    pub fn placement(&self, range: Range<usize>) -> Result<Placement, Error> {
        self.require_kind("cuMemRangeGetAttribute", "tells the place of managed memory alone", false)?;
        let (addr, count) = self.span("cuMemRangeGetAttribute", &range)?;
        let (api, device) = (self.driver.current()?, self.driver.device);
        // A driver that has `cuMemAdvise_v2` has the attributes of CUDA 12.2.
        api.advise.clone()?;
        let ask = |attribute: c_int, data: *mut c_void, size: usize| {
            // SAFETY: it writes the `size` bytes of the attribute to `data`, which has room for
            // them, and reads nothing of the managed memory of `count` bytes at `addr`.
            api.check("cuMemRangeGetAttribute", unsafe { (api.range_attribute)(data, size, attribute, addr, count) })
        };
        let int = |attribute| {
            let mut value: c_int = 0;
            ask(attribute, ptr::from_mut(&mut value).cast(), size_of::<c_int>()).map(|()| value)
        };
        let place = |kind, id| Ok(Location::from_raw(MemLocation { kind: int(kind)?, id: int(id)? }, device));
        let mut accessors = [0; ACCESSED_BY_ROOM];
        ask(ACCESSED_BY, accessors.as_mut_ptr().cast(), size_of_val(&accessors))?;
        Ok(Placement {
            read_mostly: int(READ_MOSTLY)? == 1,
            preferred: place(PREFERRED_LOCATION_KIND, PREFERRED_LOCATION_ID)?,
            accessed_by: accessors
                .into_iter()
                .filter_map(|id| match id {
                    HOST_DEVICE => Some(Location::Host),
                    _ if id == device => Some(Location::Gpu),
                    _ => None,
                })
                .collect(),
            last_prefetch: place(LAST_PREFETCH_LOCATION_KIND, LAST_PREFETCH_LOCATION_ID)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::ffi::c_uint;
    use std::marker::PhantomData;

    use super::super::tests::{gpu, gpu_with};
    use super::*;
    use crate::tests::{create_stream, destroy_stream};
    use crate::{CuResult, Cuda, Handle};

    type Result<T = (), E = Box<dyn std::error::Error + Send + Sync>> = std::result::Result<T, E>;

    thread_local! {
        /// The advice that the fake driver took last, as it was told: the address, the bytes and
        /// the advice's number.
        static ADVISED: Cell<Option<(u64, usize, c_int)>> = const { Cell::new(None) };
    }

    /// The fake driver's `cuMemAdvise_v2`: it takes the advice.
    unsafe extern "C" fn advise(addr: u64, bytes: usize, advice: c_int, _: MemLocation) -> CuResult {
        ADVISED.set(Some((addr, bytes, advice)));
        0
    }

    /// Four elements of the `kind` of memory, at an address the fake driver never reads.
    fn memory(cuda: &Cuda, kind: Kind) -> Unified<'static, u32> {
        Unified { driver: cuda.driver.clone(), ptr: ptr::without_provenance_mut(4096), len: 4, kind, of: PhantomData }
    }

    /// A GPU that cannot reach managed memory while the host does is refused a prefetch, and the
    /// advice that names it, of managed memory and of a slice lent, in GUIDE.md's words, naming
    /// the call each refuses, before the driver is asked; and one that can is asked.
    #[test]
    fn a_gpu_that_cannot_reach_managed_memory_with_the_host_is_refused_a_prefetch_and_advice_naming_it() -> Result {
        let said = |call| {
            format!(
                "{call}: the GPU cannot reach managed memory while the host does: \
                 its CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS is 0"
            )
        };
        let cuda = gpu(&[])?;
        let m = memory(&cuda, Kind::Managed);
        assert_eq!(
            m.prefetch(0..4, Location::Host).err().ok_or("prefetch gave Ok")?.to_string(),
            said("cuMemPrefetchAsync")
        );
        for advice in [Advice::PreferredLocation(Location::Gpu), Advice::AccessedBy(Location::Gpu)] {
            assert_eq!(
                m.advise(0..4, advice).err().ok_or("advise gave Ok")?.to_string(),
                said("cuMemAdvise"),
                "{advice:?}"
            );
            let lent = memory(&cuda, Kind::Lent);
            assert_eq!(
                lent.advise(0..4, advice).err().ok_or("advise gave Ok")?.to_string(),
                said("cuMemAdvise"),
                "{advice:?}"
            );
        }
        // The other advice does not name the GPU, so the driver is asked, and refuses.
        let others = [
            Advice::ReadMostly,
            Advice::NotReadMostly,
            Advice::PreferredLocation(Location::Host),
            Advice::PreferredLocation(Location::Numa(0)),
            Advice::NoPreferredLocation,
            Advice::AccessedBy(Location::Host),
            Advice::NotAccessedBy(Location::Gpu),
        ];
        for advice in others {
            assert_eq!(m.advise(0..4, advice).err().ok_or("advise gave Ok")?.call, "cuMemAdvise", "{advice:?}");
        }
        let m = memory(&gpu(&[CONCURRENT_MANAGED_ACCESS])?, Kind::Managed);
        assert_eq!(m.prefetch(0..4, Location::Gpu).err().ok_or("prefetch gave Ok")?.call, "cuMemPrefetchAsync");
        assert_eq!(
            m.advise(0..4, Advice::AccessedBy(Location::Gpu)).err().ok_or("advise gave Ok")?.call,
            "cuMemAdvise"
        );
        Ok(())
    }

    /// Memory that is page-locked, made or registered, or a slice lent is not moved or placed: each call says so, in
    /// GUIDE.md's words, before the driver is asked.
    #[test]
    fn only_managed_memory_is_moved_and_placed() -> Result {
        let cuda = gpu(&[CONCURRENT_MANAGED_ACCESS])?;
        let kinds = [(Kind::Pinned, "page-locked memory"), (Kind::Registered, "page-locked memory")];
        for (kind, this) in kinds.into_iter().chain([(Kind::Lent, "a slice lent")]) {
            let m = memory(&cuda, kind);
            let said = |call, rule| format!("{call}: the driver {rule}: this is {this}");
            let prefetch = said("cuMemPrefetchAsync", "prefetches managed memory alone");
            assert_eq!(m.prefetch(0..4, Location::Gpu).err().ok_or("prefetch gave Ok")?.to_string(), prefetch);
            let placement = said("cuMemRangeGetAttribute", "tells the place of managed memory alone");
            assert_eq!(m.placement(0..4).err().ok_or("placement gave Ok")?.to_string(), placement);
        }
        Ok(())
    }

    /// A slice lent is advised as managed memory is: the driver is told the advice, for the
    /// elements' addresses. Page-locked memory, made or registered, is refused it, in GUIDE.md's
    /// words, and the driver is not asked.
    #[test]
    fn a_slice_lent_is_advised_as_managed_memory_is_and_page_locked_memory_is_not() -> Result {
        let cuda = gpu_with(&[CONCURRENT_MANAGED_ACCESS], |api| api.advise = Ok(advise))?;
        for kind in [Kind::Managed, Kind::Lent] {
            ADVISED.set(None);
            memory(&cuda, kind).advise(1..3, Advice::ReadMostly)?;
            assert_eq!(ADVISED.get(), Some((4096 + 4, 8, 1)));
        }
        let said = "cuMemAdvise: the driver advises managed and pageable memory alone: this is page-locked memory";
        for kind in [Kind::Pinned, Kind::Registered] {
            ADVISED.set(None);
            let e = memory(&cuda, kind).advise(1..3, Advice::ReadMostly).err().ok_or("advise gave Ok")?;
            assert_eq!(e.to_string(), said);
            assert_eq!(ADVISED.get(), None, "the driver was not asked");
        }
        Ok(())
    }

    /// A NUMA node does not touch memory, so it is refused as an accessor, in GUIDE.md's words,
    /// before the driver is asked.
    #[test]
    fn a_numa_node_is_no_accessor() -> Result {
        let m = memory(&gpu(&[CONCURRENT_MANAGED_ACCESS])?, Kind::Managed);
        let said = "cuMemAdvise: only the GPU and the host access memory, not a NUMA node";
        for place in [Location::Numa(0), Location::NumaOfThread] {
            for advice in [Advice::AccessedBy(place), Advice::NotAccessedBy(place)] {
                assert_eq!(m.advise(0..4, advice).err().ok_or("advise gave Ok")?.to_string(), said, "{advice:?}");
            }
        }
        Ok(())
    }

    /// A range of no elements, or past the end, is a mistake of the caller: each call says so, and
    /// the driver is not asked.
    #[test]
    fn a_range_of_no_elements_or_past_the_end_is_an_error() -> Result {
        let m = memory(&gpu(&[CONCURRENT_MANAGED_ACCESS])?, Kind::Managed);
        let none = "has no elements";
        let past = "goes past the end of the 4 elements";
        for (range, wrong) in [(0..0, none), (2..2, none), (3..5, past), (4..5, past), (5..5, none)] {
            let said = |call| format!("{call}: the range {}..{} {wrong}", range.start, range.end);
            let prefetch = m.prefetch(range.clone(), Location::Gpu).err().ok_or("prefetch gave Ok")?;
            assert_eq!(prefetch.to_string(), said("cuMemPrefetchAsync"));
            let advise = m.advise(range.clone(), Advice::ReadMostly).err().ok_or("advise gave Ok")?;
            assert_eq!(advise.to_string(), said("cuMemAdvise"));
            let placement = m.placement(range.clone()).err().ok_or("placement gave Ok")?;
            assert_eq!(placement.to_string(), said("cuMemRangeGetAttribute"));
        }
        Ok(())
    }

    /// A driver before CUDA 12.2 lacks `cuMemPrefetchAsync_v2` and `cuMemAdvise_v2`: a prefetch,
    /// advice and a placement are the error of the lookup of the function the call needs, and
    /// memory the call is not for is refused as before.
    #[test]
    fn a_driver_before_cuda_12_2_is_the_error_of_its_lookup_for_a_prefetch_advice_and_a_placement() -> Result {
        let cuda = gpu_with(&[CONCURRENT_MANAGED_ACCESS], |api| {
            api.prefetch = Err(crate::missing(c"cuMemPrefetchAsync_v2", "undefined symbol"));
            api.advise = Err(crate::missing(c"cuMemAdvise_v2", "undefined symbol"));
        })?;
        let lacks = |name| format!("dlsym: the driver has no `{name}`: undefined symbol");
        let m = memory(&cuda, Kind::Managed);
        assert_eq!(
            m.prefetch(0..4, Location::Gpu).err().ok_or("prefetch gave Ok")?.to_string(),
            lacks("cuMemPrefetchAsync_v2")
        );
        assert_eq!(
            m.advise(0..4, Advice::ReadMostly).err().ok_or("advise gave Ok")?.to_string(),
            lacks("cuMemAdvise_v2")
        );
        assert_eq!(m.placement(0..4).err().ok_or("placement gave Ok")?.to_string(), lacks("cuMemAdvise_v2"));
        // Memory that the calls are not for is refused as before.
        let pinned = memory(&cuda, Kind::Pinned);
        assert!(
            pinned
                .prefetch(0..4, Location::Gpu)
                .err()
                .ok_or("prefetch gave Ok")?
                .to_string()
                .contains("this is page-locked memory")
        );
        Ok(())
    }

    thread_local! {
        /// The stream that the fake driver's prefetch was given last.
        static PREFETCHED_ON: Cell<Option<usize>> = const { Cell::new(None) };
    }

    /// The fake driver's `cuMemPrefetchAsync_v2`: it takes the stream.
    unsafe extern "C" fn prefetch(_: u64, _: usize, _: MemLocation, _: c_uint, stream: Handle) -> CuResult {
        PREFETCHED_ON.set(Some(stream.addr()));
        0
    }

    /// A prefetch goes to the current stream, as a launch does: the null stream, or the one a `run` made current, so
    /// that the memory is moved before the kernels launched after it on that stream need it.
    #[test]
    fn a_prefetch_goes_to_the_current_stream() -> Result {
        let cuda = gpu_with(&[CONCURRENT_MANAGED_ACCESS], |api| {
            api.prefetch = Ok(prefetch);
            api.create_stream = create_stream;
            api.destroy_stream = destroy_stream;
        })?;
        let (m, stream) = (memory(&cuda, Kind::Managed), cuda.stream()?);
        m.prefetch(0..4, Location::Gpu)?;
        assert_eq!(PREFETCHED_ON.take(), Some(0));
        // SAFETY: the fake driver runs no work, so no stream's work reaches memory.
        unsafe { stream.run(|| m.prefetch(0..4, Location::Gpu)) }?;
        assert_eq!(PREFETCHED_ON.take(), Some(0x100));
        m.prefetch(0..4, Location::Host)?;
        assert_eq!(PREFETCHED_ON.take(), Some(0));
        Ok(())
    }

    /// A place is told to the driver as the driver has it, for the GPU the memory is on, and a
    /// place it says is read back as the place.
    #[test]
    fn a_place_is_told_to_the_driver_and_read_back_as_the_place() {
        let places = [
            (Location::Gpu, MemLocation { kind: 1, id: 1 }),
            (Location::Host, MemLocation { kind: 2, id: 0 }),
            (Location::Numa(3), MemLocation { kind: 3, id: 3 }),
            (Location::NumaOfThread, MemLocation { kind: 4, id: 0 }),
        ];
        for (place, told) in places {
            assert_eq!(place.raw(1), told, "{place:?}");
            assert_eq!(Location::from_raw(told, 1), Some(place), "{place:?}");
        }
        // A node past what an `int` holds is told as the greatest, not as a negative one.
        let greatest = MemLocation { kind: 3, id: c_int::MAX };
        for node in [c_int::MAX as u32, c_int::MAX as u32 + 1, u32::MAX] {
            assert_eq!(Location::Numa(node).raw(1), greatest, "{node}");
        }
        // Another GPU, no place, and a place that is not one of these are no place of these.
        for told in [MemLocation { kind: 1, id: 0 }, MemLocation { kind: 0, id: -2 }, MemLocation { kind: 5, id: 0 }] {
            assert_eq!(Location::from_raw(told, 1), None, "{told:?}");
        }
    }
}
