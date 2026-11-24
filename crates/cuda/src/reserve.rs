//! A buffer whose addresses are reserved for more elements than it has: it backs more of them with memory, or fewer, at
//! the same addresses.

use std::ffi::{c_int, c_void};
use std::marker::PhantomData;
use std::ptr;

use super::unified::Flag;
use super::{Buffer, Cuda, Driver, Error, MemLocation, Plain, bytes, too_many};

/// The GPU maps memory at addresses reserved before it: the driver's virtual memory management.
const VIRTUAL_MEMORY_MANAGEMENT: Flag = (102, "VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED");

/// What the driver is told memory it creates is, as `CUmemAllocationProp` is.
#[repr(C)]
pub(crate) struct Properties {
    /// 1 is CU_MEM_ALLOCATION_TYPE_PINNED, the one type there is.
    kind: c_int,
    /// No handle of the memory is shared with another process.
    handles: c_int,
    /// Where the memory is.
    location: MemLocation,
    /// What Windows asks of a shared handle: none.
    win32: *mut c_void,
    /// The hints of compressible memory and of GPUDirect, and the usage: none.
    flags: [u8; 8],
}

impl Properties {
    /// Memory on the GPU `device`.
    fn on(device: c_int) -> Properties {
        Properties { kind: 1, handles: 0, location: MemLocation::gpu(device), win32: ptr::null_mut(), flags: [0; 8] }
    }
}

/// Who may use memory that is mapped, and how, as `CUmemAccessDesc` is.
#[repr(C)]
pub(crate) struct Access {
    /// The GPU that may.
    location: MemLocation,
    /// 3 is CU_MEM_ACCESS_FLAGS_PROT_READWRITE.
    flags: c_int,
}

/// What a buffer that [`Cuda::reserve`] made holds beside its memory.
#[derive(Clone, Copy)]
pub(crate) struct Reservation {
    /// How many elements the addresses are reserved for: as many as the buffer grows to.
    capacity: usize,
    /// How many bytes of addresses are reserved: whole granules.
    span: usize,
    /// How many bytes the driver backs with memory at a time.
    granule: usize,
}

impl Reservation {
    /// How many granules hold `len` elements of `T`.
    fn granules<T>(&self, len: usize) -> usize {
        (len * size_of::<T>()).div_ceil(self.granule)
    }
}

impl Driver {
    /// Backs the granules `*mapped..to` of the addresses at `ptr` with memory, each as a mapping of its own, so that
    /// each can be given back alone, and lets the GPU read and write them. `*mapped` is how many granules from the
    /// start are mapped, and is the granules mapped when it returns, whether or not it is `to`. It asks the driver
    /// for nothing where the granules are mapped already.
    fn back(&self, ptr: u64, granule: usize, mapped: &mut usize, to: usize) -> Result<(), Error> {
        let (api, from) = (self.current()?, *mapped);
        if from >= to {
            return Ok(());
        }
        while *mapped < to {
            let (mut handle, properties) = (0, Properties::on(self.device));
            // SAFETY: it writes the handle to `handle`.
            api.check("cuMemCreate", unsafe { (api.create_memory)(&mut handle, granule, &properties, 0) })?;
            let at = ptr + (*mapped * granule) as u64;
            // SAFETY: `at` is a granule of the addresses that were reserved, with nothing mapped at it. The memory is
            // mapped whole, from its start: 0 is the offset in it, and 0 the flags.
            let mapping = api.check("cuMemMap", unsafe { (api.map_memory)(at, granule, 0, handle, 0) });
            // The mapping holds the memory until it is unmapped, so the handle is let go of at once.
            // SAFETY: `cuMemCreate` gave the handle, and it is released once, here.
            unsafe { (api.release_memory)(handle) };
            mapping?;
            *mapped += 1;
        }
        let access = Access { location: MemLocation::gpu(self.device), flags: 3 };
        let at = ptr + (from * granule) as u64;
        // SAFETY: the range is the granules mapped above, whole, and `access` is the one description it is given.
        let allowed = unsafe { (api.set_access)(at, (to - from) * granule, &access, 1) };
        api.check("cuMemSetAccess", allowed)
    }

    /// Unmaps the granules `to..*mapped` of the addresses at `ptr`, the last first, so that the granules mapped are
    /// always those from the start. Each was mapped alone, whole. `*mapped` is how many granules from the start are
    /// mapped, and is the granules mapped when it returns, whether or not it is `to`.
    ///
    /// # Safety
    ///
    /// No work that is queued or running reaches the granules: the driver unmaps them at once.
    unsafe fn unmap(&self, ptr: u64, granule: usize, mapped: &mut usize, to: usize) -> Result<(), Error> {
        let api = self.current()?;
        while *mapped > to {
            let at = ptr + ((*mapped - 1) * granule) as u64;
            // SAFETY: the granule is one whole mapping, and the caller says no work reaches it.
            api.check("cuMemUnmap", unsafe { (api.unmap_memory)(at, granule) })?;
            *mapped -= 1;
        }
        Ok(())
    }
}

impl Cuda {
    /// A buffer with no elements, and addresses reserved for `capacity` of them, which stay its own until it is
    /// dropped. [`grow`](Buffer::grow) backs the elements it gains with memory at those addresses, and
    /// [`shrink`](Buffer::shrink) gives the memory of the last of them back to the GPU, so its address, `ptr()`, is
    /// the same through each growth and each shrink, and the address of an element is good for as long as the buffer
    /// has the element. Kernels, copies and `read` take it as they take any buffer, and a kernel that is no `unsafe
    /// fn` is lent it as it is lent one that [`alloc`](Cuda::alloc) made. The reserved addresses take no memory of the
    /// GPU: the driver backs the buffer's elements with it, and no more.
    ///
    /// The GPU must map memory at addresses it reserves; the error says so of one that does not: `cuMemAddressReserve:
    /// the GPU cannot map memory at addresses it reserves: its CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED
    /// is 0`.
    ///
    /// Dropping the buffer waits for the kernels that use it, as dropping any buffer does, unmaps its memory and frees
    /// its addresses.
    ///
    /// ```no_run
    /// # let cuda = kernite_cuda::Cuda::new()?;
    /// let mut cache = cuda.reserve::<f32>(1 << 28)?;
    /// let at = cache.ptr().addr();
    /// cache.grow(1000)?;
    /// cache.write(&[1.0; 1000])?;
    /// cache.grow(10_000_000)?;
    /// assert_eq!(cache.ptr().addr(), at);
    /// assert_eq!(cache.read()?[999..1001], [1.0, 0.0]);
    /// # Ok::<(), kernite_cuda::Error>(())
    /// ```
    pub fn reserve<T: Plain>(&self, capacity: usize) -> Result<Buffer<T>, Error> {
        let cannot = "the GPU cannot map memory at addresses it reserves";
        self.driver.require("cuMemAddressReserve", VIRTUAL_MEMORY_MANAGEMENT, cannot)?;
        let bytes = bytes::<T>("cuMemAddressReserve", capacity)?;
        let (api, properties, mut granule) = (self.driver.current()?, Properties::on(self.driver.device), 0);
        // SAFETY: it writes the granularity to `granule`. 0 is CU_MEM_ALLOC_GRANULARITY_MINIMUM: the least memory that
        // the driver backs at a time.
        api.check("cuMemGetAllocationGranularity", unsafe { (api.granularity)(&mut granule, &properties, 0) })?;
        // The driver reserves whole granules, and refuses no bytes, so a granule is the least.
        let span = bytes.max(1).checked_next_multiple_of(granule);
        let span = span.ok_or_else(|| too_many::<T>("cuMemAddressReserve", capacity))?;
        let mut ptr = 0;
        // SAFETY: it writes the address to `ptr`. The first 0 asks for the driver's alignment, the second for any
        // address, and the third is the flags, of which there are none.
        api.check("cuMemAddressReserve", unsafe { (api.reserve_addresses)(&mut ptr, span, 0, 0, 0) })?;
        let reservation = Some(Reservation { capacity, span, granule });
        Ok(Buffer { driver: self.driver.clone(), ptr, len: 0, reservation, of: PhantomData })
    }
}

impl<T: Plain> Buffer<T> {
    /// Backs the elements it does not have yet, up to `len`, with memory at its addresses, so that it has `len`
    /// elements: the elements it had are as they were, where they were, and the elements it gains are zero, as those of
    /// [`alloc`](Cuda::alloc) are, written on the current stream of its GPU, after the work that stream was given
    /// before. Its address, `ptr()`, does not change. The buffer is one that [`reserve`](Cuda::reserve) made: one that
    /// `alloc` or `upload` made has its memory fixed.
    ///
    /// The driver backs memory a granule at a time, 2 MiB here, and the GPU's free memory falls by whole granules: a
    /// buffer of 1 element holds a granule, and grows into the rest of it with no more memory.
    ///
    /// The call does not wait for the zeros, so a [`shrink`](Buffer::shrink) that gives back elements this growth
    /// gained comes after them: [`Cuda::sync`] waits for them.
    ///
    /// An error says what is refused, and the buffer is as it was. A buffer that `alloc` or `upload` made:
    /// `cuMemMap: a buffer of 4 elements was not made by reserve, so it does not grow`. A `len` past what it was
    /// reserved for: `cuMemMap: a buffer reserved for 268435456 elements grown to 268435457`. A `len` below the
    /// elements it has: `cuMemMap: a buffer of 1000 elements grown to 10`. And a GPU that has not the memory for the
    /// growth: the driver's own error, of `cuMemCreate`.
    pub fn grow(&mut self, len: usize) -> Result<(), Error> {
        let reservation = self.reserved("cuMemMap", "grow")?;
        if len < self.len {
            return Err(Error::new("cuMemMap", format!("a buffer of {} elements grown to {len}", self.len)));
        }
        if len > reservation.capacity {
            let message = format!("a buffer reserved for {} elements grown to {len}", reservation.capacity);
            return Err(Error::new("cuMemMap", message));
        }
        let (mut mapped, to) = (reservation.granules::<T>(self.len), reservation.granules::<T>(len));
        let from = mapped;
        let (zeros_at, zeros) = (self.ptr + (self.len * size_of::<T>()) as u64, (len - self.len) * size_of::<T>());
        let grown = self
            .driver
            .back(self.ptr, reservation.granule, &mut mapped, to)
            .and_then(|()| self.driver.zero(zeros_at, zeros));
        if let Err(e) = grown {
            // SAFETY: no work reaches the granules that were mapped only now.
            let _ = unsafe { self.driver.unmap(self.ptr, reservation.granule, &mut mapped, from) };
            return Err(e);
        }
        self.len = len;
        Ok(())
    }

    /// Gives the memory of the elements from `len` on back to the GPU, so that the buffer has `len` elements: the
    /// elements before them are as they were, where they were, and its address, `ptr()`, does not change. The buffer is
    /// one that [`reserve`](Cuda::reserve) made, as for [`grow`](Buffer::grow), and the error of one that `alloc` or
    /// `upload` made is `cuMemUnmap: a buffer of 4 elements was not made by reserve, so it does not shrink`. A `len`
    /// past the elements it has is an error, `cuMemUnmap: a buffer of 10 elements shrunk to 1000`, and the buffer is as
    /// it was.
    ///
    /// The driver takes back whole granules, 2 MiB here: the memory of a granule that has an element left stays. The
    /// elements from `len` on are zero again where the buffer grows into them.
    ///
    /// It does not wait for the kernels that use the buffer, so a buffer shrinks while the GPU works on the elements
    /// it keeps.
    ///
    /// ```no_run
    /// # let cuda = kernite_cuda::Cuda::new()?;
    /// # let mut cache = cuda.reserve::<f32>(1 << 28)?;
    /// # cache.grow(10_000_000)?;
    /// cuda.sync()?;
    /// // SAFETY: the sync waited for the zeros the growth writes as well, so no work is queued or running, and none
    /// // reaches the elements from 1000 on.
    /// unsafe { cache.shrink(1000)? };
    /// # Ok::<(), kernite_cuda::Error>(())
    /// ```
    ///
    /// # Safety
    ///
    /// No work that is queued or running on the GPU, on any stream, reaches an element from `len` on: the driver unmaps
    /// them at once, and does not wait for work that reaches them, which fails with `CUDA_ERROR_ILLEGAL_ADDRESS (700)`,
    /// as does each call after it. No pointer to them, from `ptr()` or `at()`, is used again.
    ///
    /// The zeros that [`grow`](Buffer::grow) writes to the elements it gains, and the work of [`zero`](Buffer::zero),
    /// [`Cuda::copy`] and [`copy_from`](Buffer::copy_from), are such work until the GPU has done it: each returns
    /// before then, and [`Cuda::sync`] waits for them.
    pub unsafe fn shrink(&mut self, len: usize) -> Result<(), Error> {
        let reservation = self.reserved("cuMemUnmap", "shrink")?;
        if len > self.len {
            return Err(Error::new("cuMemUnmap", format!("a buffer of {} elements shrunk to {len}", self.len)));
        }
        let mut mapped = reservation.granules::<T>(self.len);
        // SAFETY: the caller's: no work reaches an element from `len` on, which are the granules from `to` on.
        let shrunk =
            unsafe { self.driver.unmap(self.ptr, reservation.granule, &mut mapped, reservation.granules::<T>(len)) };
        // The buffer has what is mapped: `len` elements, or where the driver refused to unmap a granule, the elements
        // of the granules it still has.
        self.len = if shrunk.is_ok() { len } else { self.len.min(mapped * reservation.granule / size_of::<T>()) };
        shrunk
    }

    /// What it holds beside its memory; or the error that says the `call` is refused, as a buffer that `alloc` or
    /// `upload` made does not `does`.
    fn reserved(&self, call: &'static str, does: &str) -> Result<Reservation, Error> {
        self.reservation.ok_or_else(|| {
            Error::new(
                call,
                format!("a buffer of {} elements was not made by reserve, so it does not {does}", self.len),
            )
        })
    }

    /// Gives the GPU back what the buffer holds, as it is dropped: the kernels that may reach it are waited for, its
    /// memory is unmapped, and its addresses are freed.
    pub(super) fn release(&self, reservation: Reservation) {
        // The driver does not wait for the kernels that reach what it unmaps.
        // SAFETY: it takes nothing.
        self.driver.give_back(|api| unsafe { (api.synchronize)() });
        let mut mapped = reservation.granules::<T>(self.len);
        // SAFETY: every kernel that reached the memory has run, and none is queued after: the buffer is dropped.
        let _ = unsafe { self.driver.unmap(self.ptr, reservation.granule, &mut mapped, 0) };
        // SAFETY: the addresses are `span` bytes at `ptr`, as `cuMemAddressReserve` gave them. They are freed once,
        // here.
        self.driver.give_back(|api| unsafe { (api.free_addresses)(self.ptr, reservation.span) });
    }
}

#[cfg(test)]
mod tests {
    use std::cell::{Cell, RefCell};

    use std::ffi::c_int;

    use super::super::unified::tests::gpu_with;
    use super::super::{CuResult, Handle};
    use super::*;
    use crate::tests::{create_stream, destroy_stream};

    type Result<T = (), E = Box<dyn std::error::Error + Send + Sync>> = std::result::Result<T, E>;

    /// What the fake driver backs at a time, 2 MiB, as the driver does on the GPU here.
    const GRANULE: usize = 2 << 20;

    /// The elements of a granule, for a buffer of `u32`s.
    const WORDS: usize = GRANULE / 4;

    /// Where the fake driver reserves addresses. Nothing is there, and the calls say addresses as the bytes from it.
    const BASE: u64 = 0x7000_0000_0000;

    /// The fake GPU's attribute that says it maps memory at addresses it reserves, by the number `cuda.h` gives it.
    const MAPS: Flag = (102, "VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED");

    thread_local! {
        /// The calls the fake driver took, one line each, in order: the function's name, and what a test looks at of
        /// its arguments.
        static CALLS: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
        /// The call that the fake driver refuses, with CUDA_ERROR_OUT_OF_MEMORY: the function, and how many of its
        /// calls the driver takes before it.
        static REFUSED: Cell<Option<(&'static str, usize)>> = const { Cell::new(None) };
        /// How many handles of memory the fake driver has made: the handles are 1, 2 and on.
        static MADE: Cell<u64> = const { Cell::new(0) };
        /// Whether the fake driver has a capture on: begun, and not ended.
        static CAPTURING: Cell<bool> = const { Cell::new(false) };
    }

    /// The fake driver takes the call `line`, which starts with the function's name; and answers.
    fn took(line: String) -> CuResult {
        let name = line.split(' ').next().unwrap_or_default().to_string();
        let before = CALLS.with_borrow(|calls| calls.iter().filter(|c| c.split(' ').next() == Some(&name)).count());
        CALLS.with_borrow_mut(|calls| calls.push(line));
        // 2 is CUDA_ERROR_OUT_OF_MEMORY.
        if REFUSED.get().is_some_and(|(refused, nth)| refused == name && nth == before) { 2 } else { 0 }
    }

    /// An address as the bytes from `BASE`.
    fn from_base(at: u64) -> u64 {
        at.wrapping_sub(BASE)
    }

    /// The fake driver's `cuMemGetAllocationGranularity`.
    unsafe extern "C" fn granularity(out: *mut usize, memory: *const Properties, option: c_int) -> CuResult {
        // SAFETY: the caller gives a place for the granularity, and the properties of the memory.
        let (kind, location) = unsafe {
            out.write(GRANULE);
            ((*memory).kind, (*memory).location)
        };
        took(format!("cuMemGetAllocationGranularity {kind} {} {} {option}", location.kind, location.id))
    }

    /// The fake driver's `cuMemAddressReserve`.
    unsafe extern "C" fn reserve_addresses(out: *mut u64, bytes: usize, align: usize, at: u64, flags: u64) -> CuResult {
        // SAFETY: the caller gives a place for the address.
        unsafe { out.write(BASE) };
        took(format!("cuMemAddressReserve {bytes} {align} {at} {flags}"))
    }

    /// The fake driver's `cuMemAddressFree`.
    unsafe extern "C" fn free_addresses(at: u64, bytes: usize) -> CuResult {
        took(format!("cuMemAddressFree {} {bytes}", from_base(at)))
    }

    /// The fake driver's `cuMemCreate`.
    unsafe extern "C" fn create_memory(out: *mut u64, bytes: usize, memory: *const Properties, flags: u64) -> CuResult {
        MADE.set(MADE.get() + 1);
        // SAFETY: the caller gives a place for the handle, and the properties of the memory.
        let (kind, handles, location) = unsafe {
            out.write(MADE.get());
            ((*memory).kind, (*memory).handles, (*memory).location)
        };
        took(format!("cuMemCreate {bytes} {kind} {handles} {} {} {flags}", location.kind, location.id))
    }

    /// The fake driver's `cuMemRelease`.
    unsafe extern "C" fn release_memory(handle: u64) -> CuResult {
        took(format!("cuMemRelease {handle}"))
    }

    /// The fake driver's `cuMemMap`.
    unsafe extern "C" fn map_memory(at: u64, bytes: usize, offset: usize, handle: u64, flags: u64) -> CuResult {
        took(format!("cuMemMap {} {bytes} {offset} {handle} {flags}", from_base(at)))
    }

    /// The fake driver's `cuMemSetAccess`.
    unsafe extern "C" fn set_access(at: u64, bytes: usize, access: *const Access, count: usize) -> CuResult {
        // SAFETY: the caller gives a description of the access.
        let (location, flags) = unsafe { ((*access).location, (*access).flags) };
        took(format!("cuMemSetAccess {} {bytes} {} {} {flags} {count}", from_base(at), location.kind, location.id))
    }

    /// The fake driver's `cuMemUnmap`.
    unsafe extern "C" fn unmap_memory(at: u64, bytes: usize) -> CuResult {
        took(format!("cuMemUnmap {} {bytes}", from_base(at)))
    }

    /// The fake driver's `cuMemsetD8Async`: it says the stream, when it is not the null stream.
    unsafe extern "C" fn memset(at: u64, value: u8, bytes: usize, stream: Handle) -> CuResult {
        let on = if stream.is_null() { String::new() } else { format!(" on {:#x}", stream.addr()) };
        took(format!("cuMemsetD8Async {} {value} {bytes}{on}", from_base(at)))
    }

    /// The fake driver's `cuCtxSynchronize`: a capture refuses it, with `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`, as it
    /// does the driver.
    unsafe extern "C" fn synchronize() -> CuResult {
        let said = took("cuCtxSynchronize".to_string());
        if CAPTURING.get() { 900 } else { said }
    }

    /// The fake driver's `cuStreamBeginCapture`.
    unsafe extern "C" fn begin_capture(_: Handle, _: c_int) -> CuResult {
        CAPTURING.set(true);
        took("cuStreamBeginCapture".to_string())
    }

    /// The fake driver's `cuStreamEndCapture`: it makes no graph.
    unsafe extern "C" fn end_capture(_: Handle, graph: *mut Handle) -> CuResult {
        CAPTURING.set(false);
        // SAFETY: the caller gives a place for the graph.
        unsafe { graph.write(ptr::null_mut()) };
        took("cuStreamEndCapture".to_string())
    }

    /// The fake driver's `cuMemAlloc`, for a buffer that `reserve` did not make.
    unsafe extern "C" fn alloc(out: *mut u64, bytes: usize) -> CuResult {
        // SAFETY: the caller gives a place for the address.
        unsafe { out.write(BASE) };
        took(format!("cuMemAlloc {bytes}"))
    }

    /// The fake driver's `cuMemFree`.
    unsafe extern "C" fn free(at: u64) -> CuResult {
        took(format!("cuMemFree {}", from_base(at)))
    }

    /// A GPU that maps memory at addresses it reserves, and a driver that takes the calls above and refuses every
    /// other.
    fn fake() -> Result<Cuda> {
        CALLS.take();
        REFUSED.take();
        MADE.take();
        CAPTURING.take();
        gpu_with(&[MAPS], |api| {
            api.granularity = granularity;
            api.reserve_addresses = reserve_addresses;
            api.free_addresses = free_addresses;
            api.create_memory = create_memory;
            api.release_memory = release_memory;
            api.map_memory = map_memory;
            api.set_access = set_access;
            api.unmap_memory = unmap_memory;
            api.memset = memset;
            api.synchronize = synchronize;
            api.begin_capture = begin_capture;
            api.end_capture = end_capture;
            api.alloc = alloc;
            api.free = free;
            api.create_stream = create_stream;
            api.destroy_stream = destroy_stream;
        })
    }

    /// The calls the fake driver took since this was last called.
    fn calls() -> Vec<String> {
        CALLS.take()
    }

    /// The calls that make a granule of memory with `handle` and map it `at` bytes from the start of the addresses.
    fn backing(at: usize, handle: u64) -> [String; 3] {
        [
            format!("cuMemCreate {GRANULE} 1 0 1 0 0"),
            format!("cuMemMap {at} {GRANULE} 0 {handle} 0"),
            format!("cuMemRelease {handle}"),
        ]
    }

    /// The call that lets the GPU read and write `granules` granules from `at` bytes from the start.
    fn access(at: usize, granules: usize) -> String {
        format!("cuMemSetAccess {at} {} 1 0 3 1", granules * GRANULE)
    }

    /// The call that unmaps the granule `n`.
    fn unmap(n: usize) -> String {
        format!("cuMemUnmap {} {GRANULE}", n * GRANULE)
    }

    /// A buffer of `u32`s made by `cuda` for `capacity` elements that has `len` of them, and the calls so far
    /// forgotten.
    fn holding(cuda: &Cuda, capacity: usize, len: usize) -> Result<Buffer<u32>> {
        let mut buffer = cuda.reserve(capacity)?;
        buffer.grow(len)?;
        calls();
        Ok(buffer)
    }

    /// What `shrink` says of `len`, where the fake driver runs no work for it to disturb.
    fn shrink(buffer: &mut Buffer<u32>, len: usize) -> Result<(), Error> {
        // SAFETY: the fake driver runs no work.
        unsafe { buffer.shrink(len) }
    }

    #[test]
    fn a_buffer_is_reserved_only_for_a_gpu_that_maps_memory_at_addresses_it_reserves() -> Result {
        let cuda = gpu_with(&[], |_| {})?;
        let e = cuda.reserve::<u32>(10).err().ok_or("reserve gave Ok")?;
        assert_eq!(
            e.to_string(),
            "cuMemAddressReserve: the GPU cannot map memory at addresses it reserves: its \
             CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED is 0"
        );
        Ok(())
    }

    #[test]
    fn addresses_are_reserved_for_whole_granules_and_the_driver_is_asked_for_nothing_else() -> Result {
        let cuda = fake()?;
        // `(capacity, bytes of addresses)`: a granule is the least, and `capacity` elements of `u32` need whole ones.
        for (capacity, span) in [(0, GRANULE), (1, GRANULE), (WORDS, GRANULE), (WORDS + 1, 2 * GRANULE)] {
            let buffer = cuda.reserve::<u32>(capacity)?;
            // The memory asked about is the GPU's: 1 is the pinned type, then the kind of place that is a GPU, and
            // which.
            let asked =
                ["cuMemGetAllocationGranularity 1 1 0 0".to_string(), format!("cuMemAddressReserve {span} 0 0 0")];
            assert_eq!(calls(), asked, "capacity {capacity}");
            assert_eq!((buffer.len(), buffer.ptr().addr()), (0, BASE), "capacity {capacity}");
            drop(buffer);
            assert_eq!(calls(), ["cuCtxSynchronize".to_string(), format!("cuMemAddressFree 0 {span}")]);
        }
        // Bytes, not elements, are rounded up to a granule.
        let buffer = cuda.reserve::<u8>(GRANULE + 1)?;
        assert_eq!(calls()[1], format!("cuMemAddressReserve {} 0 0 0", 2 * GRANULE));
        drop(buffer);
        Ok(())
    }

    #[test]
    fn more_elements_than_there_are_addresses_are_refused_with_the_driver_asked_for_a_granule_at_most() -> Result {
        let cuda = fake()?;
        let e = cuda.reserve::<u32>(usize::MAX).err().ok_or("reserve gave Ok")?;
        let words = "elements of 4 bytes are more bytes than there are addresses";
        assert_eq!(e.to_string(), format!("cuMemAddressReserve: {} {words}", usize::MAX));
        assert!(calls().is_empty(), "the driver was not asked");
        // The bytes fit the addresses, and the granules that hold them do not.
        let e = cuda.reserve::<u8>(usize::MAX).err().ok_or("reserve gave Ok")?;
        let words = "elements of 1 bytes are more bytes than there are addresses";
        assert_eq!(e.to_string(), format!("cuMemAddressReserve: {} {words}", usize::MAX));
        assert_eq!(calls(), ["cuMemGetAllocationGranularity 1 1 0 0"], "the driver was asked for its granule alone");
        Ok(())
    }

    #[test]
    fn a_buffer_backs_memory_a_granule_at_a_time_and_gives_it_back_a_granule_at_a_time() -> Result {
        let cuda = fake()?;
        let mut buffer = cuda.reserve::<u32>(2 * WORDS)?;
        calls();
        // A first element needs a granule, made and mapped alone, and the handle is let go of at once, as the mapping
        // holds the memory. The GPU may use it, and the zeros of the element are written.
        buffer.grow(1)?;
        let mut first = backing(0, 1).to_vec();
        first.extend([access(0, 1), "cuMemsetD8Async 0 0 4".to_string()]);
        assert_eq!((calls(), buffer.len()), (first, 1));
        // The rest of the granule is memory it has: it grows into it with the zeros alone.
        buffer.grow(WORDS)?;
        assert_eq!((calls(), buffer.len()), ([format!("cuMemsetD8Async 4 0 {}", GRANULE - 4)].to_vec(), WORDS));
        // An element past it needs the next granule, and no other.
        buffer.grow(WORDS + 1)?;
        let mut second = backing(GRANULE, 2).to_vec();
        second.extend([access(GRANULE, 1), format!("cuMemsetD8Async {GRANULE} 0 4")]);
        assert_eq!((calls(), buffer.len()), (second, WORDS + 1));
        // The last granule is given back alone, as nothing is left in it; and no granule that has an element is.
        shrink(&mut buffer, WORDS)?;
        assert_eq!((calls(), buffer.len()), ([unmap(1)].to_vec(), WORDS));
        shrink(&mut buffer, 1)?;
        assert_eq!((calls(), buffer.len()), (Vec::new(), 1));
        shrink(&mut buffer, 0)?;
        assert_eq!((calls(), buffer.len()), ([unmap(0)].to_vec(), 0));
        Ok(())
    }

    /// The zeros that a growth writes go to the current stream of the GPU, after the work that stream was given before:
    /// the null stream where no other is current, and the stream of a `run` where one is.
    #[test]
    fn the_zeros_that_a_growth_writes_go_to_the_current_stream() -> Result {
        let cuda = fake()?;
        let (mut buffer, stream) = (cuda.reserve::<u32>(2 * WORDS)?, cuda.stream()?);
        calls();
        // SAFETY: the fake driver runs no work.
        unsafe { stream.run(|| buffer.grow(1)) }?;
        let zeros = calls().into_iter().filter(|call| call.starts_with("cuMemsetD8Async")).collect::<Vec<_>>();
        assert_eq!(zeros, ["cuMemsetD8Async 0 0 4 on 0x100"], "the zeros of a growth in a run");
        buffer.grow(2)?;
        assert_eq!(calls(), ["cuMemsetD8Async 4 0 4"], "the zeros of a growth where no stream is current");
        Ok(())
    }

    #[test]
    fn a_growth_over_granules_makes_them_in_turn_and_lets_the_gpu_use_them_all_at_once() -> Result {
        let cuda = fake()?;
        let mut buffer = holding(&cuda, 4 * WORDS, WORDS)?;
        // Granule 0 is there. Granules 1 and 2 are made and mapped, and one call lets the GPU use the two.
        buffer.grow(2 * WORDS + 3)?;
        let mut calls_made = backing(GRANULE, 2).to_vec();
        calls_made.extend(backing(2 * GRANULE, 3));
        calls_made.extend([access(GRANULE, 2), format!("cuMemsetD8Async {GRANULE} 0 {}", GRANULE + 12)]);
        assert_eq!(calls(), calls_made);
        // And the buffer gives them back the last first, as those that remain must be the first.
        shrink(&mut buffer, 1)?;
        assert_eq!(calls(), [unmap(2), unmap(1)]);
        Ok(())
    }

    /// A growth by two granules that the driver refuses at the call `call`, the `nth` of its kind: what the driver is
    /// told after, and what the buffer has, which is the one element it had before.
    fn refused_growth(call: &'static str, nth: usize) -> Result<(Vec<String>, usize)> {
        let cuda = fake()?;
        let mut buffer = holding(&cuda, 4 * WORDS, 1)?;
        REFUSED.set(Some((call, nth)));
        let e = buffer.grow(2 * WORDS + 1).err().ok_or("grow gave Ok")?;
        // The error names the call the driver refused, and gives the code the driver did: 2. The fake driver has no
        // words for its codes, as the real one has.
        assert_eq!(e.to_string(), format!("{call}: ? (2): ?"));
        let (told, len) = (calls(), buffer.len());
        REFUSED.take();
        // Nothing is left behind that stops a growth that the driver takes.
        buffer.grow(2 * WORDS + 1)?;
        assert_eq!(buffer.len(), 2 * WORDS + 1);
        Ok((told, len))
    }

    #[test]
    fn a_growth_the_driver_refuses_is_undone_and_leaves_the_buffer_as_it_was() -> Result {
        let (g1, g2) = (backing(GRANULE, 2), backing(2 * GRANULE, 3));
        let both = [g1.clone(), g2.clone()].concat();
        // Memory the driver does not make for the second granule: the first is unmapped, and not before.
        let expected = [g1.to_vec(), vec![format!("cuMemCreate {GRANULE} 1 0 1 0 0"), unmap(1)]].concat();
        assert_eq!(refused_growth("cuMemCreate", 1)?, (expected, 1));
        // A mapping the driver refuses: the handle of memory that was made is let go of all the same, so it is not
        // lost.
        let expected = [
            g1.to_vec(),
            vec![format!("cuMemCreate {GRANULE} 1 0 1 0 0"), format!("cuMemMap {} {GRANULE} 0 3 0", 2 * GRANULE)],
            vec!["cuMemRelease 3".to_string(), unmap(1)],
        ]
        .concat();
        assert_eq!(refused_growth("cuMemMap", 1)?, (expected, 1));
        // Access the driver refuses: every granule mapped for the growth is unmapped, the last first.
        let expected = [both.to_vec(), vec![access(GRANULE, 2), unmap(2), unmap(1)]].concat();
        assert_eq!(refused_growth("cuMemSetAccess", 0)?, (expected, 1));
        // Zeros the driver refuses to write: the same.
        let zeros = format!("cuMemsetD8Async 4 0 {}", 2 * GRANULE);
        let expected = [both.to_vec(), vec![access(GRANULE, 2), zeros, unmap(2), unmap(1)]].concat();
        assert_eq!(refused_growth("cuMemsetD8Async", 0)?, (expected, 1));
        Ok(())
    }

    #[test]
    fn a_shrink_the_driver_refuses_leaves_the_buffer_the_elements_of_the_granules_it_has() -> Result {
        let cuda = fake()?;
        // The buffer has three granules and 5 elements of the third; the driver unmaps the last granule first.
        let len = 2 * WORDS + 5;
        let mut buffer = holding(&cuda, 4 * WORDS, len)?;
        for (nth, held) in [(0, len), (1, 2 * WORDS)] {
            REFUSED.set(Some(("cuMemUnmap", nth)));
            let e = shrink(&mut buffer, 1).err().ok_or("shrink gave Ok")?;
            assert_eq!(e.to_string(), "cuMemUnmap: ? (2): ?");
            assert_eq!(buffer.len(), held, "the driver refused call {nth}");
            assert_eq!(calls().len(), nth + 1);
            REFUSED.take();
        }
        // What the buffer has, it still gives back.
        shrink(&mut buffer, 1)?;
        assert_eq!((calls(), buffer.len()), ([unmap(1)].to_vec(), 1));
        Ok(())
    }

    #[test]
    fn a_growth_past_the_reservation_or_below_the_buffer_and_a_shrink_past_it_are_refused_with_no_call() -> Result {
        let cuda = fake()?;
        let mut buffer = holding(&cuda, 1000, 10)?;
        let said = |result: Result<(), Error>| result.err().map(|e| e.to_string());
        assert_eq!(said(buffer.grow(1001)), Some("cuMemMap: a buffer reserved for 1000 elements grown to 1001".into()));
        assert_eq!(said(buffer.grow(9)), Some("cuMemMap: a buffer of 10 elements grown to 9".into()));
        assert_eq!(said(shrink(&mut buffer, 11)), Some("cuMemUnmap: a buffer of 10 elements shrunk to 11".into()));
        assert_eq!((calls(), buffer.len()), (Vec::new(), 10));
        // The elements it has, and those it was reserved for, are no more than it takes.
        assert_eq!((said(shrink(&mut buffer, 10)), said(buffer.grow(10))), (None, None));
        assert_eq!((said(buffer.grow(1000)), buffer.len()), (None, 1000));
        Ok(())
    }

    #[test]
    fn a_buffer_that_alloc_made_neither_grows_nor_shrinks_and_is_freed_as_a_buffer_is() -> Result {
        let cuda = fake()?;
        let mut buffer = cuda.alloc::<u32>(4)?;
        calls();
        let said = |result: Result<(), Error>| result.err().map(|e| e.to_string());
        let words = "a buffer of 4 elements was not made by reserve, so it does not";
        assert_eq!(said(buffer.grow(8)), Some(format!("cuMemMap: {words} grow")));
        assert_eq!(said(shrink(&mut buffer, 2)), Some(format!("cuMemUnmap: {words} shrink")));
        assert_eq!((calls(), buffer.len()), (Vec::new(), 4));
        drop(buffer);
        assert_eq!(calls(), ["cuMemFree 0"], "the driver frees it, and waits for no kernel that it does not");
        Ok(())
    }

    /// A buffer dropped in the closure of a capture finds that the driver refuses the wait for the kernels, as it does
    /// while a capture of its thread is on. The drop ends the capture, waits, and begins the capture again: the memory
    /// is not unmapped before the wait is made, and the driver refuses neither the unmapping nor the freeing of the
    /// addresses, which follow.
    #[test]
    fn a_buffer_dropped_in_a_capture_waits_for_the_kernels_once_the_capture_is_ended_and_unmaps_after() -> Result {
        let cuda = fake()?;
        let (buffer, stream) = (holding(&cuda, 3 * WORDS, WORDS + 1)?, cuda.stream()?);
        let e = stream
            .capture(|| {
                drop(buffer);
                Ok(())
            })
            .err()
            .ok_or("capture gave Ok")?;
        assert_eq!(e.call, "cuStreamEndCapture", "the capture is lost");
        let (begin, end, wait) = ("cuStreamBeginCapture", "cuStreamEndCapture", "cuCtxSynchronize");
        let freed = format!("cuMemAddressFree 0 {}", 3 * GRANULE);
        let want = [begin, wait, end, wait, begin, &unmap(1), &unmap(0), &freed, end];
        assert_eq!(calls(), want);
        Ok(())
    }

    #[test]
    fn dropping_a_buffer_waits_for_the_kernels_then_unmaps_the_last_granule_first_and_frees_the_addresses() -> Result {
        let cuda = fake()?;
        let buffer = holding(&cuda, 3 * WORDS, WORDS + 1)?;
        drop(buffer);
        let freed = format!("cuMemAddressFree 0 {}", 3 * GRANULE);
        assert_eq!(calls(), ["cuCtxSynchronize".to_string(), unmap(1), unmap(0), freed.clone()]);
        // A granule the driver refuses to unmap is left, and the addresses are freed all the same.
        let buffer = self::holding(&cuda, 3 * WORDS, WORDS + 1)?;
        REFUSED.set(Some(("cuMemUnmap", 0)));
        drop(buffer);
        assert_eq!(calls(), ["cuCtxSynchronize".to_string(), unmap(1), freed]);
        Ok(())
    }
}
