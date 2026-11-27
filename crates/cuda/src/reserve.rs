//! A buffer whose addresses are reserved for more elements than it has: it backs more of them with memory, or fewer, at
//! the same addresses.

use std::ffi::{c_int, c_void};
use std::marker::PhantomData;
use std::ptr;

use super::unified::Flag;
use super::{Api, Buffer, Cuda, Driver, Error, MemLocation, Plain, bytes, too_many};

/// The GPU maps memory at addresses reserved before it: the driver's virtual memory management.
const VIRTUAL_MEMORY_MANAGEMENT: Flag = (102, "VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED");

/// The GPU maps the host's memory at addresses reserved before it too.
const HOST_MEMORY_MANAGEMENT: Flag = (141, "HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED");

/// The host's memory, where the granules of a buffer from its split on are: its NUMA node 0, the one node of a machine
/// with no NUMA. 3 is `CU_MEM_LOCATION_TYPE_HOST_NUMA`.
const HOST: MemLocation = MemLocation { kind: 3, id: 0 };

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
    /// Memory at `location`: on a GPU, or in the host.
    fn at(location: MemLocation) -> Properties {
        Properties { kind: 1, handles: 0, location, win32: ptr::null_mut(), flags: [0; 8] }
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
    /// The first granule in the host's memory: those before it are on the GPU. It is the granules reserved, where none
    /// is the host's.
    split: usize,
}

impl Reservation {
    /// How many granules hold `len` elements of `T`.
    fn granules<T>(&self, len: usize) -> usize {
        (len * size_of::<T>()).div_ceil(self.granule)
    }

    /// Where the granule `n` is: in the host's memory from the split on, and on the GPU `device` before it.
    fn place(&self, n: usize, device: c_int) -> MemLocation {
        if n >= self.split { HOST } else { MemLocation::gpu(device) }
    }
}

/// Calls `copy(first, n)` for the pieces of a copy of `len` elements, each `n` elements from element `first`: cut at
/// each of `cuts` among them, and one piece of all where none is. A cut is how many elements of one side of the copy
/// are before the split of a buffer that [`Buffer::move_split`] moved, so no piece is over both sides of it: the driver
/// ends the process on such a copy from or to memory the host reaches, a slice of the host's, page-locked memory, or a
/// buffer's granules in the host's memory, say. The split is at an element, as the size of each `Plain` type divides
/// a granule.
pub(super) fn pieces(
    len: usize,
    cuts: [usize; 2],
    mut copy: impl FnMut(usize, usize) -> Result<(), Error>,
) -> Result<(), Error> {
    let [a, b] = cuts;
    let mut first = 0;
    for cut in [a.min(b), a.max(b)] {
        if first < cut && cut < len {
            copy(first, cut - first)?;
            first = cut;
        }
    }
    copy(first, len - first)
}

impl Driver {
    /// How many bytes the driver backs memory at `location` with at a time, at reserved addresses.
    fn granularity(&self, location: MemLocation) -> Result<usize, Error> {
        let (api, mut granule) = (self.current()?, 0);
        // SAFETY: it writes the granularity to `granule`. 0 is CU_MEM_ALLOC_GRANULARITY_MINIMUM: the least memory that
        // the driver backs at a time.
        let asked = unsafe { (api.granularity)(&mut granule, &Properties::at(location), 0) };
        api.check("cuMemGetAllocationGranularity", asked)?;
        Ok(granule)
    }

    /// Backs the granules `*mapped..to` of the addresses at `ptr` with memory, each as a mapping of its own, so that
    /// each can be given back alone, and lets the GPU read and write them, and the host too those in its memory: the
    /// granules from the split of `reservation` on. `*mapped` is how many granules from the start are mapped, and is
    /// the granules mapped when it returns, whether or not it is `to`. It asks the driver for nothing where the
    /// granules are mapped already.
    fn back(&self, ptr: u64, reservation: &Reservation, mapped: &mut usize, to: usize) -> Result<(), Error> {
        let (api, from, granule) = (self.current()?, *mapped, reservation.granule);
        if from >= to {
            return Ok(());
        }
        while *mapped < to {
            let (mut handle, properties) = (0, Properties::at(reservation.place(*mapped, self.device)));
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
        let split = reservation.split.clamp(from, to);
        for (first, end, host) in [(from, split, false), (split, to, true)] {
            if end > first {
                self.allow(api, ptr + (first * granule) as u64, (end - first) * granule, host)?;
            }
        }
        Ok(())
    }

    /// Lets the GPU read and write the `bytes` bytes of granules mapped at `at`, and the host too where they are in its
    /// memory.
    fn allow(&self, api: &Api, at: u64, bytes: usize, host: bool) -> Result<(), Error> {
        let who = [Access { location: MemLocation::gpu(self.device), flags: 3 }, Access { location: HOST, flags: 3 }];
        // SAFETY: the range is of granules mapped whole, and the driver reads as many descriptions from `who` as it is
        // told: the GPU's alone for the GPU's memory, which the host may not be given.
        let allowed = unsafe { (api.set_access)(at, bytes, who.as_ptr(), 1 + usize::from(host)) };
        api.check("cuMemSetAccess", allowed)
    }

    /// Gives the granule at `at` new memory, at `to`, with what it holds: the new memory is mapped at the granule of
    /// addresses `scratch` first, where the old is copied to it on the current stream, which is waited for, and then
    /// in the place of the old, which is given back. Where the driver refuses a step the granule keeps its memory, with
    /// what it holds, and the new is given back.
    ///
    /// # Safety
    ///
    /// No work that is queued or running reaches the granule, but the work the current stream was given before.
    unsafe fn move_granule(&self, at: u64, granule: usize, to: MemLocation, scratch: u64) -> Result<(), Error> {
        let (api, host, mut new) = (self.current()?, to == HOST, 0);
        // SAFETY: it writes the handle to `new`.
        api.check("cuMemCreate", unsafe { (api.create_memory)(&mut new, granule, &Properties::at(to), 0) })?;
        // SAFETY: nothing is mapped at the granule `scratch`. The memory is mapped whole, from its start.
        let copied = api.check("cuMemMap", unsafe { (api.map_memory)(scratch, granule, 0, new, 0) }).and_then(|()| {
            let stream = self.stream();
            let copy = self.allow(api, scratch, granule, host).and_then(|()| {
                // SAFETY: the two are granules of memory mapped whole, which the GPU reaches.
                api.check("cuMemcpyAsync", unsafe { (api.copy_async)(scratch, at, granule, stream) })?;
                // SAFETY: the stream is the null stream, or one that a `run` of this thread holds.
                api.check("cuStreamSynchronize", unsafe { (api.sync_stream)(stream) })
            });
            // SAFETY: the new memory is mapped at `scratch` whole, and the copy to it is done, or was not made.
            copy.and(api.check("cuMemUnmap", unsafe { (api.unmap_memory)(scratch, granule) }))
        });
        // SAFETY: the caller's: no work reaches the granule.
        let moved = copied.and_then(|()| unsafe { self.swap(api, at, granule, new, host) });
        // SAFETY: `cuMemCreate` gave the handle, and it is released once, here: where it is mapped at `at`, the mapping
        // holds the memory.
        unsafe { (api.release_memory)(new) };
        moved
    }

    /// Maps the memory `new` at the granule `at`, in the place of the memory there, which is given back, and lets the
    /// GPU reach it, and the host too where `host`. Where the driver refuses a step the old memory stays, or is mapped
    /// there again, with what it holds and the access it had.
    ///
    /// # Safety
    ///
    /// No work that is queued or running reaches the granule.
    unsafe fn swap(&self, api: &Api, at: u64, granule: usize, new: u64, host: bool) -> Result<(), Error> {
        let mut old = 0;
        // SAFETY: it writes the handle of the memory mapped at `at` to `old`, which holds the memory until it is
        // released. The address is given as the driver's `void *`, which the host reads nothing through.
        let retained = unsafe { (api.retain_memory)(&mut old, ptr::without_provenance_mut(at as usize)) };
        api.check("cuMemRetainAllocationHandle", retained)?;
        // SAFETY: the granule is one whole mapping, and the caller says no work reaches it.
        let swapped = api.check("cuMemUnmap", unsafe { (api.unmap_memory)(at, granule) }).and_then(|()| {
            // SAFETY: nothing is mapped at `at` now. The memory is mapped whole, from its start.
            let mapped = api.check("cuMemMap", unsafe { (api.map_memory)(at, granule, 0, new, 0) });
            mapped.and_then(|()| self.allow(api, at, granule, host)).inspect_err(|_| {
                // SAFETY: no work reaches the granule: the new memory leaves it, where it was mapped, and the old comes
                // back.
                unsafe {
                    (api.unmap_memory)(at, granule);
                    (api.map_memory)(at, granule, 0, old, 0);
                }
                let _ = self.allow(api, at, granule, !host);
            })
        });
        // SAFETY: this lets go of the retain's hold on `old`: where it is mapped at `at`, the mapping holds it.
        unsafe { (api.release_memory)(old) };
        swapped
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
    /// How many bytes of the GPU's memory the driver backs at a time, at addresses that [`reserve`](Cuda::reserve)
    /// reserves: 2 MiB on the GPU here. A buffer that `reserve` made grows, shrinks and moves its split by whole
    /// granules.
    pub fn granule(&self) -> Result<usize, Error> {
        self.driver.granularity(MemLocation::gpu(self.driver.device))
    }

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
        let (api, granule) = (self.driver.current()?, self.granule()?);
        // The driver reserves whole granules, and refuses no bytes, so a granule is the least. One more, past the
        // buffer's, is where `move_split` copies a granule that moves.
        let span = bytes.max(1).checked_next_multiple_of(granule).filter(|span| span.checked_add(granule).is_some());
        let span = span.ok_or_else(|| too_many::<T>("cuMemAddressReserve", capacity))?;
        let mut ptr = 0;
        // SAFETY: it writes the address to `ptr`. The first 0 asks for the driver's alignment, the second for any
        // address, and the third is the flags, of which there are none.
        api.check("cuMemAddressReserve", unsafe { (api.reserve_addresses)(&mut ptr, span + granule, 0, 0, 0) })?;
        let reservation = Some(Reservation { capacity, span, granule, split: span / granule });
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
    /// buffer of 1 element holds a granule, and grows into the rest of it with no more memory. The granules from its
    /// [`split`](Buffer::split) on are the host's memory, and the zeros of each are written alone, as the driver
    /// refuses a zeroing over two of them, or over one and the GPU's memory.
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
        let grown = self
            .driver
            .back(self.ptr, &reservation, &mut mapped, to)
            .and_then(|()| self.zero_bytes(self.len * size_of::<T>(), len * size_of::<T>()));
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

    /// Zeros the bytes `from..to` of its addresses, on the current stream: those on the GPU at once, and those of each
    /// granule in the host's memory alone, as the driver refuses a zeroing over two of them, or over one and the GPU's
    /// memory.
    pub(super) fn zero_bytes(&self, from: usize, to: usize) -> Result<(), Error> {
        let mut at = from;
        while at < to {
            let end = match self.reservation {
                Some(r) if at >= r.split * r.granule => (at / r.granule + 1) * r.granule,
                Some(r) => r.split * r.granule,
                None => to,
            };
            let end = end.min(to);
            self.driver.zero(self.ptr + at as u64, end - at)?;
            at = end;
        }
        Ok(())
    }

    /// The first of its elements in the host's memory: those from it on are there, and those before it on the GPU.
    /// Where none is in the host's memory it is as many elements as its addresses hold, or for a buffer that `alloc` or
    /// `upload` made, its length.
    pub fn split(&self) -> usize {
        self.reservation.map_or(self.len, |r| r.split * r.granule / size_of::<T>())
    }

    /// Puts the granules of its addresses that start at element `at` or past it in the host's memory, and those before
    /// them on the GPU, so that its [`split`](Buffer::split) is the first element of a granule at `at` or past it. Each
    /// granule it has that changes sides moves with what it holds, at the same addresses: the host's memory is
    /// page-locked, and the GPU's kernels reach it there over the bus, and the host too. The granules it grows into
    /// later follow the split, and a split past its addresses puts none in the host's memory.
    ///
    /// A copy over the split, by [`read`](Buffer::read), [`write`](Buffer::write), [`copy_from`](Buffer::copy_from)
    /// or [`Cuda::copy`], is made a side at a time, as the driver ends the process on one over both sides from or to
    /// memory the host reaches: page-locked memory, a slice of the host's, or the granules of a buffer in the host's
    /// memory.
    ///
    /// A granule moves through the granule of addresses past the buffer's, which [`reserve`](Cuda::reserve) reserves
    /// with them: the new memory is mapped there, the granule is copied to it on the current stream, which the call
    /// waits for, and the new memory is mapped in the place of the old, which is given back. The granules move one at
    /// a time, toward the new split, and the split moves with each, so where the driver refuses a step the split is
    /// that of the granules moved: the error is the driver's, and the granule keeps its memory, with what it holds.
    ///
    /// The GPU must map the host's memory at addresses it reserves; where the split first puts a granule there, the
    /// error says so of one that does not, and nothing moves: `cuMemCreate: the GPU cannot map the host's memory at
    /// addresses it reserves: its CU_DEVICE_ATTRIBUTE_HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED is 0`; and of one
    /// that backs the host's memory by bytes that do not divide a [`granule`](Cuda::granule) of its own: `cuMemCreate:
    /// the host's memory is backed 3145728 bytes at a time, which do not divide 2097152`. The buffer is one that
    /// `reserve` made: the error of one that `alloc` or `upload` made is `cuMemMap: a buffer of 4 elements was not made
    /// by reserve, so it does not move its split`.
    ///
    /// # Safety
    ///
    /// No work that is queued or running on the GPU, on any stream, reaches a granule that moves, but the work the
    /// current stream was given before the call, and the host reaches none: the driver unmaps each at once, and does
    /// not wait for work that reaches it, which fails with `CUDA_ERROR_ILLEGAL_ADDRESS (700)`, as does each call after
    /// it. The granules that move are those the buffer has between its split and the first granule at `at` or past it.
    pub unsafe fn move_split(&mut self, at: usize) -> Result<(), Error> {
        let reservation = self.reserved("cuMemMap", "move its split")?;
        let (granule, granules) = (reservation.granule, reservation.span / reservation.granule);
        let to = at.saturating_mul(size_of::<T>()).div_ceil(granule).min(granules);
        if to < granules && reservation.split == granules {
            let cannot = "the GPU cannot map the host's memory at addresses it reserves";
            self.driver.require("cuMemCreate", HOST_MEMORY_MANAGEMENT, cannot)?;
            let host = self.driver.granularity(HOST)?;
            if !granule.is_multiple_of(host) {
                let words =
                    format!("the host's memory is backed {host} bytes at a time, which do not divide {granule}");
                return Err(Error::new("cuMemCreate", words));
            }
        }
        let (from, mapped) = (reservation.split, reservation.granules::<T>(self.len));
        let scratch = self.ptr + reservation.span as u64;
        // Toward the new split: up from the split to the GPU, or down from it to the host.
        for n in (from..to.min(mapped)).chain((to..from.min(mapped)).rev()) {
            let (place, split) = if n < to { (MemLocation::gpu(self.driver.device), n + 1) } else { (HOST, n) };
            // SAFETY: the caller's: no work reaches the granule but the current stream's, which the move waits for.
            unsafe { self.driver.move_granule(self.ptr + (n * granule) as u64, granule, place, scratch) }?;
            self.reservation = Some(Reservation { split, ..reservation });
        }
        self.reservation = Some(Reservation { split: to, ..reservation });
        Ok(())
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
        let addresses = reservation.span + reservation.granule;
        // SAFETY: the addresses are the buffer's and a granule past them, at `ptr`, as `cuMemAddressReserve` gave them.
        // They are freed once, here.
        self.driver.give_back(|api| unsafe { (api.free_addresses)(self.ptr, addresses) });
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

    /// The fake GPU's attribute that says it maps the host's memory there too.
    const HOSTS: Flag = (141, "HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED");

    /// Where the fake driver maps the memory that a granule of a buffer reserved for four moves to, to copy the granule
    /// there: past the buffer's addresses, as bytes from `BASE`.
    const SCRATCH: usize = 4 * GRANULE;

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
        /// What the fake driver has mapped: the address of each mapping, and the handle of its memory.
        static MAPPED: RefCell<Vec<(u64, u64)>> = const { RefCell::new(Vec::new()) };
        /// How many bytes the fake driver backs the host's memory with at a time.
        static HOST_GRANULE: Cell<usize> = const { Cell::new(GRANULE) };
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

    /// What a call of the fake driver says of the stream it is given, when it is not the null stream.
    fn on(stream: Handle) -> String {
        if stream.is_null() { String::new() } else { format!(" on {:#x}", stream.addr()) }
    }

    /// The fake driver's `cuMemGetAllocationGranularity`.
    unsafe extern "C" fn granularity(out: *mut usize, memory: *const Properties, option: c_int) -> CuResult {
        // SAFETY: the caller gives a place for the granularity, and the properties of the memory.
        let (kind, location) = unsafe {
            let (kind, location) = ((*memory).kind, (*memory).location);
            out.write(if location == HOST { HOST_GRANULE.get() } else { GRANULE });
            (kind, location)
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

    /// The fake driver's `cuMemMap`: it keeps what it maps.
    unsafe extern "C" fn map_memory(at: u64, bytes: usize, offset: usize, handle: u64, flags: u64) -> CuResult {
        let said = took(format!("cuMemMap {} {bytes} {offset} {handle} {flags}", from_base(at)));
        if said == 0 {
            MAPPED.with_borrow_mut(|mapped| mapped.push((at, handle)));
        }
        said
    }

    /// The fake driver's `cuMemSetAccess`: it says each description, the kind of place and which, and the access.
    unsafe extern "C" fn set_access(at: u64, bytes: usize, access: *const Access, count: usize) -> CuResult {
        // SAFETY: the caller gives `count` descriptions of the access.
        let who = unsafe { std::slice::from_raw_parts(access, count) };
        let who = who.iter().map(|a| format!("{} {} {}", a.location.kind, a.location.id, a.flags)).collect::<Vec<_>>();
        took(format!("cuMemSetAccess {} {bytes} {}", from_base(at), who.join(" and ")))
    }

    /// The fake driver's `cuMemUnmap`.
    unsafe extern "C" fn unmap_memory(at: u64, bytes: usize) -> CuResult {
        let said = took(format!("cuMemUnmap {} {bytes}", from_base(at)));
        if said == 0 {
            MAPPED.with_borrow_mut(|mapped| mapped.retain(|&(mapping, _)| mapping != at));
        }
        said
    }

    /// The fake driver's `cuMemRetainAllocationHandle`: the handle of the memory it mapped at the address.
    unsafe extern "C" fn retain_memory(out: *mut u64, at: *mut c_void) -> CuResult {
        let at = at.addr() as u64;
        let handle = MAPPED.with_borrow(|mapped| mapped.iter().find(|&&(mapping, _)| mapping == at).map(|m| m.1));
        // SAFETY: the caller gives a place for the handle.
        unsafe { out.write(handle.unwrap_or_default()) };
        took(format!("cuMemRetainAllocationHandle {}", from_base(at)))
    }

    /// The fake driver's `cuMemsetD8Async`.
    unsafe extern "C" fn memset(at: u64, value: u8, bytes: usize, stream: Handle) -> CuResult {
        took(format!("cuMemsetD8Async {} {value} {bytes}{}", from_base(at), on(stream)))
    }

    /// The fake driver's `cuMemcpyAsync`.
    unsafe extern "C" fn copy_async(to: u64, from: u64, bytes: usize, stream: Handle) -> CuResult {
        took(format!("cuMemcpyAsync {} {} {bytes}{}", from_base(to), from_base(from), on(stream)))
    }

    /// The fake driver's `cuMemcpyHtoD`: it copies nothing.
    unsafe extern "C" fn to_device(to: u64, _: *const c_void, bytes: usize) -> CuResult {
        took(format!("cuMemcpyHtoD {} {bytes}", from_base(to)))
    }

    /// The fake driver's `cuMemcpyDtoH`: it copies nothing.
    unsafe extern "C" fn to_host(_: *mut c_void, from: u64, bytes: usize) -> CuResult {
        took(format!("cuMemcpyDtoH {} {bytes}", from_base(from)))
    }

    /// The fake driver's `cuMemcpyPeer`.
    unsafe extern "C" fn copy_peer(to: u64, _: Handle, from: u64, _: Handle, bytes: usize) -> CuResult {
        took(format!("cuMemcpyPeer {} {} {bytes}", from_base(to), from_base(from)))
    }

    /// The fake driver's `cuStreamSynchronize`.
    unsafe extern "C" fn sync_stream(stream: Handle) -> CuResult {
        took(format!("cuStreamSynchronize{}", on(stream)))
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

    /// A GPU that maps memory at addresses it reserves, and the host's memory too, and a driver that takes the calls
    /// above and refuses every other.
    fn fake() -> Result<Cuda> {
        fake_with(&[MAPS, HOSTS])
    }

    /// The same, for a GPU that says each of `flags` is 1.
    fn fake_with(flags: &[Flag]) -> Result<Cuda> {
        CALLS.take();
        REFUSED.take();
        MADE.take();
        CAPTURING.take();
        MAPPED.take();
        HOST_GRANULE.set(GRANULE);
        gpu_with(flags, |api| {
            api.granularity = granularity;
            api.reserve_addresses = reserve_addresses;
            api.free_addresses = free_addresses;
            api.create_memory = create_memory;
            api.release_memory = release_memory;
            api.map_memory = map_memory;
            api.set_access = set_access;
            api.unmap_memory = unmap_memory;
            api.retain_memory = retain_memory;
            api.memset = memset;
            api.copy_async = copy_async;
            api.to_device = to_device;
            api.to_host = to_host;
            api.copy_peer = copy_peer;
            api.sync_stream = sync_stream;
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

    /// The calls that make a granule of memory on the GPU with `handle` and map it `at` bytes from the start of the
    /// addresses. 1 0 is the kind of place that is a GPU, and which.
    fn backing(at: usize, handle: u64) -> [String; 3] {
        backing_at("1 0", at, handle)
    }

    /// The same, for a granule in the host's memory. 3 0 is the kind of place that is a NUMA node of the host, and
    /// which.
    fn host_backing(at: usize, handle: u64) -> [String; 3] {
        backing_at("3 0", at, handle)
    }

    /// The calls that make a granule of memory at `place` with `handle` and map it `at` bytes from the start.
    fn backing_at(place: &str, at: usize, handle: u64) -> [String; 3] {
        [
            format!("cuMemCreate {GRANULE} 1 0 {place} 0"),
            format!("cuMemMap {at} {GRANULE} 0 {handle} 0"),
            format!("cuMemRelease {handle}"),
        ]
    }

    /// The call that lets the GPU read and write `granules` granules from `at` bytes from the start. 3 is to read and
    /// write.
    fn access(at: usize, granules: usize) -> String {
        format!("cuMemSetAccess {at} {} 1 0 3", granules * GRANULE)
    }

    /// The call that lets the GPU and the host read and write `granules` granules in the host's memory from `at`.
    fn host_access(at: usize, granules: usize) -> String {
        format!("{} and 3 0 3", access(at, granules))
    }

    /// The call that unmaps the granule `n`.
    fn unmap(n: usize) -> String {
        format!("cuMemUnmap {} {GRANULE}", n * GRANULE)
    }

    /// The call that zeros `bytes` bytes from `at`, on the null stream.
    fn zeros(at: usize, bytes: usize) -> String {
        format!("cuMemsetD8Async {at} 0 {bytes}")
    }

    /// The calls before a granule `at` bytes from the start moves to new memory at `place` with `new`: the memory is
    /// mapped past the buffer's addresses, reached as it will be, and the granule is copied to it, on the null stream.
    fn copying(place: &str, at: usize, new: u64) -> Vec<String> {
        let allow = if place == "3 0" { host_access(SCRATCH, 1) } else { access(SCRATCH, 1) };
        let copy = format!("cuMemcpyAsync {SCRATCH} {at} {GRANULE}");
        let made = format!("cuMemCreate {GRANULE} 1 0 {place} 0");
        [made, format!("cuMemMap {SCRATCH} {GRANULE} 0 {new} 0"), allow, copy, "cuStreamSynchronize".into()].to_vec()
    }

    /// The calls that move the granule `n` of a buffer reserved for four to new memory at `place` with `new`, from its
    /// memory with `old`.
    fn moving(place: &str, n: usize, new: u64, old: u64) -> Vec<String> {
        let at = n * GRANULE;
        let allow = if place == "3 0" { host_access(at, 1) } else { access(at, 1) };
        let swap =
            [format!("cuMemRetainAllocationHandle {at}"), unmap(n), format!("cuMemMap {at} {GRANULE} 0 {new} 0")];
        let after = [allow, format!("cuMemRelease {old}"), format!("cuMemRelease {new}")];
        [copying(place, at, new), [unmap(4)].to_vec(), swap.to_vec(), after.to_vec()].concat()
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
        // `(capacity, bytes of addresses)`: a granule is the least, and `capacity` elements of `u32` need whole ones,
        // with one more past them, where a granule that moves is copied.
        for (capacity, span) in [(0, GRANULE), (1, GRANULE), (WORDS, GRANULE), (WORDS + 1, 2 * GRANULE)] {
            let buffer = cuda.reserve::<u32>(capacity)?;
            // The memory asked about is the GPU's: 1 is the pinned type, then the kind of place that is a GPU, and
            // which.
            let reserved = format!("cuMemAddressReserve {} 0 0 0", span + GRANULE);
            assert_eq!(calls(), ["cuMemGetAllocationGranularity 1 1 0 0".to_string(), reserved], "capacity {capacity}");
            assert_eq!((buffer.len(), buffer.ptr().addr(), buffer.split()), (0, BASE, span / 4), "capacity {capacity}");
            drop(buffer);
            assert_eq!(calls(), ["cuCtxSynchronize".to_string(), format!("cuMemAddressFree 0 {}", span + GRANULE)]);
        }
        // Bytes, not elements, are rounded up to a granule.
        let buffer = cuda.reserve::<u8>(GRANULE + 1)?;
        assert_eq!(calls()[1], format!("cuMemAddressReserve {} 0 0 0", 3 * GRANULE));
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
        // The granules fit the addresses, and the one past them does not.
        let e = cuda.reserve::<u8>(usize::MAX - GRANULE).err().ok_or("reserve gave Ok")?;
        assert_eq!(e.to_string(), format!("cuMemAddressReserve: {} {words}", usize::MAX - GRANULE));
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
    fn a_buffer_that_alloc_made_neither_grows_shrinks_nor_moves_its_split_and_is_freed_as_a_buffer_is() -> Result {
        let cuda = fake()?;
        let mut buffer = cuda.alloc::<u32>(4)?;
        calls();
        let said = |result: Result<(), Error>| result.err().map(|e| e.to_string());
        let words = "a buffer of 4 elements was not made by reserve, so it does not";
        assert_eq!(said(buffer.grow(8)), Some(format!("cuMemMap: {words} grow")));
        assert_eq!(said(shrink(&mut buffer, 2)), Some(format!("cuMemUnmap: {words} shrink")));
        // SAFETY: the fake driver runs no work.
        assert_eq!(said(unsafe { buffer.move_split(0) }), Some(format!("cuMemMap: {words} move its split")));
        assert_eq!((calls(), buffer.len(), buffer.split()), (Vec::new(), 4, 4));
        drop(buffer);
        assert_eq!(calls(), ["cuMemFree 0"], "the driver frees it, and waits for no kernel that it does not");
        Ok(())
    }

    #[test]
    fn the_granules_a_buffer_grows_into_from_its_split_on_are_the_hosts_and_each_is_zeroed_alone() -> Result {
        let cuda = fake()?;
        let mut buffer = cuda.reserve::<u32>(4 * WORDS)?;
        calls();
        // The split is that of whole granules, and where it first puts some in the host's memory, the driver is asked
        // how it backs the host's memory, which is the host's NUMA node 0. Nothing moves, as the buffer has none.
        // SAFETY: the fake driver runs no work.
        unsafe { buffer.move_split(2 * WORDS - 1) }?;
        let asked = ["cuMemGetAllocationGranularity 1 3 0 0".to_string()].to_vec();
        assert_eq!((calls(), buffer.split()), (asked, 2 * WORDS));
        // The granules before the split are the GPU's, and those from it on the host's, which both reach. The zeros of
        // the GPU's are written at once, and those of each of the host's alone.
        buffer.grow(3 * WORDS + 1)?;
        let made = [backing(0, 1), backing(GRANULE, 2), host_backing(2 * GRANULE, 3), host_backing(3 * GRANULE, 4)];
        let zeroed = [zeros(0, 2 * GRANULE), zeros(2 * GRANULE, GRANULE), zeros(3 * GRANULE, 4)];
        let reached = [access(0, 2), host_access(2 * GRANULE, 2)];
        assert_eq!(calls(), [made.concat(), reached.to_vec(), zeroed.to_vec()].concat());
        // The zeros of every element are written so too.
        buffer.zero()?;
        assert_eq!(calls(), zeroed);
        // A growth within a granule of the host's zeros what it gains alone, and a growth by none zeros nothing.
        buffer.grow(3 * WORDS + 2)?;
        buffer.grow(3 * WORDS + 2)?;
        assert_eq!(calls(), [zeros(3 * GRANULE + 4, 4)]);
        Ok(())
    }

    #[test]
    fn a_copy_over_a_split_is_made_a_side_at_a_time_and_one_beside_it_at_once() -> Result {
        let cuda = fake()?;
        // The granules 0 and 1 are the GPU's, and granule 2 the host's.
        let mut buffer = holding(&cuda, 4 * WORDS, 3 * WORDS)?;
        // SAFETY: the fake driver runs no work.
        unsafe { buffer.move_split(2 * WORDS) }?;
        calls();
        // The host's copies of the buffer, and of a part over the split, are a side of it each; those of a part on one
        // side are one.
        buffer.write(&[1; 3 * WORDS])?;
        buffer.read()?;
        buffer.part(WORDS..3 * WORDS).ok_or("past the end")?.read()?;
        buffer.part_mut(0..2 * WORDS).ok_or("past the end")?.write(&[1; 2 * WORDS])?;
        buffer.part(2 * WORDS + 1..3 * WORDS).ok_or("past the end")?.read()?;
        let (gpu, host) = (format!("0 {}", 2 * GRANULE), format!("{} {GRANULE}", 2 * GRANULE));
        let copies = [
            format!("cuMemcpyHtoD {gpu}"),
            format!("cuMemcpyHtoD {host}"),
            format!("cuMemcpyDtoH {gpu}"),
            format!("cuMemcpyDtoH {host}"),
            format!("cuMemcpyDtoH {GRANULE} {GRANULE}"),
            format!("cuMemcpyDtoH {host}"),
            format!("cuMemcpyHtoD {gpu}"),
            format!("cuMemcpyDtoH {} {}", 2 * GRANULE + 4, GRANULE - 4),
        ];
        assert_eq!(calls(), copies);
        // A copy between two buffers is cut at the split of each: the fake driver reserves the same addresses again,
        // and the second buffer's split is at granule 1.
        let mut other = holding(&cuda, 4 * WORDS, 3 * WORDS)?;
        // SAFETY: the fake driver runs no work.
        unsafe { other.move_split(WORDS) }?;
        calls();
        cuda.copy(&mut other, &buffer)?;
        other.copy_from(&buffer)?;
        cuda.copy(&mut other.part_mut(0..WORDS).ok_or("past the end")?, &buffer.part(0..WORDS).ok_or("past the end")?)?;
        let granules =
            |call: &str| (0..3).map(|n| format!("{call} {} {} {GRANULE}", n * GRANULE, n * GRANULE)).collect();
        let one = [format!("cuMemcpyAsync 0 0 {GRANULE}")].to_vec();
        assert_eq!(calls(), [granules("cuMemcpyAsync"), granules("cuMemcpyPeer"), one].concat());
        Ok(())
    }

    #[test]
    fn a_split_moves_each_granule_that_changes_sides_with_what_it_holds_toward_the_new_split() -> Result {
        let cuda = fake()?;
        // The granules 0 to 2 are the GPU's, with the handles 1 to 3, and granule 3 is not mapped.
        let mut buffer = holding(&cuda, 4 * WORDS, 3 * WORDS)?;
        // SAFETY: the fake driver runs no work.
        unsafe { buffer.move_split(WORDS + 1) }?;
        let asked = "cuMemGetAllocationGranularity 1 3 0 0".to_string();
        assert_eq!((calls(), buffer.split()), ([[asked].to_vec(), moving("3 0", 2, 4, 3)].concat(), 2 * WORDS));
        // The granules that go to the host's memory go the last first, and those that come back to the GPU the first
        // first; the host's memory is asked about once.
        // SAFETY: the fake driver runs no work.
        unsafe { buffer.move_split(WORDS) }?;
        assert_eq!((calls(), buffer.split()), (moving("3 0", 1, 5, 2), WORDS));
        // SAFETY: the fake driver runs no work.
        unsafe { buffer.move_split(3 * WORDS) }?;
        assert_eq!((calls(), buffer.split()), ([moving("1 0", 1, 6, 5), moving("1 0", 2, 7, 4)].concat(), 3 * WORDS));
        // A granule the buffer grows into later follows the split.
        buffer.grow(4 * WORDS)?;
        let grown = [
            host_backing(3 * GRANULE, 8).to_vec(),
            [host_access(3 * GRANULE, 1), zeros(3 * GRANULE, GRANULE)].to_vec(),
        ];
        assert_eq!(calls(), grown.concat());
        // A split past the addresses puts none in the host's memory, and the copy goes to the current stream.
        let stream = cuda.stream()?;
        calls();
        // SAFETY: the fake driver runs no work.
        unsafe { stream.run(|| buffer.move_split(usize::MAX)) }?;
        let mut moved = moving("1 0", 3, 9, 8);
        moved[3] += " on 0x100";
        moved[4] += " on 0x100";
        assert_eq!((calls(), buffer.split()), (moved, 4 * WORDS));
        // And a split where it is moves nothing.
        // SAFETY: the fake driver runs no work.
        unsafe { buffer.move_split(4 * WORDS) }?;
        assert_eq!(calls(), Vec::<String>::new());
        Ok(())
    }

    /// A move of the granules 2 and 1 of a buffer of three to the host's memory, which the driver refuses at the call
    /// `call`, the `nth` of its kind: what the driver is told, and the buffer's split after.
    fn refused_move(call: &'static str, nth: usize) -> Result<(Vec<String>, usize)> {
        let cuda = fake()?;
        let mut buffer = holding(&cuda, 4 * WORDS, 3 * WORDS)?;
        REFUSED.set(Some((call, nth)));
        // SAFETY: the fake driver runs no work.
        let e = unsafe { buffer.move_split(WORDS) }.err().ok_or("move_split gave Ok")?;
        assert_eq!(e.to_string(), format!("{call}: ? (2): ?"));
        let (told, split) = (calls(), buffer.split());
        REFUSED.take();
        // Each granule is where the split says, with its memory: a move the driver takes moves the rest.
        // SAFETY: the fake driver runs no work.
        unsafe { buffer.move_split(WORDS) }?;
        assert_eq!(buffer.split(), WORDS);
        Ok((told, split))
    }

    #[test]
    fn a_move_the_driver_refuses_leaves_the_granule_its_memory_and_the_split_that_of_the_granules_moved() -> Result {
        let asked = "cuMemGetAllocationGranularity 1 3 0 0".to_string();
        let (copy, released) = (copying("3 0", 2 * GRANULE, 4), "cuMemRelease 4".to_string());
        let (retain, at) = (format!("cuMemRetainAllocationHandle {}", 2 * GRANULE), 2 * GRANULE);
        let told = |parts: &[&[String]]| [[asked.clone()].to_vec(), parts.concat()].concat();
        // Refused before the copy is made, or as it is: the new memory leaves the granule past the buffer's, where it
        // was mapped, and is given back.
        let cases = [
            ("cuMemCreate", 0, told(&[&copy[..1]])),
            ("cuMemMap", 0, told(&[&copy[..2], std::slice::from_ref(&released)])),
            ("cuMemSetAccess", 0, told(&[&copy[..3], &[unmap(4), released.clone()]])),
            ("cuMemcpyAsync", 0, told(&[&copy[..4], &[unmap(4), released.clone()]])),
            ("cuStreamSynchronize", 0, told(&[&copy, &[unmap(4), released.clone()]])),
            ("cuMemUnmap", 0, told(&[&copy, &[unmap(4), released.clone()]])),
            ("cuMemRetainAllocationHandle", 0, told(&[&copy, &[unmap(4), retain.clone(), released.clone()]])),
            // Refused as the old memory is unmapped: it stays, and is let go of.
            (
                "cuMemUnmap",
                1,
                told(&[&copy, &[unmap(4), retain.clone(), unmap(2), "cuMemRelease 3".into(), released.clone()]]),
            ),
        ];
        for (call, nth, want) in cases {
            assert_eq!(refused_move(call, nth)?, (want, 4 * WORDS), "{call} {nth}");
        }
        // Refused as the new memory is mapped in its place, or reached there: the old is mapped there again, and the
        // GPU reaches it as it did.
        let new = format!("cuMemMap {at} {GRANULE} 0 4 0");
        let back = [
            unmap(2),
            format!("cuMemMap {at} {GRANULE} 0 3 0"),
            access(at, 1),
            "cuMemRelease 3".into(),
            released.clone(),
        ];
        let swap = [unmap(4), retain.clone(), unmap(2), new.clone()];
        assert_eq!(refused_move("cuMemMap", 1)?, (told(&[&copy, &swap, &back]), 4 * WORDS));
        let swap = [unmap(4), retain, unmap(2), new, host_access(at, 1)];
        assert_eq!(refused_move("cuMemSetAccess", 1)?, (told(&[&copy, &swap, &back]), 4 * WORDS));
        // Refused for the second granule: the split is that of the first, which moved.
        let first = moving("3 0", 2, 4, 3);
        let want = told(&[&first, &[format!("cuMemCreate {GRANULE} 1 0 3 0 0")]]);
        assert_eq!(refused_move("cuMemCreate", 1)?, (want, 2 * WORDS));
        Ok(())
    }

    #[test]
    fn a_split_puts_granules_in_the_hosts_memory_only_where_the_gpu_maps_it_by_whole_granules() -> Result {
        let cuda = fake_with(&[MAPS])?;
        let mut buffer = holding(&cuda, 4 * WORDS, 3 * WORDS)?;
        // SAFETY: the fake driver runs no work.
        let e = unsafe { buffer.move_split(0) }.err().ok_or("move_split gave Ok")?;
        let words = "the GPU cannot map the host's memory at addresses it reserves: its \
                     CU_DEVICE_ATTRIBUTE_HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED is 0";
        assert_eq!(e.to_string(), format!("cuMemCreate: {words}"));
        assert_eq!((calls(), buffer.split()), (Vec::<String>::new(), 4 * WORDS));
        // A split past the addresses puts none there, and asks nothing.
        // SAFETY: the fake driver runs no work.
        unsafe { buffer.move_split(4 * WORDS) }?;
        assert_eq!(calls(), Vec::<String>::new());
        // A GPU that maps it, but backs it by more than a granule of its own at a time.
        let cuda = fake()?;
        HOST_GRANULE.set(3 << 20);
        let mut buffer = holding(&cuda, 4 * WORDS, 3 * WORDS)?;
        // SAFETY: the fake driver runs no work.
        let e = unsafe { buffer.move_split(0) }.err().ok_or("move_split gave Ok")?;
        let words = "the host's memory is backed 3145728 bytes at a time, which do not divide 2097152";
        assert_eq!(e.to_string(), format!("cuMemCreate: {words}"));
        assert_eq!(
            (calls(), buffer.split()),
            (["cuMemGetAllocationGranularity 1 3 0 0".to_string()].to_vec(), 4 * WORDS)
        );
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
        let freed = format!("cuMemAddressFree 0 {}", 4 * GRANULE);
        let want = [begin, wait, end, wait, begin, &unmap(1), &unmap(0), &freed, end];
        assert_eq!(calls(), want);
        Ok(())
    }

    #[test]
    fn dropping_a_buffer_waits_for_the_kernels_then_unmaps_the_last_granule_first_and_frees_the_addresses() -> Result {
        let cuda = fake()?;
        let buffer = holding(&cuda, 3 * WORDS, WORDS + 1)?;
        drop(buffer);
        let freed = format!("cuMemAddressFree 0 {}", 4 * GRANULE);
        assert_eq!(calls(), ["cuCtxSynchronize".to_string(), unmap(1), unmap(0), freed.clone()]);
        // A granule the driver refuses to unmap is left, and the addresses are freed all the same.
        let buffer = self::holding(&cuda, 3 * WORDS, WORDS + 1)?;
        REFUSED.set(Some(("cuMemUnmap", 0)));
        drop(buffer);
        assert_eq!(calls(), ["cuCtxSynchronize".to_string(), unmap(1), freed]);
        Ok(())
    }
}
