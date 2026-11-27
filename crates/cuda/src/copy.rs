//! A copy from one memory to another, which does not wait: on the current stream. And the copies and zeros at
//! addresses that a program works out, which nothing checks.

use super::{Cuda, Driver, Error, Memory, MemoryMut, Plain, Ptr, bytes, pieces, same_len};

impl Cuda {
    /// Copies what `from` holds to `to`, on the current stream of this GPU: after the work that stream was given
    /// before, and before the work it is given after. The call returns once the copy is queued, and does not wait for
    /// it.
    ///
    /// Each is memory of this GPU: a [`Buffer`], or [`Unified`] memory of `'static` that is managed, page-locked or
    /// registered for good, or a [`Part`] of either, one that `part_mut` gives for `to`. Either may be of either kind:
    /// the copy tells the direction by the addresses, so one call copies from page-locked memory to a buffer, back, and
    /// from a buffer to a buffer, and a copy from `from.part(0..3)` to `to.part_mut(2..5)` copies a part into a part. A
    /// copy in a [`Stream::run`] of a stream of its own runs beside a kernel of another stream. A copy over the split
    /// of a buffer that [`Buffer::move_split`] moved is made a side of it at a time, as the driver ends the process on
    /// one over both sides from or to memory the host reaches.
    ///
    /// The host's calls that reach the memory wait for the copy, as they wait for a kernel: `read`, `write`, `host`,
    /// `host_mut` and dropping it.
    ///
    /// A copy on the null stream is as safe as a launch there, as the work of each stream and the null stream's wait
    /// for each other. In a [`Stream::run`], the `# Safety` of `run` holds for a copy as it does for a launch.
    ///
    /// ```no_run
    /// # use kernite_cuda::Cuda;
    /// let cuda = Cuda::new()?;
    /// let mut staged = cuda.pinned::<f32>(1 << 20)?;
    /// let mut weights = cuda.alloc::<f32>(1 << 20)?;
    /// staged.host_mut()?.fill(1.0);
    /// // The host goes on once the copy is queued, and a kernel launched now runs after it.
    /// cuda.copy(&mut weights, &staged)?;
    /// # Ok::<(), kernite_cuda::Error>(())
    /// ```
    ///
    /// `to` is borrowed mutably, so rustc refuses a copy of memory onto itself:
    ///
    /// ```compile_fail,E0502
    /// # let cuda = kernite_cuda::Cuda::new()?;
    /// let mut buffer = cuda.alloc::<f32>(4)?;
    /// cuda.copy(&mut buffer, &buffer)?;
    /// # Ok::<(), kernite_cuda::Error>(())
    /// ```
    ///
    /// A slice registered for less than for good is neither copied from nor to, as a `mem::forget` of what
    /// [`Cuda::register`] gave would give it back to the host with the copy still to be done. Only `Unified` memory of
    /// `'static` is [`Memory`], and `to` is [`MemoryMut`], which needs [`Memory`], so rustc refuses it in both places:
    ///
    /// ```compile_fail,E0597
    /// # let cuda = kernite_cuda::Cuda::new()?;
    /// let mut buffer = cuda.alloc::<f32>(4)?;
    /// let mut xs = vec![1.0f32; 4];
    /// cuda.copy(&mut buffer, &cuda.register(&mut xs)?)?;
    /// # Ok::<(), kernite_cuda::Error>(())
    /// ```
    ///
    /// An error says what is refused, and nothing is copied. Memory of another number of elements: `cuMemcpyAsync:
    /// memory of 4 elements copied from memory of 2`. Memory of another GPU: `cuMemcpyAsync: the memory is on another
    /// GPU than the copy is made on`: [`Buffer::copy_from`] copies a buffer to one of another GPU. A slice that
    /// [`Cuda::lend`] lent: the driver reads such a slice when the call is made, before the work the stream was given
    /// earlier has run, and writes it only after the call has waited for the stream. So its copy would not be queued in
    /// order, and it is refused. The error names [`Cuda::register`], which page-locks the slice where it is, and whose
    /// copies are queued in order.
    ///
    /// [`Buffer`]: super::Buffer
    /// [`Unified`]: super::Unified
    /// [`Part`]: super::Part
    /// [`Memory`]: super::Memory
    /// [`MemoryMut`]: super::MemoryMut
    /// [`Cuda::lend`]: super::Cuda::lend
    /// [`Cuda::register`]: super::Cuda::register
    /// [`Stream::run`]: super::Stream::run
    /// [`Buffer::copy_from`]: super::Buffer::copy_from
    /// [`Buffer::move_split`]: super::Buffer::move_split
    pub fn copy<T: Plain>(&self, to: &mut impl MemoryMut<T>, from: &impl Memory<T>) -> Result<(), Error> {
        let ((to, len, to_cut), (from, from_len, from_cut)) = (to.reach(self)?, from.reach(self)?);
        same_len("cuMemcpyAsync", "memory", len, "copied from memory of", from_len)?;
        let api = self.driver.current()?;
        pieces(len, [to_cut, from_cut], |first, n| {
            let at = (first * size_of::<T>()) as u64;
            // SAFETY: each address is of `n` elements of memory of this GPU, which the borrows keep for the call and
            // which lasts until the copy is done, as the host's calls that free or reach it wait for the copy. The
            // stream is the null stream, or one that a `run` of this thread holds.
            let copied =
                unsafe { (api.copy_async)(to.addr() + at, from.addr() + at, n * size_of::<T>(), self.driver.stream()) };
            api.check("cuMemcpyAsync", copied)
        })
    }

    /// Copies the `len` elements at `from` to `to`, as [`copy`](Cuda::copy) does: on the current stream, and the call
    /// does not wait for it. Each address is one the GPU reaches, of its memory or of the host's that is page-locked,
    /// and the copy tells the direction by them. A `len` of more bytes than there are addresses is refused, and nothing
    /// is copied: `cuMemcpyAsync: 18446744073709551615 elements of 4 bytes are more bytes than there are addresses`.
    ///
    /// # Safety
    ///
    /// Each address is of `len` elements of memory of this GPU or of the host that the GPU reaches, which the two do
    /// not share, and which lasts until the copy is done: no work and no call of the host reaches the elements at `to`,
    /// nor writes those at `from`, until then. Where the elements at one address are on both sides of the split of a
    /// buffer that [`Buffer::move_split`] moved, those at the other are of the GPU's own memory or managed memory: the
    /// driver ends the process on such a copy from or to memory the host reaches, page-locked memory, say, or a
    /// buffer's granules in the host's memory. In a [`Stream::run`], the `# Safety` of `run` holds too.
    ///
    /// [`Stream::run`]: super::Stream::run
    /// [`Buffer::move_split`]: super::Buffer::move_split
    pub unsafe fn copy_at<T: Plain>(&self, to: Ptr<T>, from: Ptr<T>, len: usize) -> Result<(), Error> {
        let bytes = bytes::<T>("cuMemcpyAsync", len)?;
        let api = self.driver.current()?;
        // SAFETY: the caller's. The stream is the null stream, or one that a `run` of this thread holds.
        api.check("cuMemcpyAsync", unsafe { (api.copy_async)(to.addr(), from.addr(), bytes, self.driver.stream()) })
    }

    /// Sets the `len` elements at `at` to zero, as [`Buffer::zero`] does: on the current stream, and the call does not
    /// wait for it.
    ///
    /// # Safety
    ///
    /// `at` is the address of `len` elements of memory of this GPU, which last until the zeros are written, and no work
    /// and no call of the host reaches them until then. They are in one buffer, or the GPU's own memory of one buffer
    /// that [`Cuda::reserve`] made, or one granule of its host's: the driver refuses a zeroing over more. In a
    /// [`Stream::run`], the `# Safety` of `run` holds too.
    ///
    /// [`Buffer::zero`]: super::Buffer::zero
    /// [`Stream::run`]: super::Stream::run
    pub unsafe fn zero_at<T: Plain>(&self, at: Ptr<T>, len: usize) -> Result<(), Error> {
        self.driver.zero(at.addr(), bytes::<T>("cuMemsetD8Async", len)?)
    }

    /// Copies `data` to the GPU at `to`, as [`Buffer::write`] does: after the kernels launched before have run, but
    /// those of a stream apart.
    ///
    /// # Safety
    ///
    /// `to` is the address of `data.len()` elements of memory of this GPU, and no work reaches them until the copy is
    /// done: the driver may still copy from its own memory to the GPU's when the call returns, before the work that a
    /// stream apart is given. The elements are on one side of the split of a buffer that [`Buffer::move_split`] moved:
    /// the driver ends the process on a copy from the host's memory over both.
    ///
    /// [`Buffer::write`]: super::Buffer::write
    /// [`Buffer::move_split`]: super::Buffer::move_split
    pub unsafe fn write_at<T: Plain>(&self, to: Ptr<T>, data: &[T]) -> Result<(), Error> {
        self.driver.write(to.addr(), data, data.len())
    }

    /// Copies the `out.len()` elements at `from` to `out`, as [`Buffer::read_into`] does: after the kernels launched
    /// before have run, but those of a stream apart.
    ///
    /// # Safety
    ///
    /// `from` is the address of `out.len()` elements of memory of this GPU, which no work writes while the copy runs.
    /// The elements are on one side of the split of a buffer that [`Buffer::move_split`] moved: the driver ends the
    /// process on a copy to the host's memory over both.
    ///
    /// [`Buffer::read_into`]: super::Buffer::read_into
    /// [`Buffer::move_split`]: super::Buffer::move_split
    pub unsafe fn read_at<T: Plain>(&self, from: Ptr<T>, out: &mut [T]) -> Result<(), Error> {
        self.driver.read_into(from.addr(), out, out.len())
    }

    /// `Ok` if `memory` is of this GPU: memory of another is not copied here.
    pub(crate) fn reaches(&self, memory: &Driver) -> Result<(), Error> {
        let message = "the memory is on another GPU than the copy is made on";
        self.driver.require_same_gpu(memory, "cuMemcpyAsync", message)
    }
}
