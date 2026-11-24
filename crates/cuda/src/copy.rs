//! A copy from one memory to another, which does not wait: on the current stream.

use super::{Cuda, Driver, Error, Memory, MemoryMut, Plain, same_len};

impl Cuda {
    /// Copies what `from` holds to `to`, on the current stream of this GPU: after the work that stream was given
    /// before, and before the work it is given after. The call returns once the copy is queued, and does not wait for
    /// it.
    ///
    /// Each is memory of this GPU: a [`Buffer`], or [`Unified`] memory of `'static` that is managed, page-locked or
    /// registered for good, and for `from` a [`View`] too. Either may be of either kind: the copy tells the direction
    /// by the addresses, so one call copies from page-locked memory to a buffer, back, and from a buffer to a buffer.
    /// A copy in a [`Stream::run`] of a stream of its own runs beside a kernel of another stream.
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
    /// [`View`]: super::View
    /// [`Memory`]: super::Memory
    /// [`MemoryMut`]: super::MemoryMut
    /// [`Cuda::lend`]: super::Cuda::lend
    /// [`Cuda::register`]: super::Cuda::register
    /// [`Stream::run`]: super::Stream::run
    /// [`Buffer::copy_from`]: super::Buffer::copy_from
    pub fn copy<T: Plain>(&self, to: &mut impl MemoryMut<T>, from: &impl Memory<T>) -> Result<(), Error> {
        let ((to, len), (from, from_len)) = (to.reach(self)?, from.reach(self)?);
        same_len("cuMemcpyAsync", "memory", len, "copied from memory of", from_len)?;
        let bytes = len * size_of::<T>();
        let api = self.driver.current()?;
        // SAFETY: each address is of `len` elements of memory of this GPU, which the borrows keep for the call and
        // which lasts until the copy is done, as the host's calls that free or reach it wait for the copy. The stream
        // is the null stream, or one that a `run` of this thread holds.
        api.check("cuMemcpyAsync", unsafe { (api.copy_async)(to.addr(), from.addr(), bytes, self.driver.stream()) })
    }

    /// `Ok` if `memory` is of this GPU: memory of another is not copied here.
    pub(crate) fn reaches(&self, memory: &Driver) -> Result<(), Error> {
        let message = "the memory is on another GPU than the copy is made on";
        self.driver.require_same_gpu(memory, "cuMemcpyAsync", message)
    }
}
