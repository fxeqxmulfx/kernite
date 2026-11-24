//! Events: points in the work of a stream.

use std::ffi::c_uint;
use std::marker::PhantomData;
use std::ptr;
use std::rc::Rc;
use std::time::Duration;

use super::{Cuda, Driver, Error, Handle};

/// What an [`Event`] is that [`Cuda::timed_event`] makes: one that times work. [`Event::duration_since`] is for it
/// alone.
pub enum Timed {}

/// What an [`Event`] is that [`Cuda::event`] makes: one that does not time work, and so costs the driver less to
/// record.
pub enum Untimed {}

impl Cuda {
    /// An event of this GPU, made not to time work: a point in the work of a stream, which says whether the work before
    /// it is done, is waited for by the host and by another stream, and costs the driver less to record than one made
    /// to time, [`timed_event`](Cuda::timed_event). See [`Event`].
    pub fn event(&self) -> Result<Event, Error> {
        self.make_event(DISABLE_TIMING)
    }

    /// An event of this GPU, made to time work: what [`event`](Cuda::event) makes, and [`Event::duration_since`] says
    /// the time between two of them.
    pub fn timed_event(&self) -> Result<Event<Timed>, Error> {
        self.make_event(0)
    }

    /// An event with `flags`, which `T` says the meaning of.
    fn make_event<T>(&self, flags: c_uint) -> Result<Event<T>, Error> {
        let api = self.driver.current()?;
        let mut handle = ptr::null_mut();
        // SAFETY: it writes the event to `handle`. `flags` is 0, `CU_EVENT_DEFAULT`, or `CU_EVENT_DISABLE_TIMING`.
        api.check("cuEventCreate", unsafe { (api.create_event)(&mut handle, flags) })?;
        Ok(Event { driver: Rc::clone(&self.driver), handle, timing: PhantomData })
    }
}

/// `CU_EVENT_DISABLE_TIMING`: the flag of an event that does not time work.
const DISABLE_TIMING: c_uint = 2;

/// A point in the work of a stream, which the GPU reaches when it has done the work before it. [`Cuda::event`] makes
/// one that does not time work, and [`Cuda::timed_event`] one that does, an `Event<Timed>`.
///
/// [`record`](Event::record) makes the event stand for the work that the current stream has been given so far: the
/// stream of the GPU that [`Stream::run`](super::Stream::run) made current on this thread, or the null stream. A record
/// again makes it stand for the work the stream was given by then, and for that alone. Then:
///
/// - [`done`](Event::done) says whether that work is done, and does not wait for it.
/// - [`sync`](Event::sync) waits for that work, and for no more.
/// - [`wait`](Event::wait) makes the work that the current stream is given after the call wait for that work, and the
///   host does not wait.
///
/// An event that was never recorded stands for no work: it is done, and nothing waits for it. An event on the null
/// stream stands for the work that every stream of the GPU was given before the record: a stream is blocking, as
/// [`Stream`](super::Stream) says.
///
/// Dropping an event destroys it, and does not wait: the work goes on, and a stream that was made to wait for it still
/// waits.
///
/// ```no_run
/// # use kernite_cuda::Cuda;
/// let cuda = Cuda::new()?;
/// let (writes, reads) = (cuda.stream()?, cuda.stream()?);
/// let written = cuda.event()?;
/// // SAFETY: what the kernel on `writes` writes, the work on `reads` does not touch until it waits for the event.
/// unsafe {
///     writes.run(|| {
///         // The kernel that writes is launched here, on the current stream.
///         written.record()
///     })
/// }?;
/// // The kernel that reads what the other wrote waits for it, and the host does not.
/// // SAFETY: as above.
/// unsafe {
///     reads.run(|| {
///         written.wait()?;
///         // The kernel that reads is launched here, on the current stream.
///         Ok::<(), kernite_cuda::Error>(())
///     })
/// }?;
/// # Ok::<(), kernite_cuda::Error>(())
/// ```
pub struct Event<T = Untimed> {
    driver: Rc<Driver>,
    handle: Handle,
    /// `Timed` if the driver was asked to time work, and `Untimed` if not.
    timing: PhantomData<T>,
}

impl<T> Event<T> {
    /// Makes this event stand for the work the current stream has been given so far, in place of what it stood for.
    /// The stream is the current one of this event's GPU: the null stream, or the stream of the [`Stream::run`] this
    /// is in. The call does not wait for the work.
    ///
    /// [`Stream::run`]: super::Stream::run
    pub fn record(&self) -> Result<(), Error> {
        let api = self.driver.current()?;
        // SAFETY: the event was made, and lasts as long as `self`. The stream is the null stream, or one that a `run`
        // of this thread holds, which lasts until the `run` ends, and so past this call.
        api.check("cuEventRecord", unsafe { (api.record_event)(self.handle, self.driver.stream()) })
    }

    /// Whether the work this event stands for is done: `false` while any of that work, a kernel or a copy, is queued
    /// or running. It is an error if the work failed, as [`Cuda::sync`] says it. An event that was never recorded
    /// stands for no work, so it is done.
    pub fn done(&self) -> Result<bool, Error> {
        let api = self.driver.current()?;
        // SAFETY: the event was made, and lasts as long as `self`.
        match unsafe { (api.query_event)(self.handle) } {
            0 => Ok(true),
            // CUDA_ERROR_NOT_READY: some of the work is not done.
            600 => Ok(false),
            result => Err(api.failed("cuEventQuery", result)),
        }
    }

    /// Waits for the work this event stands for, and for no more, and says whether it failed.
    pub fn sync(&self) -> Result<(), Error> {
        let api = self.driver.current()?;
        // SAFETY: the event was made, and lasts as long as `self`.
        api.check("cuEventSynchronize", unsafe { (api.sync_event)(self.handle) })
    }

    /// Makes the work that the current stream is given after this call wait for the work this event stands for now, as
    /// the last [`record`](Event::record) made it. The host does not wait. A record after this call does not change
    /// what the stream waits for. The stream is the current one of this event's GPU, as in `record`.
    pub fn wait(&self) -> Result<(), Error> {
        let api = self.driver.current()?;
        // SAFETY: the event was made, and lasts as long as `self`, and the stream is as in `record`. 0 is its flags,
        // `CU_EVENT_WAIT_DEFAULT`.
        api.check("cuStreamWaitEvent", unsafe { (api.wait_event)(self.driver.stream(), self.handle, 0) })
    }
}

impl Event<Timed> {
    /// The time from when the GPU reached `earlier` to when it reached this event, as the GPU's own clock says it, to
    /// about half a microsecond. It is the time that went by on the GPU, so it holds the work of other streams that ran
    /// beside the work between the two, as well as that work.
    ///
    /// Both events have been recorded, and the work before both is done, which [`sync`](Event::sync) waits for. Where
    /// one is not, the error is the driver's: `CUDA_ERROR_INVALID_HANDLE` for an event never recorded, and
    /// `CUDA_ERROR_NOT_READY` for one not done.
    ///
    /// An event that the GPU reached before `earlier` is an error, as a `Duration` is not below zero: ask the one it
    /// reached later for its time since the other. The host may have recorded them in the other order, for the GPU
    /// reaches an event when the work before it on its own stream is done. An event of another GPU, whose clock is not
    /// this one's, is an error too, and the driver is not asked.
    ///
    /// The driver must be of CUDA 12.8 (570) or later, which brought `cuEventElapsedTime_v2`:
    /// [`Cuda::new`](super::Cuda::new) opens an older driver all the same, and `duration_since` gives the error of the
    /// lookup then, ``dlsym: the driver has no `cuEventElapsedTime_v2`: libcuda.so.1: undefined symbol:
    /// cuEventElapsedTime_v2``.
    ///
    /// ```no_run
    /// # use kernite_cuda::Cuda;
    /// let cuda = Cuda::new()?;
    /// let (start, stop) = (cuda.timed_event()?, cuda.timed_event()?);
    /// start.record()?;
    /// // A kernel is launched here.
    /// stop.record()?;
    /// stop.sync()?;
    /// println!("the kernel took {:?}", stop.duration_since(&start)?);
    /// # Ok::<(), kernite_cuda::Error>(())
    /// ```
    ///
    /// An event made not to time work has no time to say, and rustc refuses it as the later event, the one that is
    /// asked:
    ///
    /// ```compile_fail,E0599
    /// # use kernite_cuda::Cuda;
    /// let cuda = Cuda::new()?;
    /// let (start, stop) = (cuda.timed_event()?, cuda.event()?);
    /// let time = stop.duration_since(&start)?;
    /// # Ok::<(), kernite_cuda::Error>(())
    /// ```
    ///
    /// As the earlier event, the one that is given:
    ///
    /// ```compile_fail,E0308
    /// # use kernite_cuda::Cuda;
    /// let cuda = Cuda::new()?;
    /// let (start, stop) = (cuda.event()?, cuda.timed_event()?);
    /// let time = stop.duration_since(&start)?;
    /// # Ok::<(), kernite_cuda::Error>(())
    /// ```
    ///
    /// And as both:
    ///
    /// ```compile_fail,E0599
    /// # use kernite_cuda::Cuda;
    /// let cuda = Cuda::new()?;
    /// let (start, stop) = (cuda.event()?, cuda.event()?);
    /// let time = stop.duration_since(&start)?;
    /// # Ok::<(), kernite_cuda::Error>(())
    /// ```
    pub fn duration_since(&self, earlier: &Event<Timed>) -> Result<Duration, Error> {
        let message = "the events are of two GPUs, and the clocks of two GPUs are not one clock";
        self.driver.require_same_gpu(&earlier.driver, "cuEventElapsedTime", message)?;
        let api = self.driver.current()?;
        let elapsed = api.elapsed_time.clone()?;
        let mut millis = 0.0f32;
        // SAFETY: it writes the milliseconds to `millis`. Both events were made, and last as long as the borrows.
        api.check("cuEventElapsedTime", unsafe { elapsed(&mut millis, earlier.handle, self.handle) })?;
        Duration::try_from_secs_f64(f64::from(millis) / 1000.0).map_err(|_| {
            let message = "the GPU reached this event before the one it is timed since: ask the later of the two";
            Error::new("cuEventElapsedTime", message.into())
        })
    }
}

impl<T> Drop for Event<T> {
    fn drop(&mut self) {
        // An event whose GPU's context cannot be made current is left, and not destroyed in another's.
        if let Ok(api) = self.driver.current() {
            // SAFETY: the event was made, and is destroyed once, here. The driver does not wait for the work it
            // stands for, and a stream that waits for it still waits.
            unsafe { (api.destroy_event)(self.handle) };
        }
    }
}
