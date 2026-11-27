//! Streams: queues of a GPU's work.

use std::cell::Cell;
use std::ffi::{c_int, c_uint};
use std::ptr;
use std::rc::Rc;
use std::thread;
use std::time::Duration;

use super::graph::Capture;
use super::{Api, CuResult, Cuda, Driver, Error, Handle};

/// `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`: what the driver says of a call that it cannot record, or that would wait
/// for the stream that captures, while a capture is on.
const CAPTURE_UNSUPPORTED: CuResult = 900;

/// `CU_STREAM_NON_BLOCKING`: the flag of a stream apart, whose work and the null stream's do not wait for each other.
const APART: c_uint = 1;

/// How long a drop waits between two tries of a call that a capture refuses: long enough not to keep a core busy for as
/// long as the capture lasts, and short enough for the drop to be done soon after the capture is.
const BETWEEN_TRIES: Duration = Duration::from_micros(100);

/// What a [`Stream::run`] keeps while its closure runs: the stream it makes current, the context of that stream's GPU,
/// the run that was the newest before it, if there was one, and, where the run is a [`capture`](Stream::capture)'s,
/// the capture. It is on the stack of the call that makes it.
pub(super) struct Run {
    context: Handle,
    /// The stream the work goes to: the one the run was made for, and the null stream where a drop ended the capture
    /// and the driver refused to begin it again, as work for a stream that does not capture would run.
    stream: Cell<Handle>,
    before: Cell<*const Run>,
    capture: Option<Rc<Capture>>,
}

thread_local! {
    /// The newest [`Stream::run`] of this thread that has not ended, and through it the ones before it, or null while
    /// none is running. A launch looks along them for its GPU's stream, and the null stream, which is no handle, is
    /// where it finds none. A `Cell` of a pointer has no destructor, so it can be read at any time: in the destructor
    /// of another thread-local too, as the thread exits, where one that has a destructor may be gone.
    static RUNNING: Cell<*const Run> = const { Cell::new(ptr::null()) };
}

impl Driver {
    /// The stream the work this thread enqueues on this GPU goes to: the one [`Stream::run`] made current for it last,
    /// or the null stream. The thread-local is all it costs a call that enqueues work, where no `run` is running.
    #[inline]
    pub(crate) fn stream(&self) -> Handle {
        let mut run = RUNNING.get();
        // SAFETY: each run in the list is on the stack of a call of `during` that has not ended: it takes itself off
        // the list before it ends, by returning or by panicking, and only its `Cell`s are written once it is on.
        while let Some(found) = unsafe { run.as_ref() } {
            if found.context == self.context {
                return found.stream.get();
            }
            run = found.before.get();
        }
        ptr::null_mut()
    }

    /// Makes `call`, a call that a drop needs to make, to wait for the GPU or to give memory, a module or a graph back,
    /// and leaves what it returns, as a drop does where it fails after an error of the GPU's work. A call is made once,
    /// as it was, where no capture refuses it. While a capture of its thread is on, the driver refuses the drops of a
    /// buffer, of managed and page-locked memory, of a slice registered, of a module and of a graph, and says
    /// `CAPTURE_UNSUPPORTED`. While any capture is on, it refuses a wait for the GPU as a whole, which the drops of a
    /// slice lent and of a reserved buffer make, and says it too. Then the captures of this thread are ended, the
    /// newest first. The call is made, and made again while a capture of another thread refuses it. Then the captures
    /// are begun again, the oldest first. So the drop waits for the kernels that reach what it gives back, as a drop
    /// does, and the closure of the capture goes on: see [`Capture`]. A capture that kernite did not begin, by the
    /// driver's `cuStreamBeginCapture` on the thread of the drop, is one it cannot end: the drop tries its call again
    /// for ever, as the driver refuses it until that capture ends, which only that thread can end.
    pub(super) fn give_back(&self, mut call: impl FnMut(&Api) -> CuResult) {
        // A drop whose GPU's context cannot be made current leaves what it gives back, and does not give it back in
        // another's.
        let Ok(api) = self.current() else { return };
        if call(api) == CAPTURE_UNSUPPORTED {
            self.give_back_outside_captures(RUNNING.get(), &mut call);
        }
    }

    /// Ends the captures of this thread from the run `run` on, the newest first, makes `call` while a capture of
    /// another thread refuses it, and begins them again, the oldest first.
    #[cold]
    fn give_back_outside_captures(&self, run: *const Run, call: &mut impl FnMut(&Api) -> CuResult) {
        // SAFETY: as in `stream`.
        let Some(found) = (unsafe { run.as_ref() }) else {
            // A capture of another thread refuses the call as long as it lasts, and the call that is refused first
            // has lost it: the thread that captures goes on with its closure, and ends the capture, which is lost.
            while self.current().is_ok_and(|api| call(api) == CAPTURE_UNSUPPORTED) {
                thread::sleep(BETWEEN_TRIES);
            }
            return;
        };
        match found.capture.as_deref().filter(|capture| capture.is_on()) {
            None => self.give_back_outside_captures(found.before.get(), call),
            Some(capture) => {
                capture.leave();
                self.give_back_outside_captures(found.before.get(), call);
                if !capture.resume() {
                    found.stream.set(ptr::null_mut());
                }
            }
        }
    }
}

impl Run {
    /// A run that makes `stream` the current stream of its GPU, which captures what `stream` is capturing, if it has
    /// `capture`.
    pub(super) fn of(stream: &Stream, capture: Option<Rc<Capture>>) -> Run {
        Run {
            context: stream.driver.context,
            stream: Cell::new(stream.handle),
            before: Cell::new(ptr::null()),
            capture,
        }
    }

    /// Runs `work`, and returns what it returns, with this run the newest of the thread's, which a launch looks
    /// along. It is the newest no more when `work` returns, or panics.
    pub(super) fn during<R>(&self, work: impl FnOnce() -> R) -> R {
        /// Puts back the run that was the newest when `work` ends, by returning or by panicking.
        struct Unlink(*const Run);
        impl Drop for Unlink {
            fn drop(&mut self) {
                RUNNING.set(self.0);
            }
        }
        let before = RUNNING.get();
        self.before.set(before);
        let _unlink = Unlink(before);
        RUNNING.set(self);
        work()
    }
}

impl Cuda {
    /// A stream of this GPU: a queue of work that the GPU runs in order, beside the work of other streams. It is a
    /// blocking stream, as the null stream is: see [`Stream`].
    pub fn stream(&self) -> Result<Stream, Error> {
        // 0 is `CU_STREAM_DEFAULT`, a blocking stream, at the priority 0, the least.
        self.stream_of(0, 0)
    }

    /// A stream apart of this GPU: a stream whose work and the null stream's do not wait for each other, as two
    /// streams' do not, so that the calls that wait for the null stream's work do not wait for its: see [`Stream`].
    /// Where `urgent`, it has the GPU's greatest priority: where the kernels of several streams wait for room, the GPU
    /// starts the urgent stream's first. It stops no kernel that runs, and its copies are not hastened.
    pub fn stream_apart(&self, urgent: bool) -> Result<Stream, Error> {
        let mut priority = 0;
        if urgent {
            let api = self.driver.current()?;
            // SAFETY: it writes the greatest priority, the least number, to `priority`, and the least nowhere, as the
            // driver takes a null pointer for a value not asked.
            let asked = unsafe { (api.priorities)(ptr::null_mut(), &mut priority) };
            api.check("cuCtxGetStreamPriorityRange", asked)?;
        }
        self.stream_of(APART, priority)
    }

    /// A stream of this GPU with the driver's `flags`, at `priority`.
    fn stream_of(&self, flags: c_uint, priority: c_int) -> Result<Stream, Error> {
        let api = self.driver.current()?;
        let mut handle = ptr::null_mut();
        // SAFETY: it writes the stream to `handle`.
        let made = unsafe { (api.create_stream)(&mut handle, flags, priority) };
        api.check("cuStreamCreateWithPriority", made)?;
        Ok(Stream { driver: Rc::clone(&self.driver), handle })
    }
}

/// A queue of a GPU's work, which the GPU runs in the order it was enqueued, and beside the work of other streams where
/// it has room. [`Cuda::stream`] makes one.
///
/// The GPU has a stream of its own besides, the null stream, which all the work goes to unless a thread says another:
/// kernels, and what the host's copies and the drops of memory wait for. A kernel is launched on the current stream
/// of its GPU, the thread's, and so are a prefetch, a copy by [`Cuda::copy`] and a zeroing: `zero`, and the zeros that
/// `alloc`, `managed`, `grow` and the load of a module write. It is the null stream, until [`run`](Stream::run) makes
/// another current for a closure. Each thread has its own, and each GPU has one on each thread.
///
/// A stream is blocking: its work and the null stream's wait for each other, as the null stream's work waits for all
/// the work enqueued before it on any stream, and a stream's work for what the null stream has enqueued before it. So
/// what waits for the null stream's work waits for a stream's: the host's `read`, `write`, `host` and `host_mut`, and
/// the drop of a buffer, of unified memory and of a module. Two streams' work does not wait for each other, and runs at
/// once where the GPU has room.
///
/// A stream apart, which [`Cuda::stream_apart`] makes, is not blocking: its work and the null stream's do not wait for
/// each other, and run at once where the GPU has room, as two streams' do. So the kernels launched before, which a
/// call of the null stream waits for, are those of the null stream and of the blocking streams alone: `read`, `write`
/// and `copy_from` do not wait for the work of a stream apart, nor do the zeros and the copies made outside a `run` of
/// it. What waits for all of the GPU's work waits for its: [`Cuda::sync`], `host` and `host_mut`.
///
/// Dropping a stream destroys it. The work it has enqueued runs on, and the memory that work reaches lasts until it is
/// done: the drop of memory waits for the work of a blocking stream, and comes after that of a stream apart, as
/// [`run`](Stream::run) says.
pub struct Stream {
    pub(super) driver: Rc<Driver>,
    pub(super) handle: Handle,
}

impl Stream {
    /// Runs `work`, and returns what it returns, with this stream the current stream of its GPU on this thread: each
    /// kernel that `work` launches on the GPU, by [`Module::launch`](super::Module::launch) or by a launcher the macros
    /// write, goes to it, through whichever [`Cuda`] of the GPU launches it, and so does each prefetch, copy and
    /// zeroing that `work` makes on the GPU, and the zeros that `alloc`, `managed`, `grow` and the load of a module
    /// write. A `run` in `work` makes another stream current, or this one, for its own closure; a `run` of a stream of
    /// another GPU is for that GPU alone. The stream that was current is current again when `work` returns, or panics.
    ///
    /// # Safety
    ///
    /// Until the work that `work` enqueues is done, no work of another stream that may run beside it may write memory
    /// that it reaches, or reach memory that it writes. This stream's work keeps the order it was enqueued in, and the
    /// null stream's work and a [`sync`](Stream::sync) order it with the rest. kernite cannot see what the work of
    /// two streams reaches, and to keep it itself it would have to make each launch wait for the others, which the
    /// streams are for not doing. In `work`, each call is as safe as it is on the null stream.
    ///
    /// The null stream's work does not order the work of a stream apart: its `sync`, an event and [`Cuda::sync`] do.
    /// The memory that its work reaches, and the module of its kernels, are dropped once the work is done.
    pub unsafe fn run<R>(&self, work: impl FnOnce() -> R) -> R {
        Run::of(self, None).during(work)
    }

    /// Waits for the work enqueued on this stream, and says whether one failed. The work of other streams is not
    /// waited for: [`Cuda::sync`] waits for all of a GPU's.
    pub fn sync(&self) -> Result<(), Error> {
        let api = self.driver.current()?;
        // SAFETY: the stream was made, and lasts as long as `self`.
        api.check("cuStreamSynchronize", unsafe { (api.sync_stream)(self.handle) })
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        // A stream whose GPU's context cannot be made current is left, and not destroyed in another's.
        if let Ok(api) = self.driver.current() {
            // SAFETY: the stream was made, and is destroyed once, here. The driver runs the work queued on it first.
            unsafe { (api.destroy_stream)(self.handle) };
        }
    }
}
