//! Streams: queues of a GPU's work.

use std::cell::Cell;
use std::ptr;
use std::rc::Rc;

use super::{Cuda, Driver, Error, Handle};

/// What a [`Stream::run`] keeps on its stack while its closure runs: the stream it makes current, the context of that
/// stream's GPU, and the run that was the newest before it, if there was one.
struct Run {
    context: Handle,
    stream: Handle,
    before: *const Run,
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
        // SAFETY: each run in the list is on the stack of a `Stream::run` that has not ended: it takes itself off the
        // list before it ends, by returning or by panicking, and nothing writes to it once it is on.
        while let Some(found) = unsafe { run.as_ref() } {
            if found.context == self.context {
                return found.stream;
            }
            run = found.before;
        }
        ptr::null_mut()
    }
}

impl Cuda {
    /// A stream of this GPU: a queue of work that the GPU runs in order, beside the work of other streams. It is a
    /// blocking stream, as the null stream is: see [`Stream`].
    pub fn stream(&self) -> Result<Stream, Error> {
        let api = self.driver.current()?;
        let mut handle = ptr::null_mut();
        // SAFETY: it writes the stream to `handle`. 0 is its flags, `CU_STREAM_DEFAULT`: a blocking stream.
        api.check("cuStreamCreate", unsafe { (api.create_stream)(&mut handle, 0) })?;
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
/// Dropping a stream destroys it. The work it has enqueued runs on, and the memory that work reaches lasts until it is
/// done, as the drop of memory waits for it.
pub struct Stream {
    driver: Rc<Driver>,
    handle: Handle,
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
    pub unsafe fn run<R>(&self, work: impl FnOnce() -> R) -> R {
        /// Puts back the run that was the newest when `work` ends, by returning or by panicking.
        struct Unlink(*const Run);
        impl Drop for Unlink {
            fn drop(&mut self) {
                RUNNING.set(self.0);
            }
        }
        let run = Run { context: self.driver.context, stream: self.handle, before: RUNNING.get() };
        let _unlink = Unlink(run.before);
        RUNNING.set(&raw const run);
        work()
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
