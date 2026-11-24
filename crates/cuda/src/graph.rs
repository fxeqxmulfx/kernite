//! Graphs: the work of a stream, captured, to be launched as often as it is asked.

use std::cell::Cell;
use std::ffi::c_int;
use std::ptr;
use std::rc::Rc;

use super::stream::Run;
use super::{Api, CuResult, Driver, Error, Handle, Stream};

/// `CU_STREAM_CAPTURE_MODE_THREAD_LOCAL`: while the capture lasts, a call of this thread that the driver cannot record,
/// as it would wait or make memory, is refused. Another thread may make and free memory, but its wait for the whole
/// GPU, or its work on the null stream, is refused, as it would wait for the stream that captures.
const THREAD_LOCAL: c_int = 1;

/// `CUDA_ERROR_STREAM_CAPTURE_INVALIDATED`: what ending a capture says where the driver could not record all of it.
const INVALIDATED: CuResult = 901;

impl Stream {
    /// Runs `work` with this stream the current stream of its GPU on this thread, as [`run`](Stream::run) does, and
    /// records the work that it enqueues on the stream, a launch, a copy or a zeroing, instead of running it. It gives
    /// the record as a [`Graph`], which [`launch`](Graph::launch) runs, all of it, as often as it is asked, for the
    /// cost of one launch. The work that was enqueued on the stream before the capture runs as it did, and is no part
    /// of the graph.
    ///
    /// The driver refuses what it cannot record: a call that waits for the GPU or for the host, as a
    /// [`read`](super::Buffer::read) and a [`sync`](Stream::sync) do; a call that makes memory; work for the null
    /// stream; a wait for an event that was recorded before the capture; and a query of an event that was recorded in
    /// it. Of the calls that make memory, it refuses those of the thread that captures alone. A call of another thread
    /// that waits for the GPU as a whole, or enqueues work on the null stream, which waits for every stream, is refused
    /// too, as it would wait for the stream that captures, and the capture is lost. The error of `capture` is the error
    /// of `work`, or the driver's where `work` ignores it: the capture is over, and no graph is made of what the driver
    /// could not record. Either way, and where `work` panics, the capture is ended, the stream captures no more, and it
    /// takes work as it did before. A capture on a stream that is capturing already is refused, and the capture that is
    /// on goes on.
    ///
    /// A drop waits for the GPU, or the driver would give memory back, or unload a module, under a kernel that reaches
    /// it. While a capture of its thread is on, the driver refuses the drops of a buffer, of managed and page-locked
    /// memory, of a slice registered, of a module and of a graph. While any capture is on, it refuses a wait for the
    /// GPU as a whole, which the drops of a slice lent and of a reserved buffer make. A drop in `work` does not skip
    /// its wait. It ends each capture of its thread, makes its call, and begins each capture again, on its stream and
    /// in its mode, so that what `work` enqueues after the drop is recorded, and does not run. The drop has lost the
    /// capture, as a refused call does: `capture` gives no graph, and its error, where `work` returns `Ok`, is
    /// `cuStreamEndCapture: CUDA_ERROR_STREAM_CAPTURE_INVALIDATED (901)`. Where the driver refuses to begin a capture
    /// again, what `work` enqueues after the drop goes to the null stream, whose work is safe. A drop of another thread
    /// whose wait for the GPU as a whole is refused, as that of a slice lent and that of a reserved buffer, has lost
    /// the capture with its first call: it tries again, with a little time between the tries, until the capture is
    /// over, and then it waits for the kernels. The drops of a buffer, of managed and page-locked memory, of a slice
    /// registered and of a module, in another thread, are not refused: they wait for the kernels, and the capture goes
    /// on. Where no capture is on, a drop makes its call once. A capture that kernite did not begin, by the driver's
    /// `cuStreamBeginCapture` on the thread of the drop, is one it cannot end: the drop tries its call again for ever,
    /// as the driver refuses it until that capture ends, which only that thread can end.
    ///
    /// Nothing that `work` enqueues on this stream runs, so `capture` is safe, as [`run`](Stream::run) is not: the
    /// `unsafe` is the graph's launch, which runs the work. And a drop still waits for the kernels that reach its
    /// memory.
    ///
    /// The driver must be of CUDA 11.4 (470) or later, as `cuGraphInstantiateWithFlags` came with it:
    /// [`Cuda::new`](super::Cuda::new) opens an older driver all the same, and `capture` gives the error of the lookup
    /// then, before it asks the driver for anything.
    pub fn capture(&self, work: impl FnOnce() -> Result<(), Error>) -> Result<Graph, Error> {
        let api = self.driver.current()?;
        // A driver that lacks it begins no capture, and runs no work.
        let instantiate = api.instantiate.clone()?;
        let capture = Rc::new(Capture::begin(self)?);
        // The work of `work` on this stream is recorded and does not run, so none of it runs beside the work of
        // another stream. What it enqueues on another stream is as safe as it is on the null stream, as `run` says.
        let worked = Run::of(self, Some(Rc::clone(&capture))).during(work);
        let ended = capture.end();
        worked?;
        if capture.lost.get() {
            // What was recorded since the drop is the work that came after it, and no graph of the work.
            return Err(api.failed("cuStreamEndCapture", INVALIDATED));
        }
        ended?;
        // `work` may have made the context of another GPU current, and the end asks for none.
        self.driver.current()?;
        let mut exec = ptr::null_mut();
        // SAFETY: it writes the graph that it makes of the capture's to `exec`. 0 is its flags.
        api.check("cuGraphInstantiate", unsafe { instantiate(&mut exec, capture.graph.get(), 0) })?;
        Ok(Graph { driver: Rc::clone(&self.driver), exec })
    }
}

/// A capture that was begun on a stream, and what became of it: [`Stream::capture`] makes it, and the run of its
/// closure keeps it, for a drop in the closure to find. It ends the capture, and destroys what the capture made, when
/// it is dropped, so that the stream is not left capturing where `work` panics.
pub(super) struct Capture {
    driver: Rc<Driver>,
    stream: Handle,
    /// What the capture made when it ended: null before, and where it failed.
    graph: Cell<Handle>,
    /// The stream is capturing.
    on: Cell<bool>,
    /// A drop ended the capture, to make a call that the driver refuses while a capture is on: what is recorded since
    /// is the work that came after the drop, and the capture gives no graph.
    lost: Cell<bool>,
}

impl Capture {
    /// Begins a capture on `stream`, in the mode that refuses the calls of its own thread alone.
    pub(super) fn begin(stream: &Stream) -> Result<Capture, Error> {
        let capture = Capture {
            driver: Rc::clone(&stream.driver),
            stream: stream.handle,
            graph: Cell::new(ptr::null_mut()),
            on: Cell::new(false),
            lost: Cell::new(false),
        };
        capture.start()?;
        Ok(capture)
    }

    /// Begins the capture on the stream, in whichever context is current: the driver begins a capture in any context,
    /// or in none, as it ends one, and a drop of memory of another GPU begins the captures again with that GPU's
    /// current.
    fn start(&self) -> Result<(), Error> {
        let api = &self.driver.api;
        // SAFETY: the stream was made, and lasts as long as the `capture` that made this. The driver says it if it is
        // capturing already.
        api.check("cuStreamBeginCapture", unsafe { (api.begin_capture)(self.stream, THREAD_LOCAL) })?;
        self.on.set(true);
        Ok(())
    }

    /// Whether the stream is capturing.
    pub(super) fn is_on(&self) -> bool {
        self.on.get()
    }

    /// Ends the capture if it is on, and keeps the graph that the driver made of it, which it destroys with the
    /// capture. It says whether the driver could record it all. The driver ends a capture in whichever context is
    /// current, or in none, so that a drop that has another GPU's context current ends it all the same.
    fn end(&self) -> Result<(), Error> {
        if !self.on.get() {
            return Ok(());
        }
        let api = &self.driver.api;
        self.on.set(false);
        let mut graph = ptr::null_mut();
        // SAFETY: it writes the graph to `graph`. The capture was begun on this stream, on this thread, and is ended
        // once, here.
        let ended = unsafe { (api.end_capture)(self.stream, &mut graph) };
        destroy(api, self.graph.replace(graph));
        api.check("cuStreamEndCapture", ended)
    }

    /// Ends the capture, for a drop to make a call that the driver refuses while it is on. The capture is lost: what
    /// the driver says of this end is left, and `capture` gives the error of a capture that is lost.
    pub(super) fn leave(&self) {
        let _ = self.end();
        self.lost.set(true);
    }

    /// Begins the capture again, after the call that `leave` ended it for, on the stream and in the mode it had: what
    /// the closure enqueues next is recorded, and does not run. It says whether the driver did.
    pub(super) fn resume(&self) -> bool {
        self.start().is_ok()
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        // The error is no one's to take: where the capture is on still, `work` panicked, and that goes on.
        let _ = self.end();
        destroy(&self.driver.api, self.graph.get());
    }
}

/// Destroys the graph that a capture made, if it made one.
fn destroy(api: &Api, graph: Handle) {
    if !graph.is_null() {
        // SAFETY: the driver made the graph, and it is destroyed once, here. The graph that was made of it holds its
        // own copy of the work.
        unsafe { (api.destroy_graph)(graph) };
    }
}

/// The work of a stream, captured by [`Stream::capture`], which [`launch`](Graph::launch) runs as often as it is
/// asked. Dropping a graph does not stop the work that was launched: it runs to its end.
pub struct Graph {
    driver: Rc<Driver>,
    exec: Handle,
}

impl Graph {
    /// Runs the captured work on the current stream of this graph's GPU, as a launch does: after the work that stream
    /// was given before, and before the work it is given after, and the call returns once the graph is queued. A
    /// launch of a graph after another launch of it runs after that. rustc refuses a launch outside an `unsafe` block:
    ///
    /// ```compile_fail,E0133
    /// # let cuda = kernite_cuda::Cuda::new()?;
    /// let graph = cuda.stream()?.capture(|| Ok(()))?;
    /// graph.launch()?;
    /// # Ok::<(), kernite_cuda::Error>(())
    /// ```
    ///
    /// # Safety
    ///
    /// Each memory that the captured work reaches, and each module whose kernels it launches, is still there until the
    /// work is done: the graph holds their addresses, which dropping them makes dangling. The contract of the work of a
    /// stream holds at each launch: until the work is done, no work of another stream that may run beside it may write
    /// memory that it reaches, or reach memory that it writes, and the host holds no slice of memory that its kernels
    /// write, as [`Module::launch`](super::Module::launch) says.
    pub unsafe fn launch(&self) -> Result<(), Error> {
        let api = self.driver.current()?;
        // SAFETY: the caller's. The graph was made, and lasts as long as `self`, and so does the stream, which is the
        // null stream or one that a `run` of this thread has not ended.
        api.check("cuGraphLaunch", unsafe { (api.launch_graph)(self.exec, self.driver.stream()) })
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        // SAFETY: the graph was made, and is destroyed once, here. The driver does not stop it if it is running: it
        // frees it when it ends. It refuses to destroy it while a capture of this thread is on, as it refuses a free.
        self.driver.give_back(|api| unsafe { (api.destroy_graph_exec)(self.exec) });
    }
}
