//! A launcher the macros write goes to the current stream of its GPU, as `Module::launch` does, which it calls: that of
//! a kernel that is `unsafe` and that of one kernite proves safe. Two kernels share a flag: `waits` spins until it sees
//! the flag, or until a time has gone by, and `sets` writes it; and `bump`, the safe one, is launched just before
//! `sets` on the same stream. A `waits` that sees the flag ran beside the `sets` that wrote it, and one that does not
//! was kept from it. Rust builds the crate without a word, under `#![deny(warnings)]`, and the test runs the kernels on
//! the GPU if there is one.

#![deny(warnings)]

#[kernite_macros::kernels]
pub mod gpu {
    /// Spins until `*flag` is not 0, or until `nanos` nanoseconds have gone by; then writes 1 to `*out` if it saw the
    /// flag, and 2 if the time went by first. The time goes by from the first read of the clock after it moves: on the
    /// GPU here a kernel's first reads of `%globaltimer` are now and then wrong, until the clock next moves, and a
    /// start taken at one would end the wait at once.
    pub unsafe extern "ptx-kernel" fn waits(flag: *const u32, out: *mut u32, nanos: u64) {
        let (first, start, now, gone): (u64, u64, u64, u64);
        let (word, result): (u32, u32);
        let (seen, early, same): (pred, pred, pred);
        mov.u64(first, globaltimer);
        label(moves);
        mov.u64(start, globaltimer);
        setp.eq.u64(same, start, first);
        if same {
            bra(moves);
        }
        label(spin);
        ld.volatile.global.u32(word, [flag]);
        setp.ne.u32(seen, word, 0);
        if seen {
            bra(found);
        }
        mov.u64(now, globaltimer);
        sub.u64(gone, now, start);
        setp.lt.u64(early, gone, nanos);
        if early {
            bra(spin);
        }
        mov.u32(result, 2);
        st.volatile.global.u32([out], result);
        ret();
        label(found);
        mov.u32(result, 1);
        st.volatile.global.u32([out], result);
    }

    /// Writes 1 to `*flag`.
    pub unsafe extern "ptx-kernel" fn sets(flag: *mut u32) {
        let one: u32;
        mov.u32(one, 1);
        st.volatile.global.u32([flag], one);
    }

    /// `y[i] += 1` for each `i < n`: a kernel that kernite proves safe, so that its launcher is no `unsafe fn`.
    pub extern "ptx-kernel" fn bump(y: *mut u32, n: u32) {
        let (block, size, thread, i): (u32, u32, u32, u32);
        mov.u32(block, ctaid.x);
        mov.u32(size, ntid.x);
        mov.u32(thread, tid.x);
        mad.lo.u32(i, block, size, thread);
        let past: pred;
        setp.ge.u32(past, i, n);
        if past {
            ret();
        }
        let at: u64;
        mad.wide.u32(at, i, 4, y);
        let value: u32;
        ld.global.u32(value, [at]);
        add.u32(value, value, 1);
        st.global.u32([at], value);
    }
}

/// What `waits` wrote, after `waits` was launched in the stream `waits_on`, and `bump` and then `sets` in `sets_on`,
/// each in no stream, the null stream's, where it is `None`. `waits` waits `nanos` nanoseconds: where it must see the
/// flag, a second, much more than `sets` takes to run beside it, and it returns as soon as it sees it; and where it
/// must not, a tenth of a second, as a `sets` that ran beside it would write the flag at once.
#[cfg(test)]
fn race(
    cuda: &kernite_cuda::Cuda,
    kernels: &gpu::Kernels,
    nanos: u64,
    waits_on: Option<&kernite_cuda::Stream>,
    sets_on: Option<&kernite_cuda::Stream>,
) -> u32 {
    let (flag, out) = (cuda.alloc::<u32>(1).unwrap(), cuda.alloc::<u32>(1).unwrap());
    let mut ys = cuda.upload(&[1u32, 2, 3]).unwrap();
    let (one, many) = (kernite_cuda::Grid::new(1, 1), kernite_cuda::Grid::new(1, 32));
    // SAFETY: the kernels share one word, the flag that `sets` writes and `waits` reads, each by one volatile access
    // that the GPU does whole, which is what the test looks for; `waits` writes its own `out`, `bump` its own `ys`, and
    // they reach no other memory in common. The pointers are of buffers that last until the kernels are done, which
    // `sync` waits for.
    unsafe {
        let launch_waits = || kernels.waits(one, flag.ptr(), out.ptr(), nanos).unwrap();
        let mut launch_sets = || {
            kernels.bump(many, &mut ys, 3).unwrap();
            kernels.sets(one, flag.ptr()).unwrap();
        };
        match waits_on {
            Some(stream) => stream.run(launch_waits),
            None => launch_waits(),
        }
        match sets_on {
            Some(stream) => stream.run(launch_sets),
            None => launch_sets(),
        }
    }
    cuda.sync().unwrap();
    assert_eq!(ys.read().unwrap(), [2, 3, 4], "bump did not run");
    out.read().unwrap()[0]
}

#[test]
fn a_launcher_goes_to_the_current_stream_and_to_the_null_stream_where_there_is_none() {
    const A_SECOND: u64 = 1_000_000_000;
    const A_TENTH: u64 = 100_000_000;
    let Ok(cuda) = kernite_cuda::Cuda::new().inspect_err(|e| eprintln!("no GPU: skipped. {e}")) else { return };
    let kernels = gpu::Kernels::load(&cuda).unwrap();
    let (a, b) = (cuda.stream().unwrap(), cuda.stream().unwrap());
    // `waits` on `a` sees the flag that `sets` writes on `b`: the launcher put each on its stream.
    assert_eq!(race(&cuda, &kernels, A_SECOND, Some(&a), Some(&b)), 1, "two streams");
    // `sets` with no stream is on the null stream, which waits for `a`'s `waits`: the time goes by.
    assert_eq!(race(&cuda, &kernels, A_TENTH, Some(&a), None), 2, "a stream, then the null stream");
}
