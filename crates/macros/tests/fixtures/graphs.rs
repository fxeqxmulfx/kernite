//! The launcher of a GPU kernel that the macros write is recorded by `Stream::capture`, whether the kernel is `unsafe`
//! or one kernite proves safe, and the graph does what the launches do one by one: `bump` is the safe kernel and
//! `double` the `unsafe` one, which do not commute, so the order of the two shows. Rust builds the crate without a
//! word, under `#![deny(warnings)]`, and the test runs the kernels on the GPU if there is one.

#![deny(warnings)]

#[kernite_macros::kernels]
pub mod gpu {
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

    /// `y[i] *= 2` for each `i < n`: the same, but a kernel that is an `unsafe fn`, whose launcher takes the pointer.
    pub unsafe extern "ptx-kernel" fn double(y: *mut u32, n: u32) {
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
        add.u32(value, value, value);
        st.global.u32([at], value);
    }
}

#[test]
fn a_graph_of_two_launchers_does_what_their_launches_do_one_by_one() {
    let Ok(cuda) = kernite_cuda::Cuda::new().inspect_err(|e| eprintln!("no GPU: skipped. {e}")) else { return };
    let kernels = gpu::Kernels::load(&cuda).unwrap();
    let (stream, grid) = (cuda.stream().unwrap(), kernite_cuda::Grid::new(1, 32));
    let (mut each, mut graphed) = (cuda.upload(&[1u32, 2, 3]).unwrap(), cuda.upload(&[1u32, 2, 3]).unwrap());
    // Three rounds, one launch after another.
    for _ in 0..3 {
        kernels.bump(grid, &mut each, 3).unwrap();
        // SAFETY: `each` has 3 words, which `double` writes by thread, inside them, and nothing else reaches them.
        unsafe { kernels.double(grid, each.ptr(), 3) }.unwrap();
    }
    // A round recorded once, and launched three times: the capture runs nothing.
    let graph = stream
        .capture(|| {
            kernels.bump(grid, &mut graphed, 3)?;
            // SAFETY: as above, for `graphed`, which lasts as long as the graph is launched.
            unsafe { kernels.double(grid, graphed.ptr(), 3) }
        })
        .unwrap();
    assert_eq!(graphed.read().unwrap(), [1, 2, 3], "the capture ran a launch");
    for _ in 0..3 {
        // SAFETY: `graphed` lasts until the end of the test, the host holds no slice of it, and no other stream's work
        // reaches it: the graph's work runs on the null stream, one launch after another.
        unsafe { graph.launch() }.unwrap();
    }
    assert_eq!(graphed.read().unwrap(), each.read().unwrap(), "the graph did what the launches did one by one");
    assert_eq!(each.read().unwrap(), [22, 30, 38], "three rounds of `(y + 1) * 2`");
}
