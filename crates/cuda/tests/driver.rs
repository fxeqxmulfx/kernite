//! The driver as kernite calls it, on the GPU here, if there is one.

use std::panic::{AssertUnwindSafe, catch_unwind};

use kernite_cuda::{Buffer, Cuda, Grid, Module, arg, bf16, f16};

/// `x[i] += 1` for each `i < n`, a thread each, written by hand.
const PTX: &str = "
.version 7.5
.target sm_61
.address_size 64

.visible .entry add_one(.param .u64 x_param, .param .u32 n_param)
{
	.reg .pred %p;
	.reg .u32 %n, %i, %block, %size, %thread;
	.reg .f32 %v;
	.reg .u64 %x, %at;

	ld.param.u64 %x, [x_param];
	cvta.to.global.u64 %x, %x;
	ld.param.u32 %n, [n_param];
	mov.u32 %block, %ctaid.x;
	mov.u32 %size, %ntid.x;
	mov.u32 %thread, %tid.x;
	mad.lo.u32 %i, %block, %size, %thread;
	setp.ge.u32 %p, %i, %n;
	@%p bra done;
	mad.wide.u32 %at, %i, 4, %x;
	ld.global.f32 %v, [%at];
	add.rn.f32 %v, %v, 0f3F800000;
	st.global.f32 [%at], %v;
done:
	ret;
}
";

/// The GPU, or `None` and why not.
fn cuda() -> Option<Cuda> {
    Cuda::new().map_err(|e| eprintln!("no GPU: skipped. {e}")).ok()
}

#[test]
fn a_kernel_runs_on_what_was_uploaded() {
    let Some(cuda) = cuda() else { return };
    let xs: Vec<f32> = (0..1000).map(|i| i as f32 / 3.0).collect();
    let buf = cuda.upload(&xs).unwrap();
    let module = Module::load(&cuda, PTX, &["add_one"], &[]).unwrap();
    // Threads past the first 999, to see the bounds checked.
    let (x, n) = (buf.ptr(), 999u32);
    // SAFETY: `add_one` takes a pointer and a `u32`, and writes only the first `n` of the
    // 1000 the buffer has.
    unsafe { module.launch(0, Grid::new(4, 256), &[arg(&x), arg(&n)]) }.unwrap();
    cuda.sync().unwrap();
    let want: Vec<f32> = xs.iter().enumerate().map(|(i, x)| if i < 999 { x + 1.0 } else { *x }).collect();
    assert_eq!(buf.read().unwrap(), want);
}

#[test]
fn a_buffer_is_zero_until_written_and_reads_back_what_was() {
    let Some(cuda) = cuda() else { return };
    let mut buf = cuda.alloc::<u32>(5).unwrap();
    assert_eq!(buf.read().unwrap(), [0; 5]);
    buf.write(&[1, 2, 3, 4, u32::MAX]).unwrap();
    let mut out = [0; 5];
    buf.read_into(&mut out).unwrap();
    assert_eq!(out, [1, 2, 3, 4, u32::MAX]);
    buf.zero().unwrap();
    assert_eq!(buf.read().unwrap(), [0; 5]);
    assert_eq!(buf.at(2).addr(), buf.ptr().addr() + 8);
    assert!(cuda.alloc::<f32>(0).unwrap().is_empty());
}

#[test]
fn ptx_the_driver_will_not_compile_is_logged() {
    let Some(cuda) = cuda() else { return };
    let e = Module::load(&cuda, &PTX.replace("add.rn.f32", "add.rn.f99"), &["add_one"], &[]).err().unwrap();
    assert_eq!(e.call, "cuModuleLoadDataEx");
    assert!(e.message.starts_with("CUDA_ERROR_INVALID_PTX"), "{e}");
    assert!(e.log.contains("f99"), "{e}");
}

#[test]
fn a_kernel_the_module_has_not_is_named() {
    let Some(cuda) = cuda() else { return };
    let e = Module::load(&cuda, PTX, &["add_one", "add_two"], &[]).err().unwrap();
    assert_eq!(e.call, "cuModuleGetFunction");
    assert!(e.message.starts_with("CUDA_ERROR_NOT_FOUND") && e.message.ends_with("for `add_two`"), "{e}");
}

/// Each block's `ntid` words of `x` reversed through the shared memory the launch gives, a
/// thread each; and `last`, which writes and reads back the last word of the `bytes` the
/// launch gives, into `x[0]`.
const SHARED: &str = "
.version 7.5
.target sm_61
.address_size 64

.extern .shared .align 4 .b8 buf[];

.visible .entry reverse(.param .u64 x_param)
{
	.reg .u32 %t, %n, %i, %j, %s, %v;
	.reg .u64 %x, %at;

	ld.param.u64 %x, [x_param];
	cvta.to.global.u64 %x, %x;
	mov.u32 %t, %tid.x;
	mov.u32 %n, %ntid.x;
	mov.u32 %i, %ctaid.x;
	mad.lo.u32 %i, %i, %n, %t;
	mad.wide.u32 %at, %i, 4, %x;
	ld.global.u32 %v, [%at];
	mov.u32 %s, buf;
	mad.lo.u32 %j, %t, 4, %s;
	st.shared.u32 [%j], %v;
	bar.sync 0;
	sub.u32 %j, %n, %t;
	sub.u32 %j, %j, 1;
	mad.lo.u32 %j, %j, 4, %s;
	ld.shared.u32 %v, [%j];
	st.global.u32 [%at], %v;
	ret;
}

.visible .entry last(.param .u64 x_param, .param .u32 bytes_param)
{
	.reg .u32 %b, %s, %v;
	.reg .u64 %x;

	ld.param.u64 %x, [x_param];
	cvta.to.global.u64 %x, %x;
	ld.param.u32 %b, [bytes_param];
	mov.u32 %s, buf;
	add.u32 %s, %s, %b;
	st.shared.u32 [%s+-4], %b;
	ld.shared.u32 %v, [%s+-4];
	st.global.u32 [%x], %v;
	ret;
}
";

#[test]
fn a_launch_gives_the_shared_memory_it_says() {
    let Some(cuda) = cuda() else { return };
    let module = Module::load(&cuda, SHARED, &["reverse", "last"], &[]).unwrap();
    let buf = cuda.upload(&(0..512).collect::<Vec<u32>>()).unwrap();
    let x = buf.ptr();
    // SAFETY: `reverse` takes a pointer, and reads and writes there the 256 words of each of
    // the two blocks, through the 1 KB each is given.
    unsafe { module.launch(0, Grid::new(2, 256).with_shared(1024), &[arg(&x)]) }.unwrap();
    let want: Vec<u32> = (0..512).map(|i| i / 256 * 256 + 255 - i % 256).collect();
    assert_eq!(buf.read().unwrap(), want);
    // As much as a block may have, which from Volta on is more than the 48 KB a launch
    // gives unasked; and then a word more, which the driver refuses.
    let most = cuda.max_shared();
    assert!(most >= 48 << 10, "{most}");
    // SAFETY: `last` takes a pointer and a `u32`, and writes the first word there, and the
    // last of the bytes it is given in shared memory.
    unsafe { module.launch(1, Grid::new(1, 1).with_shared(most), &[arg(&x), arg(&most)]) }.unwrap();
    assert_eq!(buf.read().unwrap()[0], most);
    let more = most + 4;
    // SAFETY: as before.
    let e = unsafe { module.launch(1, Grid::new(1, 1).with_shared(more), &[arg(&x), arg(&more)]) }.unwrap_err();
    assert!(e.message.starts_with("CUDA_ERROR_INVALID_VALUE"), "{e}");
    cuda.sync().unwrap();
}

/// The floats of 16 bits as the GPU's `cvt` converts them, a thread each: `widen` from `f16`
/// to `f32`, and `narrow` from `f32` to `f16` and to `bf16`, rounding to the nearest.
const HALF: &str = "
.version 7.5
.target sm_80
.address_size 64

.visible .entry widen(.param .u64 h_param, .param .u64 y_param, .param .u32 n_param)
{
	.reg .pred %p;
	.reg .u32 %n, %i, %block, %size, %thread;
	.reg .b16 %h;
	.reg .f32 %f;
	.reg .u64 %hs, %y, %at;

	ld.param.u64 %hs, [h_param];
	cvta.to.global.u64 %hs, %hs;
	ld.param.u64 %y, [y_param];
	cvta.to.global.u64 %y, %y;
	ld.param.u32 %n, [n_param];
	mov.u32 %block, %ctaid.x;
	mov.u32 %size, %ntid.x;
	mov.u32 %thread, %tid.x;
	mad.lo.u32 %i, %block, %size, %thread;
	setp.ge.u32 %p, %i, %n;
	@%p bra done;
	mad.wide.u32 %at, %i, 2, %hs;
	ld.global.b16 %h, [%at];
	cvt.f32.f16 %f, %h;
	mad.wide.u32 %at, %i, 4, %y;
	st.global.f32 [%at], %f;
done:
	ret;
}

.visible .entry narrow(.param .u64 x_param, .param .u64 h_param, .param .u64 b_param, .param .u32 n_param)
{
	.reg .pred %p;
	.reg .u32 %n, %i, %block, %size, %thread;
	.reg .f32 %f;
	.reg .b16 %h, %b;
	.reg .u64 %x, %hs, %bs, %at;

	ld.param.u64 %x, [x_param];
	cvta.to.global.u64 %x, %x;
	ld.param.u64 %hs, [h_param];
	cvta.to.global.u64 %hs, %hs;
	ld.param.u64 %bs, [b_param];
	cvta.to.global.u64 %bs, %bs;
	ld.param.u32 %n, [n_param];
	mov.u32 %block, %ctaid.x;
	mov.u32 %size, %ntid.x;
	mov.u32 %thread, %tid.x;
	mad.lo.u32 %i, %block, %size, %thread;
	setp.ge.u32 %p, %i, %n;
	@%p bra done;
	mad.wide.u32 %at, %i, 4, %x;
	ld.global.f32 %f, [%at];
	cvt.rn.f16.f32 %h, %f;
	cvt.rn.bf16.f32 %b, %f;
	mad.wide.u32 %at, %i, 2, %hs;
	st.global.b16 [%at], %h;
	mad.wide.u32 %at, %i, 2, %bs;
	st.global.b16 [%at], %b;
done:
	ret;
}
";

/// Each finite float of 16 bits at or above zero as `f32`, `to_f32` of its bits, and the
/// next: the points a conversion rounds between. Past the largest, the next is where its
/// exponent would carry to, a step on: 2^16 for `f16`, 2^128 for `bf16`.
fn steps(last: u16, to_f32: fn(u16) -> f32) -> impl Iterator<Item = (f32, f64)> {
    (0..last).map(move |bits| {
        let (x, next) = (f64::from(to_f32(bits)), f64::from(to_f32(bits + 1)));
        (to_f32(bits), if next.is_finite() { next } else { 2.0 * x - f64::from(to_f32(bits - 1)) })
    })
}

#[test]
fn floats_of_16_bits_convert_as_the_gpu_converts_them() {
    let Some(cuda) = cuda() else { return };
    if cuda.sm() < 80 {
        eprintln!("{}, sm_{}, has no `cvt.rn.bf16.f32`: skipped", cuda.name(), cuda.sm());
        return;
    }
    let module = Module::load(&cuda, HALF, &["widen", "narrow"], &[]).unwrap();
    let grid = |n: usize| Grid::new(n.div_ceil(256) as u32, 256);
    // Every f16.
    let all: Vec<f16> = (0..=u16::MAX).map(f16::from_bits).collect();
    let (hs, ys) = (cuda.upload(&all).unwrap(), cuda.alloc::<f32>(all.len()).unwrap());
    let n = all.len() as u32;
    // SAFETY: `widen` takes two pointers and a `u32`, and reads `n` halves at the first and
    // writes `n` floats at the second.
    unsafe { module.launch(0, grid(all.len()), &[arg(&hs.ptr()), arg(&ys.ptr()), arg(&n)]) }.unwrap();
    for (h, y) in all.iter().zip(ys.read().unwrap()) {
        match y.is_nan() {
            true => assert!(h.to_f32().is_nan(), "{:#06x}", h.to_bits()),
            false => assert_eq!(h.to_f32().to_bits(), y.to_bits(), "{:#06x}", h.to_bits()),
        }
    }
    // Each value of either, the points halfway to the next, an f32 to either side of those,
    // and of either sign; then f32s at random, and those at the ends.
    let mut xs = vec![];
    let f16s = steps(0x7c00, |b| f16::from_bits(b).to_f32());
    let bf16s = steps(0x7f80, |b| bf16::from_bits(b).to_f32());
    for (x, next) in f16s.chain(bf16s) {
        let half = ((f64::from(x) + next) / 2.0) as f32;
        for y in [x, half, f32::from_bits(half.to_bits() - 1), f32::from_bits(half.to_bits() + 1)] {
            xs.extend([y, -y]);
        }
    }
    let mut seed = 0x2545_f491_u32;
    xs.extend((0..1 << 20).map(|_| {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        f32::from_bits(seed)
    }));
    xs.extend([0.0, f32::INFINITY, f32::NAN, f32::MAX, f32::MIN_POSITIVE, f32::from_bits(1)]);
    assert!(xs.contains(&65520.0) && xs.contains(&f32::from_bits(0x7f7f_8000)), "the ties at the largest");
    let x = cuda.upload(&xs).unwrap();
    let (h, b) = (cuda.alloc::<f16>(xs.len()).unwrap(), cuda.alloc::<bf16>(xs.len()).unwrap());
    let n = xs.len() as u32;
    // SAFETY: `narrow` takes three pointers and a `u32`, and reads `n` floats at the first,
    // and writes `n` halves at each of the others.
    unsafe { module.launch(1, grid(xs.len()), &[arg(&x.ptr()), arg(&h.ptr()), arg(&b.ptr()), arg(&n)]) }.unwrap();
    let (h, b) = (h.read().unwrap(), b.read().unwrap());
    for (i, &x) in xs.iter().enumerate() {
        let (h, b) = (h[i].to_bits(), b[i].to_bits());
        if x.is_nan() {
            assert!(
                f16::from_bits(h).to_f32().is_nan() && bf16::from_bits(b).to_f32().is_nan(),
                "{:#010x}",
                x.to_bits()
            );
            continue;
        }
        assert_eq!(f16::from_f32(x).to_bits(), h, "f16 of {x:e}, {:#010x}", x.to_bits());
        assert_eq!(bf16::from_f32(x).to_bits(), b, "bf16 of {x:e}, {:#010x}", x.to_bits());
    }
}

/// What a kernel of the parameters of fewer bits and of `f64` is given, written to `out`: an
/// `f16`, a `bf16`, a `u8`, an `i16` and an `f64`.
const PARAMS: &str = "
.version 7.5
.target sm_61
.address_size 64

.visible .entry params(.param .u64 out_param, .param .b16 h_param, .param .b16 b_param, .param .u8 c_param, .param .s16 s_param, .param .f64 d_param)
{
	.reg .b16 %h, %b;
	.reg .u16 %c;
	.reg .s16 %s;
	.reg .f64 %d;
	.reg .u64 %out;

	ld.param.u64 %out, [out_param];
	cvta.to.global.u64 %out, %out;
	ld.param.b16 %h, [h_param];
	ld.param.b16 %b, [b_param];
	ld.param.u8 %c, [c_param];
	ld.param.s16 %s, [s_param];
	ld.param.f64 %d, [d_param];
	st.global.b16 [%out], %h;
	st.global.b16 [%out+2], %b;
	st.global.u8 [%out+4], %c;
	st.global.s16 [%out+6], %s;
	st.global.f64 [%out+8], %d;
	ret;
}
";

#[test]
fn a_kernel_is_given_the_bits_of_each_parameter() {
    let Some(cuda) = cuda() else { return };
    let module = Module::load(&cuda, PARAMS, &["params"], &[]).unwrap();
    let out = cuda.alloc::<u8>(16).unwrap();
    let (h, b, c, s, d) = (f16::from_f32(1.5), bf16::from_f32(-2.25), 0xa5u8, -12345i16, 1.0e300f64);
    let args = [arg(&out.ptr()), arg(&h), arg(&b), arg(&c), arg(&s), arg(&d)];
    // SAFETY: `params` takes these, and writes 16 bytes at the pointer.
    unsafe { module.launch(0, Grid::new(1, 1), &args) }.unwrap();
    let got = out.read().unwrap();
    let mut want = vec![0u8; 16];
    want[0..2].copy_from_slice(&h.to_bits().to_le_bytes());
    want[2..4].copy_from_slice(&b.to_bits().to_le_bytes());
    want[4] = c;
    want[6..8].copy_from_slice(&s.to_le_bytes());
    want[8..16].copy_from_slice(&d.to_le_bytes());
    assert_eq!(got, want);
}

/// Memory of the module's own, which its kernels name, and which keeps what they write from one
/// launch to the next: `count` adds 1 to word `i` of `counts` in each thread, and `copy` writes
/// the two words of `seen` and the two floats of `table` to `out`. `counts` is the module's
/// alone, as kernite declares its memory, and `seen` visible, as nvcc declares it for a program
/// linked in parts.
const GLOBALS: &str = "
.version 7.5
.target sm_61
.address_size 64

.global .align 4 .b8 counts[16];
.visible .global .align 4 .b8 seen[8] = {7, 0, 0, 0, 9};
.const .align 4 .b8 table[8] = {0, 0, 128, 63, 0, 0, 0, 64};

.visible .entry count(.param .u32 i_param)
{
	.reg .u32 %i;
	.reg .u64 %base, %at;

	ld.param.u32 %i, [i_param];
	mov.u64 %base, counts;
	mad.wide.u32 %at, %i, 4, %base;
	red.global.add.u32 [%at], 1;
	ret;
}

.visible .entry copy(.param .u64 out_param)
{
	.reg .u32 %a, %b;
	.reg .f32 %x, %y;
	.reg .u64 %out;

	ld.param.u64 %out, [out_param];
	cvta.to.global.u64 %out, %out;
	ld.global.u32 %a, [seen];
	ld.global.u32 %b, [seen+4];
	ld.const.f32 %x, [table];
	ld.const.f32 %y, [table+4];
	st.global.u32 [%out], %a;
	st.global.u32 [%out+4], %b;
	st.global.f32 [%out+8], %x;
	st.global.f32 [%out+12], %y;
	ret;
}
";

#[test]
fn a_module_s_memory_keeps_what_is_written_from_launch_to_launch() {
    let Some(cuda) = cuda() else { return };
    let module = Module::load(&cuda, GLOBALS, &["count", "copy"], &["counts", "seen", "table"]).unwrap();
    let (counts, seen, table) = (module.global::<u32>(0), module.global::<u32>(1), module.global::<f32>(2));
    // What the PTX declares: zeros, or what it is initialized to, and zeros after.
    assert_eq!(counts.read().unwrap(), [0; 4]);
    assert_eq!(seen.read().unwrap(), [7, 9]);
    assert_eq!(table.read().unwrap(), [1.0, 2.0]);
    let count = |i: u32, threads: u32| {
        // SAFETY: `count` takes a `u32`, and adds to word `i` of the four of `counts`.
        unsafe { module.launch(0, Grid::new(1, threads), &[arg(&i)]) }.unwrap()
    };
    count(2, 64);
    count(2, 32);
    count(0, 3);
    assert_eq!(counts.read().unwrap(), [3, 0, 96, 0]);
    counts.write(&[10, 20, 30, 40]).unwrap();
    count(1, 1);
    assert_eq!(counts.read().unwrap(), [10, 21, 30, 40]);
    counts.zero().unwrap();
    count(3, 5);
    assert_eq!(counts.read().unwrap(), [0, 0, 0, 5]);
    // Constant memory, which its kernels only read, the host writes.
    seen.write(&[5, 6]).unwrap();
    table.write(&[3.0, 4.0]).unwrap();
    // SAFETY: `copy` takes a pointer, and writes 16 bytes there, as many as `counts` has.
    unsafe { module.launch(1, Grid::new(1, 1), &[arg(&counts.ptr())]) }.unwrap();
    assert_eq!(counts.read().unwrap(), [5, 6, 3.0f32.to_bits(), 4.0f32.to_bits()]);
    // Its bytes as elements of any size they are a whole number of.
    assert_eq!((module.global::<u8>(1).len(), module.global::<u64>(0).len()), (8, 2));
    assert_eq!(counts.at(3).addr(), counts.ptr().addr() + 12);
}

#[test]
fn memory_the_module_has_not_is_named() {
    let Some(cuda) = cuda() else { return };
    let e = Module::load(&cuda, GLOBALS, &[], &["counts", "nothing"]).err().unwrap();
    assert_eq!(e.call, "cuModuleGetGlobal");
    assert!(e.message.starts_with("CUDA_ERROR_NOT_FOUND") && e.message.ends_with("for `nothing`"), "{e}");
}

/// Where the driver's workspace for the grid is, as `%envreg1:%envreg2` give its address, high
/// word first, which thread 0 of each block writes to `at[block]`: a cooperative launch has
/// one, whose second word is the grid's barrier. Another launch finds none, null, until the
/// context has had a cooperative one, and then that one's.
const WORKSPACE: &str = "
.version 5.0
.target sm_60
.address_size 64

.visible .entry workspace(.param .u64 at_param)
{
	.reg .pred %p;
	.reg .u32 %hi, %lo, %t, %b;
	.reg .u64 %at, %ws;

	ld.param.u64 %at, [at_param];
	cvta.to.global.u64 %at, %at;
	mov.u32 %t, %tid.x;
	setp.ne.u32 %p, %t, 0;
	@%p bra done;
	mov.u32 %hi, %envreg1;
	mov.u32 %lo, %envreg2;
	mov.b64 %ws, {%lo, %hi};
	mov.u32 %b, %ctaid.x;
	mad.wide.u32 %at, %b, 8, %at;
	st.global.u64 [%at], %ws;
done:
	ret;
}
";

#[test]
fn a_cooperative_launch_has_a_workspace_for_the_grid() {
    let Some(cuda) = cuda() else { return };
    if !cuda.cooperative() {
        eprintln!("{} launches no kernel cooperatively: skipped", cuda.name());
        return;
    }
    let module = Module::load(&cuda, WORKSPACE, &["workspace"], &[]).unwrap();
    let at = cuda.alloc::<u64>(3).unwrap();
    // SAFETY: `workspace` takes a pointer, and writes a word of 8 bytes there for each block.
    unsafe { module.launch_cooperative(0, Grid::new(3, 64), &[arg(&at.ptr())]) }.unwrap();
    let ws = at.read().unwrap();
    // An address on the GPU, which the barrier after its first word is aligned in.
    assert!(ws[0] != 0 && ws[0].is_multiple_of(8) && ws.iter().all(|&w| w == ws[0]), "{ws:#x?}");
}

#[test]
fn a_cooperative_launch_has_at_most_the_blocks_the_gpu_runs_at_once() {
    let Some(cuda) = cuda() else { return };
    if !cuda.cooperative() {
        eprintln!("{} launches no kernel cooperatively: skipped", cuda.name());
        return;
    }
    let module = Module::load(&cuda, WORKSPACE, &["workspace"], &[]).unwrap();
    let most = module.resident(0, 256, 0).unwrap();
    // Every SM holds a block of 256 threads, and none more than 2048 threads.
    let sms = cuda.multiprocessors();
    assert!((sms..=sms * 8).contains(&most), "{most} blocks on {sms} SMs");
    // Fewer threads, more blocks, up to as many as an SM holds.
    assert!(module.resident(0, 32, 0).unwrap() > most);
    let mut at = cuda.alloc::<u64>(most as usize + 1).unwrap();
    // SAFETY: `workspace` takes a pointer, and writes a word of 8 bytes there for each block.
    unsafe { module.launch_cooperative(0, Grid::new(most, 256), &[arg(&at.ptr())]) }.unwrap();
    cuda.sync().unwrap();
    // SAFETY: as above.
    let e = unsafe { module.launch_cooperative(0, Grid::new(most + 1, 256), &[arg(&at.ptr())]) }.unwrap_err();
    assert_eq!(e.call, "cuLaunchCooperativeKernel");
    assert!(e.message.starts_with("CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE (720): "), "{e}");
    let tail = format!(": the grid has {} blocks, and the GPU runs {most} of these at once", most + 1);
    assert!(e.message.ends_with(&tail), "{e}");
    // The launch was refused, and the GPU goes on.
    cuda.sync().unwrap();
    at.zero().unwrap();
    // SAFETY: as above.
    unsafe { module.launch_cooperative(0, Grid::new(most, 256), &[arg(&at.ptr())]) }.unwrap();
    let ws = at.read().unwrap();
    assert!(ws[0] != 0 && ws[..most as usize].iter().all(|&w| w == ws[0]) && ws[most as usize] == 0, "{ws:#x?}");
}

/// Stores `ctaid.x` of the last block, the one whose number is one less than the blocks along
/// x, in `out[0]`, in a module written for `sm_30`: with `sm_20` for it, the same kernel.
const LAST: &str = "
.version 3.0
.target sm_30
.address_size 64

.visible .entry last(.param .u64 out_param)
{
	.reg .pred %other;
	.reg .u32 %block, %blocks;
	.reg .u64 %out;

	ld.param.u64 %out, [out_param];
	cvta.to.global.u64 %out, %out;
	mov.u32 %block, %ctaid.x;
	mov.u32 %blocks, %nctaid.x;
	sub.u32 %blocks, %blocks, 1;
	setp.ne.u32 %other, %block, %blocks;
	@%other bra done;
	st.global.u32 [%out], %block;
done:
	ret;
}
";

/// The driver starts at most 65535 blocks along x of a module written for `sm_20` or `sm_21`,
/// whatever the GPU, and as many as the GPU takes of one written for `sm_30`, which is why a
/// module of kernite's is written for `sm_30` at the least. The `.target` says it, not the
/// `.version`.
#[test]
fn the_driver_starts_65535_blocks_of_a_module_for_fermi_and_more_of_one_for_kepler() {
    let Some(cuda) = cuda() else { return };
    let mut out = cuda.alloc::<u32>(1).unwrap();
    let at = out.ptr();
    for (target, most) in [("sm_20", 65535), ("sm_21", 65535), ("sm_30", 65536)] {
        let module = Module::load(&cuda, &LAST.replace("sm_30", target), &["last"], &[]).unwrap();
        for blocks in [1, 65535, 65536] {
            out.zero().unwrap();
            // SAFETY: `last` takes a pointer, and writes a word there.
            let launched = unsafe { module.launch(0, Grid::new(blocks, 1), &[arg(&at)]) };
            if blocks <= most {
                launched.unwrap();
                assert_eq!(out.read().unwrap(), [blocks - 1], "{blocks} blocks for {target}");
            } else {
                let e = launched.unwrap_err();
                assert_eq!(e.call, "cuLaunchKernel");
                // The driver's words alone: no size of the launch, its shared memory of none among
                // them, is past what the GPU takes.
                assert_eq!(e.message, "CUDA_ERROR_INVALID_VALUE (1): invalid argument", "{blocks} blocks for {target}");
            }
        }
    }
    cuda.sync().unwrap();
}

/// Each grid, and what a launch over it says: nothing, for one the GPU takes, at the least and at
/// the most it takes of each size; or which size is past what the GPU takes, the first that is.
fn grids() -> Vec<(Grid, Option<&'static str>)> {
    let grid = |blocks, threads| Grid { blocks, threads, shared: 0 };
    let blocks_y = "the grid has 65536 blocks along y, and the GPU takes 65535 at most";
    let blocks_z = "the grid has 65536 blocks along z, and the GPU takes 65535 at most";
    let blocks_x = "the grid has 2147483648 blocks along x, and the GPU takes 2147483647 at most";
    let no_blocks_z = "the grid has 0 blocks along z, and the GPU takes 1 at least";
    vec![
        (grid([1, 1, 1], [1, 1, 1]), None),
        (grid([1, 65535, 1], [1, 1, 1]), None),
        (grid([1, 1, 65535], [1, 1, 1]), None),
        (grid([1, 1, 1], [1024, 1, 1]), None),
        (grid([1, 1, 1], [1, 1024, 1]), None),
        (grid([1, 1, 1], [1, 1, 64]), None),
        (grid([1, 1, 1], [32, 32, 1]), None),
        (grid([1, 1, 1], [16, 16, 4]), None),
        (grid([1, 65536, 1], [1, 1, 1]), Some(blocks_y)),
        (grid([1, 1, 65536], [1, 1, 1]), Some(blocks_z)),
        (grid([1 << 31, 1, 1], [1, 1, 1]), Some(blocks_x)),
        (grid([1, 1, 1], [1025, 1, 1]), Some("each block has 1025 threads along x, and the GPU takes 1024 at most")),
        (grid([1, 1, 1], [1, 1025, 1]), Some("each block has 1025 threads along y, and the GPU takes 1024 at most")),
        (grid([1, 1, 1], [1, 1, 65]), Some("each block has 65 threads along z, and the GPU takes 64 at most")),
        (grid([1, 1, 1], [1 << 22; 3]), Some("each block has 4194304 threads along x, and the GPU takes 1024 at most")),
        (grid([1, 1, 1], [33, 32, 1]), Some("each block has 1056 threads, and the GPU takes 1024 at most")),
        (grid([1, 1, 1], [16, 16, 5]), Some("each block has 1280 threads, and the GPU takes 1024 at most")),
        (grid([1, 1, 1], [1024, 1024, 64]), Some("each block has 67108864 threads, and the GPU takes 1024 at most")),
        (grid([0, 1, 1], [1, 1, 1]), Some("the grid has 0 blocks along x, and the GPU takes 1 at least")),
        (grid([1, 0, 1], [1, 1, 1]), Some("the grid has 0 blocks along y, and the GPU takes 1 at least")),
        (grid([1, 1, 0], [1, 1, 1]), Some(no_blocks_z)),
        (grid([1, 1, 1], [0, 1, 1]), Some("each block has 0 threads along x, and the GPU takes 1 at least")),
        (grid([1, 1, 1], [1, 0, 1]), Some("each block has 0 threads along y, and the GPU takes 1 at least")),
        (grid([1, 1, 1], [1, 1, 0]), Some("each block has 0 threads along z, and the GPU takes 1 at least")),
        // The first size past is the one said, whichever limit it is past: the blocks' before the
        // threads', x before y before z.
        (grid([1, 65536, 65536], [1025, 1, 1]), Some(blocks_y)),
        (grid([1, 65536, 0], [1, 1, 1]), Some(blocks_y)),
        (grid([1, 1, 0], [1025, 1, 1]), Some(no_blocks_z)),
    ]
}

/// What a launch over a grid the GPU does not take says: the driver's `CUDA_ERROR_INVALID_VALUE`,
/// and which size is past which limit.
fn refusal(call: &str, e: &kernite_cuda::Error, said: &str) {
    assert_eq!(e.call, call);
    assert!(e.message.starts_with("CUDA_ERROR_INVALID_VALUE (1): "), "{e}");
    assert!(e.message.ends_with(&format!(": {said}")), "{e}");
}

#[test]
fn a_launch_over_a_grid_the_gpu_does_not_take_says_which_size_is_past_which_limit() {
    let Some(cuda) = cuda() else { return };
    let module = Module::load(&cuda, LAST, &["last"], &[]).unwrap();
    let out = cuda.alloc::<u32>(1).unwrap();
    let at = out.ptr();
    for (grid, said) in grids() {
        // SAFETY: `last` takes a pointer, and writes a word there.
        let launched = unsafe { module.launch(0, grid, &[arg(&at)]) };
        match (launched, said) {
            (Ok(()), None) => {}
            (Err(e), Some(said)) => refusal("cuLaunchKernel", &e, said),
            (launched, said) => panic!("{grid:?}: {launched:?}, and it should say {said:?}"),
        }
    }
    // The launches refused did not harm the GPU, nor the ones that started.
    cuda.sync().unwrap();
}

#[test]
fn a_cooperative_launch_over_a_grid_the_gpu_does_not_take_says_it_too() {
    let Some(cuda) = cuda() else { return };
    if !cuda.cooperative() {
        eprintln!("{} launches no kernel cooperatively: skipped", cuda.name());
        return;
    }
    let module = Module::load(&cuda, LAST, &["last"], &[]).unwrap();
    let out = cuda.alloc::<u32>(1).unwrap();
    let at = out.ptr();
    // Only the grids the GPU does not take: one it takes may have more blocks than it runs at
    // once, which is refused in other words.
    for (grid, said) in grids().into_iter().filter_map(|(grid, said)| said.map(|said| (grid, said))) {
        // SAFETY: `last` takes a pointer, and writes a word there.
        let e = unsafe { module.launch_cooperative(0, grid, &[arg(&at)]) }.unwrap_err();
        refusal("cuLaunchCooperativeKernel", &e, said);
    }
    cuda.sync().unwrap();
}

/// A kernel of 20 KB of shared memory of its own, `both`, and one of none, `plain`, each given
/// the launch's beside it, `bytes` of it. Each writes the last word of its own, if it has any,
/// and the last of the launch's, if it is given any, reads each back, and stores them in
/// `out[0]` and `out[1]`: 7 and 9, or 0 for the memory it has not.
const OWN: &str = "
.version 7.5
.target sm_61
.address_size 64

.extern .shared .align 4 .b8 launched[];

.visible .entry both(.param .u64 out_param, .param .u32 bytes_param)
{
	.shared .align 4 .b8 fixed[20480];
	.reg .pred %none;
	.reg .u32 %bytes, %seven, %nine, %mine, %theirs, %at;
	.reg .u64 %out;

	ld.param.u64 %out, [out_param];
	cvta.to.global.u64 %out, %out;
	ld.param.u32 %bytes, [bytes_param];
	mov.u32 %seven, 7;
	st.shared.u32 [fixed+20476], %seven;
	ld.shared.u32 %mine, [fixed+20476];
	st.global.u32 [%out], %mine;
	setp.eq.u32 %none, %bytes, 0;
	@%none bra done;
	mov.u32 %at, launched;
	add.u32 %at, %at, %bytes;
	mov.u32 %nine, 9;
	st.shared.u32 [%at+-4], %nine;
	ld.shared.u32 %theirs, [%at+-4];
	st.global.u32 [%out+4], %theirs;
done:
	ret;
}

.visible .entry plain(.param .u64 out_param, .param .u32 bytes_param)
{
	.reg .pred %none;
	.reg .u32 %bytes, %nine, %theirs, %at;
	.reg .u64 %out;

	ld.param.u64 %out, [out_param];
	cvta.to.global.u64 %out, %out;
	ld.param.u32 %bytes, [bytes_param];
	setp.eq.u32 %none, %bytes, 0;
	@%none bra done;
	mov.u32 %at, launched;
	add.u32 %at, %at, %bytes;
	mov.u32 %nine, 9;
	st.shared.u32 [%at+-4], %nine;
	ld.shared.u32 %theirs, [%at+-4];
	st.global.u32 [%out+4], %theirs;
done:
	ret;
}
";

/// Each kernel of `OWN`, by its index, and the bytes of shared memory it has of its own.
const OWNS: [(usize, u32); 2] = [(0, 20 << 10), (1, 0)];

/// Kernel `index` of `OWN` over `grid`, whose `shared` it is told, writing at `out`; launched
/// cooperatively or not.
fn launch_own(
    module: &Module,
    cooperative: bool,
    index: usize,
    grid: Grid,
    out: kernite_cuda::Ptr<u32>,
) -> Result<(), kernite_cuda::Error> {
    let bytes = grid.shared;
    let args = [arg(&out), arg(&bytes)];
    // SAFETY: both kernels of `OWN` take a pointer and a `u32`, and write two words at the
    // pointer, which has them; the last word of the `bytes` of shared memory the grid gives is
    // inside them.
    unsafe {
        match cooperative {
            false => module.launch(index, grid, &args),
            true => module.launch_cooperative(index, grid, &args),
        }
    }
}

#[test]
fn a_launch_gives_a_block_all_the_shared_memory_the_gpu_has_with_the_kernels_own_counted() {
    let Some(cuda) = cuda() else { return };
    let most = cuda.max_shared();
    if most < 99 << 10 {
        eprintln!("{}, sm_{}, gives a block {most} bytes of shared memory, not 99 KB: skipped", cuda.name(), cuda.sm());
        return;
    }
    let load = || Module::load(&cuda, OWN, &["both", "plain"], &[]).unwrap();
    let mut out = cuda.alloc::<u32>(2).unwrap();
    for (index, own) in OWNS {
        // The count of blocks a kernel runs at once lets it have the shared memory it counts by,
        // as a launch does: of a kernel nothing has let have more, it is not none.
        assert!(load().resident(index, 256, most - own).unwrap() >= 1, "{own} bytes of the kernel's");
    }
    for cooperative in [false, true] {
        if cooperative && !cuda.cooperative() {
            eprintln!("{} launches no kernel cooperatively: skipped", cuda.name());
            continue;
        }
        for (index, own) in OWNS {
            // A module of its own, which nothing has let have more than the driver gives unasked.
            let module = load();
            // What a launch gets unasked, 48 KB with the kernel's own, and what the GPU gives it.
            let (unasked, all) = ((48 << 10) - own, most - own);
            let mut sizes: Vec<u32> = [0, 16, 28, 29, 40, 48, 49, 79].map(|kb| kb << 10).into();
            sizes.extend([unasked - 4, unasked, unasked + 4, all - 4, all]);
            sizes.retain(|&bytes| bytes <= all);
            sizes.sort();
            sizes.dedup();
            // Up from none to all, and down again: a kernel let have more keeps what it was let,
            // and runs a launch of less.
            for &bytes in sizes.iter().chain(sizes.iter().rev()) {
                out.zero().unwrap();
                let grid = Grid::new(1, 1).with_shared(bytes);
                let what =
                    format!("{bytes} bytes from the launch and {own} of the kernel's, cooperative: {cooperative}");
                launch_own(&module, cooperative, index, grid, out.ptr()).unwrap_or_else(|e| panic!("{what}: {e}"));
                let want = [if own > 0 { 7 } else { 0 }, if bytes > 0 { 9 } else { 0 }];
                assert_eq!(out.read().unwrap(), want, "{what}");
            }
        }
    }
    cuda.sync().unwrap();
}

#[test]
fn a_launch_past_the_shared_memory_the_gpu_gives_a_block_says_how_much_the_block_has() {
    let Some(cuda) = cuda() else { return };
    let most = cuda.max_shared();
    let module = Module::load(&cuda, OWN, &["both", "plain"], &[]).unwrap();
    let mut out = cuda.alloc::<u32>(2).unwrap();
    let said = |own: u32, bytes: u32| {
        let all = u64::from(own) + u64::from(bytes);
        format!(
            "each block has {all} bytes of shared memory, {own} of the kernel's and {bytes} of the launch's, and the \
             GPU takes {most} at most"
        )
    };
    for (index, own) in OWNS {
        let all = most - own;
        // A byte past, a word, a kilobyte (80 KB with 20 KB of the kernel's, on a GPU of 99 KB), a
        // kilobyte past what a block has in all (100 KB with none of the kernel's), the most a count
        // of bytes holds in a `c_int` and one more, and the most a `u32` holds.
        for bytes in [all + 1, all + 4, all + 1024, most + 1024, (1 << 31) - 1, 1 << 31, u32::MAX] {
            let launch = Grid::new(1, 1).with_shared(bytes);
            for cooperative in [false, true] {
                if cooperative && !cuda.cooperative() {
                    continue;
                }
                let what =
                    format!("{bytes} bytes from the launch and {own} of the kernel's, cooperative: {cooperative}");
                let e = launch_own(&module, cooperative, index, launch, out.ptr()).expect_err(&what);
                refusal("cuFuncSetAttribute", &e, &said(own, bytes));
            }
            let e = module.resident(index, 256, bytes).expect_err("resident");
            refusal("cuFuncSetAttribute", &e, &said(own, bytes));
        }
        // A launch refused leaves the kernel as it was, and the GPU: the most it gives still runs,
        // and so does a launch of the 48 KB it gets unasked.
        for bytes in [all, (48 << 10) - own] {
            out.zero().unwrap();
            launch_own(&module, false, index, Grid::new(1, 1).with_shared(bytes), out.ptr()).unwrap();
            assert_eq!(out.read().unwrap(), [if own > 0 { 7 } else { 0 }, 9], "{bytes} bytes and {own}");
        }
        assert!(module.resident(index, 256, all).unwrap() >= 1);
    }
    cuda.sync().unwrap();
}

#[test]
fn the_first_size_past_is_said_whether_it_is_the_grids_or_the_shared_memorys() {
    let Some(cuda) = cuda() else { return };
    let most = cuda.max_shared();
    let module = Module::load(&cuda, OWN, &["plain"], &[]).unwrap();
    let out = cuda.alloc::<u32>(2).unwrap();
    let shared = most + 4;
    let grid = |blocks, threads| Grid { blocks, threads, shared };
    let all = format!(
        "each block has {shared} bytes of shared memory, 0 of the kernel's and {shared} of the launch's, and the GPU \
         takes {most} at most"
    );
    // The shared memory is past the most in each, and the sizes of the grid come before it.
    let cases = [
        (grid([1, 65536, 1], [1, 1, 1]), "the grid has 65536 blocks along y, and the GPU takes 65535 at most"),
        (grid([1, 1, 1], [1025, 1, 1]), "each block has 1025 threads along x, and the GPU takes 1024 at most"),
        (grid([0, 1, 1], [1, 1, 1]), "the grid has 0 blocks along x, and the GPU takes 1 at least"),
        (grid([1, 1, 1], [32, 32, 65]), "each block has 65 threads along z, and the GPU takes 64 at most"),
        (grid([1, 1, 1], [1024, 2, 1]), "each block has 2048 threads, and the GPU takes 1024 at most"),
        (grid([1, 1, 1], [1024, 1, 1]), all.as_str()),
        (grid([1, 1, 1], [1, 1, 1]), all.as_str()),
    ];
    for (grid, said) in cases {
        let e = launch_own(&module, false, 0, grid, out.ptr()).unwrap_err();
        refusal("cuFuncSetAttribute", &e, said);
    }
    cuda.sync().unwrap();
}

/// A module of kernels written by hand, each adding 1 to the word at its pointer for each thread
/// that runs, and each saying the directives after its name in `kernels`, one to a line.
fn counting(kernels: &[(&str, &str)]) -> String {
    let mut ptx = ".version 3.0\n.target sm_30\n.address_size 64\n".to_string();
    for (name, directives) in kernels {
        ptx += &format!(
            "\n.visible .entry {name}(.param .u64 out_param)\n{directives}\n{{\n\t.reg .u64 %out;\n\n\
             \tld.param.u64 %out, [out_param];\n\tcvta.to.global.u64 %out, %out;\n\
             \tred.global.add.u32 [%out], 1;\n\tret;\n}}\n"
        );
    }
    ptx
}

/// The kernels `counting` writes, in order: `.reqntid` of one size and of two, `.maxntid` of one
/// size and of three, one with `.minnctapersm` after its `.maxntid`, and one with `.maxnreg` before it.
const TUNED: [(&str, &str); 6] = [
    ("req", ".reqntid 256"),
    ("req_xy", ".reqntid 16, 16"),
    ("max", ".maxntid 256"),
    ("max_xyz", ".maxntid 8, 8, 4"),
    ("max_blocks", ".maxntid 256\n.minnctapersm 2"),
    ("regs", ".maxnreg 32\n.maxntid 128"),
];

/// The launches of the kernels of `TUNED`, each by its place in it, over blocks of so many threads
/// along x, y and z, and what the driver does with it: starts blocks of so many threads, or says
/// which size is past which limit.
fn launches() -> Vec<(usize, [u32; 3], Result<u32, String>)> {
    let req = |threads: u32, along: &str, exactly: u32| {
        format!(
            "each block has {threads} threads along {along}, and the kernel's `.reqntid 256` takes {exactly} exactly"
        )
    };
    let max = |threads: u32, directive: &str, most: u32| {
        format!("each block has {threads} threads, and the kernel's `.maxntid {directive}` takes {most} at most")
    };
    let xy = |threads: u32, along: &str| {
        format!("each block has {threads} threads along {along}, and the kernel's `.reqntid 16, 16` takes 16 exactly")
    };
    vec![
        // `.reqntid` takes a block of exactly its shape. A block of one thread is let through, and the
        // driver starts the kernel's own threads in it: 256, not 1.
        (0, [256, 1, 1], Ok(256)),
        (0, [1, 1, 1], Ok(256)),
        (0, [128, 1, 1], Err(req(128, "x", 256))),
        (0, [512, 1, 1], Err(req(512, "x", 256))),
        (0, [256, 2, 1], Err(req(2, "y", 1))),
        (0, [16, 16, 1], Err(req(16, "x", 256))),
        (1, [16, 16, 1], Ok(256)),
        (1, [1, 1, 1], Ok(256)),
        (1, [256, 1, 1], Err(xy(256, "x"))),
        (1, [16, 8, 2], Err(xy(8, "y"))),
        // `.maxntid` takes a block of at most that many threads in all, of any shape.
        (2, [256, 1, 1], Ok(256)),
        (2, [16, 16, 1], Ok(256)),
        (2, [1, 1, 1], Ok(1)),
        (2, [512, 1, 1], Err(max(512, "256", 256))),
        (2, [16, 16, 2], Err(max(512, "256", 256))),
        (3, [8, 8, 4], Ok(256)),
        (3, [256, 1, 1], Ok(256)),
        (3, [1024, 1, 1], Err(max(1024, "8, 8, 4", 256))),
        (4, [256, 1, 1], Ok(256)),
        (4, [257, 1, 1], Err(max(257, "256", 256))),
        (5, [128, 1, 1], Ok(128)),
        (5, [129, 1, 1], Err(max(129, "128", 128))),
        // The GPU's limits are said before the kernel's.
        (2, [2048, 1, 1], Err("each block has 2048 threads along x, and the GPU takes 1024 at most".into())),
    ]
}

#[test]
fn a_launch_over_a_block_the_kernel_refuses_says_the_directive_and_the_threads() {
    let Some(cuda) = cuda() else { return };
    let names: Vec<&str> = TUNED.iter().map(|(name, _)| *name).collect();
    let module = Module::load(&cuda, &counting(&TUNED), &names, &[]).unwrap();
    let mut count = cuda.alloc::<u32>(1).unwrap();
    for (index, threads, done) in launches() {
        let grid = Grid { blocks: [3, 1, 1], threads, shared: 0 };
        count.zero().unwrap();
        // SAFETY: each kernel takes a pointer, and adds to the word there.
        let launched = unsafe { module.launch(index, grid, &[arg(&count.ptr())]) };
        match (launched, done) {
            (Ok(()), Ok(block)) => {
                cuda.sync().unwrap();
                assert_eq!(count.read().unwrap(), [3 * block], "{} over {grid:?}", names[index]);
            }
            (Err(e), Err(said)) => refusal("cuLaunchKernel", &e, &said),
            (launched, done) => {
                panic!("{} over {grid:?}: {launched:?}, and the driver should do {done:?}", names[index])
            }
        }
    }
    // The launches refused did not harm the GPU, nor the ones that started.
    cuda.sync().unwrap();
}

#[test]
fn a_cooperative_launch_over_a_block_the_kernel_refuses_says_it_too() {
    let Some(cuda) = cuda() else { return };
    if !cuda.cooperative() {
        eprintln!("{} launches no kernel cooperatively: skipped", cuda.name());
        return;
    }
    let names: Vec<&str> = TUNED.iter().map(|(name, _)| *name).collect();
    let module = Module::load(&cuda, &counting(&TUNED), &names, &[]).unwrap();
    let count = cuda.alloc::<u32>(1).unwrap();
    // Only the blocks the kernel refuses: one it takes may have more blocks than it runs at once,
    // which is refused in other words.
    for (index, threads, said) in launches().into_iter().filter_map(|(i, t, done)| done.err().map(|said| (i, t, said)))
    {
        let grid = Grid { blocks: [3, 1, 1], threads, shared: 0 };
        // SAFETY: each kernel takes a pointer, and adds to the word there.
        let e = unsafe { module.launch_cooperative(index, grid, &[arg(&count.ptr())]) }.unwrap_err();
        refusal("cuLaunchCooperativeKernel", &e, &said);
    }
    cuda.sync().unwrap();
}

#[test]
fn the_kernels_block_is_said_after_the_gpus_limits_and_the_shared_memory() {
    let Some(cuda) = cuda() else { return };
    let names: Vec<&str> = TUNED.iter().map(|(name, _)| *name).collect();
    let module = Module::load(&cuda, &counting(&TUNED), &names, &[]).unwrap();
    let count = cuda.alloc::<u32>(1).unwrap();
    let (index, most) = (2, cuda.max_shared());
    let shared = most + 4;
    // 512 threads are past the `.maxntid 256` of kernel `max`, which is at `index`, and `shared`
    // is past what the GPU gives a block: the shared memory is said, for the kernel's block comes
    // after it. The GPU's limits on the threads come before it too, which `launches` has.
    let past_both = Grid { blocks: [3, 1, 1], threads: [512, 1, 1], shared };
    let all = format!(
        "each block has {shared} bytes of shared memory, 0 of the kernel's and {shared} of the launch's, and the GPU \
         takes {most} at most"
    );
    let args = [arg(&count.ptr())];
    // SAFETY: the kernels take a pointer, and add to the word there, if they start at all.
    let e = unsafe { module.launch(index, past_both, &args) }.unwrap_err();
    refusal("cuFuncSetAttribute", &e, &all);
    if cuda.cooperative() {
        // SAFETY: as above.
        let e = unsafe { module.launch_cooperative(index, past_both, &args) }.unwrap_err();
        refusal("cuFuncSetAttribute", &e, &all);
    }
    cuda.sync().unwrap();
}

/// The indexes of the GPUs for a test of two: 0 and 1; or, on a machine with one GPU, 0 twice,
/// and a line that says what the test could not show. `None`, and why, where the driver shows none.
fn two_gpus() -> Option<[usize; 2]> {
    // Where the driver shows none, `cuda` says so, and the test is skipped.
    cuda()?;
    let count = Cuda::count().unwrap();
    if count < 2 {
        eprintln!("two GPUs: skipped, the driver shows {count}; the test runs on GPU 0 twice");
        return Some([0, 0]);
    }
    Some([0, 1])
}

/// Launches `add_one`, kernel 0 of `module`, over the first 1000 floats of `x`.
fn add_one(module: &Module, x: &Buffer<f32>) {
    let (x, n) = (x.ptr(), 1000u32);
    // SAFETY: `add_one` takes a pointer and a `u32`, and writes only the first `n` floats there,
    // which the buffers of these tests have.
    unsafe { module.launch(0, Grid::new(4, 256), &[arg(&x), arg(&n)]) }.unwrap();
}

#[test]
fn a_gpu_is_taken_by_its_index_and_a_missing_one_is_an_error_that_says_how_many_there_are() {
    let Some(first) = cuda() else { return };
    let count = Cuda::count().unwrap();
    // The system lists a GPU in a directory of its own, unless `CUDA_VISIBLE_DEVICES` hides some.
    if let (Ok(gpus), None) = (std::fs::read_dir("/proc/driver/nvidia/gpus"), std::env::var_os("CUDA_VISIBLE_DEVICES"))
    {
        assert_eq!(gpus.count(), count);
    }
    // GPU 0 by its index is the one `Cuda::new` gives, and every index below the count is a GPU.
    let describe = |c: &Cuda| (c.name().to_string(), c.sm(), c.max_shared(), c.multiprocessors());
    assert_eq!(describe(&Cuda::on(0).unwrap()), describe(&first));
    for index in 0..count {
        Cuda::on(index).unwrap();
    }
    // The first index past the last GPU, and the greatest there is, are errors that say how many
    // GPUs there are.
    let gpus = if count == 1 { "1 GPU".to_string() } else { format!("{count} GPUs") };
    for past in [count, usize::MAX] {
        let e = Cuda::on(past).err().unwrap();
        let said = "CUDA_ERROR_INVALID_DEVICE (101): invalid device ordinal";
        assert_eq!(
            (e.call, e.message),
            ("cuDeviceGet", format!("{said}: the driver shows {gpus}, and none is numbered {past}"))
        );
    }
    if count == 1 {
        let e = Cuda::on(1).err().unwrap();
        assert_eq!(
            e.to_string(),
            "cuDeviceGet: CUDA_ERROR_INVALID_DEVICE (101): invalid device ordinal: the driver shows 1 GPU, and none is numbered 1"
        );
    }
}

#[test]
fn a_cuda_of_each_gpu_on_one_thread_sends_each_call_to_its_own_gpu() {
    let Some(gpus) = two_gpus() else { return };
    // Each `Cuda` is made, and its module loaded, with its own GPU's context current; the second
    // makes its context current in the first's place.
    let a = Cuda::on(gpus[0]).unwrap();
    let module_a = Module::load(&a, PTX, &["add_one"], &[]).unwrap();
    let globals_a = Module::load(&a, GLOBALS, &["count"], &["counts"]).unwrap();
    let b = Cuda::on(gpus[1]).unwrap();
    let module_b = Module::load(&b, PTX, &["add_one"], &[]).unwrap();
    let globals_b = Module::load(&b, GLOBALS, &["count"], &["counts"]).unwrap();
    let (mut xs_a, mut xs_b) = (a.alloc::<f32>(1000).unwrap(), b.alloc::<f32>(1000).unwrap());
    for round in 1..=4 {
        // Copies in turn, then launches on both GPUs before either is waited for, in an order
        // that changes, and each call goes to its own GPU.
        xs_a.write(&[round as f32; 1000]).unwrap();
        xs_b.write(&[10.0 * round as f32; 1000]).unwrap();
        let order: [&dyn Fn(); 4] = if round % 2 == 0 {
            [&|| add_one(&module_a, &xs_a), &|| add_one(&module_b, &xs_b), &|| add_one(&module_b, &xs_b), &|| {
                add_one(&module_a, &xs_a)
            }]
        } else {
            [&|| add_one(&module_b, &xs_b), &|| add_one(&module_a, &xs_a), &|| add_one(&module_a, &xs_a), &|| {
                add_one(&module_b, &xs_b)
            }]
        };
        order.iter().for_each(|launch| launch());
        if round % 2 == 0 {
            a.sync().unwrap();
            b.sync().unwrap();
        } else {
            b.sync().unwrap();
            a.sync().unwrap();
        }
        assert_eq!(xs_a.read().unwrap(), vec![round as f32 + 2.0; 1000], "round {round}, GPU {}", gpus[0]);
        assert_eq!(xs_b.read().unwrap(), vec![10.0 * round as f32 + 2.0; 1000], "round {round}, GPU {}", gpus[1]);
    }
    // The memory of each module is on its own GPU: the same variable on each holds its own count.
    let count = |module: &Module, threads: u32| {
        let i = 1u32;
        // SAFETY: `count` takes a `u32`, and adds to word `i` of the four of `counts`.
        unsafe { module.launch(0, Grid::new(1, threads), &[arg(&i)]) }.unwrap()
    };
    count(&globals_a, 3);
    count(&globals_b, 5);
    count(&globals_a, 3);
    assert_eq!(globals_b.global::<u32>(0).read().unwrap(), [0, 5, 0, 0]);
    assert_eq!(globals_a.global::<u32>(0).read().unwrap(), [0, 6, 0, 0]);
    // What is dropped goes to its own GPU's context too, whichever was used last.
    drop(xs_a);
    b.sync().unwrap();
    drop(globals_a);
    drop(module_a);
    assert_eq!(xs_b.read().unwrap(), vec![42.0; 1000]);
    drop(a);
    b.sync().unwrap();
}

#[test]
fn a_buffer_is_copied_from_one_gpu_to_another() {
    let Some(gpus) = two_gpus() else { return };
    let a = Cuda::on(gpus[0]).unwrap();
    let module_a = Module::load(&a, PTX, &["add_one"], &[]).unwrap();
    let b = Cuda::on(gpus[1]).unwrap();
    let module_b = Module::load(&b, PTX, &["add_one"], &[]).unwrap();
    // Whole numbers, so that each addition is exact, whichever order the sums are made in.
    let xs: Vec<f32> = (0..1000).map(|i| i as f32).collect();
    let plus = |n: u32| xs.iter().map(|x| x + n as f32).collect::<Vec<f32>>();
    let (mut from_a, mut on_b) = (a.upload(&xs).unwrap(), b.alloc::<f32>(1000).unwrap());
    // The kernels launched before the copy, on either GPU, which are not waited for, have run when
    // it is made, and those launched after it, on either, run after it.
    (0..100).for_each(|_| add_one(&module_b, &on_b));
    (0..300).for_each(|_| add_one(&module_a, &from_a));
    on_b.copy_from(&from_a).unwrap();
    (0..200).for_each(|_| add_one(&module_a, &from_a));
    (0..200).for_each(|_| add_one(&module_b, &on_b));
    assert_eq!(on_b.read().unwrap(), plus(500), "300 on the first GPU before the copy, 200 on the second after it");
    assert_eq!(from_a.read().unwrap(), plus(500), "300 before the copy and 200 after it");
    // And back.
    on_b.write(&plus(7)).unwrap();
    from_a.copy_from(&on_b).unwrap();
    assert_eq!(from_a.read().unwrap(), plus(7));
    // A buffer of none is copied to one of none, and one of another length is refused.
    b.alloc::<f32>(0).unwrap().copy_from(&a.alloc::<f32>(0).unwrap()).unwrap();
    let short = a.alloc::<f32>(999).unwrap();
    let refused = catch_unwind(AssertUnwindSafe(|| on_b.copy_from(&short))).unwrap_err();
    let said = refused.downcast_ref::<String>().cloned().unwrap_or_default();
    assert!(said.contains("a buffer of 1000 elements copied from one of 999"), "{said}");
}

#[test]
fn threads_each_take_a_cuda_of_a_gpu_and_may_share_one() {
    let Some(_) = cuda() else { return };
    let count = Cuda::count().unwrap();
    // Two threads for each GPU, each with the `Cuda` of its own thread.
    std::thread::scope(|scope| {
        for thread in 0..2 * count {
            scope.spawn(move || {
                let cuda = Cuda::on(thread % count).unwrap();
                let module = Module::load(&cuda, PTX, &["add_one"], &[]).unwrap();
                let xs = cuda.upload(&[thread as f32; 1000]).unwrap();
                for _ in 0..50 {
                    add_one(&module, &xs);
                }
                assert_eq!(xs.read().unwrap(), vec![thread as f32 + 50.0; 1000], "thread {thread}");
            });
        }
    });
}
