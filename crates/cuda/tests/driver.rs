//! The driver as kgen calls it, on the GPU here, if there is one.

use kgen_cuda::{Cuda, Grid, Module, arg, bf16, f16};

/// `x[i] += 1` for each `i < n`, a thread each, written by hand.
const PTX: &str = "
.version 7.5
.target sm_75
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
/// alone, as kgen declares its memory, and `seen` visible, as nvcc declares it for a program
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
