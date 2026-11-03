//! Kernels of compile-time arguments whose types name one: a pointer to an array of the length an
//! argument gives, in braces as well, a type of the crate's that takes it as its own argument, in a
//! parameter and in a result, with one argument, two and a `bool`, in a kernel that is cooperative.
//! There are kernels that call an instance of one argument, of two, of a `bool`, of a negative
//! number and of the caller's own, and one whose code holds a constant. A type in a kernel means in
//! each of its instances what it means in the kernel. Rust builds the crate without a word, and the
//! tests run each x86 kernel, which writes through its pointer: what it wrote is what the test
//! finds. The doc test of `Pair` is a crate of its own, which calls two of them from outside.

/// A type of the crate's with a number as its argument. A doc test is another crate, which calls an
/// instance as one would, whose code holds a constant that lives in the crate of the kernels:
///
/// ```
/// let mut bytes = [0u8; 3];
/// let mut floats = [0f32; 2];
/// unsafe {
///     argument_pointers::cpu::arr::<3>(&mut bytes);
///     argument_pointers::cpu::scaled::<2>(&mut floats);
/// }
/// assert_eq!(bytes[0], 3);
/// assert_eq!(floats[0], 0.25);
/// ```
#[repr(C)]
pub struct Pair<const N: usize>(pub [u8; N]);

/// A type of the crate's with a `bool` as its argument.
#[repr(C)]
pub struct Flag<const B: bool>(pub u8);

/// x86 kernels: each is of instances, as its name says, and `together` is cooperative as well.
#[kernite_macros::kernels]
pub mod cpu {
    use super::{Flag, Pair};

    /// Writes its length through `p`.
    #[instances(2, 3)]
    pub unsafe extern "sysv64" fn arr<const N: usize>(p: *mut [u8; N]) {
        mov(byte[p], N);
    }

    /// Writes its length through `p`, and gives it back.
    #[instances(2, 3)]
    pub unsafe extern "sysv64" fn back<const N: usize>(p: *mut [u8; N]) -> *mut [u8; N] {
        mov(byte[p], N);
        ret(p);
    }

    /// Writes the argument of a type of the crate's.
    #[instances(2, 3)]
    pub unsafe extern "sysv64" fn pair<const N: usize>(p: *mut Pair<N>) {
        mov(byte[p], N);
    }

    /// Writes its length through `p`, whose type has the argument in braces, which Rust takes as it
    /// takes the argument alone.
    #[allow(unused_braces)]
    #[instances(2, 3)]
    pub unsafe extern "sysv64" fn braced<const N: usize>(p: *mut [u8; { N }]) {
        mov(byte[p], N);
    }

    /// Writes both its arguments: the width of a row, and how many rows there are.
    #[instances((2, 3), (4, 1))]
    pub unsafe extern "sysv64" fn rows<const W: usize, const H: usize>(p: *mut [[u8; W]; H]) {
        mov(byte[p], W);
        mov(byte[p + 1], H);
    }

    /// Writes 1 for `true`, 2 for `false`.
    #[instances(true, false)]
    pub unsafe extern "sysv64" fn flag<const B: bool>(p: *mut Flag<B>) {
        if B {
            mov(byte[p], 1);
        } else {
            mov(byte[p], 2);
        }
    }

    /// Every thread writes its length through `p`.
    #[cooperative]
    #[instances(2, 3)]
    pub unsafe extern "sysv64" fn together<const N: usize>(p: *mut [u8; N]) {
        mov(byte[p], N);
    }

    /// Writes a constant through `p`, which a static of the module holds.
    #[instances(2, 3)]
    pub unsafe extern "sysv64" fn scaled<const N: usize>(p: *mut [f32; N]) {
        let q: xmm;
        vbroadcastss(q, [0.25]);
        vmovss(dword[p], q);
    }

    /// Writes a signed argument.
    #[instances(-2, 3)]
    pub unsafe extern "sysv64" fn signed<const K: i8>(p: *mut [u8; 1]) {
        mov(byte[p], K);
    }

    /// Calls `arr` for the length it is not given, 2.
    pub unsafe extern "sysv64" fn calls(p: *mut [u8; 2]) {
        call(arr::<2>, p);
    }

    /// Calls an instance of two arguments, one of a `bool` and one of a negative number.
    pub unsafe extern "sysv64" fn calls_more(p: *mut [[u8; 4]; 1], q: *mut Flag<false>, r: *mut [u8; 1]) {
        call(rows::<4, 1>, p);
        call(flag::<false>, q);
        call(signed::<-2>, r);
    }

    /// Calls `arr` for its own length, and `scaled` for 2, from an instance of its own.
    #[instances(2, 3)]
    pub unsafe extern "sysv64" fn via<const N: usize>(p: *mut [u8; N], q: *mut [f32; 2]) {
        call(arr::<N>, p);
        call(scaled::<2>, q);
    }
}

/// GPU kernels, which the launcher of each takes by the same types. They are built here, not run.
#[kernite_macros::kernels]
pub mod gpu {
    /// Writes its length through `p`.
    #[instances(2, 3)]
    pub unsafe extern "ptx-kernel" fn arr<const N: usize>(p: *mut [u8; N]) {
        st.global.u8([p], N);
    }
}

/// Each GPU kernel, launched as Rust launches it.
pub fn on_the_gpu(
    kernels: &gpu::Kernels,
    two: kernite_cuda::Ptr<[u8; 2]>,
    three: kernite_cuda::Ptr<[u8; 3]>,
) -> Result<(), kernite_cuda::Error> {
    let grid = kernite_cuda::Grid::new(1, 1);
    unsafe {
        kernels.arr::<2>(grid, two)?;
        kernels.arr::<3>(grid, three)
    }
}

#[test]
fn a_pointer_to_an_array_of_the_length_the_argument_gives() {
    let (mut two, mut three) = ([0u8; 2], [0u8; 3]);
    unsafe {
        cpu::arr::<2>(&mut two);
        cpu::arr::<3>(&mut three);
    }
    assert_eq!((two[0], three[0]), (2, 3));
    let mut calls = [0u8; 2];
    unsafe { cpu::calls(&mut calls) };
    assert_eq!(calls[0], 2);
}

#[test]
fn a_result_that_names_the_argument() {
    let (mut two, mut three) = ([0u8; 2], [0u8; 3]);
    unsafe {
        assert_eq!(cpu::back::<2>(&mut two), &raw mut two);
        assert_eq!(cpu::back::<3>(&mut three), &raw mut three);
    }
    assert_eq!((two[0], three[0]), (2, 3));
}

#[test]
fn a_type_of_the_crates_that_takes_the_argument() {
    let (mut two, mut three) = (Pair([0u8; 2]), Pair([0u8; 3]));
    unsafe {
        cpu::pair::<2>(&mut two);
        cpu::pair::<3>(&mut three);
    }
    assert_eq!((two.0[0], three.0[0]), (2, 3));
}

#[test]
fn an_argument_in_braces() {
    let (mut two, mut three) = ([0u8; 2], [0u8; 3]);
    unsafe {
        cpu::braced::<2>(&mut two);
        cpu::braced::<3>(&mut three);
    }
    assert_eq!((two[0], three[0]), (2, 3));
}

#[test]
fn a_kernel_of_two_arguments() {
    let (mut first, mut second) = ([[0u8; 2]; 3], [[0u8; 4]; 1]);
    unsafe {
        cpu::rows::<2, 3>(&mut first);
        cpu::rows::<4, 1>(&mut second);
    }
    assert_eq!((first[0], second[0]), ([2, 3], [4, 1, 0, 0]));
}

#[test]
fn a_bool_argument() {
    let (mut yes, mut no) = (Flag::<true>(0), Flag::<false>(0));
    unsafe {
        cpu::flag::<true>(&mut yes);
        cpu::flag::<false>(&mut no);
    }
    assert_eq!((yes.0, no.0), (1, 2));
}

#[test]
fn a_cooperative_kernel() {
    let (mut two, mut three) = ([0u8; 2], [0u8; 3]);
    let grid = kernite_cpu::Grid::new(1);
    unsafe {
        cpu::together::<2>(grid, &mut two).unwrap();
        cpu::together::<3>(grid, &mut three).unwrap();
    }
    assert_eq!((two[0], three[0]), (2, 3));
}

#[test]
fn a_kernel_that_calls_instances_of_two_arguments_a_bool_and_a_negative_number() {
    let (mut rows, mut flag, mut signed) = ([[0u8; 4]; 1], Flag::<false>(0), [0u8; 1]);
    unsafe { cpu::calls_more(&mut rows, &mut flag, &mut signed) };
    assert_eq!((rows[0], flag.0, signed[0]), ([4, 1, 0, 0], 2, 0xfe));
}

#[test]
fn a_kernel_that_calls_an_instance_with_its_own_argument() {
    let (mut two, mut three, mut floats) = ([0u8; 2], [0u8; 3], [0f32; 2]);
    unsafe {
        cpu::via::<2>(&mut two, &mut floats);
        cpu::via::<3>(&mut three, &mut floats);
    }
    assert_eq!((two[0], three[0], floats[0]), (2, 3, 0.25));
}
