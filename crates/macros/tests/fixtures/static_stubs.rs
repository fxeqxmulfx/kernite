//! Statics that do not compile, each said where it does not. If only its value is wrong, a static
//! stands as its type has it: the code that uses it is read as for one that compiles, though it
//! says no type for the elements, and a use the type does not fit is Rust's error. If its type is
//! wrong, it stands as one of elements of any type: the code says the type, or Rust asks for it.

#[kernite_macros::kernels]
pub mod cpu {
    pub static COUNTS: global<[u32; 4]> = [1, 2, 3, 4, 5];
    pub static TABLE: constant<[f32; 2], 16> = [1, 2];
    pub static BYTE: global<u8> = 300;
    pub static HELD: global<[u64; 2]> = [1, nope];
    pub static EMPTY: global<[u32; 0]>;

    pub unsafe extern "sysv64" fn step(x: u64) -> u64 {
        ret(x);
    }
}

#[kernite_macros::kernels]
pub mod gpu {
    pub static COUNTS: global<[u32; 4]> = [1, 2, 3, 4, 5];
    pub static HALVES: constant<[f16; 2]> = [1, 2];
    pub static HELD: global<[s64; 2]> = [1, nope];
    pub static EMPTY: global<[u32; 0]>;

    pub unsafe extern "ptx-kernel" fn k(x: *mut u32) {
        st.global.u32([x], 1);
    }
}

/// A `Global` of the process's, of the type `cpu::COUNTS` is written with.
pub fn is_counts(_: &kernite_cpu::Global<u32, 4, 4>) {}

/// The stub of a process's static is a `Global` of the type the static is written with.
pub fn cpu_type() {
    is_counts(&cpu::COUNTS);
}

pub fn cpu_read() -> u64 {
    cpu::COUNTS.read()[0] as u64
}

pub fn cpu_addresses() -> usize {
    cpu::COUNTS.ptr() as usize + cpu::COUNTS.at(1) as usize
}

pub fn cpu_table() -> f64 {
    let mut table = [0.0; 2];
    cpu::TABLE.read_into(&mut table);
    cpu::TABLE.write(&[1.5, 2.5]);
    cpu::TABLE.zero();
    f64::from(table[0])
}

pub fn cpu_byte() -> u64 {
    u64::from(cpu::BYTE.read()[0])
}

pub fn cpu_held() -> usize {
    cpu::HELD.ptr() as usize
}

/// A use that the static's type does not fit is Rust's error, as it is for one that compiles.
pub fn cpu_misfit() -> Vec<u64> {
    cpu::COUNTS.read()
}

/// A static whose type kernite does not take has elements of any type, which a use says.
pub fn cpu_empty() -> usize {
    let empty: Vec<u32> = cpu::EMPTY.read();
    empty.len()
}

/// A use that says no type for the elements is Rust's error, beside kernite's.
pub fn cpu_empty_open() -> u64 {
    cpu::EMPTY.read()[0] as u64
}

pub fn cpu_empty_address() -> usize {
    cpu::EMPTY.ptr() as usize
}

/// A `Global` of the GPU's, of the type `gpu::COUNTS` is read as.
pub fn is_gpu_counts(_: kernite_cuda::Global<'_, u32>) {}

/// A `Global` of the GPU's, of `f16`s.
pub fn is_halves(_: kernite_cuda::Global<'_, kernite_cuda::f16>) {}

/// A `Global` of the GPU's, of the type `gpu::EMPTY` is used as.
pub fn is_empty(_: kernite_cuda::Global<'_, u32>) {}

/// The accessor of a GPU static gives a `Global` of the type the static is read as.
pub fn gpu_type(kernels: &gpu::Kernels) {
    is_gpu_counts(kernels.COUNTS());
    is_halves(kernels.HALVES());
}

pub fn gpu_length(kernels: &gpu::Kernels) -> usize {
    kernels.COUNTS().len()
}

pub fn gpu_emptiness(kernels: &gpu::Kernels) -> bool {
    kernels.COUNTS().is_empty()
}

pub fn gpu_address(kernels: &gpu::Kernels) {
    let _address = kernels.COUNTS().ptr();
}

pub fn gpu_element(kernels: &gpu::Kernels) {
    let _element = kernels.COUNTS().at(1);
}

pub fn gpu_read(kernels: &gpu::Kernels) -> Result<usize, kernite_cuda::Error> {
    let read = kernels.COUNTS().read()?;
    Ok(read.len())
}

pub fn gpu_zero(kernels: &gpu::Kernels) -> Result<(), kernite_cuda::Error> {
    kernels.COUNTS().zero()
}

pub fn gpu_halves(kernels: &gpu::Kernels) -> Result<usize, kernite_cuda::Error> {
    Ok(kernels.HALVES().read()?.len())
}

pub fn gpu_held(kernels: &gpu::Kernels) -> Result<usize, kernite_cuda::Error> {
    Ok(kernels.HELD().read()?.len())
}

/// A use that the static's type does not fit is Rust's error, as it is for one that compiles.
pub fn gpu_misfit(kernels: &gpu::Kernels) -> Result<Vec<u64>, kernite_cuda::Error> {
    kernels.COUNTS().read()
}

/// A static whose type kernite does not take has elements of any type, which a use says.
pub fn gpu_empty(kernels: &gpu::Kernels) {
    is_empty(kernels.EMPTY());
}

/// A use that says no type for the elements is Rust's error, beside kernite's.
pub fn gpu_empty_open(kernels: &gpu::Kernels) -> usize {
    kernels.EMPTY().len()
}

// A static whose value is wrong and whose type the machine refuses too, `HOLLOW`, has kernite's one error at
// the value, which it reads first. It stands as one whose elements are of any type, as one whose type alone is
// wrong does.
#[kernite_macros::kernels]
pub mod cpu_hollow {
    pub static HOLLOW: global<[u32; 0]> = [nope];

    pub unsafe extern "sysv64" fn step(x: u64) -> u64 {
        ret(x);
    }
}

#[kernite_macros::kernels]
pub mod gpu_hollow {
    pub static HOLLOW: global<[u32; 0]> = [nope];

    pub unsafe extern "ptx-kernel" fn k(x: *mut u32) {
        st.global.u32([x], 1);
    }
}

/// A use that says no type for the elements is Rust's error, beside kernite's, though kernite's is at the value.
pub fn cpu_hollow_open() -> u64 {
    cpu_hollow::HOLLOW.read()[0] as u64
}

/// The same on the GPU.
pub fn gpu_hollow_open(kernels: &gpu_hollow::Kernels) -> usize {
    kernels.HOLLOW().len()
}
