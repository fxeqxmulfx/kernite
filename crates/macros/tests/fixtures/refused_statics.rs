//! A module of x86 kernels and one of GPU kernels that `#[kernels]` refuses at its attribute, `consts = 5`, each
//! with statics. Nothing of a module is compiled when it is refused, but a static stands as its type has it where
//! kernite works its type out and takes it and its name: the code that uses it is read as for one that compiles,
//! though it says no type for the elements, and a use the type does not fit is Rust's error, beside the refusal. A
//! static whose type reads a const the module would be given stands as one of elements of any type: the code says
//! the type, or Rust asks for it, beside the refusal.

#[kernite_macros::kernels(consts = 5)]
pub mod cpu {
    pub static COUNTS: global<[u32; 4]> = [1, 2, 3, 4, 5];
    pub static SIZED: global<[u32; N]>;

    pub unsafe extern "sysv64" fn step(x: u64) -> u64 {
        ret(x);
    }
}

#[kernite_macros::kernels(consts = 5)]
pub mod gpu {
    pub static COUNTS: global<[u32; 4]> = [1, 2, 3, 4, 5];
    pub static SIZED: global<[u32; N]>;

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

pub fn cpu_address() -> usize {
    cpu::COUNTS.ptr() as usize
}

/// A use that the static's type does not fit is Rust's error, as it is for one that compiles.
pub fn cpu_misfit() -> Vec<u64> {
    cpu::COUNTS.read()
}

/// A static whose type reads a const the module would be given has elements of any type, which a use says.
pub fn cpu_sized() -> usize {
    let sized: Vec<u32> = cpu::SIZED.read();
    sized.len()
}

/// A use that says no type for the elements is Rust's error, beside the refusal.
pub fn cpu_sized_open() -> u64 {
    cpu::SIZED.read()[0] as u64
}

pub fn cpu_sized_address() -> usize {
    cpu::SIZED.ptr() as usize
}

/// A `Global` of the GPU's, of the type `gpu::COUNTS` is read as.
pub fn is_gpu_counts(_: kernite_cuda::Global<'_, u32>) {}

/// The accessor of a GPU static gives a `Global` of the type the static is read as.
pub fn gpu_type(kernels: &gpu::Kernels) {
    is_gpu_counts(kernels.COUNTS());
}

pub fn gpu_length(kernels: &gpu::Kernels) -> usize {
    kernels.COUNTS().len()
}

/// A use that the static's type does not fit is Rust's error, as it is for one that compiles.
pub fn gpu_misfit(kernels: &gpu::Kernels) -> Result<Vec<u64>, kernite_cuda::Error> {
    kernels.COUNTS().read()
}

/// A static whose type reads a const the module would be given has elements of any type, which a use says.
pub fn gpu_sized(kernels: &gpu::Kernels) {
    is_gpu_counts(kernels.SIZED());
}

/// A use that says no type for the elements is Rust's error, beside the refusal.
pub fn gpu_sized_open(kernels: &gpu::Kernels) -> usize {
    kernels.SIZED().len()
}
