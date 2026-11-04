//! The CPUs a thread may run on, which Linux lets a program read and set. The calls are the C
//! library's, which only `unsafe` reaches, and this is the one module of the crate that has any:
//! each block says why it holds.

use std::fmt;
use std::io;
use std::thread::JoinHandle;

/// How many CPUs a [`Cpus`] has room for: as many as Linux is built for on x86-64, at most.
const ROOM: usize = 8192;

/// A set of CPUs by number, laid out as C's `cpu_set_t` is: CPU `n` is bit `n % 64` of word
/// `n / 64`.
#[derive(PartialEq)]
pub struct Cpus([u64; ROOM / 64]);

impl Cpus {
    /// The set with no CPU.
    pub fn empty() -> Cpus {
        Cpus([0; ROOM / 64])
    }

    /// The set of `cpu` alone; none if the set has no room for it.
    pub fn one(cpu: usize) -> Cpus {
        let mut set = Cpus::empty();
        set.insert(cpu);
        set
    }

    /// Puts `cpu` in the set, and says whether the set changed: it does not if `cpu` is in it
    /// already, or if the set has no room for it.
    pub fn insert(&mut self, cpu: usize) -> bool {
        let Some(word) = self.0.get_mut(cpu / 64) else { return false };
        let bit = 1 << (cpu % 64);
        let changed = *word & bit == 0;
        *word |= bit;
        changed
    }

    /// Whether `cpu` is in the set.
    pub fn contains(&self, cpu: usize) -> bool {
        self.0.get(cpu / 64).is_some_and(|word| word >> (cpu % 64) & 1 == 1)
    }

    /// The CPUs in the set, in order. It takes the set a word at a time and, in a word, only its
    /// CPUs, so that listing costs what the CPUs listed cost, not the room for the others.
    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.0.iter().enumerate().flat_map(|(n, &word)| {
            let mut left = word;
            std::iter::from_fn(move || {
                // The lowest bit left is the next CPU; clearing it moves on.
                (left != 0).then(|| {
                    let cpu = n * 64 + left.trailing_zeros() as usize;
                    left &= left - 1;
                    cpu
                })
            })
        })
    }

    /// The CPUs the calling thread may run on.
    pub fn current() -> io::Result<Cpus> {
        let mut set = Cpus::empty();
        sys::get(&mut set.0)?;
        Ok(set)
    }

    /// Lets `thread` run on these CPUs alone, and the calling thread if it is `None`. The system
    /// refuses a set with no CPU it lets the thread run on.
    pub fn bind(&self, thread: Option<&JoinHandle<()>>) -> io::Result<()> {
        sys::set(thread, &self.0)
    }
}

impl fmt::Display for Cpus {
    /// The CPUs as the system lists them, a run of them as its first and last: `0-5,8,10-11`.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut cpus = self.iter().peekable();
        let mut between = "";
        while let Some(first) = cpus.next() {
            let mut last = first;
            while cpus.next_if_eq(&(last + 1)).is_some() {
                last += 1;
            }
            match first == last {
                true => write!(f, "{between}{first}")?,
                false => write!(f, "{between}{first}-{last}")?,
            }
            between = ",";
        }
        Ok(())
    }
}

/// The calls, on Linux: a thread is named by its `pthread_t`, which `std` gives for a thread it
/// has started.
#[cfg(target_os = "linux")]
mod sys {
    use std::ffi::c_int;
    use std::io;
    use std::os::unix::thread::{JoinHandleExt, RawPthread};
    use std::thread::JoinHandle;

    // The C library, which every Rust program on Linux links.
    unsafe extern "C" {
        fn pthread_getaffinity_np(thread: RawPthread, size: usize, set: *mut u64) -> c_int;
        fn pthread_setaffinity_np(thread: RawPthread, size: usize, set: *const u64) -> c_int;
        /// The calling thread, which is running.
        safe fn pthread_self() -> RawPthread;
    }

    /// What a call returns: 0, or the error number.
    fn done(errno: c_int) -> io::Result<()> {
        match errno {
            0 => Ok(()),
            errno => Err(io::Error::from_raw_os_error(errno)),
        }
    }

    /// Writes to `set` the CPUs the calling thread may run on.
    pub fn get(set: &mut [u64]) -> io::Result<()> {
        // SAFETY: the thread is the calling one, which is running. The call writes at most `size`
        // bytes to `set`, which has them.
        done(unsafe { pthread_getaffinity_np(pthread_self(), size_of_val(set), set.as_mut_ptr()) })
    }

    /// Lets `thread`, or the calling thread if it is `None`, run on the CPUs of `set` alone.
    pub fn set(thread: Option<&JoinHandle<()>>, set: &[u64]) -> io::Result<()> {
        let thread = match thread {
            Some(thread) => thread.as_pthread_t(),
            None => pthread_self(),
        };
        // SAFETY: the thread is the calling one, or one whose `JoinHandle` is borrowed, which
        // keeps it from being joined or detached: that is all the call asks of it. The call reads
        // `size` bytes from `set`, which has them.
        done(unsafe { pthread_setaffinity_np(thread, size_of_val(set), set.as_ptr()) })
    }
}

/// The calls, off Linux, which has none of them to make: every one is an error.
#[cfg(not(target_os = "linux"))]
mod sys {
    use std::io;
    use std::thread::JoinHandle;

    /// Says that the system has no such call.
    pub fn get(_: &mut [u64]) -> io::Result<()> {
        Err(io::ErrorKind::Unsupported.into())
    }

    /// Says that the system has no such call.
    pub fn set(_: Option<&JoinHandle<()>>, _: &[u64]) -> io::Result<()> {
        Err(io::ErrorKind::Unsupported.into())
    }
}

#[cfg(test)]
mod tests {
    use std::hint::black_box;
    use std::time::{Duration, Instant};

    use super::{Cpus, ROOM};

    /// The set of `cpus`.
    fn set(cpus: &[usize]) -> Cpus {
        let mut set = Cpus([0; ROOM / 64]);
        for &cpu in cpus {
            set.0[cpu / 64] |= 1 << (cpu % 64);
        }
        set
    }

    /// A set of one CPU holds it and no other, at the edges of its words and of its room; a CPU it
    /// has no room for gives a set with none.
    #[test]
    fn a_set_of_one_cpu_holds_it_alone() {
        for cpu in [0, 1, 63, 64, 65, 4096, ROOM - 1] {
            let one = Cpus::one(cpu);
            assert_eq!(one.iter().collect::<Vec<_>>(), [cpu]);
            assert!(one.contains(cpu) && !one.contains(cpu + 1) && !one.contains(cpu.wrapping_sub(1)));
            assert!(one == set(&[cpu]));
        }
        for cpu in [ROOM, ROOM + 1, 1 << 20, usize::MAX] {
            assert_eq!(Cpus::one(cpu).iter().count(), 0);
            assert!(!Cpus::one(cpu).contains(cpu));
        }
        assert!(Cpus::one(3) != Cpus::one(4));
        assert!(Cpus([!0; ROOM / 64]).iter().eq(0..ROOM));
    }

    /// A CPU put in a set is in it with the CPUs the set had, at the edges of its words and of its
    /// room, and the set says whether it changed: it did not if the CPU was in it already, or if
    /// the set has no room for the CPU.
    #[test]
    fn a_cpu_put_in_a_set_is_in_it_and_the_set_says_whether_it_changed() {
        let cpus = [0, 1, 63, 64, 4096, ROOM - 1];
        let mut put = Cpus::empty();
        assert_eq!(put.iter().count(), 0);
        for (i, &cpu) in cpus.iter().enumerate() {
            assert!(put.insert(cpu), "CPU {cpu} was in the set");
            assert!(put.contains(cpu) && !put.insert(cpu));
            assert!(put == set(&cpus[..=i]));
        }
        for cpu in [ROOM, ROOM + 1, 1 << 20, usize::MAX] {
            assert!(!put.insert(cpu) && !put.contains(cpu));
        }
        assert!(put == set(&cpus));
    }

    /// Listing a set costs what the CPUs in it cost, not the room for the others: a set of four CPUs
    /// is listed in less than a quarter of the time it takes to ask the set about every CPU of the
    /// room. Each is timed as the quickest of many tries, which other work on the machine slows
    /// the same.
    #[test]
    fn a_set_is_listed_without_asking_about_each_cpu_of_the_room() {
        let few = set(&[0, 2, 64, 8191]);
        let quickest = |work: &dyn Fn() -> usize| -> Duration {
            let tries = (0..100).map(|_| {
                let began = Instant::now();
                black_box(work());
                began.elapsed()
            });
            tries.min().unwrap()
        };
        let listed = quickest(&|| black_box(&few).iter().count());
        let asked = quickest(&|| (0..ROOM).filter(|&cpu| black_box(&few).contains(cpu)).count());
        assert!(listed * 4 < asked, "listing took {listed:?} and asking about the room {asked:?}");
    }

    /// A set is listed as the system lists the CPUs a thread may run on, in
    /// `/proc/thread-self/status`: runs by their first and last CPU, the others by number.
    #[test]
    fn a_set_is_listed_as_the_system_lists_cpus() {
        let lists: [(&[usize], &str); 7] = [
            (&[], ""),
            (&[3], "3"),
            (&[0, 1], "0-1"),
            (&[0, 1, 2, 3, 4, 5], "0-5"),
            (&[0, 2, 3, 4, 8], "0,2-4,8"),
            (&[0, 1, 63, 64, 65, 8191], "0-1,63-65,8191"),
            (&[5, 7, 9], "5,7,9"),
        ];
        for (cpus, list) in lists {
            assert_eq!(set(cpus).to_string(), list);
        }
    }
}
