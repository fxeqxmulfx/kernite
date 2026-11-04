//! Where a grid's threads run: on the CPUs the caller gives, or, if it gives none, where the system
//! puts them within the CPUs the caller's thread may run on. Each thread says what Linux knows of
//! it in `/proc/thread-self`: the CPUs it may run on, and the one it runs on.

mod common;

use std::ffi::OsStr;
use std::path::Path;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
use std::sync::{Arc, Mutex};
use std::thread::{self, ThreadId};
use std::{env, fs, io, panic};

use common::listed;
use kernite_cpu::{Grid, cpus, launch, most};

/// The variable that makes a test the one another program runs, in a process of its own: it holds
/// the CPU the process may not run on, how many CPUs it may run on, and how many threads a grid has
/// at most there, as `6,6,6`.
const CHILD: &str = "KERNITE_CPU_CHILD";

/// Whether Linux says what each thread runs on, as the tests here need.
fn proc_says() -> bool {
    let said = Path::new("/proc/thread-self/status").exists();
    if !said {
        eprintln!("no /proc/thread-self: skipped");
    }
    said
}

/// The CPUs a thread may run on, as Linux lists them (`0-5,8`) in the `status` file at `path`.
fn allowed(path: &str) -> String {
    let status = fs::read_to_string(path).unwrap();
    status.lines().find_map(|line| line.strip_prefix("Cpus_allowed_list:")).unwrap().trim().to_string()
}

/// What Linux says of the calling thread: the CPUs it may run on, and the CPU it runs on.
fn here() -> (String, usize) {
    let stat = fs::read_to_string("/proc/thread-self/stat").unwrap();
    // The CPU is the 39th field. The name, in brackets, may hold spaces, so the fields are
    // counted from the 3rd, which follows it.
    let cpu = stat.rsplit_once(") ").unwrap().1.split(' ').nth(36).unwrap().parse().unwrap();
    (allowed("/proc/thread-self/status"), cpu)
}

/// The calling thread by Rust's id, and by Linux's, which names its directory in `/proc/self/task`.
fn who() -> (ThreadId, String) {
    let link = fs::read_link("/proc/thread-self").unwrap();
    (thread::current().id(), link.file_name().unwrap().to_str().unwrap().to_string())
}

/// What `said` says in each thread of a launch of `grid`, by the thread's index.
fn each<T: Send + 'static>(grid: Grid<'_>, said: fn() -> T) -> Vec<T> {
    let all = Arc::new(Mutex::new(vec![]));
    let into = Arc::clone(&all);
    launch(grid, move |place| into.lock().unwrap().push((place.thread(), said()))).unwrap();
    let mut all: Vec<(u64, T)> = all.lock().unwrap().drain(..).collect();
    all.sort_by_key(|(t, _)| *t);
    all.into_iter().map(|(_, said)| said).collect()
}

/// What each thread of a launch of `grid` was, by its index: the thread, and what `here` says of it.
fn seen(grid: Grid<'_>) -> Vec<(ThreadId, (String, usize))> {
    each(grid, || (thread::current().id(), here()))
}

/// How many threads a test's grids have: as many as the process may run on at once, and four at
/// most.
fn threads() -> usize {
    most().unwrap().min(cpus().unwrap().len()).min(4)
}

/// A placed grid runs thread `t` on CPU `cpus[t]` and on no other: the CPUs it may run on are that
/// one, and it runs there. The caller's thread is the first. The same threads of the grid, kept
/// from one launch to the next, move when the launch after places them elsewhere. The first
/// placement is the first CPUs in reverse, for the last thread to go to the first CPU while the
/// threads that have just started are on the CPU of the caller's thread, which is another.
#[test]
fn a_placed_grid_runs_each_thread_on_the_cpu_it_was_given() {
    if !proc_says() {
        return;
    }
    let (all, threads) = (cpus().unwrap(), threads());
    let first = all[..threads].to_vec();
    let reversed: Vec<usize> = first.iter().rev().copied().collect();
    let rotated: Vec<usize> = first.iter().cycle().skip(1).take(threads).copied().collect();
    let last = all[all.len() - threads..].to_vec();
    let mut ids: Vec<ThreadId> = vec![];
    for placement in [reversed, first.clone(), rotated, last, first] {
        let launched = seen(Grid::on(&placement));
        assert_eq!(launched.len(), threads);
        for (t, (_, (list, cpu))) in launched.iter().enumerate() {
            assert_eq!((listed(list), *cpu), (vec![placement[t]], placement[t]), "thread {t} of {placement:?}");
        }
        assert_eq!(launched[0].0, thread::current().id());
        let now: Vec<ThreadId> = launched.iter().map(|(id, _)| *id).collect();
        assert!(ids.is_empty() || ids == now, "the threads of the grid are the ones kept: {ids:?} then {now:?}");
        ids = now;
    }
}

/// A grid that is not placed leaves its threads to the system, on the CPUs the caller's thread may
/// run on, whatever the launches before it did: the threads an earlier launch placed go back, the
/// caller's and every kept one, those a grid of the caller's thread alone does not run, too. The
/// threads that a launch not placed starts, after one that placed the caller's thread alone, start
/// there too.
#[test]
fn a_grid_that_is_not_placed_runs_where_the_caller_may_whatever_launched_before() {
    if !proc_says() {
        return;
    }
    let (caller, _) = here();
    let (all, threads) = (cpus().unwrap(), threads());
    let placement = all[..threads].to_vec();
    launch(Grid::on(&placement[..1]), |_| {}).unwrap();
    for placed in [false, true, false, false, true, true, false] {
        let grid = if placed { Grid::on(&placement) } else { Grid::new(threads) };
        for (t, (_, (list, cpu))) in seen(grid).iter().enumerate() {
            match placed {
                true => assert_eq!(listed(list), [placement[t]], "thread {t}, placed"),
                false => assert_eq!(*list, caller, "thread {t}, not placed"),
            }
            assert!(listed(list).contains(cpu), "thread {t} runs on CPU {cpu} of {list}");
        }
    }
    let kept = each(Grid::on(&placement), who);
    seen(Grid::new(1));
    assert_eq!(here().0, caller, "the caller's thread after a grid of it alone that is not placed");
    for (t, (_, tid)) in kept.iter().enumerate().skip(1) {
        let list = allowed(&format!("/proc/self/task/{tid}/status"));
        assert_eq!(list, caller, "kept thread {t} after a grid of the caller's thread alone that is not placed");
    }
}

/// A placed launch leaves the caller's thread on `cpus[0]`, whether the grid has one thread or many
/// and whether the kernel panics, and a thread it starts meanwhile starts there. The next launch of
/// the same placement finds it there, one that places it elsewhere moves it, and one that places none
/// puts it back on the CPUs it could run on before, where a thread it starts starts again.
#[test]
fn the_callers_thread_stays_on_the_cpu_of_its_grid_until_a_launch_places_it_elsewhere_or_places_none() {
    if !proc_says() {
        return;
    }
    let (before, _) = here();
    let all = cpus().unwrap();
    for threads in 1..=threads() {
        // From the last CPU back, so that the caller is on another one, mostly.
        let placement: Vec<usize> = all.iter().rev().take(threads).copied().collect();
        let at = placement[0].to_string();
        let grid = Grid::on(&placement);
        let launched = seen(grid);
        assert_eq!(launched[0].1.0, at, "{threads} threads: the caller's thread in the launch");
        assert_eq!(here().0, at, "{threads} threads: the caller's thread after it");
        assert_eq!(thread::spawn(here).join().unwrap().0, at, "{threads} threads: a thread it starts");
        // `resume_unwind` panics without the message a hook would print.
        let panicked = panic::catch_unwind(|| {
            launch(grid, |place| {
                if place.thread() == 0 {
                    panic::resume_unwind(Box::new("the first thread panics"));
                }
            })
        });
        assert!(panicked.is_err(), "{threads} threads: the panic in the first thread ends the launch in the caller");
        assert_eq!(here().0, at, "{threads} threads: the caller's thread after a panic");
        assert_eq!(seen(grid)[0].1.0, at, "{threads} threads: the caller's thread in the next launch of it");

        // From the first CPU up, which is the last's if the caller's thread may run on as many CPUs
        // as the grid has threads.
        let elsewhere: Vec<usize> = all.iter().take(threads).copied().collect();
        let launched = seen(Grid::on(&elsewhere));
        assert_eq!(launched[0].1.0, elsewhere[0].to_string(), "{threads} threads: the caller's thread moved");
        assert_eq!(here().0, elsewhere[0].to_string(), "{threads} threads: after the move");

        seen(Grid::new(threads));
        assert_eq!(here().0, before, "{threads} threads: the caller's thread after a launch that places none");
        assert_eq!(thread::spawn(here).join().unwrap().0, before, "{threads} threads: a thread it starts then");
    }
}

/// `cpus` of a thread that a placed launch has left on one CPU lists the CPUs it could run on before,
/// and a placement is checked against those: `Grid::on(&cpus()?[..n])` gives the same grid in launch
/// after launch.
#[test]
fn a_thread_a_placed_launch_keeps_on_one_cpu_lists_the_cpus_it_could_run_on_before() {
    if !proc_says() {
        return;
    }
    let (before, _) = here();
    let n = threads();
    for round in 0..3 {
        let all = cpus().unwrap();
        assert_eq!(all, listed(&before), "round {round}: the CPUs to choose from");
        for (t, (_, (list, cpu))) in seen(Grid::on(&all[..n])).iter().enumerate() {
            assert_eq!((listed(list), *cpu), (vec![all[t]], all[t]), "round {round}: thread {t}");
        }
        assert_eq!(here().0, all[0].to_string(), "round {round}: the caller's thread after it");
    }
}

/// A program that moves the caller's thread itself between two launches puts it where kernite does
/// not know: a launch that places it on the CPU kernite left it on does not move it, one that places
/// it on another does, and one that places none puts it back on the CPUs it could run on before.
/// `taskset` moves the thread, by the number Linux gives it.
#[test]
fn a_thread_the_program_moves_itself_is_moved_by_a_launch_only_to_another_cpu_or_back() {
    if !proc_says() {
        return;
    }
    let (before, _) = here();
    let all = cpus().unwrap();
    if all.len() < 3 {
        return eprintln!(
            "the caller's thread may run on fewer than three CPUs, so there is no CPU to move to: skipped"
        );
    }
    let (left, program, another) = (all[0], all[1], all[2]);
    assert_eq!(seen(Grid::on(&[left]))[0].1.0, left.to_string());
    let moved = Command::new("taskset").args(["-cp", &program.to_string(), &who().1]).output();
    match moved {
        Ok(out) => assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr)),
        Err(e) if e.kind() == io::ErrorKind::NotFound => return eprintln!("no taskset here: skipped"),
        Err(e) => panic!("taskset did not run: {e}"),
    }
    assert_eq!(here().0, program.to_string(), "the program moved the thread");
    assert_eq!(seen(Grid::on(&[left]))[0].1.0, program.to_string(), "a launch on the CPU kernite left it on");
    assert_eq!(seen(Grid::on(&[another]))[0].1.0, another.to_string(), "a launch on another CPU");
    seen(Grid::new(1));
    assert_eq!(here().0, before, "a launch that places none");
}

/// A placement the caller's thread cannot keep is an error that says where, and nothing runs or
/// moves: no thread runs the kernel, the caller's thread stays where it is, and the threads kept stay
/// where they were placed and are the ones that run the next launch. It is so for a thread that no
/// launch has placed, whose CPUs are read, and for one a launch has left on a CPU, whose CPUs a
/// launch before it noted; for a placement of as many threads as the launch before it gave, which
/// differs in one; and for a placement given twice in a row, which the second launch does not take
/// as checked because the first was.
#[test]
fn a_placement_the_callers_thread_cannot_keep_is_an_error_and_nothing_moves() {
    if !proc_says() {
        return;
    }
    let (caller, _) = here();
    let (all, threads) = (cpus().unwrap(), threads());
    // The grid's threads are kept, each on a CPU that the placements below do not give it.
    let at: Vec<usize> = all.iter().cycle().skip(1).take(threads).copied().collect();
    let outside = all.last().unwrap() + 1;
    let may = format!("and the caller's thread may run on CPUs {caller}");
    let mut errors = vec![
        (vec![outside], format!("thread 0 of the grid is placed on CPU {outside}, {may}")),
        (vec![1 << 20], format!("thread 0 of the grid is placed on CPU 1048576, {may}")),
    ];
    let twice = "and a thread that waits needs a CPU of its own";
    if threads >= 2 {
        errors.push((vec![all[0], outside], format!("thread 1 of the grid is placed on CPU {outside}, {may}")));
        errors.push((
            vec![all[1], all[1]],
            format!("threads 0 and 1 of the grid are both placed on CPU {}, {twice}", all[1]),
        ));
        // As many threads as the placement the threads are kept on, which differs in the last.
        let last = threads - 1;
        let mut gone = at.clone();
        gone[last] = outside;
        errors.push((gone, format!("thread {last} of the grid is placed on CPU {outside}, {may}")));
        let mut doubled = at.clone();
        doubled[last] = at[0];
        errors.push((doubled, format!("threads 0 and {last} of the grid are both placed on CPU {}, {twice}", at[0])));
    }
    if threads >= 3 {
        errors.push((
            vec![all[2], all[1], all[2]],
            format!("threads 0 and 2 of the grid are both placed on CPU {}, {twice}", all[2]),
        ));
    }
    let mut kept = vec![];
    for placed_before in [false, true] {
        if placed_before {
            kept = each(Grid::on(&at), who);
        }
        let here_now = here().0;
        for (placement, message) in &errors {
            for _ in 0..2 {
                let ran = Arc::new(AtomicUsize::new(0));
                let into = Arc::clone(&ran);
                let error = launch(Grid::on(placement), move |_| _ = into.fetch_add(1, SeqCst)).unwrap_err();
                assert_eq!(error.message, *message, "{placement:?}, after a launch: {placed_before}");
                assert_eq!(ran.load(SeqCst), 0, "{placement:?}: a thread ran the kernel");
                assert_eq!(here().0, here_now, "{placement:?}: the caller's thread moved");
            }
            for (t, (_, tid)) in kept.iter().enumerate().skip(1) {
                let moved = allowed(&format!("/proc/self/task/{tid}/status"));
                assert_eq!(moved, at[t].to_string(), "{placement:?}: thread {t} moved");
            }
            if placed_before {
                assert_eq!(
                    each(Grid::on(&at), who),
                    kept,
                    "{placement:?}: the threads kept are not the ones that run next"
                );
            }
        }
    }
    let nobody = launch(Grid::on(&[]), |_| {}).unwrap_err();
    assert_eq!(nobody.message, "a grid has one thread at least, and this has none");
}

/// Grids launched from two threads at once are not checked against each other: each places its
/// thread on one CPU, and both launches run there at once, since each kernel waits for the other.
#[test]
fn grids_launched_at_once_are_not_checked_against_each_other() {
    if !proc_says() {
        return;
    }
    let cpu = cpus().unwrap()[0];
    let meet = Arc::new(std::sync::Barrier::new(2));
    let ran_on = Arc::new(Mutex::new(vec![]));
    let callers: Vec<_> = (0..2)
        .map(|_| {
            let (meet, ran_on) = (Arc::clone(&meet), Arc::clone(&ran_on));
            thread::spawn(move || {
                launch(Grid::on(&[cpu]), move |_| {
                    meet.wait();
                    ran_on.lock().unwrap().push(here().1);
                })
                .unwrap();
            })
        })
        .collect();
    for caller in callers {
        caller.join().unwrap();
    }
    assert_eq!(*ran_on.lock().unwrap(), [cpu, cpu]);
}

/// `cpus` lists the CPUs the calling thread may run on, as Linux says it does, and a grid has at
/// most as many threads: `most` counts the CPUs of the mask, or fewer.
#[test]
fn the_cpus_are_those_the_system_says_the_thread_may_run_on_and_most_counts_them() {
    if !proc_says() {
        return;
    }
    let all = cpus().unwrap();
    assert_eq!(all, listed(&here().0));
    assert!(all.windows(2).all(|pair| pair[0] < pair[1]), "{all:?}");
    assert!(most().unwrap() <= all.len(), "{} threads at most, of {} CPUs", most().unwrap(), all.len());
    // A thread that starts with the mask the caller's thread has, as the grid's do, may run on those.
    let from_a_thread = thread::spawn(|| cpus().unwrap()).join().unwrap();
    assert_eq!(from_a_thread, all);
}

/// Runs the test `name` again, in a process of its own that the program `tool` starts, given `args`
/// before the test binary, and `child` for `CHILD` to hold. It says whether that process ended well,
/// and what it printed; none if there is no `tool`.
fn again(tool: &OsStr, args: &[String], name: &str, child: String) -> Option<(bool, String)> {
    let ran = Command::new(tool)
        .args(args)
        .arg(env::current_exe().unwrap())
        .args(["--exact", name, "--nocapture", "--test-threads=1"])
        .env(CHILD, child)
        .output();
    match ran {
        Ok(out) => {
            let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
            Some((out.status.success(), text))
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => None,
        Err(e) => panic!("{} did not run: {e}", tool.to_string_lossy()),
    }
}

/// A process `numactl` keeps to some of the machine's CPUs runs a grid on those: `cpus` and `most`
/// count them, a grid of more threads than `most` is an error that says what `most` counts, so is
/// one placed on a CPU outside them, and a grid that is not placed runs on them all. A grid placed
/// on them runs each thread on its own.
#[test]
fn a_process_kept_to_some_cpus_by_numactl_runs_its_grids_on_those() {
    let name = "a_process_kept_to_some_cpus_by_numactl_runs_its_grids_on_those";
    if let Ok(said) = env::var(CHILD) {
        return limited_to_some(&said);
    }
    if !proc_says() {
        return;
    }
    let all = cpus().unwrap();
    if all.len() < 2 {
        return eprintln!("the caller's thread may run on one CPU, so numactl can keep nothing off a CPU: skipped");
    }
    // Six CPUs, the first of the mask, if the mask has a seventh to keep the process off.
    let kept = &all[..all.len().min(7) - 1];
    let list: Vec<String> = kept.iter().map(ToString::to_string).collect();
    let numactl = env::var_os("NUMACTL").unwrap_or_else(|| "numactl".into());
    let child = format!("{},{},{}", all[kept.len()], kept.len(), most().unwrap().min(kept.len()));
    let Some((ended_well, text)) = again(&numactl, &[format!("--physcpubind={}", list.join(","))], name, child) else {
        return eprintln!("no numactl here: skipped; set NUMACTL to run it");
    };
    assert!(ended_well && text.contains("1 passed"), "{text}");
}

/// How many CPUs the quota of the test below lets the process use.
const QUOTA: usize = 2;

/// A process a cgroup's CPU quota limits to two CPUs may still run on every CPU its thread may:
/// `cpus` lists all of them, `most` counts the two, and a grid of more threads than that is an
/// error that says what `most` counts, as README does: the CPUs the caller's thread may run on, or
/// fewer where a cgroup limits the process to fewer. `systemd-run` runs the test again in a scope
/// of the user's systemd manager with a quota of 200%; where there is no `systemd-run`, or it does
/// not start the test, or its manager sets no quota, the test says so and is skipped.
#[test]
fn a_process_a_cgroup_limits_to_some_cpus_runs_grids_of_that_many_threads_at_most() {
    let name = "a_process_a_cgroup_limits_to_some_cpus_runs_grids_of_that_many_threads_at_most";
    if let Ok(said) = env::var(CHILD) {
        // A manager with no `cpu` controller to give a scope sets no quota.
        let system = thread::available_parallelism().unwrap().get();
        if system != QUOTA {
            return eprintln!(
                "the quota of {QUOTA} CPUs is not the limit here, for the system says the process runs {system} at \
                 once: skipped"
            );
        }
        return limited_to_some(&said);
    }
    if !proc_says() {
        return;
    }
    let all = cpus().unwrap();
    if all.len() <= QUOTA {
        return eprintln!(
            "the caller's thread may run on {QUOTA} CPUs or fewer, so a quota of {QUOTA} limits nothing: skipped"
        );
    }
    let percent = format!("CPUQuota={}%", 100 * QUOTA);
    let quota = ["--user", "--scope", "--quiet", "-p", &percent].map(String::from);
    let child = format!("{},{},{QUOTA}", all.last().unwrap() + 1, all.len());
    let Some((ended_well, text)) = again("systemd-run".as_ref(), &quota, name, child) else {
        return eprintln!("no systemd-run here: skipped");
    };
    // libtest says it runs the test, if `systemd-run` started it.
    if !text.contains("running 1 test") {
        return eprintln!(
            "systemd-run did not start the test, which needs the user's systemd: skipped: {}",
            text.trim()
        );
    }
    // The process says so, if it found no quota.
    if let Some(skipped) = text.lines().find(|line| line.ends_with(": skipped")) {
        return eprintln!("{skipped}");
    }
    assert!(ended_well && text.contains("1 passed"), "{text}");
}

/// The body of the two tests above, in the process each runs again, as `said` tells it: the thread
/// may run on `all` CPUs, none of them `outside`, and a grid has `expected` threads at most there.
fn limited_to_some(said: &str) {
    let numbers: Vec<usize> = said.split(',').map(|n| n.parse().unwrap()).collect();
    let [outside, all, expected] = numbers[..] else { panic!("{CHILD} is `{said}`") };
    let (list, _) = here();
    let kept = cpus().unwrap();
    assert_eq!(kept, listed(&list));
    assert_eq!(kept.len(), all, "the CPUs {list} are not the {all} the thread may run on");
    assert!(!kept.contains(&outside), "CPU {outside} is outside {list}");
    let most = most().unwrap();
    assert_eq!(most, expected, "the CPUs {list} are {} in all", kept.len());
    let too_many = launch(Grid::new(most + 1), |_| {}).unwrap_err();
    assert_eq!(
        too_many.message,
        format!(
            "the grid has {} threads, and the process runs {most} at once: the CPUs the caller's thread may run \
             on, or fewer where a cgroup limits the process to fewer",
            most + 1
        )
    );
    let off = launch(Grid::on(&[outside]), |_| {}).unwrap_err();
    assert_eq!(
        off.message,
        format!("thread 0 of the grid is placed on CPU {outside}, and the caller's thread may run on CPUs {list}")
    );
    for (t, (_, (threads, _))) in seen(Grid::new(most)).iter().enumerate() {
        assert_eq!(*threads, list, "thread {t}, not placed");
    }
    let placement = &kept[..most];
    for (t, (_, (list, cpu))) in seen(Grid::on(placement)).iter().enumerate() {
        assert_eq!((listed(list), *cpu), (vec![placement[t]], placement[t]), "thread {t}, placed");
    }
}
