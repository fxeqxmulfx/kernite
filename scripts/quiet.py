"""Keeps the machine's own work off the cpus that measurements use.

The desktop renders in software, in bursts over every cpu, and whatever shares a core with a kernel
under measurement, or with one of the model's workers, slows it. `apply(cpus)` moves every thread of
every process, every interrupt and the kernel's unbound work queues onto cpus; `apply` with every
cpu undoes it.

A thread's cpus are only where it runs unless it says otherwise. make.py puts itself back on every
cpu, so builds use them all, and the benchmarks and the model's workers pin themselves where they
are meant to run. Threads the kernel binds to one cpu stay where they are. Moving other users'
threads, interrupts and the work queues takes root.
"""

import os
from pathlib import Path


def parse(text: str) -> set[int]:
    """"41,55" or "0-13,28-41" as a set of cpus."""
    cpus = set()
    for part in text.split(","):
        first, _, last = part.partition("-")
        cpus.update(range(int(first), int(last or first) + 1))
    return cpus


def apply(cpus: set[int]) -> str:
    """Every thread, interrupt and unbound work queue onto cpus, as far as permissions allow; what
    was moved."""
    listed = ",".join(map(str, sorted(cpus)))
    moved = bound = 0
    for task in Path("/proc").glob("[0-9]*/task/[0-9]*"):
        try:
            os.sched_setaffinity(int(task.name), cpus)
            moved += 1
        except OSError:  # bound to its cpu, gone, or someone else's
            bound += 1
    irqs = 0
    for irq in Path("/proc/irq").glob("[0-9]*/smp_affinity_list"):
        try:
            irq.write_text(listed)
            irqs += 1
        except OSError:  # per-cpu, managed by its driver, or no root
            pass
    mask = sum(1 << c for c in cpus)
    words = ",".join(f"{mask >> 32 * i & 0xFFFFFFFF:08x}" for i in reversed(range((os.cpu_count() + 31) // 32)))
    try:
        Path("/sys/devices/virtual/workqueue/cpumask").write_text(words)
        queues = "the unbound work queues"
    except OSError:
        queues = "not the work queues (no root)"
    return f"onto cpus {listed}: {moved} threads ({bound} not movable), {irqs} interrupts, {queues}"
