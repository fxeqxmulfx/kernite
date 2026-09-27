"""A pinned run's rate per cycle of its own, and whether the other hyperthread of its core stayed idle.

The desktop here renders in software, in bursts over every cpu. A burst on the benchmark's sibling
shares the core with it, and a burst anywhere lowers the clock. Per cycle of its own, a kernel's
rate does not depend on the clock. The sibling's cycles over the same time show whether the core
was shared: a run whose sibling was busy for more than BUSY of it is set aside.
"""

import re
import statistics
from pathlib import Path

BUSY = 0.01


def sibling(cpu: str) -> str:
    """The other hyperthread of cpu's core."""
    ids = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list").read_text()
    return next(c for c in re.split(r"[,-]", ids.strip()) if c != cpu)


# Three events a run besides cycles and instructions: with the NMI watchdog holding a counter, that
# is as many as a hyperthread's four general counters take without perf multiplexing them.
GROUPS = [
    ["uops_dispatched_port.port_0", "uops_dispatched_port.port_1", "uops_dispatched_port.port_5"],
    ["uops_dispatched_port.port_6", "uops_dispatched_port.port_2", "uops_dispatched_port.port_3"],
    ["uops_dispatched_port.port_4", "uops_dispatched_port.port_7", "uops_issued.any"],
    ["uops_executed.thread", "uops_retired.retire_slots", "idq_uops_not_delivered.core"],
    ["resource_stalls.any", "resource_stalls.rs", "ld_blocks_partial.address_alias"],
    ["idq.dsb_uops", "idq.mite_uops", "lsd.uops"],
]


def counts(csv: Path) -> dict[str, int]:
    """Each event's count in perf stat -x, output. An event perf multiplexed ends the script: its
    count would be an estimate."""
    found = {}
    for line in csv.read_text().splitlines():
        fields = line.split(",")
        if len(fields) > 4 and fields[0].isdigit():
            if fields[4] and float(fields[4]) < 99.9:
                raise SystemExit(f"{fields[2]} was counted {fields[4]}% of the time")
            found[fields[2]] = int(fields[0])
    return found


def command(cmd: list, cpu: str, tmp: Path, events: list = ()) -> list:
    """cmd pinned to cpu under two perf stats that write into tmp: one counts its own cycles and
    events in user mode, the other every cycle of its sibling over the same time."""
    own = ",".join(["cycles:u", "instructions:u", *(f"{e}:u" for e in events)])
    return ["perf", "stat", "-x,", "-o", tmp / "sibling", "-C", sibling(cpu), "-e", "cycles", "--",
            "perf", "stat", "-x,", "-o", tmp / "own", "-e", own, "--",
            "taskset", "-c", cpu, *cmd]


def result(said: str, tmp: Path) -> tuple[str, float, float, dict[str, float]]:
    """The last line a command run printed, which ends in "N weights"; its weights per cycle; its
    sibling's cycles as a share of its own; and each of its events per superblock of 256 weights."""
    line = said.strip().splitlines()[-1]
    weights = int(re.search(r"(\d+) weights", line)[1])
    own = counts(tmp / "own")
    per = {event.removesuffix(":u"): n * 256 / weights for event, n in own.items()}
    return line, weights / own["cycles:u"], counts(tmp / "sibling")["cycles"] / own["cycles:u"], per


def ports(run) -> str:
    """Every event of GROUPS per superblock, a run of run(events) each, retried while its core was
    shared; cycles and instructions from the first."""
    table = {}
    for events in GROUPS:
        for _ in range(5):
            line, rate, busy, per = run(events)
            if busy <= BUSY:
                break
        else:
            raise SystemExit(f"{events}: the core was shared in 5 runs")
        for event, n in per.items():
            table.setdefault(event, n)
    return "\n".join(f"{event:34} {n:8.2f}" for event, n in table.items())


def row(rate: float, busy: float, line: str) -> str:
    aside = "  set aside" if busy > BUSY else ""
    return f"{rate:.4f} weights a cycle, sibling busy {busy:6.2%}{aside:11} | {line}"


def summary(name: str, kept: list[float], runs: int) -> str:
    if not kept:
        return f"{name}: all {runs} runs set aside"
    return (f"{name}: {statistics.median(kept):.4f} weights a cycle, median of {len(kept)} of {runs} runs "
            f"({min(kept):.4f}-{max(kept):.4f})")
