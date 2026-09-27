#!/usr/bin/env python3
"""Everything this repository runs, one target each.

    ./make.py check                 formatting, clippy and every test
    ./make.py fmt                   formats every crate
    ./make.py speed KERNEL=ahead3   the generated kernels alone on cpu 0
    ./make.py stat KERNEL=ahead3    the same under perf stat: EVENTS, or cycles, uops and ports
    ./make.py cycles                each kernel's weights a cycle, runs whose core was shared set aside
    ./make.py ports KERNELS=ahead3  its uops a superblock by port, and the frontend's

The measuring targets move your threads onto SYSTEM (41,55) while they run, and back after.

An argument NAME=VALUE sets a variable for the targets and the environment of everything they run.
With no target, the targets are listed.
"""

import contextlib
import os
import re
import shlex
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))

import percycle
import quiet as quieting

TARGETS = {}
PORTS = ",".join(f"uops_dispatched_port.port_{p}" for p in range(8))
EVENTS = f"cycles,instructions,uops_issued.any,uops_executed.thread,idq_uops_not_delivered.core,{PORTS}"


def target(name: str):
    """Registers a target; the first line of its docstring is its line in `help`."""
    def register(fn):
        TARGETS[name] = fn
        return fn
    return register


def var(name: str, default: str = "") -> str:
    """A variable as an argument or the environment gives it, else default."""
    return os.environ.get(name) or default


def sh(cmd: list, env: dict | None = None, capture: bool = False, echo: bool = True) -> str:
    """Runs cmd in a process group of its own, killed if this script ends first. cmd failing ends
    this script with its status. Returns what it printed, stderr included, when capture is set."""
    if echo:
        print(shlex.join([*(f"{k}={v}" for k, v in (env or {}).items()), *map(str, cmd)]))
    sys.stdout.flush()
    out = subprocess.PIPE if capture else None
    child = subprocess.Popen(cmd, env=dict(os.environ, **(env or {})), stdin=subprocess.DEVNULL, stdout=out,
                             stderr=subprocess.STDOUT if capture else None, text=True, process_group=0)
    try:
        said, _ = child.communicate()
    except BaseException:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()
        raise
    if child.returncode:
        print(said or "", end="")
        sys.exit(f"{shlex.join(map(str, cmd))}: status {child.returncode}")
    return said or ""


@target("help")
def usage():
    """this list"""
    for name, fn in TARGETS.items():
        print(f"  ./make.py {name:<6} {fn.__doc__.splitlines()[0]}")


@target("check")
def check():
    """formatting, clippy on every target, and every test"""
    sh(["cargo", "fmt", "--check"])
    sh(["cargo", "clippy", "-q", "--all-targets", "--", "-D", "warnings"])
    sh(["cargo", "test", "-q"])


@target("fmt")
def fmt():
    """formats every crate"""
    sh(["cargo", "fmt"])


@target("bless")
def bless():
    """rewrites snapshots/ from the kernels as they now are"""
    sh(["cargo", "test", "-q", "-p", "kgen", "--test", "snapshots"], env={"KGEN_BLESS": "1"})


@contextlib.contextmanager
def quieted():
    """Your threads, and with root the rest of the machine's work, on SYSTEM (41,55) while cpu 0 is
    measured, and back onto every cpu after."""
    print(quieting.apply(quieting.parse(var("SYSTEM", "41,55"))))
    try:
        yield
    finally:
        print(quieting.apply(set(range(os.cpu_count()))))


def speed_cmd() -> list[str]:
    example = var("SPEED_EXAMPLE", "speed")
    sh(["cargo", "build", "-q", "--release", "--example", example])
    return ["taskset", "-c", var("CPU", "0"), f"target/release/examples/{example}"]


@target("speed")
def speed():
    """the generated kernels alone on CPU (0): KERNEL picks one, ROWS, COLS, SECS"""
    cmd = speed_cmd()
    with quieted():
        sh(cmd)


@target("stat")
def stat():
    """speed under perf stat, EVENTS or cycles, uops and the ports"""
    cmd = speed_cmd()
    with quieted():
        said = sh(["perf", "stat", "-e", var("EVENTS", EVENTS), *cmd], capture=True)
    print("\n".join(line for line in said.splitlines() if line and "Performance counter" not in line))


def kernels(exe: list) -> list[str]:
    """KERNELS, or every kernel exe runs."""
    return var("KERNELS").split() or sh([*exe], env={"KERNEL": "?"}, capture=True, echo=False).split()


def counted(exe: list, cpu: str, kernel: str, events: list = ()) -> tuple:
    """One run of kernel under percycle's counters."""
    with tempfile.TemporaryDirectory() as tmp:
        run = ["env", f"KERNEL={kernel}", f"SECS={var('SECS', '1')}", *exe]
        said = sh(percycle.command(run, cpu, Path(tmp), events), capture=True, echo=False)
        return percycle.result(said, Path(tmp))


@target("ports")
def ports():
    """each kernel's, or KERNELS', uops a superblock by port, and the frontend's, on CPU (0)"""
    cmd, cpu = speed_cmd(), var("CPU", "0")
    with quieted():
        for kernel in kernels(cmd[3:]):
            print(kernel)
            print(percycle.ports(lambda events: counted(cmd[3:], cpu, kernel, events)))


@target("cycles")
def cycles():
    """each kernel, or KERNELS, alone on CPU (0), REPS rounds of SECS (1) each: weights a cycle"""
    cmd, cpu = speed_cmd(), var("CPU", "0")
    names, runs = kernels(cmd[3:]), int(var("REPS", "5"))
    kept = {name: [] for name in names}
    with quieted():
        for _ in range(runs):
            for name in names:
                line, rate, busy, per = counted(cmd[3:], cpu, name)
                print(percycle.row(rate, busy, f"{per['instructions']:6.1f} instructions a superblock | {line}"))
                if busy <= percycle.BUSY:
                    kept[name].append(rate)
    for name in names:
        print(percycle.summary(name, kept[name], runs))


def main(args: list[str]) -> None:
    sys.stdout.reconfigure(line_buffering=True)
    for sig in signal.SIGTERM, signal.SIGHUP:
        signal.signal(sig, lambda signum, _: sys.exit(128 + signum))
    os.chdir(ROOT)
    # Back onto every cpu, from wherever a quiet machine put the shell: what this starts, builds above
    # all, should use them all, and what must run in one place pins itself there.
    os.sched_setaffinity(0, range(os.cpu_count()))
    names = []
    for arg in args:
        name, eq, value = arg.partition("=")
        if eq and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            os.environ[name] = value
        elif arg in TARGETS:
            names.append(arg)
        else:
            sys.exit(f"no target {arg!r}; ./make.py help lists them")
    for name in names or ["help"]:
        TARGETS[name]()


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except KeyboardInterrupt:
        sys.exit(130)
