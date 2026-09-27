#!/usr/bin/env python3
"""Everything this repository runs, one target each.

    ./make.py check                 formatting, clippy and every test
    ./make.py speed AHEAD=3         the generated kernels alone on cpu 0
    ./make.py stat AHEAD=3          the same under perf stat: EVENTS, or cycles, uops and ports

An argument NAME=VALUE sets a variable for the targets and the environment of everything they run.
With no target, the targets are listed.
"""

import os
import re
import shlex
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
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


def sh(cmd: list, env: dict | None = None, capture: bool = False) -> str:
    """Runs cmd in a process group of its own, killed if this script ends first. cmd failing ends
    this script with its status. Returns what it printed, stderr included, when capture is set."""
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


@target("bless")
def bless():
    """rewrites snapshots/ from the kernels as they now are"""
    sh(["cargo", "test", "-q", "-p", "kgen", "--test", "snapshots"], env={"KGEN_BLESS": "1"})


def speed_cmd() -> list[str]:
    sh(["cargo", "build", "-q", "--release", "--example", "speed"])
    return ["taskset", "-c", var("CPU", "0"), "target/release/examples/speed"]


@target("speed")
def speed():
    """the generated kernels alone on CPU (0): AHEAD picks one, ROWS, COLS, SECS"""
    sh(speed_cmd())


@target("stat")
def stat():
    """speed under perf stat, EVENTS or cycles, uops and the ports"""
    said = sh(["perf", "stat", "-e", var("EVENTS", EVENTS), *speed_cmd()], capture=True)
    print("\n".join(line for line in said.splitlines() if line and "Performance counter" not in line))


def main(args: list[str]) -> None:
    sys.stdout.reconfigure(line_buffering=True)
    for sig in signal.SIGTERM, signal.SIGHUP:
        signal.signal(sig, lambda signum, _: sys.exit(128 + signum))
    os.chdir(ROOT)
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
