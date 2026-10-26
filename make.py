#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# ///
"""Everything this repository runs, one target each.

    ./make.py check                 formatting, clippy, every test, and the docs
    ./make.py fmt                   formats every crate

An argument NAME=VALUE sets a variable in the environment of everything the targets run.
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


def target(name: str):
    """Registers a target; the first line of its docstring is its line in `help`."""
    def register(fn):
        TARGETS[name] = fn
        return fn
    return register


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
    """formatting, clippy on every target, every test, and the docs, without a warning"""
    sh(["cargo", "fmt", "--check"])
    sh(["cargo", "clippy", "-q", "--all-targets", "--", "-D", "warnings"])
    sh(["cargo", "test", "-q"])
    sh(["cargo", "doc", "-q", "--no-deps", "--workspace"], env={"RUSTDOCFLAGS": "-D warnings"})


@target("fmt")
def fmt():
    """formats every crate"""
    sh(["cargo", "fmt"])


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
