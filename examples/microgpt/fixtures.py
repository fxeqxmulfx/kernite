#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# ///
"""Writes what Python gives microgpt, for the Rust port to be held to.

- `fixtures/random.txt`: draws of Python's `random`, one kind a line.
- `fixtures/docs.txt`: the first thousand names in the order microgpt.py shuffles them, the
  ones its thousand steps train on. It reads `input.txt`, and downloads it as microgpt.py
  does if it is not here.
- `fixtures/python.txt`, with `--python`: what microgpt.py itself prints, run from the gist.
  It takes Python some minutes.
"""

import pathlib
import random
import subprocess
import sys
import tempfile
import urllib.request

HERE = pathlib.Path(__file__).parent
NAMES = "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt"
GIST = "https://gist.githubusercontent.com/karpathy/8627fe009c40f57531cb18360106ce95/raw/microgpt.py"


def write(name, text):
    path = HERE / "fixtures" / name
    path.write_text(text)
    print(f"wrote {path}")


lines = []
random.seed(42)
lines.append(["random", *(repr(random.random()) for _ in range(6))])
random.seed(42)
lines.append(["bits", *(str(random.getrandbits(k)) for k in (1, 5, 15, 31, 32))])
random.seed(42)
lines.append(["gauss", *(repr(random.gauss(0, 0.08)) for _ in range(7))])
random.seed(42)
weights = [0.1, 0.25, 0.0, 0.4, 0.25]
lines.append(["choices", *(str(random.choices(range(5), weights=weights)[0]) for _ in range(24))])
# What microgpt.py draws, in its order: the shuffle of its 32033 names, which depends only on
# how many there are, then the weights.
random.seed(42)
order = list(range(32033))
random.shuffle(order)
lines.append(["shuffle", *map(str, order[:12])])
weights = [random.gauss(0, 0.08) for _ in range(4192)]
lines.append(["weights", *(repr(w) for w in weights[:4]), repr(weights[-1]), repr(sum(weights))])
write("random.txt", "".join(" ".join(line) + "\n" for line in lines))

input_txt = HERE / "input.txt"
if not input_txt.exists():
    urllib.request.urlretrieve(NAMES, input_txt)
docs = [line.strip() for line in open(input_txt) if line.strip()]
random.seed(42)
random.shuffle(docs)
write("docs.txt", "".join(doc + "\n" for doc in docs[:1000]))

if "--python" in sys.argv:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        urllib.request.urlretrieve(GIST, tmp / "microgpt.py")
        (tmp / "input.txt").write_bytes(input_txt.read_bytes())
        run = subprocess.run([sys.executable, "microgpt.py"], cwd=tmp, capture_output=True, check=True)
    # Each step ends its line with a carriage return, to print over the one before.
    write("python.txt", run.stdout.decode().replace("\r", "\n"))
