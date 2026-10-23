"""Shared setup for the checks on downstream_amp.py runs (gate_opening.py, amp_neurons.py, amp_opener_shape.py,
amp_pc_sparsity.py, amp_transcoder.py, amp_basis_alignment.py).

setup(args) runs downstream_amp.py up to its fitting loop with the given arguments and returns its globals (model adapter,
fitting / held-out rows, whitening, sweep, coords, ...). from_run(run) does the same with the arguments saved in a run's .pt,
so a check always sees exactly the rows, whitening and τ that the run was fitted and checked on.
Each check writes its output to downstream_amp/checks/<check>_<run>.log as well as to stdout.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent
SCRIPT = ROOT_DIR / "downstream_amp.py"
RUNS = ROOT_DIR / "downstream_amp"
CHECKS = RUNS / "checks"
FIT_LOOP = "g0 = torch.Generator"           # first line of downstream_amp.py after held-out τ is computed
BEFORE_TAU = "tau_f, tau_h = "              # first line that computes τ (setups that do not need it stop here)


def setup(args, stop=FIT_LOOP):
    argv = [str(SCRIPT)]
    for k, v in args.items():
        if v not in ("", None):
            argv += [f"--{k.replace('_', '-')}", str(v)]
    saved, sys.argv = sys.argv, argv
    g = {"__file__": str(SCRIPT), "__name__": "amp_setup"}
    try:
        src = SCRIPT.read_text()
        exec(compile(src[:src.index(stop)], str(SCRIPT), "exec"), g)
    finally:
        sys.argv = saved
    return SimpleNamespace(**g)


def load_run(run):
    path = Path(run)
    if not path.exists():
        path = RUNS / run
    return path, torch.load(path, weights_only=False)


def from_run(run, stop=FIT_LOOP, **override):
    path, res = load_run(run)
    return path, res, setup({**res["args"], **override}, stop)


def openers(res, k=6):
    """Indices of the run's reported attractors (at most k) and whether each recurred (basin ≥ 2 starts)."""
    reps = list(res["reps"])[:k]
    basin = np.bincount(res["label"])
    return reps, [bool(basin[res["label"][j]] >= 2) for j in reps]


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


def log_to(name):
    CHECKS.mkdir(parents=True, exist_ok=True)
    path = CHECKS / f"{name}.log"
    sys.stdout = _Tee(sys.__stdout__, open(path, "w"))
    return path
