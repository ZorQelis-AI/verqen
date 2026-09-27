"""Multi-GPU helpers (torchrun + DDP) that also run unchanged on one GPU, a CPU, or Windows.

Rules baked in from past DDP postmortems (SoreQen/LESSONS.md, NexReason DDP forensics):

* Every rank must issue the same collectives in the same order. Anything that decides
  control flow (stop now? which LR?) is decided on rank 0 and broadcast, never computed
  per rank from local wall-clock timing.
* A crash on one rank must print its own traceback *before* any teardown barrier,
  otherwise the barrier deadlocks and the real error is never shown.
* Rank 0 alone writes files; saves are atomic (write to a temp dir, then rename).
"""

from __future__ import annotations

import contextlib
import os
import shutil
import sys
import traceback
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class DistInfo:
    rank: int = 0
    world: int = 1
    local_rank: int = 0
    device: torch.device = torch.device("cpu")

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def enabled(self) -> bool:
        return self.world > 1


def init(prefer_cuda: bool = True) -> DistInfo:
    """Initialize from torchrun's env vars; a plain `python` launch is a world of one."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    use_cuda = prefer_cuda and torch.cuda.is_available()
    device = torch.device(f"cuda:{local}") if use_cuda else torch.device("cpu")
    if use_cuda:
        torch.cuda.set_device(device)
    if world > 1 and not dist.is_initialized():
        backend = "nccl" if use_cuda and sys.platform != "win32" else "gloo"
        dist.init_process_group(backend=backend, rank=rank, world_size=world,
                                device_id=device if backend == "nccl" else None)
    return DistInfo(rank, world, local, device)


def cleanup(info: DistInfo) -> None:
    if info.enabled and dist.is_initialized():
        dist.destroy_process_group()


@contextlib.contextmanager
def guarded(info: DistInfo):
    """Print this rank's traceback first, then tear down (never a blocking barrier in `finally`)."""
    try:
        yield
    except BaseException:
        print(f"[rank {info.rank}] FAILED:\n{traceback.format_exc()}", file=sys.stderr, flush=True)
        if info.enabled and dist.is_initialized():
            dist.destroy_process_group()  # non-blocking abort path; peers time out with our error printed
        raise
    else:
        cleanup(info)


def broadcast_floats(values: list[float], info: DistInfo) -> list[float]:
    """Rank 0's values on every rank (used for stop decisions, LR, step totals)."""
    if not info.enabled:
        return values
    t = torch.tensor(values, dtype=torch.float64, device=info.device)
    dist.broadcast(t, src=0)
    return t.tolist()


def mean_over_ranks(t: torch.Tensor, info: DistInfo) -> torch.Tensor:
    if not info.enabled:
        return t
    t = t.detach().clone()
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t / info.world


def assert_in_sync(module, info: DistInfo, tol: float = 1e-6) -> float:
    """Checksum every parameter and compare across ranks; DDP replicas that drifted apart fail loudly."""
    with torch.no_grad():
        c = torch.zeros((), dtype=torch.float64, device=info.device)
        for i, p in enumerate(unwrap(module).parameters()):
            c += p.detach().double().sum() * (1 + (i % 7) * 1e-3)
    if not info.enabled:
        return float(c)
    lo, hi = c.clone(), c.clone()
    dist.all_reduce(lo, op=dist.ReduceOp.MIN)
    dist.all_reduce(hi, op=dist.ReduceOp.MAX)
    if float(hi - lo) > tol * max(1.0, abs(float(c))):
        raise RuntimeError(f"DDP replicas diverged: checksum spread {float(hi - lo):.3e}")
    return float(c)


def barrier(info: DistInfo) -> None:
    if info.enabled:
        dist.barrier()


def maybe_no_sync(module, active: bool):
    """DDP gradient sync only on the last micro-step of an accumulation cycle."""
    if active and hasattr(module, "no_sync"):
        return module.no_sync()
    return contextlib.nullcontext()


def atomic_dir_write(final_dir: str, write_fn) -> None:
    """write_fn(tmp_dir) then rename over final_dir; a crash mid-save never corrupts the last good copy."""
    tmp = final_dir.rstrip("/\\") + ".tmp"
    old = final_dir.rstrip("/\\") + ".old"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    write_fn(tmp)
    shutil.rmtree(old, ignore_errors=True)
    if os.path.exists(final_dir):
        os.replace(final_dir, old)
    os.replace(tmp, final_dir)
    shutil.rmtree(old, ignore_errors=True)


def unwrap(module):
    """Strip DDP (`module.`) and torch.compile (`_orig_mod.`) wrappers."""
    while True:
        if hasattr(module, "module"):
            module = module.module
        elif hasattr(module, "_orig_mod"):
            module = module._orig_mod
        else:
            return module
