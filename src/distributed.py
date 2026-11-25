from __future__ import annotations

import os

import torch
import torch.distributed as dist


def setup_distributed() -> dict:
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if distributed:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device("cuda", local_rank)
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return {
        "distributed": distributed,
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "device": device,
        "is_main": rank == 0,
    }


def gather_objects(value, world_size: int) -> list:
    if world_size == 1:
        return [value]
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, value)
    return gathered


def reduce_pair(total: float, count: int, device: torch.device) -> tuple[float, int]:
    values = torch.tensor([total, float(count)], dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return float(values[0].item()), int(values[1].item())


def cleanup() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

