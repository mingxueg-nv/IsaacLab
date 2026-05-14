from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch


parser = argparse.ArgumentParser(description="Benchmark direct GR00T policy inference latency.")
parser.add_argument("--model_path", required=True)
parser.add_argument("--config_dir", required=True)
parser.add_argument("--warmup", type=int, default=5)
parser.add_argument("--iters", type=int, default=30)
parser.add_argument("--height", type=int, default=224)
parser.add_argument("--width", type=int, default=224)
parser.add_argument("--denoising_steps", type=int, default=4)
parser.add_argument("--output", default="/tmp/gr00t_policy_benchmark.json")
args = parser.parse_args()

config_dir = Path(args.config_dir).resolve()
if str(config_dir) not in sys.path:
    sys.path.insert(0, str(config_dir))

from gr00t.model.policy import Gr00tPolicy  # noqa: E402
from gr00t_config import IsaacLabDataConfig  # noqa: E402


def _stats(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean_ms": statistics.fmean(ordered),
        "median_ms": statistics.median(ordered),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "p95_ms": ordered[min(len(ordered) - 1, max(0, int(round(0.95 * (len(ordered) - 1)))))],
    }


def make_obs(height: int, width: int) -> dict[str, object]:
    image = np.zeros((1, height, width, 3), dtype=np.uint8)
    image[..., 0] = 96
    image[..., 1] = 112
    image[..., 2] = 128
    return {
        "video.left_wrist_view": image.copy(),
        "video.right_wrist_view": image.copy(),
        "video.room_view": image.copy(),
        "state.left_arm": np.zeros((1, 7), dtype=np.float32),
        "state.right_arm": np.zeros((1, 7), dtype=np.float32),
        "state.left_hand": np.zeros((1, 7), dtype=np.float32),
        "state.right_hand": np.zeros((1, 7), dtype=np.float32),
        "annotation.human.task_description": np.array(["install trocar from box"]),
    }


def main() -> None:
    data_config = IsaacLabDataConfig()
    load_t0 = time.perf_counter()
    policy = Gr00tPolicy(
        model_path=args.model_path,
        embodiment_tag="new_embodiment",
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        denoising_steps=args.denoising_steps,
        device="cuda:0",
    )
    torch.cuda.synchronize()
    load_s = time.perf_counter() - load_t0

    obs = make_obs(args.height, args.width)
    for _ in range(args.warmup):
        _ = policy.get_action(obs)
    torch.cuda.synchronize()

    times_ms: list[float] = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        action = policy.get_action(obs)
        torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    action_shapes = {
        key: list(np.asarray(value).shape)
        for key, value in action.items()
    }
    metrics = {
        "model_path": args.model_path,
        "height": args.height,
        "width": args.width,
        "warmup": args.warmup,
        "iters": args.iters,
        "denoising_steps": args.denoising_steps,
        "load_s": load_s,
        "latency": _stats(times_ms),
        "raw_latency_ms": times_ms,
        "action_shapes": action_shapes,
        "cuda_max_memory_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3),
        "cuda_max_memory_reserved_gb": torch.cuda.max_memory_reserved() / (1024**3),
    }
    Path(args.output).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in metrics.items() if k != "raw_latency_ms"}, indent=2))


if __name__ == "__main__":
    main()
