#!/usr/bin/env python3
"""Print percentile summaries from profile_gr00t_server.py JSONL output."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


FIELDS = (
    "request_bytes",
    "deserialize_ms",
    "processor_ms",
    "collate_ms",
    "dtype_cast_ms",
    "model_wall_ms",
    "model_cuda_ms",
    "decode_ms",
    "serialize_ms",
    "policy_total_ms",
    "server_total_ms",
)

BRIDGE_FIELDS = (
    "camera_read_ms",
    "state_get_ms",
    "observation_total_ms",
    "rpc_get_action_ms",
    "inference_total_ms",
)


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    index = (len(values) - 1) * q
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--include-warmup", action="store_true")
    parser.add_argument("--warmup", type=int, default=3, help="Rows to ignore by default")
    args = parser.parse_args()

    rows = []
    for lineno, line in enumerate(args.log.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSON at line {lineno}: {exc}") from exc
        is_server_row = row.get("endpoint") == "get_action" and row.get("status") == "ok"
        is_bridge_row = row.get("kind") in {"observation", "inference"} and row.get("ok")
        if is_server_row or is_bridge_row:
            rows.append(row)

    if not args.include_warmup:
        rows = rows[args.warmup :]
    if not rows:
        raise SystemExit("No successful get_action rows after warmup filtering.")

    print(f"Samples: {len(rows)}")
    print(f"{'stage':<20} {'mean':>10} {'p50':>10} {'p95':>10} {'max':>10}")
    print("-" * 64)
    for field in (*FIELDS, *BRIDGE_FIELDS):
        values = [float(row[field]) for row in rows if field in row]
        if not values:
            continue
        suffix = " B" if field.endswith("_bytes") else " ms"
        print(
            f"{field:<20} "
            f"{sum(values) / len(values):>9.2f}{suffix:<1} "
            f"{percentile(values, 0.50):>9.2f}{suffix:<1} "
            f"{percentile(values, 0.95):>9.2f}{suffix:<1} "
            f"{max(values):>9.2f}{suffix:<1}"
        )

    cuda = [float(row.get("model_cuda_ms", 0.0)) for row in rows]
    wall = [float(row.get("model_wall_ms", 0.0)) for row in rows]
    if any(cuda) and any(wall):
        print(
            "\nModel CPU/dispatch estimate (model_wall - CUDA kernels), mean: "
            f"{sum(max(0.0, w - c) for w, c in zip(wall, cuda)) / len(rows):.2f} ms"
        )


if __name__ == "__main__":
    main()
