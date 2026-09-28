#!/usr/bin/env python3
"""Run the affective VLA bridge with JSONL timing, without editing the bridge.

Use the same command-line arguments as run_inference_affective_vla.py. Set
BRIDGE_PROFILE_LOG to choose the output path (default: /tmp/bridge_profile.jsonl).
The wrapper patches functions only in this Python process.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any


class JsonlLogger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("a", encoding="utf-8", buffering=1)
        self._lock = threading.Lock()
        print(f"[bridge-profile] Writing JSONL records to {path}", flush=True)

    def write(self, record: dict[str, Any]) -> None:
        record["timestamp_unix"] = time.time()
        with self._lock:
            self._file.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def main() -> None:
    # Import after startup so this script remains importable from a plain repo
    # checkout; the caller activates the same environment as the normal bridge.
    import tyro
    from gear_sonic.scripts import run_inference_affective_vla as bridge

    logger = JsonlLogger(Path(os.environ.get("BRIDGE_PROFILE_LOG", "/tmp/bridge_profile.jsonl")))
    original_prepare = bridge.prepare_observation_from_sensors
    original_inference = bridge.run_policy_inference_and_process

    def profiled_prepare(*args: Any, **kwargs: Any):
        camera_subscriber = kwargs.get("camera_subscriber")
        state_subscriber = kwargs.get("state_subscriber")
        if camera_subscriber is None and len(args) >= 1:
            camera_subscriber = args[0]
        if state_subscriber is None and len(args) >= 2:
            state_subscriber = args[1]

        timings: dict[str, float] = {}
        original_camera_read = camera_subscriber.read
        original_state_get = state_subscriber.get_msg

        def camera_read(*read_args: Any, **read_kwargs: Any):
            start = time.perf_counter()
            try:
                return original_camera_read(*read_args, **read_kwargs)
            finally:
                timings["camera_read_ms"] = (time.perf_counter() - start) * 1_000.0

        def state_get(*state_args: Any, **state_kwargs: Any):
            start = time.perf_counter()
            try:
                return original_state_get(*state_args, **state_kwargs)
            finally:
                timings["state_get_ms"] = (time.perf_counter() - start) * 1_000.0

        start = time.perf_counter()
        camera_subscriber.read = camera_read
        state_subscriber.get_msg = state_get
        try:
            observation = original_prepare(*args, **kwargs)
        finally:
            camera_subscriber.read = original_camera_read
            state_subscriber.get_msg = original_state_get

        record: dict[str, Any] = {
            "kind": "observation",
            "observation_total_ms": round((time.perf_counter() - start) * 1_000.0, 3),
            **{key: round(value, 3) for key, value in timings.items()},
            "ok": observation is not None,
        }
        if observation is not None:
            record["video_shapes"] = {
                key: list(value.shape) for key, value in observation.get("video", {}).items()
            }
        logger.write(record)
        return observation

    def profiled_inference(policy: Any, observation: dict[str, Any], robot_model: Any):
        # The original helper combines PolicyClient.get_action with local output
        # validation/concat_action. Time both, and isolate the RPC inside it.
        rpc: dict[str, float] = {}
        original_get_action = policy.get_action

        def timed_get_action(*call_args: Any, **call_kwargs: Any):
            start = time.perf_counter()
            try:
                return original_get_action(*call_args, **call_kwargs)
            finally:
                rpc["rpc_get_action_ms"] = (time.perf_counter() - start) * 1_000.0

        start = time.perf_counter()
        policy.get_action = timed_get_action
        try:
            result = original_inference(policy, observation, robot_model)
        finally:
            policy.get_action = original_get_action

        logger.write(
            {
                "kind": "inference",
                "inference_total_ms": round((time.perf_counter() - start) * 1_000.0, 3),
                **{key: round(value, 3) for key, value in rpc.items()},
                "ok": result is not None,
            }
        )
        return result

    bridge.prepare_observation_from_sensors = profiled_prepare
    bridge.run_policy_inference_and_process = profiled_inference
    tyro.cli(bridge.main)


if __name__ == "__main__":
    main()
