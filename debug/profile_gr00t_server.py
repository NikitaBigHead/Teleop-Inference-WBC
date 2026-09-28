#!/usr/bin/env python3
"""GR00T PolicyServer with per-request latency profiling.

Run this *instead of* ``gr00t/eval/run_gr00t_server.py`` for a diagnostic
session.  It uses exactly the same checkpoint and ZMQ protocol, but records
where every get_action request spends its time.

The JSONL output is deliberately one record per request so it can be analysed
without parsing coloured terminal output.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1_000.0, 3)


def _resolve_embodiment(tag: str):
    from gr00t.data.embodiment_tags import EmbodimentTag

    normalized = tag.replace("-", "_").upper()
    if normalized in EmbodimentTag.__members__:
        return EmbodimentTag[normalized]
    for member in EmbodimentTag:
        if member.value.upper() == normalized:
            return member
    values = ", ".join(member.name for member in EmbodimentTag)
    raise ValueError(f"Unknown embodiment tag '{tag}'. Choices: {values}")


class JsonlLogger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("a", encoding="utf-8", buffering=1)
        print(f"[profile] Writing JSONL records to {path}", flush=True)

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, sort_keys=True, default=str)
        self._file.write(line + "\n")
        print("[profile] " + line, flush=True)

    def close(self) -> None:
        self._file.close()


def build_profiled_policy(device: str):
    """Create a Gr00tPolicy subclass after GR00T is importable."""
    import numpy as np
    import torch
    from gr00t.policy.gr00t_policy import Gr00tPolicy, _rec_to_dtype

    class ProfiledGr00tPolicy(Gr00tPolicy):
        # The PolicyServer processes requests serially, so this is safe to read
        # immediately after get_action returns.
        last_profile: dict[str, Any] | None = None

        def _get_action(self, observation: dict[str, Any], options=None):
            profile: dict[str, Any] = {
                "model_device": str(self.model.device),
                "model_dtype": str(self.model.dtype),
                "cuda_available": torch.cuda.is_available(),
            }
            total_start = time.perf_counter()

            start = time.perf_counter()
            unbatched_observations = self._unbatch_observation(observation)
            profile["unbatch_ms"] = _ms(start)

            processed_inputs = []
            states = []
            start = time.perf_counter()
            for obs in unbatched_observations:
                vla_step_data = self._to_vla_step_data(obs)
                states.append(vla_step_data.states)
                messages = [{"type": "episode_step", "content": vla_step_data}]
                processed_inputs.append(self.processor(messages))
            profile["processor_ms"] = _ms(start)

            start = time.perf_counter()
            collated_inputs = self.collate_fn(processed_inputs)
            profile["collate_ms"] = _ms(start)

            start = time.perf_counter()
            collated_inputs = _rec_to_dtype(collated_inputs, dtype=torch.bfloat16)
            profile["dtype_cast_ms"] = _ms(start)

            cuda_device = self.model.device if torch.cuda.is_available() else None
            if cuda_device is not None:
                # Synchronisation is intentionally diagnostic-only: it makes
                # wall time and CUDA event time attributable to this request.
                torch.cuda.synchronize(cuda_device)
                event_start = torch.cuda.Event(enable_timing=True)
                event_end = torch.cuda.Event(enable_timing=True)
                event_start.record()
            else:
                event_start = event_end = None

            start = time.perf_counter()
            with torch.inference_mode():
                model_pred = self.model.get_action(**collated_inputs)
            if event_end is not None:
                event_end.record()
                torch.cuda.synchronize(cuda_device)
                profile["model_cuda_ms"] = round(event_start.elapsed_time(event_end), 3)
                profile["gpu_memory_allocated_mb"] = round(
                    torch.cuda.memory_allocated(cuda_device) / 1024**2, 1
                )
            profile["model_wall_ms"] = _ms(start)

            start = time.perf_counter()
            normalized_action = model_pred["action_pred"].float()
            batched_states = {
                key: np.stack([state[key] for state in states], axis=0)
                for key in self.modality_configs["state"].modality_keys
            }
            unnormalized_action = self.processor.decode_action(
                normalized_action.cpu().numpy(), self.embodiment_tag, batched_states
            )
            profile["decode_ms"] = _ms(start)

            start = time.perf_counter()
            casted_action = {
                key: value.astype(np.float32) for key, value in unnormalized_action.items()
            }
            profile["action_cast_ms"] = _ms(start)
            profile["policy_total_ms"] = _ms(total_start)
            self.last_profile = profile
            return casted_action, {}

    return ProfiledGr00tPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gr00t-root", default=".", help="Path to the Isaac-GR00T checkout")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--embodiment-tag", default="NEW_EMBODIMENT")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--profile-log", default="/tmp/gr00t_profile.jsonl")
    parser.add_argument("--no-strict", action="store_true")
    args = parser.parse_args()

    gr00t_root = str(Path(args.gr00t_root).resolve())
    if gr00t_root not in sys.path:
        sys.path.insert(0, gr00t_root)

    import torch
    from gr00t.policy.server_client import MsgSerializer, PolicyServer
    import zmq

    ProfiledGr00tPolicy = build_profiled_policy(args.device)
    policy = ProfiledGr00tPolicy(
        embodiment_tag=_resolve_embodiment(args.embodiment_tag),
        model_path=args.model_path,
        device=args.device,
        strict=not args.no_strict,
    )
    logger = JsonlLogger(Path(args.profile_log))

    class ProfiledPolicyServer(PolicyServer):
        def run(self):
            address = self.socket.getsockopt_string(zmq.LAST_ENDPOINT)
            print(f"Profiled GR00T server listening on {address}", flush=True)
            request_index = 0
            while self.running:
                request_index += 1
                total_start = time.perf_counter()
                record: dict[str, Any] = {"request_index": request_index}
                try:
                    raw_message = self.socket.recv()
                    record["request_bytes"] = len(raw_message)

                    start = time.perf_counter()
                    request = MsgSerializer.from_bytes(raw_message)
                    record["deserialize_ms"] = _ms(start)
                    endpoint = request.get("endpoint", "get_action")
                    record["endpoint"] = endpoint

                    if not self._validate_token(request):
                        response = {"error": "Unauthorized: Invalid API token"}
                    elif endpoint not in self._endpoints:
                        response = {"error": f"Unknown endpoint: {endpoint}"}
                    else:
                        handler = self._endpoints[endpoint]
                        start = time.perf_counter()
                        response = (
                            handler.handler(**request.get("data", {}))
                            if handler.requires_input
                            else handler.handler()
                        )
                        record["handler_ms"] = _ms(start)
                        if endpoint == "get_action" and policy.last_profile is not None:
                            record.update(policy.last_profile)

                    start = time.perf_counter()
                    encoded_response = MsgSerializer.to_bytes(response)
                    record["serialize_ms"] = _ms(start)
                    record["response_bytes"] = len(encoded_response)
                    self.socket.send(encoded_response)
                    record["status"] = "ok"
                except Exception as exc:  # Keep REP socket usable after an error.
                    record["status"] = "error"
                    record["error"] = repr(exc)
                    try:
                        self.socket.send(MsgSerializer.to_bytes({"error": str(exc)}))
                    except zmq.ZMQError:
                        pass
                finally:
                    record["server_total_ms"] = _ms(total_start)
                    record["timestamp_unix"] = time.time()
                    logger.write(record)

    print(f"CUDA available: {torch.cuda.is_available()}; requested device: {args.device}")
    server = ProfiledPolicyServer(policy=policy, host=args.host, port=args.port)
    try:
        server.run()
    except KeyboardInterrupt:
        print("Profiled GR00T server stopped.")
    finally:
        logger.close()
        server.socket.close()
        server.context.term()


if __name__ == "__main__":
    main()
