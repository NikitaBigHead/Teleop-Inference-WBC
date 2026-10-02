"""Human-intent wire protocol and the safety state machine used by VLA inference."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Mapping


SCHEMA_VERSION = 1
INTENT_TO_PROMPT = {
    "hug": "hug",
    "handshake": "handshake",
    "fist_bump": "fist_bump",
    "no_interaction": "none",
}


def validate_intent_event(message: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalise one message from ``dual_camera_robot``."""
    if message.get("type") != "human_intent":
        raise ValueError(f"unexpected intent type: {message.get('type')!r}")
    if int(message.get("version", -1)) != SCHEMA_VERSION:
        raise ValueError(f"unsupported intent schema version: {message.get('version')!r}")

    label = str(message.get("label", ""))
    if label not in {*INTENT_TO_PROMPT, "unknown"}:
        raise ValueError(f"unsupported intent label: {label!r}")

    event = dict(message)
    event["label"] = label
    event["accepted"] = bool(message.get("accepted", label != "unknown"))
    event["confidence"] = float(message.get("confidence", 0.0))
    event["seq"] = int(message.get("seq", -1))
    return event


class HumanIntentSubscriber:
    """Non-blocking latest-only subscriber for human intent decisions."""

    def __init__(self, host: str, port: int):
        import zmq

        self._zmq = zmq
        self.endpoint = f"tcp://{host}:{port}"
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self._socket.setsockopt(zmq.CONFLATE, True)
        self._socket.setsockopt(zmq.RCVHWM, 1)
        self._socket.connect(self.endpoint)
        self._last_warning_time = 0.0

    def read_latest(self) -> dict[str, Any] | None:
        latest = None
        while self._socket.poll(0):
            try:
                latest = validate_intent_event(
                    self._socket.recv_json(flags=self._zmq.NOBLOCK)
                )
            except self._zmq.Again:
                break
            except (TypeError, ValueError) as exc:
                now = time.monotonic()
                if now - self._last_warning_time >= 2.0:
                    print(f"[intent] ignored invalid message: {exc}", flush=True)
                    self._last_warning_time = now
        return latest

    def close(self) -> None:
        self._socket.close(0)
        self._context.term()


@dataclass
class IntentController:
    """Own prompt selection and fail closed when automatic intent is unavailable."""

    mode: str
    initial_prompt: str
    max_age: float = 0.5
    unknown_grace: float = 0.3
    unknown_to_none_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in {"manual", "auto"}:
            raise ValueError("intent mode must be 'manual' or 'auto'")
        if self.max_age <= 0:
            raise ValueError("intent max_age must be positive")
        if self.unknown_grace < 0:
            raise ValueError("intent unknown_grace must be non-negative")
        if self.unknown_to_none_seconds < 0:
            raise ValueError("intent unknown_to_none_seconds must be non-negative")
        if 0 < self.unknown_to_none_seconds <= self.unknown_grace:
            raise ValueError("intent unknown_to_none_seconds must exceed unknown_grace")

        self.prompt = self.initial_prompt
        self.label: str | None = None
        self.epoch = 0
        self.last_message_monotonic: float | None = None
        self.unknown_since: float | None = None
        self.fallback_unknown_since: float | None = None
        self.unknown_reason: str | None = None
        self.unknown_fallback_active = False
        self.inference_enabled = self.mode == "manual"
        self.execution_hold = True
        self.hold_reason = (
            "waiting for first action chunk"
            if self.mode == "manual"
            else "waiting for human intent"
        )

    def _invalidate(self, reason: str) -> None:
        self.epoch += 1
        self.execution_hold = True
        self.hold_reason = reason

    def _enter_hold(self, reason: str) -> bool:
        if not self.inference_enabled and self.execution_hold and self.hold_reason == reason:
            return False
        self.inference_enabled = False
        self._invalidate(reason)
        return True

    def set_manual_prompt(self, prompt: str) -> bool:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("manual prompt must not be empty")
        if self.mode == "manual" and self.prompt == prompt and self.inference_enabled:
            return False
        self.mode = "manual"
        self.prompt = prompt
        self.label = None
        self.unknown_since = None
        self.fallback_unknown_since = None
        self.unknown_reason = None
        self.unknown_fallback_active = False
        self.inference_enabled = True
        self._invalidate(f"waiting for manual prompt {prompt!r}")
        return True

    def enable_auto(self) -> bool:
        if self.mode == "auto":
            return False
        self.mode = "auto"
        self.label = None
        self.last_message_monotonic = None
        self.unknown_since = None
        self.fallback_unknown_since = None
        self.unknown_reason = None
        self.unknown_fallback_active = False
        return self._enter_hold("waiting for human intent")

    def restart(self, reason: str) -> bool:
        """Discard the current chunk and require a fresh result for this prompt."""
        if not self.inference_enabled:
            return False
        self._invalidate(reason)
        return True

    def _clear_unknown(self) -> None:
        self.unknown_since = None
        self.fallback_unknown_since = None
        self.unknown_reason = None
        self.unknown_fallback_active = False

    def _accept_label(self, label: str, reason: str | None = None) -> bool:
        prompt = INTENT_TO_PROMPT[label]
        self._clear_unknown()
        if self.inference_enabled and self.label == label and self.prompt == prompt:
            return False
        self.label = label
        self.prompt = prompt
        self.inference_enabled = True
        self._invalidate(reason or f"waiting for VLA chunk for {label}")
        return True

    def _apply_unknown_fallback(self, now: float) -> bool:
        if (
            self.unknown_to_none_seconds <= 0
            or self.fallback_unknown_since is None
            or now - self.fallback_unknown_since < self.unknown_to_none_seconds
        ):
            return False
        changed = self._accept_label(
            "no_interaction",
            "waiting for VLA chunk for no_interaction (unknown fallback)",
        )
        self.unknown_fallback_active = True
        return changed

    def process_event(self, event: Mapping[str, Any], now: float) -> bool:
        """Apply one validated event. Return whether the epoch changed."""
        if self.mode != "auto":
            return False

        self.last_message_monotonic = now
        label = str(event["label"])
        accepted = bool(event.get("accepted", label != "unknown"))
        if not accepted or label == "unknown":
            if self.unknown_fallback_active:
                return False
            if self.unknown_since is None:
                self.unknown_since = now
            reason = str(event.get("reason", "unknown")) or "unknown"
            self.unknown_reason = reason
            if self.fallback_unknown_since is None:
                self.fallback_unknown_since = now
            if self._apply_unknown_fallback(now):
                return True
            if now - self.unknown_since >= self.unknown_grace:
                return self._enter_hold(f"intent rejected: {reason}")
            return False

        return self._accept_label(label)

    def tick(self, now: float) -> bool:
        """Apply time-based unknown and publisher-staleness transitions."""
        if self.mode != "auto":
            return False
        if self.last_message_monotonic is None:
            return False
        if now - self.last_message_monotonic > self.max_age:
            self.fallback_unknown_since = None
            self.unknown_fallback_active = False
            return self._enter_hold("human intent stream is stale")
        if self._apply_unknown_fallback(now):
            return True
        if self.unknown_since is not None and now - self.unknown_since >= self.unknown_grace:
            if (
                not self.inference_enabled
                and self.execution_hold
                and self.hold_reason.startswith("intent rejected:")
            ):
                return False
            reason = self.unknown_reason or "unknown"
            return self._enter_hold(f"intent rejected: {reason}")
        return False

    def accept_result(self, epoch: int) -> bool:
        """Open the execution gate only for the current prompt generation."""
        if epoch != self.epoch or not self.inference_enabled:
            return False
        self.execution_hold = False
        self.hold_reason = ""
        return True
