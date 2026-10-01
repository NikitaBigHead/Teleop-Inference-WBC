"""Select and publish arm-compliance profiles for VLA prompts."""

from __future__ import annotations

import json
import time
from typing import Any


PROMPT_TO_COMPLIANCE_PROFILE = {
    "hug": "HUG",
    "none": "HUG",
    "handshake": "HANDSHAKE",
    "fist_bump": "FISTBUMP_SOFTWRIST",
}


def compliance_profile_for_prompt(prompt: str) -> str:
    """Return a safe built-in profile for an effective VLA prompt."""
    return PROMPT_TO_COMPLIANCE_PROFILE.get(prompt.strip().lower(), "RIGID")


class ComplianceProfilePublisher:
    """Publish the selected profile immediately and then as a heartbeat."""

    def __init__(self, socket: Any, *, topic: str = "compliance", rate_hz: float = 10.0):
        if rate_hz <= 0:
            raise ValueError("compliance publish rate must be positive")
        self._socket = socket
        self.topic = topic
        self.period_s = 1.0 / float(rate_hz)
        self.profile: str | None = None
        self._next_publish = 0.0

    def _publish(self, now: float) -> None:
        if self.profile is None:
            return
        command = json.dumps({"profile": self.profile}, separators=(",", ":"))
        self._socket.send_string(f"{self.topic} {command}")
        self._next_publish = now + self.period_s

    def set_prompt(self, prompt: str, now: float | None = None) -> bool:
        """Select the prompt's profile; return True only when it changed."""
        profile = compliance_profile_for_prompt(prompt)
        changed = profile != self.profile
        if changed:
            self.profile = profile
            self._publish(time.monotonic() if now is None else float(now))
        return changed

    def tick(self, now: float | None = None) -> bool:
        """Send the current profile when its heartbeat is due."""
        now = time.monotonic() if now is None else float(now)
        if self.profile is None or now < self._next_publish:
            return False
        self._publish(now)
        return True
