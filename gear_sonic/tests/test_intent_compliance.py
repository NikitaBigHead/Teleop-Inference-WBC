import json
import unittest

from gear_sonic.utils.inference.intent_compliance import (
    ComplianceProfilePublisher,
    compliance_profile_for_prompt,
)


class FakeSocket:
    def __init__(self):
        self.messages = []

    def send_string(self, message):
        self.messages.append(message)


class IntentComplianceTest(unittest.TestCase):
    def test_prompt_mapping_matches_robot_profiles(self):
        self.assertEqual(compliance_profile_for_prompt("hug"), "HUG")
        self.assertEqual(compliance_profile_for_prompt("none"), "HUG")
        self.assertEqual(compliance_profile_for_prompt("handshake"), "HANDSHAKE")
        self.assertEqual(
            compliance_profile_for_prompt("fist_bump"), "FISTBUMP_SOFTWRIST"
        )
        self.assertEqual(compliance_profile_for_prompt("custom task"), "RIGID")

    def test_change_is_immediate_and_same_profile_uses_heartbeat(self):
        socket = FakeSocket()
        publisher = ComplianceProfilePublisher(socket, rate_hz=10.0)

        self.assertIs(publisher.set_prompt("hug", now=1.0), True)
        self.assertIs(publisher.set_prompt("none", now=1.01), False)
        self.assertIs(publisher.tick(now=1.09), False)
        self.assertIs(publisher.tick(now=1.10), True)

        self.assertEqual(len(socket.messages), 2)
        topic, payload = socket.messages[-1].split(" ", 1)
        self.assertEqual(topic, "compliance")
        self.assertEqual(json.loads(payload), {"profile": "HUG"})

    def test_profile_changes_with_executed_prompt(self):
        socket = FakeSocket()
        publisher = ComplianceProfilePublisher(socket, rate_hz=10.0)
        publisher.set_prompt("hug", now=1.0)

        self.assertIs(publisher.set_prompt("handshake", now=1.01), True)
        self.assertIn('"HANDSHAKE"', socket.messages[-1])


if __name__ == "__main__":
    unittest.main()
