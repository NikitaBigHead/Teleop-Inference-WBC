import unittest

from gear_sonic.utils.inference.human_intent import (
    IntentController,
    validate_intent_event,
)


def event(label, accepted=True, reason=""):
    return validate_intent_event(
        {
            "type": "human_intent",
            "version": 1,
            "seq": 1,
            "label": label,
            "accepted": accepted,
            "confidence": 0.8,
            "reason": reason,
        }
    )


class IntentControllerTest(unittest.TestCase):
    def test_auto_intent_switch_holds_until_matching_vla_result(self):
        controller = IntentController("auto", "none")
        self.assertIs(controller.inference_enabled, False)
        self.assertIs(controller.execution_hold, True)

        self.assertIs(controller.process_event(event("handshake"), now=1.0), True)
        self.assertEqual(controller.prompt, "handshake")
        self.assertEqual(controller.epoch, 1)
        self.assertIs(controller.execution_hold, True)
        self.assertIs(controller.accept_result(0), False)
        self.assertIs(controller.accept_result(1), True)
        self.assertIs(controller.execution_hold, False)

        self.assertIs(controller.process_event(event("handshake"), now=1.2), False)
        self.assertEqual(controller.epoch, 1)

    def test_no_interaction_maps_to_none(self):
        controller = IntentController("auto", "hug")
        controller.process_event(event("no_interaction"), now=1.0)
        self.assertEqual(controller.prompt, "none")

    def test_unknown_and_stale_stream_fail_closed(self):
        controller = IntentController("auto", "none", max_age=0.5, unknown_grace=0.3)
        controller.process_event(event("hug"), now=1.0)
        controller.accept_result(controller.epoch)

        self.assertIs(
            controller.process_event(
                event("unknown", accepted=False, reason="low_confidence"), now=1.1
            ),
            False,
        )
        self.assertIs(controller.execution_hold, False)
        self.assertIs(controller.tick(now=1.41), True)
        self.assertIs(controller.execution_hold, True)
        self.assertIs(controller.inference_enabled, False)

        controller.process_event(event("hug"), now=2.0)
        controller.accept_result(controller.epoch)
        self.assertIs(controller.tick(now=2.6), True)
        self.assertEqual(controller.hold_reason, "human intent stream is stale")

    def test_manual_override_and_return_to_auto(self):
        controller = IntentController("auto", "none")
        self.assertIs(controller.set_manual_prompt("hug"), True)
        self.assertEqual(controller.mode, "manual")
        self.assertIs(controller.inference_enabled, True)
        self.assertIs(controller.enable_auto(), True)
        self.assertEqual(controller.mode, "auto")
        self.assertIs(controller.inference_enabled, False)

    def test_fist_bump_is_accepted_like_other_known_intents(self):
        controller = IntentController("auto", "none", unknown_grace=0.0)
        self.assertIs(controller.process_event(event("fist_bump"), now=1.0), True)
        self.assertIs(controller.inference_enabled, True)
        self.assertEqual(controller.prompt, "fist_bump")

    def test_protocol_rejects_unknown_schema_and_label(self):
        with self.assertRaises(ValueError):
            validate_intent_event(
                {"type": "human_intent", "version": 2, "label": "hug"}
            )
        with self.assertRaises(ValueError):
            validate_intent_event(
                {"type": "human_intent", "version": 1, "label": "wave"}
            )


if __name__ == "__main__":
    unittest.main()
