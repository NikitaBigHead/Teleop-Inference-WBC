"""
AffectiveVLA two-camera inference runner — NO ROS 2 DEPENDENCY.

Runs an Isaac-GR00T VLA policy against the Sonic whole-body control stack.
All communication uses ZMQ:
  1. Robot state  -> ZMQ SUB on ``g1_debug`` topic (from C++ zmq_output_handler)
  2. Actions out  -> ZMQ PUB (latent protocol v4: motion token + hand joints)
  3. Camera       -> ZMQ/TCP via ComposedCameraClientSensor
  4. Keyboard     -> ZMQ SUB via ZMQKeyboardSubscriber
  5. Compliance   -> optional ZMQ PUB of runtime Kp/Kd profiles to C++ deploy

Uses the Isaac-GR00T PolicyClient (ZMQ REQ/REP) to communicate with a
running PolicyServer. The robot camera transport names are mapped to the
two video modality names used during training:

  ego_view -> depth_camera_rgb
  head     -> webcam

Keyboard commands (received via ZMQ from the standalone keyboard publisher):
  p  -> pause / resume the policy loop
  k  -> start / stop the C++ control loop
  i  -> send initial pose and switch to POSE mode
  pr <text> -> change prompt at runtime (received via ZMQ)
  pr auto -> return prompt control to the human-intent predictor
  [  -> toggle left hand open/closed for initial pose
  ]  -> toggle right hand open/closed for initial pose
  c  -> start recording (handled by data exporter if running)
  s  -> stop recording success (handled by data exporter)
  f  -> stop recording failure (handled by data exporter)
"""

from dataclasses import dataclass
import os
import queue
import sys as _sys
import threading
import time

import numpy as np
_sys.path.append(os.path.expanduser('~'))  # PROBE-PATCH
from chunk_probe import ChunkProbe  # PROBE-PATCH
import tyro
import zmq

from gear_sonic.camera.composed_camera import ComposedCameraClientSensor
from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model
from gear_sonic.utils.data_collection.keyboard_subscriber import (
    DEFAULT_ZMQ_KEYBOARD_PORT,
    ZMQKeyboardSubscriber,
)
from gear_sonic.utils.data_collection.telemetry import Telemetry
from gear_sonic.utils.data_collection.transforms import compute_projected_gravity
from gear_sonic.utils.data_collection.zmq_state_subscriber import ZMQStateSubscriber
from gear_sonic.utils.inference.initial_poses import LATENT_INITIAL_MOTION_TOKEN
from gear_sonic.utils.inference.human_intent import (
    HumanIntentSubscriber,
    INTENT_TO_PROMPT,
    IntentController,
)
from gear_sonic.utils.inference.intent_compliance import ComplianceProfilePublisher
from gear_sonic.utils.inference.vla_utils import (
    calculate_latency_compensated_index,
    concat_action,
    prepare_observation_for_eval,
    should_trigger_new_inference,
)
from gear_sonic.utils.teleop.solver.hand.g1_gripper_ik_solver import (
    G1GripperInverseKinematicsSolver,
)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    build_command_message,
    pack_pose_message,
)


@dataclass
class InferenceConfig:
    """CLI config for the VLA inference runner."""

    # Policy server (Isaac-GR00T PolicyServer)
    host: str = "localhost"
    """The host address of the Isaac-GR00T PolicyServer."""

    port: int = 5550
    """The port of the Isaac-GR00T PolicyServer."""

    # Control
    action_publish_rate: int = 50
    """Rate at which individual actions are published to the C++ control loop (Hz)."""

    action_horizon: int = 50
    """Action horizon of the VLA policy (number of future actions per inference)."""

    rate: float = 1 / 0.2
    """Rate at which we run the forward pass of the VLA policy (Hz)."""

    # Camera
    camera_host: str = "localhost"
    """Camera server host."""

    camera_port: int = 5555
    """Camera server port."""

    # ZMQ: Robot state (from C++ zmq_output_handler, g1_debug topic)
    state_zmq_host: str = "localhost"
    """ZMQ host for robot state (g1_debug topic from C++ deploy)."""

    state_zmq_port: int = 5557
    """ZMQ port for robot state (same socket as robot_config topic)."""

    # ZMQ: Action output (latent actions to C++ control loop)
    action_zmq_host: str = "localhost"
    """ZMQ host for action output (PUB socket)."""

    action_zmq_port: int = 5556
    """ZMQ port for action output."""

    # ZMQ: Keyboard input
    keyboard_zmq_host: str = "localhost"
    """ZMQ host for keyboard input."""

    keyboard_zmq_port: int = DEFAULT_ZMQ_KEYBOARD_PORT
    """ZMQ port for keyboard input."""

    # Embodiment
    embodiment_tag: str = "unitree_g1_sonic"
    """Embodiment tag for policy inference."""

    # Prompt / eval
    prompt: str = "demo"
    """The language prompt for the VLA policy."""

    intent_mode: str = "manual"
    """Prompt source: 'manual' keeps --prompt/pr commands; 'auto' follows the predictor."""

    intent_host: str = "localhost"
    """Host running the dual-camera human-intent publisher."""

    intent_port: int = 5562
    """Port of the dual-camera human-intent publisher."""

    intent_max_age: float = 0.5
    """Enter safe hold when no predictor message arrives for this many seconds."""

    intent_unknown_grace: float = 0.3
    """How long a continuous unknown prediction is tolerated before safe hold."""

    intent_compliance: bool = False
    """Switch C++ arm-compliance profiles when a new VLA prompt starts executing."""

    compliance_port: int = 5565
    """Port bound by the compliance command publisher."""

    compliance_rate: float = 10.0
    """Compliance profile heartbeat rate in Hz."""

    # BrainCo contact detection
    brainco_contact_position_error_threshold: float = 0.05
    """Minimum command/state position lag used to detect contact."""

    brainco_contact_count: int = 3
    """Consecutive contact detections required to block finger closing."""

    brainco_contact_hold_seconds: float = 1.0
    """Seconds to block further finger closing after contact."""

    # Debug
    verbose_timing: bool = False
    """Whether to always print timing info (not just when loop is slow)."""


def print_green(x):
    print(f"\033[92m{x}\033[0m")


@dataclass(frozen=True)
class InferenceRequest:
    intent_epoch: int
    prompt: str


@dataclass(frozen=True)
class InferenceResult:
    intent_epoch: int
    prompt: str
    action: dict
    inference_start_time: float


# ---------------------------------------------------------------------------
# Action packing (latent protocol v4)
# ---------------------------------------------------------------------------


def pack_latent_action_message(
    motion_token: np.ndarray,
    frame_index: np.ndarray,
    left_hand_joints: np.ndarray = None,
    right_hand_joints: np.ndarray = None,
) -> bytes:
    """Pack a single motion-token action into a ZMQ message (Protocol v4).

    Args:
        motion_token: Shape ``[64]`` (flat) or ``[1, 64]``.
        frame_index:  Shape ``[1]``.
        left_hand_joints:  Shape ``[7]`` or ``[1, 7]``, optional.
        right_hand_joints: Shape ``[7]`` or ``[1, 7]``, optional.

    Returns:
        Packed ZMQ message bytes.
    """
    motion_token = np.asarray(motion_token, dtype=np.float32)
    frame_index = np.asarray(frame_index, dtype=np.int64)

    if frame_index.ndim == 0:
        frame_index = np.array([frame_index], dtype=np.int64)
    elif frame_index.shape[0] != 1:
        frame_index = frame_index[:1]

    if motion_token.ndim == 1:
        motion_token = motion_token.reshape(1, -1)

    pose_data = {
        "token_state": motion_token,
        "frame_index": frame_index,
    }

    if left_hand_joints is not None:
        left_hand_joints = np.asarray(left_hand_joints, dtype=np.float32)
        if left_hand_joints.ndim == 1:
            if left_hand_joints.shape[0] != 7:
                raise ValueError(
                    f"left_hand_joints must have shape [7], got {left_hand_joints.shape}"
                )
            left_hand_joints = left_hand_joints.reshape(1, 7)
        pose_data["left_hand_joints"] = left_hand_joints

    if right_hand_joints is not None:
        right_hand_joints = np.asarray(right_hand_joints, dtype=np.float32)
        if right_hand_joints.ndim == 1:
            if right_hand_joints.shape[0] != 7:
                raise ValueError(
                    f"right_hand_joints must have shape [7], got {right_hand_joints.shape}"
                )
            right_hand_joints = right_hand_joints.reshape(1, 7)
        pose_data["right_hand_joints"] = right_hand_joints

    return pack_pose_message(pose_data, topic="pose", version=4)


def get_action_field(action_dict: dict, key: str):
    """Get action field from dict, checking both with and without 'action.' prefix."""
    value = action_dict.get(key)
    if value is not None:
        return value
    value = action_dict.get(f"action.{key}")
    if value is not None:
        return value
    raise AssertionError(
        f"Required action field '{key}' (or 'action.{key}') not found in processed_action. "
        f"Available keys: {list(action_dict.keys())}"
    )


# ---------------------------------------------------------------------------
# Observation / inference helpers
# ---------------------------------------------------------------------------


# --- BrainCo hand state override (6 DOF, как при сборе данных) ---
_BRAINCO_CTRL = None
_BRAINCO_STATE_LOCK = threading.Lock()
_BRAINCO_STATES = {"left": None, "right": None}
_BRAINCO_STATE_SUBSCRIBERS = []

_BRAINCO_NUM_MOTORS = 6
_BRAINCO_MOTOR_NAMES = ("thumb", "thumb_aux", "index", "middle", "ring", "pinky")

_brainco_last_command = {"left": None, "right": None}
_brainco_contact_blocked_until = {
    "left": np.zeros(_BRAINCO_NUM_MOTORS, dtype=np.float64),
    "right": np.zeros(_BRAINCO_NUM_MOTORS, dtype=np.float64),
}
_brainco_contact_counter = {
    "left": np.zeros(_BRAINCO_NUM_MOTORS, dtype=np.int32),
    "right": np.zeros(_BRAINCO_NUM_MOTORS, dtype=np.int32),
}


def _set_brainco_state(hand: str, msg) -> None:
    with _BRAINCO_STATE_LOCK:
        _BRAINCO_STATES[hand] = msg

def _get_brainco():
    global _BRAINCO_CTRL, _BRAINCO_STATE_SUBSCRIBERS
    if _BRAINCO_CTRL is None:
        import sys as _sys
        if "/home/unitree/gr00t-g1-bridge" not in _sys.path:
            _sys.path.insert(0, "/home/unitree/gr00t-g1-bridge")
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorStates_
        try:
            ChannelFactoryInitialize(0, "wlxfc23cd997021")
        except Exception as e:
            print(f"[BrainCo] ChannelFactoryInitialize: {e}", flush=True)

        for hand, topic in (
            ("left", "rt/brainco/left/state"),
            ("right", "rt/brainco/right/state"),
        ):
            subscriber = ChannelSubscriber(topic, MotorStates_)
            subscriber.Init(
                lambda msg, hand=hand: _set_brainco_state(hand, msg),
                10,
            )
            _BRAINCO_STATE_SUBSCRIBERS.append(subscriber)

        from brainco_hand import BraincoHandController
        _BRAINCO_CTRL = BraincoHandController()
        _BRAINCO_CTRL.wait_ready(10.0)
        print("[BrainCo] hand state reader ready (6 DOF)", flush=True)
    return _BRAINCO_CTRL

def _to7(h):
    """6 моторов BrainCo -> 7-мерный формат, который ждёт WBC."""
    h = np.asarray(h, dtype=np.float32).reshape(-1)[:6]
    out = np.zeros(7, dtype=np.float32)
    out[:6] = h
    out[6] = h[5]
    return out


_slew_prev = [None]  # SLEW-PATCH


def _slew_limit(token):  # SLEW-PATCH
    """
    Cap the per-tick change of motion_token.

    Measured chunk-boundary jumps reach 0.49 in L2 against a normal playback
    step of 0.017, which is what the arm feels as a jolt. This spreads such a
    jump over several ticks. Returns the token unchanged when SLEW_ENABLE is
    not set, so the default path is untouched.
    """
    import os as _os
    if _os.environ.get("SLEW_ENABLE", "0") != "1":
        return token
    try:
        import numpy as _np
        cap = float(_os.environ.get("SLEW_MAX", "0.05"))
        cur = _np.asarray(token, dtype=_np.float32).reshape(-1)
        prev = _slew_prev[0]
        if prev is None or prev.shape != cur.shape:
            _slew_prev[0] = cur.copy()
            return token
        delta = cur - prev
        dist = float(_np.linalg.norm(delta))
        if dist > cap:
            cur = prev + delta * (cap / dist)
        _slew_prev[0] = cur.copy()
        return cur.reshape(_np.asarray(token).shape)
    except Exception as e:  # noqa: BLE001
        print(f"[slew] limiter failed, passing through: {e}", flush=True)
        return token


_slew_hand_prev = {"left": None, "right": None}  # SLEW-PATCH


def _slew_hand(vec, hand: str):  # SLEW-PATCH
    """
    Cap the per-tick change of one hand command.

    Measured: at chunk boundaries the flex command snaps BACKWARD by up to
    0.499, because the new plan expects the hand less closed than it already
    is. A genuine release moves about 0.014 per tick, so the default cap of
    0.03 passes real motion at double speed and stretches a snap over ~14
    ticks. Left and right history are tracked independently. SLEW_HAND=0
    disables just this part.
    """
    import os as _os
    if _os.environ.get("SLEW_ENABLE", "0") != "1":
        return vec
    try:
        import numpy as _np
        cap = float(_os.environ.get("SLEW_HAND", "0.03"))
        if cap <= 0:
            return vec
        cur = _np.asarray(vec, dtype=_np.float32).reshape(-1).copy()
        prev = _slew_hand_prev[hand]
        if prev is None or prev.shape != cur.shape:
            _slew_hand_prev[hand] = cur.copy()
            return vec
        step = _np.clip(cur - prev, -cap, cap)
        cur = prev + step
        _slew_hand_prev[hand] = cur.copy()
        return cur.reshape(_np.asarray(vec).shape)
    except Exception as e:  # noqa: BLE001
        print(f"[slew] {hand} hand limiter failed, passing through: {e}", flush=True)
        return vec


def _filter_brainco_contacts(
    hand: str,
    target,
    position_error_threshold: float,
    contact_count: int,
    hold_seconds: float,
) -> np.ndarray:
    """Block further closing after contact while always allowing opening."""
    target = np.clip(
        np.asarray(target, dtype=np.float32).reshape(_BRAINCO_NUM_MOTORS),
        0.0,
        1.0,
    )
    previous = _brainco_last_command[hand]
    if previous is None:
        _brainco_last_command[hand] = target.copy()
        return target

    with _BRAINCO_STATE_LOCK:
        state = _BRAINCO_STATES[hand]
        if state is not None and len(state.states) >= _BRAINCO_NUM_MOTORS:
            state_q = np.asarray(
                [state.states[idx].q for idx in range(_BRAINCO_NUM_MOTORS)],
                dtype=np.float32,
            )
        else:
            state_q = None

    output = previous.copy()
    blocked_until = _brainco_contact_blocked_until[hand]
    counters = _brainco_contact_counter[hand]
    now = time.monotonic()

    for idx in range(_BRAINCO_NUM_MOTORS):
        target_q = float(target[idx])
        previous_q = float(previous[idx])

        # Smaller normalized q opens the finger. Opening always passes through
        # and rearms contact detection for the next closing motion.
        if target_q < previous_q:
            if now < blocked_until[idx]:
                print(
                    f"[BrainCo] RELEASE {hand} {_BRAINCO_MOTOR_NAMES[idx]}: "
                    f"q_cmd={target_q:.3f}",
                    flush=True,
                )
            output[idx] = target_q
            blocked_until[idx] = 0.0
            counters[idx] = 0
            continue

        if now < blocked_until[idx]:
            # Ignore equal or larger VLA targets until the contact timer expires.
            output[idx] = previous_q
            continue

        if blocked_until[idx] > 0.0:
            blocked_until[idx] = 0.0
            counters[idx] = 0
            print(
                f"[BrainCo] CONTACT TIMER EXPIRED {hand} "
                f"{_BRAINCO_MOTOR_NAMES[idx]}",
                flush=True,
            )

        if target_q > previous_q and state_q is not None:
            contact = previous_q - float(state_q[idx]) > position_error_threshold
            counters[idx] = counters[idx] + 1 if contact else 0
            if counters[idx] >= contact_count:
                output[idx] = float(np.clip(state_q[idx], 0.0, 1.0))
                blocked_until[idx] = now + hold_seconds
                counters[idx] = 0
                print(
                    f"[BrainCo] CONTACT {hand} {_BRAINCO_MOTOR_NAMES[idx]}: "
                    f"q={output[idx]:.3f}, block={hold_seconds:.2f}s",
                    flush=True,
                )
                continue
        else:
            counters[idx] = 0

        output[idx] = target_q

    _brainco_last_command[hand] = output.copy()
    return output


def _send_brainco_hands(
    left,
    right,
    position_error_threshold: float,
    contact_count: int,
    hold_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter and send 6-DOF commands directly to BrainCo DDS."""
    import os as _os
    enable = _os.environ.get("HAND_ENABLE", "0") == "1"
    l = np.clip(np.asarray(left, dtype=np.float32).reshape(-1)[:6], 0.0, 1.0)
    r = np.clip(np.asarray(right, dtype=np.float32).reshape(-1)[:6], 0.0, 1.0)
    try:
        ctrl = _get_brainco()
        if enable:
            l = _filter_brainco_contacts(
                "left", l, position_error_threshold, contact_count, hold_seconds
            )
            r = _filter_brainco_contacts(
                "right", r, position_error_threshold, contact_count, hold_seconds
            )
            # Keep slew history aligned with the command that was actually sent.
            _slew_hand_prev["left"] = l.copy()
            _slew_hand_prev["right"] = r.copy()
        ctrl.send_targets_normalized(l, r, dry_run=not enable)
        if not enable:
            print(f"[BrainCo] DRY-RUN right={r.round(3)}", flush=True)
    except Exception as e:
        print(f"[BrainCo] send failed: {e}", flush=True)
    return l, r


def _override_hand_state(observation):
    try:
        left, right = _get_brainco().get_state_normalized()
        left = np.asarray(left, dtype=np.float32).reshape(-1)
        right = np.asarray(right, dtype=np.float32).reshape(-1)
        if left.size != 6 or right.size != 6:
            raise ValueError(
                "BrainCo state must contain exactly 6 motors per hand, got "
                f"left={left.size}, right={right.size}"
            )
        observation["state"]["left_hand"] = left.reshape(1, 1, 6)
        observation["state"]["right_hand"] = right.reshape(1, 1, 6)
    except Exception as e:
        print(
            f"[BrainCo] state override failed; skipping inference: {e}",
            flush=True,
        )
        return None
    return observation


def prepare_observation_from_sensors(
    camera_subscriber,
    state_subscriber,
    robot_model,
    language_prompt: str,
    log_errors: bool = False,
):
    """Read sensors and prepare observation for the VLA policy.

    Returns:
        observation dict, or None if sensor data not yet available.
    """
    camera_msg = camera_subscriber.read()
    if camera_msg is None:
        if log_errors:
            print("[DEBUG] prepare_observation: waiting for camera msg..", flush=True)
        return None

    state_msg = state_subscriber.get_msg()
    if state_msg is None:
        if log_errors:
            print("[DEBUG] prepare_observation: waiting for state msg..", flush=True)
        return None

    images = camera_msg.get("images", {})
    missing_cameras = [name for name in ("ego_view", "head") if images.get(name) is None]
    if missing_cameras:
        if log_errors:
            print(
                "[DEBUG] prepare_observation: missing required camera stream(s) "
                f"{missing_cameras}; available={sorted(images)}",
                flush=True,
            )
        return None

    import cv2 as _cv2

    def resize_for_policy(image: np.ndarray) -> np.ndarray:
        """Reduce transport size without changing the camera aspect ratio."""
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected an HxWx3 camera image, got {getattr(image, 'shape', None)}")
        height, width = image.shape[:2]
        if min(height, width) <= 256:
            return image
        scale = 256 / min(height, width)
        return _cv2.resize(
            image,
            (int(round(width * scale)), int(round(height * scale))),
            interpolation=_cv2.INTER_AREA,
        )

    ego_img = resize_for_policy(images["ego_view"])
    head_img = resize_for_policy(images["head"])

    # Copy index finger data to middle finger (hardware coupling)
    state_msg["left_hand_q"][5] = state_msg["left_hand_q"][3]
    state_msg["left_hand_q"][6] = state_msg["left_hand_q"][4]

    qpos = robot_model.get_configuration_from_actuated_joints(
        body_actuated_joint_values=state_msg["body_q"],
        left_hand_actuated_joint_values=state_msg["left_hand_q"],
        right_hand_actuated_joint_values=state_msg["right_hand_q"],
    )

    # These keys must exactly match the two-camera ModalityConfig saved in the
    # GR00T checkpoint. The source camera names remain ego_view and head.
    video = {
        "depth_camera_rgb": ego_img[np.newaxis, np.newaxis],
        "webcam": head_img[np.newaxis, np.newaxis],
    }

    observation = {
        "video": video,
        "state": {},
        "language": {
            "annotation.human.task_description": [[language_prompt]],
        },
        "q": np.asarray(qpos, dtype=np.float32)[np.newaxis, np.newaxis],
        "timestamps": camera_msg["timestamps"]["ego_view"],
    }

    observation = prepare_observation_for_eval(robot_model, observation)
    observation = _override_hand_state(observation)
    if observation is None:
        return None

    # Projected gravity for Sonic latent embodiment
    assert "base_quat" in state_msg, "base_quat not found in state_msg"
    base_quat = np.asarray(state_msg["base_quat"], dtype=np.float64)
    assert base_quat.shape == (4,), "base_quat must have shape (4,)"
    projected_gravity = compute_projected_gravity(base_quat)
    observation["state"]["projected_gravity"] = np.asarray(
        projected_gravity, dtype=np.float32
    )[np.newaxis, np.newaxis]

    return observation


def run_policy_inference_and_process(policy, observation, robot_model):
    """Run policy inference via Isaac-GR00T PolicyClient and process results.

    Returns:
        processed_action dict or None on error.
    """
    try:
        try:
            for k, vv in observation.get("video", {}).items():
                print(f"[DBG] video[{k}] type={type(vv)}", flush=True)
                if isinstance(vv, dict):
                    print(f"[DBG]   keys={list(vv.keys())[:5]}", flush=True)
            for k, vv in observation.get("state", {}).items():
                print(f"[DBG] state[{k}] shape={getattr(vv,'shape',None)}", flush=True)
        except Exception as e:
            print("[DBG] fail:", e, flush=True)
        action, _info = policy.get_action(observation)

        action.pop("task_progress", None)
        action.pop("action.task_progress", None)

        motion_key = "motion_token" if "motion_token" in action else "action.motion_token"
        if np.abs(action[motion_key]).max() > 1.25:
            print(
                f"[Warning] action['{motion_key}'] max "
                f"({np.abs(action[motion_key]).max():.4f}) > 1.25. "
                "Exceeds action bound, skipping."
            )
            return None

        processed_action = concat_action(robot_model, action)
        return processed_action
    except Exception as e:
        print(f"Error in inference: {e}")
        import traceback

        traceback.print_exc()
        return None


def _inference_worker_loop(
    inference_queue: queue.Queue,
    result_queue: queue.Queue,
    stop_event: threading.Event,
    busy_event: threading.Event,
    prepare_obs_fn,
    inference_fn,
):
    """Persistent worker thread for async inference."""
    while not stop_event.is_set():
        try:
            try:
                request = inference_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            busy_event.set()
            try:
                observation = prepare_obs_fn(request.prompt)
                if observation is None:
                    print("[DEBUG] Worker thread: Observation is None, skipping", flush=True)
                    continue

                inference_start_time = time.monotonic()
                processed_action = inference_fn(observation)

                if processed_action is not None:
                    try:
                        result_queue.put_nowait(
                            InferenceResult(
                                intent_epoch=request.intent_epoch,
                                prompt=request.prompt,
                                action=processed_action,
                                inference_start_time=inference_start_time,
                            )
                        )
                    except queue.Full:
                        try:
                            result_queue.get_nowait()
                            result_queue.put_nowait(
                                InferenceResult(
                                    intent_epoch=request.intent_epoch,
                                    prompt=request.prompt,
                                    action=processed_action,
                                    inference_start_time=inference_start_time,
                                )
                            )
                        except queue.Empty:
                            result_queue.put_nowait(
                                InferenceResult(
                                    intent_epoch=request.intent_epoch,
                                    prompt=request.prompt,
                                    action=processed_action,
                                    inference_start_time=inference_start_time,
                                )
                            )
            finally:
                busy_event.clear()
        except Exception as e:
            print(f"Error in inference worker thread: {e}")
            import traceback

            traceback.print_exc()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _compute_closed_hand_joints(side: str) -> np.ndarray:
    """Compute closed hand joint positions using G1GripperInverseKinematicsSolver."""
    side_str = "left" if side.upper() == "L" else "right"
    solver = G1GripperInverseKinematicsSolver(side=side_str)
    return solver._get_middle_close_q_desired().astype(np.float32)


def main(config: InferenceConfig):
    if config.brainco_contact_position_error_threshold < 0:
        raise ValueError(
            "brainco_contact_position_error_threshold must be non-negative"
        )
    if config.brainco_contact_count <= 0:
        raise ValueError("brainco_contact_count must be positive")
    if config.brainco_contact_hold_seconds < 0:
        raise ValueError("brainco_contact_hold_seconds must be non-negative")
    if not 1 <= config.compliance_port <= 65535:
        raise ValueError("compliance_port must be between 1 and 65535")
    if config.compliance_rate <= 0:
        raise ValueError("compliance_rate must be positive")

    intent_controller = IntentController(
        mode=config.intent_mode,
        initial_prompt=config.prompt,
        max_age=config.intent_max_age,
        unknown_grace=config.intent_unknown_grace,
    )

    pause_loop = True
    freeze_body = False        # body frozen: skip motion_token, WBC holds pose

    robot_model = instantiate_g1_robot_model(waist_location="lower_and_upper_body")

    # Isaac-GR00T PolicyClient
    import sys; sys.path.insert(0, "/home/unitree/gr00t-g1-bridge")
    from groot_client_liza import Gr00tClient as PolicyClient

    n1_policy = PolicyClient(host=config.host, port=config.port)

    print(f"Connecting to PolicyServer at {config.host}:{config.port}...")
    if n1_policy.ping():
        print_green("PolicyServer is reachable.")
    else:
        print("WARNING: PolicyServer not reachable. Inference will fail until server is up.")

    state_subscriber = ZMQStateSubscriber(
        host=config.state_zmq_host,
        port=config.state_zmq_port,
    )

    camera_subscriber = ComposedCameraClientSensor(
        server_ip=config.camera_host, port=config.camera_port
    )
    intent_subscriber = HumanIntentSubscriber(
        host=config.intent_host, port=config.intent_port
    )
    print_green(
        f"Human intent SUB connected to {intent_subscriber.endpoint} "
        f"(mode={intent_controller.mode})"
    )

    zmq_context = zmq.Context()
    zmq_socket = zmq_context.socket(zmq.PUB)
    zmq_socket.bind(f"tcp://{config.action_zmq_host}:{config.action_zmq_port}")
    time.sleep(0.1)
    print_green(
        f"ZMQ action socket bound to tcp://{config.action_zmq_host}:{config.action_zmq_port}"
    )
    print_green(f"Using embodiment tag: {config.embodiment_tag}")

    compliance_socket = None
    compliance_publisher = None
    if config.intent_compliance:
        compliance_socket = zmq_context.socket(zmq.PUB)
        compliance_socket.setsockopt(zmq.LINGER, 0)
        compliance_socket.setsockopt(zmq.SNDHWM, 1)
        compliance_socket.bind(f"tcp://*:{config.compliance_port}")
        compliance_publisher = ComplianceProfilePublisher(
            compliance_socket, rate_hz=config.compliance_rate
        )
        compliance_publisher.set_prompt(intent_controller.prompt)
        print_green(
            f"Intent compliance PUB bound to tcp://*:{config.compliance_port}; "
            f"initial profile={compliance_publisher.profile}; "
            f"heartbeat={config.compliance_rate:.1f} Hz"
        )

    keyboard_listener = ZMQKeyboardSubscriber(
        port=config.keyboard_zmq_port, host=config.keyboard_zmq_host
    )

    telemetry = Telemetry(window_size=100)

    loop_rate = config.action_publish_rate
    loop_period = 1.0 / loop_rate

    # Track C++ control loop state
    cpp_loop_running = False
    cpp_mode = "OFF"  # "OFF", "PLANNER", or "POSE"

    # Track initial pose hand states
    initial_pose_left_hand_closed = False
    initial_pose_right_hand_closed = False

    def publish_initial_pose():
        """Publish initial pose command to move robot to starting position."""
        print("Moving to initial pose")
        left_hand = (
            _compute_closed_hand_joints("L")
            if initial_pose_left_hand_closed
            else np.zeros(7, dtype=np.float32)
        )
        right_hand = (
            _compute_closed_hand_joints("R")
            if initial_pose_right_hand_closed
            else np.zeros(7, dtype=np.float32)
        )
        zmq_message = pack_latent_action_message(
            motion_token=LATENT_INITIAL_MOTION_TOKEN,
            frame_index=np.array([0], dtype=np.int64),
            left_hand_joints=left_hand,
            right_hand_joints=right_hand,
        )
        zmq_socket.send(zmq_message)
        print_green("Sent latent initial pose via ZMQ")
        time.sleep(1.0)
        print("Initial pose done.")

    def send_cpp_control_command(start: bool, planner: bool = False):
        """Send C++ control loop start/stop commands via ZMQ."""
        nonlocal cpp_loop_running, cpp_mode
        try:
            cmd_msg = build_command_message(start=start, stop=not start, planner=planner)
            zmq_socket.send(cmd_msg)
            time.sleep(0.01)
            action_str = "start" if start else "stop"
            mode_str = "planner" if planner else "pose"
            cpp_loop_running = start
            if start:
                cpp_mode = "PLANNER" if planner else "POSE"
            else:
                cpp_mode = "OFF"
            print_green(f"Sent ZMQ command: {action_str} control loop ({mode_str} mode)")
            return True
        except Exception as e:
            action_str = "start" if start else "stop"
            print(f"Warning: Failed to send {action_str} command message: {e}")
            return False

    # Async inference state
    cached_action_chunk = None
    action_chunk_index = 0
    last_inference_time = 0.0
    probe = ChunkProbe(enabled=os.environ.get('PROBE_ENABLE') == '1')  # PROBE-PATCH
    print(f'[probe] enabled={probe.enabled}')  # PROBE-PATCH
    inference_interval = 1.0 / config.rate

    zmq_frame_counter = 0

    language_prompt_ref: list[str] = [intent_controller.prompt]
    inference_queue = queue.Queue(maxsize=1)
    result_queue = queue.Queue(maxsize=1)
    inference_stop_event = threading.Event()
    inference_busy_event = threading.Event()
    applied_intent_epoch = intent_controller.epoch
    last_hold_log_time = 0.0

    def drain(q: queue.Queue) -> None:
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                return

    def apply_intent_state() -> None:
        """Invalidate cached/queued actions whenever prompt authority changes."""
        nonlocal applied_intent_epoch, cached_action_chunk, action_chunk_index
        nonlocal last_inference_time
        if applied_intent_epoch == intent_controller.epoch:
            return

        old_prompt = language_prompt_ref[0]
        language_prompt_ref[0] = intent_controller.prompt
        applied_intent_epoch = intent_controller.epoch
        cached_action_chunk = None
        action_chunk_index = 0
        last_inference_time = 0.0
        drain(inference_queue)
        drain(result_queue)
        if compliance_publisher is not None and intent_controller.inference_enabled:
            if compliance_publisher.set_prompt(intent_controller.prompt):
                print_green(
                    f"Arm compliance -> {compliance_publisher.profile} "
                    f'(preparing prompt "{intent_controller.prompt}")'
                )
        print_green(
            f'Intent epoch {intent_controller.epoch}: prompt "{old_prompt}" -> '
            f'"{intent_controller.prompt}"; mode={intent_controller.mode}; '
            f'hold={intent_controller.execution_hold} ({intent_controller.hold_reason})'
        )

    # ``pr <text>`` is the command format emitted by the ZMQ controller.
    # Keep ``prompt:<text>`` for compatibility with the previous publisher.
    PROMPT_MSG_PREFIX = "prompt:"
    PROMPT_COMMAND_PREFIX = "pr "

    def check_keyboard_input():
        nonlocal pause_loop, freeze_body, cpp_loop_running, cpp_mode
        nonlocal initial_pose_left_hand_closed, initial_pose_right_hand_closed
        nonlocal cached_action_chunk, action_chunk_index, last_inference_time
        nonlocal zmq_frame_counter

        key = keyboard_listener.read_msg()
        if key is None:
            return

        if key.startswith(PROMPT_COMMAND_PREFIX):
            new_prompt = key[len(PROMPT_COMMAND_PREFIX):].strip()
        elif key.startswith(PROMPT_MSG_PREFIX):
            new_prompt = key[len(PROMPT_MSG_PREFIX):].strip()
        else:
            new_prompt = None

        if new_prompt is not None:
            if new_prompt:
                if new_prompt.lower() == "auto":
                    intent_controller.enable_auto()
                else:
                    intent_controller.set_manual_prompt(new_prompt)
                apply_intent_state()
            else:
                print("Received empty prompt change -- ignoring.")
            return

        if key == "c":
            print("Keyboard: 'c' (start recording -- handled by data exporter)")
        elif key == "s":
            print("Keyboard: 's' (stop recording success -- handled by data exporter)")
        elif key == "f":
            print("Keyboard: 'f' (stop recording failure -- handled by data exporter)")
        elif key == "i":
            print("Moving to initial pose")
            zmq_frame_counter = 0
            print("Reset ZMQ frame counter")
            publish_initial_pose()
            intent_controller.restart("waiting for fresh chunk after initial pose")
            apply_intent_state()
            print("Cleared cached action chunk")
            if cpp_loop_running and cpp_mode == "PLANNER":
                if send_cpp_control_command(start=True, planner=False):
                    print("Switched to POSE mode (from PLANNER mode)")
                else:
                    print("Warning: Failed to switch to POSE mode")
            elif not cpp_loop_running:
                print("Note: C++ loop not running - press 'k' to start")
        elif key == "g":
            freeze_body = not freeze_body
            print(f"{'FROZEN body (fingers live)' if freeze_body else 'Body resumed'} - WBC holds pose")
        elif key == "p":
            pause_loop = not pause_loop
            print(f"{'Paused' if pause_loop else 'Resumed'} policy loop")
            if pause_loop:
                print("Policy loop paused (C++ loop still running - press 'k' to stop)")
            else:
                print("Policy loop resumed")
        elif key == "k":
            if cpp_loop_running:
                current_planner = cpp_mode == "PLANNER"
                print(f"Stopping C++ control loop (from {cpp_mode} mode)...")
                if send_cpp_control_command(start=False, planner=current_planner):
                    print("Stopped C++ control loop")
            else:
                print("Starting C++ control loop in PLANNER mode...")
                if send_cpp_control_command(start=True, planner=True):
                    print("Started C++ control loop in PLANNER mode")
                    print("Press 'i' to send initial pose and switch to POSE mode")
                    if pause_loop:
                        print("Note: Policy loop is paused - press 'p' to resume")
        elif key == "[":
            initial_pose_left_hand_closed = not initial_pose_left_hand_closed
            print(
                f"Initial pose left hand: {'closed' if initial_pose_left_hand_closed else 'open'}"
            )
        elif key == "]":
            initial_pose_right_hand_closed = not initial_pose_right_hand_closed
            print(
                f"Initial pose right hand: "
                f"{'closed' if initial_pose_right_hand_closed else 'open'}"
            )

    print(
        f"Starting the policy loop with language prompt: {language_prompt_ref[0]} "
        f"(intent mode: {intent_controller.mode})"
    )

    inference_worker_thread = threading.Thread(
        target=_inference_worker_loop,
        args=(
            inference_queue,
            result_queue,
            inference_stop_event,
            inference_busy_event,
            lambda prompt: prepare_observation_from_sensors(
                camera_subscriber=camera_subscriber,
                state_subscriber=state_subscriber,
                robot_model=robot_model,
                language_prompt=prompt,
                log_errors=True,
            ),
            lambda obs: run_policy_inference_and_process(
                policy=n1_policy,
                observation=obs,
                robot_model=robot_model,
            ),
        ),
        daemon=True,
    )
    inference_worker_thread.start()

    try:
        while True:
            t_start = time.monotonic()
            check_keyboard_input()
            if compliance_publisher is not None:
                compliance_publisher.tick(t_start)

            intent_event = intent_subscriber.read_latest()
            intent_now = time.monotonic()
            if intent_event is not None:
                received_label = intent_event["label"]
                received_prompt = INTENT_TO_PROMPT.get(received_label)
                accepted = bool(intent_event["accepted"])
                prompt_text = f'"{received_prompt}"' if accepted and received_prompt else "<SAFE HOLD>"
                print(
                    f"[intent] received seq={intent_event['seq']}: "
                    f"label={received_label} -> prompt={prompt_text}; "
                    f"confidence={intent_event['confidence']:.3f}; "
                    f"status={'accepted' if accepted else 'rejected'}",
                    flush=True,
                )
                intent_controller.process_event(intent_event, intent_now)
            intent_controller.tick(intent_now)
            apply_intent_state()

            # Consume result first so last_inference_time is fresh before trigger check
            try:
                inference_result = result_queue.get_nowait()
            except queue.Empty:
                inference_result = None

            if inference_result is not None:
                if (
                    inference_result.intent_epoch != intent_controller.epoch
                    or not intent_controller.inference_enabled
                ):
                    print(
                        f"[intent] discarded stale VLA chunk: "
                        f"result epoch={inference_result.intent_epoch}, "
                        f"current epoch={intent_controller.epoch}",
                        flush=True,
                    )
                else:
                    processed_action = inference_result.action
                    inference_delay = (
                        time.monotonic() - inference_result.inference_start_time
                    )
                    action_chunk_index = calculate_latency_compensated_index(
                        inference_delay,
                        config.action_publish_rate,
                        config.action_horizon,
                    )
                    cached_action_chunk = processed_action
                    intent_controller.accept_result(inference_result.intent_epoch)
                    if probe.enabled:  # PROBE-PATCH
                        probe.note_delay(inference_delay, action_chunk_index)  # PROBE-PATCH
                    last_inference_time = time.monotonic()
                    print_green(
                        f'New action chunk (prompt: "{inference_result.prompt}", '
                        f"intent epoch: {inference_result.intent_epoch}, "
                        f"latency: {inference_delay:.3f}s)"
                    )

            worker_is_busy = inference_busy_event.is_set()
            should_start = should_trigger_new_inference(
                cached_chunk_exists=(cached_action_chunk is not None),
                inference_thread_running=worker_is_busy,
                time_since_last_inference=(time.monotonic() - last_inference_time),
                inference_interval=inference_interval,
            )

            if should_start and intent_controller.inference_enabled:
                try:
                    inference_queue.put_nowait(
                        InferenceRequest(
                            intent_epoch=intent_controller.epoch,
                            prompt=intent_controller.prompt,
                        )
                    )
                except queue.Full:
                    pass

            if pause_loop:
                print("Pausing...", end="", flush=True)
                time.sleep(0.2)
                print(".", end="", flush=True)
                continue

            if intent_controller.execution_hold:
                now = time.monotonic()
                if now - last_hold_log_time >= 2.0:
                    print(
                        f"[intent] SAFE HOLD: {intent_controller.hold_reason}",
                        flush=True,
                    )
                    last_hold_log_time = now
                _sleep_remaining(t_start, loop_period)
                continue

            with telemetry.timer("total_loop"):
                if cached_action_chunk is None:
                    print("[DEBUG] No cached chunk yet, waiting...", flush=True)
                    _sleep_remaining(t_start, loop_period)
                    continue

                processed_action = cached_action_chunk

                if processed_action is None or not processed_action:
                    print("[DEBUG] processed_action is None or empty, skipping", flush=True)
                else:
                    motion_token = np.asarray(
                        get_action_field(processed_action, "motion_token"),
                        dtype=np.float32,
                    )
                    left_hand_joints = np.asarray(
                        get_action_field(processed_action, "left_hand_joints"),
                        dtype=np.float32,
                    )
                    right_hand_joints = np.asarray(
                        get_action_field(processed_action, "right_hand_joints"),
                        dtype=np.float32,
                    )

                    # Action arrays arrive as (B, T, D) from the model.
                    # Squeeze batch dim to get (T, D), then index by time step.
                    if motion_token.ndim == 3:
                        motion_token = motion_token[0]
                    if left_hand_joints.ndim == 3:
                        left_hand_joints = left_hand_joints[0]
                    if right_hand_joints.ndim == 3:
                        right_hand_joints = right_hand_joints[0]

                    horizon = motion_token.shape[0] if motion_token.ndim == 2 else 1
                    current_idx = min(action_chunk_index, horizon - 1)

                    if probe.enabled:  # PROBE-PATCH
                        probe.on_step(left_hand_joints if os.environ.get('PROBE_HAND')=='left' else right_hand_joints, current_idx)  # PROBE-PATCH

                    if motion_token.ndim == 2:
                        motion_token = motion_token[current_idx]
                    if left_hand_joints.ndim == 2:
                        left_hand_joints = left_hand_joints[current_idx]
                    if right_hand_joints.ndim == 2:
                        right_hand_joints = right_hand_joints[current_idx]

                    frame_index = np.array([zmq_frame_counter], dtype=np.int64)
                    zmq_frame_counter += 1

                    # Apply independent slew limiting to both VLA hand commands.
                    left_hand_joints = _slew_hand(left_hand_joints, "left")
                    right_hand_out = np.asarray(right_hand_joints).copy()

                    # THUMBLOCK-PATCH: большой палец защёлкивается от первой же
                    # команды на сгиб (13 авг) — не даём её вообще.
                    # Снять после ремонта: --revert
                    right_hand_out[0] = 0.0  # THUMBLOCK-PATCH
                    right_hand_out = _slew_hand(right_hand_out, "right")
                    right_hand_out[0] = 0.0  # Keep the thumb hard-blocked after slew.

                    left_hand_joints, right_hand_out = _send_brainco_hands(
                        left_hand_joints,
                        right_hand_out,
                        position_error_threshold=(
                            config.brainco_contact_position_error_threshold
                        ),
                        contact_count=config.brainco_contact_count,
                        hold_seconds=config.brainco_contact_hold_seconds,
                    )
                    if probe.enabled:  # PROBE-PATCH
                        probe.on_sent(left_hand_joints if os.environ.get('PROBE_HAND')=='left' else right_hand_out, motion_token)  # PROBE-PATCH
                    # body/motion_token skipped when frozen - WBC holds the last pose itself
                    if not freeze_body:
                        motion_token = _slew_limit(motion_token)  # SLEW-PATCH
                        zmq_message = pack_latent_action_message(
                            motion_token,
                            frame_index,
                            left_hand_joints=_to7(left_hand_joints),
                            right_hand_joints=_to7(right_hand_out),
                        )
                        zmq_socket.send(zmq_message)
                        if zmq_frame_counter % 50 == 0:
                            print_green(
                                f"ZMQ: Sent latent action - "
                                f"frame: {frame_index[0]}, "
                                f"token shape: {motion_token.shape}"
                            )

                action_chunk_index = min(action_chunk_index + 1, config.action_horizon - 1)

            end_time = time.monotonic()

            if config.verbose_timing:
                telemetry.log_timing_info(context="VLA Inference Loop", threshold=0.0)
            elif (end_time - t_start) > (1 / config.rate):
                telemetry.log_timing_info(
                    context="VLA Inference Loop Missed", threshold=0.001
                )

            _sleep_remaining(t_start, loop_period)

    except KeyboardInterrupt:
        print("VLA inference loop terminated by user")

    finally:
        inference_stop_event.set()
        inference_worker_thread.join(timeout=1.0)
        if compliance_socket is not None:
            compliance_socket.close()
        zmq_socket.close()
        zmq_context.term()
        state_subscriber.close()
        camera_subscriber.close()
        intent_subscriber.close()
        keyboard_listener.close()
        print("Shutdown complete.")


def _sleep_remaining(t_start: float, loop_period: float):
    """Sleep for the remainder of the loop period."""
    elapsed = time.monotonic() - t_start
    remaining = loop_period - elapsed
    if remaining > 0:
        time.sleep(remaining)


if __name__ == "__main__":
    config = tyro.cli(InferenceConfig)
    main(config)
