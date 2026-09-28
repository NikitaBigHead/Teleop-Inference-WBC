# GR00T latency diagnostics

These tools profile a diagnostic PolicyServer without changing the normal
bridge or checkpoint. They isolate network serialization, CPU preprocessing,
CUDA kernels, action decoding, and response serialization.

The robot-side wrapper additionally records the local camera ZMQ receive and
decode path, state read, observation construction and the full remote RPC.

## 1. Copy the tools to the running container

From this repository, copy the profiler to the A100 and then into
`tactile-train`:

```bash
scp debug/profile_gr00t_server.py root@100.64.0.21:/tmp/
ssh root@100.64.0.21 \
  'docker cp /tmp/profile_gr00t_server.py tactile-train:/opt/Isaac-GR00T/debug_profile_gr00t_server.py'
```

Keep the host `socat` process running. Stop only the normal GR00T server, then
inside the container run the profiled replacement:

```bash
source /opt/Isaac-GR00T/.venv/bin/activate
cd /opt/Isaac-GR00T
python debug_profile_gr00t_server.py \
  --gr00t-root /opt/Isaac-GR00T \
  --model-path /workspace/checkpoints/tasks-affect-500-episodes-two-cam/checkpoint-30000 \
  --embodiment-tag NEW_EMBODIMENT \
  --device cuda:0 \
  --port 5555 \
  --profile-log /tmp/gr00t_profile.jsonl
```

Run 15–30 actions after the model has warmed up, then stop it with `Ctrl-C`.

## 2. Collect host metrics simultaneously

In a second terminal on the A100 host:

```bash
bash /path/to/monitor_a100_runtime.sh /tmp/gr00t_runtime 60 0.5
```

This creates `metrics.csv` with GPU utilization/memory and PolicyServer CPU,
plus `tcp.txt` with TCP send/receive queues for port 5555.

## 3. Profile the bridge on the robot

Copy `profile_affective_bridge.py` to the robot repository. Launch it with the
same arguments as the normal bridge; only the script path and an environment
variable change:

```bash
cd /home/unitree/teleop-ws/Teleop-Inference-WBC
source ~/GR00T-WholeBodyControl/.venv_data_collection/bin/activate
BRIDGE_PROFILE_LOG=/tmp/bridge_profile.jsonl \
python debug/profile_affective_bridge.py \
  --host 100.64.0.21 --port 5555 \
  --camera-host localhost --camera-port 5555 \
  --state-zmq-host localhost --state-zmq-port 5557 \
  --action-zmq-host localhost --action-zmq-port 5556 \
  --embodiment-tag NEW_EMBODIMENT --prompt hug
```

In another terminal on the robot:

```bash
bash debug/monitor_robot_runtime.sh /tmp/vla_robot_runtime 60 0.5
```

`profile_affective_bridge.py` does not alter the source bridge: it imports it
and instruments that process only. Both the normal bridge and profiled bridge
must not run simultaneously because they would publish actions to the same WBC.

## 4. Analyse either JSONL log

Copy the JSONL log back, then run:

```bash
python debug/analyze_gr00t_profile.py /path/to/gr00t_profile.jsonl
```

The same command accepts `/tmp/bridge_profile.jsonl`. For the bridge output,
compare `camera_read_ms`, `observation_total_ms`, and `rpc_get_action_ms`.

Interpretation:

| Largest field | Meaning | Next action |
| --- | --- | --- |
| `deserialize_ms` or `request_bytes` | Raw NumPy image payload / network transport | Send final `uint8 256x256` images; measure Tailscale throughput. |
| `processor_ms` | CPU image transforms and tokenizer | Preprocess the exact evaluation transform on the robot; use fixed 256x256 shapes. |
| `model_wall_ms` and `model_cuda_ms` | Model is genuinely compute-bound | Keep BF16/FlashAttention; evaluate `torch.compile`; test fewer diffusion steps. |
| `model_wall_ms` much larger than `model_cuda_ms` | CPU dispatch, host→GPU copies, or Python inside model | Inspect `prepare_input`; use fixed shapes and pinned/preallocated inputs. |
| `decode_ms` | CPU unnormalization/output conversion | Keep output arrays small; profile `decode_action` separately. |
| `serialize_ms` | Response packing | Usually minor because an action chunk is small. |

On the robot, a high `camera_read_ms` means local ZMQ msgpack/JPEG decode or
camera backlog. A high `observation_total_ms - camera_read_ms` points to image
resize, robot state conversion, or BrainCo state access. A high
`rpc_get_action_ms` with low local timings is remote transport/server time.

The checkpoint has an action horizon of 50 at a 50 Hz controller: it covers
only one second. A policy result slower than one second is already stale, even
if WBC itself is healthy.
