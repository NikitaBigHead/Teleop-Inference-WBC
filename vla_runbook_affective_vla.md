# Ранбук: affective VLA
---
## 1. Порядок запуска

### A100 (в контейнере, одной строкой)

```bash
ssh root@100.64.0.21
docker start tactile-train
docker exec -it tactile-train bash
source /opt/Isaac-GR00T/.venv/bin/activate && cd /opt/Isaac-GR00T
pkill -f run_gr00t_server; sleep 3
python gr00t/eval/run_gr00t_server.py --model-path /workspace/checkpoints/tasks-affect-500-episodes-two-cam/checkpoint-30000 --embodiment-tag NEW_EMBODIMENT --port 5555
```

### A100, хост, в tmux

```bash
tmux new -s socat
socat TCP-LISTEN:5555,fork,reuseaddr TCP:172.17.0.2:5555
```

### Робот, по порядку

```bash
# 1. камера
cd ~/GR00T-WholeBodyControl && source .venv_camera/bin/activate
cd /home/unitree/teleop-ws/Teleop-Data-Collection
python -m gear_sonic.camera.composed_camera \
  --ego-view-camera realsense \
  --ego-view-device-id 243422071979 \
  --head-camera usb \
  --head-device-id /dev/video6 \
  --realsense-width 1280 \
  --realsense-height 720 \
  --head-camera-width 1280 \
  --head-camera-height 720 \
  --head-camera-fps 15 \
  --head-camera-quality 90 \
  --head-camera-fourcc MJPG \
  --no-realsense-depth \
  --fps 15 \
  --port 5555
```

##  2. Brainco  — РОВНО ОДИН экземпляр

```bash
/home/unitree/teleop-ws/Teleop-Inference-WBC/gear_sonic/scripts/start_brainco_and_check.sh   --interface wlxfc23cd997021
```

# 3. WBC — ждать «Init done»


## Default 
```bash
export TensorRT_ROOT="$HOME/TensorRT"
cd ~/GR00T-WholeBodyControl/gear_sonic_deploy && source scripts/setup_env.sh
./deploy.sh --cp policy/sonic_v1_1/model --obs-config policy/sonic_v1_1/observation_config.yaml --input-type zmq_manager real
```

## Impedance profiles for deploy

### Hugging params
```bash
export TensorRT_ROOT="$HOME/TensorRT"
cd ~/GR00T-WholeBodyControl/gear_sonic_deploy && source scripts/setup_env.sh
./deploy.sh   --input-type zmq_manager   --motor-kp-scale 15-28=0.6   --motor-kd-scale 15-28=0.83   real
```

### Handshake params 
```bash
export TensorRT_ROOT="$HOME/TensorRT"
cd ~/GR00T-WholeBodyControl/gear_sonic_deploy && source scripts/setup_env.sh
./deploy.sh \
  --input-type zmq_manager \
  --motor-kp-scale 18,25=0.6 \
  --motor-kd-scale 18,25=0.83 \
  real
```

### Fist bump params 
```bash
export TensorRT_ROOT="$HOME/TensorRT"
cd ~/GR00T-WholeBodyControl/gear_sonic_deploy && source scripts/setup_env.sh
./deploy.sh \
  --input-type zmq_manager \
  --motor-kp-scale 22-25=0.25 \
  --motor-kd-scale 22-25=0.5 \
  --motor-kp-scale 15-18=0.4 \
  --motor-kd-scale 15-18=0.632 \
  real

```
# 4. паблишер клавиш

```bash
source ~/GR00T-WholeBodyControl/.venv_data_collection/bin/activate
python3 ~/keypress.py
```
# 5. мост — команды в §3

### База (VLA одна)

```bash
cd ~/GR00T-WholeBodyControl && source .venv_data_collection/bin/activate && AGENT_HAND=left AGENT_ENABLE=0 SLEW_ENABLE=1 PROBE_HAND=left PROBE_ENABLE=1 HAND_ENABLE=1 PYTHONPATH=/home/unitree/unitree_sdk2_python python gear_sonic/scripts/run_inference_affective_vla.py --host 100.64.0.21 --port 5555 --camera-host localhost --camera-port 5555 --state-zmq-host localhost --state-zmq-port 5557 --action-zmq-host localhost --action-zmq-port 5556 --embodiment-tag NEW_EMBODIMENT --prompt "hug" 2>&1 | tee ~/bridge_base_$(date +%Y%m%d_%H%M).log
```

**Клавиши:** `k` старт → `i` начальная поза (обязательно) → `p` снять паузу (тумблер) → `g` заморозка тела.





# Copying the project to Unitree over SSH

```bash
rsync -avh --partial --info=progress2 \
  --filter=':- .gitignore' \
  --exclude='.git/' \
  /home/nikita/Skoltech/ICRA-HRI/Teleop-Inference-WBC/ \
  unitree@192.168.50.132:/home/unitree/teleop-ws/Teleop-Inference-WBC/