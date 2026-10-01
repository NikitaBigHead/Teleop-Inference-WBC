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
cd /home/unitree/teleop-ws/Teleop-Inference-WBC
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
cd /home/unitree/teleop-ws/Teleop-Inference-WBC/gear_sonic/scripts 
chmod 777 ./start_brainco_and_check.sh 
./start_brainco_and_check.sh   --interface wlxfc23cd997021
```

# 3. WBC — ждать «Init done»



## Impedance profiles for deploy

### Hugging params
```bash
export TensorRT_ROOT="$HOME/TensorRT"
cd ~/GR00T-WholeBodyControl/gear_sonic_deploy && source scripts/setup_env.sh
./deploy.sh \
  --cp policy/sonic_v1_1/model \
  --obs-config policy/sonic_v1_1/observation_config.yaml \
  --input-type zmq_manager \
  --motor-kp-scale 15-28=0.6 \
  --motor-kd-scale 15-28=0.83 \
  --safe-stop-profile SOFT \
  --safe-stop-soften 0.5 \
  --safe-stop-hand-speed 0.2 \
  --safe-stop-status-port 5571 \
  --safe-stop-voice-port 5570 \
  real
```
```

### Рекомендуемый запуск: динамическая жёсткость по intent

Вместо перезапуска deploy для каждого действия используется runtime-слой `arm_compliance`. При таком запуске **не добавлять** статические `--motor-kp-scale` / `--motor-kd-scale`: runtime-профиль применяется поверх базовых gains, и статические коэффициенты иначе перемножатся с динамическими.

```bash
export TensorRT_ROOT="$HOME/TensorRT"
cd ~/GR00T-WholeBodyControl/gear_sonic_deploy && source scripts/setup_env.sh
./deploy.sh \
  --input-type zmq_manager \
  --arm-compliance \
  --compliance-host localhost \
  --compliance-port 5565 \
  --compliance-profile HUG \
  --compliance-soften 0.3 \
  --compliance-stiffen 1.0 \
  --safe-stop-profile SOFT \
  --safe-stop-soften 0.5 \
  --safe-stop-hand-speed 0.2 \
  --safe-stop-status-port 5571 \
  --safe-stop-voice-port 5570 \
  real
```

Bridge выбирает встроенный профиль сразу после принятия нового prompt. Старый action chunk в этот момент уже сброшен, поэтому minimum-jerk переход gains начинается во время расчёта нового VLA chunk, до исполнения нового действия:

| Intent / prompt | Runtime-профиль | Эффективные коэффициенты рук |
|---|---|---|
| `hug` | `HUG` | моторы 15–28: Kp × 0.6, Kd × 0.83 |
| `no_interaction` / `none` | `HUG` | моторы 15–28: Kp × 0.6, Kd × 0.83 |
| `handshake` | `HANDSHAKE` | моторы 18 и 25: Kp × 0.6, Kd × 0.83; остальные × 1 |
| `fist_bump` | `FISTBUMP_SOFTWRIST` | моторы 26–28: Kp × 0.5, Kd × 0.7; остальные × 1 |
| другой ручной prompt | `RIGID` | моторы 15–28: Kp × 1, Kd × 1 |

Переход выполняется minimum-jerk: смягчение за 0.3 с, возврат к более жёсткому профилю за 1.0 с. Во время safe stop применяется более мягкое из значений текущего runtime-профиля и `SOFT`.

# 4. паблишер клавиш

```bash
source ~/GR00T-WholeBodyControl/.venv_data_collection/bin/activate
python3 ~/keypress.py
```
# 5. мост — команды в §3

### База (VLA одна)

```bash
cd ~/GR00T-WholeBodyControl &&
source .venv_data_collection/bin/activate &&
PYTHONPATH="/home/unitree/unitree_sdk2_python:${PYTHONPATH:-}" \
SLEW_ENABLE=1 \
HAND_ENABLE=1 \
python /home/unitree/teleop-ws/Teleop-Inference-WBC/gear_sonic/scripts/run_inference_affective_vla.py \
  --host 100.64.0.21 \
  --port 5555 \
  --camera-host localhost \
  --camera-port 5555 \
  --state-zmq-host localhost \
  --state-zmq-port 5557 \
  --action-zmq-host localhost \
  --action-zmq-port 5556 \
  --embodiment-tag NEW_EMBODIMENT \
  --prompt "none" \
  2>&1 | tee ~/bridge_base_$(date +%Y%m%d_%H%M).log
```

**Клавиши:** `k` старт → `i` начальная поза (обязательно) → `p` снять паузу (тумблер) → `g` заморозка тела.

### Автоматический prompt от dual-camera predictor

На компьютере с окружением `action` и доступом к камерному серверу робота:

```bash
cd /home/nikita/Skoltech/ICRA-HRI/dual_camera_robot
conda activate action
python -B run_dualcam.py robot \
  --host 192.168.50.132 \
  --port 5555 \
  --device cuda \
  --intent-bind 'tcp://*:5562' \
  --hold-hug-on-unknown \
  --hug-unknown-hold-seconds 3 \
  --print
```

Два `hug`-флага опциональны. С ними predictor продолжает публиковать последний подтверждённый `hug` вместо кратковременного `unknown`, но не дольше указанного времени. Любой другой известный класс отменяет удержание.

На роботе запускайте мост с дополнительными параметрами:



```bash
cd ~/GR00T-WholeBodyControl &&
source .venv_data_collection/bin/activate &&
PYTHONPATH="/home/unitree/unitree_sdk2_python:${PYTHONPATH:-}"

cd /home/unitree/teleop-ws/Teleop-Inference-WBC

SAFE_STOP_HOST=localhost \
SAFE_STOP_PORT=5571 \
SAFE_STOP_HAND_OPEN_RATE=0.5 \
SAFE_STOP_HAND_RESUME_RATE=0.5 \
SLEW_ENABLE=1 \
HAND_ENABLE=1 \
python gear_sonic/scripts/run_inference_pose_predictor_affective_vla.py \
  --host 100.64.0.21 \
  --port 5555 \
  --camera-host localhost \
  --camera-port 5555 \
  --state-zmq-host localhost \
  --state-zmq-port 5557 \
  --action-zmq-host localhost \
  --action-zmq-port 5556 \
  --embodiment-tag NEW_EMBODIMENT \
  --prompt none \
  --intent-mode auto \
  --intent-host 192.168.50.42 \
  --intent-port 5562 \
  --intent-compliance \
  --compliance-port 5565 \
  --compliance-rate 10
```

В auto-режиме `hug`, `handshake`, `fist_bump` и `no_interaction` переключают prompt (`no_interaction` → `none`). `unknown`, пропавший intent-поток и смена prompt включают safe hold; исполнение возобновляется только после получения VLA chunk для нового `intent_epoch`. Для ручного prompt отправьте `pr hug`/`pr handshake`; для возврата к predictor — `pr auto`.

`--intent-compliance` поднимает PUB на `tcp://*:5565`, немедленно отправляет профиль при подготовке нового prompt и повторяет его с частотой `--compliance-rate`. Исполнение остаётся в safe hold до свежего VLA chunk, поэтому профиль успевает начать плавный переход раньше движения. Не запускайте одновременно `arm_compliance_cli.py`: он тоже пытается занять порт 5565. В логе bridge ожидается `Arm compliance -> HANDSHAKE/HUG/FISTBUMP_SOFTWRIST`, в deploy — `[ArmCompliance] -> ...`.

## Голосовой safe stop со встроенного микрофона G1

Safe stop работает через отдельный процесс `voice_safe_stop.py`: он получает PCM-поток встроенного микрофона G1, распознаёт только короткий список команд и публикует команду на порт `5570`. Deploy должен быть запущен с `--safe-stop-voice-port 5570`, как в команде Hugging params выше.



Рабочий запуск:

```bash
cd /home/unitree/teleop-ws/Teleop-Inference-WBC
python gear_sonic/scripts/voice_safe_stop.py \
  --g1-mic \
  --model ~/vosk-model-small-en-us-0.15 \
  --port 5570 \
  --verbose
```

Команды по умолчанию:

| Голос | Действие |
|---|---|
| `stop` | latched safe stop |
| `hold on` | latched safe stop |
| release голосом | выключен; освобождение только через `u` в терминале deploy |

При рабочем запуске voice-процесс печатает `publishing on tcp://*:5570`, при распознавании — `STOP sent`, а deploy — `[SafeStop] STOP (voice)`. Если написано `dry run, nothing sent`, команда роботу не ушла.

### Клавиши и терминалы

Важно: клавиша `k` имеет разный смысл в терминале deploy и в паблишере клавиш VLA.

| Где вводить | Команда | Действие |
|---|---|---|
| Терминал **deploy** | `k` | включить latched safe stop: VLA игнорируется, робот переходит в planner idle, руки смягчаются |
| Терминал **deploy** | `u` | снять latch safe stop; робот остаётся в idle и сам движение не возобновляет |
| Терминал **deploy** | `O` | аварийная остановка всего робота; это не soft safe stop |
| Паблишер клавиш **VLA** | `k` | запустить или остановить C++ control loop |
| Паблишер клавиш **VLA** | `i` | отправить initial pose и перейти из PLANNER в POSE |
| Паблишер клавиш **VLA** | `p` | pause/resume VLA policy loop |
| Паблишер клавиш **VLA** | `g` | заморозить/возобновить корпус; пальцы продолжают работать |
| Паблишер клавиш **VLA** | `[` / `]` | переключить открытую/закрытую начальную позу левой/правой кисти |
| Паблишер клавиш **VLA** | `pr hug`, `pr handshake`, `pr fist_bump`, `pr none` | ручной prompt |
| Паблишер клавиш **VLA** | `pr auto` | вернуть автоматический prompt от predictor |

### Как вернуть робота после `stop` / `hold on`

Safe stop специально защёлкивается: восстановление predictor или новые VLA actions не запускают робота автоматически.

1. Убедиться, что рядом с роботом безопасно и человек больше не находится в контакте с руками.
2. В терминале **deploy** нажать `u`. Дождаться `[SafeStop] RELEASED` и сообщения о возврате жёсткости. Робот останется в PLANNER idle — это нормально.
3. Если до stop стандартная последовательность `k → i → p` уже была выполнена и bridge считает C++ loop запущенным, в паблишере клавиш VLA нажать `k` один раз для остановки и ещё раз для чистого запуска в PLANNER. Это синхронизирует внутреннее состояние bridge с deploy после safe stop.
4. Нажать `i`: отправится initial pose, затем режим переключится в POSE и будет запрошен свежий VLA chunk.
5. Нажать `p` только если policy loop находится в состоянии `Paused`. Если до safe stop он работал, повторное `p` не требуется.

Короткая надёжная последовательность после voice stop для ранее работавшего VLA:

```text
deploy:       u
VLA keyboard: k → k → i
VLA keyboard: p только если в логе написано Paused
```

Если C++ loop ещё до stop был выключен, достаточно `u → k → i → p`: первый `k` сразу запускает его в PLANNER.

Переменные `SAFE_STOP_HOST`, `SAFE_STOP_PORT` и скорости раскрытия относятся только к отдельному `SafeHandGuard` для BrainCo. Основное тело и плечи останавливает C++ deploy независимо от них. В локальной версии `run_inference_pose_predictor_affective_vla.py` guard должен быть явно подключён к `_send_brainco_hands`; одни переменные окружения не включают раскрытие пальцев.





# Copying the project to Unitree over SSH

```bash
rsync -avh --partial --info=progress2 \
  --filter=':- .gitignore' \
  --exclude='.git/' \
  /home/nikita/Skoltech/ICRA-HRI/Teleop-Inference-WBC/ \
  unitree@192.168.50.132:/home/unitree/teleop-ws/Teleop-Inference-WBC/
