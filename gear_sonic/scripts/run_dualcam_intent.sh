#!/usr/bin/env bash
# Start exactly one human-intent predictor as the VLA intent publisher.
set -euo pipefail

VERSION="${1:-}"
if [[ "$VERSION" != "v1" && "$VERSION" != "v2" && "$VERSION" != "video" && "$VERSION" != "hri-video" ]]; then
  echo "usage: $0 {v1|v2|video} [predictor robot options]" >&2
  echo "example: $0 video --host 192.168.50.132 --device cuda --print" >&2
  exit 2
fi
shift

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TELEOP_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
DEFAULT_WORKSPACE_ROOT="$(dirname -- "$TELEOP_ROOT")"
WORKSPACE_ROOT="${ICRA_HRI_ROOT:-$DEFAULT_WORKSPACE_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python}"
INTENT_BIND="${INTENT_BIND:-tcp://*:5562}"

case "$VERSION" in
  v1)
    PREDICTOR_ROOT="$WORKSPACE_ROOT/dual_camera_robot"
    PREDICTOR_SCRIPT="$PREDICTOR_ROOT/run_dualcam.py"
    ;;
  v2)
    PREDICTOR_ROOT="$WORKSPACE_ROOT/dual_camera_robot_v2"
    PREDICTOR_SCRIPT="$PREDICTOR_ROOT/run_dualcam.py"
    ;;
  video|hri-video)
    VERSION="video"
    PREDICTOR_ROOT="$WORKSPACE_ROOT/hri_video_robot"
    PREDICTOR_SCRIPT="$PREDICTOR_ROOT/run_video.py"
    PYTHON_BIN="${HRI_VIDEO_PYTHON_BIN:-$PYTHON_BIN}"
    ;;
esac

if [[ ! -f "$PREDICTOR_SCRIPT" ]]; then
  echo "predictor $VERSION not found: $PREDICTOR_SCRIPT" >&2
  echo "set ICRA_HRI_ROOT to the directory containing the predictor repositories" >&2
  exit 1
fi

echo "Starting human-intent predictor $VERSION from $PREDICTOR_ROOT" >&2
echo "Publishing human intent on $INTENT_BIND" >&2
if [[ "$VERSION" == "video" ]]; then
  exec env -u PYTHONPATH "$PYTHON_BIN" -B -u "$PREDICTOR_SCRIPT" robot \
    --intent-bind "$INTENT_BIND" \
    "$@"
else
  exec "$PYTHON_BIN" -B "$PREDICTOR_SCRIPT" robot \
    --intent-bind "$INTENT_BIND" \
    "$@"
fi
