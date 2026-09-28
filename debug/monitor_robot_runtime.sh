#!/usr/bin/env bash
# Snapshot robot-side bridge/camera resources and the two relevant TCP paths.
# Run on the robot. Arguments: output-dir [duration-seconds] [interval-seconds]
set -euo pipefail

out_dir=${1:-/tmp/vla_robot_runtime_$(date +%Y%m%d_%H%M%S)}
duration_seconds=${2:-60}
interval_seconds=${3:-0.5}
mkdir -p "$out_dir"

metrics="$out_dir/process_metrics.csv"
network="$out_dir/network.txt"
printf 'timestamp,bridge_pid,bridge_cpu_pct,bridge_rss_kb,camera_pid,camera_cpu_pct,camera_rss_kb\n' > "$metrics"
end=$(( $(date +%s) + duration_seconds ))

process_metrics() {
  local pattern=$1
  local pid cpu rss
  pid=$(pgrep -f "$pattern" | tail -n 1 || true)
  if [[ -z "$pid" ]]; then
    printf 'NA,NA,NA'
    return
  fi
  read -r cpu rss < <(ps -p "$pid" -o '%cpu=,rss=')
  printf '%s,%s,%s' "$pid" "${cpu:-NA}" "${rss:-NA}"
}

echo "Writing robot runtime samples to $out_dir for ${duration_seconds}s"
while (( $(date +%s) < end )); do
  bridge=$(process_metrics 'run_inference_affective_vla.py|profile_affective_bridge.py')
  camera=$(process_metrics 'gear_sonic.camera.composed_camera')
  printf '%s,%s,%s\n' "$(date +%s.%N)" "$bridge" "$camera" >> "$metrics"
  {
    printf '\n--- %s ---\n' "$(date -Is)"
    printf '[remote PolicyServer]\n'
    ss -tnpi '( dport = :5555 or sport = :5555 )' || true
    printf '[tailscale]\n'
    ip -s link show tailscale0 2>/dev/null || true
    printf '[local camera ZMQ]\n'
    ss -tnpi '( dport = :5555 or sport = :5555 )' | grep -v '100.64.0.' || true
  } >> "$network"
  sleep "$interval_seconds"
done

echo "Done. Process metrics: $metrics; socket snapshots: $network"
