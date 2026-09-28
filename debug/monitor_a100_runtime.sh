#!/usr/bin/env bash
# Snapshot GPU, PolicyServer CPU and TCP queues while profiling. Run on A100 host.
set -euo pipefail

out_dir=${1:-/tmp/gr00t_runtime_$(date +%Y%m%d_%H%M%S)}
duration_seconds=${2:-60}
interval_seconds=${3:-0.5}
mkdir -p "$out_dir"

metrics="$out_dir/metrics.csv"
tcp="$out_dir/tcp.txt"
printf 'timestamp,pid,cpu_pct,rss_kb,gpu_util_pct,gpu_mem_util_pct,gpu_mem_used_mib\n' > "$metrics"

end=$(( $(date +%s) + duration_seconds ))
echo "Writing runtime samples to $out_dir for ${duration_seconds}s"
while (( $(date +%s) < end )); do
  pid=$(pgrep -f 'run_gr00t_server.py' | tail -n 1 || true)
  process_csv='NA,NA,NA'
  if [[ -n "$pid" ]]; then
    process_csv=$(ps -p "$pid" -o '%cpu=,rss=' | awk '{print $1 "," $2}')
  fi
  gpu_csv=$(nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used \
    --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
  printf '%s,%s,%s,%s\n' "$(date +%s.%N)" "${pid:-NA}" "$process_csv" "$gpu_csv" >> "$metrics"
  {
    printf '\n--- %s ---\n' "$(date -Is)"
    ss -tnpi '( sport = :5555 or dport = :5555 )' || true
  } >> "$tcp"
  sleep "$interval_seconds"
done

echo "Done. Metrics: $metrics"
