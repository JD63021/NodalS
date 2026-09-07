#!/usr/bin/env bash
set -euo pipefail
OUT="${1:?output file required}"
INTERVAL="${GPU_MEM_POLL_INTERVAL:-0.20}"

base="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')"
echo "baselineMiB=$base" > "$OUT"

while true; do
  x="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || true)"
  if [[ "$x" =~ ^[0-9]+$ ]]; then
    echo "sampleMiB=$x" >> "$OUT.samples"
  fi
  sleep "$INTERVAL"
done
