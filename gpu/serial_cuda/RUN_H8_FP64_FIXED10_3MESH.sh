#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
GPU="$ROOT"
OUT="$ROOT/results_fixed10"
mkdir -p "$OUT"

if ! command -v nvcc >/dev/null 2>&1; then
  for C in /usr/local/cuda-13.0 /usr/local/cuda /usr/local/cuda-12.2; do
    if [[ -x "$C/bin/nvcc" ]]; then
      export CUDA_HOME="$C"
      export PATH="$C/bin:$PATH"
      export LD_LIBRARY_PATH="$C/lib64:${LD_LIBRARY_PATH:-}"
      break
    fi
  done
fi
command -v nvcc >/dev/null || { echo "H8_FP64_FIXED10_RUNNER status=FAIL reason=nvcc_missing"; exit 40; }

CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')"
ARCH="sm_${CC/.}"
echo "H8_FP64_CUDA computeCapability=$CC arch=$ARCH"

"$ROOT/AUDIT_H8_FP64_SOURCE.sh"

# Reuse the binary from the 111k gate if present; otherwise build.
if [[ ! -x "$GPU/nodals_gpu_h8_fp64" ]]; then
  make -C "$GPU" ARCH="$ARCH" nodals_gpu_h8_fp64 2>&1 | tee "$OUT/build.log"
fi
"$GPU/audit_no_petsc.sh" "$GPU/nodals_gpu_h8_fp64"

PIPE="$HOME/Desktop/meshes/pipe"
declare -A MESH
MESH[768k]="$PIPE/768k/constant/polyMesh"
MESH[1.1M]="$PIPE/1.1M/constant/polyMesh"
MESH[2M]="$PIPE/2M/constant/polyMesh"

finish_monitor () {
  local pid="$1" f="$2"
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  local base peak
  base="$(awk -F= '/^baselineMiB=/{print $2}' "$f" | tail -1)"
  peak="$base"
  if [[ -s "$f.samples" ]]; then
    peak="$(awk -F= '/^sampleMiB=/{if($2>m)m=$2}END{print m+0}' "$f.samples")"
  fi
  {
    echo "baselineMiB=${base:-0}"
    echo "peakMiB=${peak:-0}"
    echo "peakDeltaMiB=$(( ${peak:-0} - ${base:-0} ))"
  } > "$f.final"
  mv "$f.final" "$f"
  rm -f "$f.samples"
}

for tag in 768k 1.1M 2M; do
  mesh="${MESH[$tag]}"
  [[ -f "$mesh/owner" ]] || {
    echo "H8_FP64_FIXED10_CASE status=FAIL tag=$tag reason=mesh_missing path=$mesh"
    exit 41
  }

  log="$OUT/$tag.log"
  timef="$OUT/$tag.time"
  gpuf="$OUT/$tag.gpu_mem"

  echo
  echo "============================================================"
  echo "H8_FP64_FIXED10_CASE_START tag=$tag mesh=$mesh exactSimpleIts=10"
  echo "============================================================"

  "$ROOT/MONITOR_GPU_MEMORY.sh" "$gpuf" &
  monpid=$!
  sleep 0.25

  set +e
  /usr/bin/time -v -o "$timef" \
    "$GPU/nodals_gpu_h8_fp64" \
      --mesh "$mesh" \
      --tag "$tag" \
      --run-mode fixed10 \
      --momentum-work fgs1 \
      --alpha-u 0.50 \
      --alpha-p 0.50 \
      --mom-omega 1.0 \
      --p-rtol 0.5 \
      --p-atol 1e-12 \
      --p-max 20 \
      --simple-tol 1e-6 \
      --max-outer 10 \
      --snapshot-tol 5e-10 \
    2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}
  set -e

  finish_monitor "$monpid" "$gpuf"

  if (( rc != 0 )); then
    echo "H8_FP64_FIXED10_CASE status=FAIL tag=$tag rc=$rc log=$log"
    exit "$rc"
  fi

  grep -q "NODALS_GPU_H8_CONFIG tag=$tag precision=fp64 .*runMode=fixed10" "$log" || {
    echo "H8_FP64_FIXED10_CASE status=FAIL tag=$tag reason=fp64_config_gate_missing"; exit 42;
  }
  grep -q "NODALS_GPU_RESULT gate=H8 tag=$tag .*precision=fp64 runMode=fixed10 .*outer=10 .*status=PASS" "$log" || {
    echo "H8_FP64_FIXED10_CASE status=FAIL tag=$tag reason=result_gate_missing"; exit 43;
  }

  echo "H8_FP64_FIXED10_CASE status=PASS tag=$tag"
done

python3 "$ROOT/ANALYZE_H8_FP64.py" "$OUT" | tee "$OUT/H8_FP64_FIXED10_SUMMARY.txt"

echo "H8_FP64_FIXED10_RUNNER status=PASS cases='768k 1.1M 2M' exactSimpleIts=10"
echo "SUMMARY=$OUT/H8_FP64_FIXED10_SUMMARY.txt"
