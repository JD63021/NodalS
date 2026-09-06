#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
GPU="$ROOT"
OUT="$ROOT/results_111k_convergence"
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
command -v nvcc >/dev/null || { echo "H8_FP64_111K_RUNNER status=FAIL reason=nvcc_missing"; exit 40; }

CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')"
ARCH="sm_${CC/.}"
echo "H8_FP64_CUDA computeCapability=$CC arch=$ARCH"

"$ROOT/AUDIT_H8_FP64_SOURCE.sh"

make -C "$GPU" clean >/dev/null
make -C "$GPU" ARCH="$ARCH" nodals_gpu_h8_fp64 2>&1 | tee "$OUT/build.log"
"$GPU/audit_no_petsc.sh" "$GPU/nodals_gpu_h8_fp64"

MESH="$HOME/Desktop/meshes/pipe/111k/constant/polyMesh"
[[ -f "$MESH/owner" ]] || {
  echo "H8_FP64_111K_RUNNER status=FAIL reason=mesh_missing path=$MESH"
  exit 41
}

LOG="$OUT/111k_fp64_converge.log"
TIMEF="$OUT/111k_fp64_converge.time"
GPUF="$OUT/111k_fp64_converge.gpu_mem"

"$ROOT/MONITOR_GPU_MEMORY.sh" "$GPUF" &
monpid=$!
sleep 0.25

set +e
/usr/bin/time -v -o "$TIMEF" \
  "$GPU/nodals_gpu_h8_fp64" \
    --mesh "$MESH" \
    --tag 111k_fp64_converge \
    --run-mode converge \
    --momentum-work fgs1 \
    --alpha-u 0.50 \
    --alpha-p 0.50 \
    --mom-omega 1.0 \
    --p-rtol 0.5 \
    --p-atol 1e-12 \
    --p-max 20 \
    --simple-tol 1e-3 \
    --max-outer 2500 \
    --snapshot-tol 5e-10 \
  2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e

kill "$monpid" 2>/dev/null || true
wait "$monpid" 2>/dev/null || true

base="$(awk -F= '/^baselineMiB=/{print $2}' "$GPUF" | tail -1)"
peak="$base"
if [[ -s "$GPUF.samples" ]]; then
  peak="$(awk -F= '/^sampleMiB=/{if($2>m)m=$2}END{print m+0}' "$GPUF.samples")"
fi
{
  echo "baselineMiB=${base:-0}"
  echo "peakMiB=${peak:-0}"
  echo "peakDeltaMiB=$(( ${peak:-0} - ${base:-0} ))"
} > "$GPUF.final"
mv "$GPUF.final" "$GPUF"
rm -f "$GPUF.samples"

if (( rc != 0 )); then
  echo "H8_FP64_111K_RUNNER status=FAIL rc=$rc log=$LOG"
  exit "$rc"
fi

grep -q 'NODALS_GPU_H8_CONFIG tag=111k_fp64_converge precision=fp64 .*runMode=converge' "$LOG" || {
  echo "H8_FP64_111K_RUNNER status=FAIL reason=fp64_config_gate_missing"; exit 42;
}
grep -q 'NODALS_GPU_H8_FULLCONV tag=111k_fp64_converge .*converged=1 .*status=PASS' "$LOG" || {
  echo "H8_FP64_111K_RUNNER status=FAIL reason=fullconv_gate_missing"; exit 43;
}
grep -q 'NODALS_GPU_RESULT gate=H8 tag=111k_fp64_converge .*precision=fp64 runMode=converge .*status=PASS' "$LOG" || {
  echo "H8_FP64_111K_RUNNER status=FAIL reason=result_gate_missing"; exit 44;
}

python3 "$ROOT/ANALYZE_FP64_111K.py" "$LOG" "$GPUF" "$TIMEF" \
  | tee "$OUT/FP64_111K_CONVERGENCE_SUMMARY.txt"

echo "H8_FP64_111K_RUNNER status=PASS convergence=PASS"
echo "SUMMARY=$OUT/FP64_111K_CONVERGENCE_SUMMARY.txt"
