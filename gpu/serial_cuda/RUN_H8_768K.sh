#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
GPU="$ROOT"
OUT="$ROOT/h8_results"
mkdir -p "$OUT"
exec > >(tee "$OUT/h8_master.log") 2>&1

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
command -v nvcc >/dev/null || {
  echo "H8_RUNNER_RESULT status=FAIL reason=nvcc_missing"
  exit 40
}

CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')"
ARCH="sm_${CC/.}"
echo "H8_CUDA_ARCH computeCapability=$CC nvccArch=$ARCH"

make -C "$GPU" clean >/dev/null
make -C "$GPU" ARCH="$ARCH" 2>&1 | tee "$OUT/build.log"
"$GPU/audit_no_petsc.sh" "$GPU/nodals_gpu_h8_fp32"

MESH="${PIPE_ROOT:-$HOME/Desktop/meshes/pipe}/768k/constant/polyMesh"
[[ -f "$MESH/owner" ]] || {
  echo "H8_RUNNER_RESULT status=FAIL reason=mesh_missing path=$MESH"
  exit 41
}

TAG="768k_fgs1_h8_precomputed_b"

echo
echo "=== H8 768k / FGS1 / exactly 10 SIMPLE / H6 + H7 retained ==="

/usr/bin/time -v -o "$OUT/$TAG.time" \
  "$GPU/nodals_gpu_h8_fp32" \
    --mesh "$MESH" \
    --tag "$TAG" \
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
    --snapshot-tol 5e-6 \
  2>&1 | tee "$OUT/$TAG.log"

grep -q "NODALS_GPU_RESULT gate=H8 tag=$TAG.*outer=10.*status=PASS" "$OUT/$TAG.log" || {
  echo "H8_RUNNER_RESULT status=FAIL reason=case_failed"
  exit 42
}

python3 "$ROOT/ANALYZE_H8.py" "$OUT/$TAG.log" | tee "$OUT/h8_summary.txt"

echo "H8_RUNNER_RESULT status=PASS runs=1 mesh=768k momentum=fgs1 exactOuter=10 H6CoarseSpMV=KEPT H7Assembly=KEPT out=$OUT"
echo "RETURN_SUMMARY=$OUT/h8_summary.txt"
