#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
GPU="$ROOT/gpu_h4"
OUT="$ROOT/h4_results"
mkdir -p "$OUT"
exec > >(tee "$OUT/h4_master.log") 2>&1

if ! command -v nvcc >/dev/null 2>&1; then
  for C in /usr/local/cuda-13.0 /usr/local/cuda /usr/local/cuda-12.2; do
    if [[ -x "$C/bin/nvcc" ]]; then
      export CUDA_HOME="$C"; export PATH="$C/bin:$PATH"; export LD_LIBRARY_PATH="$C/lib64:${LD_LIBRARY_PATH:-}"
      break
    fi
  done
fi
command -v nvcc >/dev/null || { echo "H4_RUNNER_RESULT status=FAIL reason=nvcc_missing"; exit 40; }

CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')"
ARCH="sm_${CC/.}"
echo "H4_CUDA_ARCH computeCapability=$CC nvccArch=$ARCH"
make -C "$GPU" clean >/dev/null
make -C "$GPU" ARCH="$ARCH" 2>&1 | tee "$OUT/build.log"
"$GPU/audit_no_petsc.sh" "$GPU/nodals_gpu_h4_fp32"

PIPE_ROOT="${PIPE_ROOT:-$HOME/Desktop/meshes/pipe}"
declare -A MPATH=(
  [768k]="$PIPE_ROOT/768k/constant/polyMesh"
  [1.1M]="$PIPE_ROOT/1.1M/constant/polyMesh"
)

ARGS=(
  --run-mode fixed10
  --alpha-u 0.50 --alpha-p 0.50
  --mom-omega 1.0
  --p-rtol 0.5 --p-atol 1e-12 --p-max 20
  --simple-tol 1e-6 --max-outer 10 --snapshot-tol 5e-6
)

run_one () {
  local meshTag="$1"
  local mode="$2"
  local mesh="${MPATH[$meshTag]}"
  local tag="${meshTag}_${mode}"

  [[ -f "$mesh/owner" ]] || {
    echo "H4_RUNNER_RESULT status=FAIL reason=mesh_missing tag=$tag path=$mesh"; exit 41;
  }

  echo
  echo "=== H4 $tag : persistent diffusion + convection-only, EXACTLY 10 SIMPLE ==="
  /usr/bin/time -v -o "$OUT/$tag.time" \
    "$GPU/nodals_gpu_h4_fp32" \
      --mesh "$mesh" --tag "$tag" --momentum-work "$mode" \
      "${ARGS[@]}" \
    2>&1 | tee "$OUT/$tag.log"

  grep -q "NODALS_GPU_RESULT gate=H4 tag=$tag.*outer=10.*status=PASS" "$OUT/$tag.log" || {
    echo "H4_RUNNER_RESULT status=FAIL reason=case_failed tag=$tag"; exit 42;
  }
}

run_one 768k fgs1
run_one 768k altgs1
run_one 1.1M fgs1
run_one 1.1M altgs1

python3 "$ROOT/ANALYZE_H4.py" "$OUT" | tee "$OUT/analysis_console.txt"

echo "H4_RUNNER_RESULT status=PASS runs=4 exactOuter=10 meshes=768k_1p1M modes=fgs1_altgs1 noLongCases=1 out=$OUT"
echo "RETURN_SUMMARY=$OUT/h4_summary.txt"
