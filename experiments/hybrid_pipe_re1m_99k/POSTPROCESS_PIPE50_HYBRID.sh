#!/usr/bin/env bash
# Standalone postprocessing. No solver invocation, CUDA or build.
# Usage:
#   bash POSTPROCESS_PIPE50_HYBRID.sh /path/to/run          # latest checkpoint
#   bash POSTPROCESS_PIPE50_HYBRID.sh /path/to/run 1000     # exactly it1000
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN="${1:-$HOME/NodalS_SST_LOCAL_RUNS/PIPE50_RE1M_HYBRID_99K_ONEPOINT}"
IT="${2:-latest}"

if [[ ! -d "$RUN" ]]; then
    echo "ERROR: missing run folder: $RUN" >&2
    exit 2
fi

if [[ "$IT" == latest ]]; then
    VTU="$(find "$RUN" -maxdepth 1 -type f -name 'NODALS_SST_G2_CHECKPOINT_it*_flow.vtu' | sort -V | tail -n 1)"
    if [[ -z "$VTU" ]]; then
        echo "ERROR: no checkpoint *_flow.vtu found in $RUN" >&2
        exit 2
    fi
    bn="$(basename "$VTU")"
    IT="${bn#NODALS_SST_G2_CHECKPOINT_it}"
    IT="${IT%_flow.vtu}"
else
    if ! [[ "$IT" =~ ^[0-9]+$ ]]; then
        echo "Iteration must be a nonnegative integer or latest" >&2
        exit 2
    fi
    VTU="$RUN/NODALS_SST_G2_CHECKPOINT_it${IT}_flow.vtu"
fi

if [[ ! -s "$VTU" ]]; then
    echo "ERROR: checkpoint file not found: $VTU" >&2
    find "$RUN" -maxdepth 1 -type f -name '*_flow.vtu' -print | sort -V >&2
    exit 2
fi
OUT="$RUN/POSTPROCESS_it${IT}/MOODY"
echo "POSTPROCESS_VTU=$VTU"
echo "POSTPROCESS_OUT=$OUT"
python3 -u "$HERE/POSTPROCESS_PIPE50_HYBRID.py" "$VTU" \
    --diameter 0.889 \
    --bulk 16.431211404701912 \
    --re 1000000 \
    --output "$OUT"

echo "POSTPROCESS_50D_STATUS=PASS"
echo "CSV=$OUT/local_pressure_friction.csv"
echo "JSON=$OUT/result.json"
echo "PLOT=$OUT/pressure_20to50D.png"
