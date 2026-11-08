#!/usr/bin/env bash
# GitHub PR helper. The complete, audited importer is also provided in
# FREEZE_NODALS_SST_V2_20261108.sh in the accompanying workstation handoff.
# This helper performs the same minimal source-only freeze.
set -euo pipefail
REPO="${NODALS_GIT_REPO:-$HOME/NodalS}"
SRC="${NODALS_V2_SOURCE:-$HOME/Downloads/NodalS_SST_SOFTWARE_V2}"
WT="${NODALS_FREEZE_WORKTREE:-$HOME/NodalS_SST_V2_PR_20261108}"
BRANCH="freeze/sst-v2-unified-supg-centroid-20261108"
[[ -s "$SRC/nodals.py" && -f "$SRC/source/puretet_profile5_eq5_hxt5b/Makefile" ]] || { echo "Active V2 source missing" >&2; exit 2; }
[[ -d "$REPO" && "$(git -C "$REPO" rev-parse --show-toplevel)" == "$REPO" ]] || { echo "NodalS Git repo missing" >&2; exit 2; }
[[ "$(git -C "$REPO" remote get-url origin)" == *"JD63021/NodalS"* ]] || { echo "Unexpected remote" >&2; exit 2; }
grep -RFlq 'NODALS_UNIFIED_FULL_MOMENTUM_SUPG_V1' "$SRC/source" || { echo "Missing unified SUPG" >&2; exit 2; }
grep -RFlq 'NODALS_AUTO_THIN_CENTROID_V13' "$SRC/source" || { echo "Missing thin-channel fallback" >&2; exit 2; }
git -C "$REPO" fetch origin "refs/heads/$BRANCH:refs/remotes/origin/$BRANCH"
if [[ ! -e "$WT" ]]; then
  if git -C "$REPO" show-ref --verify --quiet "refs/heads/$BRANCH"; then
    git -C "$REPO" worktree add "$WT" "$BRANCH"
  else
    git -C "$REPO" worktree add -b "$BRANCH" "$WT" "origin/$BRANCH"
  fi
fi
[[ "$(git -C "$WT" branch --show-current)" == "$BRANCH" ]] || { echo "Wrong worktree branch" >&2; exit 2; }
[[ -z "$(git -C "$WT" status --porcelain)" ]] || { echo "Review worktree dirty; inspect before retrying" >&2; exit 2; }
git -C "$WT" pull --ff-only origin "$BRANCH"
DST="$WT/turbulence/sst_gpu_v2"
mkdir -p "$DST"
rsync -a --delete --delete-excluded \
 --exclude='build/' --exclude='.cache/' --exclude='runs/' --exclude='__pycache__/' \
 --exclude='*.o' --exclude='*.so' --exclude='*.a' --exclude='*.pyc' \
 --exclude='*.log' --exclude='*.vtu' --exclude='*.bin' --exclude='*.ckpt' \
 --exclude='*.msh' --exclude='*.tar*' --exclude='*.zip' \
 --exclude='*.BEFORE*' --exclude='*.before*' --exclude='*.bak' \
 --exclude='*_BACKUP*' --exclude='nodals_hxt5b_fp64' \
 "$SRC/source/" "$DST/source/"
cp -p "$SRC/nodals.py" "$DST/nodals.py"
for f in Makefile CMakeLists.txt build.sh; do
 [[ ! -f "$SRC/$f" ]] || cp -p "$SRC/$f" "$DST/$f"
done
for dir in tools runner scripts tests examples; do
 [[ ! -d "$SRC/$dir" ]] || rsync -a --exclude='*.msh' --exclude='build/' \
 --exclude='__pycache__/' --exclude='*.vtu' --exclude='*.ckpt' \
 "$SRC/$dir/" "$DST/$dir/"
done
python3 -m py_compile "$DST/nodals.py"
git -C "$WT" add -A -- turbulence/sst_gpu_v2
git -C "$WT" diff --cached --stat
if ! git -C "$WT" diff --cached --quiet; then
 git -C "$WT" commit -m 'Freeze working SST V2 unified SUPG and centroid wall fallback'
fi
git -C "$WT" push -u origin "$BRANCH"
echo "Review PR at: https://github.com/JD63021/NodalS/pulls"
