#!/usr/bin/env bash
# Run repeated recovery-matrix inversion + identification eval for error bars.
#
# For each rep (a different --seed) and each of the 4 opt_spaces used in the figure,
# it runs recovery_matrix_invert_reps.py then recovery_check_eval.py. Results land in
#   <OUT_ROOT>/rep<NN>/<opt_space>/{source,recovered,recovery_check_identification.pkl}
# which Fig4_recon_and_identification_errorbar.py aggregates.
#
# Usage:
#   bash scripts/experiments/run_recovery_reps.sh              # all reps 0..N_REPS-1
#   bash scripts/experiments/run_recovery_reps.sh 0 1 2 3 4    # only these rep indices
#   CUDA_VISIBLE_DEVICES=0 bash scripts/experiments/run_recovery_reps.sh 0 1 2 3 4 &
#   CUDA_VISIBLE_DEVICES=1 bash scripts/experiments/run_recovery_reps.sh 5 6 7 8 9 &
#
# Override defaults via env vars, e.g.:
#   N_REPS=5 SEED_BASE=200 OUT_ROOT=results/my_reps bash scripts/experiments/run_recovery_reps.sh
set -euo pipefail

N_REPS="${N_REPS:-10}"                                  # total reps when no indices are passed
SEED_BASE="${SEED_BASE:-100}"                           # rep i uses seed SEED_BASE + i
OUT_ROOT="${OUT_ROOT:-results/recovery_from_rand_images}"  # parent dir for rep*/ output
SPACES="${SPACES:-clip_vitb32 openclip_laion alexnet_conv5 alexnet_rand_conv5}"
EVALS="${EVALS:-clip_vitb32,clip_laion,alexnet_conv5,alexnet_rand_conv5,lpips,dreamsim}"
INVERT_ARGS="${INVERT_ARGS:---no_crop}"                 # extra args to the invert step (match recovery_nocrop25)
EVAL_ARGS="${EVAL_ARGS:-}"                              # e.g. EVAL_ARGS=--clip_aug to match a crop-aug original

# rep indices: CLI args if given, else 0..N_REPS-1
if [ "$#" -gt 0 ]; then
  REPS=("$@")
else
  REPS=($(seq 0 $((N_REPS - 1))))
fi

echo "[run] reps=${REPS[*]}  seeds=$SEED_BASE+i  out=$OUT_ROOT"
echo "[run] spaces: $SPACES"
echo "[run] evals : $EVALS"

for i in "${REPS[@]}"; do
  REP=$(printf "%s/rep%02d" "$OUT_ROOT" "$i")
  SEED=$((SEED_BASE + i))
  for sp in $SPACES; do
    echo "==== rep$i (seed $SEED) | $sp ===="
    uv run python scripts/experiments/recovery_matrix_invert_reps.py \
        --opt_space "$sp" --seed "$SEED" --out_root "$REP" $INVERT_ARGS
    uv run python scripts/experiments/recovery_check_eval.py \
        --out_dir "$REP/$sp" --evaluators "$EVALS" $EVAL_ARGS
  done
done

# No --root: it defaults to the first rep, so the example images come from a seed the
# bars actually summarise. Naming an unrelated single run there is what the figure
# script warns against.
echo "[run] DONE reps=${REPS[*]}. Make the figure with:"
echo "  uv run python scripts/create_figure_assets/Fig4_recon_and_identification_errorbar.py \\"
echo "      --reps_root $OUT_ROOT --err sd"
