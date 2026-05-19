#!/usr/bin/env bash
set -euo pipefail

# Run the redesigned ETH/UCY adaptive-prior ablation experiments on Linux.
# Assumption: the star conda environment has already been activated.
#
# Examples:
#   bash scripts/run_eth_ucy_ablation_remaining.sh
#   bash scripts/run_eth_ucy_ablation_remaining.sh univ
#   bash scripts/run_eth_ucy_ablation_remaining.sh zara1 zara2
#   bash scripts/run_eth_ucy_ablation_remaining.sh --dry-run univ

cd "$(dirname "$0")/.."

BASE_ARGS=(
  --dataset eth5
  --start_test 10
  --sample_num 20
  --fde_weight 0.5
  --spatial_prior_mix 0.6
  --spatial_sigma 2.0
  --diversity_weight 0.02
  --diversity_margin 0.2
  --motion_gate_bias 1.0
  --spatial_gate_bias -1.0
)

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi

dataset_args() {
  case "$1" in
    eth)
      echo "--num_epochs 300 --learning_rate 0.001 --router_top_p 0.5 --rt_layers 3 --num_virtual_nodes 3"
      ;;
    hotel)
      echo "--num_epochs 300 --learning_rate 0.0018 --router_top_p 0.6 --rt_layers 3 --num_virtual_nodes 3"
      ;;
    univ)
      echo "--num_epochs 300 --learning_rate 0.001 --router_top_p 0.5 --rt_layers 3 --num_virtual_nodes 3"
      ;;
    zara1)
      echo "--num_epochs 300 --learning_rate 0.0012 --router_top_p 0.6 --rt_layers 3 --num_virtual_nodes 3"
      ;;
    zara2)
      echo "--num_epochs 300 --learning_rate 0.0012 --router_top_p 0.5 --rt_layers 3 --num_virtual_nodes 3"
      ;;
    *)
      echo "[ERROR] Unsupported ETH/UCY test_set: $1" >&2
      echo "        Expected one of: eth, hotel, univ, zara1, zara2" >&2
      return 1
      ;;
  esac
}

run_one() {
  local setname="$1"
  local mode="$2"
  local model="ab_${setname}_${mode}"
  local outdir="output/${setname}/${model}"
  local extra_args
  extra_args="$(dataset_args "$setname")"

  echo
  echo "[TRAIN] test_set=${setname} ablation=${mode} model=${model}"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY ] mkdir -p \"${outdir}\""
    echo "[DRY ] rm -f \"${outdir}/config_train.yaml\""
    echo "[DRY ] python trainval.py --phase train --test_set \"${setname}\" --train_model \"${model}\" --ablation \"${mode}\" ${BASE_ARGS[*]} ${extra_args} > \"${outdir}/train_stdout.txt\" 2>&1"
    echo "[DRY ] rm -f \"${outdir}/config_test.yaml\""
    echo "[DRY ] python trainval.py --phase test --test_set \"${setname}\" --train_model \"${model}\" --ablation \"${mode}\" --load_model best ${BASE_ARGS[*]} ${extra_args} > \"${outdir}/test_stdout_best.txt\" 2>&1"
    return 0
  fi

  mkdir -p "$outdir"
  rm -f "${outdir}/config_train.yaml"
  python trainval.py \
    --phase train \
    --test_set "$setname" \
    --train_model "$model" \
    --ablation "$mode" \
    "${BASE_ARGS[@]}" \
    ${extra_args} \
    > "${outdir}/train_stdout.txt" 2>&1

  echo "[TEST ] test_set=${setname} ablation=${mode} model=${model}"
  rm -f "${outdir}/config_test.yaml"
  python trainval.py \
    --phase test \
    --test_set "$setname" \
    --train_model "$model" \
    --ablation "$mode" \
    --load_model best \
    "${BASE_ARGS[@]}" \
    ${extra_args} \
    > "${outdir}/test_stdout_best.txt" 2>&1

  echo "[OK   ] ${model}"
}

run_dataset() {
  local setname="$1"
  local extra_args
  extra_args="$(dataset_args "$setname")"

  echo
  echo "============================================================"
  echo "[DATASET] ${setname}"
  echo "[ARGS] ${extra_args}"
  echo "============================================================"

  run_one "$setname" adaptive
  run_one "$setname" adaptive_no_motion
  run_one "$setname" adaptive_no_spatial
  run_one "$setname" adaptive_motion_only
  run_one "$setname" vite_only
  run_one "$setname" motion_only
  run_one "$setname" spatial_only
}

if [[ "$#" -eq 0 ]]; then
  run_dataset univ
  run_dataset zara1
  run_dataset zara2
else
  for setname in "$@"; do
    run_dataset "$setname"
  done
fi

echo "[DONE] Adaptive-prior ETH/UCY ablation experiments finished."
