#!/usr/bin/env bash
# Muller-Brown ablation studies for the thesis SC (semigroup-consistency) chapter:
# loss_type=sc with sc_switch=True (below force_sigma_max, samples use the
# sg_type-weighted force anchor; above it, the semigroup-consistency term),
# swept over sg_type in {noise_cutoff, mode_mixture} and delta_s in
# {none, 0.2, 0.1, 0.05}, each run 5x with different seeds.
#
# sg_lambda and tsm_lambda are always set to the same LAMBDA so the sg-schedule
# weight and the additive schedule-based TSM term (tsm_type=constant, evaluated
# at the sampled ts) share one lambda, as in experiments_sc.sh. For
# sg_type=mode_mixture the raw sg_lambda float is unused internally (the
# mode-mixture kappa/lambda come from sg_lambda_scheme instead), but it is
# still set to LAMBDA to keep the pairing consistent everywhere.
#
# All run directories (and their out/metrics.json) land under
# $BASE_DIR/<experiment>/seed_<n>. A run's success/failure is appended to
# $BASE_DIR/run_log.txt; a failed run does not stop the rest of the sweep.

set -uo pipefail

BASE_DIR="/ds/students/go57nuk_maximilian_gordin/out/analysis/mueller_brown_ablations_thesis"
SEEDS=(1 2 3 4 5)
LOG_FILE="${BASE_DIR}/run_log.txt"
mkdir -p "${BASE_DIR}"

LAMBDA=0.5

COMMON_ARGS=(
  dataset=mueller_brown
  +architecture=mlp/small_potential
  training_schedule.epochs=180
  training_schedule.losses.0.time_weighting.midpoint=0.5
  evaluation.partial_denoise_eval_ts=[0.1,0.05,0.01]
  wandb.enabled=False
  training_schedule.losses.0.loss.loss_type=sc
  training_schedule.losses.0.loss.sc_switch=true
  training_schedule.losses.0.loss.beta=0
  training_schedule.losses.0.loss.sg_lambda="${LAMBDA}"
  training_schedule.losses.0.loss.tsm_lambda="${LAMBDA}"
)

run_experiment() {
  local exp_name="$1"
  shift
  local extra_args=("$@")
  local seed run_dir
  for seed in "${SEEDS[@]}"; do
    run_dir="${BASE_DIR}/${exp_name}/seed_${seed}"
    echo "=== ${exp_name} (seed=${seed}) -> ${run_dir} ==="
    if python train.py \
      "${COMMON_ARGS[@]}" \
      "${extra_args[@]}" \
      seed="${seed}" \
      hydra.run.dir="${run_dir}" \
      +wandb.name="mb-sc-$(echo "${exp_name}" | tr '/' '-')-seed${seed}"
    then
      echo "OK   ${exp_name} seed=${seed}" >> "${LOG_FILE}"
    else
      echo "FAIL ${exp_name} seed=${seed}" >> "${LOG_FILE}"
    fi
  done
}

delta_s_args() {
  local ds="$1"
  if [ "${ds}" != "none" ]; then
    echo "training_schedule.losses.0.loss.delta_s=${ds}"
  fi
}

### sc, sg_type=noise_cutoff: gamma(t) = sg_lambda if sigma_t <= sg_sigma_max else 0, ###
### shared between the force anchor (std <= force_sigma_max) and the semigroup       ###
### consistency term (std > force_sigma_max).                                        ###

for sgsmax in 0.2 0.4 0.6; do
  for ds in none 0.2 0.1 0.05; do
    run_experiment "sc_noise_cutoff/force_sigma_max_0.1/sg_sigma_max_${sgsmax}/delta_s_${ds}" \
      training_schedule.losses.0.loss.sg_type=noise_cutoff \
      training_schedule.losses.0.loss.force_sigma_max=0.1 \
      training_schedule.losses.0.loss.sg_sigma_max="${sgsmax}" \
      $(delta_s_args "${ds}")
  done
done

for sgsmax in 0.6 0.8; do
  for ds in none 0.2 0.1 0.05; do
    run_experiment "sc_noise_cutoff/force_sigma_max_0.5/sg_sigma_max_${sgsmax}/delta_s_${ds}" \
      training_schedule.losses.0.loss.sg_type=noise_cutoff \
      training_schedule.losses.0.loss.force_sigma_max=0.5 \
      training_schedule.losses.0.loss.sg_sigma_max="${sgsmax}" \
      $(delta_s_args "${ds}")
  done
done

### sc, sg_type=mode_mixture: uniform lambda schedule, closed-form Muller-Brown mode ###
### variance estimator; sweeping force_sigma_max and delta_s.                        ###

for fsmax in 0.1 0.5 0.8; do
  for ds in none 0.2 0.1 0.05; do
    run_experiment "sc_mode_mixture/force_sigma_max_${fsmax}/delta_s_${ds}" \
      training_schedule.losses.0.loss.sg_type=mode_mixture \
      training_schedule.losses.0.loss.sg_lambda_scheme=uniform \
      training_schedule.losses.0.loss.force_sigma_max="${fsmax}" \
      dataset.mode_var_computation=potential \
      $(delta_s_args "${ds}")
  done
done

echo "All experiments submitted. See ${LOG_FILE} for a pass/fail summary."
