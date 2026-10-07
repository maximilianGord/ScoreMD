#!/bin/bash
# H3: in atomistic ALDP the stiff bond-stretch residuals dominate the TSM loss, so the network spends its
# capacity on bonds and the slow phi/psi landscape (basin populations, barriers) is learned worse.
#
# Intervention: tsm_project_bonds=True removes the 21 bond-stretch directions (computed from x_t) from the
# TSM residual. Bond stretches are then trained by DSM only; TSM still constrains every other direction.
# The projector depends only on x_t, so the TSM target in the remaining directions is unchanged (unbiased).
#
# Arms (same gamma / cutoff for both TSM arms; 3 seeds each):
#   dsm        plain DSM (beta=0)                        -> baseline
#   tsm        TSM noise_cutoff, gamma=3e-4, sigma_max=0.2
#   tsm_proj   same + tsm_project_bonds=True
#
# Prediction if H3 is true:   tsm_proj has better Langevin phi/psi than tsm (lower langevin JS, alpha_R/beta
#                             closer to 0.40/0.59, alpha_L entries/ns closer to DSM), while its bond widths move
#                             back towards DSM (it gives up the local-structure gain on bonds).
# H3 rejected if:             tsm_proj ~ tsm on Langevin phi/psi (difference within the seed spread).
#
# Readout after the runs (fill in the run directories):
#   python evaluation/aldp_langevin_basins.py --runs <dirs> --labels <labels>
#   python evaluation/aldp_langevin_diagnostics.py --runs <dirs> --labels <labels> --true-dts
#   plus eval/aldp_langevin_js_divergence and eval/aldp_iid_js_divergence from out/metrics.json

SEEDS="${SEEDS:-0}"

COMMON="dataset=aldp \
  dataset.limit_samples=50_000 \
  dataset.validation=False \
  +architecture=transformer/potential \
  training_schedule.epochs.0=5000 \
  training_schedule.BS=512 \
  checkpoint_options.save_interval_steps=1000 \
  training_schedule.losses.0.loss.beta=0 \
  +training_schedule/augment=random_rotations \
  dataset.coarse_graining_level=none \
  evaluation.num_iid_samples=50000 \
  evaluation.num_parallel_langevin_samples=50 \
  evaluation.num_langevin_samples=10000 \
  evaluation.num_langevin_intermediate_steps=50 \
  evaluation.langevin_dt=2e-3 \
  evaluation.num_fp_timepoints=0 \
  evaluation.eval_t=1e-5 \
  wandb.enabled=False"

TSM="training_schedule.losses.0.loss.loss_type=tsm \
  training_schedule.losses.0.loss.tsm_type=noise_cutoff \
  training_schedule.losses.0.loss.tsm_lambda=3e-4 \
  training_schedule.losses.0.loss.tsm_sigma_max=0.2"

for seed in $SEEDS; do
  python train.py $COMMON seed=$seed \
    +wandb.name=h3-dsm-seed$seed

  python train.py $COMMON seed=$seed $TSM \
    +wandb.name=h3-tsm-seed$seed

  python train.py $COMMON seed=$seed $TSM \
    training_schedule.losses.0.loss.tsm_project_bonds=True \
    +wandb.name=h3-tsm-proj-seed$seed
done
