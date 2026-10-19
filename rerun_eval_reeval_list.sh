#!/usr/bin/env bash
### Re-evaluate every run listed in reeval_list.json with long Langevin settings
### (100 x 120k Langevin, dt=2e-3, 50 intermediate steps).
###
### No training: each run's own .hydra config is loaded with load_dir=<run> and
### continue_from=null, so the latest checkpoint is evaluated directly.
### Results go next to the original results, into <run>/out_12ns/ (metrics.json,
### plots, trajectories). The original out/, .hydra/ and train.log are untouched:
### this run's hydra config goes to <run>/.hydra_12ns/ and its log to
### <run>/train_12ns.log.
###
### 120000 samples x 50 steps x 2 fs = 12 ns per trajectory, 100 trajectories.
###
### Usage:
###   bash rerun_eval_reeval_list.sh                 # all datasets
###   bash rerun_eval_reeval_list.sh aldp_noH        # only some keys
###   FORCE=1 ...                                          # redo finished runs
set -uo pipefail

base=/ds/students/go57nuk_maximilian_gordin/out
list=/home/go57nuk/ScoreMD/reeval_list.json
OUT="${OUT:-out_12ns}"
log=/home/go57nuk/ScoreMD/rerun_eval_reeval_list.log

mapfile -t RUNS < <(python3 - "$list" "$@" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
keys = sys.argv[2:] or list(d)
for k in keys:
    for run, name in d[k].items():
        print(f"{k}/{run}\t{name}")
EOF
)

echo "Re-evaluating ${#RUNS[@]} runs, output ${OUT}" | tee -a "$log"

for entry in "${RUNS[@]}"; do
  rel="${entry%%$'\t'*}"
  name="${entry#*$'\t'}"
  d="$base/$rel"
  if [ ! -d "$d/.hydra" ]; then
    echo "SKIP $rel ($name): no .hydra" | tee -a "$log"; continue
  fi
  if [ -f "$d/$OUT/metrics.json" ] && [ -z "${FORCE:-}" ]; then
    echo "DONE $rel ($name): metrics.json exists" | tee -a "$log"; continue
  fi
  echo "=== $rel ($name) -> $d/$OUT ===" | tee -a "$log"
  if pixi run python train.py \
    --config-path "$d/.hydra" --config-name config.yaml \
    hydra.run.dir="$d" \
    hydra.output_subdir=".hydra${OUT#out}" \
    hydra.job_logging.handlers.file.filename="$d/train_${OUT#out_}.log" \
    ++out_subdir="$OUT" \
    load_dir="$d" continue_from=null \
    evaluation.num_parallel_langevin_samples=100 \
    evaluation.langevin_dt=2e-3 \
    evaluation.num_langevin_intermediate_steps=50 \
    evaluation.num_langevin_samples=120000 \
    wandb.enabled=False
    # evaluation.num_iid_samples=50000 \
    # evaluation.num_fp_timepoints=0 \
    # evaluation.eval_t=1e-5 \
  then
    echo "OK   $rel ($name)" | tee -a "$log"
  else
    echo "FAIL $rel ($name)" | tee -a "$log"
  fi
done
