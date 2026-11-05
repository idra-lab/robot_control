#!/bin/bash
# @author: riccardobussola
# Figures and numbers for one robot, from the outputs of run_pipeline.sh.  Runs in the lab container
# or, from the host, through run_in_image.sh, in robot_control/base_controllers:
#
#   bash components/safe_rl/summarize.sh <robot>
#   components/safe_rl/run_in_image.sh bash components/safe_rl/summarize.sh <robot>
#
# Reads OUT/<robot>/ (fall_check, abs, mps, calibration, closed_loop, the train_, calibrate_ and
# evaluate_ logs) and MODELS/VF_<name>_<robot>.json.  Writes FIGURES/<robot>_summary.txt, the numbers
# value_function_comparison.md quotes, the closed-loop tables FIGURES/<robot>_loop_<table>.md and the
# figures <robot>_fall_rule.png, _dataset_abs.png, _dataset_mps.png, _loop_<table>.png (and
# _vf_<model>.png from the evaluate stage).  OUT, MODELS and FIGURES default to the package's
# observation_datasets, models and report_figures.
set -e
if [ $# -ne 1 ]; then sed -n 3,13p "$0"; exit 1; fi
robot=$1
SR=components/safe_rl
OUT=${OUT:-$SR/value_function/observation_datasets}
MODELS=${MODELS:-$SR/value_function/models}
FIGURES=${FIGURES:-$SR/report_figures}
D=$OUT/$robot
npy() { ls "$1"/batch*/observations_dataset_abs_test.npy 2>/dev/null; }
models=$(cd $D && ls train_*.log | sed 's/train_\(.*\)\.log/\1/')
mkdir -p "$FIGURES"
# a closed-loop table of the arms that exist: table <name> <arm to test against, or -> <arm> ...; the
# threshold table also plots falls against needless switches (<robot>_tradeoff.png)
table() {
  local name=$1 versus=$2; shift 2; local args=()
  for arm in "$@"; do [ -d $D/closed_loop/$arm ] && args+=(--arm $arm $(npy $D/closed_loop/$arm)); done
  [ ${#args[@]} -gt 0 ] || return 0
  [ "$versus" != - ] && [ -d $D/closed_loop/$versus ] && args+=(--versus $versus)
  [ $name = threshold ] && args+=(--tradeoff $FIGURES/${robot}_tradeoff.png)
  echo; echo "======== closed loop: $name (compare_filter.py)"
  python3 $SR/compare_filter.py "${args[@]}" --table $FIGURES/${robot}_loop_$name.md --plot $FIGURES/${robot}_loop_$name.png
}
{
echo "################ $robot, $(date +%F)"
echo; echo "======== fall rule (fall_rule.py)"
python3 $SR/fall_rule.py $(npy $D/fall_check) --title "${robot^}: " --plot $FIGURES/${robot}_fall_rule.png
for method in abs mps; do
  echo; echo "======== $method dataset (dataset_statistics.py, training and test batches)"
  python3 $SR/dataset_statistics.py --robot $robot $(npy $D/$method/episodes) $(npy $D/$method/test) \
    --plot $FIGURES/${robot}_dataset_$method.png | grep -v "^$OUT"
  python3 - $D/$method <<'EOF'
import glob, sys
import numpy as np
e = np.concatenate([np.atleast_1d(np.genfromtxt(f, delimiter=",", names=True))
                    for f in glob.glob(sys.argv[1] + "/*/batch*/observations_dataset_abs_test_episodes.csv")])
e = e[e["discarded"] == 0]
print(f"Outcomes of the {len(e)} episodes kept: fallen {int(e['fallen'].sum())}, captured {int(e['captured'].sum())}, "
      f"survived {int(e['survived'].sum())}, static equilibrium {int(e['static'].sum())}, support polygon "
      f"{int(e['polygon'].sum())}, timeout {int(e['timeout'].sum())}")
if np.any(e["method"] == 1):
    t = e["static_time"] - e["backup_time"]
    print(f"  safe policy: static equilibrium {np.nanmedian(t):.2f} s after the switch (median, 90th percentile "
          f"{np.nanpercentile(t, 90):.2f} s); switched while walking undisturbed {int(np.sum(e['kick'] == 0))}")
EOF
done
for name in $models; do
  echo; echo "======== model $name: training (train_$name.log)"
  tr '\r' '\n' < $D/train_$name.log | grep -a "Transitions\|left out\|best at epoch\|ONNX export" | sed 's/^ *//'
  echo "-------- calibration (calibrate_filter.py)"
  grep -v "GPU/TPU" $D/calibrate_$name.log || true
  echo "-------- evaluation on the test batches (evaluate_value_function.py)"
  grep -v "GPU/TPU\|^Plot" $D/evaluate_$name.log || true
done
# the filters at their threshold (walking rule) and margin; the threshold rules; the margin; the
# falls against needless switches curve; switching back; the ablations at both threshold rules
table main abs alone safe abs absfull mps mps_early
table rule mps alone abs abs_pct absfull absfull_pct mps mps_pct mps_early mps_early_pct
table margin mps alone mps mps_L0 mps_pct mps_pct_L0
table threshold - alone safe abs_t0.5 abs_pct abs abs_t0.99 mps_t0.5 mps_pct mps_t0.9 mps mps_t0.95 mps_early mps_early_t0.9
table switch_back - alone abs abs_back abs_pct abs_pct_back mps mps_back mps_pct mps_pct_back
table ablation mps alone mps mps_early mps_polygon mps_obsnoise mps_nonoise mps_ou2
table ablation_pct mps_pct alone mps_pct mps_early_pct mps_polygon_pct mps_obsnoise_pct mps_nonoise_pct mps_ou2_pct
} 2>&1 | grep -v "GPU/TPU" | tee $FIGURES/${robot}_summary.txt
