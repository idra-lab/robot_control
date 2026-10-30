#!/bin/bash
# @author: riccardobussola
# Figures and numbers for one robot, from the outputs of run_pipeline.sh.  Runs in the lab container
# or, from the host, through run_in_image.sh, in robot_control/base_controllers:
#
#   bash components/safe_rl/summarize.sh <robot>
#   components/safe_rl/run_in_image.sh bash components/safe_rl/summarize.sh <robot>
#
# Reads OUT/<robot>/ (fall_check, episodes, test, closed_loop/{alone,filter_<threshold>}, train.log)
# and MODELS/VF_push_end_<robot>.onnx.  Writes FIGURES/<robot>_fall_rule.png, _dataset.png,
# _value_function.png, _closed_loop.png and <robot>_summary.txt, the numbers the report quotes.
# OUT, MODELS and FIGURES default to the package's observation_datasets, models and report_figures.
set -e
if [ $# -ne 1 ]; then sed -n 3,12p "$0"; exit 1; fi
robot=$1
SR=components/safe_rl
OUT=${OUT:-$SR/value_function/observation_datasets}
MODELS=${MODELS:-$SR/value_function/models}
FIGURES=${FIGURES:-$SR/report_figures}
D=$OUT/$robot
# the kick below which the walking policy falls about 1 % of the time: an alarm there is false
case $robot in go2) light=0.75;; *) light=1.0;; esac
# the model the filter loaded before this work, as a baseline
case $robot in aliengo) baseline=$SR/value_function/models/VF_se_100_cp_terminationB.pkl;; *) baseline=;; esac
npy() { ls "$1"/batch*/observations_dataset_abs_test.npy; }
mkdir -p "$FIGURES"
{
echo "################ $robot, $(date +%F)"
echo; echo "======== fall rule (fall_rule.py)"
python3 $SR/fall_rule.py $(npy $D/fall_check) --title "${robot^}: " --plot $FIGURES/${robot}_fall_rule.png
echo; echo "======== dataset (dataset_statistics.py, training and test batches)"
python3 $SR/dataset_statistics.py --robot $robot $(npy $D/episodes) $(npy $D/test) --plot $FIGURES/${robot}_dataset.png \
  | grep -v "^$OUT"
grep -h "Discarded" $D/episodes/batch*.log $D/test/batch*.log | sort | uniq -c || true
echo; echo "======== training (train.log)"
tr '\r' '\n' < $D/train.log | grep -a "Transitions\|best at epoch\|ONNX export" | sed 's/^ *//'
echo; echo "======== evaluation on the test batches, the ONNX model (evaluate_value_function.py)"
python3 $SR/value_function/evaluate_value_function.py --model $MODELS/VF_push_end_$robot.onnx --data $(npy $D/test) \
  --light-kick $light --plot $FIGURES/${robot}_value_function.png
if [ -n "$baseline" ]; then
  echo; echo "======== evaluation of the model deployed before, $(basename $baseline)"
  python3 $SR/value_function/evaluate_value_function.py --model $baseline --data $(npy $D/test) --light-kick $light
fi
for f in $D/closed_loop/filter_*; do
  thr=${f##*_}
  echo; echo "======== closed loop, threshold $thr (compare_filter.py)"
  plot=$([ "$thr" = 0.5 ] && echo "--plot $FIGURES/${robot}_closed_loop.png" || true)
  python3 $SR/compare_filter.py --light-kick $light --alone $(npy $D/closed_loop/alone) --filtered $(npy $f) $plot
done
} 2>&1 | grep -v "GPU/TPU" | tee $FIGURES/${robot}_summary.txt
