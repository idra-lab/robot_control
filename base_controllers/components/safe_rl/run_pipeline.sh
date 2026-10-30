#!/bin/bash
# @author: riccardobussola
# The whole pipeline for one or more robots, from the host (README.md, "Run it all"):
#   1. fall-rule check: 8 batches of FALL_EPISODES episodes run to the ground (--fall-check)
#   2. dataset: 6 batches of EPISODES for training (the 6th stops it), 2 recorded from the push
#      start for the test (--from-push-start)
#   3. training (run_in_image.sh), into MODELS/VF_push_end_<robot>.pkl and its ONNX export .onnx,
#      the model the filter loads
#   4. closed loop: 3 batches of LOOP_EPISODES with the walking policy alone, and the same with
#      the filter at each of THRESHOLDS
#   5. summarize.sh: figures and numbers in FIGURES/
#
#   run_pipeline.sh [robot ...]        default: aliengo go2
#
# About 20 minutes per robot on 24 cores (8 simulations at a time, in real time).  It overwrites
# MODELS/VF_push_end_<robot>.pkl and .onnx and OUT/<robot>/.  Seeds: aliengo
# 401 (fall check), 501 (dataset), 701 (closed loop); go2 421, 521, 711; another robot needs
# SEEDS="<fall check> <dataset> <closed loop>".  Sizes: FALL_EPISODES (30), EPISODES (70),
# LOOP_EPISODES (35), EPOCHS (600), THRESHOLDS ("0.5 0.8").  Outputs, relative to
# robot_control/base_controllers: OUT (components/safe_rl/value_function/observation_datasets)/<robot>/,
# MODELS (.../value_function/models), FIGURES (components/safe_rl/report_figures).
set -e
SR_HOST=$(dirname "$(readlink -f "$0")")
BC_HOST=$(readlink -f "$SR_HOST/../..")
SR=components/safe_rl
export OUT=${OUT:-$SR/value_function/observation_datasets}
export MODELS=${MODELS:-$SR/value_function/models}
export FIGURES=${FIGURES:-$SR/report_figures}
FALL_EPISODES=${FALL_EPISODES:-30}; EPISODES=${EPISODES:-70}; LOOP_EPISODES=${LOOP_EPISODES:-35}
EPOCHS=${EPOCHS:-600}; THRESHOLDS=${THRESHOLDS:-"0.5 0.8"}
ROBOTS=${*:-aliengo go2}
batches() { "$SR_HOST/run_batches.sh" "$@" > /dev/null; }
wait_batches() { sleep 20; while [ -n "$(docker ps -q --filter name=safe_rl_)" ]; do sleep 10; done; }
stamp() { echo "$(date +%T) $*"; }

for robot in $ROBOTS; do
  case $robot in
    aliengo) seeds="401 501 701";;
    go2) seeds="421 521 711";;
    *) seeds=${SEEDS:?set SEEDS=\"<fall check> <dataset> <closed loop>\" for $robot};;
  esac
  read -r fall_seed data_seed loop_seed <<< "$seeds"
  D=$OUT/$robot
  model=$MODELS/VF_push_end_$robot.pkl   # the training's; the filter runs ${model%.pkl}.onnx
  if [ -n "$(docker ps -q --filter name=safe_rl_)" ]; then
    echo "safe_rl_ containers are running (docker ps): wait for them or stop them first"; exit 1
  fi

  stamp "$robot: 1. fall-rule check"
  batches $fall_seed 8 $FALL_EPISODES $D/fall_check --robot $robot --fall-check; wait_batches

  stamp "$robot: 2. dataset"
  batches $data_seed 6 $EPISODES $D/episodes --robot $robot
  batches $((data_seed + 6)) 2 $EPISODES $D/test --robot $robot --from-push-start; wait_batches

  stamp "$robot: 3. training"
  "$SR_HOST/run_in_image.sh" python3 $SR/value_function/train_value_function.py \
    --data $(cd "$BC_HOST" && ls $D/episodes/batch[1-5]/observations_dataset_abs_test.npy) \
    --val $D/episodes/batch6/observations_dataset_abs_test.npy --model $model --epochs $EPOCHS > "$BC_HOST/$D/train.log" 2>&1
  grep -a "best at epoch" "$BC_HOST/$D/train.log"

  stamp "$robot: 4. closed loop"
  batches $loop_seed 3 $LOOP_EPISODES $D/closed_loop/alone --robot $robot
  for thr in $THRESHOLDS; do
    batches $loop_seed 3 $LOOP_EPISODES $D/closed_loop/filter_$thr --robot $robot --filter ${model%.pkl}.onnx --threshold $thr
  done
  wait_batches

  stamp "$robot: 5. summary"
  "$SR_HOST/run_in_image.sh" bash $SR/summarize.sh $robot > /dev/null
  stamp "$robot: done, $FIGURES/${robot}_summary.txt"
done
