#!/bin/bash
# @author: riccardobussola
# The v2 study (components/safe_rl/v2/README.md), from the host: value functions on the policy's own
# observation, ours as the delayed-switch value Q(o, delta), ABS as published, then both filters on
# push families the value functions never saw.  Every simulation with the motors' torque-speed curve.
#
#   run_v2.sh [robot ...]            default: go2 aliengo;  STAGES="train test" for some stages
#
# Stages:
#   data   mps: MPS_BATCHES x MPS_EPISODES, the walking policy, a training push (0.2 s, kick uniform in
#          the robot's kick_range), the safe policy from the first inference 0.02-0.7 s after the push
#          starts (30 % of the episodes: no push, a handover while walking); abs: ABS_BATCHES x
#          ABS_EPISODES walking-policy rollouts with a training push
#   train  VF_mps_<robot> (EPOCHS), VF_abs_<robot> (ABS_EPOCHS: its TD propagates over the episode),
#          VF_paper_<robot> (TD(0)) and VF_paper_mc_<robot> (EPOCHS, on the mps data) (train.py) into
#          MODELS, unless the .onnx exists
#   calibrate  calibrate.py: ours' walking-rule threshold from the training data (the abs rollouts' undisturbed
#          walking: switch in at most 1 % of them), threshold_walking in the model's .json; calibrate_paper.py:
#          the paper's V_thr and Lipschitz margin L for VF_paper and VF_paper_mc
#   test   per push family, TEST_BATCHES x TEST_EPISODES pushes, the same for every arm (seeded per
#          family): the walking policy alone, the safe policy alone (from the first inference after the
#          push starts), ABS at its rule (0.05) and at 0.3, ours at 0.9, 0.5, 0.95, 0.99 and at the walking rule (mps_wr); the kicks KICKS
#          (impulse / mass, m/s), episodes LENGTH s unless the robot leaves the safe set
#          paper / paper_wr: the paper's V(o) (train.py --method paper, TD(0) on the safe policy's
#          transitions) at Sec. II-C's V_thr + L, and at the walking rule (calibrate_paper.py);
#          paper_mc: the same V(o) with Monte Carlo targets (train.py --method paper_mc) at a probability
#          fixed in advance plus the Lipschitz margin, PAPER_P (0.8) + L
#          families: train (0.2 s, as the training data: the in-distribution reference), impulse
#          (0.05 s), long (0.6 s), offset (0.2 s applied 0.25 m ahead of or behind the base), down
#          (0.2 s with as much force downward as horizontal)
#   summary  summarize.py: tables and figures into FIGURES
#
# Outputs (relative to robot_control/base_controllers): OUT (components/safe_rl/v2/data)/<robot>/,
# MODELS (components/safe_rl/v2/models), FIGURES (components/safe_rl/v2/report).  A stage skips what
# is complete; delete a folder to redo it.  Seeds: SEED (go2 7000, aliengo 8000) + 0 mps, + 100 abs,
# + 1000 + 100 x family the tests.
set -e
SR_HOST=$(readlink -f "$(dirname "$(readlink -f "$0")")/..")
BC_HOST=$(readlink -f "$SR_HOST/../..")
V2=components/safe_rl/v2
export OUT=${OUT:-$V2/data}
export MODELS=${MODELS:-$V2/models}
export FIGURES=${FIGURES:-$V2/report}
MPS_BATCHES=${MPS_BATCHES:-10}; MPS_EPISODES=${MPS_EPISODES:-100}
ABS_BATCHES=${ABS_BATCHES:-10}; ABS_EPISODES=${ABS_EPISODES:-60}
TEST_BATCHES=${TEST_BATCHES:-10}; TEST_EPISODES=${TEST_EPISODES:-20}; LENGTH=${LENGTH:-7}
EPOCHS=${EPOCHS:-60}; ABS_EPOCHS=${ABS_EPOCHS:-200}
FAMILIES=${FAMILIES:-"train impulse long offset down"}
ARMS=${ARMS:-"walk safe abs abs_03 mps mps_05 mps_wr mps_95 mps_99 paper paper_wr paper_mc"}
STAGES=${STAGES:-"data train calibrate test summary"}
ROBOTS=${*:-go2 aliengo}
stamp() { echo "$(date +%T) $*"; }
stage() { [[ " $STAGES " == *" $1 "* ]]; }
wait_batches() { sleep 20; while [ -n "$(docker ps -q --filter name=safe_rl_)" ]; do sleep 10; done; }
batches() {
  local first=$1 n=$2 episodes=$3 out=$4; shift 4
  local done=1
  for i in $(seq 1 "$n"); do [ -f "$BC_HOST/$out/batch$i/v2_dataset.npz" ] || done=0; done
  if [ $done = 1 ]; then echo "  $out: complete"; return; fi
  rm -rf "$BC_HOST/$out"
  "$SR_HOST/run_batches.sh" "$first" "$n" "$episodes" "$out" --v2 --motor-limit "$@" > /dev/null
  wait_batches
  echo "  $out: $(cat "$BC_HOST/$out"/batch*/observations_dataset_abs_test_episodes.csv | grep -vc episode) episodes"
}
in_image() { "$SR_HOST/run_in_image.sh" "$@"; }
family_args() {
  case $1 in
    train) echo "--push-duration 0.2";;
    impulse) echo "--push-duration 0.05";;
    long) echo "--push-duration 0.6";;
    offset) echo "--push-duration 0.2 --push-lever 0.25";;
    down) echo "--push-duration 0.2 --push-down 1.0";;
    *) echo "unknown family $1" >&2; exit 1;;
  esac
}
arm_args() {
  local robot=$1 arm=$2 M=$MODELS
  case $arm in
    walk) echo "";;
    safe) echo "--backup-delay 0.001";;
    abs) echo "--shield $M/VF_abs_$robot.onnx";;
    abs_03) echo "--shield $M/VF_abs_$robot.onnx --shield-threshold 0.3";;
    mps) echo "--shield $M/VF_mps_$robot.onnx";;
    mps_05) echo "--shield $M/VF_mps_$robot.onnx --shield-threshold 0.5";;
    mps_wr) echo "--shield $M/VF_mps_$robot.onnx --shield-threshold $(python3 -c "import json; print(json.load(open('$BC_HOST/$M/VF_mps_$robot.json'))['threshold_walking'])")";;
    paper) echo "--shield $M/VF_paper_$robot.onnx --shield-threshold $(python3 -c "import json; print(json.load(open('$BC_HOST/$M/VF_paper_$robot.json'))['threshold_paper'])")";;
    paper_wr) echo "--shield $M/VF_paper_$robot.onnx --shield-threshold $(python3 -c "import json; print(json.load(open('$BC_HOST/$M/VF_paper_$robot.json'))['threshold_walking'])")";;
    paper_mc) echo "--shield $M/VF_paper_mc_$robot.onnx --shield-threshold $(python3 -c "import json; d = json.load(open('$BC_HOST/$M/VF_paper_mc_$robot.json')); print(${PAPER_P:-0.8} + d['lipschitz_L'])")";;
    mps_*) echo "--shield $M/VF_mps_$robot.onnx --shield-threshold 0.${arm#mps_}";;
    *) echo "unknown arm $arm" >&2; exit 1;;
  esac
}

if [ -n "$(docker ps -q --filter name=safe_rl_)" ]; then
  echo "safe_rl_ containers are running (docker ps): wait for them or stop them"; exit 1
fi
for robot in $ROBOTS; do
  case $robot in
    go2) seed=${SEED:-7000}; kicks=${KICKS:-"1.0 1.2 1.4 1.6 1.8"};;
    aliengo) seed=${SEED:-8000}; kicks=${KICKS:-"1.2 1.4 1.6 1.8 2.0"};;
    *) seed=${SEED:?set SEED}; kicks=${KICKS:?set KICKS};;
  esac
  D=$OUT/$robot
  R="--robot $robot"

  if stage data; then
    stamp "$robot: data"
    batches $seed "$MPS_BATCHES" "$MPS_EPISODES" $D/mps $R --method mps --switch-from push_start \
      --switch-window 0.02 0.7 --walk-switch 0.3
    batches $((seed + 100)) "$ABS_BATCHES" "$ABS_EPISODES" $D/abs $R --method abs
  fi

  if stage train; then
    for method in mps abs paper paper_mc; do
      model=$MODELS/VF_${method}_$robot
      if [ -f "$BC_HOST/$model.onnx" ]; then echo "  $model: exists"; continue; fi
      stamp "$robot: train $method"
      mkdir -p "$BC_HOST/$MODELS"
      epochs=$EPOCHS; [ $method = abs ] && epochs=$ABS_EPOCHS
      data=$method; [[ $method = paper* ]] && data=mps  # the paper's V: the safe policy's part of the mps episodes
      folders=$(cd "$BC_HOST" && ls -d $D/$data/batch*/ | tr '\n' ' ')
      in_image python3 $V2/train.py --method $method --data $folders --out $model --epochs "$epochs" \
        | tee "$BC_HOST/${model}_train.log" | grep -v "^epoch" || true
    done
  fi

  if stage calibrate; then
    stamp "$robot: calibrate"
    in_image bash -c "python3 $V2/calibrate.py --model $MODELS/VF_mps_$robot.onnx --abs-data $D/abs/batch*/ \
      --mps-data $D/mps/batch*/ --write" | tail -4
    for method in paper paper_mc; do
      in_image bash -c "python3 $V2/calibrate_paper.py --model $MODELS/VF_${method}_$robot.onnx --abs-data $D/abs/batch*/ \
        --write" | tail -4
    done
  fi

  if stage test; then
    f=0
    for family in $FAMILIES; do
      for arm in $ARMS; do
        stamp "$robot: test $family $arm"
        batches $((seed + 1000 + 100 * f)) "$TEST_BATCHES" "$TEST_EPISODES" $D/test/$family/$arm $R --method abs \
          --episode-length "$LENGTH" --kicks $kicks $(family_args $family) $(arm_args $robot $arm)
      done
      f=$((f + 1))
    done
  fi

  if stage summary; then
    in_image python3 $V2/summarize.py --robot $robot --data $D/test --out $FIGURES
  fi
done
